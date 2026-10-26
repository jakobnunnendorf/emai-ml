"""UCI Taxi Service Trajectory (Porto, ECML PKDD 2015): ~1.7M trips.

Target CALL_TYPE: A = central, B = taxi stand, C = street. ORIGIN_CALL and
ORIGIN_STAND are not used (each is filled only for one class, so they leak the
label). Features: hour, weekday, n_points (trip duration in 15 s steps),
start/end coordinates, straight-line distance.

The features are cached in data/taxi_features.parquet (building takes ~1 min).
"""

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from lab import DATA_DIR

TARGET = "CALL_TYPE"
CATEGORICAL = ["hour", "weekday"]

# Bounding box around Porto (city centre is about -8.61, 41.15).
PORTO_LON = (-9.0, -8.0)
PORTO_LAT = (40.7, 41.6)


def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in km between two (lon, lat) points (vectorized)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def _features_from_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Turn one chunk of raw rows into feature rows."""
    chunk = chunk[chunk["MISSING_DATA"].astype(str) == "False"]

    # POLYLINE: JSON list of [lon, lat]
    points = chunk["POLYLINE"].map(json.loads)
    n_points = points.map(len)
    keep = n_points > 0
    chunk, points, n_points = chunk[keep], points[keep], n_points[keep]

    time = pd.to_datetime(chunk["TIMESTAMP"], unit="s", utc=True).dt.tz_convert("Europe/Lisbon")
    features = pd.DataFrame({
        "hour": time.dt.hour,
        "weekday": time.dt.weekday,
        "n_points": n_points,
        "start_lon": points.map(lambda p: p[0][0]),
        "start_lat": points.map(lambda p: p[0][1]),
        "end_lon": points.map(lambda p: p[-1][0]),
        "end_lat": points.map(lambda p: p[-1][1]),
        TARGET: chunk[TARGET],
        # only for splitting
        "time": time,
    })
    features["distance_km"] = haversine_km(
        features["start_lon"], features["start_lat"], features["end_lon"], features["end_lat"]
    )

    # drop trips with broken GPS points (well under 1%)
    in_box = (
        features["start_lon"].between(*PORTO_LON) & features["end_lon"].between(*PORTO_LON)
        & features["start_lat"].between(*PORTO_LAT) & features["end_lat"].between(*PORTO_LAT)
    )
    return features[in_box]


def build_taxi_features(csv_path=None, out_path=None, chunksize=100_000):
    """Read the raw CSV in chunks and write the feature table."""
    csv_path = csv_path or DATA_DIR / "taxi" / "train.csv"
    out_path = out_path or DATA_DIR / "taxi_features.parquet"

    columns = [TARGET, "TIMESTAMP", "MISSING_DATA", "POLYLINE"]
    parts = []
    for i, chunk in enumerate(pd.read_csv(csv_path, usecols=columns, chunksize=chunksize)):
        parts.append(_features_from_chunk(chunk))
        print(f"processed {(i + 1) * chunksize:,} rows", flush=True)

    features = pd.concat(parts, ignore_index=True)
    features.to_parquet(out_path, index=False)
    return out_path


def load_taxi(max_rows=None, seed=0):
    """Return (X, y, time). time is only used for splitting, not a feature.
    max_rows: random subset (None = all trips)."""
    path = DATA_DIR / "taxi_features.parquet"
    if not path.exists() or "time" not in pq.read_schema(path).names:
        print("building the feature cache (takes about a minute) ...")
        build_taxi_features(out_path=path)

    df = pd.read_parquet(path)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    # one-hot encoded by the pipeline
    for column in CATEGORICAL:
        df[column] = df[column].astype("category")

    X = df.drop(columns=[TARGET, "time"])
    y = df[TARGET]
    assert not {"ORIGIN_CALL", "ORIGIN_STAND"} & set(X.columns), "leaky columns must not be features"
    return X, y, df["time"]


def split_taxi(X, y, time, val_fraction=0.2):
    """Temporal split: the last ~20% of days are validation (cut at midnight).
    Realistic (train on the past) and keeps trips of the same day together."""
    cutoff = time.quantile(1 - val_fraction).normalize()
    is_val = time >= cutoff
    print(f"training: trips before {cutoff:%Y-%m-%d}, validation: from {cutoff:%Y-%m-%d} on")
    return X[~is_val], X[is_val], y[~is_val], y[is_val]


def month_groups(time):
    """Month of each trip, e.g. "2013-07" (CV groups)."""
    return time.dt.strftime("%Y-%m")
