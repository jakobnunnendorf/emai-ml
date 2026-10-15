"""Porto taxi dataset (UCI "Taxi Service Trajectory - ECML PKDD 2015").

Big classification dataset: ~1.7M taxi trips in Porto (2013-2014).
Target: CALL_TYPE, how the trip was requested:
    A = dispatched from the central (phone call)
    B = taxi stand
    C = hailed on the street / other

Raw file: data/taxi/train.csv (1.9 GB), see README.md for how to unzip it.

Leakage: two raw columns give the answer away. ORIGIN_CALL (caller id) is
filled only for A trips and ORIGIN_STAND (stand id) only for B trips. We
therefore never read them; the model has to guess the call type from WHEN
and WHERE the trip happened. TRIP_ID / TAXI_ID are identifiers and DAY_TYPE
is the same value in every row, so they are skipped as well.

Features we build (one row per trip):
    hour, weekday          from TIMESTAMP (Porto local time), categorical
    n_points               number of GPS points; one every 15 s, so this is
                           the trip duration in units of 15 s
    start_lon, start_lat   first GPS point
    end_lon, end_lat       last GPS point
    distance_km            straight-line distance from start to end

Building the features takes about a minute, so the result is cached in
data/taxi_features.parquet and later loads take about a second. Delete that
file to rebuild it after changing this module.
"""

import json

import numpy as np
import pandas as pd

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
    # MISSING_DATA is read as the strings "True"/"False"; keep complete trips.
    chunk = chunk[chunk["MISSING_DATA"].astype(str) == "False"]

    # POLYLINE is a JSON list of [lon, lat] pairs, e.g. "[[-8.61,41.14],...]".
    points = chunk["POLYLINE"].map(json.loads)
    n_points = points.map(len)
    keep = n_points > 0  # some trips have no GPS points at all
    chunk, points, n_points = chunk[keep], points[keep], n_points[keep]

    time = pd.to_datetime(chunk["TIMESTAMP"], unit="s", utc=True).dt.tz_convert("Europe/Lisbon")
    features = pd.DataFrame({
        "hour": time.dt.hour,
        "weekday": time.dt.weekday,  # 0 = Monday
        "n_points": n_points,
        "start_lon": points.map(lambda p: p[0][0]),
        "start_lat": points.map(lambda p: p[0][1]),
        "end_lon": points.map(lambda p: p[-1][0]),
        "end_lat": points.map(lambda p: p[-1][1]),
        TARGET: chunk[TARGET],
    })
    features["distance_km"] = haversine_km(
        features["start_lon"], features["start_lat"], features["end_lon"], features["end_lat"]
    )

    # A few trips have broken GPS points (e.g. longitude 52.9 instead of -8.6,
    # i.e. 1000+ km away). Squared in the quadratic model these would dwarf
    # every normal value, so we keep only trips that start and end inside a
    # generous box (~50 km) around Porto. This removes well under 1% of trips.
    in_box = (
        features["start_lon"].between(*PORTO_LON) & features["end_lon"].between(*PORTO_LON)
        & features["start_lat"].between(*PORTO_LAT) & features["end_lat"].between(*PORTO_LAT)
    )
    return features[in_box]


def build_taxi_features(csv_path=None, out_path=None, chunksize=100_000):
    """Read the raw CSV in chunks (keeps memory low) and write the feature table."""
    csv_path = csv_path or DATA_DIR / "taxi" / "train.csv"
    out_path = out_path or DATA_DIR / "taxi_features.parquet"

    # Only these columns are read; the leaky ones never enter memory.
    columns = [TARGET, "TIMESTAMP", "MISSING_DATA", "POLYLINE"]
    parts = []
    for i, chunk in enumerate(pd.read_csv(csv_path, usecols=columns, chunksize=chunksize)):
        parts.append(_features_from_chunk(chunk))
        print(f"processed {(i + 1) * chunksize:,} rows", flush=True)

    features = pd.concat(parts, ignore_index=True)
    features.to_parquet(out_path, index=False)
    return out_path


def load_taxi(max_rows=None, seed=0):
    """Return (X, y) for the taxi dataset.

    max_rows: use a random subset of this many trips (handy on a laptop);
              None = all ~1.7M trips.
    """
    path = DATA_DIR / "taxi_features.parquet"
    if not path.exists():
        print("feature cache not found, building it (takes about a minute) ...")
        build_taxi_features(out_path=path)

    df = pd.read_parquet(path)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    # hour and weekday are labels, not quantities (hour 23 is not "23x" hour 1),
    # so we make them categorical and the pipeline one-hot encodes them.
    for column in CATEGORICAL:
        df[column] = df[column].astype("category")

    X = df.drop(columns=TARGET)
    y = df[TARGET]
    assert not {"ORIGIN_CALL", "ORIGIN_STAND"} & set(X.columns), "leaky columns must not be features"
    return X, y
