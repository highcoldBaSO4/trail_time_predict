from __future__ import annotations

import numpy as np
import pandas as pd


MOVEMENT_WINDOW_SECONDS = 15
MINIMUM_MOVEMENT_SPEED_MPS = 0.05
MAXIMUM_MOVEMENT_SPEED_MPS = 12.0


def add_interval_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add point and rolling movement metrics used by all capability models."""
    source_attrs = dict(frame.attrs)
    data = frame.copy().sort_values("timestamp").reset_index(drop=True)
    data["dt_seconds"] = data["timestamp"].diff().dt.total_seconds()
    data["dd_m"] = data["distance"].diff()

    # A short rolling median removes isolated barometer/GPS altitude spikes.
    altitude = data["altitude"].rolling(5, center=True, min_periods=1).median()
    data["smoothed_altitude"] = altitude
    data["delev_m"] = altitude.diff()
    data["speed_mps"] = data["dd_m"] / data["dt_seconds"]
    data["grade_pct"] = data["delev_m"] / data["dd_m"] * 100.0

    valid = (
        data["dt_seconds"].between(0.2, 120.0)
        & data["dd_m"].between(0.0, 1000.0)
        & data["speed_mps"].between(0.0, 12.0)
    )
    data["valid_interval"] = valid
    data.loc[~valid, ["speed_mps", "grade_pct"]] = np.nan
    data["grade_pct"] = data["grade_pct"].clip(-60.0, 60.0)

    # FIT distance is commonly quantized: a slow climb may be recorded as
    # 0 m, 0 m, 1 m on three consecutive seconds. Requiring every individual
    # record to contain distance therefore drops real movement and makes pace
    # look too fast. A centred time window assigns those seconds the local
    # movement speed/grade while still excluding sustained stops.
    timestamp = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    interval_valid = (
        timestamp.notna()
        & data["dt_seconds"].between(0.2, 120.0)
        & data["dd_m"].between(0.0, 1000.0)
    )
    rolling_values = pd.DataFrame(
        {
            "distance": data["dd_m"].where(interval_valid, 0.0).clip(lower=0.0),
            "seconds": data["dt_seconds"].where(interval_valid, 0.0).clip(lower=0.0),
            "elevation": data["delev_m"].where(interval_valid, 0.0),
        }
    )
    usable_timestamp = timestamp.notna()
    rolling_source = rolling_values.loc[usable_timestamp].copy()
    rolling_source.index = pd.DatetimeIndex(timestamp.loc[usable_timestamp])
    rolling = rolling_source.rolling(
        f"{MOVEMENT_WINDOW_SECONDS}s", center=True, min_periods=1
    ).sum()
    rolling_speed = rolling["distance"] / rolling["seconds"].replace(0.0, np.nan)
    rolling_grade = rolling["elevation"] / rolling["distance"].replace(0.0, np.nan) * 100.0
    data["movement_speed_mps"] = np.nan
    data["movement_grade_pct"] = np.nan
    data.loc[usable_timestamp, "movement_speed_mps"] = rolling_speed.to_numpy()
    data.loc[usable_timestamp, "movement_grade_pct"] = rolling_grade.clip(-60.0, 60.0).to_numpy()
    data["moving_interval"] = (
        interval_valid
        & data["movement_speed_mps"].between(
            MINIMUM_MOVEMENT_SPEED_MPS, MAXIMUM_MOVEMENT_SPEED_MPS
        )
    )
    data.attrs.update(source_attrs)
    return data


def analyze_activity(frame: pd.DataFrame, name: str | None = None) -> dict[str, float | str | None]:
    """Calculate the V0.1 basic metrics for one FIT activity."""
    data = add_interval_metrics(frame)
    valid = data["valid_interval"].fillna(False)
    distance_m = float(data.loc[valid, "dd_m"].sum())
    duration_s = float(data.loc[valid, "dt_seconds"].sum())
    elevation = data.loc[valid, "delev_m"].dropna()

    return {
        "name": name,
        "distance_km": round(distance_m / 1000.0, 3),
        "duration_hour": round(duration_s / 3600.0, 3),
        "elevation_gain": round(float(elevation.clip(lower=0).sum()), 1),
        "elevation_loss": round(float(-elevation.clip(upper=0).sum()), 1),
        "avg_hr": _safe_mean(data["heart_rate"]),
        "max_hr": _safe_max(data["heart_rate"]),
        "avg_cadence": _safe_mean(data["cadence"]),
        "avg_temperature": _estimated_ambient_mean(data),
        "avg_device_temperature": _safe_mean(data["device_temperature"]) if "device_temperature" in data else None,
    }


def _safe_mean(series: pd.Series) -> float | None:
    value = series.dropna().mean()
    return None if pd.isna(value) else round(float(value), 1)


def _safe_max(series: pd.Series) -> float | None:
    value = series.dropna().max()
    return None if pd.isna(value) else round(float(value), 1)


def _estimated_ambient_mean(data: pd.DataFrame) -> float | None:
    if "device_temperature" not in data:
        return _safe_mean(data["temperature"]) if "temperature" in data else None
    return None
