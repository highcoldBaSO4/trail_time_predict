from __future__ import annotations

import numpy as np
import pandas as pd


def add_interval_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add robust point-to-point metrics used by all capability models."""
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
    }


def _safe_mean(series: pd.Series) -> float | None:
    value = series.dropna().mean()
    return None if pd.isna(value) else round(float(value), 1)


def _safe_max(series: pd.Series) -> float | None:
    value = series.dropna().max()
    return None if pd.isna(value) else round(float(value), 1)

