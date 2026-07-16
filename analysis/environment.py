from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from config import load_config


def solar_elevation_degrees(timestamp: datetime | pd.Timestamp, latitude: float, longitude: float) -> float:
    """Approximate solar elevation from an absolute time and geographic position."""
    moment = pd.Timestamp(timestamp)
    if moment.tzinfo is None:
        moment = moment.tz_localize("UTC")
    else:
        moment = moment.tz_convert("UTC")
    julian_day = moment.timestamp() / 86400.0 + 2440587.5
    days = julian_day - 2451545.0
    mean_longitude = math.radians((280.460 + 0.9856474 * days) % 360.0)
    mean_anomaly = math.radians((357.528 + 0.9856003 * days) % 360.0)
    ecliptic_longitude = mean_longitude + math.radians(1.915) * math.sin(mean_anomaly) + math.radians(0.020) * math.sin(2 * mean_anomaly)
    obliquity = math.radians(23.439 - 0.0000004 * days)
    right_ascension = math.atan2(math.cos(obliquity) * math.sin(ecliptic_longitude), math.cos(ecliptic_longitude))
    declination = math.asin(math.sin(obliquity) * math.sin(ecliptic_longitude))
    sidereal = math.radians((280.46061837 + 360.98564736629 * days + float(longitude)) % 360.0)
    hour_angle = (sidereal - right_ascension + math.pi) % (2 * math.pi) - math.pi
    latitude_radians = math.radians(float(latitude))
    elevation = math.asin(
        math.sin(latitude_radians) * math.sin(declination)
        + math.cos(latitude_radians) * math.cos(declination) * math.cos(hour_angle)
    )
    return math.degrees(elevation)


def is_night(timestamp: datetime | pd.Timestamp, latitude: float, longitude: float) -> bool:
    """Return whether the sun is below the configured twilight threshold."""
    threshold = float(load_config()["environment"]["night_solar_elevation_degrees"])
    return solar_elevation_degrees(timestamp, latitude, longitude) <= threshold


def build_environment_profile(activities: list[pd.DataFrame]) -> dict[str, Any]:
    """Summarize historical night exposure and altitude coverage from FIT tracks."""
    night_seconds = 0.0
    geo_seconds = 0.0
    terrain_night_seconds = {"flat": 0.0, "uphill": 0.0, "downhill": 0.0}
    terrain_geo_seconds = {"flat": 0.0, "uphill": 0.0, "downhill": 0.0}
    flat_limit = float(load_config()["terrain"]["flat_grade_abs_percent"])
    altitude_values: list[float] = []
    altitude_weights: list[float] = []
    for frame in activities:
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        seconds = timestamps.diff().dt.total_seconds()
        latitude = pd.to_numeric(frame["latitude"], errors="coerce")
        longitude = pd.to_numeric(frame["longitude"], errors="coerce")
        altitude = pd.to_numeric(frame["altitude"], errors="coerce")
        grade_source = frame["grade_pct"] if "grade_pct" in frame else pd.Series(np.nan, index=frame.index)
        grade = pd.to_numeric(grade_source, errors="coerce")
        valid_time = seconds.between(0.2, 120.0)
        for index in frame.index[valid_time]:
            duration = float(seconds.loc[index])
            if np.isfinite(latitude.loc[index]) and np.isfinite(longitude.loc[index]):
                geo_seconds += duration
                terrain = (
                    "uphill" if np.isfinite(grade.loc[index]) and grade.loc[index] > flat_limit
                    else "downhill" if np.isfinite(grade.loc[index]) and grade.loc[index] < -flat_limit
                    else "flat"
                )
                terrain_geo_seconds[terrain] += duration
                night = is_night(timestamps.loc[index], float(latitude.loc[index]), float(longitude.loc[index]))
                if night:
                    night_seconds += duration
                    terrain_night_seconds[terrain] += duration
            if np.isfinite(altitude.loc[index]):
                altitude_values.append(float(altitude.loc[index]))
                altitude_weights.append(duration)

    if altitude_values:
        values = np.asarray(altitude_values, dtype=float)
        weights = np.asarray(altitude_weights, dtype=float)
        mean_altitude = float(np.average(values, weights=weights))
        p90_altitude = _weighted_percentile(values, weights, 90.0)
        maximum_altitude = float(np.max(values))
        altitude_seconds = float(weights.sum())
    else:
        mean_altitude = p90_altitude = maximum_altitude = 0.0
        altitude_seconds = 0.0
    return {
        "night": {
            "ratio": round(night_seconds / geo_seconds, 4) if geo_seconds > 0 else 0.0,
            "night_seconds": round(night_seconds, 1),
            "sample_duration_seconds": round(geo_seconds, 1),
            "source": "fit_coordinates" if geo_seconds > 0 else "unavailable",
            "terrain": {
                terrain: {
                    "ratio": round(terrain_night_seconds[terrain] / terrain_geo_seconds[terrain], 4)
                    if terrain_geo_seconds[terrain] > 0 else 0.0,
                    "night_seconds": round(terrain_night_seconds[terrain], 1),
                    "sample_duration_seconds": round(terrain_geo_seconds[terrain], 1),
                }
                for terrain in ("flat", "uphill", "downhill")
            },
        },
        "altitude": {
            "mean_m": round(mean_altitude, 1),
            "p90_m": round(p90_altitude, 1),
            "max_m": round(maximum_altitude, 1),
            "sample_duration_seconds": round(altitude_seconds, 1),
            "source": "fit_altitude" if altitude_seconds > 0 else "unavailable",
        },
    }


def relative_altitude_factor(target_elevation_m: float, historical_elevation_m: float) -> float:
    """Return an incremental altitude factor relative to historical training altitude."""
    settings = load_config()["environment"]["altitude"]
    target = _absolute_altitude_factor(target_elevation_m, settings)
    baseline = _absolute_altitude_factor(historical_elevation_m, settings)
    return max(float(settings["minimum_relative_factor"]), min(float(settings["maximum_relative_factor"]), target / baseline))


def _absolute_altitude_factor(elevation_m: float, settings: dict[str, Any]) -> float:
    excess = max(0.0, float(elevation_m) - float(settings["threshold_m"]))
    return 1.0 + excess / 1000.0 * float(settings["time_increase_per_1000m"])


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    target = percentile / 100.0 * ordered_weights.sum()
    index = min(int(np.searchsorted(np.cumsum(ordered_weights), target, side="left")), len(ordered_values) - 1)
    return float(ordered_values[index])
