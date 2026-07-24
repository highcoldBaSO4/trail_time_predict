from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.confidence import calculate_confidence
from config import load_config


TERRAIN_METRICS = {"flat": "speed_mps", "uphill": "vam", "downhill": "speed_mps"}
FATIGUE_STAGE_THRESHOLDS = (("fresh_end_hour", 0.97), ("mild_end_hour", 0.90), ("moderate_end_hour", 0.80))
FATIGUE_STAGE_FALLBACK_HOURS = {"fresh_end_hour": 3.0, "mild_end_hour": 5.0, "moderate_end_hour": 8.0}


def interpolate_fatigue(elapsed_hours: float, points: list[tuple[float, float]] | list[dict[str, float]]) -> float:
    """Continuously interpolate retained performance over elapsed time."""
    if not points:
        raise ValueError("疲劳曲线不能为空")
    normalized = [(float(p["hour"]), float(p["factor"])) if isinstance(p, dict) else (float(p[0]), float(p[1])) for p in points]
    normalized.sort()
    elapsed = max(0.0, float(elapsed_hours))
    last_hour, last_factor = normalized[-1]
    if elapsed <= last_hour:
        return float(np.interp(elapsed, [p[0] for p in normalized], [p[1] for p in normalized]))

    # New profiles carry an explicit extrapolated tail point.  For old profiles
    # retain the historical flat-tail behaviour so they remain reproducible.
    last_point = max(
        (point for point in points if isinstance(point, dict)),
        key=lambda point: float(point["hour"]),
        default=None,
    )
    if not isinstance(last_point, dict) or "tail_decline_per_hour" not in last_point:
        return float(last_factor)
    if str(last_point.get("source")) == "extrapolated":
        return float(last_factor)
    decline = max(0.0, float(last_point["tail_decline_per_hour"]))
    maximum = max(0.0, float(last_point.get("maximum_extrapolation_hours", 0.0)))
    floor = float(last_point.get("minimum_retention_factor", 0.45))
    extra = min(maximum, elapsed - last_hour)
    return float(max(floor, last_factor * (1.0 - decline) ** extra))


def build_fatigue_profile(activities: list[pd.DataFrame]) -> dict[str, object]:
    """Build terrain-specific fatigue curves from eligible node windows.

    A node is evidence only when its activity reaches that duration.  This
    deliberately avoids treating an activity's first 3/5/8 hours as the
    retained ability *at* 3/5/8 hours.
    """
    defaults = load_config()["default_profile"]["fatigue"]
    config = load_config()["fatigue_model"]
    node_window_hours = float(config["node_window_minutes"]) / 60.0
    minimum_window_seconds = float(config["minimum_window_minutes"]) * 60.0
    coverage_ratio = float(config["minimum_activity_coverage_ratio"])
    fresh_start = float(config["fresh_reference_start_hour"])
    fresh_end = float(config["fresh_reference_end_hour"])
    full_node_activity_count = max(1, int(config["full_node_activity_count"]))
    curves: dict[str, list[dict[str, float | int | str]]] = {}
    for terrain in ("flat", "uphill", "downhill"):
        observations: dict[float, list[dict[str, float]]] = {3.0: [], 5.0: [], 8.0: []}
        for activity in activities:
            samples = _terrain_samples(activity, terrain)
            baseline_window = samples[samples["elapsed_h"].between(fresh_start, fresh_end, inclusive="both")]
            baseline = _metric(baseline_window, terrain)
            if not np.isfinite(baseline) or baseline <= 0:
                continue
            activity_hours = _activity_moving_hours(activity)
            for hour in observations:
                if activity_hours < hour * coverage_ratio:
                    continue
                window = samples[samples["elapsed_h"].between(hour - node_window_hours, hour, inclusive="both")]
                window_seconds = float(window["dt_seconds"].sum()) if not window.empty else 0.0
                value = _metric(window, terrain)
                if window_seconds >= minimum_window_seconds and np.isfinite(value):
                    observations[hour].append(
                        {
                            "factor": float(np.clip(value / baseline, 0.45, 1.05)),
                            "duration_seconds": window_seconds,
                            "activity_hours": activity_hours,
                        }
                    )
        curve = []
        for default in defaults[terrain]:
            hour = float(default["hour"])
            if hour == 0.0:
                curve.append({"hour": 0.0, "factor": 1.0, "sample_count": 0,
                              "source": "anchor", "confidence": None})
                continue
            values = observations.get(hour, [])
            observed_count = len(values)
            observed_duration = sum(item["duration_seconds"] for item in values)
            observed_max_hours = max((item["activity_hours"] for item in values), default=0.0)
            personal = float(np.median([item["factor"] for item in values])) if values else float(default["factor"])
            blend_weight = min(1.0, observed_count / full_node_activity_count)
            source = "personal" if blend_weight >= 1.0 else "blended" if values else "default"
            factor = personal * blend_weight + float(default["factor"]) * (1.0 - blend_weight)
            confidence = calculate_confidence(observed_duration, observed_count, source="personal" if values else "default")
            curve.append({"hour": hour, "factor": round(factor, 3), "sample_count": observed_count,
                          "observed_activity_count": observed_count, "observed_duration_seconds": round(observed_duration, 1),
                          "observed_max_hours": round(observed_max_hours, 3), "source": source,
                          "confidence": confidence, "extrapolation_distance_hours": 0.0})
        _enforce_monotonic_retention(curve)
        last = curve[-1]
        tail_hours = float(config["maximum_extrapolation_hours"])
        tail_factor = max(
            float(config["minimum_retention_factor"]),
            float(last["factor"]) * (1.0 - float(config["tail_decline_prior"])) ** tail_hours,
        )
        curve.append({
            "hour": round(float(last["hour"]) + tail_hours, 3), "factor": round(tail_factor, 3),
            "sample_count": 0, "observed_activity_count": 0, "observed_duration_seconds": 0.0,
            "observed_max_hours": float(last.get("observed_max_hours", 0.0)), "source": "extrapolated",
            "confidence": calculate_confidence(0.0, 0, source="default"),
            "extrapolation_distance_hours": tail_hours,
            "tail_decline_per_hour": float(config["tail_decline_prior"]),
            "maximum_extrapolation_hours": tail_hours,
            "minimum_retention_factor": float(config["minimum_retention_factor"]),
        })
        curves[terrain] = curve
    # Legacy keys keep the V0.1 UI/report functional during Phase 1.
    curves["3h"] = interpolate_fatigue(3.0, curves["flat"])
    curves["5h"] = interpolate_fatigue(5.0, curves["flat"])
    curves["8h"] = interpolate_fatigue(8.0, curves["flat"])
    return curves


def _activity_moving_hours(activity: pd.DataFrame) -> float:
    moving = activity.get("moving_interval", pd.Series(False, index=activity.index)).fillna(False)
    seconds = activity.get("dt_seconds", pd.Series(0.0, index=activity.index)).where(moving, 0.0).fillna(0.0).sum()
    return max(0.0, float(seconds) / 3600.0)


def _enforce_monotonic_retention(curve: list[dict[str, object]]) -> None:
    retained = 1.0
    for point in curve:
        factor = min(retained, max(0.45, float(point["factor"])))
        point["factor"] = round(factor, 3)
        retained = factor


def build_fatigue_stages(fatigue_profile: dict[str, object]) -> dict[str, object]:
    """Convert continuous fatigue curves into explainable retention bands."""
    terrain_stages: dict[str, dict[str, float]] = {}
    for terrain in ("flat", "uphill", "downhill"):
        curve = list(fatigue_profile.get(terrain, []))
        terrain_stages[terrain] = {
            name: round(_retention_crossing_hour(curve, retention, FATIGUE_STAGE_FALLBACK_HOURS[name]), 3)
            for name, retention in FATIGUE_STAGE_THRESHOLDS
        }
    overall = {
        name: round(min(terrain_stages[terrain][name] for terrain in terrain_stages), 3)
        for name, _ in FATIGUE_STAGE_THRESHOLDS
    }
    return {
        "thresholds": {name: retention for name, retention in FATIGUE_STAGE_THRESHOLDS},
        "terrain": terrain_stages,
        "overall": overall,
        "method": "continuous_retention_crossings",
    }


def fatigue_stage_for_duration(duration_hours: float, stages: dict[str, object]) -> str:
    overall = dict(stages.get("overall", {}))
    duration = max(0.0, float(duration_hours))
    if duration <= float(overall.get("fresh_end_hour", 3.0)):
        return "fresh"
    if duration <= float(overall.get("mild_end_hour", 5.0)):
        return "mild"
    if duration <= float(overall.get("moderate_end_hour", 8.0)):
        return "moderate"
    return "severe"


def _retention_crossing_hour(curve: list[dict[str, object]], retention: float, fallback: float) -> float:
    points = sorted((float(point["hour"]), float(point["factor"])) for point in curve)
    if not points:
        return fallback
    for (left_hour, left_factor), (right_hour, right_factor) in zip(points, points[1:]):
        if left_factor >= retention >= right_factor and left_factor != right_factor:
            weight = (left_factor - retention) / (left_factor - right_factor)
            return left_hour + weight * (right_hour - left_hour)
        if left_factor == retention:
            return left_hour
    if points[-1][1] <= retention:
        return points[-1][0]
    return fallback


def _terrain_samples(activity: pd.DataFrame, terrain: str) -> pd.DataFrame:
    moving = activity["moving_interval"].fillna(False)
    moving_elapsed = activity["dt_seconds"].where(moving, 0.0).fillna(0.0).cumsum() / 3600.0
    valid = activity[moving].copy()
    valid["elapsed_h"] = moving_elapsed.loc[valid.index]
    valid["grade_pct"] = valid["movement_grade_pct"]
    valid["speed_mps"] = valid["movement_speed_mps"]
    grade = valid["grade_pct"]
    flat_limit = float(load_config()["terrain"]["flat_grade_abs_percent"])
    mask = (
        grade.between(-flat_limit, flat_limit)
        if terrain == "flat"
        else grade > flat_limit
        if terrain == "uphill"
        else grade < -flat_limit
    )
    selected = valid[mask].copy()
    selected["vam"] = (
        selected["speed_mps"] * selected["grade_pct"].clip(lower=0.0) / 100.0 * 3600.0
    )
    return selected


def _metric(frame: pd.DataFrame, terrain: str) -> float:
    seconds = float(frame["dt_seconds"].sum()) if not frame.empty else 0.0
    if seconds <= 0:
        return np.nan
    return float(frame["delev_m"].clip(lower=0).sum() / seconds * 3600.0) if terrain == "uphill" else float(frame["dd_m"].sum() / seconds)
