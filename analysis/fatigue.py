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
    return float(np.interp(max(0.0, float(elapsed_hours)), [p[0] for p in normalized], [p[1] for p in normalized]))


def build_fatigue_profile(activities: list[pd.DataFrame]) -> dict[str, object]:
    """Build separate terrain-normalized fatigue curves with safe defaults."""
    defaults = load_config()["default_profile"]["fatigue"]
    curves: dict[str, list[dict[str, float | int | str]]] = {}
    for terrain in ("flat", "uphill", "downhill"):
        ratios: dict[float, list[float]] = {3.0: [], 5.0: [], 8.0: []}
        for activity in activities:
            samples = _terrain_samples(activity, terrain)
            baseline = _metric(samples[samples["elapsed_h"] <= 3.0], terrain)
            if not np.isfinite(baseline) or baseline <= 0:
                continue
            for hour, low in ((3.0, 0.0), (5.0, 3.0), (8.0, 5.0)):
                value = _metric(samples[(samples["elapsed_h"] > low) & (samples["elapsed_h"] <= hour)], terrain)
                if np.isfinite(value):
                    ratios[hour].append(float(np.clip(value / baseline, 0.45, 1.05)))
        curve = []
        for default in defaults[terrain]:
            hour = float(default["hour"])
            if hour == 0.0:
                curve.append({"hour": 0.0, "factor": 1.0, "sample_count": 0,
                              "source": "anchor", "confidence": None})
                continue
            values = ratios.get(hour, [])
            source = "personal" if values else "default"
            curve.append({"hour": hour, "factor": round(float(np.median(values)), 3) if values else float(default["factor"]),
                          "sample_count": len(values), "source": source,
                          "confidence": calculate_confidence(len(values) * 1800.0, len(values), source=source)})
        curves[terrain] = curve
    # Legacy keys keep the V0.1 UI/report functional during Phase 1.
    curves["3h"] = interpolate_fatigue(3.0, curves["flat"])
    curves["5h"] = interpolate_fatigue(5.0, curves["flat"])
    curves["8h"] = interpolate_fatigue(8.0, curves["flat"])
    return curves


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
