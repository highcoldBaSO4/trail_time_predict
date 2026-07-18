from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analysis.downhill import interpolate_downhill_speed
from analysis.fatigue import build_fatigue_stages, fatigue_stage_for_duration, interpolate_fatigue
from analysis.uphill import interpolate_uphill_vam


TERRAINS = ("flat", "uphill", "downhill")
PHASE_EDGES = (0.0, 0.25, 0.50, 0.75, 1.000001)
PHASE_CENTERS = (0.125, 0.375, 0.625, 0.875)


def build_pacing_strategy_profile(
    segments: pd.DataFrame,
    flat: dict[str, Any],
    uphill: dict[str, Any],
    downhill: dict[str, Any],
    fatigue: dict[str, Any],
    quality_score: float,
) -> dict[str, Any]:
    """Extract terrain-normalised whole-activity pacing curves.

    The saved factor is a *time* factor: values below one mean a more
    aggressive use of the runner's fresh terrain ability.  Fatigue is removed
    before the curve is calculated, so the predictor can apply it separately.
    """
    if segments.empty:
        return {"version": 1, "phase_centers": list(PHASE_CENTERS), "samples": [], "confidence": 0.2}

    prepared = segments.copy()
    prepared["distance_progress"] = 0.0
    for _, indexes in prepared.groupby("activity", sort=False).groups.items():
        selected = prepared.loc[indexes]
        total = float(selected["distance_m"].clip(lower=0).sum())
        if total > 0:
            starts = selected["distance_m"].clip(lower=0).cumsum() - selected["distance_m"].clip(lower=0)
            prepared.loc[indexes, "distance_progress"] = (
                starts + selected["distance_m"].clip(lower=0) / 2.0
            ) / total

    fatigue_stages = build_fatigue_stages(fatigue)
    samples: list[dict[str, Any]] = []
    for activity_name, activity in prepared.groupby("activity", sort=False):
        sample = _activity_strategy(activity, flat, uphill, downhill, fatigue, quality_score)
        if sample is not None:
            sample["activity"] = str(activity_name)
            sample["fatigue_stage"] = fatigue_stage_for_duration(float(sample["duration_hours"]), fatigue_stages)
            samples.append(sample)

    if not samples:
        confidence = 0.2
    else:
        weights = np.asarray([float(item["model_weight"]) for item in samples], dtype=float)
        values = np.asarray([float(item["confidence"]) for item in samples], dtype=float)
        confidence = float(np.average(values, weights=np.maximum(weights, 1e-6)))
    return {
        "version": 1,
        "phase_centers": list(PHASE_CENTERS),
        "samples": samples,
        "fatigue_stages": fatigue_stages,
        "confidence": round(max(0.2, min(0.95, confidence)), 3),
    }


def _activity_strategy(
    activity: pd.DataFrame,
    flat: dict[str, Any],
    uphill: dict[str, Any],
    downhill: dict[str, Any],
    fatigue: dict[str, Any],
    quality_score: float,
) -> dict[str, Any] | None:
    rows: list[dict[str, float | str]] = []
    for _, segment in activity.iterrows():
        baseline = _baseline_seconds(segment, flat, uphill, downhill)
        observed = float(segment["duration_s"])
        if baseline <= 0 or observed <= 0:
            continue
        terrain = str(segment["type"])
        retained = interpolate_fatigue(
            float(segment.get("activity_elapsed_h", 0.0)), list(fatigue.get(terrain, []))
        )
        # observed = baseline * strategy_time_factor / retained_ability
        strategy_factor = observed / baseline * max(retained, 0.1)
        if not np.isfinite(strategy_factor):
            continue
        rows.append({
            "terrain": terrain,
            "progress": float(segment["distance_progress"]),
            "factor": float(np.clip(strategy_factor, 0.70, 1.45)),
            "duration": observed,
            "distance": float(segment["distance_m"]),
            "gain": float(segment.get("gain_m", 0.0)),
            "loss": float(segment.get("loss_m", 0.0)),
            "model_weight": float(segment.get("model_weight", 1.0)),
        })
    if not rows:
        return None

    frame = pd.DataFrame(rows)
    total_distance = float(frame["distance"].sum())
    total_duration = float(frame["duration"].sum())
    if total_distance <= 0 or total_duration < 600:
        return None

    overall_curve = _phase_curve(frame)
    terrain_curves: dict[str, list[float]] = {}
    terrain_counts: dict[str, int] = {}
    for terrain in TERRAINS:
        selected = frame[frame["terrain"] == terrain]
        terrain_counts[terrain] = len(selected)
        terrain_curves[terrain] = _phase_curve(selected, overall_curve)

    first, last = overall_curve[0], overall_curve[-1]
    delta = last - first
    if delta <= -0.04:
        strategy_type = "negative_split"
    elif delta >= 0.04:
        strategy_type = "positive_split"
    elif max(overall_curve) - min(overall_curve) <= 0.06:
        strategy_type = "even"
    else:
        strategy_type = "variable"

    activity_type = str(activity["activity_type"].iloc[0])
    model_weight = float(activity["model_weight"].median())
    terrain_distance = frame.groupby("terrain")["distance"].sum().to_dict()
    usable_phases = sum(
        bool(((frame["progress"] >= PHASE_EDGES[index]) & (frame["progress"] < PHASE_EDGES[index + 1])).any())
        for index in range(4)
    )
    coverage = usable_phases / 4.0
    sample_strength = min(1.0, total_duration / 7200.0)
    confidence = 0.2 + 0.35 * coverage + 0.25 * sample_strength + 0.15 * float(quality_score)
    if activity_type != "trail":
        confidence *= 0.82

    gain = float(frame["gain"].sum())
    loss = float(frame["loss"].sum())
    return {
        "activity_type": activity_type,
        "distance_km": round(total_distance / 1000.0, 3),
        "elevation_gain_m": round(gain, 1),
        "elevation_loss_m": round(loss, 1),
        "climb_density_m_per_km": round(gain / max(total_distance / 1000.0, 0.001), 2),
        "load_km": round(total_distance / 1000.0 + gain / 100.0, 3),
        "duration_hours": round(total_duration / 3600.0, 3),
        "terrain_share": {
            terrain: round(float(terrain_distance.get(terrain, 0.0)) / total_distance, 4)
            for terrain in TERRAINS
        },
        "overall_curve": [round(value, 4) for value in overall_curve],
        "terrain_curves": {terrain: [round(value, 4) for value in curve] for terrain, curve in terrain_curves.items()},
        "strategy_type": strategy_type,
        "negative_split_delta": round(delta, 4),
        "terrain_sample_count": terrain_counts,
        "confidence": round(max(0.2, min(0.90, confidence)), 3),
        "model_weight": round(max(0.05, model_weight), 3),
    }


def _phase_curve(frame: pd.DataFrame, fallback: list[float] | None = None) -> list[float]:
    result: list[float] = []
    for index in range(4):
        selected = frame[
            (frame["progress"] >= PHASE_EDGES[index])
            & (frame["progress"] < PHASE_EDGES[index + 1])
        ]
        if selected.empty:
            result.append(float(fallback[index]) if fallback is not None else np.nan)
            continue
        result.append(_weighted_median(
            selected["factor"].to_numpy(dtype=float),
            selected["duration"].to_numpy(dtype=float) * selected["model_weight"].to_numpy(dtype=float),
        ))
    if fallback is None:
        finite = np.asarray(result, dtype=float)
        known = np.isfinite(finite)
        if not known.any():
            return [1.0] * 4
        finite = np.interp(np.arange(4), np.flatnonzero(known), finite[known])
        result = finite.tolist()
    return [float(value) for value in result]


def _baseline_seconds(
    segment: pd.Series, flat: dict[str, Any], uphill: dict[str, Any], downhill: dict[str, Any]
) -> float:
    terrain = str(segment["type"])
    distance = float(segment["distance_m"])
    grade = float(segment.get("grade_pct", 0.0))
    if terrain == "uphill":
        curve = [(float(point["grade"]), float(point["value"])) for point in uphill.get("curve", [])]
        if not curve:
            return 0.0
        vam = interpolate_uphill_vam(grade, curve)
        return float(segment.get("gain_m", 0.0)) / max(vam, 1.0) * 3600.0
    if terrain == "downhill":
        curve = [(float(point["grade"]), float(point["speed_mps"])) for point in downhill.get("curve", [])]
        if not curve:
            return 0.0
        speed = interpolate_downhill_speed(grade, curve)
        return distance / max(speed, 0.1)
    return distance / 1000.0 * float(flat["aerobic_pace"])


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = np.maximum(weights[order], 0.0)
    if weights.sum() <= 0:
        return float(np.median(values))
    index = min(int(np.searchsorted(np.cumsum(weights), weights.sum() / 2.0, side="left")), len(values) - 1)
    return float(values[index])
