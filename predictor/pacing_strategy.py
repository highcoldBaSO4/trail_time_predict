from __future__ import annotations

from typing import Any

import numpy as np

from predictor.duration_adjustment import duration_match


TERRAINS = ("flat", "uphill", "downhill")
DEFAULT_PHASE_CENTERS = (0.125, 0.375, 0.625, 0.875)


def match_route_pacing_strategy(
    profile: dict[str, Any], segments: list[dict[str, Any]], estimated_hours: float
) -> dict[str, Any]:
    """Match a target GPX to historical distance/ascent pacing strategies."""
    target = route_features(segments)
    strategy_profile = dict(profile.get("pacing_strategy", {}))
    candidates = []
    for sample in strategy_profile.get("samples", []):
        similarity = _similarity(target, sample)
        if similarity <= 0:
            continue
        candidates.append((similarity, sample))
    candidates.sort(key=lambda item: item[0], reverse=True)
    candidates = candidates[:5]

    best_similarity = candidates[0][0] if candidates else 0.0
    if not candidates or best_similarity < 0.22:
        return _duration_fallback(profile, estimated_hours, target, best_similarity)

    weights = np.asarray([
        similarity ** 2
        * float(sample.get("confidence", 0.2))
        * float(sample.get("model_weight", 1.0))
        for similarity, sample in candidates
    ], dtype=float)
    if weights.sum() <= 0:
        return _duration_fallback(profile, estimated_hours, target, best_similarity)

    curves: dict[str, list[float]] = {}
    for terrain in TERRAINS:
        values = np.asarray([
            list(sample.get("terrain_curves", {}).get(terrain, sample.get("overall_curve", [1.0] * 4)))
            for _, sample in candidates
        ], dtype=float)
        curve = np.average(values, axis=0, weights=weights)
        curves[terrain] = [round(float(np.clip(value, 0.78, 1.28)), 4) for value in curve]

    overall_values = np.asarray([sample.get("overall_curve", [1.0] * 4) for _, sample in candidates], dtype=float)
    overall_curve = np.average(overall_values, axis=0, weights=weights)
    confidence_values = np.asarray([float(sample.get("confidence", 0.2)) for _, sample in candidates])
    evidence = min(1.0, len(candidates) / 3.0)
    confidence = float(np.average(confidence_values, weights=weights)) * (0.55 + 0.45 * evidence) * best_similarity
    confidence = max(0.2, min(0.92, confidence))
    source = "historical_route_strategy"
    matched = [
        {
            "activity": str(sample.get("activity", "")),
            "distance_km": float(sample.get("distance_km", 0.0)),
            "elevation_gain_m": float(sample.get("elevation_gain_m", 0.0)),
            "strategy_type": str(sample.get("strategy_type", "unknown")),
            "similarity": round(float(similarity), 3),
            "weight": round(float(weight / weights.sum()), 3),
        }
        for (similarity, sample), weight in zip(candidates, weights)
    ]
    return {
        "source": source,
        "target": target,
        "phase_centers": list(strategy_profile.get("phase_centers", DEFAULT_PHASE_CENTERS)),
        "overall_curve": [round(float(value), 4) for value in overall_curve],
        "terrain_curves": curves,
        "confidence": round(confidence, 3),
        "best_similarity": round(best_similarity, 3),
        "matched_activities": matched,
        "strategy_type": _strategy_type(overall_curve),
        "fallback_reason": None,
    }


def pacing_factor(match: dict[str, Any], terrain: str, progress: float) -> dict[str, Any]:
    centers = np.asarray(match.get("phase_centers", DEFAULT_PHASE_CENTERS), dtype=float)
    curve = np.asarray(match.get("terrain_curves", {}).get(terrain, [1.0] * len(centers)), dtype=float)
    factor = float(np.interp(float(np.clip(progress, 0.0, 1.0)), centers, curve))
    return {
        "factor": factor,
        "confidence": float(match.get("confidence", 0.2)),
        "source": str(match.get("source", "duration_fallback")),
        "weights": {
            item["activity"]: float(item["weight"])
            for item in match.get("matched_activities", [])
        },
    }


def route_features(segments: list[dict[str, Any]]) -> dict[str, Any]:
    distance_m = sum(max(0.0, float(segment.get("distance", 0.0))) for segment in segments)
    gain = sum(max(0.0, float(segment.get("gain", 0.0))) for segment in segments)
    loss = sum(max(0.0, float(segment.get("loss", 0.0))) for segment in segments)
    distance_km = distance_m / 1000.0
    terrain_distance = {
        terrain: sum(
            max(0.0, float(segment.get("distance", 0.0)))
            for segment in segments if str(segment.get("type", "flat")) == terrain
        )
        for terrain in TERRAINS
    }
    return {
        "distance_km": round(distance_km, 3),
        "elevation_gain_m": round(gain, 1),
        "elevation_loss_m": round(loss, 1),
        "climb_density_m_per_km": round(gain / max(distance_km, 0.001), 2),
        "load_km": round(distance_km + gain / 100.0, 3),
        "terrain_share": {
            terrain: round(terrain_distance[terrain] / max(distance_m, 0.001), 4)
            for terrain in TERRAINS
        },
    }


def _similarity(target: dict[str, Any], sample: dict[str, Any]) -> float:
    distance = _ratio_similarity(float(target["distance_km"]), float(sample.get("distance_km", 0.0)), 0.75)
    gain = _ratio_similarity(float(target["elevation_gain_m"]), float(sample.get("elevation_gain_m", 0.0)), 1.00, offset=100.0)
    density = _ratio_similarity(
        float(target["climb_density_m_per_km"]),
        float(sample.get("climb_density_m_per_km", 0.0)),
        0.85,
        offset=10.0,
    )
    load = _ratio_similarity(float(target["load_km"]), float(sample.get("load_km", 0.0)), 0.80)
    target_share = target.get("terrain_share", {})
    sample_share = sample.get("terrain_share", {})
    terrain_difference = sum(abs(float(target_share.get(key, 0.0)) - float(sample_share.get(key, 0.0))) for key in TERRAINS) / 2.0
    terrain = max(0.0, 1.0 - terrain_difference)
    activity_type_penalty = 1.0 if sample.get("activity_type") == "trail" else 0.78
    return float(np.clip((0.30 * distance + 0.23 * gain + 0.17 * density + 0.15 * load + 0.15 * terrain) * activity_type_penalty, 0.0, 1.0))


def _ratio_similarity(left: float, right: float, scale: float, offset: float = 0.01) -> float:
    if left < 0 or right < 0:
        return 0.0
    return float(np.exp(-abs(np.log((left + offset) / (right + offset))) / scale))


def _duration_fallback(
    profile: dict[str, Any], estimated_hours: float, target: dict[str, Any], best_similarity: float
) -> dict[str, Any]:
    terrain_curves = {}
    confidences = []
    for terrain in TERRAINS:
        match = duration_match(profile, estimated_hours, terrain)
        terrain_curves[terrain] = [float(match["factor"])] * 4
        confidences.append(float(match["confidence"]))
    return {
        "source": "duration_fallback",
        "target": target,
        "phase_centers": list(DEFAULT_PHASE_CENTERS),
        "overall_curve": [float(np.mean([terrain_curves[key][0] for key in TERRAINS]))] * 4,
        "terrain_curves": terrain_curves,
        "confidence": round(float(np.mean(confidences)), 3),
        "best_similarity": round(float(best_similarity), 3),
        "matched_activities": [],
        "strategy_type": "duration_fallback",
        "fallback_reason": "no_similar_historical_route",
    }


def _strategy_type(curve: np.ndarray) -> str:
    delta = float(curve[-1] - curve[0])
    if delta <= -0.04:
        return "negative_split"
    if delta >= 0.04:
        return "positive_split"
    if float(np.max(curve) - np.min(curve)) <= 0.06:
        return "even"
    return "variable"
