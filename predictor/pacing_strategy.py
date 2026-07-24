from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from analysis.route_features import route_structure_features
from config import load_config
from predictor.duration_adjustment import duration_match


TERRAINS = ("flat", "uphill", "downhill")
DEFAULT_PHASE_CENTERS = (0.125, 0.375, 0.625, 0.875)
GRADE_BAND_KEYS = tuple(
    f"{direction}_{threshold}_{measure}_share"
    for direction, measure in (("uphill", "distance"), ("uphill", "gain"), ("downhill", "distance"), ("downhill", "loss"))
    for threshold in (10, 15, 20)
)
CONTINUOUS_KEYS = (
    "longest_uphill_distance_km", "longest_uphill_gain_m", "longest_uphill_average_grade_pct",
    "longest_uphill_start_progress", "longest_uphill_end_progress", "longest_downhill_distance_km",
    "longest_downhill_loss_m", "longest_downhill_average_grade_pct", "longest_downhill_start_progress",
    "longest_downhill_end_progress", "maximum_single_ascent_m", "maximum_single_descent_m",
)
PHASE_KEYS = ("gain_share", "loss_share", "hard_uphill_gain_share", "flat_distance_share", "uphill_distance_share", "downhill_distance_share")
SEQUENCE_KEYS = ("uphill_to_long_downhill_transition_share", "late_hard_uphill_gain_share", "terrain_run_count_per_10km")


def match_route_pacing_strategy(
    profile: dict[str, Any], segments: list[dict[str, Any]], estimated_hours: float,
    matching_mode: str = "structural",
) -> dict[str, Any]:
    """Match a target GPX to historical pacing with route-structure evidence."""
    target = route_features(segments)
    config = dict(load_config().get("route_similarity", {}))
    strategy_profile = dict(profile.get("pacing_strategy", {}))
    if matching_mode not in {"structural", "legacy", "duration_fallback"}:
        raise ValueError("未知路线配速匹配模式")
    if matching_mode == "duration_fallback":
        return _duration_fallback(profile, estimated_hours, target, 0.0, config, [], matching_mode)
    candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for sample in strategy_profile.get("samples", []):
        detail = (
            route_similarity_details(target, sample, config)
            if matching_mode == "structural" else _legacy_similarity_details(target, sample)
        )
        similarity = float(detail["score"])
        if similarity <= 0:
            continue
        candidates.append((similarity, sample, detail))
    candidates.sort(key=lambda item: item[0], reverse=True)
    candidates = candidates[:int(config.get("maximum_matched_activities", 5))]

    best_similarity = candidates[0][0] if candidates else 0.0
    fallback_threshold = max(
        float(config.get("minimum_historical_similarity", 0.22)),
        float(config.get("low_similarity_threshold", 0.42)),
    ) if matching_mode == "structural" else float(config.get("minimum_historical_similarity", 0.22))
    if not candidates or best_similarity < fallback_threshold:
        return _duration_fallback(profile, estimated_hours, target, best_similarity, config, candidates, matching_mode)

    weights = np.asarray([
        similarity ** 2
        * float(sample.get("confidence", 0.2))
        * float(sample.get("model_weight", 1.0))
        for similarity, sample, _ in candidates
    ], dtype=float)
    if weights.sum() <= 0:
        return _duration_fallback(profile, estimated_hours, target, best_similarity, config, candidates, matching_mode)

    curves: dict[str, list[float]] = {}
    for terrain in TERRAINS:
        values = np.asarray([
            list(sample.get("terrain_curves", {}).get(terrain, sample.get("overall_curve", [1.0] * 4)))
            for _, sample, _ in candidates
        ], dtype=float)
        curve = np.average(values, axis=0, weights=weights)
        curves[terrain] = [round(float(np.clip(value, 0.78, 1.28)), 4) for value in curve]

    overall_values = np.asarray([sample.get("overall_curve", [1.0] * 4) for _, sample, _ in candidates], dtype=float)
    overall_curve = np.average(overall_values, axis=0, weights=weights)
    confidence_values = np.asarray([float(sample.get("confidence", 0.2)) for _, sample, _ in candidates])
    evidence = min(1.0, len(candidates) / 3.0)
    missing_structure_factor = 0.88 if matching_mode == "structural" and candidates[0][2]["missing_feature_groups"] else 1.0
    confidence = (
        float(np.average(confidence_values, weights=weights))
        * (0.55 + 0.45 * evidence)
        * best_similarity
        * missing_structure_factor
    )
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
            "similarity_groups": detail["groups"],
            "missing_feature_groups": detail["missing_feature_groups"],
        }
        for (similarity, sample, detail), weight in zip(candidates, weights)
    ]
    uncertainty = _route_uncertainty(target, candidates, config) if matching_mode == "structural" else _empty_route_uncertainty()
    return {
        "source": source,
        "matching_mode": matching_mode,
        "target": target,
        "phase_centers": list(strategy_profile.get("phase_centers", DEFAULT_PHASE_CENTERS)),
        "overall_curve": [round(float(value), 4) for value in overall_curve],
        "terrain_curves": curves,
        "confidence": round(confidence, 3),
        "best_similarity": round(best_similarity, 3),
        "matched_activities": matched,
        "similarity_groups": candidates[0][2]["groups"],
        "missing_feature_groups": candidates[0][2]["missing_feature_groups"],
        "similarity_reasons": candidates[0][2]["reasons"],
        "uncertainty": uncertainty,
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
        "route_structure": route_structure_features(segments),
    }


def route_similarity_details(
    target: dict[str, Any], sample: dict[str, Any], config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return grouped route similarity with an explainable missing-data path."""
    config = dict(load_config().get("route_similarity", {})) if config is None else config
    group_weights = dict(config.get("group_weights", {}))
    groups: dict[str, float] = {"scale": _scale_similarity(target, sample)}
    missing: list[str] = []
    target_structure = _mapping(target.get("route_structure"))
    sample_structure = _mapping(sample.get("route_structure"))
    for name, scorer in (
        ("grade_structure", _grade_structure_similarity),
        ("continuous_slope", _continuous_slope_similarity),
        ("terrain_sequence", _terrain_sequence_similarity),
    ):
        score = scorer(target_structure, sample_structure)
        if score is None:
            missing.append(name)
        else:
            groups[name] = score
    groups["activity_type"] = 1.0 if sample.get("activity_type") == "trail" else 0.78
    usable_weights = {name: float(group_weights.get(name, 0.0)) for name in groups if float(group_weights.get(name, 0.0)) > 0}
    if not usable_weights:
        return {"score": 0.0, "groups": {}, "missing_feature_groups": list(group_weights), "reasons": ["缺少可比较路线特征"]}
    total_weight = sum(usable_weights.values())
    score = float(np.exp(sum(weight / total_weight * np.log(max(groups[name], 0.01)) for name, weight in usable_weights.items())))
    ordered = sorted(groups.items(), key=lambda item: item[1])
    reasons = [_group_reason(name, value) for name, value in ordered[:2] if value < 0.72]
    if missing:
        reasons.append("历史活动缺少部分结构特征，已降低对应特征权重")
    return {
        "score": round(float(np.clip(score, 0.0, 1.0)), 4),
        "groups": {name: round(value, 4) for name, value in groups.items()},
        "missing_feature_groups": missing,
        "reasons": reasons,
    }


def _similarity(target: dict[str, Any], sample: dict[str, Any]) -> float:
    """Compatibility wrapper used by older callers and tests."""
    return float(route_similarity_details(target, sample)["score"])


def _legacy_similarity_details(target: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    """Pre-stage-2 route comparison, retained only for rolling-backtest ablation."""
    score = _scale_similarity(target, sample)
    return {
        "score": round(score, 4),
        "groups": {"legacy_scale": round(score, 4)},
        "missing_feature_groups": [],
        "reasons": ["使用阶段 2 前的路线规模相似度"],
    }


def _scale_similarity(target: dict[str, Any], sample: dict[str, Any]) -> float:
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
    return float(np.clip(0.30 * distance + 0.23 * gain + 0.17 * density + 0.15 * load + 0.15 * terrain, 0.0, 1.0))


def _grade_structure_similarity(target: dict[str, Any], sample: dict[str, Any]) -> float | None:
    return _mapping_similarity(target.get("grade_bands"), sample.get("grade_bands"), scale=0.55, allowed_keys=GRADE_BAND_KEYS)


def _continuous_slope_similarity(target: dict[str, Any], sample: dict[str, Any]) -> float | None:
    return _mapping_similarity(target.get("continuous"), sample.get("continuous"), scale=0.65, allowed_keys=CONTINUOUS_KEYS)


def _terrain_sequence_similarity(target: dict[str, Any], sample: dict[str, Any]) -> float | None:
    target_phases = _mapping(target.get("phase_distribution"))
    sample_phases = _mapping(sample.get("phase_distribution"))
    target_sequence = target.get("sequence")
    sample_sequence = sample.get("sequence")
    phase_scores = [
        _mapping_similarity(target_phases.get(phase), sample_phases.get(phase), scale=0.40, allowed_keys=PHASE_KEYS)
        for phase in ("first_25", "second_25", "third_25", "last_25")
    ]
    phase_scores = [score for score in phase_scores if score is not None]
    sequence_score = _mapping_similarity(target_sequence, sample_sequence, scale=0.45, allowed_keys=SEQUENCE_KEYS)
    scores = phase_scores + ([] if sequence_score is None else [sequence_score])
    return float(np.mean(scores)) if scores else None


def _mapping_similarity(
    target: object, sample: object, scale: float, allowed_keys: tuple[str, ...]
) -> float | None:
    target_mapping = _mapping(target)
    sample_mapping = _mapping(sample)
    if not target_mapping or not sample_mapping:
        return None
    keys = [key for key in allowed_keys if key in target_mapping and key in sample_mapping]
    if not keys:
        return None
    values = []
    for key in keys:
        try:
            left, right = abs(float(target_mapping[key])), abs(float(sample_mapping[key]))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        if left == right == 0.0:
            values.append(1.0)
        else:
            values.append(_ratio_similarity(left, right, scale, offset=0.01))
    return float(np.mean(values)) if values else None


def _mapping(value: object) -> Mapping[str, Any]:
    """Accept only mapping-shaped, summary-level route features.

    Older saved profiles can carry experimental/raw route payloads.  Never copy
    or iterate their arbitrary keys during a prediction; only the known feature
    keys are read by the group scorers above.
    """
    return value if isinstance(value, Mapping) else {}


def _ratio_similarity(left: float, right: float, scale: float, offset: float = 0.01) -> float:
    if left < 0 or right < 0:
        return 0.0
    return float(np.exp(-abs(np.log((left + offset) / (right + offset))) / scale))


def _duration_fallback(
    profile: dict[str, Any], estimated_hours: float, target: dict[str, Any], best_similarity: float,
    config: dict[str, Any], candidates: list[tuple[float, dict[str, Any], dict[str, Any]]],
    matching_mode: str = "structural",
) -> dict[str, Any]:
    terrain_curves = {}
    confidences = []
    for terrain in TERRAINS:
        match = duration_match(profile, estimated_hours, terrain)
        terrain_curves[terrain] = [float(match["factor"])] * 4
        confidences.append(float(match["confidence"]))
    return {
        "source": "duration_fallback",
        "matching_mode": matching_mode,
        "target": target,
        "phase_centers": list(DEFAULT_PHASE_CENTERS),
        "overall_curve": [float(np.mean([terrain_curves[key][0] for key in TERRAINS]))] * 4,
        "terrain_curves": terrain_curves,
        "confidence": round(float(np.mean(confidences)), 3),
        "best_similarity": round(float(best_similarity), 3),
        "matched_activities": [],
        "similarity_groups": candidates[0][2]["groups"] if candidates else {},
        "missing_feature_groups": candidates[0][2]["missing_feature_groups"] if candidates else [],
        "similarity_reasons": candidates[0][2]["reasons"] if candidates else ["没有可用的历史路线匹配"],
        "uncertainty": _route_uncertainty(target, candidates, config),
        "strategy_type": "duration_fallback",
        "fallback_reason": "no_similar_historical_route",
    }


def _route_uncertainty(
    target: dict[str, Any], candidates: list[tuple[float, dict[str, Any], dict[str, Any]]], config: dict[str, Any]
) -> dict[str, Any]:
    best_score = float(candidates[0][0]) if candidates else 0.0
    best_sample = candidates[0][1] if candidates else None
    threshold = float(config.get("low_similarity_threshold", 0.42))
    sigma_config = dict(config.get("uncertainty_sigma", {}))
    global_sigma = float(sigma_config.get("low_similarity", 0.045)) if best_score < threshold else 0.0
    terrain_sigma = {terrain: 0.0 for terrain in TERRAINS}
    reasons: list[str] = []
    if best_score < threshold:
        reasons.append("最佳历史路线相似度偏低，已扩大整体概率区间")
    if best_sample is not None:
        if not best_sample.get("route_structure"):
            per_terrain = float(sigma_config.get("uncovered_structure", 0.030))
            terrain_sigma["uphill"] += per_terrain
            terrain_sigma["downhill"] += per_terrain
            reasons.append("历史路线缺少坡度结构摘要，长坡与坡序按保守区间处理")
        uncovered = _uncovered_structure(target, best_sample, float(config.get("structure_extrapolation_ratio", 1.25)))
        per_terrain = float(sigma_config.get("uncovered_structure", 0.030))
        for terrain in uncovered:
            terrain_sigma[terrain] += per_terrain
        reasons.extend(uncovered.values())
    else:
        terrain_sigma["uphill"] = float(sigma_config.get("uncovered_structure", 0.030))
        terrain_sigma["downhill"] = float(sigma_config.get("uncovered_structure", 0.030))
        reasons.append("没有相似历史路线，陡坡与坡序结构按时长能力层回退")
    return {
        "additional_global_sigma": round(global_sigma, 4),
        "terrain_sigma": {terrain: round(value, 4) for terrain, value in terrain_sigma.items()},
        "reasons": reasons,
    }


def _empty_route_uncertainty() -> dict[str, Any]:
    return {"additional_global_sigma": 0.0, "terrain_sigma": {terrain: 0.0 for terrain in TERRAINS}, "reasons": []}


def _uncovered_structure(target: dict[str, Any], sample: dict[str, Any], ratio: float) -> dict[str, str]:
    target_continuous = _mapping(_mapping(target.get("route_structure")).get("continuous"))
    sample_continuous = _mapping(_mapping(sample.get("route_structure")).get("continuous"))
    result: dict[str, str] = {}
    checks = (
        ("uphill", "longest_uphill_distance_km", 0.25, "目标路线最长连续上坡超出历史覆盖"),
        ("uphill", "maximum_single_ascent_m", 100.0, "目标路线单次爬升超出历史覆盖"),
        ("downhill", "longest_downhill_distance_km", 0.25, "目标路线最长连续下坡超出历史覆盖"),
        ("downhill", "maximum_single_descent_m", 100.0, "目标路线单次下降超出历史覆盖"),
    )
    for terrain, key, additive_tolerance, message in checks:
        target_value = float(target_continuous.get(key, 0.0))
        sample_value = float(sample_continuous.get(key, 0.0))
        if target_value > 0 and target_value > max(sample_value * ratio, sample_value + additive_tolerance):
            result.setdefault(terrain, message)
    target_last_phase = _mapping(_mapping(_mapping(target.get("route_structure")).get("phase_distribution")).get("last_25"))
    sample_last_phase = _mapping(_mapping(_mapping(sample.get("route_structure")).get("phase_distribution")).get("last_25"))
    target_late_hard = float(target_last_phase.get("hard_uphill_gain_share", 0.0))
    sample_late_hard = float(sample_last_phase.get("hard_uphill_gain_share", 0.0))
    if target_late_hard > max(sample_late_hard * ratio, sample_late_hard + 0.08):
        result.setdefault("uphill", "目标路线后25%陡坡爬升超出历史覆盖")
    return result


def _group_reason(name: str, score: float) -> str:
    labels = {
        "scale": "路线规模差异较大", "grade_structure": "陡坡结构差异较大",
        "continuous_slope": "连续上坡或下坡结构差异较大", "terrain_sequence": "爬升/下降出现顺序差异较大",
        "activity_type": "历史样本活动类型与越野比赛不完全一致",
    }
    return f"{labels.get(name, name)}（匹配度 {score:.0%}）"


def _strategy_type(curve: np.ndarray) -> str:
    delta = float(curve[-1] - curve[0])
    if delta <= -0.04:
        return "negative_split"
    if delta >= 0.04:
        return "positive_split"
    if float(np.max(curve) - np.min(curve)) <= 0.06:
        return "even"
    return "variable"
