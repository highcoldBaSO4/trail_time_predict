from __future__ import annotations

from datetime import datetime
from math import exp, log
from typing import Any

import numpy as np

from analysis.residuals import residual_from_record
from config import load_config


TERRAINS = ("flat", "uphill", "downhill")


def calibration_summary(records: list[dict[str, Any]]) -> dict[str, object]:
    """Build a conservative, portable model from leakage-free backtest records.

    The model contains only residuals and coarse route summaries from rolling
    backtests.  It deliberately excludes FIT samples, traces and any target or
    future activity evidence.  ``calibrate_prediction_interval`` is the only
    consumer that may turn this evidence into a future-prediction adjustment.
    """
    model = build_calibration_model(records)
    return {
        "status": model["status"],
        "enabled": bool(model["enabled"]),
        "valid_backtest_count": int(model["valid_backtest_count"]),
        "effective_sample_size": model["effective_sample_size"],
        "mean_log_residual": model["mean_log_residual"],
        "p50_factor": model["p50"]["factor"],
        "interval_source": model["interval"]["source"],
        "note": model["note"],
        "model": model,
    }


def build_calibration_model(records: list[dict[str, Any]]) -> dict[str, object]:
    """Create a bounded calibration model from historical rolling backtests."""
    settings = _settings()
    evidence = [_evidence_from_record(record) for record in records]
    evidence = [item for item in evidence if item is not None]
    evidence.sort(key=lambda item: str(item["target_activity_time"]))
    count = len(evidence)
    weights = _evidence_weights(evidence, settings)
    residuals = _robust_residuals(
        np.asarray([float(item["residual_log"]) for item in evidence], dtype=float), weights, settings
    )
    mean = _weighted_mean(residuals, weights)
    median_residual = _weighted_quantile(residuals, weights, 0.50)
    effective_sample_size = _effective_sample_size(weights)
    direction_support = _direction_support(residuals, weights, mean)
    stable = (
        count >= int(settings["minimum_valid_backtests"])
        and abs(median_residual) >= float(settings["p50"]["minimum_stable_log_bias"])
        and direction_support >= float(settings["p50"]["minimum_direction_support"])
    )
    minimum = int(settings["minimum_valid_backtests"])
    full = int(settings["full_enable_backtests"])
    if count < minimum:
        status = "样本不足"
        mode = "display_only"
        enabled = False
    elif count < full:
        status = "建议观察" if stable else "建议观察（方向不稳定）"
        mode = "observe"
        enabled = stable
    elif stable:
        status = "已启用"
        mode = "enabled"
        enabled = True
    else:
        status = "证据不稳定"
        mode = "display_only"
        enabled = False

    shrinkage = count / max(count + float(settings["p50"]["shrinkage_prior_count"]), 1.0)
    mode_scale = float(settings["p50"]["observe_strength"]) if mode == "observe" else 1.0
    maximum = float(
        settings["p50"]["observe_max_log_adjustment"]
        if mode == "observe" else settings["p50"]["enabled_max_log_adjustment"]
    )
    p50_log_adjustment = _clip(mean * shrinkage * mode_scale, -maximum, maximum) if enabled else 0.0
    q10 = _weighted_quantile(residuals, weights, 0.10)
    q90 = _weighted_quantile(residuals, weights, 0.90)
    terrain = _terrain_models(evidence, weights, mean, settings)
    interval_source = "历史残差分位数" if count >= full else "全局保守先验"
    note = _model_note(status, count, enabled, interval_source)
    return {
        "schema_version": "1.0",
        "source": "rolling_backtest_no_leak",
        "status": status,
        "mode": mode,
        "enabled": enabled,
        "valid_backtest_count": count,
        "effective_sample_size": _round_or_none(effective_sample_size, 2),
        "mean_log_residual": _round_or_none(mean, 5),
        "median_log_residual": _round_or_none(median_residual, 5),
        "direction_support": _round_or_none(direction_support, 3),
        "reference_activity_time": evidence[-1]["target_activity_time"] if evidence else None,
        "quality_threshold": float(settings["minimum_quality_score"]),
        "p50": {
            "raw_log_residual": _round_or_none(mean, 5),
            "shrunken_log_adjustment": round(p50_log_adjustment, 5),
            "factor": round(exp(p50_log_adjustment), 5),
            "maximum_log_adjustment": maximum,
            "shrinkage": round(shrinkage * mode_scale, 4) if enabled else 0.0,
        },
        "interval": {
            "source": interval_source,
            "residual_p10_log": _round_or_none(q10, 5),
            "residual_p90_log": _round_or_none(q90, 5),
            "global_prior_log_half_width": float(settings["interval"]["global_prior_log_half_width"]),
        },
        "terrain": terrain,
        "evidence_records": evidence,
        "note": note,
    }


def calibrate_prediction_interval(
    probability: dict[str, object],
    model: dict[str, object] | None,
    target_route: dict[str, object] | None,
    terrain_time_share: dict[str, float] | None,
    external_context: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Apply a bounded P50/interval correction to a future probability result.

    Calibration is intentionally one-way conservative: it can shift P50 only
    after the sample gate is met, and it never narrows the existing simulated
    interval.  Inputs and return values remain JSON-friendly so the payload can
    be persisted inside an ability file or prediction report.
    """
    original = dict(probability)
    raw_p10 = float(original["p10_seconds"])
    raw_p50 = float(original["p50_seconds"])
    raw_p90 = float(original["p90_seconds"])
    if not isinstance(model, dict) or not model:
        return original, _disabled_result("没有可用的无泄漏历史回测校准模型", raw_p10, raw_p50, raw_p90)

    settings = _settings()
    evidence = [item for item in model.get("evidence_records", []) if isinstance(item, dict)]
    weights, similarities = _target_evidence_weights(evidence, target_route or {}, settings)
    residuals = _robust_residuals(
        np.asarray([_number(item.get("residual_log")) for item in evidence], dtype=float), weights, settings
    )
    count = len(evidence)
    effective_sample_size = _effective_sample_size(weights)
    runtime_mean = _weighted_mean(residuals, weights)
    runtime_median = _weighted_quantile(residuals, weights, 0.50)
    direction_support = _direction_support(residuals, weights, runtime_mean)
    route_similarity = _weighted_mean(np.asarray(similarities, dtype=float), weights)
    minimum = int(settings["minimum_valid_backtests"])
    full = int(settings["full_enable_backtests"])
    stable = (
        count >= minimum
        and abs(runtime_median) >= float(settings["p50"]["minimum_stable_log_bias"])
        and direction_support >= float(settings["p50"]["minimum_direction_support"])
    )
    if count < minimum:
        return original, _disabled_result(
            f"仅有 {count} 条高质量无泄漏回测，少于 {minimum} 条门槛；仅展示校准证据，不改预测。",
            raw_p10, raw_p50, raw_p90, count, effective_sample_size,
        )

    mode = "observe" if count < full else "enabled"
    enabled = stable and bool(model.get("enabled", False))
    shrinkage = count / max(count + float(settings["p50"]["shrinkage_prior_count"]), 1.0)
    if mode == "observe":
        shrinkage *= float(settings["p50"]["observe_strength"])
        maximum = float(settings["p50"]["observe_max_log_adjustment"])
    else:
        maximum = float(settings["p50"]["enabled_max_log_adjustment"])
    similarity_strength = float(settings["target_similarity"]["minimum_strength"]) + (
        1.0 - float(settings["target_similarity"]["minimum_strength"])
    ) * max(0.0, min(1.0, route_similarity))
    overall_log_adjustment = (
        _clip(runtime_mean * shrinkage * similarity_strength, -maximum, maximum) if enabled else 0.0
    )
    terrain_log_adjustment, terrain_details = _terrain_adjustment(
        evidence, weights, runtime_mean, terrain_time_share or {}, settings, enabled
    )
    p50_log_adjustment = _clip(
        overall_log_adjustment + terrain_log_adjustment,
        -float(settings["p50"]["enabled_max_log_adjustment"]),
        float(settings["p50"]["enabled_max_log_adjustment"]),
    ) if enabled else 0.0
    calibrated_p50 = raw_p50 * exp(p50_log_adjustment)

    raw_lower = max(0.0, log(max(raw_p50, 1.0) / max(raw_p10, 1.0)))
    raw_upper = max(0.0, log(max(raw_p90, 1.0) / max(raw_p50, 1.0)))
    q10 = _weighted_quantile(residuals, weights, 0.10)
    q90 = _weighted_quantile(residuals, weights, 0.90)
    prior = float(settings["interval"]["global_prior_log_half_width"])
    if count >= full:
        historical_lower = max(0.0, runtime_mean - q10)
        historical_upper = max(0.0, q90 - runtime_mean)
        interval_source = "历史残差分位数"
    else:
        historical_lower = prior
        historical_upper = prior
        interval_source = "全局保守先验（样本仍在观察）"
    external_width, external_reasons = _external_interval_width(external_context or {}, settings)
    lower_width = min(
        float(settings["interval"]["maximum_log_half_width"]),
        max(raw_lower, historical_lower, prior if count < full else 0.0) + external_width,
    )
    upper_width = min(
        float(settings["interval"]["maximum_log_half_width"]),
        max(raw_upper, historical_upper, prior if count < full else 0.0) + external_width,
    )
    calibrated_p10 = calibrated_p50 * exp(-lower_width)
    calibrated_p90 = calibrated_p50 * exp(upper_width)
    calibrated = _recenter_samples(
        original, calibrated_p10, calibrated_p50, calibrated_p90, raw_lower, raw_upper, lower_width, upper_width
    )
    applied = bool(enabled)
    note = (
        "校准已启用：P50 使用近期、高质量且与目标路线相近的无泄漏回测残差，并向 ×1.00 收缩。"
        if applied and mode == "enabled" else
        "校准建议观察：样本为 3–4 条，P50 仅作轻度、有上限的修正。"
        if applied else
        "回测残差方向不稳定，保留原始预测；区间仅按保守先验处理。"
    )
    result = {
        "status": "已启用" if applied and mode == "enabled" else "建议观察" if mode == "observe" else "证据不稳定",
        "enabled": applied,
        "mode": mode,
        "valid_backtest_count": count,
        "effective_sample_size": _round_or_none(effective_sample_size, 2),
        "target_route_similarity": _round_or_none(route_similarity, 3),
        "p50_before_seconds": round(raw_p50, 1),
        "p50_after_seconds": round(calibrated_p50, 1),
        "p50_factor": round(exp(p50_log_adjustment), 5),
        "overall_log_adjustment": round(overall_log_adjustment, 5),
        "terrain_log_adjustment": round(terrain_log_adjustment, 5),
        "terrain": terrain_details,
        "interval_source": interval_source,
        "interval_external_log_width": round(external_width, 5),
        "interval_external_reasons": external_reasons,
        "p10_before_seconds": round(raw_p10, 1),
        "p10_after_seconds": round(calibrated_p10, 1),
        "p90_before_seconds": round(raw_p90, 1),
        "p90_after_seconds": round(calibrated_p90, 1),
        "note": note,
    }
    return calibrated, result


def _settings() -> dict[str, Any]:
    return dict(load_config().get("prediction_calibration", {}))


def _evidence_from_record(record: dict[str, Any]) -> dict[str, object] | None:
    try:
        residual = residual_from_record(record)
        target_time = str(record["target_activity_time"])
        _parse_time(target_time)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    quality = _quality_score(dict(record.get("data_quality", {})))
    if quality < float(_settings()["minimum_quality_score"]):
        return None
    route = dict(record.get("route", {}))
    terrain_errors = dict(record.get("terrain_errors", {}))
    terrain_residuals: dict[str, float] = {}
    for terrain in TERRAINS:
        value = terrain_errors.get(terrain)
        if value is None:
            continue
        terrain_residuals[terrain] = round(log(max(0.1, 1.0 + float(value) / 100.0)), 5)
    terrain_share = {
        terrain: max(0.0, _number(dict(route.get("terrain_share", {})).get(terrain)))
        for terrain in TERRAINS
    }
    return {
        "target_activity_time": target_time,
        "residual_log": round(residual, 5),
        "quality_score": round(quality, 3),
        "route": {
            "distance_km": _round_or_none(_number(route.get("distance_km")), 3),
            "elevation_gain_m": _round_or_none(_number(route.get("elevation_gain_m")), 1),
            "elevation_loss_m": _round_or_none(_number(route.get("elevation_loss_m")), 1),
            "terrain_share": {terrain: round(value, 4) for terrain, value in terrain_share.items()},
        },
        "terrain_residual_log": terrain_residuals,
    }


def _evidence_weights(evidence: list[dict[str, object]], settings: dict[str, Any]) -> np.ndarray:
    if not evidence:
        return np.asarray([], dtype=float)
    latest = max(_parse_time(str(item["target_activity_time"])) for item in evidence)
    half_life = max(1.0, float(settings["recency_half_life_days"]))
    values = []
    for item in evidence:
        days = max(0.0, (latest - _parse_time(str(item["target_activity_time"]))).total_seconds() / 86400.0)
        recency = 0.5 ** (days / half_life)
        values.append(recency * max(0.05, _number(item.get("quality_score"))))
    return _safe_weights(values)


def _target_evidence_weights(
    evidence: list[dict[str, object]], target_route: dict[str, object], settings: dict[str, Any]
) -> tuple[np.ndarray, list[float]]:
    base = _evidence_weights(evidence, settings)
    similarities = [_route_similarity(dict(item.get("route", {})), target_route) for item in evidence]
    return _safe_weights(base * np.asarray(similarities, dtype=float)), similarities


def _terrain_models(
    evidence: list[dict[str, object]], weights: np.ndarray, overall_mean: float, settings: dict[str, Any]
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    minimum = int(settings["terrain"]["minimum_valid_backtests"])
    for terrain in TERRAINS:
        values: list[float] = []
        selected_weights: list[float] = []
        for item, weight in zip(evidence, weights):
            raw = dict(item.get("terrain_residual_log", {})).get(terrain)
            if raw is not None:
                values.append(_number(raw))
                selected_weights.append(float(weight))
        array = np.asarray(values, dtype=float)
        terrain_weights = _safe_weights(selected_weights)
        mean = _weighted_mean(array, terrain_weights)
        delta = mean - overall_mean
        stable = (
            len(values) >= minimum
            and abs(delta) >= float(settings["terrain"]["minimum_stable_log_bias"])
            and _direction_support(array - overall_mean, terrain_weights, delta)
            >= float(settings["terrain"]["minimum_direction_support"])
        )
        result[terrain] = {
            "enabled": bool(stable),
            "valid_backtest_count": len(values),
            "mean_log_residual": _round_or_none(mean, 5),
            "relative_log_adjustment": round(_clip(
                delta, -float(settings["terrain"]["maximum_log_adjustment"]),
                float(settings["terrain"]["maximum_log_adjustment"]),
            ), 5) if stable else 0.0,
        }
    return result


def _terrain_adjustment(
    evidence: list[dict[str, object]], weights: np.ndarray, overall_mean: float,
    terrain_time_share: dict[str, float], settings: dict[str, Any], enabled: bool,
) -> tuple[float, dict[str, dict[str, object]]]:
    details: dict[str, dict[str, object]] = {}
    if not enabled:
        return 0.0, details
    total = 0.0
    minimum = int(settings["terrain"]["minimum_valid_backtests"])
    for terrain in TERRAINS:
        values: list[float] = []
        selected_weights: list[float] = []
        for item, weight in zip(evidence, weights):
            raw = dict(item.get("terrain_residual_log", {})).get(terrain)
            if raw is not None:
                values.append(_number(raw))
                selected_weights.append(float(weight))
        terrain_weights = _safe_weights(selected_weights)
        mean = _weighted_mean(np.asarray(values, dtype=float), terrain_weights)
        delta = mean - overall_mean
        stable = (
            len(values) >= minimum
            and abs(delta) >= float(settings["terrain"]["minimum_stable_log_bias"])
            and _direction_support(np.asarray(values, dtype=float) - overall_mean, terrain_weights, delta)
            >= float(settings["terrain"]["minimum_direction_support"])
        )
        adjustment = _clip(
            delta, -float(settings["terrain"]["maximum_log_adjustment"]),
            float(settings["terrain"]["maximum_log_adjustment"]),
        ) if stable else 0.0
        share = max(0.0, _number(terrain_time_share.get(terrain)))
        total += share * adjustment
        details[terrain] = {
            "enabled": bool(stable),
            "valid_backtest_count": len(values),
            "time_share": round(share, 4),
            "relative_log_adjustment": round(adjustment, 5),
        }
    return total, details


def _external_interval_width(context: dict[str, object], settings: dict[str, Any]) -> tuple[float, list[str]]:
    external = dict(settings["interval"].get("external_log_width", {}))
    width = 0.0
    reasons: list[str] = []
    route = dict(context.get("route_uncertainty", {}))
    route_sigma = max(0.0, _number(route.get("additional_global_sigma")))
    terrain_sigma = dict(route.get("terrain_sigma", {}))
    terrain_share = dict(context.get("terrain_time_share", {}))
    route_sigma += sum(max(0.0, _number(terrain_share.get(terrain))) * max(0.0, _number(terrain_sigma.get(terrain))) for terrain in TERRAINS)
    if route_sigma > 0:
        width += min(float(external.get("route_similarity_max", 0.04)), route_sigma)
        reasons.append("路线相似度或坡序结构外推")
    if bool(context.get("fatigue_extrapolated")):
        width += float(external.get("fatigue_extrapolation", 0.025))
        reasons.append("疲劳曲线外推")
    dynamic = dict(context.get("dynamic_environment", {}))
    sources = dict(dynamic.get("sources", {}))
    if any(str(mode) == "unknown" for mode in sources.values()):
        width += float(external.get("unknown_condition", 0.02))
        reasons.append("比赛条件存在未知保守先验")
    return width, reasons


def _recenter_samples(
    probability: dict[str, object], p10: float, p50: float, p90: float,
    raw_lower: float, raw_upper: float, lower_width: float, upper_width: float,
) -> dict[str, object]:
    result = dict(probability)
    raw_p50 = max(float(probability["p50_seconds"]), 1.0)
    samples = np.asarray(probability.get("samples_seconds", []), dtype=float)
    if samples.size:
        relative = np.log(np.maximum(samples, 1.0) / raw_p50)
        lower_scale = lower_width / max(raw_lower, 1e-6)
        upper_scale = upper_width / max(raw_upper, 1e-6)
        adjusted = p50 * np.exp(np.where(relative < 0.0, relative * lower_scale, relative * upper_scale))
        result["samples_seconds"] = [round(float(value), 1) for value in adjusted]
        result["sigma"] = round(float(np.std(adjusted) / max(np.mean(adjusted), 1.0)), 4)
    result["p10_seconds"] = round(p10, 1)
    result["p50_seconds"] = round(p50, 1)
    result["p90_seconds"] = round(p90, 1)
    return result


def _disabled_result(
    note: str, raw_p10: float, raw_p50: float, raw_p90: float,
    count: int = 0, effective_sample_size: float | None = None,
) -> dict[str, object]:
    return {
        "status": "样本不足",
        "enabled": False,
        "mode": "display_only",
        "valid_backtest_count": count,
        "effective_sample_size": _round_or_none(effective_sample_size, 2),
        "p50_before_seconds": round(raw_p50, 1),
        "p50_after_seconds": round(raw_p50, 1),
        "p50_factor": 1.0,
        "overall_log_adjustment": 0.0,
        "terrain_log_adjustment": 0.0,
        "terrain": {},
        "interval_source": "未启用",
        "interval_external_log_width": 0.0,
        "interval_external_reasons": [],
        "p10_before_seconds": round(raw_p10, 1),
        "p10_after_seconds": round(raw_p10, 1),
        "p90_before_seconds": round(raw_p90, 1),
        "p90_after_seconds": round(raw_p90, 1),
        "note": note,
    }


def _model_note(status: str, count: int, enabled: bool, interval_source: str) -> str:
    if count < 3:
        return "高质量无泄漏回测少于 3 条：仅展示残差，不会改写未来预测。"
    if not enabled:
        return "回测残差方向不稳定：保留原始预测，避免由单次异常活动驱动校准。"
    if status.startswith("建议观察"):
        return "已有 3–4 条同向高质量无泄漏回测：P50 仅作轻度、带上限的修正。"
    return f"至少 5 条同向高质量无泄漏回测：P50 可在上限内校准；区间来源为{interval_source}。"


def _quality_score(data_quality: dict[str, object]) -> float:
    values = [_number(data_quality.get(key), default=np.nan) for key in ("target", "route")]
    usable = [value for value in values if np.isfinite(value)]
    return float(sum(usable) / len(usable)) if usable else 0.0


def _route_similarity(source: dict[str, object], target: dict[str, object]) -> float:
    if not target:
        return 1.0
    distance = _ratio_similarity(_number(source.get("distance_km")), _number(target.get("distance_km")))
    gain = _ratio_similarity(_number(source.get("elevation_gain_m")), _number(target.get("elevation_gain"), _number(target.get("elevation_gain_m"))))
    source_share = dict(source.get("terrain_share", {}))
    target_share = dict(target.get("terrain_share", {}))
    if source_share and target_share:
        terrain = 1.0 - min(1.0, sum(abs(_number(source_share.get(key)) - _number(target_share.get(key))) for key in TERRAINS) / 2.0)
    else:
        terrain = 0.7
    return max(0.05, min(1.0, distance * 0.45 + gain * 0.35 + terrain * 0.20))


def _ratio_similarity(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        return 0.7
    return float(np.exp(-abs(log(left / right))))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    if weights.size != values.size or float(weights.sum()) <= 0:
        return float(values.mean())
    return float(np.average(values, weights=weights))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if values.size == 0:
        return 0.0
    if weights.size != values.size or float(weights.sum()) <= 0:
        return float(np.quantile(values, quantile))
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    target = max(0.0, min(1.0, quantile)) * float(ordered_weights.sum())
    index = min(int(np.searchsorted(np.cumsum(ordered_weights), target, side="left")), len(ordered_values) - 1)
    return float(ordered_values[index])


def _robust_residuals(values: np.ndarray, weights: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    """Winsorize a single pathological backtest before it can steer P50/P90."""
    if values.size == 0:
        return values
    centre = _weighted_quantile(values, weights, 0.50)
    limit = max(0.01, float(settings["p50"]["outlier_log_clip"]))
    return np.clip(values, centre - limit, centre + limit)


def _direction_support(values: np.ndarray, weights: np.ndarray, mean: float) -> float:
    if values.size == 0 or abs(mean) < 1e-9:
        return 0.0
    direction = 1.0 if mean > 0 else -1.0
    if weights.size != values.size or float(weights.sum()) <= 0:
        return float(np.mean(values * direction > 0.0))
    return float(weights[values * direction > 0.0].sum() / weights.sum())


def _effective_sample_size(weights: np.ndarray) -> float | None:
    if weights.size == 0 or float(weights.sum()) <= 0:
        return None
    return float(weights.sum() ** 2 / max(float(np.square(weights).sum()), 1e-12))


def _safe_weights(values: Any) -> np.ndarray:
    weights = np.asarray(values, dtype=float)
    if weights.size == 0:
        return weights
    weights[~np.isfinite(weights)] = 0.0
    return np.maximum(weights, 0.0)


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _number(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _round_or_none(value: float | None, digits: int) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value), digits)
