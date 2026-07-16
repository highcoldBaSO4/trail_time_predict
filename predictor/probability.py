from __future__ import annotations

from typing import Any

import numpy as np

from config import load_config


TERRAINS = ("flat", "uphill", "downhill")
CONDITION_SOURCES = (
    "heart_rate_pacing", "form", "technical", "mud", "night", "altitude", "carried_weight",
    "temperature_fatigue", "heart_rate_fatigue", "weather",
)


def simulate_segmented_finish_times(
    profile: dict[str, object],
    segment_rows: list[dict[str, object]],
    aid_seconds: float,
    gpx_quality_score: float = 1.0,
    simulations: int | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    """Simulate the race segment by segment with terrain-specific ability and fatigue uncertainty."""
    if not segment_rows:
        raise ValueError("概率模拟至少需要一个比赛分段")
    config = load_config()["monte_carlo"]
    count = _simulation_count(simulations, config)
    rng = np.random.default_rng(int(config["seed"] if seed is None else seed))
    correlation = float(config.get("terrain_correlation", 0.70))
    fatigue_scale = float(config.get("fatigue_sigma_scale", 0.60))
    duration_scale = float(config.get("duration_sigma_scale", 0.50))

    ability_samples, ability_confidence = _ability_samples(profile, count, rng, config, correlation)
    fatigue_samples, fatigue_confidence = _fatigue_samples(
        profile, count, rng, config, correlation, fatigue_scale
    )
    duration_samples, duration_confidence = _duration_samples(
        segment_rows, count, rng, config, duration_scale
    )
    condition_latents = _condition_latents(count, rng, config)
    condition_summary = _condition_uncertainty_summary(segment_rows, config)
    gpx_sigma = _gpx_sigma(gpx_quality_score, config)
    gpx_shared = rng.normal(size=count)
    gpx_correlation = float(config.get("gpx_segment_correlation", 0.55))

    elapsed = np.zeros(count, dtype=float)
    terrain_seconds = {terrain: 0.0 for terrain in TERRAINS}
    gpx_affected_seconds = 0.0
    for row in segment_rows:
        terrain = str(row.get("type", "flat"))
        deterministic_duration = float(row.get("duration_factor", 1.0))
        deterministic_condition = float(row.get("condition_factor", 1.0))
        raw_seconds, sampled_grade = _sample_gpx_geometry(
            profile, row, count, rng, gpx_shared, gpx_sigma, gpx_correlation
        )

        ability_time_factor = _ability_factor_for_grade(ability_samples[terrain], sampled_grade)
        sustainable = (
            raw_seconds
            * ability_time_factor
            * deterministic_duration
            * duration_samples[terrain]
        )
        fatigue = _interpolate_fatigue_samples(
            elapsed / 3600.0,
            fatigue_samples[terrain]["hours"],
            fatigue_samples[terrain]["values"],
        )
        condition_uncertainty = _condition_noise_for_row(
            row, terrain, condition_latents, config
        )
        seconds = (
            sustainable
            / np.maximum(fatigue, 0.1)
            * deterministic_condition
            * condition_uncertainty
        )
        elapsed += seconds
        terrain_seconds[terrain] += float(row["predicted_time_seconds"])
        if terrain != "flat" and bool(row.get("elevation_available", True)):
            gpx_affected_seconds += float(row["predicted_time_seconds"])

    samples = np.maximum(elapsed + max(0.0, float(aid_seconds)), 1.0)
    p10, p50, p90 = np.percentile(samples, [10, 50, 90])
    total_deterministic = sum(terrain_seconds.values())
    terrain_share = {
        terrain: round(terrain_seconds[terrain] / total_deterministic, 4) if total_deterministic > 0 else 0.0
        for terrain in TERRAINS
    }
    return {
        "p10_seconds": round(float(p10), 1),
        "p50_seconds": round(float(p50), 1),
        "p90_seconds": round(float(p90), 1),
        "simulations": count,
        "sigma": round(float(np.std(samples) / np.mean(samples)), 4),
        "samples_seconds": [round(float(value), 1) for value in samples],
        "method": "segmented_source_condition_physical_gpx",
        "uncertainty": {
            "terrain_time_share": terrain_share,
            "ability_confidence": ability_confidence,
            "fatigue_confidence": fatigue_confidence,
            "duration_confidence": duration_confidence,
            "condition_sources": condition_summary,
            "condition_sigma": round(_weighted_condition_sigma(condition_summary), 4),
            "gpx_sigma": round(gpx_sigma, 4),
            "gpx": {
                "mode": "segment_elevation_grade",
                "quality_score": round(max(0.0, min(1.0, float(gpx_quality_score))), 3),
                "vertical_sigma": round(gpx_sigma, 4),
                "affected_time_share": round(gpx_affected_seconds / total_deterministic, 4)
                if total_deterministic > 0 else 0.0,
            },
        },
    }


def simulate_finish_times(
    adjusted_moving_seconds: float,
    aid_seconds: float,
    confidence: float,
    gpx_quality_score: float = 1.0,
    simulations: int | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    """Compatibility wrapper for callers that only have an aggregate moving time."""
    config = load_config()["monte_carlo"]
    count = _simulation_count(simulations, config)
    confidence = max(0.0, min(1.0, float(confidence)))
    ability_sigma = _sigma_from_confidence(confidence, config)
    gpx_sigma = _gpx_sigma(gpx_quality_score, config)
    rng = np.random.default_rng(int(config["seed"] if seed is None else seed))
    ability = rng.lognormal(0.0, ability_sigma, count)
    condition = rng.lognormal(0.0, float(config["condition_sigma"]), count)
    gpx = rng.lognormal(0.0, gpx_sigma, count)
    samples = np.maximum(adjusted_moving_seconds * ability * condition * gpx + aid_seconds, 1.0)
    p10, p50, p90 = np.percentile(samples, [10, 50, 90])
    return {
        "p10_seconds": round(float(p10), 1),
        "p50_seconds": round(float(p50), 1),
        "p90_seconds": round(float(p90), 1),
        "simulations": count,
        "sigma": round(float(np.std(samples) / np.mean(samples)), 4),
        "samples_seconds": [round(float(value), 1) for value in samples],
        "method": "aggregate_compatibility",
    }


def _ability_samples(
    profile: dict[str, object],
    count: int,
    rng: np.random.Generator,
    config: dict[str, Any],
    correlation: float,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    flat_confidence = float(profile.get("flat", {}).get("confidence", 0.2))
    flat_sigma = _sigma_from_confidence(flat_confidence, config)
    flat_values = rng.lognormal(0.0, flat_sigma, count)
    result: dict[str, dict[str, object]] = {
        "flat": {"grades": np.asarray([0.0]), "values": flat_values.reshape(1, count)}
    }
    confidence_summary: dict[str, object] = {"flat": round(flat_confidence, 3)}
    for terrain, value_key in (("uphill", "value"), ("downhill", "speed_mps")):
        curve = sorted(
            list(profile.get(terrain, {}).get("curve", [])), key=lambda point: float(point["grade"])
        )
        grades = np.asarray([float(point["grade"]) for point in curve], dtype=float)
        confidences = np.asarray([float(point.get("confidence", 0.2) or 0.2) for point in curve])
        sigmas = np.asarray([_sigma_from_confidence(value, config) for value in confidences])
        values = _correlated_lognormal(sigmas, count, rng, correlation)
        result[terrain] = {"grades": grades, "values": values, "value_key": value_key}
        confidence_summary[terrain] = [round(float(value), 3) for value in confidences]
    return result, confidence_summary


def _fatigue_samples(
    profile: dict[str, object],
    count: int,
    rng: np.random.Generator,
    config: dict[str, Any],
    correlation: float,
    sigma_scale: float,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list[float | None]]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    confidence_summary: dict[str, list[float | None]] = {}
    for terrain in TERRAINS:
        curve = list(profile.get("fatigue", {}).get(terrain, []))
        hours = np.asarray([float(point["hour"]) for point in curve], dtype=float)
        base = np.asarray([float(point["factor"]) for point in curve], dtype=float)
        confidences = [None if point.get("confidence") is None else float(point["confidence"]) for point in curve]
        sigmas = np.asarray([
            0.0 if confidence is None else _sigma_from_confidence(confidence, config) * sigma_scale
            for confidence in confidences
        ])
        multipliers = _correlated_lognormal(sigmas, count, rng, correlation)
        values = np.clip(base[:, None] * multipliers, 0.40, 1.05)
        values[hours == 0.0, :] = 1.0
        result[terrain] = {"hours": hours, "values": values}
        confidence_summary[terrain] = [None if value is None else round(value, 3) for value in confidences]
    return result, confidence_summary


def _duration_samples(
    rows: list[dict[str, object]],
    count: int,
    rng: np.random.Generator,
    config: dict[str, Any],
    sigma_scale: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    result: dict[str, np.ndarray] = {}
    confidence_summary: dict[str, float] = {}
    for terrain in TERRAINS:
        terrain_rows = [row for row in rows if str(row.get("type", "flat")) == terrain]
        if terrain_rows:
            weights = np.asarray([float(row["predicted_time_seconds"]) for row in terrain_rows])
            confidences = np.asarray([float(row.get("duration_confidence", 0.2)) for row in terrain_rows])
            confidence = float(np.average(confidences, weights=weights)) if weights.sum() > 0 else float(confidences.mean())
        else:
            confidence = 0.2
        sigma = _sigma_from_confidence(confidence, config) * sigma_scale
        result[terrain] = rng.lognormal(0.0, sigma, count)
        confidence_summary[terrain] = round(confidence, 3)
    return result, confidence_summary


def _ability_factor_for_grade(samples: dict[str, object], grade: float | np.ndarray) -> np.ndarray:
    grades = np.asarray(samples["grades"], dtype=float)
    values = np.asarray(samples["values"], dtype=float)
    if len(grades) == 1:
        return values[0]
    grade_values = np.broadcast_to(np.asarray(grade, dtype=float), values.shape[1:])
    upper = np.searchsorted(grades, grade_values, side="right")
    upper = np.clip(upper, 1, len(grades) - 1)
    lower = upper - 1
    weight = np.clip(
        (grade_values - grades[lower]) / np.maximum(grades[upper] - grades[lower], 1e-9),
        0.0,
        1.0,
    )
    columns = np.arange(values.shape[1])
    interpolated = values[lower, columns] * (1.0 - weight) + values[upper, columns] * weight
    interpolated[grade_values <= grades[0]] = values[0, grade_values <= grades[0]]
    interpolated[grade_values >= grades[-1]] = values[-1, grade_values >= grades[-1]]
    return interpolated


def _condition_latents(
    count: int, rng: np.random.Generator, config: dict[str, Any]
) -> dict[str, dict[str, np.ndarray]]:
    correlation = max(0.0, min(1.0, float(config.get("condition_terrain_correlation", 0.85))))
    result: dict[str, dict[str, np.ndarray]] = {}
    for source in CONDITION_SOURCES:
        shared = rng.normal(size=count)
        independent = rng.normal(size=(len(TERRAINS), count))
        result[source] = {
            terrain: np.sqrt(correlation) * shared + np.sqrt(1.0 - correlation) * independent[index]
            for index, terrain in enumerate(TERRAINS)
        }
    return result


def _condition_noise_for_row(
    row: dict[str, object],
    terrain: str,
    latents: dict[str, dict[str, np.ndarray]],
    config: dict[str, Any],
) -> np.ndarray:
    factors = dict(row.get("condition_factors", {}))
    confidences = dict(row.get("condition_confidence", {}))
    sample = np.ones_like(next(iter(latents.values()))[terrain])
    for source in CONDITION_SOURCES:
        sigma = _condition_sigma_for_factor(
            source, float(factors.get(source, 1.0)), config, float(confidences.get(source, 0.7))
        )
        if sigma > 0:
            sample *= np.exp(sigma * latents[source][terrain])
    return sample


def _condition_sigma_for_factor(
    source: str, factor: float, config: dict[str, Any], confidence: float = 0.7
) -> float:
    source_sigmas = dict(config.get("condition_source_sigma", {}))
    maximum = float(source_sigmas.get(source, config.get("condition_sigma", 0.02)))
    reference = max(float(config.get("condition_activation_reference", 0.10)), 1e-6)
    activation = min(abs(float(factor) - 1.0) / reference, 1.0)
    confidence_scale = 1.25 - 0.5 * max(0.0, min(1.0, confidence))
    return maximum * activation * confidence_scale


def _condition_uncertainty_summary(
    rows: list[dict[str, object]], config: dict[str, Any]
) -> dict[str, dict[str, float]]:
    total = sum(float(row.get("predicted_time_seconds", 0.0)) for row in rows)
    summary: dict[str, dict[str, float]] = {}
    for source in CONDITION_SOURCES:
        weighted_sigma = 0.0
        active_seconds = 0.0
        for row in rows:
            seconds = float(row.get("predicted_time_seconds", 0.0))
            factor = float(dict(row.get("condition_factors", {})).get(source, 1.0))
            confidence = float(dict(row.get("condition_confidence", {})).get(source, 0.7))
            sigma = _condition_sigma_for_factor(source, factor, config, confidence)
            weighted_sigma += seconds * sigma
            if abs(factor - 1.0) > 1e-6:
                active_seconds += seconds
        summary[source] = {
            "effective_sigma": round(weighted_sigma / total, 4) if total > 0 else 0.0,
            "active_time_share": round(active_seconds / total, 4) if total > 0 else 0.0,
        }
    return summary


def _weighted_condition_sigma(summary: dict[str, dict[str, float]]) -> float:
    return float(np.sqrt(sum(float(item["effective_sigma"]) ** 2 for item in summary.values())))


def _sample_gpx_geometry(
    profile: dict[str, object],
    row: dict[str, object],
    count: int,
    rng: np.random.Generator,
    shared: np.ndarray,
    sigma: float,
    correlation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Perturb vertical metres, then recalculate grade and terrain base time."""
    micro_segments = list(row.get("micro_segments", []))
    if micro_segments:
        sampled_seconds = np.zeros(count, dtype=float)
        weighted_grade = np.zeros(count, dtype=float)
        total_distance = 0.0
        for micro in micro_segments:
            unit_seconds, unit_grade = _sample_gpx_geometry_unit(
                profile, micro, count, rng, shared, sigma, correlation
            )
            distance = max(float(micro.get("distance", 0.0)), 0.0)
            sampled_seconds += unit_seconds
            weighted_grade += unit_grade * distance
            total_distance += distance
        return sampled_seconds, weighted_grade / max(total_distance, 0.1)
    return _sample_gpx_geometry_unit(profile, row, count, rng, shared, sigma, correlation)


def _sample_gpx_geometry_unit(
    profile: dict[str, object],
    row: dict[str, object],
    count: int,
    rng: np.random.Generator,
    shared: np.ndarray,
    sigma: float,
    correlation: float,
) -> tuple[np.ndarray, np.ndarray]:
    original_seconds = _geometry_seconds(profile, row)
    original_grade = float(row.get("grade", 0.0))
    terrain = str(row.get("type", "flat"))
    if terrain == "flat" or not bool(row.get("elevation_available", True)):
        return np.full(count, original_seconds), np.full(count, original_grade)

    correlation = max(0.0, min(1.0, correlation))
    z = np.sqrt(correlation) * shared + np.sqrt(1.0 - correlation) * rng.normal(size=count)
    distance = max(float(row.get("distance", 0.0)), 0.1)
    if terrain == "uphill":
        vertical = max(float(row.get("gain", 0.0)), distance * max(original_grade, 0.0) / 100.0, 0.1)
        sampled_vertical = vertical * np.exp(sigma * z)
        sampled_grade = sampled_vertical / distance * 100.0
        curve = list(profile.get("uphill", {}).get("curve", []))
        if not curve:
            return np.full(count, original_seconds), sampled_grade
        grades = np.asarray([float(point["grade"]) for point in curve])
        values = np.asarray([float(point["value"]) for point in curve])
        vam = np.interp(sampled_grade, grades, values)
        climbing = sampled_vertical / np.maximum(vam, 1.0) * 3600.0
        flat_floor = distance / 1000.0 * float(profile["flat"]["aerobic_pace"])
        return np.maximum(climbing, flat_floor), sampled_grade

    vertical = max(float(row.get("loss", 0.0)), distance * max(-original_grade, 0.0) / 100.0, 0.1)
    sampled_vertical = vertical * np.exp(sigma * z)
    sampled_grade = -sampled_vertical / distance * 100.0
    curve = list(profile.get("downhill", {}).get("curve", []))
    if not curve:
        return np.full(count, original_seconds), sampled_grade
    grades = np.asarray([float(point["grade"]) for point in curve])
    speeds = np.asarray([float(point["speed_mps"]) for point in curve])
    order = np.argsort(grades)
    speed = np.interp(sampled_grade, grades[order], speeds[order])
    return distance / np.maximum(speed, 0.1), sampled_grade


def _geometry_seconds(profile: dict[str, object], row: dict[str, object]) -> float:
    if row.get("base_time_seconds") is not None:
        return float(row["base_time_seconds"])
    distance = max(float(row.get("distance", 0.0)), 0.1)
    grade = float(row.get("grade", 0.0))
    terrain = str(row.get("type", "flat"))
    flat_seconds = distance / 1000.0 * float(profile["flat"]["aerobic_pace"])
    if terrain == "flat":
        return flat_seconds
    if terrain == "uphill":
        curve = list(profile.get("uphill", {}).get("curve", []))
        if not curve:
            return flat_seconds
        grades = np.asarray([float(point["grade"]) for point in curve])
        values = np.asarray([float(point["value"]) for point in curve])
        vam = float(np.interp(grade, grades, values))
        vertical = max(float(row.get("gain", 0.0)), distance * max(grade, 0.0) / 100.0)
        return max(vertical / max(vam, 1.0) * 3600.0, flat_seconds)
    curve = list(profile.get("downhill", {}).get("curve", []))
    if not curve:
        return flat_seconds
    grades = np.asarray([float(point["grade"]) for point in curve])
    speeds = np.asarray([float(point["speed_mps"]) for point in curve])
    order = np.argsort(grades)
    speed = float(np.interp(grade, grades[order], speeds[order]))
    return distance / max(speed, 0.1)


def _interpolate_fatigue_samples(
    elapsed_hours: np.ndarray,
    hours: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    upper = np.searchsorted(hours, elapsed_hours, side="right")
    upper = np.clip(upper, 1, len(hours) - 1)
    lower = upper - 1
    beyond = elapsed_hours >= hours[-1]
    before = elapsed_hours <= hours[0]
    span = np.maximum(hours[upper] - hours[lower], 1e-9)
    weight = np.clip((elapsed_hours - hours[lower]) / span, 0.0, 1.0)
    columns = np.arange(len(elapsed_hours))
    interpolated = values[lower, columns] * (1.0 - weight) + values[upper, columns] * weight
    interpolated[beyond] = values[-1, columns[beyond]]
    interpolated[before] = values[0, columns[before]]
    return interpolated


def _correlated_lognormal(
    sigmas: np.ndarray,
    count: int,
    rng: np.random.Generator,
    correlation: float,
) -> np.ndarray:
    shared = rng.normal(size=count)
    independent = rng.normal(size=(len(sigmas), count))
    correlation = max(0.0, min(1.0, correlation))
    z = np.sqrt(correlation) * shared[None, :] + np.sqrt(1.0 - correlation) * independent
    return np.exp(sigmas[:, None] * z)


def _simulation_count(simulations: int | None, config: dict[str, Any]) -> int:
    count = int(config["simulations"] if simulations is None else simulations)
    return max(int(config["min_simulations"]), min(int(config["max_simulations"]), count))


def _sigma_from_confidence(confidence: float, config: dict[str, Any]) -> float:
    confidence = max(0.0, min(1.0, float(confidence)))
    return (
        float(config["sigma_at_confidence_zero"]) * (1.0 - confidence)
        + float(config["sigma_at_confidence_one"]) * confidence
    )


def _gpx_sigma(gpx_quality_score: float, config: dict[str, Any]) -> float:
    quality = max(0.0, min(1.0, float(gpx_quality_score)))
    return float(config["gpx_sigma_high_quality"]) + (1.0 - quality) * (
        float(config["gpx_sigma_low_quality"]) - float(config["gpx_sigma_high_quality"])
    )
