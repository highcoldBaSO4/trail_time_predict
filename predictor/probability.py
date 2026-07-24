from __future__ import annotations

from typing import Callable
from typing import Any

import numpy as np

from analysis.environment import solar_elevation_degrees_vector
from analysis.temperature import (
    heart_rate_heat_fatigue_time_factor_vector,
    humidity_time_factor_vector,
    race_temperature_at_elapsed_vector,
    temperature_fatigue_time_factor_vector,
    temperature_time_factor_vector,
)
from config import load_config
from models import RaceCondition


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
    route_strategy_uncertainty: dict[str, object] | None = None,
    condition: RaceCondition | None = None,
    estimated_hours: float | None = None,
    progress: Callable[[str], None] | None = None,
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
    race_condition = condition.normalized() if condition is not None else None
    dynamic_source_modes = _dynamic_environment_source_modes(segment_rows, race_condition)
    condition_summary = _condition_uncertainty_summary(segment_rows, config, dynamic_source_modes)
    gpx_sigma = _gpx_sigma(gpx_quality_score, config)
    gpx_shared = rng.normal(size=count)
    gpx_correlation = float(config.get("gpx_segment_correlation", 0.55))
    route_uncertainty = dict(route_strategy_uncertainty or {})
    route_global_sigma = max(0.0, float(route_uncertainty.get("additional_global_sigma", 0.0)))
    route_global_noise = rng.lognormal(0.0, route_global_sigma, count)
    route_terrain_sigma = {
        terrain: max(0.0, float(dict(route_uncertainty.get("terrain_sigma", {})).get(terrain, 0.0)))
        for terrain in TERRAINS
    }
    route_terrain_noise = {
        terrain: rng.lognormal(0.0, route_terrain_sigma[terrain], count)
        for terrain in TERRAINS
    }

    elapsed = np.zeros(count, dtype=float)
    terrain_seconds = {terrain: 0.0 for terrain in TERRAINS}
    gpx_affected_seconds = 0.0
    dynamic_night_seconds = np.zeros(count, dtype=float)
    dynamic_temperature_seconds = np.zeros(count, dtype=float)
    dynamic_temperature_weighted = np.zeros(count, dtype=float)
    progress_interval = max(1, len(segment_rows) // 5)
    for index, row in enumerate(segment_rows, start=1):
        terrain = str(row.get("type", "flat"))
        deterministic_duration = float(row.get("duration_factor", 1.0))
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
        before_condition = sustainable / np.maximum(fatigue, 0.1)
        dynamic_condition = _dynamic_condition_factors(
            profile, row, race_condition, elapsed, before_condition,
            float(estimated_hours if estimated_hours is not None else max(float(row.get("cumulative_time_seconds", 0.0)) / 3600.0, 0.01)),
        )
        condition_uncertainty = _condition_noise_for_row(
            row, terrain, condition_latents, config,
            dynamic_factors=dynamic_condition["sources"], source_modes=dynamic_source_modes,
        )
        seconds = (
            before_condition
            * dynamic_condition["total"]
            * condition_uncertainty
            * route_global_noise
            * route_terrain_noise[terrain]
        )
        elapsed += seconds
        dynamic_night_seconds += seconds * dynamic_condition["night_ratio"]
        if dynamic_condition["temperature_c"] is not None:
            dynamic_temperature_seconds += seconds
            dynamic_temperature_weighted += seconds * dynamic_condition["temperature_c"]
        terrain_seconds[terrain] += float(row["predicted_time_seconds"])
        if terrain != "flat" and bool(row.get("elevation_available", True)):
            gpx_affected_seconds += float(row["predicted_time_seconds"])
        if progress is not None and (index == len(segment_rows) or index % progress_interval == 0):
            progress(f"Monte Carlo 概率模拟：已处理 {index}/{len(segment_rows)} 个自然地形段……")

    samples = np.maximum(elapsed + max(0.0, float(aid_seconds)), 1.0)
    p10, p50, p90 = np.percentile(samples, [10, 50, 90])
    total_deterministic = sum(terrain_seconds.values())
    moving_samples = np.maximum(elapsed, 1.0)
    dynamic_temperature = np.divide(
        dynamic_temperature_weighted, dynamic_temperature_seconds,
        out=np.full(count, np.nan), where=dynamic_temperature_seconds > 0,
    )
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
        "method": "segmented_dynamic_environment_source_condition_physical_gpx",
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
            "route_similarity": {
                "additional_global_sigma": round(route_global_sigma, 4),
                "terrain_sigma": {terrain: round(value, 4) for terrain, value in route_terrain_sigma.items()},
                "reasons": list(route_uncertainty.get("reasons", [])),
            },
            "dynamic_environment": {
                "mode": "per_simulation_elapsed_time",
                "enabled": race_condition is not None,
                "sources": dynamic_source_modes,
                "mean_night_ratio": round(float(np.mean(dynamic_night_seconds / moving_samples)), 4),
                "night_ratio_p10": round(float(np.percentile(dynamic_night_seconds / moving_samples, 10)), 4),
                "night_ratio_p90": round(float(np.percentile(dynamic_night_seconds / moving_samples, 90)), 4),
                "mean_temperature_c": None if np.isnan(dynamic_temperature).all() else round(float(np.nanmean(dynamic_temperature)), 2),
                "temperature_c_p10": None if np.isnan(dynamic_temperature).all() else round(float(np.nanpercentile(dynamic_temperature, 10)), 2),
                "temperature_c_p90": None if np.isnan(dynamic_temperature).all() else round(float(np.nanpercentile(dynamic_temperature, 90)), 2),
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
    dynamic = dict(load_config().get("dynamic_environment", {}))
    grouped_correlation = dict(dynamic.get("correlation", {}))
    group_sources = {
        "weather_heat": {"weather", "temperature_fatigue", "heart_rate_fatigue"},
        "technical_mud": {"technical", "mud"},
    }
    group_shared = {name: rng.normal(size=count) for name in group_sources}
    result: dict[str, dict[str, np.ndarray]] = {}
    for source in CONDITION_SOURCES:
        group_name = next((name for name, sources in group_sources.items() if source in sources), None)
        if group_name is None:
            shared = rng.normal(size=count)
        else:
            source_correlation = max(0.0, min(1.0, float(grouped_correlation.get(group_name, 0.0))))
            shared = (
                np.sqrt(source_correlation) * group_shared[group_name]
                + np.sqrt(1.0 - source_correlation) * rng.normal(size=count)
            )
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
    dynamic_factors: dict[str, float | np.ndarray] | None = None,
    source_modes: dict[str, str] | None = None,
) -> np.ndarray:
    factors = dict(row.get("condition_factors", {}))
    confidences = dict(row.get("condition_confidence", {}))
    sample = np.ones_like(next(iter(latents.values()))[terrain])
    for source in CONDITION_SOURCES:
        factor = factors.get(source, 1.0) if dynamic_factors is None else dynamic_factors.get(source, factors.get(source, 1.0))
        source_mode = "route_confirmed" if source_modes is None else source_modes.get(source, "route_confirmed")
        sigma = _condition_sigma_for_factor_vector(
            source, factor, config, float(confidences.get(source, 0.7)),
            _source_uncertainty_scale(source, source_modes, config), source_mode,
        )
        if np.any(sigma > 0):
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


def _condition_sigma_for_factor_vector(
    source: str,
    factor: float | np.ndarray,
    config: dict[str, Any],
    confidence: float = 0.7,
    source_scale: float = 1.0,
    source_mode: str = "route_confirmed",
) -> np.ndarray:
    source_sigmas = dict(config.get("condition_source_sigma", {}))
    maximum = float(source_sigmas.get(source, config.get("condition_sigma", 0.02)))
    reference = max(float(config.get("condition_activation_reference", 0.10)), 1e-6)
    activation = np.minimum(np.abs(np.asarray(factor, dtype=float) - 1.0) / reference, 1.0)
    if source_mode == "unknown":
        unknown_prior = float(load_config().get("dynamic_environment", {}).get("uncertainty", {}).get("unknown_prior_activation", 0.0))
        activation = np.maximum(activation, max(0.0, min(1.0, unknown_prior)))
    confidence_scale = 1.25 - 0.5 * max(0.0, min(1.0, confidence))
    return maximum * activation * confidence_scale * max(0.0, source_scale)


def _source_uncertainty_scale(source: str, source_modes: dict[str, str] | None, config: dict[str, Any]) -> float:
    mode = "route_confirmed" if source_modes is None else source_modes.get(source, "route_confirmed")
    settings = dict(load_config().get("dynamic_environment", {}).get("uncertainty", {}))
    return float(settings.get(f"{mode}_sigma_scale", 1.0))


def _dynamic_environment_source_modes(
    rows: list[dict[str, object]], condition: RaceCondition | None
) -> dict[str, str]:
    has_route_clock = bool(
        condition is not None and condition.race_start_time_utc is not None
        and any("latitude" in row and "longitude" in row for row in rows)
    )
    has_temperature = bool(condition is not None and condition.temperature_c is not None)
    modes = {source: "route_confirmed" for source in CONDITION_SOURCES}
    modes.update({
        "heart_rate_pacing": "user_input", "form": "user_input", "technical": "user_input",
        "mud": "user_input", "carried_weight": "user_input",
    })
    modes["weather"] = "user_input" if has_temperature else "unknown"
    modes["temperature_fatigue"] = modes["weather"]
    modes["heart_rate_fatigue"] = modes["weather"]
    modes["night"] = "route_confirmed" if has_route_clock else "unknown"
    modes["altitude"] = "route_confirmed" if any(bool(row.get("elevation_available", False)) for row in rows) else "unknown"
    return modes


def _dynamic_condition_factors(
    profile: dict[str, object],
    row: dict[str, object],
    condition: RaceCondition | None,
    elapsed_seconds: np.ndarray,
    before_condition_seconds: np.ndarray,
    estimated_hours: float,
) -> dict[str, object]:
    """Re-evaluate time-dependent conditions for one simulated segment."""
    stored = {name: float(value) for name, value in dict(row.get("condition_factors", {})).items()}
    count = len(elapsed_seconds)
    sources: dict[str, float | np.ndarray] = {
        name: np.full(count, stored.get(name, 1.0), dtype=float)
        for name in CONDITION_SOURCES
    }
    if condition is None:
        total = np.ones(count, dtype=float)
        for value in sources.values():
            total *= value
        return {"total": total, "sources": sources, "night_ratio": np.zeros(count), "temperature_c": None}

    midpoint_hours = (elapsed_seconds + before_condition_seconds / 2.0) / 3600.0
    temperatures = race_temperature_at_elapsed_vector(condition, midpoint_hours, estimated_hours)
    if temperatures is not None:
        sources["weather"] = temperature_time_factor_vector(profile, temperatures) * humidity_time_factor_vector(
            temperatures, condition.humidity_percent
        )
        sources["temperature_fatigue"] = temperature_fatigue_time_factor_vector(profile, temperatures, midpoint_hours)
        sources["heart_rate_fatigue"] = heart_rate_heat_fatigue_time_factor_vector(profile, temperatures, midpoint_hours)

    night_ratio = np.full(count, float(dict(row.get("environment", {})).get("night_ratio", condition.night_running_ratio)), dtype=float)
    if condition.race_start_time_utc is not None and "latitude" in row and "longitude" in row:
        solar = solar_elevation_degrees_vector(
            condition.race_start_time_utc, elapsed_seconds + before_condition_seconds / 2.0,
            float(row["latitude"]), float(row["longitude"]),
        )
        night_ratio = (solar <= float(load_config()["environment"]["night_solar_elevation_degrees"])).astype(float)
        terrain = str(row.get("type", "flat"))
        historical_night = float(
            profile.get("environment", {}).get("night", {}).get("terrain", {}).get(terrain, {}).get(
                "ratio", profile.get("environment", {}).get("night", {}).get("ratio", 0.0)
            )
        )
        night_penalty = float(load_config()["conditions"]["night_max"][terrain])
        sources["night"] = (1.0 + night_ratio * night_penalty) / (1.0 + historical_night * night_penalty)

    total = np.ones(count, dtype=float)
    for value in sources.values():
        total *= value
    return {"total": total, "sources": sources, "night_ratio": night_ratio, "temperature_c": temperatures}


def _condition_uncertainty_summary(
    rows: list[dict[str, object]], config: dict[str, Any], source_modes: dict[str, str] | None = None,
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
            if source_modes is not None and source_modes.get(source) == "unknown":
                maximum = float(dict(config.get("condition_source_sigma", {})).get(source, config.get("condition_sigma", 0.02)))
                unknown_prior = float(load_config().get("dynamic_environment", {}).get("uncertainty", {}).get("unknown_prior_activation", 0.0))
                source_scale = _source_uncertainty_scale(source, source_modes, config)
                sigma = max(sigma, maximum * max(0.0, min(1.0, unknown_prior)) * (1.25 - 0.5 * max(0.0, min(1.0, confidence))) * source_scale)
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
        return _sample_gpx_micro_batch(profile, micro_segments, count, rng, shared, sigma, correlation)
    return _sample_gpx_geometry_unit(profile, row, count, rng, shared, sigma, correlation)


def _sample_gpx_micro_batch(
    profile: dict[str, object],
    micro_segments: list[dict[str, object]],
    count: int,
    rng: np.random.Generator,
    shared: np.ndarray,
    sigma: float,
    correlation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorise GPX perturbation within one natural terrain segment.

    A long GPX has thousands of 25m micro-segments.  Sampling one vector per
    micro-segment made the Python loop dominate a 3,000-run simulation.  The
    batch remains bounded by one natural segment, so it avoids allocating an
    entire-route matrix while retaining the original global/segment correlation
    model.
    """
    total_seconds = np.zeros(count, dtype=float)
    weighted_grade = np.zeros(count, dtype=float)
    total_distance = 0.0
    correlation = max(0.0, min(1.0, correlation))
    shared_scale = np.sqrt(correlation)
    independent_scale = np.sqrt(1.0 - correlation)

    for terrain in TERRAINS:
        selected = [
            micro for micro in micro_segments
            if str(micro.get("type", "flat")) == terrain and bool(micro.get("elevation_available", True))
        ]
        if not selected:
            continue
        distances = np.asarray([max(float(micro.get("distance", 0.0)), 0.1) for micro in selected])
        grades = np.asarray([float(micro.get("grade", 0.0)) for micro in selected])
        total_distance += float(distances.sum())
        if terrain == "flat":
            total_seconds += float(distances.sum()) / 1000.0 * float(profile["flat"]["aerobic_pace"])
            weighted_grade += float(np.dot(grades, distances))
            continue

        z = shared_scale * shared[None, :] + independent_scale * rng.normal(size=(len(selected), count))
        if terrain == "uphill":
            vertical = np.asarray([
                max(float(micro.get("gain", 0.0)), distance * max(grade, 0.0) / 100.0, 0.1)
                for micro, distance, grade in zip(selected, distances, grades)
            ])
            sampled_vertical = vertical[:, None] * np.exp(sigma * z)
            sampled_grade = sampled_vertical / distances[:, None] * 100.0
            curve = list(profile.get("uphill", {}).get("curve", []))
            if curve:
                curve_grades = np.asarray([float(point["grade"]) for point in curve])
                curve_values = np.asarray([float(point["value"]) for point in curve])
                vam = np.interp(sampled_grade, curve_grades, curve_values)
                climbing = sampled_vertical / np.maximum(vam, 1.0) * 3600.0
                flat_floor = distances[:, None] / 1000.0 * float(profile["flat"]["aerobic_pace"])
                total_seconds += np.maximum(climbing, flat_floor).sum(axis=0)
            else:
                total_seconds += np.asarray([
                    _geometry_seconds(profile, micro) for micro in selected
                ]).sum()
            weighted_grade += (sampled_grade * distances[:, None]).sum(axis=0)
            continue

        vertical = np.asarray([
            max(float(micro.get("loss", 0.0)), distance * max(-grade, 0.0) / 100.0, 0.1)
            for micro, distance, grade in zip(selected, distances, grades)
        ])
        sampled_vertical = vertical[:, None] * np.exp(sigma * z)
        sampled_grade = -sampled_vertical / distances[:, None] * 100.0
        curve = list(profile.get("downhill", {}).get("curve", []))
        if curve:
            curve_grades = np.asarray([float(point["grade"]) for point in curve])
            curve_speeds = np.asarray([float(point["speed_mps"]) for point in curve])
            order = np.argsort(curve_grades)
            speed = np.interp(sampled_grade, curve_grades[order], curve_speeds[order])
            total_seconds += (distances[:, None] / np.maximum(speed, 0.1)).sum(axis=0)
        else:
            total_seconds += np.asarray([_geometry_seconds(profile, micro) for micro in selected]).sum()
        weighted_grade += (sampled_grade * distances[:, None]).sum(axis=0)

    unavailable = [micro for micro in micro_segments if not bool(micro.get("elevation_available", True))]
    if unavailable:
        distances = np.asarray([max(float(micro.get("distance", 0.0)), 0.1) for micro in unavailable])
        grades = np.asarray([float(micro.get("grade", 0.0)) for micro in unavailable])
        total_distance += float(distances.sum())
        total_seconds += np.asarray([_geometry_seconds(profile, micro) for micro in unavailable]).sum()
        weighted_grade += float(np.dot(grades, distances))
    return total_seconds, weighted_grade / max(total_distance, 0.1)


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
