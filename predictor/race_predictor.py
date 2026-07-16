from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta, timezone
from pathlib import Path

from analysis.downhill import interpolate_downhill_speed
from analysis.environment import relative_altitude_factor, solar_elevation_degrees
from analysis.fatigue import interpolate_fatigue
from analysis.uphill import interpolate_uphill_vam
from config import load_config
from models import PredictionResult, RaceCondition, RunnerProfile
from parser.gpx_reader import route_summary
from predictor.condition_adjustment import condition_factors
from predictor.duration_adjustment import duration_match
from predictor.probability import simulate_finish_times


def predict_race(
    profile: dict[str, object],
    segments: list[dict[str, float | str]],
    aid_minutes: float = 0.0,
    condition: RaceCondition | None = None,
    simulations: int | None = None,
    seed: int | None = None,
    gpx_quality_score: float = 1.0,
) -> dict[str, object]:
    """Predict standard, condition-adjusted and probabilistic race times."""
    if not segments:
        raise ValueError("比赛路线没有分段数据")
    profile = RunnerProfile.from_profile_dict(profile).to_profile_dict()
    race_condition = (condition or RaceCondition(aid_station_minutes=aid_minutes)).normalized()
    if condition is not None and aid_minutes > 0 and race_condition.aid_station_minutes == 0:
        race_condition = replace(race_condition, aid_station_minutes=aid_minutes)

    estimate_hours = _initial_estimate_hours(profile, segments)
    duration_config = load_config()["duration_capability"]
    converged = False
    standard_rows: list[dict[str, object]] = []
    standard_seconds = 0.0
    for iteration in range(1, int(duration_config["max_iterations"]) + 1):
        standard_rows, standard_seconds, _ = _predict_once(
            profile, segments, estimate_hours, RaceCondition(), apply_environment=False
        )
        updated_hours = standard_seconds / 3600.0
        if abs(updated_hours - estimate_hours) * 3600.0 <= float(duration_config["convergence_tolerance_seconds"]):
            converged = True
            estimate_hours = updated_hours
            break
        estimate_hours = updated_hours

    adjusted_rows, adjusted_seconds, breakdown = _predict_once(profile, segments, estimate_hours, race_condition)
    aid_seconds = race_condition.aid_station_minutes * 60.0
    confidence = _prediction_confidence(profile, adjusted_rows)
    probability = simulate_finish_times(adjusted_seconds, aid_seconds, confidence, gpx_quality_score, simulations, seed)
    risks = _risk_notes(profile, adjusted_rows, gpx_quality_score, converged)
    environment_summary = _environment_summary(profile, adjusted_rows)

    payload = {
        "schema_version": "0.2",
        "route": route_summary(segments),
        "condition": race_condition.to_dict(),
        "standard_moving_time_seconds": round(standard_seconds, 1),
        "adjusted_moving_time_seconds": round(adjusted_seconds, 1),
        "aid_station_time_seconds": round(aid_seconds, 1),
        "median_finish_time_seconds": probability["p50_seconds"],
        "optimistic_time_seconds": probability["p10_seconds"],
        "conservative_time_seconds": probability["p90_seconds"],
        "confidence": round(confidence, 3),
        "duration_match": {"estimated_hours": round(estimate_hours, 3), "converged": converged,
                           "iterations": iteration, "terrain": {terrain: duration_match(profile, estimate_hours, terrain) for terrain in ("flat", "uphill", "downhill")}},
        "adjustment_breakdown": {**breakdown, "aid_station": round(aid_seconds, 1)},
        "probability": probability,
        "risk_notes": risks,
        "environment": environment_summary,
        "segments": adjusted_rows,
        # V0.1 compatibility fields.
        "moving_time_seconds": round(adjusted_seconds, 1),
        "aid_time_seconds": round(aid_seconds, 1),
        "total_time_seconds": round(adjusted_seconds + aid_seconds, 1),
    }
    return PredictionResult.from_dict(payload).to_dict()


def _predict_once(
    profile: dict[str, object], segments: list[dict[str, float | str]], estimated_hours: float,
    condition: RaceCondition, apply_environment: bool = True,
) -> tuple[list[dict[str, object]], float, dict[str, float]]:
    elapsed = 0.0
    rows: list[dict[str, object]] = []
    totals = {"base_terrain": 0.0, "duration_adaptation": 0.0, "fatigue": 0.0,
              "form": 0.0, "technical": 0.0, "mud": 0.0, "night": 0.0,
              "altitude": 0.0, "carried_weight": 0.0, "weather": 0.0}
    for segment in segments:
        terrain = str(segment.get("type", "flat"))
        raw_seconds, basis = _base_segment_seconds(profile, segment)
        match = duration_match(profile, estimated_hours, terrain)
        sustainable_seconds = raw_seconds * float(match["factor"])
        fatigue = fatigue_factor(profile, elapsed / 3600.0, terrain)
        standard_seconds = sustainable_seconds / max(fatigue, 0.1)
        environment = _segment_environment(
            profile, segment, condition, elapsed, standard_seconds, apply_environment
        )
        factors = condition_factors(
            condition,
            terrain,
            target_night_ratio=float(environment["night_ratio"]),
            historical_night_ratio=float(environment["historical_night_ratio"]),
            automatic_altitude_factor=float(environment["altitude_factor"]),
        )
        adjusted = standard_seconds
        factor_increases: dict[str, float] = {}
        for name, factor in factors.items():
            increase = adjusted * (float(factor) - 1.0)
            factor_increases[name] = increase
            totals[name] += increase
            adjusted *= float(factor)
        elapsed += adjusted
        totals["base_terrain"] += raw_seconds
        totals["duration_adaptation"] += sustainable_seconds - raw_seconds
        totals["fatigue"] += standard_seconds - sustainable_seconds
        rows.append({**segment, "base_time_seconds": round(raw_seconds, 1),
                     "duration_factor": round(float(match["factor"]), 4),
                     "duration_confidence": float(match["confidence"]),
                     "duration_weights": match["weights"], "duration_source": match["source"],
                     "sustainable_time_seconds": round(sustainable_seconds, 1),
                     "fatigue_factor": round(fatigue, 3),
                     "standard_time_seconds": round(standard_seconds, 1),
                     "condition_factors": {key: round(value, 4) for key, value in factors.items()},
                     "condition_factor": round(adjusted / max(standard_seconds, 0.1), 4),
                     "condition_increase_seconds": {key: round(value, 1) for key, value in factor_increases.items()},
                     "environment": environment,
                     "predicted_time_seconds": round(adjusted, 1),
                     "cumulative_time_seconds": round(elapsed, 1), "basis": basis})
    return rows, elapsed, {key: round(value, 1) for key, value in totals.items()}


def fatigue_factor(profile: dict[str, object], elapsed_hour: float, terrain: str = "flat") -> float:
    fatigue = profile["fatigue"]
    curve = fatigue.get(terrain)
    if curve:
        return interpolate_fatigue(elapsed_hour, curve)
    if elapsed_hour <= 3.0:
        return float(fatigue["3h"])
    if elapsed_hour <= 5.0:
        return float(fatigue["5h"])
    return float(fatigue["8h"])


def save_prediction(prediction: dict[str, object], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")


def _base_segment_seconds(profile: dict[str, object], segment: dict[str, float | str]) -> tuple[float, str]:
    grade = float(segment["grade"])
    distance = float(segment["distance"])
    segment_type = str(segment.get("type", "flat"))
    if segment_type == "uphill":
        curve = profile["uphill"].get("curve", [])
        if curve:
            vam = interpolate_uphill_vam(grade, [(float(point["grade"]), float(point["value"])) for point in curve])
        else:
            label = "15_percent" if grade >= 15 else "10_percent" if grade >= 10 else "5_percent" if grade >= 5 else "1_percent"
            vam = float(profile["uphill"][label])
        gain = max(float(segment["gain"]), distance * grade / 100.0)
        climbing_seconds = gain / max(vam, 1.0) * 3600.0
        flat_seconds = distance / 1000.0 * float(profile["flat"]["aerobic_pace"])
        return max(climbing_seconds, flat_seconds), f"{grade:.1f}%坡 / VAM {vam:.0f} m/h"
    if segment_type == "downhill":
        curve = profile["downhill"].get("curve", [])
        if curve:
            speed = interpolate_downhill_speed(grade, [(float(point["grade"]), float(point["speed_mps"])) for point in curve])
        else:
            label = "-15_percent" if grade <= -15 else "-10_percent" if grade <= -10 else "-5_percent" if grade <= -5 else "-1_percent"
            speed = float(profile["downhill"][label]["speed_mps"])
        return distance / max(speed, 0.1), f"{grade:.1f}%坡 / 下坡速度 {speed:.2f} m/s"
    pace = float(profile["flat"]["aerobic_pace"])
    return distance / 1000.0 * pace, f"平路配速 {format_pace(pace)}/km"


def _initial_estimate_hours(profile: dict[str, object], segments: list[dict[str, float | str]]) -> float:
    return max(0.1, sum(_base_segment_seconds(profile, segment)[0] for segment in segments) / 3600.0)


def _prediction_confidence(profile: dict[str, object], rows: list[dict[str, object]]) -> float:
    values = [float(profile.get("flat", {}).get("confidence", 0.2))]
    values.extend(float(point.get("confidence", 0.2)) for point in profile.get("uphill", {}).get("curve", []))
    values.extend(float(point.get("confidence", 0.2)) for point in profile.get("downhill", {}).get("curve", []))
    values.extend(float(row.get("duration_confidence", 0.2)) for row in rows)
    values.append(float(profile.get("data_quality", {}).get("score", 0.2)))
    confidence = sum(values) / len(values)
    altitude_profile = profile.get("environment", {}).get("altitude", {})
    historical_p90 = float(altitude_profile.get("p90_m", 0.0))
    route_max = max((float(row.get("environment", {}).get("elevation_m", 0.0)) for row in rows), default=0.0)
    environment_config = load_config()["environment"]["altitude"]
    if route_max > historical_p90 + float(environment_config["coverage_margin_m"]):
        confidence -= float(environment_config["confidence_penalty"])
    return max(0.2, min(0.95, confidence))


def _risk_notes(profile: dict[str, object], rows: list[dict[str, object]], gpx_quality: float, converged: bool) -> list[str]:
    notes: list[str] = []
    if any(row.get("duration_source") == "fallback" for row in rows):
        notes.append("目标时长附近的个人样本不足，持续能力使用了保守折减。")
    if gpx_quality < 0.7:
        notes.append("GPX 数据质量较低，预测区间已扩大。")
    if not converged:
        notes.append("持续能力与预计时长迭代未完全收敛。")
    if float(profile.get("data_quality", {}).get("score", 1.0)) < 0.6:
        notes.append("历史 FIT 数据质量偏低。")
    altitude_profile = profile.get("environment", {}).get("altitude", {})
    historical_p90 = float(altitude_profile.get("p90_m", 0.0))
    route_max = max((float(row.get("environment", {}).get("elevation_m", 0.0)) for row in rows), default=0.0)
    margin = float(load_config()["environment"]["altitude"]["coverage_margin_m"])
    if route_max > historical_p90 + margin:
        notes.append(
            f"比赛最高海拔约 {route_max:.0f}m，明显高于历史训练覆盖 {historical_p90:.0f}m，已降低预测可信度。"
        )
    if any(bool(row.get("environment", {}).get("night")) for row in rows):
        night_source = profile.get("environment", {}).get("night", {}).get("source")
        if night_source == "unavailable":
            notes.append("比赛包含夜间路段，但历史 FIT 缺少可用经纬度，夜间能力使用默认折减。")
    return notes


def _segment_environment(
    profile: dict[str, object],
    segment: dict[str, object],
    condition: RaceCondition,
    elapsed_seconds: float,
    segment_seconds: float,
    apply_environment: bool,
) -> dict[str, object]:
    history = profile.get("environment", {})
    historical_night = float(history.get("night", {}).get("ratio", 0.0)) if apply_environment else 0.0
    target_night = float(condition.night_running_ratio) if apply_environment else 0.0
    solar_elevation: float | None = None
    night = target_night >= 0.5
    if apply_environment and condition.race_start_time_utc is not None and "latitude" in segment and "longitude" in segment:
        start_time = condition.race_start_time_utc
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        midpoint = start_time + timedelta(seconds=elapsed_seconds + segment_seconds / 2.0)
        solar_elevation = solar_elevation_degrees(midpoint, float(segment["latitude"]), float(segment["longitude"]))
        threshold = float(load_config()["environment"]["night_solar_elevation_degrees"])
        night = solar_elevation <= threshold
        target_night = 1.0 if night else 0.0
    elevation_available = bool(segment.get("elevation_available", False))
    elevation = float(segment.get("elevation", 0.0)) if elevation_available else 0.0
    historical_elevation = float(history.get("altitude", {}).get("mean_m", 0.0))
    altitude_factor = (
        relative_altitude_factor(elevation, historical_elevation)
        if apply_environment and elevation_available
        else float(condition.altitude_factor) if apply_environment else 1.0
    )
    return {
        "night": night,
        "night_ratio": round(target_night, 3),
        "historical_night_ratio": round(historical_night, 3),
        "solar_elevation_degrees": None if solar_elevation is None else round(solar_elevation, 1),
        "elevation_m": round(elevation, 1),
        "elevation_available": elevation_available,
        "historical_elevation_m": round(historical_elevation, 1),
        "altitude_factor": round(altitude_factor, 4),
    }


def _environment_summary(profile: dict[str, object], rows: list[dict[str, object]]) -> dict[str, object]:
    total_seconds = sum(float(row["predicted_time_seconds"]) for row in rows)
    night_seconds = sum(
        float(row["predicted_time_seconds"]) * float(row.get("environment", {}).get("night_ratio", 0.0))
        for row in rows
    )
    altitude_rows = [row for row in rows if bool(row.get("environment", {}).get("elevation_available", False))]
    distance = sum(float(row["distance"]) for row in altitude_rows)
    average_elevation = (
        sum(float(row["environment"]["elevation_m"]) * float(row["distance"]) for row in altitude_rows) / distance
        if distance > 0 else None
    )
    maximum_elevation = max((float(row["environment"]["elevation_m"]) for row in altitude_rows), default=None)
    return {
        "historical_night_ratio": float(profile.get("environment", {}).get("night", {}).get("ratio", 0.0)),
        "race_night_ratio": round(night_seconds / total_seconds, 4) if total_seconds > 0 else 0.0,
        "race_night_seconds": round(night_seconds, 1),
        "historical_mean_elevation_m": float(profile.get("environment", {}).get("altitude", {}).get("mean_m", 0.0)),
        "historical_p90_elevation_m": float(profile.get("environment", {}).get("altitude", {}).get("p90_m", 0.0)),
        "race_average_elevation_m": None if average_elevation is None else round(average_elevation, 1),
        "race_maximum_elevation_m": None if maximum_elevation is None else round(maximum_elevation, 1),
    }


def format_duration(seconds: float) -> str:
    sign = "-" if seconds < 0 else ""
    total_minutes = abs(int(round(seconds / 60.0)))
    hours, minutes = divmod(total_minutes, 60)
    text = f"{hours}小时{minutes:02d}分钟" if hours else f"{minutes}分钟"
    return sign + text


def format_pace(seconds_per_km: float) -> str:
    minutes, seconds = divmod(int(round(seconds_per_km)), 60)
    return f"{minutes}:{seconds:02d}"
