from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta, timezone
from pathlib import Path

from analysis.downhill import interpolate_downhill_speed
from analysis.environment import relative_altitude_factor, solar_elevation_degrees
from analysis.fatigue import interpolate_fatigue
from analysis.uphill import interpolate_uphill_vam
from analysis.heart_rate import heart_rate_pacing_adjustment, interpolate_hr_drift
from analysis.temperature import (
    heart_rate_heat_fatigue_time_factor,
    humidity_time_factor,
    race_temperature_at_elapsed,
    temperature_fatigue_time_factor,
    temperature_time_factor,
)
from config import load_config
from models import PredictionResult, RaceCondition, RunnerProfile
from parser.gpx_reader import route_summary
from predictor.condition_adjustment import condition_factors
from predictor.duration_adjustment import duration_match
from predictor.pacing_strategy import match_route_pacing_strategy, pacing_factor
from predictor.probability import simulate_segmented_finish_times


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
        route_pacing_match = match_route_pacing_strategy(profile, segments, estimate_hours)
        standard_rows, standard_seconds, _ = _predict_once(
            profile, segments, estimate_hours, RaceCondition(), route_pacing_match, apply_environment=False
        )
        updated_hours = standard_seconds / 3600.0
        if abs(updated_hours - estimate_hours) * 3600.0 <= float(duration_config["convergence_tolerance_seconds"]):
            converged = True
            estimate_hours = updated_hours
            break
        estimate_hours = updated_hours

    route_pacing_match = match_route_pacing_strategy(profile, segments, estimate_hours)
    standard_rows, standard_seconds, _ = _predict_once(
        profile, segments, estimate_hours, RaceCondition(), route_pacing_match, apply_environment=False
    )
    adjusted_rows, adjusted_seconds, breakdown = _predict_once(
        profile, segments, estimate_hours, race_condition, route_pacing_match
    )
    aid_seconds = race_condition.aid_station_minutes * 60.0
    confidence_details = _prediction_confidence_details(profile, adjusted_rows, gpx_quality_score)
    confidence = float(confidence_details["overall"])
    probability = simulate_segmented_finish_times(
        profile, adjusted_rows, aid_seconds, gpx_quality_score, simulations, seed
    )
    probability.setdefault("uncertainty", {})["route_weighted_confidence"] = confidence_details
    risks = _risk_notes(profile, adjusted_rows, gpx_quality_score, converged)
    environment_summary = _environment_summary(profile, adjusted_rows)

    payload = {
        "schema_version": "0.3",
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
        "pacing_strategy_match": route_pacing_match,
        "adjustment_breakdown": {**breakdown, "aid_station": round(aid_seconds, 1)},
        "probability": probability,
        "risk_notes": risks,
        "environment": environment_summary,
        "physiology": _physiology_summary(profile, race_condition, adjusted_rows),
        "segments": adjusted_rows,
        # V0.1 compatibility fields.
        "moving_time_seconds": round(adjusted_seconds, 1),
        "aid_time_seconds": round(aid_seconds, 1),
        "total_time_seconds": round(adjusted_seconds + aid_seconds, 1),
    }
    return PredictionResult.from_dict(payload).to_dict()


def _predict_once(
    profile: dict[str, object], segments: list[dict[str, float | str]], estimated_hours: float,
    condition: RaceCondition, route_pacing_match: dict[str, object], apply_environment: bool = True,
) -> tuple[list[dict[str, object]], float, dict[str, float]]:
    elapsed = 0.0
    rows: list[dict[str, object]] = []
    totals = {"base_terrain": 0.0, "heart_rate_pacing": 0.0,
              "pacing_strategy": 0.0, "fatigue": 0.0,
              "form": 0.0, "technical": 0.0, "mud": 0.0, "night": 0.0,
              "altitude": 0.0, "carried_weight": 0.0, "weather": 0.0,
              "temperature_fatigue": 0.0, "heart_rate_fatigue": 0.0}
    total_distance = sum(max(0.0, float(item.get("distance", 0.0))) for item in segments)
    completed_distance = 0.0
    for segment in segments:
        terrain = str(segment.get("type", "flat"))
        raw_seconds, basis = _base_segment_seconds(profile, segment)
        segment_distance = max(0.0, float(segment.get("distance", 0.0)))
        progress = (completed_distance + segment_distance / 2.0) / max(total_distance, 0.1)
        match = pacing_factor(route_pacing_match, terrain, progress)
        sustainable_seconds = raw_seconds * float(match["factor"])
        fatigue = fatigue_factor(profile, elapsed / 3600.0, terrain)
        standard_seconds = sustainable_seconds / max(fatigue, 0.1)
        base_output = _segment_output(segment, raw_seconds)
        pacing = (
            heart_rate_pacing_adjustment(
                profile,
                terrain,
                float(segment.get("grade", 0.0)),
                estimated_hours,
                condition.pacing_strategy,
                base_output,
                (elapsed + standard_seconds / 2.0) / 3600.0,
            )
            if apply_environment
            else {
                "strategy": "standard", "intensity_label": "标准能力", "target_hr_bpm": None,
                "expected_hr_bpm": None, "time_factor": 1.0, "output_factor": 1.0,
                "confidence": 0.2, "source": "baseline", "extrapolated": False,
            }
        )
        environment = _segment_environment(
            profile, segment, condition, elapsed, standard_seconds, apply_environment
        )
        physiology_elapsed_hours = (elapsed + standard_seconds / 2.0) / 3600.0
        segment_temperature = race_temperature_at_elapsed(
            condition, physiology_elapsed_hours, estimated_hours
        ) if apply_environment else None
        direct_temperature = (
            temperature_time_factor(profile, segment_temperature)
            * humidity_time_factor(segment_temperature, condition.humidity_percent)
            if apply_environment else 1.0
        )
        temperature_fatigue = (
            temperature_fatigue_time_factor(profile, segment_temperature, physiology_elapsed_hours)
            if apply_environment else 1.0
        )
        heart_rate_fatigue = (
            heart_rate_heat_fatigue_time_factor(profile, segment_temperature, physiology_elapsed_hours)
            if apply_environment else 1.0
        )
        factors = condition_factors(
            condition,
            terrain,
            target_night_ratio=float(environment["night_ratio"]),
            historical_night_ratio=float(environment["historical_night_ratio"]),
            automatic_altitude_factor=float(environment["altitude_factor"]),
            temperature_factor=direct_temperature,
        )
        factors = {"heart_rate_pacing": float(pacing["time_factor"]), **factors}
        weather = factors.pop("weather")
        factors["temperature_fatigue"] = temperature_fatigue
        factors["heart_rate_fatigue"] = heart_rate_fatigue
        factors["weather"] = weather
        temperature_confidence = float(profile.get("temperature", {}).get("confidence", 0.2))
        heart_rate_confidence = float(profile.get("heart_rate", {}).get("confidence", 0.2))
        condition_confidence = {
            "heart_rate_pacing": float(pacing.get("confidence", 0.2)),
            "weather": temperature_confidence,
            "temperature_fatigue": temperature_confidence,
            "heart_rate_fatigue": heart_rate_confidence,
        }
        adjusted = standard_seconds
        factor_increases: dict[str, float] = {}
        for name, factor in factors.items():
            increase = adjusted * (float(factor) - 1.0)
            factor_increases[name] = increase
            totals[name] += increase
            adjusted *= float(factor)
        elapsed += adjusted
        completed_distance += segment_distance
        totals["base_terrain"] += raw_seconds
        totals["pacing_strategy"] += sustainable_seconds - raw_seconds
        totals["fatigue"] += standard_seconds - sustainable_seconds
        rows.append({**segment, "base_time_seconds": round(raw_seconds, 1),
                     "duration_factor": round(float(match["factor"]), 4),
                     "duration_confidence": float(match["confidence"]),
                     "duration_weights": match["weights"], "duration_source": match["source"],
                     "pacing_strategy_factor": round(float(match["factor"]), 4),
                     "pacing_strategy_progress": round(progress, 4),
                     "sustainable_time_seconds": round(sustainable_seconds, 1),
                     "fatigue_factor": round(fatigue, 3),
                     "standard_time_seconds": round(standard_seconds, 1),
                     "condition_factors": {key: round(value, 4) for key, value in factors.items()},
                     "condition_confidence": condition_confidence,
                     "condition_factor": round(adjusted / max(standard_seconds, 0.1), 4),
                     "condition_increase_seconds": {key: round(value, 1) for key, value in factor_increases.items()},
                     "environment": environment,
                     "physiology": {
                         "temperature_c": segment_temperature,
                         "temperature_direct_factor": round(direct_temperature, 4),
                         "temperature_fatigue_factor": round(temperature_fatigue, 4),
                         "heart_rate_fatigue_factor": round(heart_rate_fatigue, 4),
                         "pacing": pacing,
                         "expected_hr_drift_bpm": round(interpolate_hr_drift(profile, physiology_elapsed_hours), 1),
                         "temperature_model_source": str(profile.get("temperature", {}).get("source", "default")),
                         "heart_rate_model_source": str(profile.get("heart_rate", {}).get("source", "unavailable")),
                     },
                     "predicted_time_seconds": round(adjusted, 1),
                     "cumulative_time_seconds": round(elapsed, 1), "basis": basis})
    return rows, elapsed, {key: round(value, 1) for key, value in totals.items()}


def _segment_output(segment: dict[str, float | str], raw_seconds: float) -> float:
    if raw_seconds <= 0:
        return 0.0
    if str(segment.get("type", "flat")) == "uphill":
        gain = max(float(segment.get("gain", 0.0)), float(segment["distance"]) * max(float(segment["grade"]), 0.0) / 100.0)
        return gain / raw_seconds * 3600.0
    return float(segment["distance"]) / raw_seconds


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


def _base_segment_seconds(profile: dict[str, object], segment: dict[str, object]) -> tuple[float, str]:
    micro_segments = list(segment.get("micro_segments", []))
    if micro_segments:
        results = [_base_terrain_unit_seconds(profile, micro) for micro in micro_segments]
        seconds = sum(result[0] for result in results)
        grades = [float(micro.get("grade", 0.0)) for micro in micro_segments]
        return seconds, f"{len(micro_segments)} 个微分段 / 局部坡度 {min(grades):.1f}%～{max(grades):.1f}%"
    return _base_terrain_unit_seconds(profile, segment)


def _base_terrain_unit_seconds(profile: dict[str, object], segment: dict[str, object]) -> tuple[float, str]:
    grade = float(segment["grade"])
    distance = float(segment["distance"])
    segment_type = str(segment.get("type", "flat"))
    if segment_type == "uphill":
        curve = profile["uphill"].get("curve", [])
        if curve:
            vam = interpolate_uphill_vam(grade, [(float(point["grade"]), float(point["value"])) for point in curve])
        else:
            label = "20_percent" if grade >= 20 else "15_percent" if grade >= 15 else "10_percent" if grade >= 10 else "5_percent" if grade >= 5 else "1_percent"
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
            label = "-20_percent" if grade <= -20 else "-15_percent" if grade <= -15 else "-10_percent" if grade <= -10 else "-5_percent" if grade <= -5 else "-1_percent"
            speed = float(profile["downhill"][label]["speed_mps"])
        return distance / max(speed, 0.1), f"{grade:.1f}%坡 / 下坡速度 {speed:.2f} m/s"
    pace = float(profile["flat"]["aerobic_pace"])
    return distance / 1000.0 * pace, f"平路配速 {format_pace(pace)}/km"


def _initial_estimate_hours(profile: dict[str, object], segments: list[dict[str, float | str]]) -> float:
    return max(0.1, sum(_base_segment_seconds(profile, segment)[0] for segment in segments) / 3600.0)


def _prediction_confidence(
    profile: dict[str, object], rows: list[dict[str, object]], gpx_quality_score: float = 1.0
) -> float:
    return float(_prediction_confidence_details(profile, rows, gpx_quality_score)["overall"])


def _prediction_confidence_details(
    profile: dict[str, object], rows: list[dict[str, object]], gpx_quality_score: float = 1.0
) -> dict[str, object]:
    total_seconds = sum(float(row.get("predicted_time_seconds", 0.0)) for row in rows)
    terrain_totals = {terrain: 0.0 for terrain in ("flat", "uphill", "downhill")}
    terrain_weighted = {
        terrain: {"ability": 0.0, "fatigue": 0.0, "duration": 0.0, "combined": 0.0}
        for terrain in terrain_totals
    }
    route_weighted = 0.0
    for row in rows:
        terrain = str(row.get("type", "flat"))
        seconds = float(row.get("predicted_time_seconds", 0.0))
        ability = _ability_confidence_for_grade(profile, terrain, float(row.get("grade", 0.0)))
        fatigue = _fatigue_confidence_for_terrain(profile, terrain)
        duration = float(row.get("duration_confidence", 0.2))
        combined = ability * 0.50 + fatigue * 0.25 + duration * 0.25
        terrain_totals[terrain] += seconds
        terrain_weighted[terrain]["ability"] += seconds * ability
        terrain_weighted[terrain]["fatigue"] += seconds * fatigue
        terrain_weighted[terrain]["duration"] += seconds * duration
        terrain_weighted[terrain]["combined"] += seconds * combined
        route_weighted += seconds * combined

    route_capability = route_weighted / total_seconds if total_seconds > 0 else 0.2
    data_quality = float(profile.get("data_quality", {}).get("score", 0.2))
    confidence = route_capability * 0.85 + data_quality * 0.15
    altitude_penalty = 0.0
    altitude_profile = profile.get("environment", {}).get("altitude", {})
    historical_p90 = float(altitude_profile.get("p90_m", 0.0))
    route_max = max((float(row.get("environment", {}).get("elevation_m", 0.0)) for row in rows), default=0.0)
    environment_config = load_config()["environment"]["altitude"]
    if route_max > historical_p90 + float(environment_config["coverage_margin_m"]):
        altitude_penalty = float(environment_config["confidence_penalty"])
        confidence -= altitude_penalty
    gpx_quality = max(0.0, min(1.0, float(gpx_quality_score)))
    confidence = confidence * 0.90 + gpx_quality * 0.10
    physiology_confidence: float | None = None
    temperature_active = any(row.get("physiology", {}).get("temperature_c") is not None for row in rows)
    pacing_active = any(
        row.get("physiology", {}).get("pacing", {}).get("source") == "personal" for row in rows
    )
    if temperature_active or pacing_active:
        temperature_confidence = float(profile.get("temperature", {}).get("confidence", 0.2))
        hr_active = any(
            float(row.get("physiology", {}).get("heart_rate_fatigue_factor", 1.0)) > 1.0001
            for row in rows
        )
        heart_rate_confidence = float(profile.get("heart_rate", {}).get("confidence", 0.2))
        if temperature_active:
            physiology_confidence = (
                temperature_confidence * 0.75 + heart_rate_confidence * 0.25
                if hr_active or pacing_active else temperature_confidence
            )
        else:
            physiology_confidence = heart_rate_confidence
        confidence = confidence * 0.90 + physiology_confidence * 0.10

    terrain_details: dict[str, dict[str, float]] = {}
    for terrain, seconds in terrain_totals.items():
        terrain_details[terrain] = {
            "time_share": round(seconds / total_seconds, 4) if total_seconds > 0 else 0.0,
            **{
                key: round(value / seconds, 3) if seconds > 0 else 0.0
                for key, value in terrain_weighted[terrain].items()
            },
        }
    return {
        "overall": round(max(0.2, min(0.95, confidence)), 3),
        "route_capability": round(route_capability, 3),
        "data_quality": round(data_quality, 3),
        "gpx_quality": round(gpx_quality, 3),
        "physiology_confidence": None if physiology_confidence is None else round(physiology_confidence, 3),
        "altitude_penalty": round(altitude_penalty, 3),
        "terrain": terrain_details,
    }


def _ability_confidence_for_grade(profile: dict[str, object], terrain: str, grade: float) -> float:
    if terrain == "flat":
        return float(profile.get("flat", {}).get("confidence", 0.2))
    curve = sorted(
        list(profile.get(terrain, {}).get("curve", [])), key=lambda point: float(point["grade"])
    )
    if not curve:
        return 0.2
    if grade <= float(curve[0]["grade"]):
        return float(curve[0].get("confidence", 0.2) or 0.2)
    if grade >= float(curve[-1]["grade"]):
        return float(curve[-1].get("confidence", 0.2) or 0.2)
    for lower, upper in zip(curve, curve[1:]):
        lower_grade = float(lower["grade"])
        upper_grade = float(upper["grade"])
        if lower_grade <= grade <= upper_grade:
            weight = (grade - lower_grade) / max(upper_grade - lower_grade, 1e-9)
            lower_confidence = float(lower.get("confidence", 0.2) or 0.2)
            upper_confidence = float(upper.get("confidence", 0.2) or 0.2)
            return lower_confidence * (1.0 - weight) + upper_confidence * weight
    return 0.2


def _fatigue_confidence_for_terrain(profile: dict[str, object], terrain: str) -> float:
    values = [
        float(point["confidence"])
        for point in profile.get("fatigue", {}).get(terrain, [])
        if point.get("confidence") is not None
    ]
    return sum(values) / len(values) if values else 0.2


def _risk_notes(profile: dict[str, object], rows: list[dict[str, object]], gpx_quality: float, converged: bool) -> list[str]:
    notes: list[str] = []
    if any(row.get("duration_source") == "duration_fallback" for row in rows):
        notes.append("没有足够相似的历史距离与爬升样本，比赛配速策略已回退到时长能力层。")
    if gpx_quality < 0.7:
        notes.append("GPX 数据质量较低，预测区间已扩大。")
    if not converged:
        notes.append("配速策略、疲劳与预计时长的迭代未完全收敛。")
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
    if any(row.get("physiology", {}).get("temperature_c") is not None for row in rows):
        temperature_source = str(profile.get("temperature", {}).get("source", "unavailable"))
        if temperature_source in {"unavailable", "default"}:
            notes.append("历史 FIT 温度样本不足，本场温度影响使用系统默认曲线，概率区间已考虑较低可信度。")
        if str(profile.get("heart_rate", {}).get("source", "unavailable")) == "unavailable":
            notes.append("历史 FIT 心率覆盖不足，未应用个人心率热应激疲劳修正。")
    calibration = profile.get("temperature", {}).get("calibration", {})
    if calibration.get("source") == "wrist_relative_only":
        notes.append("历史温度来自腕表设备，未反推环境气温；个人绝对耐热曲线已停用，本场使用系统默认温度曲线。")
    elif calibration.get("source") == "historical_weather":
        notes.append("历史环境温度来自约9–11km再分析天气网格；山区林下、山谷和暴晒路段仍可能与网格气温不同。")
    if any(row.get("physiology", {}).get("pacing", {}).get("extrapolated") for row in rows):
        notes.append("目标比赛强度超出部分历史心率样本范围，配速已按历史输出上限限制。")
    if rows and rows[0].get("physiology", {}).get("pacing", {}).get("strategy") == "aggressive" and not any(
        row.get("physiology", {}).get("pacing", {}).get("source") == "personal" for row in rows
    ):
        notes.append("选择了积极策略，但对应坡度缺少可靠心率—输出样本，未强行提高配速。")
    return notes


def _physiology_summary(
    profile: dict[str, object], condition: RaceCondition, rows: list[dict[str, object]]
) -> dict[str, object]:
    temperature = profile.get("temperature", {})
    heart_rate = profile.get("heart_rate", {})
    return {
        "race_temperature_c": condition.temperature_c,
        "race_temperature_schedule": {
            "start_c": condition.temperature_c,
            "peak_c": condition.temperature_peak_c,
            "peak_hour": condition.temperature_peak_hour,
            "finish_c": condition.temperature_finish_c,
        },
        "temperature_model_source": str(temperature.get("source", "unavailable")),
        "temperature_model_confidence": float(temperature.get("confidence", 0.2)),
        "reference_temperature_c": temperature.get("reference_temperature_c"),
        "best_temperature_range_c": temperature.get("best_range_c"),
        "direct_temperature_factor": max(
            (float(row.get("physiology", {}).get("temperature_direct_factor", 1.0)) for row in rows),
            default=1.0,
        ),
        "maximum_temperature_fatigue_factor": max(
            (float(row.get("physiology", {}).get("temperature_fatigue_factor", 1.0)) for row in rows),
            default=1.0,
        ),
        "heart_rate_model_source": str(heart_rate.get("source", "unavailable")),
        "heart_rate_model_confidence": float(heart_rate.get("confidence", 0.2)),
        "pacing_strategy": condition.pacing_strategy,
        "pacing_strategy_label": {"conservative": "保守", "standard": "标准", "aggressive": "积极"}.get(condition.pacing_strategy, "标准"),
        "heart_rate_pacing_applied": any(
            row.get("physiology", {}).get("pacing", {}).get("source") == "personal" for row in rows
        ),
        "target_hr_bpm_range": _pacing_hr_range(rows, "target_hr_bpm"),
        "expected_hr_bpm_range": _pacing_hr_range(rows, "expected_hr_bpm"),
        "aerobic_range": heart_rate.get("aerobic_range", {}),
        "threshold": heart_rate.get("threshold", {}),
        "expected_hr_drift_at_finish_bpm": round(interpolate_hr_drift(profile, rows[-1]["cumulative_time_seconds"] / 3600.0), 1) if rows else 0.0,
        "maximum_heart_rate_fatigue_factor": max(
            (float(row.get("physiology", {}).get("heart_rate_fatigue_factor", 1.0)) for row in rows),
            default=1.0,
        ),
    }


def _pacing_hr_range(rows: list[dict[str, object]], field: str) -> list[float] | None:
    values = [
        float(row["physiology"]["pacing"][field])
        for row in rows
        if row.get("physiology", {}).get("pacing", {}).get(field) is not None
    ]
    return [round(min(values), 1), round(max(values), 1)] if values else None


def _segment_environment(
    profile: dict[str, object],
    segment: dict[str, object],
    condition: RaceCondition,
    elapsed_seconds: float,
    segment_seconds: float,
    apply_environment: bool,
) -> dict[str, object]:
    history = profile.get("environment", {})
    night_history = history.get("night", {})
    historical_night = (
        float(night_history.get("terrain", {}).get(str(segment.get("type", "flat")), {}).get("ratio", night_history.get("ratio", 0.0)))
        if apply_environment else 0.0
    )
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
        "historical_night_by_terrain": {
            terrain: float(profile.get("environment", {}).get("night", {}).get("terrain", {}).get(terrain, {}).get("ratio", 0.0))
            for terrain in ("flat", "uphill", "downhill")
        },
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
