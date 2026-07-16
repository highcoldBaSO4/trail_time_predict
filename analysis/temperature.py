from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from analysis.confidence import calculate_confidence
from analysis.fatigue import interpolate_fatigue
from analysis.heart_rate import interpolate_hr_drift
from config import load_config


def calibrate_activity_temperature(activity: pd.DataFrame) -> pd.DataFrame:
    """Keep wrist temperature as relative exposure, never as ambient truth.

    FIT does not provide enough information to invert wrist-device temperature
    into ambient temperature. Frames without device_temperature are treated as
    already-ambient data to preserve explicit/programmatic inputs.
    """
    data = activity.copy()
    existing_calibration = dict(data.attrs.get("temperature_calibration", {}))
    existing_ambient = (
        pd.to_numeric(data["temperature"], errors="coerce")
        if "temperature" in data.columns else pd.Series(dtype=float)
    )
    if existing_calibration.get("source") == "historical_weather" and existing_ambient.notna().any():
        return data
    if "device_temperature" not in data:
        data.attrs["temperature_calibration"] = {
            "source": "ambient_assumed",
            "absolute_temperature_available": True,
            "model_weight": 1.0,
        }
        return data
    device = pd.to_numeric(data["device_temperature"], errors="coerce")
    if not device.notna().any():
        data["temperature"] = np.nan
        data["temperature_weight"] = 0.0
        data.attrs["temperature_calibration"] = {
            "source": "unavailable",
            "absolute_temperature_available": False,
            "model_weight": 0.0,
        }
        return data
    timestamp = pd.to_datetime(data.get("timestamp"), errors="coerce", utc=True)
    elapsed_minutes = (timestamp - timestamp.min()).dt.total_seconds() / 60.0
    warmup_minutes = float(load_config()["temperature_model"]["wrist_relative"]["warmup_minutes"])
    settled = device[elapsed_minutes >= warmup_minutes].dropna()
    baseline = float(settled.median()) if not settled.empty else (
        float(device.dropna().median()) if device.notna().any() else np.nan
    )
    data["device_temperature_relative"] = device - baseline
    data["temperature"] = np.nan
    data["temperature_weight"] = 0.0
    data.attrs["temperature_calibration"] = {
        "source": "wrist_relative_only",
        "warmup_minutes": warmup_minutes,
        "relative_baseline_c": None if not np.isfinite(baseline) else round(baseline, 1),
        "absolute_temperature_available": False,
        "model_weight": 0.0,
    }
    return data


def build_temperature_profile(
    activities: list[pd.DataFrame], fatigue_profile: dict[str, object]
) -> dict[str, object]:
    """Build a terrain/fatigue-normalized personal temperature time curve."""
    config = load_config()["temperature_model"]
    default_curve = [dict(point) for point in config["default_curve"]]
    prepared = [_prepare_samples(activity, fatigue_profile) for activity in activities]
    prepared = [sample for sample in prepared if not sample.empty]
    if not prepared:
        profile = _default_profile(default_curve, "unavailable")
        profile["coverage"].update(_device_coverage_from_activities(activities))
        profile["calibration"] = _activity_calibration_summary(activities)
        return profile
    data = pd.concat(prepared, ignore_index=True)
    activity_count = int(data["activity"].nunique())
    observed_duration_seconds = float(data["dt_seconds"].sum())
    duration_seconds = float(data["weight_seconds"].sum())
    baseline = data.groupby("grade_band")["fresh_output"].transform("median")
    data = data[baseline > 0].copy()
    data["relative_output"] = (data["fresh_output"] / baseline[baseline > 0]).clip(0.35, 2.5)
    nodes = np.asarray([float(value) for value in config["nodes_c"]], dtype=float)
    midpoints = (nodes[:-1] + nodes[1:]) / 2.0
    data["temperature_node"] = nodes[np.digitize(data["temperature"], midpoints)]

    personal: dict[float, dict[str, object]] = {}
    for node, sample in data.groupby("temperature_node"):
        seconds = float(sample["weight_seconds"].sum())
        count = int(sample["activity"].nunique())
        personal[float(node)] = {
            "performance": _weighted_percentile(sample["relative_output"], sample["weight_seconds"], 50.0),
            "sample_duration_seconds": seconds,
            "activity_count": count,
            "confidence": calculate_confidence(seconds, count, source="personal"),
        }
    comfort = [entry["performance"] for node, entry in personal.items() if 10.0 <= node <= 20.0]
    reference_performance = max(comfort) if comfort else max(entry["performance"] for entry in personal.values())
    sufficient = (
        activity_count >= int(config["minimum_personal_activities"])
        and duration_seconds >= float(config["minimum_personal_duration_minutes"]) * 60.0
        and len(personal) >= 2
    )
    overall_confidence = calculate_confidence(duration_seconds, activity_count, source="personal" if sufficient else "default")
    coverage_factor = min(1.0, len(personal) / 4.0) * min(1.0, activity_count / 3.0)
    overall_confidence = max(0.2, min(0.95, overall_confidence * coverage_factor)) if sufficient else 0.2
    defaults = {float(point["temperature_c"]): float(point["time_factor"]) for point in default_curve}
    comfort_min = float(config.get("comfort_min_c", 10.0))
    comfort_max = float(config.get("comfort_max_c", 20.0))
    curve: list[dict[str, object]] = []
    for node in nodes:
        entry = personal.get(float(node))
        default_factor = defaults[float(node)]
        personal_factor = float(np.clip(reference_performance / entry["performance"], 0.85, 1.35)) if entry else None
        evidence = _node_evidence_strength(entry, config) if entry and sufficient else 0.0
        raw_confidence = float(entry["confidence"]) if entry else 0.2
        effective_confidence = 0.2 + (raw_confidence - 0.2) * evidence
        node_weight = max(0.0, min(1.0, (effective_confidence - 0.2) / 0.75))
        in_comfort_range = comfort_min <= float(node) <= comfort_max
        if in_comfort_range:
            factor = 1.0
            source = "comfort_anchor"
            node_weight = 0.0
        elif personal_factor is not None and node_weight > 0:
            factor = default_factor * (1.0 - node_weight) + personal_factor * node_weight
            source = "personal_blend"
        else:
            factor = default_factor
            source = "node_default" if entry else "default"
        curve.append(
            {
                "temperature_c": float(node),
                "time_factor": round(factor, 4),
                "default_time_factor": round(default_factor, 4),
                "personal_time_factor": round(personal_factor, 4) if personal_factor is not None else None,
                "sample_duration_seconds": round(float(entry["sample_duration_seconds"]), 1) if entry else 0.0,
                "activity_count": int(entry["activity_count"]) if entry else 0,
                "confidence": round(effective_confidence, 3),
                "raw_confidence": round(raw_confidence, 3),
                "personal_weight": round(node_weight, 3),
                "source": source,
            }
        )
    _stabilize_temperature_curve(curve, comfort_min, comfort_max, config)
    has_personal_nodes = any(point.get("source") == "personal_blend" for point in curve)
    return {
        "source": "personal_blend" if has_personal_nodes else "default",
        "confidence": round(overall_confidence, 3),
        "reference_temperature_c": round((comfort_min + comfort_max) / 2.0, 1),
        "best_range_c": [round(comfort_min, 1), round(comfort_max, 1)],
        "coverage": {
            "activity_count": activity_count,
            "valid_duration_seconds": round(duration_seconds, 1),
            "observed_duration_seconds": round(observed_duration_seconds, 1),
            "minimum_c": round(float(data["temperature"].min()), 1),
            "maximum_c": round(float(data["temperature"].max()), 1),
            "device_minimum_c": _optional_round(data.get("device_temperature"), "min"),
            "device_maximum_c": _optional_round(data.get("device_temperature"), "max"),
            "covered_node_count": len(personal),
        },
        "calibration": _calibration_summary(data),
        "curve": curve,
    }


def temperature_time_factor(profile: dict[str, object], temperature_c: float | None) -> float:
    if temperature_c is None:
        return 1.0
    curve = list(profile.get("temperature", {}).get("curve", []))
    if not curve:
        curve = list(load_config()["temperature_model"]["default_curve"])
    ordered = sorted(
        (float(point["temperature_c"]), float(point["time_factor"])) for point in curve
    )
    return float(np.interp(float(temperature_c), [item[0] for item in ordered], [item[1] for item in ordered]))


def temperature_fatigue_time_factor(
    profile: dict[str, object], temperature_c: float | None, elapsed_hours: float
) -> float:
    if temperature_c is None or elapsed_hours <= 0:
        return 1.0
    config = load_config()["temperature_model"]
    reference = float(profile.get("temperature", {}).get("reference_temperature_c", 15.0))
    threshold = reference + float(config["fatigue_threshold_above_reference_c"])
    heat_degrees = max(0.0, float(temperature_c) - threshold)
    factor = 1.0 + heat_degrees * max(0.0, elapsed_hours) * float(config["fatigue_time_increase_per_degree_hour"])
    return min(float(config["maximum_fatigue_time_factor"]), factor)


def heart_rate_heat_fatigue_time_factor(
    profile: dict[str, object], temperature_c: float | None, elapsed_hours: float
) -> float:
    """Return only the personal HR response beyond the generic heat-fatigue effect."""
    if temperature_c is None or elapsed_hours <= 0:
        return 1.0
    heart_rate = profile.get("heart_rate", {})
    heat = heart_rate.get("heat_sensitivity", {})
    if heat.get("source") != "personal":
        return 1.0
    config = load_config()["heart_rate_model"]
    temperature_config = load_config()["temperature_model"]
    reference = float(profile.get("temperature", {}).get("reference_temperature_c", 15.0))
    heat_degrees = max(
        0.0,
        float(temperature_c) - reference - float(temperature_config["fatigue_threshold_above_reference_c"]),
    )
    if heat_degrees <= 0:
        return 1.0
    drift_at_five = interpolate_hr_drift(profile, 5.0)
    drift_excess = max(0.0, drift_at_five - float(config["expected_drift_bpm_at_5h"]))
    sensitivity_excess = max(
        0.0,
        float(heat.get("bpm_per_degree", 0.0)) - float(config["default_heat_sensitivity_bpm_per_degree"]),
    ) * 10.0
    confidence = float(heat.get("confidence", 0.2))
    heat_ratio = min(2.0, heat_degrees / 10.0)
    progress = min(2.0, elapsed_hours / 5.0)
    evidence = drift_excess + sensitivity_excess
    factor = 1.0 + heat_ratio * progress * evidence * float(config["heat_drift_time_factor_scale"]) * confidence
    return min(float(config["maximum_heat_drift_time_factor"]), factor)


def _prepare_samples(activity: pd.DataFrame, fatigue_profile: dict[str, object]) -> pd.DataFrame:
    config = load_config()["temperature_model"]
    required = {"valid_interval", "dt_seconds", "dd_m", "speed_mps", "grade_pct", "delev_m", "temperature"}
    if not required <= set(activity.columns):
        return pd.DataFrame()
    valid = (
        activity["valid_interval"].fillna(False)
        & activity["temperature"].between(float(config["valid_min_c"]), float(config["valid_max_c"]))
        & activity["dt_seconds"].between(0.2, 120.0)
        & (activity["dd_m"] > 0)
    )
    data = activity.loc[valid].copy()
    if data.empty:
        return data
    flat_limit = float(load_config()["terrain"]["flat_grade_abs_percent"])
    data["terrain"] = np.where(
        data["grade_pct"] > flat_limit,
        "uphill",
        np.where(data["grade_pct"] < -flat_limit, "downhill", "flat"),
    )
    data["output"] = data["speed_mps"]
    uphill = data["terrain"] == "uphill"
    data.loc[uphill, "output"] = data.loc[uphill, "delev_m"].clip(lower=0) / data.loc[uphill, "dt_seconds"] * 3600.0
    data = data[np.isfinite(data["output"]) & (data["output"] > 0)].copy()
    if data.empty:
        return data
    start = pd.to_datetime(activity["timestamp"], errors="coerce", utc=True).min()
    data["elapsed_h"] = (
        pd.to_datetime(data["timestamp"], errors="coerce", utc=True) - start
    ).dt.total_seconds().clip(lower=0) / 3600.0
    data["retention"] = [
        interpolate_fatigue(float(hour), fatigue_profile.get(str(terrain), []))
        for hour, terrain in zip(data["elapsed_h"], data["terrain"])
    ]
    data["fresh_output"] = data["output"] / data["retention"].clip(lower=0.4)
    data["activity"] = str(activity.get("_activity_name", pd.Series(["activity"])).iloc[0])
    calibration = activity.attrs.get("temperature_calibration", {"source": "ambient_assumed"})
    data["temperature_calibration_source"] = str(calibration.get("source", "ambient_assumed"))
    if "device_temperature" in data:
        data["device_temperature"] = pd.to_numeric(data["device_temperature"], errors="coerce")
    model_weight = data["_model_weight"] if "_model_weight" in data else pd.Series(1.0, index=data.index)
    temperature_weight = (
        data["temperature_weight"]
        if "temperature_weight" in data
        else pd.Series(1.0, index=data.index)
    )
    data["weight_seconds"] = (
        data["dt_seconds"]
        * pd.to_numeric(model_weight, errors="coerce").fillna(1.0)
        * pd.to_numeric(temperature_weight, errors="coerce").fillna(1.0).clip(0.0, 1.0)
    )
    data["grade_band"] = [_grade_band(terrain, grade) for terrain, grade in zip(data["terrain"], data["grade_pct"])]
    return data


def _node_evidence_strength(entry: dict[str, object], config: dict[str, object]) -> float:
    rules = dict(config.get("node_evidence", {}))
    minimum_activities = int(rules.get("minimum_activities", 2))
    full_activities = max(minimum_activities, int(rules.get("full_activities", 3)))
    minimum_minutes = float(rules.get("minimum_duration_minutes", 60.0))
    full_minutes = max(minimum_minutes, float(rules.get("full_duration_minutes", 180.0)))
    count = int(entry.get("activity_count", 0))
    minutes = float(entry.get("sample_duration_seconds", 0.0)) / 60.0
    if count < minimum_activities or minutes < minimum_minutes:
        return 0.0
    count_span = max(1, full_activities - minimum_activities + 1)
    count_strength = min(1.0, (count - minimum_activities + 1) / count_span)
    duration_strength = min(1.0, minutes / full_minutes)
    return max(0.0, min(count_strength, duration_strength))


def _stabilize_temperature_curve(
    curve: list[dict[str, object]],
    comfort_min: float,
    comfort_max: float,
    config: dict[str, object],
) -> None:
    """Apply local bounds without allowing one node to overwrite a whole side."""
    maximum_adjustment = float(config.get("maximum_personal_factor_adjustment", 0.05))
    for point in curve:
        temperature = float(point["temperature_c"])
        if comfort_min <= temperature <= comfort_max:
            point["time_factor"] = 1.0
            continue
        if point.get("source") != "personal_blend":
            continue
        default_factor = float(point.get("default_time_factor", point["time_factor"]))
        point["time_factor"] = round(
            float(np.clip(float(point["time_factor"]), 1.0, default_factor + maximum_adjustment)),
            4,
        )

    temperatures = [float(point["temperature_c"]) for point in curve]
    cold_indices = [index for index, value in enumerate(temperatures) if value < comfort_min]
    hot_indices = [index for index, value in enumerate(temperatures) if value > comfort_max]
    _stabilize_side(curve, reversed(cold_indices))
    _stabilize_side(curve, hot_indices)


def _stabilize_side(curve: list[dict[str, object]], indices: Iterable[int]) -> None:
    ordered = list(indices)
    previous = 1.0
    for position, index in enumerate(ordered):
        point = curve[int(index)]
        if point.get("source") == "personal_blend":
            outer_fixed = next(
                (
                    float(curve[int(outer_index)]["time_factor"])
                    for outer_index in ordered[position + 1:]
                    if curve[int(outer_index)].get("source") != "personal_blend"
                ),
                None,
            )
            factor = float(point["time_factor"])
            if outer_fixed is not None:
                factor = min(factor, outer_fixed)
            point["time_factor"] = round(max(previous, factor), 4)
        previous = float(point["time_factor"])


def _grade_band(terrain: str, grade: float) -> str:
    if terrain == "flat":
        return "flat"
    magnitude = abs(float(grade))
    label = "2_5" if magnitude < 5 else "5_10" if magnitude < 10 else "10_15" if magnitude < 15 else "15_plus"
    return f"{terrain}_{label}"


def _weighted_percentile(values: pd.Series, weights: pd.Series, percentile: float) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    numeric = values[mask].to_numpy(dtype=float)
    numeric_weights = weights[mask].to_numpy(dtype=float)
    if len(numeric) == 0:
        return float("nan")
    order = np.argsort(numeric)
    target = np.clip(percentile / 100.0, 0.0, 1.0) * numeric_weights.sum()
    index = min(int(np.searchsorted(np.cumsum(numeric_weights[order]), target, side="left")), len(numeric) - 1)
    return float(numeric[order][index])


def _default_profile(default_curve: list[dict[str, object]], source: str) -> dict[str, object]:
    return {
        "source": source,
        "confidence": 0.2,
        "reference_temperature_c": 15.0,
        "best_range_c": [10.0, 20.0],
        "coverage": {"activity_count": 0, "valid_duration_seconds": 0.0,
                     "observed_duration_seconds": 0.0, "minimum_c": None, "maximum_c": None,
                     "device_minimum_c": None, "device_maximum_c": None, "covered_node_count": 0},
        "calibration": {"source": "unavailable", "absolute_temperature_available": False,
                        "model_weight": 0.0},
        "curve": [
            {**point, "default_time_factor": point["time_factor"], "personal_time_factor": None,
             "sample_duration_seconds": 0.0, "activity_count": 0, "confidence": 0.2,
             "raw_confidence": 0.2, "personal_weight": 0.0, "source": "default"}
            for point in default_curve
        ],
    }


def _optional_round(series: pd.Series | None, operation: str) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    value = values.min() if operation == "min" else values.max()
    return round(float(value), 1)


def _calibration_summary(data: pd.DataFrame) -> dict[str, object]:
    sources = set(data.get("temperature_calibration_source", pd.Series(dtype=str)).dropna().astype(str))
    if "historical_weather" in sources:
        config = load_config()["historical_weather"]
        return {
            "source": "historical_weather",
            "provider": "Open-Meteo Historical Weather API",
            "absolute_temperature_available": True,
            "model_weight": float(config["model_weight"]),
            "local_exposure_weight_adjustment": True,
            "spatial_resolution_note": "约9–11km再分析网格，山区微气候可能存在偏差",
        }
    if "wrist_relative_only" not in sources:
        return {"source": "ambient_assumed", "absolute_temperature_available": True,
                "model_weight": 1.0}
    return {
        "source": "wrist_relative_only",
        "absolute_temperature_available": False,
        "model_weight": 0.0,
    }


def _device_coverage_from_activities(activities: list[pd.DataFrame]) -> dict[str, object]:
    values = [
        pd.to_numeric(activity["device_temperature"], errors="coerce")
        for activity in activities if "device_temperature" in activity
    ]
    device = pd.concat(values, ignore_index=True).dropna() if values else pd.Series(dtype=float)
    return {
        "device_minimum_c": None if device.empty else round(float(device.min()), 1),
        "device_maximum_c": None if device.empty else round(float(device.max()), 1),
        "device_activity_count": sum(
            1 for activity in activities
            if "device_temperature" in activity and pd.to_numeric(activity["device_temperature"], errors="coerce").notna().any()
        ),
    }


def _activity_calibration_summary(activities: list[pd.DataFrame]) -> dict[str, object]:
    calibrations = [activity.attrs.get("temperature_calibration", {}) for activity in activities]
    historical = [item for item in calibrations if item.get("source") == "historical_weather"]
    if historical:
        config = load_config()["historical_weather"]
        return {
            "source": "historical_weather",
            "provider": "Open-Meteo Historical Weather API",
            "absolute_temperature_available": True,
            "model_weight": float(config["model_weight"]),
            "local_exposure_weight_adjustment": True,
            "spatial_resolution_note": "约9–11km再分析网格，山区微气候可能存在偏差",
        }
    wrist = [item for item in calibrations if item.get("source") == "wrist_relative_only"]
    if not wrist:
        return {"source": "unavailable", "absolute_temperature_available": False, "model_weight": 0.0}
    return {
        "source": "wrist_relative_only",
        "warmup_minutes": wrist[0].get("warmup_minutes"),
        "absolute_temperature_available": False,
        "model_weight": 0.0,
    }
