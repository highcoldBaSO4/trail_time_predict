from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.confidence import calculate_confidence
from config import load_config


TERRAINS = ("flat", "uphill", "downhill")
INTENSITY_LABELS = {
    "easy": "轻松",
    "aerobic": "有氧",
    "steady": "稳态",
    "threshold": "阈值",
    "high": "高强度",
}


def build_heart_rate_profile(
    activities: list[pd.DataFrame], natural_segments: pd.DataFrame | None = None
) -> dict[str, object]:
    """Build explanatory terrain HR response and cardiac-drift curves.

    Heart rate does not redefine the base capability model. Stable historical
    HR/output windows provide a guarded race-intensity layer on top of it.
    """
    samples = [_prepare_samples(activity) for activity in activities]
    samples = [sample for sample in samples if not sample.empty]
    if not samples:
        return _default_profile()
    combined = pd.concat(samples, ignore_index=True)
    duration_seconds = float(combined["dt_seconds"].sum())
    activity_count = int(combined["activity"].nunique())
    confidence = calculate_confidence(duration_seconds, activity_count, source="personal")

    stable = _stable_windows(combined)
    threshold = _weighted_percentile(
        stable["heart_rate"], stable["weight_seconds"], float(load_config()["heart_rate_model"]["threshold_percentile"])
    ) if not stable.empty else _weighted_percentile(combined["heart_rate"], combined["weight_seconds"], 90.0)
    zones = _intensity_zones(threshold)
    aerobic_low = zones["aerobic"][0] * threshold
    aerobic_high = zones["aerobic"][1] * threshold
    aerobic_average = (aerobic_low + aerobic_high) / 2.0
    natural = _prepare_natural_segment_samples(natural_segments)
    response_samples = natural if not natural.empty else combined
    intensity_samples = natural if not natural.empty else stable
    intensity_output = _intensity_output_profile(intensity_samples, threshold)
    terrain_response = {
        "flat": _response_entries(response_samples[response_samples["terrain"] == "flat"], "flat"),
        "uphill": _response_entries(response_samples[response_samples["terrain"] == "uphill"], "uphill"),
        "downhill": _response_entries(response_samples[response_samples["terrain"] == "downhill"], "downhill"),
    }
    drift = {
        "overall": _drift_curve(combined),
        "terrain": {
            terrain: _drift_curve(combined[combined["terrain"] == terrain])
            for terrain in TERRAINS
        },
    }
    return {
        "source": "personal",
        "confidence": round(float(confidence), 3),
        "coverage": {
            "activity_count": activity_count,
            "valid_duration_seconds": round(duration_seconds, 1),
            "record_count": len(combined),
        },
        "aerobic_range": {
            "low_bpm": round(aerobic_low, 1),
            "high_bpm": round(aerobic_high, 1),
            "average_bpm": round(aerobic_average, 1),
            "source": "observed_stable_output",
            "confidence": round(float(confidence), 3),
        },
        "threshold": {
            "bpm": round(threshold, 1),
            "source": "estimated_upper_output",
            "confidence": round(float(confidence) * 0.8, 3),
        },
        "intensity_zones": [
            {
                "name": name,
                "label": INTENSITY_LABELS[name],
                "minimum_ratio": round(low, 3),
                "maximum_ratio": round(high, 3),
                "minimum_bpm": round(low * threshold, 1),
                "maximum_bpm": round(high * threshold, 1),
            }
            for name, (low, high) in zones.items()
        ],
        "intensity_output": intensity_output,
        "terrain_response": terrain_response,
        "drift": drift,
        "heat_sensitivity": _heat_sensitivity(combined),
    }


def heart_rate_pacing_adjustment(
    profile: dict[str, object],
    terrain: str,
    grade: float,
    estimated_hours: float,
    strategy: str,
    base_output: float,
    elapsed_hours: float = 0.0,
) -> dict[str, object]:
    """Select a guarded terrain output from historical HR/intensity samples."""
    heart_rate = profile.get("heart_rate", {})
    strategy = strategy if strategy in {"conservative", "standard", "aggressive"} else "standard"
    threshold = heart_rate.get("threshold", {}).get("bpm")
    entries = list(heart_rate.get("intensity_output", {}).get(terrain, []))
    if threshold is None or not entries or base_output <= 0:
        return _unavailable_pacing(strategy, estimated_hours)
    config = load_config()["heart_rate_model"]
    target_ratio = _target_ratio(estimated_hours, strategy, config)
    band = _grade_band(terrain, grade)
    candidates = [entry for entry in entries if entry.get("grade_band") == band]
    if not candidates:
        return _unavailable_pacing(strategy, estimated_hours, target_ratio, float(threshold))
    candidates.sort(key=lambda entry: float(entry["heart_rate_ratio"]))
    ratios = np.asarray([float(entry["heart_rate_ratio"]) for entry in candidates], dtype=float)
    output_field = "fast_output" if strategy == "aggressive" else "median_output"
    outputs = np.asarray([float(entry[output_field]) for entry in candidates], dtype=float)
    confidences = np.asarray([float(entry.get("confidence", 0.2)) for entry in candidates], dtype=float)
    target_output = float(np.interp(target_ratio, ratios, outputs))
    confidence = float(np.interp(target_ratio, ratios, confidences))
    blend = max(0.0, min(1.0, (confidence - 0.2) / 0.75))
    selected_output = base_output * (1.0 - blend) + target_output * blend
    limits = config["strategy_output_limits"][strategy]
    maximum = float(limits["downhill_max"] if terrain == "downhill" else limits["max"])
    output_factor = max(float(limits["min"]), min(maximum, selected_output / base_output))
    drift = interpolate_hr_drift(profile, elapsed_hours)
    target_bpm = float(threshold) * target_ratio
    expected_bpm = min(float(threshold) * 1.03, target_bpm + max(0.0, drift))
    return {
        "strategy": strategy,
        "intensity": _intensity_name(target_ratio),
        "intensity_label": INTENSITY_LABELS[_intensity_name(target_ratio)],
        "target_hr_ratio": round(target_ratio, 3),
        "target_hr_bpm": round(target_bpm, 1),
        "expected_hr_bpm": round(expected_bpm, 1),
        "output_factor": round(output_factor, 4),
        "time_factor": round(1.0 / output_factor, 4),
        "selected_output": round(base_output * output_factor, 2),
        "output_unit": "vertical_metres_per_hour" if terrain == "uphill" else "metres_per_second",
        "grade_band": band,
        "confidence": round(confidence, 3),
        "source": "personal",
        "extrapolated": bool(target_ratio < ratios.min() or target_ratio > ratios.max()),
    }


def interpolate_hr_drift(profile: dict[str, object], elapsed_hours: float) -> float:
    points = list(profile.get("heart_rate", {}).get("drift", {}).get("overall", []))
    if not points:
        return 0.0
    ordered = sorted((float(point["hour"]), float(point["drift_bpm"])) for point in points)
    return float(np.interp(max(0.0, elapsed_hours), [item[0] for item in ordered], [item[1] for item in ordered]))


def _prepare_samples(activity: pd.DataFrame) -> pd.DataFrame:
    config = load_config()["heart_rate_model"]
    required = {"valid_interval", "dt_seconds", "dd_m", "speed_mps", "grade_pct", "delev_m", "heart_rate"}
    if not required <= set(activity.columns):
        return pd.DataFrame()
    valid = (
        activity["valid_interval"].fillna(False)
        & activity["heart_rate"].between(float(config["valid_min_bpm"]), float(config["valid_max_bpm"]))
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
    data["vertical_speed_mph"] = np.where(
        data["terrain"] == "uphill",
        data["output"],
        np.where(
            data["terrain"] == "downhill",
            data["delev_m"] / data["dt_seconds"] * 3600.0,
            0.0,
        ),
    )
    data = data[np.isfinite(data["output"]) & (data["output"] > 0)]
    data = data[
        ((data["terrain"] == "uphill") & data["output"].between(30.0, 2500.0))
        | ((data["terrain"] != "uphill") & data["output"].between(0.2, 12.0))
    ].copy()
    if data.empty:
        return data
    start = pd.to_datetime(activity["timestamp"], errors="coerce", utc=True).min()
    data["elapsed_h"] = (
        pd.to_datetime(data["timestamp"], errors="coerce", utc=True) - start
    ).dt.total_seconds().clip(lower=0) / 3600.0
    data["activity"] = str(activity.get("_activity_name", pd.Series(["activity"])).iloc[0])
    model_weight = data["_model_weight"] if "_model_weight" in data else pd.Series(1.0, index=data.index)
    data["model_weight"] = pd.to_numeric(model_weight, errors="coerce").fillna(1.0)
    data["weight_seconds"] = data["dt_seconds"] * data["model_weight"]
    temperature_weight = (
        pd.to_numeric(data["temperature_weight"], errors="coerce").fillna(1.0)
        if "temperature_weight" in data
        else pd.Series(1.0, index=data.index)
    )
    data["temperature_weight_seconds"] = data["weight_seconds"] * temperature_weight.clip(0.0, 1.0)
    data["grade_band"] = [_grade_band(terrain, grade) for terrain, grade in zip(data["terrain"], data["grade_pct"])]
    data["temperature"] = (
        pd.to_numeric(data["temperature"], errors="coerce") if "temperature" in data else np.nan
    )
    return data


def _stable_windows(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate records into stable two-minute terrain windows."""
    if frame.empty:
        return pd.DataFrame()
    config = load_config()["heart_rate_model"]
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data = data.dropna(subset=["timestamp"])
    data["window"] = data["timestamp"].dt.floor(f"{int(config['stable_window_seconds'])}s")
    rows: list[dict[str, object]] = []
    for (activity, terrain, grade_band, window), sample in data.groupby(
        ["activity", "terrain", "grade_band", "window"], sort=False
    ):
        duration = float(sample["dt_seconds"].sum())
        if duration < float(config["minimum_stable_window_seconds"]):
            continue
        output_mean = _weighted_average(sample["output"], sample["weight_seconds"])
        output_std = float(sample["output"].std(ddof=0))
        if not np.isfinite(output_mean) or output_mean <= 0:
            continue
        if output_std / output_mean > float(config["maximum_output_cv"]):
            continue
        rows.append(
            {
                "activity": activity,
                "terrain": terrain,
                "grade_band": grade_band,
                "window": window,
                "heart_rate": _weighted_average(sample["heart_rate"], sample["weight_seconds"]),
                "output": output_mean,
                "speed_mps": _weighted_average(sample["speed_mps"], sample["weight_seconds"]),
                "vertical_speed_mph": (
                    output_mean if terrain == "uphill"
                    else -abs(_weighted_average(sample["delev_m"], sample["weight_seconds"]) /
                              max(_weighted_average(sample["dt_seconds"], sample["weight_seconds"]), 1e-6) * 3600.0)
                    if terrain == "downhill" else 0.0
                ),
                "grade_pct": _weighted_average(sample["grade_pct"], sample["weight_seconds"]),
                "weight_seconds": float(sample["weight_seconds"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _intensity_zones(threshold: float) -> dict[str, tuple[float, float]]:
    del threshold
    values = load_config()["heart_rate_model"]["intensity_zones"]
    return {
        str(name): (float(bounds["min_ratio"]), float(bounds["max_ratio"]))
        for name, bounds in values.items()
    }


def _intensity_output_profile(frame: pd.DataFrame, threshold: float) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {terrain: [] for terrain in TERRAINS}
    if frame.empty or not np.isfinite(threshold) or threshold <= 0:
        return result
    data = frame.copy()
    data["heart_rate_ratio"] = data["heart_rate"] / threshold
    data["intensity"] = [_intensity_name(float(value)) for value in data["heart_rate_ratio"]]
    for (terrain, band, intensity), sample in data.groupby(["terrain", "grade_band", "intensity"], sort=True):
        duration = float(sample["weight_seconds"].sum())
        activity_count = int(sample["activity"].nunique())
        median_speed = _weighted_percentile(sample["speed_mps"], sample["weight_seconds"], 50.0)
        fast_speed = _weighted_percentile(sample["speed_mps"], sample["weight_seconds"], 70.0)
        median_vertical = _weighted_percentile(sample["vertical_speed_mph"], sample["weight_seconds"], 50.0)
        fast_vertical = _weighted_percentile(sample["vertical_speed_mph"], sample["weight_seconds"], 70.0)
        result[str(terrain)].append(
            {
                "grade_band": str(band),
                "intensity": str(intensity),
                "intensity_label": INTENSITY_LABELS[str(intensity)],
                "heart_rate_ratio": round(_weighted_average(sample["heart_rate_ratio"], sample["weight_seconds"]), 3),
                "average_hr_bpm": round(_weighted_average(sample["heart_rate"], sample["weight_seconds"]), 1),
                "median_output": round(_weighted_percentile(sample["output"], sample["weight_seconds"], 50.0), 2),
                "fast_output": round(_weighted_percentile(sample["output"], sample["weight_seconds"], 70.0), 2),
                "median_speed_mps": round(median_speed, 3),
                "fast_speed_mps": round(fast_speed, 3),
                "median_vertical_speed_mph": round(median_vertical, 1),
                "fast_vertical_speed_mph": round(fast_vertical, 1),
                "output_unit": "vertical_metres_per_hour" if terrain == "uphill" else "metres_per_second",
                "sample_duration_seconds": round(duration, 1),
                "activity_count": activity_count,
                "confidence": round(calculate_confidence(duration, activity_count, source="personal"), 3),
            }
        )
    return result


def _target_ratio(estimated_hours: float, strategy: str, config: dict[str, object]) -> float:
    strategy = strategy if strategy in {"conservative", "standard", "aggressive"} else "standard"
    for row in config["race_intensity_targets"]:
        maximum = row.get("max_hours")
        if maximum is None or estimated_hours <= float(maximum):
            return float(row[strategy])
    return float(config["race_intensity_targets"][-1][strategy])


def _intensity_name(ratio: float) -> str:
    zones = _intensity_zones(1.0)
    for name, (low, high) in zones.items():
        if low <= ratio < high or (name == "high" and ratio >= low):
            return name
    return "easy"


def _unavailable_pacing(
    strategy: str, estimated_hours: float, target_ratio: float | None = None, threshold: float | None = None
) -> dict[str, object]:
    ratio = target_ratio if target_ratio is not None else _target_ratio(
        estimated_hours, strategy, load_config()["heart_rate_model"]
    )
    intensity = _intensity_name(ratio)
    return {
        "strategy": strategy,
        "intensity": intensity,
        "intensity_label": INTENSITY_LABELS[intensity],
        "target_hr_ratio": round(ratio, 3),
        "target_hr_bpm": None if threshold is None else round(threshold * ratio, 1),
        "expected_hr_bpm": None,
        "output_factor": 1.0,
        "time_factor": 1.0,
        "selected_output": None,
        "output_unit": None,
        "grade_band": None,
        "confidence": 0.2,
        "source": "unavailable",
        "extrapolated": False,
    }


def _response_entries(frame: pd.DataFrame, terrain: str) -> list[dict[str, object]]:
    if frame.empty:
        return []
    result: list[dict[str, object]] = []
    for band, sample in frame.groupby("grade_band", sort=True):
        duration = float(sample["weight_seconds"].sum())
        activity_count = int(sample["activity"].nunique())
        result.append(
            {
                "grade_band": str(band),
                "average_grade": round(_weighted_average(sample["grade_pct"], sample["weight_seconds"]), 2),
                "average_hr_bpm": round(_weighted_average(sample["heart_rate"], sample["weight_seconds"]), 1),
                "median_output": round(_weighted_percentile(sample["output"], sample["weight_seconds"], 50.0), 2),
                "median_speed_mps": round(_weighted_percentile(sample["speed_mps"], sample["weight_seconds"], 50.0), 3),
                "vertical_speed_mph": round(_weighted_percentile(sample["vertical_speed_mph"], sample["weight_seconds"], 50.0), 1),
                "output_unit": "vertical_metres_per_hour" if terrain == "uphill" else "metres_per_second",
                "sample_duration_seconds": round(duration, 1),
                "activity_count": activity_count,
                "confidence": round(calculate_confidence(duration, activity_count, source="personal"), 3),
            }
        )
    return result


def _prepare_natural_segment_samples(segments: pd.DataFrame | None) -> pd.DataFrame:
    required = {
        "activity", "type", "distance_m", "duration_s", "gain_m", "loss_m", "grade_pct",
        "average_hr_bpm", "heart_rate_duration_s", "model_weight",
    }
    if segments is None or segments.empty or not required <= set(segments.columns):
        return pd.DataFrame()
    config = load_config()["heart_rate_model"]
    data = segments.copy()
    data["heart_rate"] = pd.to_numeric(data["average_hr_bpm"], errors="coerce")
    data["terrain"] = data["type"].astype(str)
    data["speed_mps"] = pd.to_numeric(data["distance_m"], errors="coerce") / pd.to_numeric(
        data["duration_s"], errors="coerce"
    ).replace(0, np.nan)
    data["vertical_speed_mph"] = np.where(
        data["terrain"] == "uphill",
        pd.to_numeric(data["gain_m"], errors="coerce") / data["duration_s"] * 3600.0,
        np.where(
            data["terrain"] == "downhill",
            -pd.to_numeric(data["loss_m"], errors="coerce") / data["duration_s"] * 3600.0,
            0.0,
        ),
    )
    data["output"] = np.where(
        data["terrain"] == "uphill", data["vertical_speed_mph"], data["speed_mps"]
    )
    data["weight_seconds"] = (
        pd.to_numeric(data["heart_rate_duration_s"], errors="coerce").fillna(0.0)
        * pd.to_numeric(data["model_weight"], errors="coerce").fillna(1.0)
    )
    data["dt_seconds"] = pd.to_numeric(data["duration_s"], errors="coerce")
    data["grade_band"] = [
        _grade_band(str(terrain), float(grade))
        for terrain, grade in zip(data["terrain"], data["grade_pct"])
    ]
    valid = (
        data["terrain"].isin(TERRAINS)
        & data["heart_rate"].between(float(config["valid_min_bpm"]), float(config["valid_max_bpm"]))
        & (data["weight_seconds"] > 0)
        & np.isfinite(data["output"])
        & (data["output"] > 0)
    )
    return data.loc[valid].copy()


def _drift_curve(frame: pd.DataFrame) -> list[dict[str, object]]:
    config = load_config()["heart_rate_model"]
    hours = [float(value) for value in config["drift_hours"]]
    result: list[dict[str, object]] = [
        {"hour": 0.0, "drift_bpm": 0.0, "output_retention": 1.0, "sample_count": 0,
         "source": "anchor", "confidence": None}
    ]
    if frame.empty or "activity" not in frame:
        result.extend(
            {"hour": hour, "drift_bpm": 0.0, "output_retention": 1.0,
             "sample_count": 0, "source": "default", "confidence": 0.2}
            for hour in hours[1:]
        )
        return result
    minimum_hours = float(config["minimum_activity_hours_for_drift"])
    for hour in hours[1:]:
        drift_values: list[float] = []
        retention_values: list[float] = []
        for _, activity in frame.groupby("activity"):
            if activity.empty or float(activity["elapsed_h"].max()) < min(hour * 0.75, minimum_hours):
                continue
            normalized = _normalize_output_by_grade(activity)
            early = normalized[normalized["elapsed_h"] <= min(1.5, max(0.75, hour * 0.3))]
            low = 1.5 if hour == 3.0 else 3.0 if hour == 5.0 else 5.0
            late = normalized[(normalized["elapsed_h"] > low) & (normalized["elapsed_h"] <= hour)]
            if float(early["dt_seconds"].sum()) < 300 or float(late["dt_seconds"].sum()) < 300:
                continue
            early_cost = _weighted_percentile(early["heart_rate"] / early["normalized_output"], early["weight_seconds"], 50.0)
            late_cost = _weighted_percentile(late["heart_rate"] / late["normalized_output"], late["weight_seconds"], 50.0)
            early_efficiency = _weighted_percentile(early["normalized_output"] / early["heart_rate"], early["weight_seconds"], 50.0)
            late_efficiency = _weighted_percentile(late["normalized_output"] / late["heart_rate"], late["weight_seconds"], 50.0)
            if np.isfinite(early_cost) and np.isfinite(late_cost):
                drift_values.append(float(np.clip(late_cost - early_cost, -10.0, 40.0)))
            if early_efficiency > 0 and np.isfinite(late_efficiency):
                retention_values.append(float(np.clip(late_efficiency / early_efficiency, 0.5, 1.1)))
        source = "personal" if drift_values else "default"
        result.append(
            {
                "hour": hour,
                "drift_bpm": round(float(np.median(drift_values)), 1) if drift_values else 0.0,
                "output_retention": round(float(np.median(retention_values)), 3) if retention_values else 1.0,
                "sample_count": len(drift_values),
                "source": source,
                "confidence": round(calculate_confidence(len(drift_values) * 1800.0, len(drift_values), source=source), 3),
            }
        )
    return result


def _normalize_output_by_grade(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    baselines = data[data["elapsed_h"] <= 1.5].groupby("grade_band")["output"].median()
    data["baseline_output"] = data["grade_band"].map(baselines)
    data = data[data["baseline_output"].notna() & (data["baseline_output"] > 0)].copy()
    data["normalized_output"] = (data["output"] / data["baseline_output"]).clip(0.25, 2.0)
    return data


def _heat_sensitivity(frame: pd.DataFrame) -> dict[str, object]:
    config = load_config()["heart_rate_model"]
    valid = frame[frame["temperature"].between(-30.0, 60.0)].copy()
    activity_count = int(valid["activity"].nunique()) if not valid.empty else 0
    if valid.empty or activity_count < 2 or float(valid["temperature"].max() - valid["temperature"].min()) < 5.0:
        return {
            "bpm_per_degree": float(config["default_heat_sensitivity_bpm_per_degree"]),
            "source": "default",
            "confidence": 0.2,
            "activity_count": activity_count,
        }
    group = valid.groupby(["activity", "grade_band"])
    valid["hr_residual"] = valid["heart_rate"] - group["heart_rate"].transform("median")
    valid["temperature_residual"] = valid["temperature"] - group["temperature"].transform("median")
    heat_weights = valid.get("temperature_weight_seconds", valid["weight_seconds"])
    variance = float(np.average(valid["temperature_residual"] ** 2, weights=heat_weights))
    covariance = float(np.average(valid["temperature_residual"] * valid["hr_residual"], weights=heat_weights))
    slope = covariance / variance if variance > 1e-9 else float(config["default_heat_sensitivity_bpm_per_degree"])
    slope = float(np.clip(slope, 0.0, float(config["maximum_heat_sensitivity_bpm_per_degree"])))
    confidence = calculate_confidence(float(heat_weights.sum()), activity_count, source="personal")
    return {
        "bpm_per_degree": round(slope, 3),
        "source": "personal",
        "confidence": round(confidence, 3),
        "activity_count": activity_count,
    }


def _grade_band(terrain: str, grade: float) -> str:
    if terrain == "flat":
        return "flat"
    magnitude = abs(float(grade))
    label = (
        "2_5" if magnitude < 5 else "5_10" if magnitude < 10 else
        "10_15" if magnitude < 15 else "15_20" if magnitude < 20 else "20_plus"
    )
    return f"{terrain}_{label}"


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    return float(np.average(values[mask], weights=weights[mask])) if mask.any() else float("nan")


def _weighted_percentile(values: pd.Series, weights: pd.Series, percentile: float) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    numeric = values[mask].to_numpy(dtype=float)
    numeric_weights = weights[mask].to_numpy(dtype=float)
    order = np.argsort(numeric)
    target = np.clip(percentile / 100.0, 0.0, 1.0) * numeric_weights.sum()
    index = min(int(np.searchsorted(np.cumsum(numeric_weights[order]), target, side="left")), len(numeric) - 1)
    return float(numeric[order][index])


def _default_profile() -> dict[str, object]:
    return {
        "source": "unavailable",
        "confidence": 0.2,
        "coverage": {"activity_count": 0, "valid_duration_seconds": 0.0, "record_count": 0},
        "aerobic_range": {"low_bpm": None, "high_bpm": None, "average_bpm": None,
                          "source": "unavailable", "confidence": 0.2},
        "threshold": {"bpm": None, "source": "unavailable", "confidence": 0.2},
        "intensity_zones": [],
        "intensity_output": {terrain: [] for terrain in TERRAINS},
        "terrain_response": {terrain: [] for terrain in TERRAINS},
        "drift": {"overall": _drift_curve(pd.DataFrame()),
                  "terrain": {terrain: _drift_curve(pd.DataFrame()) for terrain in TERRAINS}},
        "heat_sensitivity": {
            "bpm_per_degree": float(load_config()["heart_rate_model"]["default_heat_sensitivity_bpm_per_degree"]),
            "source": "default", "confidence": 0.2, "activity_count": 0,
        },
    }
