from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.activity_analysis import add_interval_metrics, analyze_activity
from analysis.confidence import aggregate_quality_score, calculate_confidence
from analysis.data_quality import diagnose_fit
from analysis.fatigue import build_fatigue_profile
from config import load_config
from parser.gpx_reader import group_terrain_chunks


_CONFIG_DEFAULTS = load_config()["default_profile"]
DEFAULT_PROFILE = {
    "flat": _CONFIG_DEFAULTS["flat"],
    "uphill": dict(zip(("1_percent", "5_percent", "10_percent", "15_percent"), (p["value"] for p in _CONFIG_DEFAULTS["uphill"]))),
    "downhill": dict(zip(("-1_percent", "-5_percent", "-10_percent", "-15_percent"),
                            ({"speed_mps": p["speed_mps"], "vertical_speed_mph": p["vertical_speed_mph"]} for p in _CONFIG_DEFAULTS["downhill"]))),
}


def build_runner_profile(activities: dict[str, pd.DataFrame]) -> dict[str, object]:
    if not activities:
        raise ValueError("至少需要一个 FIT 活动才能生成能力画像")

    enriched: list[pd.DataFrame] = []
    terrain_segments: list[dict[str, float | str]] = []
    activity_types: dict[str, str] = {}
    quality_reports: dict[str, dict[str, object]] = {}
    for name, frame in activities.items():
        activity_type = _activity_type(name, frame)
        activity_types[name] = activity_type
        quality_reports[name] = diagnose_fit(frame)
        activity = add_interval_metrics(frame)
        activity["_activity_name"] = name
        activity["_activity_type"] = activity_type
        activity["_model_weight"] = _activity_weight(name, activity)
        enriched.append(activity)
        terrain_segments.extend(_activity_terrain_segments(activity, name, activity_type))
    segment_frame = pd.DataFrame(terrain_segments)
    quality_score = aggregate_quality_score(list(quality_reports.values()))
    flat = _build_flat_profile(segment_frame, quality_score)
    uphill = {**DEFAULT_PROFILE["uphill"], **_build_uphill_profile(segment_frame)}
    downhill = {**DEFAULT_PROFILE["downhill"], **_build_downhill_profile(segment_frame)}
    uphill["curve"] = _uphill_curve(uphill, quality_score)
    downhill["curve"] = _downhill_curve(downhill, quality_score)
    activity_summaries = []
    for name, frame in activities.items():
        summary = analyze_activity(frame, name)
        summary["activity_type"] = activity_types[name]
        summary["data_quality"] = quality_reports[name]
        activity_summaries.append(summary)
    profile: dict[str, object] = {
        "schema_version": "0.2-phase1",
        "units": {
            "flat_pace": "seconds_per_km",
            "uphill": "vertical_metres_per_hour",
            "downhill_speed": "metres_per_second",
            "fatigue": "retained_performance_ratio",
        },
        "flat": flat,
        "uphill": uphill,
        "downhill": downhill,
        "fatigue": build_fatigue_profile(enriched),
        "activities": activity_summaries,
        "terrain_segments": {
            "total": len(terrain_segments),
            "uphill": sum(item["type"] == "uphill" for item in terrain_segments),
            "downhill": sum(item["type"] == "downhill" for item in terrain_segments),
            "flat": sum(item["type"] == "flat" for item in terrain_segments),
        },
        "sample_count": len(activities),
        "data_quality": {
            "score": round(quality_score, 3),
            "activities": quality_reports,
            "recommended_count": sum(bool(report["recommended_for_model"]) for report in quality_reports.values()),
        },
    }
    return profile


def save_runner_profile(profile: dict[str, object], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_flat_profile(segments: pd.DataFrame, quality_score: float = 1.0) -> dict[str, object]:
    """Blend trail flats with discounted road flats; heart rate is not used."""
    if segments.empty:
        flat_segments = segments
    else:
        flat_segments = segments[
            (segments["type"] == "flat")
            & (segments["distance_m"] >= 200.0)
            & segments["pace"].between(150.0, 1800.0)
        ]
    if flat_segments.empty:
        return {
            "aerobic_pace": DEFAULT_PROFILE["flat"]["aerobic_pace"],
            "threshold_pace": DEFAULT_PROFILE["flat"]["threshold_pace"],
            "method": "natural_terrain_flat_weighted",
            "source": "default_no_qualified_segment",
            "qualified_segments": 0,
            "sample_distance_km": 0.0,
            "confidence": calculate_confidence(0, 0, source="default"),
        }

    trail = flat_segments[flat_segments["activity_type"] == "trail"]
    road = flat_segments[flat_segments["activity_type"] == "road"]
    road_penalty = 1.10
    trail_weight, road_weight = 0.70, 0.30
    trail_pace = _weighted_percentile(trail, "pace", 50)
    trail_fast = _weighted_percentile(trail, "pace", 30)
    road_pace = _weighted_percentile(road, "pace", 50)
    road_fast = _weighted_percentile(road, "pace", 30)
    if np.isfinite(trail_pace) and np.isfinite(road_pace):
        baseline = trail_weight * trail_pace + road_weight * road_pace * road_penalty
        faster = trail_weight * trail_fast + road_weight * road_fast * road_penalty
        source = "trail_70pct_plus_discounted_road_30pct"
    elif np.isfinite(trail_pace):
        baseline, faster, source = trail_pace, trail_fast, "trail_only"
    else:
        baseline, faster, source = road_pace * road_penalty, road_fast * road_penalty, "discounted_road_only"
    duration_seconds = float(flat_segments["duration_s"].sum())
    variability = float(flat_segments["pace"].std() / flat_segments["pace"].mean()) if len(flat_segments) > 1 else None
    return {
        "aerobic_pace": round(float(baseline), 1),
        "threshold_pace": round(float(faster), 1),
        "method": "natural_terrain_flat_weighted",
        "source": source,
        "qualified_segments": len(flat_segments),
        "sample_distance_km": round(float(flat_segments["distance_m"].sum()) / 1000.0, 3),
        "trail": _flat_source_summary(trail),
        "road": _flat_source_summary(road),
        "weights": {"trail": trail_weight, "road": road_weight},
        "road_to_trail_pace_factor": road_penalty,
        "sample_duration_seconds": round(duration_seconds, 1),
        "confidence": calculate_confidence(duration_seconds, len(flat_segments), quality_score, variability),
    }


def _activity_terrain_segments(
    activity: pd.DataFrame, activity_name: str, activity_type: str, sample_distance_m: float = 100.0
) -> list[dict[str, float | str]]:
    valid = (
        activity["valid_interval"].fillna(False)
        & (activity["dd_m"] > 0)
        & activity["delev_m"].notna()
    )
    block_ids = (~valid).cumsum()
    segments: list[dict[str, float | str]] = []
    for _, block in activity.loc[valid].groupby(block_ids[valid], sort=False):
        distance = block["dd_m"].to_numpy(dtype=float)
        seconds = block["dt_seconds"].to_numpy(dtype=float)
        elevation = block["delev_m"].to_numpy(dtype=float)
        total_distance = float(distance.sum())
        if total_distance < 1.0:
            continue

        cumulative_distance = np.concatenate(([0.0], np.cumsum(distance)))
        cumulative_seconds = np.concatenate(([0.0], np.cumsum(seconds)))
        cumulative_elevation = np.concatenate(([0.0], np.cumsum(elevation)))
        edges = np.arange(0.0, total_distance, sample_distance_m)
        if total_distance - edges[-1] > 1e-6:
            edges = np.append(edges, total_distance)
        edge_seconds = np.interp(edges, cumulative_distance, cumulative_seconds)
        edge_elevation = np.interp(edges, cumulative_distance, cumulative_elevation)
        terrain_elevation = edge_elevation.copy()
        if len(terrain_elevation) >= 3:
            padded = np.pad(terrain_elevation, (1, 1), mode="edge")
            terrain_elevation = np.convolve(padded, np.ones(3) / 3.0, mode="valid")
        chunk_distance = np.diff(edges)
        chunk_seconds = np.diff(edge_seconds)
        terrain_delta = np.diff(terrain_elevation)
        chunks = [
            {
                "distance": float(distance_m),
                "elevation_delta": float(elevation_delta),
                "grade": float(smooth_delta / distance_m * 100.0),
                "seconds": float(seconds_s),
            }
            for distance_m, seconds_s, elevation_delta, smooth_delta in zip(
                chunk_distance, chunk_seconds, np.diff(edge_elevation), terrain_delta
            )
            if distance_m > 1e-3 and seconds_s > 0
        ]
        for start, end, terrain_type in group_terrain_chunks(chunks, sample_distance_m):
            selected = chunks[start:end]
            segment_distance = sum(item["distance"] for item in selected)
            duration = sum(item["seconds"] for item in selected)
            gain = sum(max(item["elevation_delta"], 0.0) for item in selected)
            loss = sum(max(-item["elevation_delta"], 0.0) for item in selected)
            grade = sum(item["grade"] * item["distance"] for item in selected) / segment_distance
            segments.append(
                {
                    "activity": activity_name,
                    "activity_type": activity_type,
                    "type": terrain_type,
                    "distance_m": segment_distance,
                    "duration_s": duration,
                    "gain_m": gain,
                    "loss_m": loss,
                    "grade_pct": grade,
                    "pace": duration / segment_distance * 1000.0,
                    "speed_mps": segment_distance / duration,
                    "model_weight": float(block["_model_weight"].iloc[0]),
                }
            )
    return segments


def _build_uphill_profile(segments: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {}
    samples: dict[str, dict[str, float | int]] = {}
    for low, high, label in (
        (1.0, 5.0, "1_percent"),
        (5.0, 10.0, "5_percent"),
        (10.0, 15.0, "10_percent"),
        (15.0, 100.0, "15_percent"),
    ):
        sample = segments[
            (segments["type"] == "uphill")
            & segments["grade_pct"].between(low, high, inclusive="left")
            & (segments["gain_m"] > 0)
        ] if not segments.empty else segments
        seconds = float(sample["duration_s"].sum()) if not sample.empty else 0.0
        samples[label] = _slope_sample_summary(sample, "gain_m")
        if seconds > 0:
            result[label] = round(float(sample["gain_m"].sum()) / seconds * 3600.0, 1)
    result["_samples"] = samples
    return result


def _build_downhill_profile(segments: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {}
    if segments.empty:
        result["_samples"] = {}
        return result
    grade = segments["grade_pct"]
    bins = (
        ((grade > -5.0) & (grade < -1.0), "-1_percent"),
        ((grade > -10.0) & (grade <= -5.0), "-5_percent"),
        ((grade > -15.0) & (grade <= -10.0), "-10_percent"),
        ((grade <= -15.0), "-15_percent"),
    )
    samples: dict[str, dict[str, float | int]] = {}
    for grade_mask, label in bins:
        sample = segments[
            (segments["type"] == "downhill")
            & grade_mask
            & (segments["loss_m"] > 0)
        ]
        seconds = float(sample["duration_s"].sum()) if not sample.empty else 0.0
        samples[label] = _slope_sample_summary(sample, "loss_m")
        if seconds > 0:
            result[label] = {
                "speed_mps": round(float(sample["distance_m"].sum()) / seconds, 3),
                "vertical_speed_mph": round(float(sample["loss_m"].sum()) / seconds * 3600.0, 1),
            }
    result["_samples"] = samples
    return result


def _activity_type(name: str, frame: pd.DataFrame) -> str:
    sub_sport = str(frame.attrs.get("sub_sport") or "").lower()
    lowered = name.lower()
    if sub_sport == "trail" or "越野" in name or "trail" in lowered:
        return "trail"
    return "road"


def _percentile_or_nan(series: pd.Series, percentile: float) -> float:
    return float(np.percentile(series.to_numpy(dtype=float), percentile)) if len(series) else np.nan


def _weighted_percentile(frame: pd.DataFrame, column: str, percentile: float) -> float:
    if frame.empty:
        return np.nan
    values = frame[column].to_numpy(dtype=float)
    weights = frame.get("model_weight", pd.Series(1.0, index=frame.index)).to_numpy(dtype=float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    target = np.clip(percentile / 100.0, 0.0, 1.0) * weights.sum()
    index = min(int(np.searchsorted(np.cumsum(weights), target, side="left")), len(values) - 1)
    return float(values[index])


def _activity_weight(name: str, frame: pd.DataFrame) -> float:
    """Combine configured activity recency and purpose weights."""
    config = load_config()
    valid_time = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).dropna()
    age_days = max(0, (datetime.now(timezone.utc) - valid_time.max().to_pydatetime()).days) if not valid_time.empty else 9999
    recency = 0.4
    for band in config["activity_recency_weights"]:
        if band["max_days"] is None or age_days <= int(band["max_days"]):
            recency = float(band["weight"])
            break
    lowered = name.lower()
    purpose = "race" if any(word in lowered for word in ("race", "比赛")) else "recovery" if any(word in lowered for word in ("recovery", "恢复")) else "specific_training" if any(word in lowered for word in ("trail", "越野")) else "normal_training"
    return recency * float(config["activity_type_weights"][purpose])


def _flat_source_summary(segments: pd.DataFrame) -> dict[str, float | int | None]:
    if segments.empty:
        return {"segments": 0, "distance_km": 0.0, "median_pace": None, "fast_pace_p25": None}
    return {
        "segments": len(segments),
        "distance_km": round(float(segments["distance_m"].sum()) / 1000.0, 3),
        "median_pace": round(_percentile_or_nan(segments["pace"], 50), 1),
        "fast_pace_p25": round(_percentile_or_nan(segments["pace"], 25), 1),
    }


def _slope_sample_summary(segments: pd.DataFrame, vertical_column: str) -> dict[str, float | int]:
    return {
        "segments": len(segments),
        "distance_km": round(float(segments["distance_m"].sum()) / 1000.0, 3) if not segments.empty else 0.0,
        "vertical_m": round(float(segments[vertical_column].sum()), 1) if not segments.empty else 0.0,
        "duration_hour": round(float(segments["duration_s"].sum()) / 3600.0, 3) if not segments.empty else 0.0,
    }


def _uphill_curve(profile: dict[str, object], quality_score: float) -> list[dict[str, object]]:
    result = []
    for grade, key in zip((3.0, 7.5, 12.5, 18.0), ("1_percent", "5_percent", "10_percent", "15_percent")):
        sample = profile.get("_samples", {}).get(key, {})
        seconds = float(sample.get("duration_hour", 0)) * 3600.0
        count = int(sample.get("segments", 0))
        source = "personal" if count else "default"
        result.append({"grade": grade, "value": float(profile[key]), "unit": "vertical_metres_per_hour",
                       "confidence": calculate_confidence(seconds, count, quality_score, source=source),
                       "sample_count": count, "sample_duration_seconds": seconds,
                       "sample_distance_m": float(sample.get("distance_km", 0)) * 1000.0,
                       "sample_elevation_m": float(sample.get("vertical_m", 0)), "source": source})
    return result


def _downhill_curve(profile: dict[str, object], quality_score: float) -> list[dict[str, object]]:
    result = []
    for grade, key in zip((-3.0, -7.5, -12.5, -18.0), ("-1_percent", "-5_percent", "-10_percent", "-15_percent")):
        sample = profile.get("_samples", {}).get(key, {})
        seconds = float(sample.get("duration_hour", 0)) * 3600.0
        count = int(sample.get("segments", 0))
        source = "personal" if count else "default"
        ability = profile[key]
        result.append({"grade": grade, "speed_mps": float(ability["speed_mps"]),
                       "vertical_speed_mph": float(ability["vertical_speed_mph"]),
                       "confidence": calculate_confidence(seconds, count, quality_score, source=source),
                       "sample_count": count, "sample_duration_seconds": seconds,
                       "sample_distance_m": float(sample.get("distance_km", 0)) * 1000.0,
                       "sample_elevation_m": float(sample.get("vertical_m", 0)), "source": source})
    return result
