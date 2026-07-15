from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.activity_analysis import add_interval_metrics, analyze_activity
from analysis.fatigue import build_fatigue_profile
from parser.gpx_reader import group_terrain_chunks


DEFAULT_PROFILE = {
    "flat": {"aerobic_pace": 420.0, "threshold_pace": 360.0},
    "uphill": {"1_percent": 300.0, "5_percent": 550.0, "10_percent": 500.0, "15_percent": 400.0},
    "downhill": {
        "-1_percent": {"speed_mps": 2.8, "vertical_speed_mph": 400.0},
        "-5_percent": {"speed_mps": 2.5, "vertical_speed_mph": 900.0},
        "-10_percent": {"speed_mps": 2.2, "vertical_speed_mph": 1100.0},
        "-15_percent": {"speed_mps": 1.8, "vertical_speed_mph": 1250.0},
    },
    "fatigue": {"3h": 1.0, "5h": 0.90, "8h": 0.80},
}


def build_runner_profile(activities: dict[str, pd.DataFrame]) -> dict[str, object]:
    if not activities:
        raise ValueError("至少需要一个 FIT 活动才能生成能力画像")

    enriched: list[pd.DataFrame] = []
    terrain_segments: list[dict[str, float | str]] = []
    activity_types: dict[str, str] = {}
    for name, frame in activities.items():
        activity_type = _activity_type(name, frame)
        activity_types[name] = activity_type
        activity = add_interval_metrics(frame)
        activity["_activity_name"] = name
        activity["_activity_type"] = activity_type
        enriched.append(activity)
        terrain_segments.extend(_activity_terrain_segments(activity, name, activity_type))
    segment_frame = pd.DataFrame(terrain_segments)
    flat = _build_flat_profile(segment_frame)
    uphill = {**DEFAULT_PROFILE["uphill"], **_build_uphill_profile(segment_frame)}
    downhill = {**DEFAULT_PROFILE["downhill"], **_build_downhill_profile(segment_frame)}
    activity_summaries = []
    for name, frame in activities.items():
        summary = analyze_activity(frame, name)
        summary["activity_type"] = activity_types[name]
        activity_summaries.append(summary)
    profile: dict[str, object] = {
        "schema_version": "0.1",
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
    }
    return profile


def save_runner_profile(profile: dict[str, object], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_flat_profile(segments: pd.DataFrame) -> dict[str, object]:
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
        }

    trail = flat_segments[flat_segments["activity_type"] == "trail"]
    road = flat_segments[flat_segments["activity_type"] == "road"]
    road_penalty = 1.10
    trail_weight, road_weight = 0.70, 0.30
    trail_pace = _percentile_or_nan(trail["pace"], 50)
    trail_fast = _percentile_or_nan(trail["pace"], 25)
    road_pace = _percentile_or_nan(road["pace"], 50)
    road_fast = _percentile_or_nan(road["pace"], 25)
    if np.isfinite(trail_pace) and np.isfinite(road_pace):
        baseline = trail_weight * trail_pace + road_weight * road_pace * road_penalty
        faster = trail_weight * trail_fast + road_weight * road_fast * road_penalty
        source = "trail_70pct_plus_discounted_road_30pct"
    elif np.isfinite(trail_pace):
        baseline, faster, source = trail_pace, trail_fast, "trail_only"
    else:
        baseline, faster, source = road_pace * road_penalty, road_fast * road_penalty, "discounted_road_only"
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
