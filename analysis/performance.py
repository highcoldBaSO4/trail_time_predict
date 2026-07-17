from __future__ import annotations

from datetime import timezone
from typing import Callable

import numpy as np
import pandas as pd

from analysis.activity_analysis import add_interval_metrics
from analysis.capability import build_runner_profile
from analysis.data_quality import diagnose_fit, diagnose_gpx
from models.performance_result import PerformanceResult
from models.race_condition import RaceCondition
from parser.gpx_reader import build_race_segments
from predictor.race_predictor import predict_race


TERRAIN_LABELS = {"flat": "平路", "uphill": "上坡", "downhill": "下坡"}


def analyze_performance(
    baseline_activities: dict[str, pd.DataFrame],
    baseline_types: dict[str, str],
    target_activity: pd.DataFrame,
    target_name: str,
    sample_distance_m: float = 100.0,
    simulations: int = 3000,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Compare one independent FIT activity with a profile built from baseline FITs."""
    if not baseline_activities:
        raise ValueError("活动表现诊断至少需要一个历史基准 FIT")
    if target_activity.empty:
        raise ValueError("待诊断 FIT 没有有效记录")

    _emit(progress, "建立不包含待诊断活动的基准能力画像……")
    profile = build_runner_profile(baseline_activities, baseline_types, progress=progress)
    points = fit_route_points(target_activity)
    if len(points) < 2:
        raise ValueError("待诊断 FIT 缺少有效轨迹坐标，无法生成诊断路线")
    route_quality = diagnose_gpx(points)
    _emit(progress, "使用待诊断活动路线识别自然地形段……")
    segments = build_race_segments(points, sample_distance_m)

    target_quality = diagnose_fit(target_activity)
    condition = _observed_condition(target_activity)
    _emit(progress, "按基准能力计算该活动路线的预期表现……")
    prediction = predict_race(
        profile,
        segments,
        aid_minutes=0.0,
        condition=condition,
        simulations=simulations,
        gpx_quality_score=float(route_quality["score"]),
    )
    diagnosis = _compare_actual_with_prediction(
        target_activity,
        target_name,
        prediction,
        float(target_quality["score"]),
        float(route_quality["score"]),
    )
    route_distance_m = sum(float(segment.get("distance", 0.0)) for segment in segments)
    activity_trace = _build_activity_trace(target_activity, route_distance_m)
    _emit(progress, "活动表现对比完成")
    return {
        "profile": profile,
        "prediction": prediction,
        "diagnosis": diagnosis,
        "activity_trace": activity_trace,
        "target_data_quality": target_quality,
        "route_data_quality": route_quality,
    }


def fit_route_points(frame: pd.DataFrame) -> list[dict[str, float | None]]:
    """Convert valid FIT positions into the route-point shape used by GPX analysis."""
    required = {"latitude", "longitude"}
    if not required <= set(frame.columns):
        return []
    latitude = pd.to_numeric(frame["latitude"], errors="coerce")
    longitude = pd.to_numeric(frame["longitude"], errors="coerce")
    altitude = pd.to_numeric(frame.get("altitude"), errors="coerce")
    valid = latitude.between(-90.0, 90.0) & longitude.between(-180.0, 180.0)
    points: list[dict[str, float | None]] = []
    previous: tuple[float, float] | None = None
    for lat, lon, ele in zip(latitude[valid], longitude[valid], altitude[valid]):
        coordinates = (float(lat), float(lon))
        if previous == coordinates:
            continue
        points.append(
            {
                "latitude": coordinates[0],
                "longitude": coordinates[1],
                "elevation": None if pd.isna(ele) else float(ele),
            }
        )
        previous = coordinates
    return points


def _build_activity_trace(
    frame: pd.DataFrame,
    route_distance_m: float,
    bin_distance_m: float = 100.0,
) -> list[dict[str, float]]:
    """Build a compact distance-aligned elevation/speed trace for the UI."""
    data = add_interval_metrics(frame)
    valid = data["valid_interval"].fillna(False) & data["dt_seconds"].notna()
    cumulative = data["dd_m"].where(data["dd_m"] > 0.0, 0.0).fillna(0.0).cumsum()
    activity_distance_m = float(cumulative.max())
    if activity_distance_m <= 0.0 or route_distance_m <= 0.0:
        return []

    data = data.loc[valid].copy()
    cumulative = cumulative.loc[data.index]
    interval_distance = data["dd_m"].fillna(0.0).clip(lower=0.0)
    # The diagnosis may use a separately uploaded GPX. Align the FIT trace to
    # that route by progress while retaining the unscaled FIT distance for the
    # actual speed calculation.
    route_position_m = (
        cumulative - interval_distance / 2.0
    ) * route_distance_m / activity_distance_m
    data["_route_position_m"] = route_position_m.clip(0.0, route_distance_m)
    data["_trace_bin"] = np.floor(data["_route_position_m"] / max(bin_distance_m, 1.0)).astype(int)

    rows: list[dict[str, float]] = []
    for _, sample in data.groupby("_trace_bin", sort=True):
        seconds = float(sample["dt_seconds"].sum())
        distance_m = float(sample["dd_m"].clip(lower=0.0).sum())
        if seconds <= 0.0 or distance_m <= 0.0:
            continue
        altitude = pd.to_numeric(sample["smoothed_altitude"], errors="coerce").median()
        if pd.isna(altitude):
            continue
        rows.append(
            {
                "distance_km": round(float(sample["_route_position_m"].median()) / 1000.0, 3),
                "elevation_m": round(float(altitude), 1),
                "speed_kmh": distance_m / seconds * 3.6,
            }
        )
    if not rows:
        return []
    speed = pd.Series([row["speed_kmh"] for row in rows], dtype=float)
    smoothed_speed = speed.rolling(3, center=True, min_periods=1).median().clip(0.0, 30.0)
    for row, value in zip(rows, smoothed_speed):
        row["speed_kmh"] = round(float(value), 2)
    return rows


def _compare_actual_with_prediction(
    frame: pd.DataFrame,
    target_name: str,
    prediction: dict[str, object],
    fit_quality_score: float,
    route_quality_score: float,
) -> dict[str, object]:
    data = add_interval_metrics(frame)
    valid = data["valid_interval"].fillna(False) & data["dt_seconds"].notna()
    moving = data["moving_interval"].fillna(False) & data["dt_seconds"].notna()
    actual_moving = float(data.loc[moving, "dt_seconds"].sum())
    if actual_moving <= 0:
        raise ValueError("待诊断 FIT 没有可计算的移动时间")

    timestamps = pd.to_datetime(data["timestamp"], errors="coerce", utc=True).dropna()
    timestamp_elapsed = float((timestamps.max() - timestamps.min()).total_seconds()) if len(timestamps) > 1 else actual_moving
    actual_elapsed = _positive_attr(frame, "total_elapsed_time", timestamp_elapsed)
    actual_timer = _positive_attr(frame, "total_timer_time", min(actual_elapsed, float(data.loc[valid, "dt_seconds"].sum())))
    actual_elapsed = max(actual_elapsed, actual_timer, actual_moving)
    stopped = max(0.0, actual_elapsed - actual_moving)
    paused = max(0.0, actual_elapsed - actual_timer)
    nonmoving_timer = max(0.0, actual_timer - actual_moving)

    predicted_moving = float(prediction["adjusted_moving_time_seconds"])
    deviation = actual_moving - predicted_moving
    deviation_percent = deviation / max(predicted_moving, 1.0) * 100.0
    samples = np.asarray(prediction.get("probability", {}).get("samples_seconds", []), dtype=float)
    percentile = float(np.mean(samples <= actual_moving) * 100.0) if samples.size else 50.0

    segment_rows = _actual_segment_rows(data, moving, list(prediction["segments"]))
    terrain_analysis = _aggregate_by_terrain(segment_rows)
    progress_analysis = _aggregate_by_progress(data, moving, list(prediction["segments"]))
    confidence = min(float(prediction.get("confidence", 0.2)), fit_quality_score, route_quality_score)
    result = PerformanceResult(
        activity_name=target_name,
        actual_elapsed_seconds=actual_elapsed,
        actual_timer_seconds=actual_timer,
        actual_moving_seconds=actual_moving,
        stopped_seconds=stopped,
        paused_seconds=paused,
        nonmoving_timer_seconds=nonmoving_timer,
        predicted_moving_seconds=predicted_moving,
        deviation_seconds=deviation,
        deviation_percent=deviation_percent,
        prediction_percentile=percentile,
        confidence=confidence,
        terrain_analysis=terrain_analysis,
        progress_analysis=progress_analysis,
        segment_analysis=segment_rows,
        metadata={
            "prediction_range": {
                "p10_seconds": float(prediction["optimistic_time_seconds"]),
                "p50_seconds": float(prediction["median_finish_time_seconds"]),
                "p90_seconds": float(prediction["conservative_time_seconds"]),
            },
            "performance_label": _performance_label(percentile),
            "comparison_basis": "actual_moving_vs_condition_adjusted_predicted_moving",
            "stopped_time_note": "非移动差额拆分为 FIT 暂停/未计时和计时中的停留；移动时间由 15 秒局部窗口估算",
        },
    )
    return result.to_dict()


def _actual_segment_rows(
    data: pd.DataFrame, moving: pd.Series, predicted_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    route_distance = sum(float(row["distance"]) for row in predicted_rows)
    cumulative = data["dd_m"].where(data["dd_m"] > 0, 0.0).fillna(0.0).cumsum()
    activity_distance = float(cumulative.max())
    if activity_distance <= 0 or route_distance <= 0:
        return []
    position = (cumulative - data["dd_m"].fillna(0.0).clip(lower=0.0) / 2.0) * route_distance / activity_distance
    rows: list[dict[str, object]] = []
    for index, predicted in enumerate(predicted_rows):
        start_m = float(predicted["start_km"]) * 1000.0
        end_m = float(predicted["end_km"]) * 1000.0
        boundary = position.between(start_m, end_m, inclusive="both" if index == len(predicted_rows) - 1 else "left")
        selected = moving & boundary
        actual_seconds = float(data.loc[selected, "dt_seconds"].sum())
        predicted_seconds = float(predicted["predicted_time_seconds"])
        hr_seconds = float(data.loc[selected & data["heart_rate"].notna(), "dt_seconds"].sum())
        average_hr = (
            float((data.loc[selected, "heart_rate"] * data.loc[selected, "dt_seconds"]).sum() / hr_seconds)
            if hr_seconds > 0 else None
        )
        rows.append(
            {
                "start_km": float(predicted["start_km"]),
                "end_km": float(predicted["end_km"]),
                "terrain": str(predicted.get("type", "flat")),
                "terrain_label": str(predicted.get("terrain", TERRAIN_LABELS.get(str(predicted.get("type")), "未知"))),
                "grade": float(predicted.get("grade", 0.0)),
                "distance_m": float(predicted["distance"]),
                "predicted_seconds": round(predicted_seconds, 1),
                "actual_seconds": round(actual_seconds, 1),
                "deviation_seconds": round(actual_seconds - predicted_seconds, 1),
                "deviation_percent": round((actual_seconds / predicted_seconds - 1.0) * 100.0, 2) if predicted_seconds > 0 else 0.0,
                "average_hr_bpm": None if average_hr is None else round(average_hr, 1),
            }
        )
    return rows


def _aggregate_by_terrain(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for terrain in ("flat", "uphill", "downhill"):
        selected = [row for row in rows if row["terrain"] == terrain]
        predicted = sum(float(row["predicted_seconds"]) for row in selected)
        actual = sum(float(row["actual_seconds"]) for row in selected)
        result[terrain] = {
            "predicted_seconds": round(predicted, 1),
            "actual_seconds": round(actual, 1),
            "deviation_seconds": round(actual - predicted, 1),
            "deviation_percent": round((actual / predicted - 1.0) * 100.0, 2) if predicted > 0 else 0.0,
        }
    return result


def _aggregate_by_progress(
    data: pd.DataFrame, moving: pd.Series, predicted_rows: list[dict[str, object]]
) -> dict[str, dict[str, float]]:
    cumulative = data["dd_m"].where(data["dd_m"] > 0, 0.0).fillna(0.0).cumsum()
    total = float(cumulative.max())
    actual_first = float(data.loc[moving & (cumulative <= total / 2.0), "dt_seconds"].sum()) if total > 0 else 0.0
    actual_second = float(data.loc[moving & (cumulative > total / 2.0), "dt_seconds"].sum()) if total > 0 else 0.0
    route_total = sum(float(row["distance"]) for row in predicted_rows)
    predicted_first = _predicted_overlap_seconds(predicted_rows, 0.0, route_total / 2.0)
    predicted_second = _predicted_overlap_seconds(predicted_rows, route_total / 2.0, route_total)
    return {
        "first_half": _comparison_values(predicted_first, actual_first),
        "second_half": _comparison_values(predicted_second, actual_second),
    }


def _predicted_overlap_seconds(rows: list[dict[str, object]], start_m: float, end_m: float) -> float:
    seconds = 0.0
    for row in rows:
        row_start = float(row["start_km"]) * 1000.0
        row_end = float(row["end_km"]) * 1000.0
        overlap = max(0.0, min(end_m, row_end) - max(start_m, row_start))
        row_distance = max(float(row["distance"]), 1e-9)
        seconds += float(row["predicted_time_seconds"]) * overlap / row_distance
    return seconds


def _comparison_values(predicted: float, actual: float) -> dict[str, float]:
    return {
        "predicted_seconds": round(predicted, 1),
        "actual_seconds": round(actual, 1),
        "deviation_seconds": round(actual - predicted, 1),
        "deviation_percent": round((actual / predicted - 1.0) * 100.0, 2) if predicted > 0 else 0.0,
    }


def _observed_condition(frame: pd.DataFrame) -> RaceCondition:
    timestamps = pd.to_datetime(frame.get("timestamp"), errors="coerce", utc=True).dropna()
    start = timestamps.min().to_pydatetime().astimezone(timezone.utc) if not timestamps.empty else None
    temperature = _median_column(frame, "temperature", -30.0, 60.0)
    humidity = _median_column(frame, "weather_relative_humidity_2m", 0.0, 100.0)
    return RaceCondition(
        current_form="normal",
        pacing_strategy="standard",
        temperature_c=temperature,
        humidity_percent=humidity,
        race_start_time_utc=start,
    )


def _median_column(frame: pd.DataFrame, column: str, minimum: float, maximum: float) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[values.between(minimum, maximum)].dropna()
    return None if values.empty else float(values.median())


def _positive_attr(frame: pd.DataFrame, key: str, fallback: float) -> float:
    try:
        value = float(frame.attrs.get(key))
        return value if np.isfinite(value) and value > 0 else max(0.0, fallback)
    except (TypeError, ValueError):
        return max(0.0, fallback)


def _performance_label(percentile: float) -> str:
    if percentile < 10.0:
        return "明显快于模型合理区间"
    if percentile <= 90.0:
        return "处于模型合理区间"
    return "明显慢于模型合理区间"


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
