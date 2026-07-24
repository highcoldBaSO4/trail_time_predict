from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Callable

import pandas as pd

from analysis.ability_file import AbilityBundle, profile_before_activity
from analysis.calibration import calibration_summary
from analysis.performance import analyze_performance
from analysis.residuals import grouped_metric_summary, metric_summary
from config import load_config
from models import BacktestResult


def run_rolling_backtest(
    bundle: AbilityBundle,
    simulations: int | None = None,
    seed: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Backtest each FIT using only the evidence available before it started.

    The returned payload contains route and condition summaries only; FIT
    records, geographic traces and other raw activity evidence are never copied
    into the result.
    """
    if not bundle.supports_update:
        raise ValueError("滚动回测需要包含历史活动证据的新版个人能力文件")
    config = load_config()
    backtest_config = dict(config.get("backtest", {}))
    run_seed = int(backtest_config.get("seed", config["monte_carlo"]["seed"]) if seed is None else seed)
    run_simulations = int(backtest_config.get("simulations", config["monte_carlo"]["min_simulations"]) if simulations is None else simulations)
    ordered = _ordered_activities(bundle)
    records: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    run_route_ablations = bool(backtest_config.get("route_similarity_ablation", True))
    ablation_records: dict[str, list[dict[str, object]]] = {
        "structural": records,
        "legacy": [],
        "duration_fallback": [],
    }
    for index, (name, frame, started_at) in enumerate(ordered):
        _emit(progress, f"滚动回测 {index + 1}/{len(ordered)}：{name}")
        digest = bundle.activity_hashes.get(name)
        try:
            profile, _ = profile_before_activity(bundle, frame, target_hash=digest)
        except ValueError as exc:
            skipped.append({"target_activity_time": started_at.isoformat(), "reason": str(exc)})
            continue
        result = analyze_performance(
            {}, {}, frame, name, simulations=run_simulations, seed=run_seed + index, profile=profile
        )
        records.append(_record_from_result(bundle, name, frame, started_at, profile, result, _config_version(config)))
        if run_route_ablations:
            for mode in ("legacy", "duration_fallback"):
                ablation_result = analyze_performance(
                    {}, {}, frame, name, simulations=run_simulations, seed=run_seed + index,
                    profile=profile, route_strategy_matching_mode=mode,
                )
                ablation_records[mode].append(
                    _record_from_result(bundle, name, frame, started_at, profile, ablation_result, _config_version(config))
                )
    metrics = _metrics(records)
    calibration = dict(metrics.get("calibration", {}))
    model = dict(calibration.get("model", {}))
    if model:
        model["activity_evidence_fingerprint"] = _activity_evidence_fingerprint(bundle)
        calibration["model"] = model
        metrics["calibration"] = calibration
    if run_route_ablations:
        metrics["route_similarity_ablation"] = {
            mode: metric_summary(items) for mode, items in ablation_records.items()
        }
    result = BacktestResult(
        records=records,
        skipped=skipped,
        metrics=metrics,
        metadata={
            "model_version": "phase4_conservative_calibration",
            "config_version": _config_version(config),
            "simulations": run_simulations,
            "seed": run_seed,
            "raw_trajectory_stored": False,
            "method": "profile_before_activity_then_target_fit_route",
            "route_similarity_ablation_enabled": run_route_ablations,
            "route_similarity_ablations": {
                mode: items for mode, items in ablation_records.items()
            } if run_route_ablations else {},
        },
    )
    return result.to_dict()


def _ordered_activities(bundle: AbilityBundle) -> list[tuple[str, pd.DataFrame, datetime]]:
    ordered: list[tuple[str, pd.DataFrame, datetime]] = []
    for name, frame in bundle.activities.items():
        timestamps = pd.to_datetime(frame.get("timestamp"), errors="coerce", utc=True).dropna()
        if timestamps.empty:
            continue
        ordered.append((name, frame, timestamps.min().to_pydatetime()))
    return sorted(ordered, key=lambda item: (item[2], item[0]))


def _record_from_result(
    bundle: AbilityBundle,
    name: str,
    frame: pd.DataFrame,
    started_at: datetime,
    profile: dict[str, object],
    result: dict[str, object],
    config_version: str,
) -> dict[str, object]:
    diagnosis = dict(result["diagnosis"])
    prediction = dict(result["prediction"])
    route = dict(prediction.get("route", {}))
    condition = dict(prediction.get("condition", {}))
    historical = _baseline_dates(bundle, started_at)
    terrain_analysis = dict(diagnosis.get("terrain_analysis", {}))
    progress_analysis = dict(diagnosis.get("progress_analysis", {}))
    return {
        "target_activity_time": started_at.isoformat(),
        "baseline_activity_count": historical["count"],
        "baseline_earliest_activity_time": historical["earliest"],
        "baseline_latest_activity_time": historical["latest"],
        "model_version": str(profile.get("schema_version", "unknown")),
        "config_version": config_version,
        "route": {
            "distance_km": route.get("distance_km"),
            "elevation_gain_m": route.get("elevation_gain"),
            "terrain_share": prediction.get("pacing_strategy_match", {}).get("target", {}).get("terrain_share", {}),
            "best_similarity": prediction.get("pacing_strategy_match", {}).get("best_similarity"),
        },
        "conditions": {
            "temperature_c": condition.get("temperature_c"),
            "humidity_percent": condition.get("humidity_percent"),
            "night_ratio": prediction.get("environment", {}).get("race_night_ratio"),
            "maximum_elevation_m": prediction.get("environment", {}).get("race_maximum_elevation_m"),
        },
        "actual_moving_seconds": float(diagnosis["actual_moving_seconds"]),
        "p10_seconds": float(prediction["optimistic_time_seconds"]),
        "p50_seconds": float(prediction["median_finish_time_seconds"]),
        "p90_seconds": float(prediction["conservative_time_seconds"]),
        "p50_error": round(float(diagnosis["actual_moving_seconds"]) / float(prediction["median_finish_time_seconds"]) - 1.0, 5),
        "covered_by_p10_p90": bool(
            float(prediction["optimistic_time_seconds"]) <= float(diagnosis["actual_moving_seconds"])
            <= float(prediction["conservative_time_seconds"])
        ),
        "terrain_errors": {terrain: values.get("deviation_percent") for terrain, values in terrain_analysis.items()},
        "progress_errors": {stage: values.get("deviation_percent") for stage, values in progress_analysis.items()},
        "confidence": float(diagnosis["confidence"]),
        "data_quality": {
            "target": result.get("target_data_quality", {}).get("score"),
            "route": result.get("route_data_quality", {}).get("score"),
        },
    }


def _baseline_dates(bundle: AbilityBundle, target_started_at: datetime) -> dict[str, object]:
    dates: list[datetime] = []
    for frame in bundle.activities.values():
        timestamps = pd.to_datetime(frame.get("timestamp"), errors="coerce", utc=True).dropna()
        if not timestamps.empty and timestamps.min().to_pydatetime() < target_started_at:
            dates.append(timestamps.min().to_pydatetime())
    return {
        "count": len(dates),
        "earliest": min(dates).isoformat() if dates else None,
        "latest": max(dates).isoformat() if dates else None,
    }


def _metrics(records: list[dict[str, object]]) -> dict[str, object]:
    terrain_records = {
        terrain: _records_for_error(records, "terrain_errors", terrain)
        for terrain in ("flat", "uphill", "downhill")
    }
    return {
        "overall": metric_summary(records),
        "by_terrain": {terrain: _error_metric_summary(items) for terrain, items in terrain_records.items()},
        "by_duration": grouped_metric_summary(records, _duration_group),
        "by_route_similarity": grouped_metric_summary(records, _similarity_group),
        "by_temperature": grouped_metric_summary(records, _temperature_group),
        "by_evidence": grouped_metric_summary(records, _evidence_group),
        "by_progress": {
            stage: _error_metric_summary(_records_for_error(records, "progress_errors", stage))
            for stage in ("first_half", "second_half")
        },
        "calibration": calibration_summary(records),
    }


def _records_for_error(records: list[dict[str, object]], field: str, key: str) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for record in records:
        error = record.get(field, {}).get(key)
        if error is not None:
            selected.append({**record, "_error": float(error) / 100.0})
    return selected


def _error_metric_summary(records: list[dict[str, object]]) -> dict[str, float | int | None]:
    if not records:
        return {"count": 0, "signed_mean_error": None, "median_absolute_percentage_error": None,
                "mean_absolute_percentage_error": None}
    values = pd.Series([float(record["_error"]) for record in records])
    return {
        "count": len(records), "signed_mean_error": round(float(values.mean()), 5),
        "median_absolute_percentage_error": round(float(values.abs().median()), 5),
        "mean_absolute_percentage_error": round(float(values.abs().mean()), 5),
    }


def _duration_group(record: dict[str, object]) -> str:
    hours = float(record["actual_moving_seconds"]) / 3600.0
    return "<2h" if hours < 2 else "2-5h" if hours < 5 else "5-8h" if hours < 8 else ">8h"


def _similarity_group(record: dict[str, object]) -> str | None:
    value = record.get("route", {}).get("best_similarity")
    if value is None:
        return None
    score = float(value)
    return "高" if score >= 0.65 else "中" if score >= 0.35 else "低"


def _temperature_group(record: dict[str, object]) -> str:
    temperature = record.get("conditions", {}).get("temperature_c")
    return "未知" if temperature is None else "高温" if float(temperature) >= 25.0 else "正常"


def _evidence_group(record: dict[str, object]) -> str:
    return "充分" if int(record["baseline_activity_count"]) >= 5 else "不充分"


def _config_version(config: dict[str, object]) -> str:
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _activity_evidence_fingerprint(bundle: AbilityBundle) -> str:
    """Stable, non-reversible identifier used to reject stale calibration."""
    payload = json.dumps(sorted(bundle.activity_hashes.values()), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
