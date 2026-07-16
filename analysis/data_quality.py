from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import load_config
from parser.gpx_reader import haversine_m


def diagnose_fit(frame: pd.DataFrame) -> dict[str, Any]:
    """Diagnose whether a parsed FIT activity is suitable for modelling."""
    issues: list[str] = []
    if frame.empty:
        return _report(0.0, ["活动没有有效记录"], False)
    timestamps = pd.to_datetime(frame.get("timestamp"), errors="coerce", utc=True)
    distance = pd.to_numeric(frame.get("distance"), errors="coerce")
    altitude = pd.to_numeric(frame.get("altitude"), errors="coerce")
    dt = timestamps.diff().dt.total_seconds()
    dd = distance.diff()
    speed = dd / dt
    settings = load_config()["quality"]
    if timestamps.isna().mean() > 0.02:
        issues.append("时间戳缺失较多")
    if (dt > float(settings["max_timestamp_gap_seconds"])).mean() > 0.02:
        issues.append("时间戳存在较多中断")
    if (dd < 0).any():
        issues.append("累计距离存在回退")
    if (speed > float(settings["max_speed_mps"])).mean() > 0.01:
        issues.append("存在较多异常瞬时速度")
    if altitude.isna().mean() > 0.20:
        issues.append("海拔缺失率较高")
    if (altitude.diff().abs() > float(settings["max_altitude_jump_m"])).any():
        issues.append("海拔存在明显跳变")
    duration = float(dt[dt > 0].sum())
    if duration < 600:
        issues.append("活动时长不足10分钟")
    if "heart_rate" in frame and frame["heart_rate"].isna().mean() > 0.50:
        issues.append("心率大量缺失")
    score = max(0.1, 1.0 - len(issues) * 0.14)
    report = _report(score, issues, score >= 0.55 and duration >= 600)
    temperature = pd.to_numeric(frame.get("temperature"), errors="coerce") if "temperature" in frame else pd.Series(dtype=float)
    valid_temperature = temperature.between(-30.0, 60.0)
    coverage = float(valid_temperature.mean()) if len(temperature) else 0.0
    device_temperature = (
        pd.to_numeric(frame.get("device_temperature"), errors="coerce")
        if "device_temperature" in frame else pd.Series(dtype=float)
    )
    device_coverage = float(device_temperature.between(-30.0, 70.0).mean()) if len(device_temperature) else 0.0
    temperature_issues: list[str] = []
    if coverage == 0 and device_coverage > 0:
        temperature_issues.append("FIT仅包含腕表温度，不作为环境温度")
    elif coverage == 0:
        temperature_issues.append("FIT没有可用温度数据")
    elif coverage < 0.50:
        temperature_issues.append("FIT温度覆盖率较低")
    if valid_temperature.any() and temperature[valid_temperature].diff().abs().max() > 10.0:
        temperature_issues.append("FIT温度存在明显跳变")
    report["heart_rate_coverage"] = round(float(frame["heart_rate"].notna().mean()), 3) if "heart_rate" in frame else 0.0
    report["temperature_coverage"] = round(coverage, 3)
    report["device_temperature_coverage"] = round(device_coverage, 3)
    report["temperature_issues"] = temperature_issues
    return report


def diagnose_gpx(points: list[dict[str, float | None]]) -> dict[str, Any]:
    """Measure GPX point spacing, elevation coverage and gain stability."""
    issues: list[str] = []
    spacings = [
        haversine_m(float(a["latitude"]), float(a["longitude"]), float(b["latitude"]), float(b["longitude"]))
        for a, b in zip(points, points[1:])
    ]
    elevations = np.asarray([np.nan if p.get("elevation") is None else float(p["elevation"]) for p in points])
    missing_rate = float(np.isnan(elevations).mean()) if len(elevations) else 1.0
    known = np.isfinite(elevations)
    max_jump = float(np.max(np.abs(np.diff(elevations[known])))) if known.sum() > 1 else 0.0
    raw_gain = float(np.clip(np.diff(elevations[known]), 0, None).sum()) if known.sum() > 1 else 0.0
    gains = {str(int(scale)): _resampled_gain(points, float(scale)) for scale in load_config()["quality"]["gpx_resample_distances_m"]}
    nonzero = [value for value in gains.values() if value > 0]
    variation = (max(nonzero) - min(nonzero)) / max(nonzero) if nonzero else 0.0
    if missing_rate > 0.20:
        issues.append("GPX海拔缺失率较高")
    if max_jump > float(load_config()["quality"]["max_altitude_jump_m"]):
        issues.append("GPX存在明显单点高度跳变")
    if variation > 0.25:
        issues.append("不同重采样尺度的累计爬升差异较大")
    score = max(0.1, 1.0 - len(issues) * 0.2)
    report = _report(score, issues, len(points) >= 2 and missing_rate < 0.8)
    report.update({"point_count": len(points), "average_spacing_m": round(float(np.mean(spacings)), 2) if spacings else 0.0,
                   "elevation_missing_rate": round(missing_rate, 3), "max_elevation_jump_m": round(max_jump, 1),
                   "raw_elevation_gain_m": round(raw_gain, 1),
                   "resampled_gain_m": {key: round(value, 1) for key, value in gains.items()}, "gain_variation": round(variation, 3)})
    return report


def _resampled_gain(points: list[dict[str, float | None]], scale: float) -> float:
    if len(points) < 2:
        return 0.0
    distances = [0.0]
    for a, b in zip(points, points[1:]):
        distances.append(distances[-1] + haversine_m(float(a["latitude"]), float(a["longitude"]), float(b["latitude"]), float(b["longitude"])))
    elevation = np.asarray([np.nan if p.get("elevation") is None else float(p["elevation"]) for p in points])
    known = np.isfinite(elevation)
    if known.sum() < 2 or distances[-1] <= 0:
        return 0.0
    sampled_distance = np.arange(0.0, distances[-1] + scale, scale).clip(max=distances[-1])
    sampled_distance = np.unique(sampled_distance)
    filled = np.interp(distances, np.asarray(distances)[known], elevation[known])
    sampled = np.interp(sampled_distance, distances, filled)
    return float(np.clip(np.diff(sampled), 0, None).sum())


def _report(score: float, issues: list[str], recommended: bool) -> dict[str, Any]:
    level = "高" if score >= 0.8 else "中" if score >= 0.55 else "低"
    return {"level": level, "score": round(score, 3), "issues": issues, "recommended_for_model": recommended}
