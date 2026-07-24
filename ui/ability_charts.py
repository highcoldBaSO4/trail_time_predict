from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


TERRAIN_STYLE = {
    "flat": ("平路", "#2a9d8f"),
    "uphill": ("上坡", "#d97706"),
    "downhill": ("下坡", "#2563eb"),
}
SOURCE_MARKERS = {"personal": "o", "blended": "o", "default": "s", "extrapolated": "^", "anchor": "o"}


def terrain_ability_figure(profile: dict[str, Any], terrain: str) -> Figure:
    """Plot the continuous uphill VAM or downhill speed curve."""
    if terrain not in {"uphill", "downhill"}:
        raise ValueError("能力曲线仅支持 uphill 或 downhill")
    points = sorted(list(profile.get(terrain, {}).get("curve", [])), key=lambda point: float(point.get("grade", 0.0)))
    figure, axis = _figure()
    if not points:
        _no_data(axis, "没有可绘制的坡度能力数据")
        return figure

    grades = np.asarray([float(point["grade"]) for point in points])
    values = np.asarray([
        float(point["value"]) if terrain == "uphill" else float(point["speed_mps"]) * 3.6
        for point in points
    ])
    label, color = TERRAIN_STYLE[terrain]
    axis.plot(grades, values, color=color, linewidth=2.2, label=f"{label}能力")
    for grade, value, point in zip(grades, values, points):
        source = str(point.get("source", "personal"))
        marker = SOURCE_MARKERS.get(source, "o")
        face = color if source not in {"default", "extrapolated"} else "white"
        axis.scatter(grade, value, s=44, marker=marker, facecolor=face, edgecolor=color, linewidth=1.5, zorder=3)
        axis.annotate(
            f"{value:.0f}" if terrain == "uphill" else f"{value:.1f}",
            (grade, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9,
        )
    axis.set_xlabel("平均坡度 (%)")
    axis.set_ylabel("VAM (m/h)" if terrain == "uphill" else "下坡速度 (km/h)")
    axis.set_xticks(grades)
    axis.legend(loc="best", frameon=False)
    _finish(axis)
    return figure


def fatigue_figure(fatigue: dict[str, Any]) -> Figure:
    """Plot terrain-specific retained-performance curves and evidence markers."""
    figure, axis = _figure(height=3.9)
    has_points = False
    for terrain, (label, color) in TERRAIN_STYLE.items():
        points = sorted(list(fatigue.get(terrain, [])), key=lambda point: float(point.get("hour", 0.0)))
        if not points:
            continue
        has_points = True
        hours = np.asarray([float(point["hour"]) for point in points])
        values = np.asarray([float(point["factor"]) * 100.0 for point in points])
        axis.plot(hours, values, color=color, linewidth=2.2, label=label)
        for hour, value, point in zip(hours, values, points):
            source = str(point.get("source", "default"))
            marker = SOURCE_MARKERS.get(source, "o")
            face = color if source not in {"default", "extrapolated"} else "white"
            axis.scatter(hour, value, s=42, marker=marker, facecolor=face, edgecolor=color, linewidth=1.4, zorder=3)
    if not has_points:
        _no_data(axis, "没有可绘制的疲劳曲线")
        return figure
    for retention in (100, 90, 80):
        axis.axhline(retention, color="#9ca3af", linewidth=0.8, alpha=0.55, zorder=0)
    axis.set_xlabel("累计移动时间 (小时)")
    axis.set_ylabel("能力保留 (%)")
    axis.set_ylim(max(40.0, axis.get_ylim()[0] - 3.0), 103.0)
    axis.legend(loc="best", frameon=False, ncols=3)
    _finish(axis)
    return figure


def temperature_figure(temperature: dict[str, Any]) -> Figure:
    """Plot final temperature tolerance beside the system baseline."""
    points = sorted(list(temperature.get("curve", [])), key=lambda point: float(point.get("temperature_c", 0.0)))
    figure, axis = _figure(height=3.9)
    if not points:
        _no_data(axis, "没有可绘制的温度耐受曲线")
        return figure
    temperatures = np.asarray([float(point["temperature_c"]) for point in points])
    final = np.asarray([float(point["time_factor"]) for point in points])
    baseline = np.asarray([float(point.get("default_time_factor", point["time_factor"])) for point in points])
    axis.axvspan(10.0, 20.0, color="#2a9d8f", alpha=0.10, label="舒适区 10–20℃")
    axis.plot(temperatures, baseline, color="#6b7280", linewidth=1.4, linestyle="--", label="系统先验")
    axis.plot(temperatures, final, color="#b45309", linewidth=2.2, label="最终耐受曲线")
    for temperature_c, factor, point in zip(temperatures, final, points):
        source = str(point.get("source", "default"))
        marker = "o" if source == "personal_blend" else "s"
        face = "#b45309" if source == "personal_blend" else "white"
        axis.scatter(temperature_c, factor, s=42, marker=marker, facecolor=face, edgecolor="#b45309", linewidth=1.4, zorder=3)
    axis.set_xlabel("环境温度 (℃)")
    axis.set_ylabel("耗时系数")
    axis.set_xticks(temperatures)
    axis.yaxis.set_major_formatter(lambda value, _: f"×{value:.2f}")
    axis.legend(loc="best", frameon=False)
    _finish(axis)
    return figure


def _figure(height: float = 3.5) -> tuple[Figure, Any]:
    figure, axis = plt.subplots(figsize=(8.4, height))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    return figure, axis


def _finish(axis: Any) -> None:
    axis.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.margins(x=0.06)
    axis.figure.tight_layout()


def _no_data(axis: Any, message: str) -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()
