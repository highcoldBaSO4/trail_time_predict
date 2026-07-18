from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FONT_PATH = PROJECT_ROOT / "assets" / "fonts" / "NotoSansCJKsc-Regular.otf"
CHINESE_FONT_FAMILY = "Noto Sans CJK SC"


def configure_chart_font() -> None:
    """Register the bundled CJK font for Linux-based cloud deployments."""
    if not FONT_PATH.is_file():
        raise RuntimeError(f"Bundled chart font is missing: {FONT_PATH}")
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        CHINESE_FONT_FAMILY,
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


configure_chart_font()


SLOPE_STYLES = {
    "flat": ("平路 ±2%", "#8A94A3"),
    "uphill_2_5": ("微坡 2%～5%", "#EAB308"),
    "uphill_5_10": ("缓坡 5%～10%", "#F59E0B"),
    "uphill_10_15": ("中坡 10%～15%", "#F97316"),
    "uphill_15_20": ("较陡坡 15%～20%", "#EF4444"),
    "uphill_20_plus": ("陡坡 ≥20%", "#B91C1C"),
    "downhill_2_5": ("微下降 -2%～-5%", "#93C5FD"),
    "downhill_5_10": ("缓下降 -5%～-10%", "#60A5FA"),
    "downhill_10_15": ("中下降 -10%～-15%", "#2563EB"),
    "downhill_15_20": ("较陡下降 -15%～-20%", "#1D4ED8"),
    "downhill_20_plus": ("陡下降 ≤-20%", "#1E3A8A"),
}


def elevation_figure(
    segments: list[dict[str, Any]],
    *,
    actual_trace: list[dict[str, Any]] | None = None,
    figsize: tuple[float, float] = (11, 5.2),
) -> plt.Figure:
    detailed = [
        micro
        for segment in segments
        for micro in list(segment.get("micro_segments", []))
    ]
    absolute_elevation = bool(detailed) and all(bool(item.get("elevation_available", False)) for item in detailed)
    units = detailed if detailed else segments
    distances = [float(units[0].get("start_km", 0.0))]
    elevations = [
        float(units[0]["elevation_start"])
        if absolute_elevation else 0.0
    ]
    for unit in units:
        distances.append(float(unit["end_km"]))
        elevations.append(
            float(unit["elevation_end"])
            if absolute_elevation else elevations[-1] + float(unit["gain"]) - float(unit["loss"])
        )

    figure, axis = plt.subplots(figsize=figsize)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    baseline = min(elevations)
    axis.fill_between(distances, elevations, baseline, color="#98A2B3", alpha=0.10)

    for segment in segments:
        axis.axvspan(
            float(segment.get("start_km", 0.0)),
            float(segment["end_km"]),
            color=SLOPE_STYLES[_slope_band(segment)][1],
            alpha=0.075,
            linewidth=0,
        )

    for index, unit in enumerate(units):
        band = _slope_band(unit)
        color = SLOPE_STYLES[band][1]
        start_km, end_km = distances[index], distances[index + 1]
        axis.plot(
            [start_km, end_km],
            [elevations[index], elevations[index + 1]],
            color=color,
            linewidth=2.7,
            solid_capstyle="round",
            zorder=3,
        )

    legend = [
        Line2D([0], [0], color=color, linewidth=3, label=label)
        for label, color in SLOPE_STYLES.values()
    ]
    axis.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        ncol=3,
        frameon=False,
        fontsize=8.2,
        handlelength=2.0,
        columnspacing=1.3,
    )
    axis.set_xlabel("距离（km）", color="#667085")
    axis.set_ylabel("海拔（m）" if absolute_elevation else "相对海拔（m）", color="#667085")
    axis.grid(axis="y", color="#e8ebef", linewidth=0.8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#dfe4ea")
    axis.tick_params(colors="#667085", labelsize=9)

    if actual_trace:
        speed_axis = axis.twinx()
        trace_distance = [float(item["distance_km"]) for item in actual_trace]
        trace_speed = [float(item["speed_kmh"]) for item in actual_trace]
        speed_axis.plot(
            trace_distance,
            trace_speed,
            color="#087E5B",
            linewidth=1.65,
            alpha=0.9,
            label="实际速度",
            zorder=5,
        )
        speed_axis.set_ylabel("实际速度（km/h）", color="#087E5B")
        speed_axis.tick_params(axis="y", colors="#087E5B", labelsize=9)
        speed_axis.spines[["top", "left"]].set_visible(False)
        speed_axis.spines["right"].set_color("#b9d8cd")
        speed_axis.set_ylim(bottom=0.0)
        axis.text(
            0.995,
            0.975,
            "绿色曲线：实际速度（100m分箱，3点中值平滑）",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color="#087E5B",
            fontsize=8.3,
        )
    figure.tight_layout()
    return figure


def save_elevation_chart(segments: list[dict[str, Any]], path: str | Path) -> None:
    figure = elevation_figure(segments, figsize=(10, 4.0))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _slope_band(segment: dict[str, Any]) -> str:
    terrain = str(segment.get("type", "flat"))
    if terrain == "flat":
        return "flat"
    magnitude = abs(float(segment.get("grade", 0.0)))
    level = (
        "2_5" if magnitude < 5 else "5_10" if magnitude < 10 else
        "10_15" if magnitude < 15 else "15_20" if magnitude < 20 else "20_plus"
    )
    return f"{terrain}_{level}"
