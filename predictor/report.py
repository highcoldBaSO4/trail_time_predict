from __future__ import annotations

from pathlib import Path

from predictor.race_predictor import format_duration, format_pace


def build_markdown_report(profile: dict[str, object], prediction: dict[str, object]) -> str:
    route = prediction["route"]
    flat = profile["flat"]
    uphill = profile["uphill"]
    downhill = profile["downhill"]
    fatigue = profile["fatigue"]
    lines = [
        "# 越野跑比赛时间预测报告",
        "",
        "## 个人能力",
        "",
        f"- 越野平路综合配速：{format_pace(float(flat['aerobic_pace']))}/km",
        f"- 越野平路较快配速（P25）：{format_pace(float(flat['threshold_pace']))}/km",
        f"- 合格自然平路样本：{int(flat.get('qualified_segments', 0))} 段，"
        f"共 {float(flat.get('sample_distance_km', 0)):.2f} km",
        "",
        "### 上坡能力",
        "",
        "| 档位 | 平均坡度 | 等效配速 | VAM | 历史样本 | 累计距离 | 累计高度 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    uphill_rows = (
        ("微坡", "1%–5%", "1_percent"),
        ("缓坡", "5%–10%", "5_percent"),
        ("中坡", "10%–15%", "10_percent"),
        ("陡坡", "≥15%", "15_percent"),
    )
    for name, grade_range, key in uphill_rows:
        sample = uphill.get("_samples", {}).get(key, {})
        lines.append(
            f"| {name} | {grade_range} | {_sample_pace(sample)} | {float(uphill[key]):.0f} m/h | "
            f"{int(sample.get('segments', 0))} 段 | {float(sample.get('distance_km', 0)):.2f} km | "
            f"+{float(sample.get('vertical_m', 0)):.0f} m |"
        )
    lines.extend(
        [
            "",
            "### 下坡能力",
            "",
            "| 档位 | 平均坡度 | 等效配速 | VAM | 历史样本 | 累计距离 | 累计高度 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    downhill_rows = (
        ("微下降", "-1%～-5%", "-1_percent"),
        ("缓下降", "-5%～-10%", "-5_percent"),
        ("中下降", "-10%～-15%", "-10_percent"),
        ("陡下降", "≤-15%", "-15_percent"),
    )
    for name, grade_range, key in downhill_rows:
        ability = downhill[key]
        sample = downhill.get("_samples", {}).get(key, {})
        speed = float(ability["speed_mps"])
        lines.append(
            f"| {name} | {grade_range} | {_sample_pace(sample, speed) } | "
            f"-{float(ability['vertical_speed_mph']):.0f} m/h | {int(sample.get('segments', 0))} 段 | "
            f"{float(sample.get('distance_km', 0)):.2f} km | -{float(sample.get('vertical_m', 0)):.0f} m |"
        )
    lines.extend(
        [
            "",
            "### 长时间疲劳衰减",
            "",
            "| 累计移动时间 | 能力保留比例 | 耗时修正倍率 |",
            "| --- | ---: | ---: |",
            _fatigue_row("0–3 小时", float(fatigue["3h"])),
            _fatigue_row("3–5 小时", float(fatigue["5h"])),
            _fatigue_row("5 小时以上", float(fatigue["8h"])),
            "",
            "> 疲劳修正规则：分段基础耗时 ÷ 能力保留比例。比如保留比例为 80%，该段耗时按 1.25 倍计算。",
        ]
    )
    lines.extend(
        [
        "",
        "## 比赛预测",
        "",
        f"- 距离：{float(route['distance_km']):.2f} km",
        f"- 累计爬升：{float(route['elevation_gain']):.0f} m",
        f"- 累计下降：{float(route['elevation_loss']):.0f} m",
        f"- 自然爬坡：{int(route.get('climbs', 0))} 个",
        f"- 自然下降：{int(route.get('descents', 0))} 个",
        f"- 预计移动时间：{format_duration(float(prediction['moving_time_seconds']))}",
        f"- 补给时间：{format_duration(float(prediction['aid_time_seconds']))}",
        f"- 最终预测：**{format_duration(float(prediction['total_time_seconds']))}**",
        "",
        "## 分段预测",
        "",
        "| 公里 | 地形 | 距离 | 平均坡度 | 最大坡度 | 爬升/下降 | 疲劳因子 | 预测时间 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in prediction["segments"]:
        lines.append(
            f"| {float(row['start_km']):.1f}-{float(row['end_km']):.1f} | "
            f"{row.get('terrain', row['type'])} | {float(row['distance']) / 1000.0:.2f} km | "
            f"{float(row['grade']):.1f}% | {float(row.get('max_grade', row['grade'])):.1f}% | "
            f"+{float(row['gain']):.0f}/-{float(row['loss']):.0f} m | "
            f"{float(row['fatigue_factor']) * 100:.0f}% | "
            f"{format_duration(float(row['predicted_time_seconds']))} |"
        )
    lines.extend(
        [
            "",
            "> 说明：这是基于历史运动表现和路线海拔的移动时间估算。天气、路况、技术难度、补给和停留需另行修正。",
            "",
        ]
    )
    return "\n".join(lines)


def save_markdown_report(report: str, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")


def _sample_pace(sample: dict[str, object], fallback_speed_mps: float | None = None) -> str:
    distance_km = float(sample.get("distance_km", 0) or 0)
    duration_hour = float(sample.get("duration_hour", 0) or 0)
    if distance_km > 0 and duration_hour > 0:
        return f"{format_pace(duration_hour * 3600.0 / distance_km)}/km"
    if fallback_speed_mps and fallback_speed_mps > 0:
        return f"{format_pace(1000.0 / fallback_speed_mps)}/km"
    return "—"


def _fatigue_row(time_range: str, retained_ratio: float) -> str:
    safe_ratio = max(retained_ratio, 0.1)
    return f"| {time_range} | {retained_ratio * 100:.0f}% | ×{1.0 / safe_ratio:.2f} |"
