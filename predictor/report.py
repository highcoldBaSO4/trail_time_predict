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
        f"- 平路能力可信度：{float(flat.get('confidence', 0.2)):.0%}",
        f"- 合格自然平路样本：{int(flat.get('qualified_segments', 0))} 段，"
        f"共 {float(flat.get('sample_distance_km', 0)):.2f} km",
        "",
        "### 上坡能力",
        "",
        "| 档位 | 平均坡度 | 等效配速 | VAM | 可信度 | 历史样本 | 累计距离 | 累计高度 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    uphill_rows = (
        ("微坡", ">2%–5%", "1_percent"),
        ("缓坡", "5%–10%", "5_percent"),
        ("中坡", "10%–15%", "10_percent"),
        ("陡坡", "≥15%", "15_percent"),
    )
    for name, grade_range, key in uphill_rows:
        sample = uphill.get("_samples", {}).get(key, {})
        curve_point = next((point for point in uphill.get("curve", []) if point.get("grade") == {"1_percent": 3.0, "5_percent": 7.5, "10_percent": 12.5, "15_percent": 18.0}[key]), {})
        lines.append(
            f"| {name} | {grade_range} | {_sample_pace(sample)} | {float(uphill[key]):.0f} m/h | "
            f"{float(curve_point.get('confidence', 0.2)):.0%} | "
            f"{int(sample.get('segments', 0))} 段 | {float(sample.get('distance_km', 0)):.2f} km | "
            f"+{float(sample.get('vertical_m', 0)):.0f} m |"
        )
    lines.extend(
        [
            "",
            "### 下坡能力",
            "",
            "| 档位 | 平均坡度 | 等效配速 | VAM | 可信度 | 历史样本 | 累计距离 | 累计高度 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    downhill_rows = (
        ("微下降", "-2%～-5%", "-1_percent"),
        ("缓下降", "-5%～-10%", "-5_percent"),
        ("中下降", "-10%～-15%", "-10_percent"),
        ("陡下降", "≤-15%", "-15_percent"),
    )
    for name, grade_range, key in downhill_rows:
        ability = downhill[key]
        sample = downhill.get("_samples", {}).get(key, {})
        curve_point = next((point for point in downhill.get("curve", []) if point.get("grade") == {"-1_percent": -3.0, "-5_percent": -7.5, "-10_percent": -12.5, "-15_percent": -18.0}[key]), {})
        speed = float(ability["speed_mps"])
        lines.append(
            f"| {name} | {grade_range} | {_sample_pace(sample, speed) } | "
            f"-{float(ability['vertical_speed_mph']):.0f} m/h | {float(curve_point.get('confidence', 0.2)):.0%} | {int(sample.get('segments', 0))} 段 | "
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
    lines.extend(["", "### 地形归一化连续疲劳曲线", "", "| 地形 | 时间 | 能力保留 | 可信度 |", "| --- | ---: | ---: | ---: |"])
    for terrain, label in (("flat", "平路"), ("uphill", "上坡"), ("downhill", "下坡")):
        for point in fatigue.get(terrain, []):
            confidence = "—（固定基准）" if point.get("source") == "anchor" else f"{float(point.get('confidence', 0.2)):.0%}"
            lines.append(f"| {label} | {float(point['hour']):g}h | {float(point['factor']):.0%} | {confidence} |")
    quality = profile.get("data_quality", {})
    lines.extend(["", "### 数据质量", "", f"- 综合评分：{float(quality.get('score', 0.2)):.0%}",
                  f"- 建议用于建模：{int(quality.get('recommended_count', 0))}/{int(profile.get('sample_count', 0))} 个活动"])
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
        f"- 标准能力移动时间：{format_duration(float(prediction.get('standard_moving_time_seconds', prediction['moving_time_seconds'])))}",
        f"- 状态与条件修正后移动时间：{format_duration(float(prediction.get('adjusted_moving_time_seconds', prediction['moving_time_seconds'])))}",
        f"- 补给时间：{format_duration(float(prediction['aid_time_seconds']))}",
        f"- 最快合理时间 P10：{format_duration(float(prediction.get('optimistic_time_seconds', prediction['total_time_seconds'])))}",
        f"- 最终预测（中位 P50）：**{format_duration(float(prediction.get('median_finish_time_seconds', prediction['total_time_seconds'])))}**",
        f"- 保守预测时间 P90：{format_duration(float(prediction.get('conservative_time_seconds', prediction['total_time_seconds'])))}",
        f"- 预测可信度：{float(prediction.get('confidence', 0.2)):.0%}",
        "",
        "### 目标时长能力匹配",
        "",
        f"- 迭代预计时长：{float(prediction.get('duration_match', {}).get('estimated_hours', 0)):.2f} 小时",
        f"- 是否收敛：{'是' if prediction.get('duration_match', {}).get('converged', True) else '否'}",
        "",
        "| 地形 | 持续能力层权重 | 耗时系数 | 可信度 | 来源 |",
        "| --- | --- | ---: | ---: | --- |",
        "## 分段预测",
        "",
        "| 公里 | 地形 | 距离 | 平均坡度 | 最大坡度 | 爬升/下降 | 时长适配 | 疲劳因子 | 条件系数 | 预测时间 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    terrain_labels = {"flat": "平路", "uphill": "上坡", "downhill": "下坡"}
    duration_terrain = prediction.get("duration_match", {}).get("terrain", {})
    duration_rows = []
    for terrain in ("flat", "uphill", "downhill"):
        match = duration_terrain.get(terrain, {})
        weights = "、".join(f"{name} {float(weight):.0%}" for name, weight in match.get("weights", {}).items()) or "—"
        duration_rows.append(f"| {terrain_labels[terrain]} | {weights} | ×{float(match.get('factor', 1)):.3f} | {float(match.get('confidence', .2)):.0%} | {match.get('source', 'legacy')} |")
    insertion = lines.index("## 分段预测")
    lines[insertion:insertion] = duration_rows + ["", "### 时间损耗拆解", "", "| 项目 | 时间影响 |", "| --- | ---: |"] + [
        f"| {_breakdown_label(key)} | {format_duration(float(value))} |" for key, value in prediction.get("adjustment_breakdown", {}).items()
    ] + [""]
    for row in prediction["segments"]:
        lines.append(
            f"| {float(row['start_km']):.1f}-{float(row['end_km']):.1f} | "
            f"{row.get('terrain', row['type'])} | {float(row['distance']) / 1000.0:.2f} km | "
            f"{float(row['grade']):.1f}% | {float(row.get('max_grade', row['grade'])):.1f}% | "
            f"+{float(row['gain']):.0f}/-{float(row['loss']):.0f} m | "
            f"×{float(row.get('duration_factor', 1)):.3f} | "
            f"{float(row['fatigue_factor']) * 100:.0f}% | "
            f"×{float(row.get('condition_factor', 1)):.3f} | "
            f"{format_duration(float(row['predicted_time_seconds']))} |"
        )
    lines.extend(
        [
            "",
            "> 说明：标准能力时间表示正常状态和正常条件下的移动时间；最终区间进一步考虑用户填写的状态、天气、路况、技术难度、负重、补给及模型不确定性。",
            "",
        ]
    )
    return "\n".join(lines)


def _breakdown_label(key: str) -> str:
    return {"base_terrain": "基础地形耗时", "duration_adaptation": "目标时长适配", "fatigue": "疲劳增加",
            "form": "当前状态", "technical": "技术难度", "mud": "泥泞", "night": "夜间",
            "altitude": "高海拔", "carried_weight": "装备负重", "weather": "温湿度",
            "aid_station": "补给停留"}.get(key, key)


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
