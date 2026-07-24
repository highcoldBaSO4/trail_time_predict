from __future__ import annotations

from pathlib import Path

from predictor.race_predictor import format_duration, format_pace

HEART_RATE_GRADE_BANDS = (
    ("flat", "flat", "平路", "±2%"),
    ("uphill", "uphill_2_5", "微坡", ">2%～5%"),
    ("uphill", "uphill_5_10", "缓坡", "5%～10%"),
    ("uphill", "uphill_10_15", "中坡", "10%～15%"),
    ("uphill", "uphill_15_20", "较陡坡", "15%～20%"),
    ("uphill", "uphill_20_plus", "陡坡", "≥20%"),
    ("downhill", "downhill_2_5", "微下降", "-2%～-5%"),
    ("downhill", "downhill_5_10", "缓下降", "-5%～-10%"),
    ("downhill", "downhill_10_15", "中下降", "-10%～-15%"),
    ("downhill", "downhill_15_20", "较陡下降", "-15%～-20%"),
    ("downhill", "downhill_20_plus", "陡下降", "≤-20%"),
)
HEART_RATE_INTENSITIES = (("easy", "轻松"), ("aerobic", "有氧"), ("steady", "稳态"),
                          ("threshold", "阈值"), ("high", "高强度"))


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
    movement = profile.get("movement_detection", {})
    movement_rows = []
    for activity in profile.get("activities", []):
        detail = activity.get("movement_detection", {})
        movement_rows.append(
            f"| {activity.get('name', '—')} | {format_duration(float(detail.get('moving_seconds', 0)))} | "
            f"{format_duration(float(detail.get('recovered_zero_distance_seconds', 0)))} |"
        )
    if movement_rows:
        insertion = lines.index("### 上坡能力")
        lines[insertion:insertion] = [
            "### FIT 移动时间识别",
            "",
            f"零距离量化时间共补回 {format_duration(float(movement.get('recovered_zero_distance_seconds', 0)))}。"
            "仅补回短间隔，或有步频/功率证据且前后距离继续增长的区间；较长无运动信号区间按停留处理。",
            "",
            "| FIT | 识别移动时间 | 补回的零距离时间 |",
            "| --- | ---: | ---: |",
            *movement_rows,
            "",
        ]
    uphill_rows = (
        ("微坡", ">2%–5%", "1_percent"),
        ("缓坡", "5%–10%", "5_percent"),
        ("中坡", "10%–15%", "10_percent"),
        ("较陡坡", "15%～20%", "15_percent"),
        ("陡坡", "≥20%", "20_percent"),
    )
    for name, grade_range, key in uphill_rows:
        sample = uphill.get("_samples", {}).get(key, {})
        curve_point = next((point for point in uphill.get("curve", []) if point.get("grade") == {"1_percent": 3.0, "5_percent": 7.5, "10_percent": 12.5, "15_percent": 17.5, "20_percent": 22.5}[key]), {})
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
        ("较陡下降", "-15%～-20%", "-15_percent"),
        ("陡下降", "≤-20%", "-20_percent"),
    )
    for name, grade_range, key in downhill_rows:
        ability = downhill[key]
        sample = downhill.get("_samples", {}).get(key, {})
        curve_point = next((point for point in downhill.get("curve", []) if point.get("grade") == {"-1_percent": -3.0, "-5_percent": -7.5, "-10_percent": -12.5, "-15_percent": -17.5, "-20_percent": -22.5}[key]), {})
        speed = float(ability["speed_mps"])
        lines.append(
            f"| {name} | {grade_range} | {_sample_pace(sample, speed) } | "
            f"-{float(ability['vertical_speed_mph']):.0f} m/h | {float(curve_point.get('confidence', 0.2)):.0%} | {int(sample.get('segments', 0))} 段 | "
            f"{float(sample.get('distance_km', 0)):.2f} km | -{float(sample.get('vertical_m', 0)):.0f} m |"
        )
    lines.extend(
        [
            "",
            "### 长时间疲劳衰减（按地形与证据）",
            "",
            "| 地形 | 节点 | 能力保留 | 耗时修正倍率 | 来源 | 证据 | 可信度 |",
            "| --- | ---: | ---: | ---: | --- | --- | ---: |",
        ]
    )
    for terrain, label in (("flat", "平路"), ("uphill", "上坡"), ("downhill", "下坡")):
        for point in fatigue.get(terrain, []):
            source = str(point.get("source", "default"))
            node = f"{float(point['hour']):g}h" + ("（外推）" if source == "extrapolated" else "")
            retained = float(point["factor"])
            confidence = "—" if source == "anchor" else f"{float(point.get('confidence', 0.2)):.0%}"
            lines.append(
                f"| {label} | {node} | {retained:.1%} | ×{1.0 / max(retained, 0.1):.2f} | "
                f"{_source_label(source)} | {_fatigue_evidence_label(point)} | {confidence} |"
            )
    lines.extend([
        "",
        "> 疲劳修正规则：分段基础耗时 ÷ 能力保留比例。系统先验表示该节点缺少个人证据；保守外推用于超过个人最长观测时长的比赛后程。",
    ])
    stages = profile.get("fatigue_stages", {})
    base_sampling = profile.get("base_ability_sampling", {})
    overall_stages = stages.get("overall", {})
    lines.extend([
        "",
        "### 疲劳阶段与基础能力取样",
        "",
        f"- 新鲜阶段结束（97%能力保留）：{float(overall_stages.get('fresh_end_hour', 3)):.2f}h",
        f"- 轻度疲劳结束（90%能力保留）：{float(overall_stages.get('mild_end_hour', 5)):.2f}h",
        f"- 中度疲劳结束（80%能力保留）：{float(overall_stages.get('moderate_end_hour', 8)):.2f}h",
        f"- 基础能力来源：{int(base_sampling.get('selected_activity_count', 0))} 条完整落在新鲜阶段的活动；"
        "长距离活动继续用于比赛策略和疲劳，不按全程数据量主导基础能力。",
        "",
    ])
    temperature = profile.get("temperature", {})
    temperature_coverage = temperature.get("coverage", {})
    temperature_calibration = temperature.get("calibration", {})
    lines.extend(
        [
            "",
            "### 个人温度耐受",
            "",
            f"- 模型来源：{_source_label(temperature.get('source'))}",
            f"- 模型可信度：{float(temperature.get('confidence', 0.2)):.0%}",
            f"- 最佳温度范围：{_temperature_range(temperature.get('best_range_c'))}",
            f"- 历史温度覆盖：{_temperature_coverage(temperature_coverage)}",
            f"- 腕表原始温度覆盖：{_device_temperature_coverage(temperature_coverage)}",
            f"- 温度数据处理：{_temperature_calibration_text(temperature_calibration)}",
            f"- 绝对温度基础建模权重：{float(temperature_calibration.get('model_weight', 0)):.0%}",
            "",
            "| 温度 | 默认系数 | 个人原始系数 | 最终系数 | 活动数 | 有效时长 | 节点可信度 | 个人权重 | 来源 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for point in temperature.get("curve", []):
        personal_factor = point.get("personal_time_factor")
        personal_factor_text = "—" if personal_factor is None else f"×{float(personal_factor):.3f}"
        lines.append(
            f"| {float(point['temperature_c']):g}℃ | ×{float(point.get('default_time_factor', point['time_factor'])):.3f} | "
            f"{personal_factor_text} | "
            f"×{float(point['time_factor']):.3f} | {int(point.get('activity_count', 0))} | "
            f"{float(point.get('sample_duration_seconds', 0)) / 3600.0:.1f}h | "
            f"{float(point.get('confidence', 0.2)):.0%} | {float(point.get('personal_weight', 0)):.0%} | "
            f"{_source_label(point.get('source'))} |"
        )
    heart_rate = profile.get("heart_rate", {})
    aerobic = heart_rate.get("aerobic_range", {})
    threshold = heart_rate.get("threshold", {})
    heat_sensitivity = heart_rate.get("heat_sensitivity", {})
    lines.extend(
        [
            "",
            "### 心率响应与漂移",
            "",
            f"- 模型来源：{_source_label(heart_rate.get('source'))}",
            f"- 模型可信度：{float(heart_rate.get('confidence', 0.2)):.0%}",
            f"- 有氧稳定心率范围：{_bpm_range(aerobic.get('low_bpm'), aerobic.get('high_bpm'))}",
            f"- 估算阈值心率：{_bpm_value(threshold.get('bpm'))}",
            f"- 个人热应激响应：{float(heat_sensitivity.get('bpm_per_degree', 0.3)):.2f} bpm/℃"
            f"（{_source_label(heat_sensitivity.get('source'))}）",
            "",
            "| 时间 | HR漂移 | 输出保留 | 可信度 |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for point in heart_rate.get("drift", {}).get("overall", []):
        confidence = "—" if point.get("confidence") is None else f"{float(point.get('confidence', 0.2)):.0%}"
        lines.append(
            f"| {float(point['hour']):g}h | {float(point.get('drift_bpm', 0)):+.1f} bpm | "
            f"{float(point.get('output_retention', 1)):.0%} | {confidence} |"
        )
    lines.extend(
        [
            "",
            "#### 分地形心率成本",
            "",
            "| 地形 | 坡度 | 等效配速 | VAM | 平均心率 | 活动数 | 有效时长 | 可信度 | 来源 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    response_entries = {
        (terrain, str(entry.get("grade_band"))): entry
        for terrain in ("flat", "uphill", "downhill")
        for entry in heart_rate.get("terrain_response", {}).get(terrain, [])
    }
    for terrain, band, label, grade_label in HEART_RATE_GRADE_BANDS:
        entry = response_entries.get((terrain, band))
        lines.append(
            f"| {label} | {grade_label} | {_hr_pace(entry)} | {_hr_vam(entry, terrain)} | "
            f"{_hr_bpm(entry)} | "
            f"{0 if entry is None else int(entry.get('activity_count', 0))} | "
            f"{_hr_duration(entry)} | {_hr_confidence(entry)} | "
            f"{'数据不足' if entry is None else '自然坡个人数据'} |"
        )
    lines.extend(
        [
            "",
            "#### 心率强度—输出能力",
            "",
            "| 地形 | 坡度 | 强度 | 平均心率 | 典型配速 | 典型VAM | 偏快配速 | 偏快VAM | 活动数 | 有效时长 | 可信度 | 来源 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    intensity_entries = {
        (terrain, str(entry.get("grade_band")), str(entry.get("intensity"))): entry
        for terrain in ("flat", "uphill", "downhill")
        for entry in heart_rate.get("intensity_output", {}).get(terrain, [])
    }
    for terrain, band, label, grade_label in HEART_RATE_GRADE_BANDS:
        for intensity, intensity_label in HEART_RATE_INTENSITIES:
            entry = intensity_entries.get((terrain, band, intensity))
            lines.append(
                f"| {label} | {grade_label} | {intensity_label} | "
                f"{_hr_bpm(entry)} | "
                f"{_hr_pace(entry, 'median_speed_mps')} | {_hr_vam(entry, terrain, 'median_vertical_speed_mph')} | "
                f"{_hr_pace(entry, 'fast_speed_mps')} | {_hr_vam(entry, terrain, 'fast_vertical_speed_mph')} | "
                f"{0 if entry is None else int(entry.get('activity_count', 0))} | "
                f"{_hr_duration(entry)} | {_hr_confidence(entry)} | "
                f"{'数据不足' if entry is None else '自然坡个人数据'} |"
            )
    quality = profile.get("data_quality", {})
    lines.extend(["", "### 数据质量", "", f"- 综合评分：{float(quality.get('score', 0.2)):.0%}",
                  f"- 建议用于建模：{int(quality.get('recommended_count', 0))}/{int(profile.get('sample_count', 0))} 个活动"])
    historical_environment = profile.get("environment", {})
    historical_night = historical_environment.get("night", {})
    night_terrain = historical_night.get("terrain", {})
    historical_altitude = historical_environment.get("altitude", {})
    lines.extend(
        [
            "",
            "### 历史环境覆盖",
            "",
            f"- 历史夜间运动占比：{float(historical_night.get('ratio', 0)):.1%}",
            f"- 历史夜间上坡占比：{float(night_terrain.get('uphill', {}).get('ratio', 0)):.1%}",
            f"- 历史夜间下坡占比：{float(night_terrain.get('downhill', {}).get('ratio', 0)):.1%}",
            f"- 历史训练平均海拔：{float(historical_altitude.get('mean_m', 0)):.0f} m",
            f"- 历史训练 P90 海拔：{float(historical_altitude.get('p90_m', 0)):.0f} m",
            f"- 历史训练最高海拔：{float(historical_altitude.get('max_m', 0)):.0f} m",
        ]
    )
    race_environment = prediction.get("environment", {})
    physiology = prediction.get("physiology", {})
    prediction_calibration = dict(prediction.get("calibration", {}))
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
        f"- 比赛强度策略：{physiology.get('pacing_strategy_label', '标准')}",
        f"- 心率配速调整：{'已应用个人模型' if physiology.get('heart_rate_pacing_applied') else '样本不足，保持标准能力'}",
        f"- 目标心率范围：{_bpm_list_range(physiology.get('target_hr_bpm_range'))}",
        f"- 含漂移预计心率范围：{_bpm_list_range(physiology.get('expected_hr_bpm_range'))}",
        f"- 比赛预计夜间占比：{float(race_environment.get('race_night_ratio', 0)):.1%}",
        f"- 环境到达时间计算：{'条件总时长迭代收敛' if race_environment.get('arrival_time_converged') else '条件总时长迭代未完全收敛，已保守保留最后一次结果'}"
        f"（{int(race_environment.get('arrival_time_iterations', 0))} 次）",
        f"- 比赛平均海拔：{_format_elevation(race_environment.get('race_average_elevation_m'))}",
        f"- 比赛最高海拔：{_format_elevation(race_environment.get('race_maximum_elevation_m'))}",
        f"- 比赛温度曲线：{_temperature_schedule_text(physiology.get('race_temperature_schedule', {}))}",
        f"- 温度模型：{_source_label(physiology.get('temperature_model_source'))}，"
        f"可信度 {float(physiology.get('temperature_model_confidence', 0.2)):.0%}",
        f"- 温度直接耗时系数：×{float(physiology.get('direct_temperature_factor', 1)):.3f}",
        f"- 后程最大温度疲劳系数：×{float(physiology.get('maximum_temperature_fatigue_factor', 1)):.3f}",
        f"- 心率热应激最大附加系数：×{float(physiology.get('maximum_heart_rate_fatigue_factor', 1)):.3f}",
        f"- 预计终点HR漂移：{float(physiology.get('expected_hr_drift_at_finish_bpm', 0)):+.1f} bpm",
        "",
        "### 历史比赛配速策略匹配",
        "",
        f"- 迭代预计时长：{float(prediction.get('duration_match', {}).get('estimated_hours', 0)):.2f} 小时",
        f"- 是否收敛：{'是' if prediction.get('duration_match', {}).get('converged', True) else '否'}",
        "## 分段预测",
        "",
        "| 公里 | 地形 | 距离 | 平均坡度 | 强度 | 目标HR | 目标配速 | 心率配速 | 海拔 | 昼夜 | 爬升/下降 | 路线配速策略 | 疲劳因子 | 温度疲劳 | HR疲劳 | 海拔系数 | 条件系数 | 预测时间 |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if prediction_calibration:
        calibration_lines = [
            "",
            "### 历史回测校准",
            "",
            f"- 状态：{prediction_calibration.get('status', '未启用')}；高质量无泄漏回测：{int(prediction_calibration.get('valid_backtest_count', 0))} 条。",
            f"- P50：{format_duration(float(prediction_calibration.get('p50_before_seconds', prediction.get('median_finish_time_seconds', 0))))}"
            f" → {format_duration(float(prediction_calibration.get('p50_after_seconds', prediction.get('median_finish_time_seconds', 0))))}"
            f"（×{float(prediction_calibration.get('p50_factor', 1.0)):.3f}）。",
            f"- P10/P90：{format_duration(float(prediction_calibration.get('p10_before_seconds', prediction.get('optimistic_time_seconds', 0))))}"
            f"～{format_duration(float(prediction_calibration.get('p90_before_seconds', prediction.get('conservative_time_seconds', 0))))}"
            f" → {format_duration(float(prediction_calibration.get('p10_after_seconds', prediction.get('optimistic_time_seconds', 0))))}"
            f"～{format_duration(float(prediction_calibration.get('p90_after_seconds', prediction.get('conservative_time_seconds', 0))))}。",
            f"- 区间来源：{prediction_calibration.get('interval_source', '未启用')}。",
            f"- 说明：{prediction_calibration.get('note', '未使用历史回测校准。')}",
        ]
        extra_reasons = list(prediction_calibration.get("interval_external_reasons", []))
        if extra_reasons:
            calibration_lines.append("- 额外保守放宽：" + "、".join(str(item) for item in extra_reasons) + "。")
        lines[lines.index("### 历史比赛配速策略匹配"):lines.index("### 历史比赛配速策略匹配")] = calibration_lines
    terrain_labels = {"flat": "平路", "uphill": "上坡", "downhill": "下坡"}
    strategy_match = prediction.get("pacing_strategy_match", {})
    target = strategy_match.get("target", {})
    strategy_labels = {"negative_split": "负分割", "positive_split": "正分割", "even": "均匀", "variable": "波动", "duration_fallback": "时长层回退"}
    strategy_source = "历史相似路线" if strategy_match.get("source") == "historical_route_strategy" else "旧时长能力层回退"
    duration_rows = [
        f"- 目标路线：{float(target.get('distance_km', 0)):.1f} km / +{float(target.get('elevation_gain_m', 0)):.0f} m，"
        f"爬升密度 {float(target.get('climb_density_m_per_km', 0)):.0f} m/km",
        f"- 匹配来源：{strategy_source}；策略类型：{strategy_labels.get(strategy_match.get('strategy_type'), strategy_match.get('strategy_type', '—'))}；"
        f"可信度 {float(strategy_match.get('confidence', .2)):.0%}",
        "",
        "| 地形 | 前25% | 25%–50% | 50%–75% | 后25% |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    structure = dict(target.get("route_structure", {}))
    grade_bands = dict(structure.get("grade_bands", {}))
    continuous = dict(structure.get("continuous", {}))
    phases = dict(structure.get("phase_distribution", {}))
    sequence = dict(structure.get("sequence", {}))
    if grade_bands or continuous or phases:
        duration_rows += ["", "#### 目标路线结构摘要", ""]
        if grade_bands:
            duration_rows += [
                "| 坡度结构 | 里程占比 | 垂直量占比 |",
                "| --- | ---: | ---: |",
                *[
                    f"| +{threshold}% 及以上上坡 | {float(grade_bands.get(f'uphill_{threshold}_distance_share', 0)):.1%} | "
                    f"{float(grade_bands.get(f'uphill_{threshold}_gain_share', 0)):.1%} 爬升 |"
                    for threshold in (10, 15, 20)
                ],
                *[
                    f"| {threshold}% 及以下下坡 | {float(grade_bands.get(f'downhill_{abs(threshold)}_distance_share', 0)):.1%} | "
                    f"{float(grade_bands.get(f'downhill_{abs(threshold)}_loss_share', 0)):.1%} 下降 |"
                    for threshold in (-10, -15, -20)
                ],
                "",
            ]
        if continuous:
            duration_rows += [
                f"- 最长连续上坡：{float(continuous.get('longest_uphill_distance_km', 0)):.2f} km / "
                f"+{float(continuous.get('longest_uphill_gain_m', 0)):.0f} m / {float(continuous.get('longest_uphill_average_grade_pct', 0)):.1f}% ，"
                f"位于赛程 {float(continuous.get('longest_uphill_start_progress', 0)):.0%}–{float(continuous.get('longest_uphill_end_progress', 0)):.0%}；"
                f"最大单次爬升 +{float(continuous.get('maximum_single_ascent_m', 0)):.0f} m",
                f"- 最长连续下坡：{float(continuous.get('longest_downhill_distance_km', 0)):.2f} km / "
                f"-{float(continuous.get('longest_downhill_loss_m', 0)):.0f} m / {float(continuous.get('longest_downhill_average_grade_pct', 0)):.1f}% ，"
                f"位于赛程 {float(continuous.get('longest_downhill_start_progress', 0)):.0%}–{float(continuous.get('longest_downhill_end_progress', 0)):.0%}；"
                f"最大单次下降 -{float(continuous.get('maximum_single_descent_m', 0)):.0f} m",
            ]
        if phases:
            phase_labels = (("first_25", "前25%"), ("second_25", "25%–50%"), ("third_25", "50%–75%"), ("last_25", "后25%"))
            duration_rows += ["", "| 比赛进程 | 爬升占比 | 下降占比 | 陡坡爬升占比 |", "| --- | ---: | ---: | ---: |"]
            duration_rows += [
                f"| {label} | {float(dict(phases.get(name, {})).get('gain_share', 0)):.1%} | "
                f"{float(dict(phases.get(name, {})).get('loss_share', 0)):.1%} | "
                f"{float(dict(phases.get(name, {})).get('hard_uphill_gain_share', 0)):.1%} |"
                for name, label in phase_labels
            ]
        if sequence:
            duration_rows += [
                f"- 上坡后紧接 ≥1 km 下坡的上坡占比：{float(sequence.get('uphill_to_long_downhill_transition_share', 0)):.1%}；"
                f"地形切换频率：{float(sequence.get('terrain_run_count_per_10km', 0)):.1f} 段 / 10 km",
            ]
    for terrain in ("flat", "uphill", "downhill"):
        curve = list(strategy_match.get("terrain_curves", {}).get(terrain, [1.0] * 4))
        duration_rows.append(f"| {terrain_labels[terrain]} | " + " | ".join(f"×{float(value):.3f}" for value in curve) + " |")
    matched = list(strategy_match.get("matched_activities", []))
    if matched:
        duration_rows += ["", "| 匹配 FIT | 距离 | 爬升 | 历史策略 | 相似度 | 权重 |", "| --- | ---: | ---: | --- | ---: | ---: |"]
        duration_rows += [
            f"| {item.get('activity', '—')} | {float(item.get('distance_km', 0)):.1f} km | +{float(item.get('elevation_gain_m', 0)):.0f} m | "
            f"{strategy_labels.get(item.get('strategy_type'), item.get('strategy_type', '—'))} | {float(item.get('similarity', 0)):.0%} | {float(item.get('weight', 0)):.0%} |"
            for item in matched
        ]
    similarity_labels = {
        "scale": "路线规模", "grade_structure": "陡坡结构", "continuous_slope": "连续坡",
        "terrain_sequence": "坡序与后程分布", "activity_type": "活动类型可靠性",
    }
    similarity_explanations = {
        "scale": "距离、爬升、密度、负荷与地形占比", "grade_structure": "±10/15/20% 坡段占比",
        "continuous_slope": "最长连续爬升/下降与单次升降", "terrain_sequence": "四阶段坡段分布及上后接下坡",
        "activity_type": "越野活动优先于其他活动类型",
    }
    similarity_groups = dict(strategy_match.get("similarity_groups", {}))
    if similarity_groups:
        duration_rows += ["", "| 相似度维度 | 匹配度 | 比较依据 |", "| --- | ---: | --- |"]
        duration_rows += [
            f"| {similarity_labels.get(name, name)} | {float(score):.0%} | {similarity_explanations.get(name, '—')} |"
            for name, score in similarity_groups.items()
        ]
    match_reasons = list(strategy_match.get("similarity_reasons", []))
    missing_groups = list(strategy_match.get("missing_feature_groups", []))
    uncertainty_reasons = list(strategy_match.get("uncertainty", {}).get("reasons", []))
    if match_reasons or missing_groups or uncertainty_reasons:
        duration_rows += ["", "匹配与区间说明："]
        duration_rows += [f"- {reason}" for reason in match_reasons]
        if missing_groups:
            duration_rows.append("- 未参与结构比对的维度：" + "、".join(similarity_labels.get(name, name) for name in missing_groups) + "。")
        duration_rows += [f"- {reason}" for reason in uncertainty_reasons if reason not in match_reasons]
    insertion = lines.index("## 分段预测")
    probability_uncertainty = prediction.get("probability", {}).get("uncertainty", {})
    dynamic_environment = dict(probability_uncertainty.get("dynamic_environment", {}))
    confidence_details = probability_uncertainty.get("route_weighted_confidence", {})
    route_terrain_confidence = confidence_details.get("terrain", {})
    probability_rows = []
    terrain_labels = {"flat": "平路", "uphill": "上坡", "downhill": "下坡"}
    for terrain in ("flat", "uphill", "downhill"):
        route_confidence = route_terrain_confidence.get(terrain, {})
        probability_rows.append(
            f"| {terrain_labels[terrain]} | "
            f"{float(probability_uncertainty.get('terrain_time_share', {}).get(terrain, 0)):.1%} | "
            f"{float(route_confidence.get('ability', _mean_confidence(probability_uncertainty.get('ability_confidence', {}).get(terrain, 0.2)))):.0%} | "
            f"{float(route_confidence.get('fatigue', _mean_confidence(probability_uncertainty.get('fatigue_confidence', {}).get(terrain, [])))):.0%} | "
            f"{float(route_confidence.get('duration', probability_uncertainty.get('duration_confidence', {}).get(terrain, 0.2))):.0%} |"
        )
    condition_sources = probability_uncertainty.get("condition_sources", {})
    condition_rows = [
        f"| {_breakdown_label(source)} | {float(values.get('active_time_share', 0)):.1%} | "
        f"{float(values.get('effective_sigma', 0)):.1%} |"
        for source, values in condition_sources.items()
        if float(values.get("active_time_share", 0)) > 0
    ]
    gpx_uncertainty = probability_uncertainty.get("gpx", {})
    uncertainty_details = [
        "",
        f"路线加权综合可信度：{float(confidence_details.get('overall', prediction.get('confidence', .2))):.0%}；"
        f"历史数据质量：{float(confidence_details.get('data_quality', .2)):.0%}；"
        f"GPX 质量：{float(confidence_details.get('gpx_quality', 1)):.0%}"
        + (
            f"；温度/心率模型：{float(confidence_details['physiology_confidence']):.0%}。"
            if confidence_details.get("physiology_confidence") is not None else "。"
        ),
        "",
    ]
    if dynamic_environment.get("enabled"):
        temperature_text = "—" if dynamic_environment.get("mean_temperature_c") is None else (
            f"{float(dynamic_environment['mean_temperature_c']):.1f}℃"
            f"（P10–P90：{float(dynamic_environment.get('temperature_c_p10', dynamic_environment['mean_temperature_c'])):.1f}–"
            f"{float(dynamic_environment.get('temperature_c_p90', dynamic_environment['mean_temperature_c'])):.1f}℃）"
        )
        uncertainty_details += [
            "#### 动态到达时间环境",
            "",
            "每次 Monte Carlo 按该样本的累计到达时间重算温度、昼夜、高温后程疲劳和心率热应激；海拔与路线技术因素保持静态。",
            f"- 模拟平均温度：{temperature_text}；夜间占比 P10–P90："
            f"{float(dynamic_environment.get('night_ratio_p10', 0)):.1%}–{float(dynamic_environment.get('night_ratio_p90', 0)):.1%}。",
            "- 条件证据：" + "；".join(
                f"{label}{ {'user_input': '用户输入', 'route_confirmed': '路线可确认', 'unknown': '未知保守先验'}.get(mode, mode)}"
                for label, mode in (("温度/热疲劳：", dynamic_environment.get("sources", {}).get("weather", "unknown")),
                                    ("昼夜：", dynamic_environment.get("sources", {}).get("night", "unknown")),
                                    ("海拔：", dynamic_environment.get("sources", {}).get("altitude", "unknown")))
            ) + "。",
            "",
        ]
    if condition_rows:
        uncertainty_details += [
            "#### 条件不确定性来源",
            "",
            "| 条件 | 影响时间占比 | 有效波动 |",
            "| --- | ---: | ---: |",
            *condition_rows,
            "",
        ]
    uncertainty_details += [
        "#### GPX 几何误差",
        "",
        f"采用坡段爬升/下降扰动并重新计算坡度和基础时间；坡段影响占比 "
        f"{float(gpx_uncertainty.get('affected_time_share', 0)):.1%}，垂直误差波动 "
        f"{float(gpx_uncertainty.get('vertical_sigma', probability_uncertainty.get('gpx_sigma', 0))):.1%}。",
        "",
    ]
    lines[insertion:insertion] = duration_rows + [
        "",
        "### 概率区间依据",
        "",
        "Monte Carlo 按分地形能力和疲劳曲线逐段重算，并分别采样温度直接影响、高温后程疲劳与心率热应激；同一场模拟内同类来源保持相关。",
        "",
        "| 地形 | 预计耗时占比 | 能力可信度 | 疲劳可信度 | 配速策略可信度 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ] + probability_rows + uncertainty_details + ["### 时间损耗拆解", "", "| 项目 | 时间影响 |", "| --- | ---: |"] + [
        f"| {_breakdown_label(key)} | {format_duration(float(value))} |" for key, value in prediction.get("adjustment_breakdown", {}).items()
    ] + [""]
    for row in prediction["segments"]:
        pacing = row.get("physiology", {}).get("pacing", {})
        distance_km = float(row["distance"]) / 1000.0
        target_pace = (
            "—" if distance_km <= 0
            else f"{format_pace(float(row['predicted_time_seconds']) / distance_km)}/km"
        )
        lines.append(
            f"| {float(row['start_km']):.1f}-{float(row['end_km']):.1f} | "
            f"{row.get('terrain', row['type'])} | {distance_km:.2f} km | "
            f"{float(row['grade']):.1f}% | {pacing.get('intensity_label', '—')} | "
            f"{_bpm_value(pacing.get('target_hr_bpm'))} | {target_pace} | ×{float(pacing.get('time_factor', 1)):.3f} | "
            f"{_format_elevation(row.get('environment', {}).get('elevation_m'))} | "
            f"{'夜间' if row.get('environment', {}).get('night') else '白天'} | "
            f"+{float(row['gain']):.0f}/-{float(row['loss']):.0f} m | "
            f"×{float(row.get('duration_factor', 1)):.3f} | "
            f"{float(row['fatigue_factor']) * 100:.0f}% | "
            f"×{float(row.get('physiology', {}).get('temperature_fatigue_factor', 1)):.3f} | "
            f"×{float(row.get('physiology', {}).get('heart_rate_fatigue_factor', 1)):.3f} | "
            f"×{float(row.get('environment', {}).get('altitude_factor', 1)):.3f} | "
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
    return {"base_terrain": "基础地形耗时", "heart_rate_pacing": "心率强度配速", "duration_adaptation": "目标时长适配", "pacing_strategy": "比赛配速策略", "fatigue": "疲劳增加",
            "form": "当前状态", "technical": "技术难度", "mud": "泥泞", "night": "夜间",
            "altitude": "高海拔", "carried_weight": "装备负重", "weather": "温度/湿度直接影响",
            "temperature_fatigue": "高温后程疲劳", "heart_rate_fatigue": "心率热应激疲劳",
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


def _format_elevation(value: object) -> str:
    return "—" if value is None else f"{float(value):.0f} m"


def _mean_confidence(value: object) -> float:
    values = value if isinstance(value, list) else [value]
    numeric = [float(item) for item in values if item is not None]
    return sum(numeric) / len(numeric) if numeric else 0.2


def _source_label(value: object) -> str:
    return {
        "personal": "个人数据", "personal_blend": "个人与默认加权", "blended": "个人与先验混合",
        "default": "系统先验", "extrapolated": "保守外推",
        "node_default": "节点样本不足，使用默认", "comfort_anchor": "10～20℃最佳区间基准",
        "unavailable": "数据不足", "observed_stable_output": "历史稳定输出",
        "estimated_upper_output": "历史高输出估算", "anchor": "固定基准",
    }.get(str(value), str(value or "数据不足"))


def _fatigue_evidence_label(point: dict[str, object]) -> str:
    source = str(point.get("source", "default"))
    if source == "anchor":
        return "固定为 100%"
    if source == "extrapolated":
        observed = float(point.get("observed_max_hours", 0.0))
        distance = float(point.get("extrapolation_distance_hours", 0.0))
        return f"最长 {observed:g}h 后外推 {distance:g}h"
    count = int(point.get("observed_activity_count", point.get("sample_count", 0)))
    minutes = float(point.get("observed_duration_seconds", 0.0)) / 60.0
    return f"{count} 场 / {minutes:.0f} 分钟窗口" if count else "无个人节点证据"


def _temperature_value(value: object) -> str:
    return "未填写" if value is None else f"{float(value):.1f}℃"


def _temperature_schedule_text(value: object) -> str:
    if not isinstance(value, dict) or value.get("start_c") is None:
        return "未填写"
    start = float(value["start_c"])
    peak = value.get("peak_c")
    finish = value.get("finish_c")
    peak_hour = value.get("peak_hour")
    if peak is None and finish is None:
        return f"全程按 {start:.1f}℃"
    peak = start if peak is None else float(peak)
    finish = peak if finish is None else float(finish)
    when = "中途" if peak_hour is None else f"赛后 {float(peak_hour):g}h"
    return f"起跑 {start:.1f}℃ → {when} {peak:.1f}℃ → 终点 {finish:.1f}℃"


def _temperature_range(value: object) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return "数据不足"
    return f"{float(value[0]):g}～{float(value[1]):g}℃"


def _temperature_coverage(coverage: dict[str, object]) -> str:
    low, high = coverage.get("minimum_c"), coverage.get("maximum_c")
    return "无有效样本" if low is None or high is None else f"{float(low):g}～{float(high):g}℃，{int(coverage.get('activity_count', 0))}个活动"


def _device_temperature_coverage(coverage: dict[str, object]) -> str:
    low, high = coverage.get("device_minimum_c"), coverage.get("device_maximum_c")
    return "无腕表原始值" if low is None or high is None else f"{float(low):g}～{float(high):g}℃"


def _temperature_calibration_text(calibration: dict[str, object]) -> str:
    if calibration.get("source") == "historical_weather":
        return (
            f"使用 {calibration.get('provider', '历史天气服务')} 作为绝对气温基准；"
            "太阳辐射和腕表相对升温仅降低局地热暴露样本权重，不改变环境气温；"
            f"{calibration.get('spatial_resolution_note', '山区微气候可能存在偏差')}"
        )
    if calibration.get("source") == "wrist_relative_only":
        return "腕表温度仅保留相对变化，不反推环境温度，也不训练绝对耐热曲线"
    if calibration.get("source") != "ambient_assumed":
        return "没有可用的绝对环境温度"
    return "输入按环境温度处理"


def _bpm_range(low: object, high: object) -> str:
    return "数据不足" if low is None or high is None else f"{float(low):.0f}～{float(high):.0f} bpm"


def _bpm_value(value: object) -> str:
    return "数据不足" if value is None else f"{float(value):.0f} bpm"


def _heart_rate_output(entry: dict[str, object], field: str = "median_output") -> str:
    value = float(entry.get(field, entry.get("median_output", 0.0)))
    if entry.get("output_unit") == "vertical_metres_per_hour":
        return f"{value:.0f} m/h"
    return "—" if value <= 0 else f"{format_pace(1000.0 / value)}/km"


def _hr_pace(entry: dict[str, object] | None, field: str = "median_speed_mps") -> str:
    if entry is None:
        return "—"
    speed = float(entry.get(field, 0.0) or 0.0)
    return "—" if speed <= 0 else f"{format_pace(1000.0 / speed)}/km"


def _hr_vam(
    entry: dict[str, object] | None,
    terrain: str,
    field: str = "vertical_speed_mph",
) -> str:
    if entry is None or terrain == "flat":
        return "—"
    value = float(entry.get(field, 0.0) or 0.0)
    return f"{value:+.0f} m/h" if terrain == "downhill" else f"{value:.0f} m/h"


def _hr_bpm(entry: dict[str, object] | None) -> str:
    return "—" if entry is None else f"{float(entry.get('average_hr_bpm', 0)):.0f} bpm"


def _hr_duration(entry: dict[str, object] | None) -> str:
    return "—" if entry is None else f"{float(entry.get('sample_duration_seconds', 0)) / 3600.0:.1f}h"


def _hr_confidence(entry: dict[str, object] | None) -> str:
    return "20%" if entry is None else f"{float(entry.get('confidence', 0.2)):.0%}"


def _bpm_list_range(value: object) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return "数据不足"
    return f"{float(value[0]):.0f}～{float(value[1]):.0f} bpm"
