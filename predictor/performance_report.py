from __future__ import annotations

from predictor.race_predictor import format_duration


TERRAIN_LABELS = {"flat": "平路", "uphill": "上坡", "downhill": "下坡"}


def build_performance_report(result: dict[str, object]) -> str:
    diagnosis = dict(result["diagnosis"])
    prediction_range = dict(diagnosis.get("prediction_range", {}))
    sign = "+" if float(diagnosis["deviation_seconds"]) >= 0 else "-"
    lines = [
        "# 活动表现诊断报告",
        "",
        f"- 活动：{diagnosis['activity_name']}",
        f"- 实际经过时间：{format_duration(float(diagnosis['actual_elapsed_seconds']))}",
        f"- FIT 计时时间：{format_duration(float(diagnosis['actual_timer_seconds']))}",
        f"- 估算移动时间：{format_duration(float(diagnosis['actual_moving_seconds']))}",
        f"- 暂停或未计时：{format_duration(float(diagnosis.get('paused_seconds', 0)))}",
        f"- 计时中的停留：{format_duration(float(diagnosis.get('nonmoving_timer_seconds', 0)))}",
        f"- 模型预测移动时间：{format_duration(float(diagnosis['predicted_moving_seconds']))}",
        f"- 移动时间偏差：{sign}{format_duration(abs(float(diagnosis['deviation_seconds'])))}（{float(diagnosis['deviation_percent']):+.1f}%）",
        f"- 实际表现百分位：P{float(diagnosis['prediction_percentile']):.0f}",
        f"- 判断：{diagnosis.get('performance_label', '—')}",
        f"- 诊断可信度：{float(diagnosis['confidence']):.0%}",
        "",
        "## 预测区间",
        "",
        f"- P10：{format_duration(float(prediction_range.get('p10_seconds', 0)))}",
        f"- P50：{format_duration(float(prediction_range.get('p50_seconds', 0)))}",
        f"- P90：{format_duration(float(prediction_range.get('p90_seconds', 0)))}",
        "",
        "## 分地形表现",
        "",
        "| 地形 | 预测时间 | 实际时间 | 偏差 | 偏差比例 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for terrain in ("flat", "uphill", "downhill"):
        values = dict(diagnosis.get("terrain_analysis", {}).get(terrain, {}))
        lines.append(
            f"| {TERRAIN_LABELS[terrain]} | {format_duration(float(values.get('predicted_seconds', 0)))} | "
            f"{format_duration(float(values.get('actual_seconds', 0)))} | "
            f"{format_duration(float(values.get('deviation_seconds', 0)))} | "
            f"{float(values.get('deviation_percent', 0)):+.1f}% |"
        )
    lines.extend(
        [
            "",
            "## 前后半程表现",
            "",
            "| 阶段 | 预测时间 | 实际时间 | 偏差比例 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key, label in (("first_half", "前半程"), ("second_half", "后半程")):
        values = dict(diagnosis.get("progress_analysis", {}).get(key, {}))
        lines.append(
            f"| {label} | {format_duration(float(values.get('predicted_seconds', 0)))} | "
            f"{format_duration(float(values.get('actual_seconds', 0)))} | "
            f"{float(values.get('deviation_percent', 0)):+.1f}% |"
        )
    lines.extend(
        [
            "",
            "> 时间口径：经过时间来自 FIT session 或时间戳跨度；计时时间来自 FIT timer；移动时间使用 15 秒局部窗口识别，避免慢速爬坡的距离量化漏掉耗时。",
            "",
            "> 当前为 V0.4 Phase 1 基础表现对比。技术难度、泥泞、负重和当天主观状态未知时保持中性，未解释偏差不会被强行归因。",
            "",
        ]
    )
    return "\n".join(lines)
