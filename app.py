from __future__ import annotations

import json
import hashlib
import tempfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analysis.capability import build_runner_profile
from analysis.activity_selection import ACTIVITY_TYPE_LABELS, LABEL_TO_ACTIVITY_TYPE, apply_activity_review, build_activity_review
from analysis.data_quality import diagnose_gpx
from analysis.temperature import calibrate_activity_temperature
from analysis.weather import enrich_activity_with_historical_weather
from models import RaceCondition
from parser.fit_reader import read_fit
from parser.gpx_reader import build_race_segments, read_gpx
from predictor.race_predictor import format_duration, format_pace, predict_race
from predictor.report import build_markdown_report


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HEART_RATE_GRADE_BANDS = (
    ("flat", "flat", "平路", "±2%"),
    ("uphill", "uphill_2_5", "微坡", ">2%～5%"),
    ("uphill", "uphill_5_10", "缓坡", "5%～10%"),
    ("uphill", "uphill_10_15", "中坡", "10%～15%"),
    ("uphill", "uphill_15_plus", "陡坡", "≥15%"),
    ("downhill", "downhill_2_5", "微下降", "-2%～-5%"),
    ("downhill", "downhill_5_10", "缓下降", "-5%～-10%"),
    ("downhill", "downhill_10_15", "中下降", "-10%～-15%"),
    ("downhill", "downhill_15_plus", "陡下降", "≤-15%"),
)
HEART_RATE_INTENSITIES = (
    ("easy", "轻松"),
    ("aerobic", "有氧"),
    ("steady", "稳态"),
    ("threshold", "阈值"),
    ("high", "高强度"),
)

st.set_page_config(
    page_title="越野跑比赛时间预测",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --trail-green: #16833d;
            --trail-green-dark: #0f6a31;
            --trail-orange: #f56b0a;
            --trail-text: #17202a;
            --trail-muted: #667085;
            --trail-border: #dfe4ea;
            --trail-bg: #f5f7f9;
        }
        .stApp { background: var(--trail-bg); color: var(--trail-text); }
        [data-testid="stHeader"] { background: rgba(245, 247, 249, .92); }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--trail-border);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
        .block-container { max-width: 1500px; padding-top: 1.35rem; padding-bottom: 3rem; }
        h1, h2, h3 { color: var(--trail-text); letter-spacing: -.02em; }
        h1 { font-size: 1.85rem !important; font-weight: 720 !important; margin-bottom: .2rem !important; }
        h2 { font-size: 1.25rem !important; }
        h3 { font-size: 1.05rem !important; }
        .app-subtitle { color: var(--trail-muted); font-size: .95rem; margin-bottom: 1.3rem; }
        .sidebar-title { font-size: 1.1rem; font-weight: 720; color: var(--trail-text); margin-bottom: .15rem; }
        .sidebar-copy { color: var(--trail-muted); font-size: .82rem; line-height: 1.55; margin-bottom: 1.15rem; }
        .step-label { color: var(--trail-text); font-size: .92rem; font-weight: 680; margin: 1.2rem 0 .45rem; }
        .step-number {
            display: inline-flex; width: 1.45rem; height: 1.45rem; align-items: center;
            justify-content: center; margin-right: .5rem; border-radius: 50%;
            background: var(--trail-green); color: #fff; font-size: .78rem;
        }
        .result-kicker { color: var(--trail-muted); font-size: .85rem; margin-bottom: .1rem; }
        .result-time { color: var(--trail-orange); font-size: 2.65rem; line-height: 1.08; font-weight: 760; letter-spacing: -.04em; }
        .empty-state {
            min-height: 420px; display: flex; flex-direction: column; align-items: center;
            justify-content: center; text-align: center; border: 1px dashed #cbd3dc;
            background: #fff; border-radius: 10px; padding: 3rem;
        }
        .empty-mark { font-size: 2.8rem; margin-bottom: .6rem; filter: grayscale(.25); }
        .empty-title { font-size: 1.22rem; font-weight: 700; margin-bottom: .4rem; }
        .empty-copy { color: var(--trail-muted); max-width: 520px; line-height: 1.65; }
        [data-testid="stMetric"] {
            background: #fff; border: 1px solid var(--trail-border); border-radius: 8px;
            padding: .85rem 1rem;
        }
        [data-testid="stMetricLabel"] { color: var(--trail-muted); }
        [data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 700; }
        [data-testid="stFileUploaderDropzone"] {
            background: #f8fafb; border: 1px dashed #b9c4cf; border-radius: 8px;
        }
        .stButton > button[kind="primary"] {
            background: var(--trail-green); border-color: var(--trail-green); font-weight: 680;
            min-height: 2.8rem;
        }
        .stButton > button[kind="primary"]:hover {
            background: var(--trail-green-dark); border-color: var(--trail-green-dark);
        }
        .stDownloadButton > button { min-height: 2.65rem; font-weight: 650; }
        [data-baseweb="tab-list"] { gap: 1.25rem; border-bottom: 1px solid var(--trail-border); }
        [data-baseweb="tab"] { padding: .75rem .15rem; font-weight: 650; }
        [aria-selected="true"][data-baseweb="tab"] { color: var(--trail-green); }
        [data-testid="stDataFrame"] { border: 1px solid var(--trail-border); border-radius: 8px; overflow: hidden; }
        div[data-testid="stExpander"] { background: #fff; border-color: var(--trail-border); }
        @media (max-width: 900px) {
            .block-container { padding-top: 1rem; }
            .result-time { font-size: 2.1rem; }
            [data-testid="stMetricValue"] { font-size: 1.15rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def step_heading(number: int, text: str) -> None:
    st.markdown(
        f'<div class="step-label"><span class="step-number">{number}</span>{text}</div>',
        unsafe_allow_html=True,
    )


def calculate_prediction(
    activities: dict[str, pd.DataFrame],
    activity_types: dict[str, str],
    gpx_file: Any,
    sample_distance: float,
    aid_minutes: float,
    condition: RaceCondition | None = None,
    simulations: int = 3000,
) -> dict[str, Any]:
    total_steps = 4
    completed = 0
    progress = st.progress(0, text="准备建立能力画像……")
    with st.status("正在建立个人能力画像", expanded=True) as status:
        status.write(f"使用已确认的 {len(activities)} 个活动识别自然爬坡、下降和平路……")
        profile = build_runner_profile(
            activities,
            activity_types,
            progress=lambda message: status.write(message.strip()),
        )
        completed += 1
        progress.progress(completed / total_steps, text="个人能力画像已生成")

        status.write("解析比赛路线并识别自然坡段……")
        with tempfile.TemporaryDirectory() as temp_dir:
            gpx_path = Path(temp_dir) / "race.gpx"
            gpx_path.write_bytes(gpx_file.getvalue())
            points = read_gpx(gpx_path)
            gpx_quality = diagnose_gpx(points)
            segments = build_race_segments(points, sample_distance)
        completed += 1
        progress.progress(completed / total_steps, text=f"已识别 {len(segments)} 个自然地形段")

        status.write("匹配个人能力并计算逐段时间……")
        prediction = predict_race(profile, segments, aid_minutes, condition=condition,
                                  simulations=simulations, gpx_quality_score=float(gpx_quality["score"]))
        prediction["gpx_data_quality"] = gpx_quality
        completed += 1
        progress.progress(completed / total_steps, text="逐段预测已完成")

        report = build_markdown_report(profile, prediction)
        completed += 1
        progress.progress(1.0, text="报告生成完成")
        status.update(label="计算完成", state="complete", expanded=False)

    return {"profile": profile, "prediction": prediction, "report": report}


def upload_signature(files: list[Any]) -> str:
    """Identify the selected upload set so stale reviews are never reused."""
    digest = hashlib.sha256()
    for uploaded in files:
        digest.update(uploaded.name.encode("utf-8", errors="replace"))
        digest.update(uploaded.getvalue())
    return digest.hexdigest()


def parse_activity_uploads(files: list[Any]) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Parse FIT files, attach historical weather, then prepare review rows."""
    names = [uploaded.name for uploaded in files]
    if len(names) != len(set(names)):
        raise ValueError("存在同名 FIT 文件，请重命名后重新上传")
    activities: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    progress = st.progress(0, text="准备解析活动并匹配历史天气……")
    with st.status("正在解析活动、匹配历史天气并检查数据质量", expanded=True) as status:
        for index, uploaded in enumerate(files, start=1):
            detail = st.empty()
            try:
                parsed = read_fit(
                    uploaded,
                    progress=lambda message, target=detail: target.write(message.strip()),
                )
                calibrated = calibrate_activity_temperature(parsed)
                activities[uploaded.name] = enrich_activity_with_historical_weather(
                    calibrated,
                    uploaded.name,
                    progress=lambda message, target=detail: target.write(message.strip()),
                )
            except ValueError as exc:
                failures.append(f"{uploaded.name}：{exc}")
                detail.error(f"已跳过 {uploaded.name}：{exc}")
            progress.progress(index / len(files), text=f"已解析并匹配天气 {index}/{len(files)} 个活动")
        if not activities:
            raise ValueError("上传的 FIT 文件均无法用于活动确认")
        status.update(label=f"活动解析及天气匹配完成：成功 {len(activities)} 个，失败 {len(failures)} 个", state="complete")
    for failure in failures:
        st.warning(failure)
    return activities, build_activity_review(activities)


def render_activity_review(rows: list[dict[str, Any]], editor_key: str) -> list[dict[str, Any]]:
    """Render the simplified road/trail review table and return normalized rows."""
    display = pd.DataFrame(
        [
            {
                "用于建模": row["use_for_model"],
                "文件名": row["filename"],
                "日期": row["date"],
                "距离": row["distance_km"],
                "时长": row["duration_hour"],
                "爬升": row["elevation_gain_m"],
                "自动类型": ACTIVITY_TYPE_LABELS[row["auto_type"]],
                "确认类型": ACTIVITY_TYPE_LABELS[row["confirmed_type"]],
                "质量": row["quality_level"],
                "心率覆盖": f"{float(row.get('heart_rate_coverage', 0.0)):.0%}",
                "环境温度覆盖": f"{float(row.get('temperature_coverage', 0.0)):.0%}",
                "腕表温度覆盖": f"{float(row.get('device_temperature_coverage', 0.0)):.0%}",
                "温度区间": row.get("temperature_range", "无有效温度"),
                "问题": "；".join(row["quality_issues"]) or "无",
            }
            for row in rows
        ]
    )
    edited = st.data_editor(
        display,
        hide_index=True,
        width="stretch",
        key=editor_key,
        disabled=["文件名", "日期", "距离", "时长", "爬升", "自动类型", "质量", "心率覆盖", "环境温度覆盖", "腕表温度覆盖", "温度区间", "问题"],
        column_config={
            "用于建模": st.column_config.CheckboxColumn("用于建模", help="取消后该 FIT 不参与能力画像"),
            "确认类型": st.column_config.SelectboxColumn("确认类型", options=["越野", "路跑"], required=True),
            "距离": st.column_config.NumberColumn("距离（km）", format="%.2f"),
            "时长": st.column_config.NumberColumn("时长（h）", format="%.2f"),
            "爬升": st.column_config.NumberColumn("爬升（m）", format="%.0f"),
        },
    )
    original = {str(row["filename"]): row for row in rows}
    normalized: list[dict[str, Any]] = []
    for record in edited.to_dict("records"):
        source = dict(original[str(record["文件名"])])
        source["use_for_model"] = bool(record["用于建模"])
        source["confirmed_type"] = LABEL_TO_ACTIVITY_TYPE[str(record["确认类型"])]
        normalized.append(source)
    selected_count = sum(bool(row["use_for_model"]) for row in normalized)
    not_recommended_count = sum(not bool(row["use_for_model"]) for row in rows)
    st.caption(
        f"已选择 {selected_count}/{len(normalized)} 个活动用于建模；"
        f"系统初始判定不建议建模 {not_recommended_count} 个。"
    )
    return normalized


def elevation_figure(segments: list[dict[str, Any]]) -> plt.Figure:
    distances = [0.0]
    elevations = [0.0]
    for segment in segments:
        distances.append(float(segment["end_km"]))
        elevations.append(elevations[-1] + float(segment["gain"]) - float(segment["loss"]))
    figure, axis = plt.subplots(figsize=(12, 3.6))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.plot(distances, elevations, color="#f56b0a", linewidth=2.35)
    axis.fill_between(distances, elevations, min(elevations), color="#f56b0a", alpha=.10)
    axis.set_xlabel("距离（km）", color="#667085")
    axis.set_ylabel("相对海拔（m）", color="#667085")
    axis.grid(axis="y", color="#e8ebef", linewidth=.8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#dfe4ea")
    axis.tick_params(colors="#667085", labelsize=9)
    figure.tight_layout()
    return figure


def ability_table(profile: dict[str, Any], direction: str) -> pd.DataFrame:
    if direction == "uphill":
        labels = (("微坡", ">2%～5%", "1_percent"), ("缓坡", "5%～10%", "5_percent"),
                  ("中坡", "10%～15%", "10_percent"), ("陡坡", "≥15%", "15_percent"))
        ability = profile["uphill"]
        rows = []
        for index, (name, grade, key) in enumerate(labels):
            sample = ability.get("_samples", {}).get(key, {})
            point = ability.get("curve", [{}] * 4)[index]
            pace = _sample_pace(sample)
            rows.append([name, grade, pace, f"{float(ability[key]):.0f} m/h", f"{float(point.get('confidence', .2)):.0%}", int(sample.get("segments", 0)),
                         f"{float(sample.get('distance_km', 0)):.2f} km", f"+{float(sample.get('vertical_m', 0)):.0f} m"])
    else:
        labels = (("微下降", "-2%～-5%", "-1_percent"), ("缓下降", "-5%～-10%", "-5_percent"),
                  ("中下降", "-10%～-15%", "-10_percent"), ("陡下降", "≤-15%", "-15_percent"))
        ability = profile["downhill"]
        rows = []
        for index, (name, grade, key) in enumerate(labels):
            sample = ability.get("_samples", {}).get(key, {})
            point = ability.get("curve", [{}] * 4)[index]
            pace = _sample_pace(sample, float(ability[key]["speed_mps"]))
            rows.append([name, grade, pace, f"-{float(ability[key]['vertical_speed_mph']):.0f} m/h", f"{float(point.get('confidence', .2)):.0%}",
                         int(sample.get("segments", 0)), f"{float(sample.get('distance_km', 0)):.2f} km",
                         f"-{float(sample.get('vertical_m', 0)):.0f} m"])
    return pd.DataFrame(rows, columns=["档位", "平均坡度", "等效配速", "VAM", "可信度", "样本", "累计距离", "累计高度"])


def _sample_pace(sample: dict[str, Any], fallback_speed: float | None = None) -> str:
    distance = float(sample.get("distance_km", 0) or 0)
    duration = float(sample.get("duration_hour", 0) or 0)
    if distance > 0 and duration > 0:
        return f"{format_pace(duration * 3600 / distance)}/km"
    return f"{format_pace(1000 / fallback_speed)}/km" if fallback_speed else "—"


def _mean_confidence(value: Any) -> float:
    values = value if isinstance(value, list) else [value]
    numeric = [float(item) for item in values if item is not None]
    return sum(numeric) / len(numeric) if numeric else 0.2


def _display_bpm_range(value: Any) -> str:
    return (
        f"{float(value[0]):.0f}～{float(value[1]):.0f} bpm"
        if isinstance(value, (list, tuple)) and len(value) == 2
        else "数据不足"
    )


def _display_hr_output(entry: dict[str, Any], field: str) -> str:
    output = float(entry.get(field, 0.0))
    if output <= 0:
        return "—"
    if entry.get("output_unit") == "vertical_metres_per_hour":
        return f"{output:.0f} m/h"
    return f"{format_pace(1000.0 / output)}/km"


def _temperature_source_label(value: object) -> str:
    return {
        "personal_blend": "个人与默认加权",
        "default": "系统默认",
        "node_default": "节点样本不足，使用默认",
        "comfort_anchor": "10～20℃最佳区间基准",
    }.get(str(value), str(value))


def _pace_from_speed(value: object) -> str:
    speed = float(value or 0.0)
    return "—" if speed <= 0 else f"{format_pace(1000.0 / speed)}/km"


def _vertical_speed_text(value: object, terrain: str) -> str:
    if value is None or terrain == "flat":
        return "—"
    speed = float(value)
    return f"{speed:+.0f} m/h" if terrain == "downhill" else f"{speed:.0f} m/h"


def render_result(result: dict[str, Any]) -> None:
    prediction = result["prediction"]
    profile = result["profile"]
    route = prediction["route"]

    header_left, download_md, download_json = st.columns([2.2, 1, 1])
    with header_left:
        st.markdown('<div class="result-kicker">预计完赛时间</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result-time">{format_duration(float(prediction["median_finish_time_seconds"]))}</div>',
            unsafe_allow_html=True,
        )
    with download_md:
        st.download_button(
            "下载 Markdown 报告",
            result["report"],
            "trail_prediction_report.md",
            "text/markdown",
            width="stretch",
        )
    with download_json:
        st.download_button(
            "下载预测 JSON",
            json.dumps(prediction, ensure_ascii=False, indent=2),
            "trail_prediction.json",
            "application/json",
            width="stretch",
        )

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("最快合理 P10", format_duration(float(prediction["optimistic_time_seconds"])))
    metric2.metric("中位预测 P50", format_duration(float(prediction["median_finish_time_seconds"])))
    metric3.metric("保守预测 P90", format_duration(float(prediction["conservative_time_seconds"])))
    metric4.metric("预测可信度", f"{float(prediction['confidence']):.0%}")

    st.subheader("路线海拔剖面")
    figure = elevation_figure(prediction["segments"])
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    overview_tab, ability_tab, segment_tab, report_tab = st.tabs(
        ["预测概览", "个人能力", "自然坡分段", "完整报告"]
    )
    with overview_tab:
        route_metric1, route_metric2, route_metric3, route_metric4 = st.columns(4)
        route_metric1.metric("比赛总距离", f"{float(route.get('distance_km', 0)):.2f} km")
        route_metric2.metric("累计爬升", f"{float(route.get('elevation_gain', 0)):.0f} m")
        route_metric3.metric("累计下降", f"{float(route.get('elevation_loss', 0)):.0f} m")
        route_metric4.metric(
            "自然坡数量",
            f"{int(route.get('climbs', 0))} 爬 / {int(route.get('descents', 0))} 降",
        )
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("时间构成")
            overview = pd.DataFrame(
                [
                    ["预计移动时间", format_duration(float(prediction["moving_time_seconds"]))],
                    ["标准能力移动时间", format_duration(float(prediction["standard_moving_time_seconds"]))],
                    ["条件修正后移动时间", format_duration(float(prediction["adjusted_moving_time_seconds"]))],
                    ["补给与停留", format_duration(float(prediction["aid_time_seconds"]))],
                    ["最终中位预测 P50", format_duration(float(prediction["median_finish_time_seconds"]))],
                ],
                columns=["项目", "结果"],
            )
            st.dataframe(overview, hide_index=True, width="stretch")
        with col2:
            st.subheader("路线结构")
            route_data = pd.DataFrame(
                [
                    ["自然爬坡", f"{int(route.get('climbs', 0))} 个"],
                    ["自然下降", f"{int(route.get('descents', 0))} 个"],
                    ["平均海拔", "—" if route.get("average_elevation_m") is None else f"{float(route['average_elevation_m']):.0f} m"],
                    ["最高海拔", "—" if route.get("maximum_elevation_m") is None else f"{float(route['maximum_elevation_m']):.0f} m"],
                ],
                columns=["项目", "结果"],
            )
            st.dataframe(route_data, hide_index=True, width="stretch")
        uncertainty = prediction["probability"].get("uncertainty", {})
        if uncertainty:
            st.subheader("概率区间依据")
            confidence_details = uncertainty.get("route_weighted_confidence", {})
            route_terrain_confidence = confidence_details.get("terrain", {})
            terrain_labels = {"flat": "平路", "uphill": "上坡", "downhill": "下坡"}
            uncertainty_rows = []
            for terrain in ("flat", "uphill", "downhill"):
                route_confidence = route_terrain_confidence.get(terrain, {})
                uncertainty_rows.append(
                    [
                        terrain_labels[terrain],
                        f"{float(uncertainty.get('terrain_time_share', {}).get(terrain, 0)):.1%}",
                        f"{float(route_confidence.get('ability', _mean_confidence(uncertainty.get('ability_confidence', {}).get(terrain, 0.2)))):.0%}",
                        f"{float(route_confidence.get('fatigue', _mean_confidence(uncertainty.get('fatigue_confidence', {}).get(terrain, [])))):.0%}",
                        f"{float(route_confidence.get('duration', uncertainty.get('duration_confidence', {}).get(terrain, 0.2))):.0%}",
                    ]
                )
            st.dataframe(
                pd.DataFrame(
                    uncertainty_rows,
                    columns=["地形", "预计耗时占比", "能力可信度", "疲劳可信度", "持续能力可信度"],
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                f"路线加权综合可信度 {float(confidence_details.get('overall', prediction.get('confidence', .2))):.0%} · "
                f"历史数据质量 {float(confidence_details.get('data_quality', .2)):.0%} · "
                f"GPX 质量 {float(confidence_details.get('gpx_quality', 1)):.0%}"
            )
            condition_labels = {
                "heart_rate_pacing": "心率强度配速", "form": "当前状态", "technical": "技术难度", "mud": "泥泞", "night": "夜间",
                "altitude": "高海拔", "carried_weight": "负重", "weather": "温度/湿度直接影响",
                "temperature_fatigue": "高温后程疲劳", "heart_rate_fatigue": "心率热应激疲劳",
            }
            condition_rows = [
                [condition_labels.get(source, source),
                 f"{float(values.get('active_time_share', 0)):.1%}",
                 f"{float(values.get('effective_sigma', 0)):.1%}"]
                for source, values in uncertainty.get("condition_sources", {}).items()
                if float(values.get("active_time_share", 0)) > 0
            ]
            if condition_rows:
                st.markdown("**条件不确定性来源**")
                st.dataframe(
                    pd.DataFrame(condition_rows, columns=["条件", "影响时间占比", "有效波动"]),
                    hide_index=True,
                    width="stretch",
                )
            gpx_uncertainty = uncertainty.get("gpx", {})
            st.caption(
                "GPX 使用坡段爬升/下降物理扰动："
                f"坡段影响占比 {float(gpx_uncertainty.get('affected_time_share', 0)):.1%}，"
                f"垂直误差波动 {float(gpx_uncertainty.get('vertical_sigma', uncertainty.get('gpx_sigma', 0))):.1%}。"
            )
        st.subheader("时间损耗拆解")
        labels = {"base_terrain": "基础地形", "heart_rate_pacing": "心率强度配速",
                  "duration_adaptation": "目标时长适配", "fatigue": "疲劳",
                  "form": "当前状态", "technical": "技术难度", "mud": "泥泞", "night": "夜间",
                  "altitude": "高海拔", "carried_weight": "负重", "weather": "温度/湿度直接影响",
                  "temperature_fatigue": "高温后程疲劳", "heart_rate_fatigue": "心率热应激疲劳",
                  "aid_station": "补给停留"}
        breakdown = pd.DataFrame([[labels.get(key, key), format_duration(float(value))]
                                  for key, value in prediction["adjustment_breakdown"].items()], columns=["项目", "时间影响"])
        st.dataframe(breakdown, hide_index=True, width="stretch")
        for note in prediction.get("risk_notes", []):
            st.warning(note)
        environment = prediction.get("environment", {})
        st.subheader("自动环境识别")
        environment_rows = [
            ["历史夜间占比", f"{float(environment.get('historical_night_ratio', 0)):.1%}"],
            ["历史夜间上坡占比", f"{float(environment.get('historical_night_by_terrain', {}).get('uphill', 0)):.1%}"],
            ["历史夜间下坡占比", f"{float(environment.get('historical_night_by_terrain', {}).get('downhill', 0)):.1%}"],
            ["比赛预计夜间占比", f"{float(environment.get('race_night_ratio', 0)):.1%}"],
            ["历史训练平均海拔", f"{float(environment.get('historical_mean_elevation_m', 0)):.0f} m"],
            ["历史训练P90海拔", f"{float(environment.get('historical_p90_elevation_m', 0)):.0f} m"],
            ["比赛平均海拔", "—" if environment.get("race_average_elevation_m") is None else f"{float(environment['race_average_elevation_m']):.0f} m"],
            ["比赛最高海拔", "—" if environment.get("race_maximum_elevation_m") is None else f"{float(environment['race_maximum_elevation_m']):.0f} m"],
        ]
        st.dataframe(pd.DataFrame(environment_rows, columns=["项目", "结果"]), hide_index=True, width="stretch")
        physiology = prediction.get("physiology", {})
        st.subheader("温度与心率响应")
        physiology_rows = [
            ["比赛温度", "未填写" if physiology.get("race_temperature_c") is None else f"{float(physiology['race_temperature_c']):.1f}℃"],
            ["温度模型", f"{physiology.get('temperature_model_source', 'unavailable')} / {float(physiology.get('temperature_model_confidence', .2)):.0%}"],
            ["温度直接系数", f"×{float(physiology.get('direct_temperature_factor', 1)):.3f}"],
            ["最大温度疲劳系数", f"×{float(physiology.get('maximum_temperature_fatigue_factor', 1)):.3f}"],
            ["心率模型", f"{physiology.get('heart_rate_model_source', 'unavailable')} / {float(physiology.get('heart_rate_model_confidence', .2)):.0%}"],
            ["比赛强度策略", physiology.get("pacing_strategy_label", "标准")],
            ["心率配速调整", "已应用个人模型" if physiology.get("heart_rate_pacing_applied") else "样本不足，保持标准能力"],
            ["目标心率", _display_bpm_range(physiology.get("target_hr_bpm_range"))],
            ["含漂移预计心率", _display_bpm_range(physiology.get("expected_hr_bpm_range"))],
            ["预计终点HR漂移", f"{float(physiology.get('expected_hr_drift_at_finish_bpm', 0)):+.1f} bpm"],
            ["最大心率热应激系数", f"×{float(physiology.get('maximum_heart_rate_fatigue_factor', 1)):.3f}"],
        ]
        st.dataframe(pd.DataFrame(physiology_rows, columns=["项目", "结果"]), hide_index=True, width="stretch")

    with ability_tab:
        flat = profile["flat"]
        flat1, flat2, flat3, flat4 = st.columns(4)
        flat1.metric("越野平路综合配速", f"{format_pace(float(flat['aerobic_pace']))}/km")
        flat2.metric("较快配速（P25）", f"{format_pace(float(flat['threshold_pace']))}/km")
        flat3.metric("自然平路样本", f"{int(flat.get('qualified_segments', 0))} 段")
        flat4.metric("平路可信度", f"{float(flat.get('confidence', .2)):.0%}")
        st.subheader("上坡能力")
        st.dataframe(ability_table(profile, "uphill"), hide_index=True, width="stretch")
        st.subheader("下坡能力")
        st.dataframe(ability_table(profile, "downhill"), hide_index=True, width="stretch")
        st.subheader("长时间疲劳衰减")
        fatigue = profile["fatigue"]
        fatigue_rows = []
        for time_range, key in (("0–3 小时", "3h"), ("3–5 小时", "5h"), ("5 小时以上", "8h")):
            retained = float(fatigue[key])
            fatigue_rows.append(
                [time_range, f"{retained * 100:.0f}%", f"×{1.0 / max(retained, 0.1):.2f}"]
            )
        st.dataframe(
            pd.DataFrame(fatigue_rows, columns=["累计移动时间", "能力保留比例", "耗时修正倍率"]),
            hide_index=True,
            width="stretch",
        )
        st.caption("预测按“分段基础耗时 ÷ 能力保留比例”应用疲劳修正。")
        st.subheader("地形归一化连续疲劳曲线")
        curve_rows = []
        for terrain, label in (("flat", "平路"), ("uphill", "上坡"), ("downhill", "下坡")):
            for point in fatigue.get(terrain, []):
                confidence = "—（固定基准）" if point.get("source") == "anchor" else f"{float(point.get('confidence', .2)):.0%}"
                curve_rows.append([label, f"{float(point['hour']):g}h", f"{float(point['factor']):.0%}", confidence])
        st.dataframe(pd.DataFrame(curve_rows, columns=["地形", "时间", "能力保留", "可信度"]), hide_index=True, width="stretch")
        temperature_profile = profile.get("temperature", {})
        st.subheader("个人温度耐受")
        temperature_coverage = temperature_profile.get("coverage", {})
        temperature_calibration = temperature_profile.get("calibration", {})
        best_range = temperature_profile.get("best_range_c", [])
        best_text = (
            f"{float(best_range[0]):g}～{float(best_range[1]):g}℃"
            if isinstance(best_range, list) and len(best_range) == 2 else "数据不足"
        )
        temp1, temp2, temp3 = st.columns(3)
        temp1.metric("最佳温度范围", best_text)
        temp2.metric("温度模型可信度", f"{float(temperature_profile.get('confidence', .2)):.0%}")
        temp3.metric("温度活动覆盖", f"{int(temperature_coverage.get('activity_count', 0))} 个")
        device_low = temperature_coverage.get("device_minimum_c")
        device_high = temperature_coverage.get("device_maximum_c")
        if temperature_calibration.get("source") == "wrist_relative_only":
            st.caption(
                f"FIT 腕表原始温度：{float(device_low):g}～{float(device_high):g}℃；"
                "该温度只保留为相对热暴露，不反推环境温度，不参与个人绝对耐热曲线；"
                "比赛气温使用用户填写值和系统默认温度曲线。"
            )
        elif temperature_calibration.get("source") == "historical_weather":
            st.caption(
                f"历史环境温度来自 {temperature_calibration.get('provider', 'Open-Meteo')}，"
                f"基础建模权重 {float(temperature_calibration.get('model_weight', .75)):.0%}；"
                "太阳辐射和腕表相对升温只用于降低局地暴露较强样本的权重，不会被当成环境绝对温度；"
                f"{temperature_calibration.get('spatial_resolution_note', '山区微气候可能存在偏差')}。"
            )
        temperature_rows = [
            [
                f"{float(point['temperature_c']):g}℃",
                f"×{float(point.get('default_time_factor', point['time_factor'])):.3f}",
                "—" if point.get("personal_time_factor") is None else f"×{float(point['personal_time_factor']):.3f}",
                f"×{float(point['time_factor']):.3f}",
                int(point.get("activity_count", 0)),
                f"{float(point.get('sample_duration_seconds', 0)) / 3600.0:.1f} h",
                f"{float(point.get('confidence', .2)):.0%}",
                f"{float(point.get('personal_weight', 0)):.0%}",
                _temperature_source_label(point.get("source", "default")),
            ]
            for point in temperature_profile.get("curve", [])
        ]
        if temperature_rows:
            st.dataframe(
                pd.DataFrame(
                    temperature_rows,
                    columns=["温度", "默认系数", "个人原始系数", "最终系数", "活动数", "有效时长", "节点可信度", "个人权重", "来源"],
                ),
                hide_index=True, width="stretch",
            )
        heart_rate_profile = profile.get("heart_rate", {})
        aerobic = heart_rate_profile.get("aerobic_range", {})
        threshold = heart_rate_profile.get("threshold", {})
        st.subheader("心率响应与漂移")
        hr1, hr2, hr3 = st.columns(3)
        aerobic_text = (
            "数据不足" if aerobic.get("low_bpm") is None or aerobic.get("high_bpm") is None
            else f"{float(aerobic['low_bpm']):.0f}～{float(aerobic['high_bpm']):.0f} bpm"
        )
        threshold_text = "数据不足" if threshold.get("bpm") is None else f"{float(threshold['bpm']):.0f} bpm"
        hr1.metric("有氧稳定心率", aerobic_text)
        hr2.metric("估算阈值心率", threshold_text)
        hr3.metric("心率模型可信度", f"{float(heart_rate_profile.get('confidence', .2)):.0%}")
        drift_rows = [
            [f"{float(point['hour']):g}h", f"{float(point.get('drift_bpm', 0)):+.1f} bpm",
             f"{float(point.get('output_retention', 1)):.0%}",
             "—" if point.get("confidence") is None else f"{float(point.get('confidence', .2)):.0%}"]
            for point in heart_rate_profile.get("drift", {}).get("overall", [])
        ]
        if drift_rows:
            st.dataframe(
                pd.DataFrame(drift_rows, columns=["时间", "HR漂移", "输出保留", "可信度"]),
                hide_index=True, width="stretch",
            )
        response_entries = {
            (terrain, str(entry.get("grade_band"))): entry
            for terrain in ("flat", "uphill", "downhill")
            for entry in heart_rate_profile.get("terrain_response", {}).get(terrain, [])
        }
        response_rows = []
        for terrain, band, label, grade_label in HEART_RATE_GRADE_BANDS:
            entry = response_entries.get((terrain, band))
            response_rows.append(
                [
                    label,
                    grade_label,
                    "—" if entry is None else _pace_from_speed(entry.get("median_speed_mps")),
                    "—" if entry is None else _vertical_speed_text(entry.get("vertical_speed_mph"), terrain),
                    "—" if entry is None else f"{float(entry.get('average_hr_bpm', 0)):.0f} bpm",
                    0 if entry is None else int(entry.get("activity_count", 0)),
                    "—" if entry is None else f"{float(entry.get('sample_duration_seconds', 0)) / 3600.0:.1f} h",
                    "20%" if entry is None else f"{float(entry.get('confidence', .2)):.0%}",
                    "数据不足" if entry is None else "自然坡个人数据",
                ]
            )
        if response_rows:
            st.markdown("**分地形心率成本**")
            st.dataframe(
                pd.DataFrame(
                    response_rows,
                    columns=["地形", "坡度", "等效配速", "VAM", "平均心率", "活动数", "有效时长", "可信度", "来源"],
                ),
                hide_index=True, width="stretch",
            )
        intensity_entries = {
            (terrain, str(entry.get("grade_band")), str(entry.get("intensity"))): entry
            for terrain in ("flat", "uphill", "downhill")
            for entry in heart_rate_profile.get("intensity_output", {}).get(terrain, [])
        }
        intensity_rows = []
        for terrain, band, label, grade_label in HEART_RATE_GRADE_BANDS:
            for intensity, intensity_label in HEART_RATE_INTENSITIES:
                entry = intensity_entries.get((terrain, band, intensity))
                intensity_rows.append(
                    [
                        label,
                        grade_label,
                        intensity_label,
                        "—" if entry is None else f"{float(entry.get('average_hr_bpm', 0)):.0f} bpm",
                        "—" if entry is None else _pace_from_speed(entry.get("median_speed_mps")),
                        "—" if entry is None else _vertical_speed_text(entry.get("median_vertical_speed_mph"), terrain),
                        "—" if entry is None else _pace_from_speed(entry.get("fast_speed_mps")),
                        "—" if entry is None else _vertical_speed_text(entry.get("fast_vertical_speed_mph"), terrain),
                        0 if entry is None else int(entry.get("activity_count", 0)),
                        "—" if entry is None else f"{float(entry.get('sample_duration_seconds', 0)) / 3600.0:.1f} h",
                        "20%" if entry is None else f"{float(entry.get('confidence', .2)):.0%}",
                        "数据不足" if entry is None else "自然坡个人数据",
                    ]
                )
        if intensity_rows:
            st.markdown("**心率强度—输出能力**")
            st.dataframe(
                pd.DataFrame(
                    intensity_rows,
                    columns=["地形", "坡度", "强度", "平均心率", "典型配速", "典型VAM", "偏快配速", "偏快VAM", "活动数", "有效时长", "可信度", "来源"],
                ),
                hide_index=True, width="stretch", height=520,
            )
        quality = profile.get("data_quality", {})
        st.caption(f"数据质量综合评分：{float(quality.get('score', .2)):.0%}；建议用于建模 {int(quality.get('recommended_count', 0))}/{int(profile.get('sample_count', 0))} 个活动。")

    with segment_tab:
        segment_frame = pd.DataFrame(prediction["segments"])
        display = segment_frame[
            ["start_km", "end_km", "terrain", "distance", "grade", "max_grade", "gain", "loss", "elevation", "duration_factor", "fatigue_factor", "condition_factor", "predicted_time_seconds"]
        ].copy()
        display["夜间"] = segment_frame["environment"].map(lambda value: "是" if value.get("night") else "否")
        display["海拔系数"] = segment_frame["environment"].map(lambda value: f"×{float(value.get('altitude_factor', 1)):.3f}")
        physiology_series = (
            segment_frame["physiology"]
            if "physiology" in segment_frame
            else pd.Series([{} for _ in range(len(segment_frame))], index=segment_frame.index)
        )
        display["温度疲劳"] = physiology_series.map(lambda value: f"×{float(value.get('temperature_fatigue_factor', 1)):.3f}")
        display["HR疲劳"] = physiology_series.map(lambda value: f"×{float(value.get('heart_rate_fatigue_factor', 1)):.3f}")
        display["强度"] = physiology_series.map(lambda value: value.get("pacing", {}).get("intensity_label", "—"))
        display["目标HR"] = physiology_series.map(
            lambda value: "—" if value.get("pacing", {}).get("target_hr_bpm") is None
            else f"{float(value['pacing']['target_hr_bpm']):.0f} bpm"
        )
        display["心率配速"] = physiology_series.map(lambda value: f"×{float(value.get('pacing', {}).get('time_factor', 1)):.3f}")
        display.columns = ["起点km", "终点km", "地形", "距离m", "平均坡度%", "最陡坡度%", "爬升m", "下降m", "海拔m", "时长适配", "疲劳因子", "条件系数", "预测秒", "夜间", "海拔系数", "温度疲劳", "HR疲劳", "强度", "目标HR", "心率配速"]
        display["时长适配"] = display["时长适配"].map(lambda factor: f"×{float(factor):.3f}")
        display["疲劳因子"] = display["疲劳因子"].map(lambda factor: f"{float(factor) * 100:.0f}%")
        display["条件系数"] = display["条件系数"].map(lambda factor: f"×{float(factor):.3f}")
        display["预测时间"] = display.pop("预测秒").map(lambda seconds: format_duration(float(seconds)))
        st.dataframe(display, hide_index=True, width="stretch", height=460)

    with report_tab:
        st.markdown(result["report"])
        st.download_button(
            "下载完整 Markdown 报告",
            result["report"],
            "trail_prediction_report.md",
            "text/markdown",
        )


inject_styles()

with st.sidebar:
    st.markdown('<div class="sidebar-title">预测工作流</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-copy">数据只在当前电脑中解析，不会上传到外部服务。</div>',
        unsafe_allow_html=True,
    )
    step_heading(1, "上传历史 Activity FIT")
    fit_files = st.file_uploader(
        "选择多个 .fit 文件",
        type=["fit"],
        accept_multiple_files=True,
        help="建议同时包含路跑、越野跑和长距离活动。",
    )
    if fit_files:
        st.caption(f"已选择 {len(fit_files)} 个 FIT 文件")
        signature = upload_signature(fit_files)
        if st.session_state.get("fit_upload_signature") != signature:
            st.session_state["fit_upload_signature"] = signature
            st.session_state.pop("parsed_activities", None)
            st.session_state.pop("activity_review_rows", None)
            st.session_state.pop("activity_review_signature", None)
            st.session_state.pop("prediction_result", None)
        if st.button("解析并确认活动", width="stretch", disabled=not fit_files):
            try:
                parsed, review_rows = parse_activity_uploads(fit_files)
                st.session_state["parsed_activities"] = parsed
                st.session_state["activity_review_rows"] = review_rows
            except Exception as exc:
                st.session_state.pop("parsed_activities", None)
                st.session_state.pop("activity_review_rows", None)
                st.error(f"活动解析失败：{exc}")
    else:
        st.session_state.pop("fit_upload_signature", None)
        st.session_state.pop("parsed_activities", None)
        st.session_state.pop("activity_review_rows", None)
        st.session_state.pop("activity_review_signature", None)
        st.session_state.pop("prediction_result", None)

st.markdown("# 越野跑比赛时间概率预测")
st.markdown(
    '<div class="app-subtitle">结合持续能力、当天状态和比赛条件，输出可解释的 P10–P90 完赛时间区间。</div>',
    unsafe_allow_html=True,
)

parsed_activities: dict[str, pd.DataFrame] = st.session_state.get("parsed_activities", {})
review_rows: list[dict[str, Any]] = st.session_state.get("activity_review_rows", [])
selected_activities: dict[str, pd.DataFrame] = {}
confirmed_types: dict[str, str] = {}
if parsed_activities and review_rows:
    with st.expander("活动确认与筛选", expanded="prediction_result" not in st.session_state):
        st.caption("系统先自动判断越野或路跑，并匹配可用的历史环境温度。请确认类型，并取消不希望进入能力画像的活动。")
        normalized_review = render_activity_review(
            review_rows,
            f"activity_review_{st.session_state.get('fit_upload_signature', 'none')}",
        )
    review_signature = tuple(
        (row["filename"], bool(row["use_for_model"]), row["confirmed_type"])
        for row in normalized_review
    )
    if st.session_state.get("activity_review_signature") != review_signature:
        st.session_state["activity_review_signature"] = review_signature
        st.session_state.pop("prediction_result", None)
    try:
        selected_activities, confirmed_types = apply_activity_review(parsed_activities, normalized_review)
    except ValueError as exc:
        st.warning(str(exc))

with st.sidebar:
    step_heading(2, "上传比赛路线 GPX")
    gpx_file = st.file_uploader("选择一个 .gpx 文件", type=["gpx"])

    step_heading(3, "分析参数")
    sample_distance = st.number_input(
        "坡度采样窗口（米）", min_value=50, max_value=500, value=100, step=50,
        help="窗口越小越敏感；默认100米适合大多数越野路线。",
    )
    aid_minutes = st.number_input(
        "预计补给与停留（分钟）", min_value=0, max_value=600, value=0, step=5,
    )
    st.markdown("**当前状态与比赛条件**")
    form_labels = {"状态很好": "very_good", "状态正常": "normal", "轻微疲劳": "slight_fatigue",
                   "状态较差": "poor", "生病或伤病": "ill_or_injured"}
    current_form_label = st.selectbox("当前状态", list(form_labels), index=1)
    pacing_labels = {"标准": "standard", "保守": "conservative", "积极": "aggressive"}
    pacing_label = st.selectbox(
        "比赛强度策略",
        list(pacing_labels),
        index=0,
        help="根据历史心率—输出关系选择可持续强度；积极策略只在有可靠样本时提高配速。",
    )
    temperature = st.number_input("温度（℃）", min_value=-20, max_value=60, value=20, step=1)
    humidity = st.number_input("湿度（%）", min_value=0, max_value=100, value=60, step=5)
    technical_level = st.select_slider("技术难度", options=[0, 1, 2, 3, 4], value=0,
                                       format_func=lambda value: ["接近平时训练", "略难", "更难", "明显更难", "极高技术"][value])
    mud_level = st.select_slider("相对平时的额外泥泞", options=[0, 1, 2, 3, 4], value=0)
    carried_weight = st.number_input("比平时额外携带重量（kg）", min_value=0.0, max_value=20.0, value=0.0, step=0.5)
    st.markdown("**自动夜间与海拔分析**")
    race_date = st.date_input("比赛日期", value=date.today())
    race_start_clock = st.time_input("出发时间", value=time(8, 0), step=900)
    utc_offset = st.selectbox(
        "比赛时区",
        options=list(range(-12, 15)),
        index=20,
        format_func=lambda value: f"UTC{value:+d}",
        help="用于将比赛当地出发时间转换为太阳位置计算所需的绝对时间。",
    )
    st.caption("夜间路段由比赛时间和路线经纬度自动识别；海拔修正由历史 FIT 与比赛 GPX 自动计算。")
    simulations = st.select_slider("模拟次数", options=[1000, 3000, 5000, 10000], value=3000)

    step_heading(4, "开始计算")
    can_calculate = bool(selected_activities and gpx_file)
    if st.button("开始计算", type="primary", disabled=not can_calculate, width="stretch"):
        try:
            local_start = datetime.combine(race_date, race_start_clock).replace(
                tzinfo=timezone(timedelta(hours=int(utc_offset)))
            )
            st.session_state["prediction_result"] = calculate_prediction(
                selected_activities, confirmed_types, gpx_file, float(sample_distance), float(aid_minutes),
                RaceCondition(current_form=form_labels[current_form_label],
                              pacing_strategy=pacing_labels[pacing_label], temperature_c=float(temperature),
                              humidity_percent=float(humidity), altitude_factor=1.0,
                              terrain_technical_level=int(technical_level), mud_level=int(mud_level),
                              night_running_ratio=0.0,
                              carried_weight_kg=float(carried_weight), aid_station_minutes=float(aid_minutes),
                              race_start_time_utc=local_start.astimezone(timezone.utc)),
                int(simulations),
            )
        except Exception as exc:
            st.session_state.pop("prediction_result", None)
            st.error(f"计算失败：{exc}")
    if not can_calculate:
        st.caption("解析并确认至少一个 FIT，同时上传 GPX 后即可开始计算。")

if "prediction_result" in st.session_state:
    render_result(st.session_state["prediction_result"])
elif parsed_activities:
    st.info("活动已解析。确认上方活动类型和建模选择后，在左侧上传 GPX 并开始计算。")
else:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-mark">⌁</div>
            <div class="empty-title">上传数据，开始一次路线预测</div>
            <div class="empty-copy">
                在左侧选择历史 FIT，点击“解析并确认活动”。确认越野/路跑类型后上传比赛 GPX，
                系统将建立个人能力画像并生成概率预测报告。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
