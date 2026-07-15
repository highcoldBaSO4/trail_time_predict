from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analysis.capability import build_runner_profile
from parser.fit_reader import read_fit
from parser.gpx_reader import build_race_segments, read_gpx
from predictor.race_predictor import format_duration, format_pace, predict_race
from predictor.report import build_markdown_report


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

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
    fit_files: list[Any],
    gpx_file: Any,
    sample_distance: float,
    aid_minutes: float,
) -> dict[str, Any]:
    total_steps = len(fit_files) + 4
    completed = 0
    progress = st.progress(0, text="准备读取文件……")
    with st.status("正在建立个人能力画像", expanded=True) as status:
        activities: dict[str, pd.DataFrame] = {}
        for index, uploaded in enumerate(fit_files, start=1):
            detail = st.empty()
            detail.write(f"解析 FIT {index}/{len(fit_files)}：{uploaded.name}")
            activities[uploaded.name] = read_fit(
                uploaded,
                progress=lambda message, target=detail: target.write(message.strip()),
            )
            completed += 1
            progress.progress(completed / total_steps, text=f"已完成 {completed}/{total_steps} 个步骤")

        status.write("识别历史活动中的自然爬坡、下降和平路……")
        profile = build_runner_profile(activities)
        completed += 1
        progress.progress(completed / total_steps, text="个人能力画像已生成")

        status.write("解析比赛路线并识别自然坡段……")
        with tempfile.TemporaryDirectory() as temp_dir:
            gpx_path = Path(temp_dir) / "race.gpx"
            gpx_path.write_bytes(gpx_file.getvalue())
            segments = build_race_segments(read_gpx(gpx_path), sample_distance)
        completed += 1
        progress.progress(completed / total_steps, text=f"已识别 {len(segments)} 个自然地形段")

        status.write("匹配个人能力并计算逐段时间……")
        prediction = predict_race(profile, segments, aid_minutes)
        completed += 1
        progress.progress(completed / total_steps, text="逐段预测已完成")

        report = build_markdown_report(profile, prediction)
        completed += 1
        progress.progress(1.0, text="报告生成完成")
        status.update(label="计算完成", state="complete", expanded=False)

    return {"profile": profile, "prediction": prediction, "report": report}


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
        labels = (("微坡", "1%～5%", "1_percent"), ("缓坡", "5%～10%", "5_percent"),
                  ("中坡", "10%～15%", "10_percent"), ("陡坡", "≥15%", "15_percent"))
        ability = profile["uphill"]
        rows = []
        for name, grade, key in labels:
            sample = ability.get("_samples", {}).get(key, {})
            pace = _sample_pace(sample)
            rows.append([name, grade, pace, f"{float(ability[key]):.0f} m/h", int(sample.get("segments", 0)),
                         f"{float(sample.get('distance_km', 0)):.2f} km", f"+{float(sample.get('vertical_m', 0)):.0f} m"])
    else:
        labels = (("微下降", "-1%～-5%", "-1_percent"), ("缓下降", "-5%～-10%", "-5_percent"),
                  ("中下降", "-10%～-15%", "-10_percent"), ("陡下降", "≤-15%", "-15_percent"))
        ability = profile["downhill"]
        rows = []
        for name, grade, key in labels:
            sample = ability.get("_samples", {}).get(key, {})
            pace = _sample_pace(sample, float(ability[key]["speed_mps"]))
            rows.append([name, grade, pace, f"-{float(ability[key]['vertical_speed_mph']):.0f} m/h",
                         int(sample.get("segments", 0)), f"{float(sample.get('distance_km', 0)):.2f} km",
                         f"-{float(sample.get('vertical_m', 0)):.0f} m"])
    return pd.DataFrame(rows, columns=["档位", "平均坡度", "等效配速", "VAM", "样本", "累计距离", "累计高度"])


def _sample_pace(sample: dict[str, Any], fallback_speed: float | None = None) -> str:
    distance = float(sample.get("distance_km", 0) or 0)
    duration = float(sample.get("duration_hour", 0) or 0)
    if distance > 0 and duration > 0:
        return f"{format_pace(duration * 3600 / distance)}/km"
    return f"{format_pace(1000 / fallback_speed)}/km" if fallback_speed else "—"


def render_result(result: dict[str, Any]) -> None:
    prediction = result["prediction"]
    profile = result["profile"]
    route = prediction["route"]

    header_left, download_md, download_json = st.columns([2.2, 1, 1])
    with header_left:
        st.markdown('<div class="result-kicker">预计完赛时间</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result-time">{format_duration(float(prediction["total_time_seconds"]))}</div>',
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
    metric1.metric("比赛距离", f"{float(route['distance_km']):.2f} km")
    metric2.metric("累计爬升", f"{float(route['elevation_gain']):.0f} m")
    metric3.metric("自然爬坡", f"{int(route.get('climbs', 0))} 个")
    metric4.metric("历史活动", f"{int(profile['sample_count'])} 个")

    st.subheader("路线海拔剖面")
    figure = elevation_figure(prediction["segments"])
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    overview_tab, ability_tab, segment_tab, report_tab = st.tabs(
        ["预测概览", "个人能力", "自然坡分段", "完整报告"]
    )
    with overview_tab:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("时间构成")
            overview = pd.DataFrame(
                [
                    ["预计移动时间", format_duration(float(prediction["moving_time_seconds"]))],
                    ["补给与停留", format_duration(float(prediction["aid_time_seconds"]))],
                    ["最终预测", format_duration(float(prediction["total_time_seconds"]))],
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
                    ["累计下降", f"{float(route['elevation_loss']):.0f} m"],
                ],
                columns=["项目", "结果"],
            )
            st.dataframe(route_data, hide_index=True, width="stretch")

    with ability_tab:
        flat = profile["flat"]
        flat1, flat2, flat3 = st.columns(3)
        flat1.metric("越野平路综合配速", f"{format_pace(float(flat['aerobic_pace']))}/km")
        flat2.metric("较快配速（P25）", f"{format_pace(float(flat['threshold_pace']))}/km")
        flat3.metric("自然平路样本", f"{int(flat.get('qualified_segments', 0))} 段")
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

    with segment_tab:
        segment_frame = pd.DataFrame(prediction["segments"])
        display = segment_frame[
            ["start_km", "end_km", "terrain", "distance", "grade", "max_grade", "gain", "loss", "fatigue_factor", "predicted_time_seconds"]
        ].copy()
        display.columns = ["起点km", "终点km", "地形", "距离m", "平均坡度%", "最陡坡度%", "爬升m", "下降m", "疲劳因子", "预测秒"]
        display["疲劳因子"] = display["疲劳因子"].map(lambda factor: f"{float(factor) * 100:.0f}%")
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

    step_heading(4, "开始计算")
    can_calculate = bool(fit_files and gpx_file)
    if st.button("开始计算", type="primary", disabled=not can_calculate, width="stretch"):
        try:
            st.session_state["prediction_result"] = calculate_prediction(
                fit_files, gpx_file, float(sample_distance), float(aid_minutes)
            )
        except Exception as exc:
            st.session_state.pop("prediction_result", None)
            st.error(f"计算失败：{exc}")
    if not can_calculate:
        st.caption("上传 FIT 和 GPX 后即可开始计算。")

st.markdown("# 越野跑比赛时间预测")
st.markdown(
    '<div class="app-subtitle">从历史活动建立个人地形能力画像，并按比赛自然坡段预测完成时间。</div>',
    unsafe_allow_html=True,
)

if "prediction_result" in st.session_state:
    render_result(st.session_state["prediction_result"])
else:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-mark">⌁</div>
            <div class="empty-title">上传数据，开始一次路线预测</div>
            <div class="empty-copy">
                在左侧选择历史 FIT 活动和比赛 GPX。系统会识别自然爬坡、下降和平路，
                建立个人能力画像，并生成可在线查看和下载的完整报告。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
