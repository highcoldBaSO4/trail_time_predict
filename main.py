from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from analysis.capability import build_runner_profile, save_runner_profile
from parser.fit_reader import read_fit_directory
from parser.gpx_reader import build_race_segments, read_gpx, save_segments
from analysis.data_quality import diagnose_gpx
from elevation_chart import save_elevation_chart
from models import RaceCondition
from predictor.race_predictor import format_duration, predict_race, save_prediction
from predictor.report import build_markdown_report, save_markdown_report


def run_pipeline(
    activities_dir: str | Path,
    race_gpx: str | Path,
    output_dir: str | Path,
    segment_distance_m: float = 100.0,
    aid_minutes: float = 0.0,
    progress: Callable[[str], None] | None = None,
    condition: RaceCondition | None = None,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    _emit(progress, "[1/5] 正在解析历史 FIT 文件……")
    activities = read_fit_directory(activities_dir, progress=progress)
    _emit(progress, f"[2/5] 正在汇总 {len(activities)} 个活动并建立个人能力画像……")
    profile = build_runner_profile(activities, progress=progress)
    save_runner_profile(profile, output / "runner_profile.json")

    _emit(progress, "[3/5] 正在解析比赛 GPX 并生成路线分段……")
    points = read_gpx(race_gpx)
    gpx_quality = diagnose_gpx(points)
    _emit(progress, f"    GPX 数据质量：{gpx_quality['level']}（{float(gpx_quality['score']):.0%}）")
    _emit(progress, f"    GPX 读取完成，共 {len(points):,} 个轨迹点；正在按 {segment_distance_m:g} 米分段……")
    segments = build_race_segments(points, segment_distance_m)
    micro_count = sum(len(list(segment.get("micro_segments", []))) for segment in segments)
    _emit(progress, f"    路线分段完成，共 {len(segments):,} 个自然地形段、{micro_count:,} 个计时微分段")
    save_segments(segments, output / "race_segments.json")

    _emit(progress, f"[4/5] 正在预测 {len(segments)} 个路线分段……")
    prediction = predict_race(profile, segments, aid_minutes, condition=condition, gpx_quality_score=float(gpx_quality["score"]))
    prediction["gpx_data_quality"] = gpx_quality
    save_prediction(prediction, output / "prediction.json")
    _emit(progress, "[5/5] 正在生成报告和海拔图……")
    save_markdown_report(build_markdown_report(profile, prediction), output / "report.md")
    save_elevation_chart(segments, output / "elevation_profile.png")
    _emit(progress, "全部处理完成。")
    return prediction


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="越野跑比赛时间概率预测 V0.3")
    parser.add_argument("--activities", default="data/activities", help="历史 FIT 文件目录")
    parser.add_argument("--race", default="data/races/race.gpx", help="比赛 GPX 路径")
    parser.add_argument("--output", default="output", help="报告输出目录")
    parser.add_argument("--segment-distance", type=float, default=100.0, help="地形坡度采样窗口（米）")
    parser.add_argument("--aid-minutes", type=float, default=0.0, help="预计补给/停留总时间（分钟）")
    parser.add_argument("--current-form", choices=["very_good", "normal", "slight_fatigue", "poor", "ill_or_injured"], default="normal", help="当前身体状态")
    parser.add_argument("--pacing-strategy", choices=["conservative", "standard", "aggressive"], default="standard", help="比赛强度策略")
    parser.add_argument("--temperature", type=float, default=None, help="比赛温度（摄氏度）")
    parser.add_argument("--humidity", type=float, default=None, help="相对湿度百分比")
    parser.add_argument("--technical-level", type=int, choices=range(-4, 5), default=0, help="相对平时的技术难度 -4 至 4")
    parser.add_argument("--mud-level", type=int, choices=range(-4, 5), default=0, help="相对平时的泥泞程度 -4 至 4")
    parser.add_argument("--night-ratio", type=float, default=0.0, help="夜间路段比例 0-1")
    parser.add_argument("--altitude-factor", type=float, default=1.0, help="高海拔耗时系数")
    parser.add_argument("--carried-weight", type=float, default=0.0, help="相对日常训练额外携带重量（kg）")
    parser.add_argument(
        "--race-start",
        type=_parse_race_start,
        default=None,
        help="比赛出发时间，ISO 8601格式并包含时区，例如 2026-08-01T06:00:00+08:00",
    )
    return parser.parse_args()


def _parse_race_start(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"无效的比赛出发时间：{value}") from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("比赛出发时间必须包含时区，例如 +08:00")
    return parsed.astimezone(timezone.utc)


if __name__ == "__main__":
    args = parse_args()
    result = run_pipeline(
        args.activities,
        args.race,
        args.output,
        args.segment_distance,
        args.aid_minutes,
        condition=RaceCondition(current_form=args.current_form, pacing_strategy=args.pacing_strategy,
                                temperature_c=args.temperature,
                                humidity_percent=args.humidity, altitude_factor=args.altitude_factor,
                                terrain_technical_level=args.technical_level, mud_level=args.mud_level,
                                night_running_ratio=args.night_ratio, carried_weight_kg=args.carried_weight,
                                aid_station_minutes=args.aid_minutes, race_start_time_utc=args.race_start),
        progress=lambda message: print(message, flush=True),
    )
    print(f"预测完成（P50）：{format_duration(float(result['median_finish_time_seconds']))}")
    print(f"报告目录：{Path(args.output).resolve()}")
