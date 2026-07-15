from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from analysis.capability import build_runner_profile, save_runner_profile
from parser.fit_reader import read_fit_directory
from parser.gpx_reader import build_race_segments, read_gpx, save_segments
from analysis.data_quality import diagnose_gpx
from predictor.race_predictor import format_duration, predict_race, save_prediction
from predictor.report import build_markdown_report, save_markdown_report


def run_pipeline(
    activities_dir: str | Path,
    race_gpx: str | Path,
    output_dir: str | Path,
    segment_distance_m: float = 100.0,
    aid_minutes: float = 0.0,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    _emit(progress, "[1/5] 正在解析历史 FIT 文件……")
    activities = read_fit_directory(activities_dir, progress=progress)
    _emit(progress, f"[2/5] 正在汇总 {len(activities)} 个活动并建立个人能力画像……")
    profile = build_runner_profile(activities)
    save_runner_profile(profile, output / "runner_profile.json")

    _emit(progress, "[3/5] 正在解析比赛 GPX 并生成路线分段……")
    points = read_gpx(race_gpx)
    gpx_quality = diagnose_gpx(points)
    _emit(progress, f"    GPX 数据质量：{gpx_quality['level']}（{float(gpx_quality['score']):.0%}）")
    _emit(progress, f"    GPX 读取完成，共 {len(points):,} 个轨迹点；正在按 {segment_distance_m:g} 米分段……")
    segments = build_race_segments(points, segment_distance_m)
    _emit(progress, f"    路线分段完成，共 {len(segments):,} 段")
    save_segments(segments, output / "race_segments.json")

    _emit(progress, f"[4/5] 正在预测 {len(segments)} 个路线分段……")
    prediction = predict_race(profile, segments, aid_minutes)
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


def save_elevation_chart(segments: list[dict[str, object]], path: str | Path) -> None:
    import matplotlib.pyplot as plt

    distances = [0.0]
    elevations = [0.0]
    for segment in segments:
        distances.append(float(segment["end_km"]))
        elevations.append(elevations[-1] + float(segment["gain"]) - float(segment["loss"]))
    figure, axis = plt.subplots(figsize=(10, 3.5))
    axis.plot(distances, elevations, color="#d95f02", linewidth=2)
    axis.fill_between(distances, elevations, min(elevations), color="#fdb863", alpha=0.35)
    axis.set(xlabel="Distance (km)", ylabel="Relative elevation (m)", title="Race elevation profile")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="越野跑比赛时间预测 V0.1")
    parser.add_argument("--activities", default="data/activities", help="历史 FIT 文件目录")
    parser.add_argument("--race", default="data/races/race.gpx", help="比赛 GPX 路径")
    parser.add_argument("--output", default="output", help="报告输出目录")
    parser.add_argument("--segment-distance", type=float, default=100.0, help="地形坡度采样窗口（米）")
    parser.add_argument("--aid-minutes", type=float, default=0.0, help="预计补给/停留总时间（分钟）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_pipeline(
        args.activities,
        args.race,
        args.output,
        args.segment_distance,
        args.aid_minutes,
        progress=lambda message: print(message, flush=True),
    )
    print(f"预测完成：{format_duration(float(result['total_time_seconds']))}")
    print(f"报告目录：{Path(args.output).resolve()}")
