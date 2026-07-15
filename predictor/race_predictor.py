from __future__ import annotations

import json
from pathlib import Path

from parser.gpx_reader import route_summary


def predict_race(
    profile: dict[str, object],
    segments: list[dict[str, float | str]],
    aid_minutes: float = 0.0,
) -> dict[str, object]:
    """Predict every route segment and return a JSON-serializable result."""
    if not segments:
        raise ValueError("比赛路线没有分段数据")
    elapsed_s = 0.0
    predicted: list[dict[str, float | str]] = []

    for segment in segments:
        raw_seconds, basis = _base_segment_seconds(profile, segment)
        fatigue = fatigue_factor(profile, elapsed_s / 3600.0)
        # The profile stores retained performance; lower performance means more time.
        seconds = raw_seconds / max(fatigue, 0.1)
        elapsed_s += seconds
        predicted.append(
            {
                **segment,
                "base_time_seconds": round(raw_seconds, 1),
                "fatigue_factor": round(fatigue, 3),
                "predicted_time_seconds": round(seconds, 1),
                "cumulative_time_seconds": round(elapsed_s, 1),
                "basis": basis,
            }
        )

    aid_seconds = max(0.0, float(aid_minutes)) * 60.0
    return {
        "route": route_summary(segments),
        "moving_time_seconds": round(elapsed_s, 1),
        "aid_time_seconds": round(aid_seconds, 1),
        "total_time_seconds": round(elapsed_s + aid_seconds, 1),
        "segments": predicted,
    }


def fatigue_factor(profile: dict[str, object], elapsed_hour: float) -> float:
    fatigue = profile["fatigue"]
    if elapsed_hour <= 3.0:
        return float(fatigue["3h"])
    if elapsed_hour <= 5.0:
        return float(fatigue["5h"])
    return float(fatigue["8h"])


def save_prediction(prediction: dict[str, object], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")


def _base_segment_seconds(
    profile: dict[str, object], segment: dict[str, float | str]
) -> tuple[float, str]:
    grade = float(segment["grade"])
    distance = float(segment["distance"])
    segment_type = str(segment.get("type", "flat"))
    if segment_type == "uphill":
        label = "15_percent" if grade >= 15 else "10_percent" if grade >= 10 else "5_percent" if grade >= 5 else "1_percent"
        vam = float(profile["uphill"][label])
        gain = max(float(segment["gain"]), distance * grade / 100.0)
        climbing_seconds = gain / max(vam, 1.0) * 3600.0
        flat_seconds = distance / 1000.0 * float(profile["flat"]["aerobic_pace"])
        return max(climbing_seconds, flat_seconds), f"{grade:.1f}%坡 / VAM {vam:.0f} m/h"
    if segment_type == "downhill":
        label = "-15_percent" if grade <= -15 else "-10_percent" if grade <= -10 else "-5_percent" if grade <= -5 else "-1_percent"
        speed = float(profile["downhill"][label]["speed_mps"])
        return distance / max(speed, 0.1), f"{grade:.1f}%坡 / 下坡速度 {speed:.2f} m/s"
    pace = float(profile["flat"]["aerobic_pace"])
    return distance / 1000.0 * pace, f"平路配速 {format_pace(pace)}/km"


def format_duration(seconds: float) -> str:
    total_minutes = int(round(seconds / 60.0))
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}小时{minutes:02d}分钟" if hours else f"{minutes}分钟"


def format_pace(seconds_per_km: float) -> str:
    minutes, seconds = divmod(int(round(seconds_per_km)), 60)
    return f"{minutes}:{seconds:02d}"
