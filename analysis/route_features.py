from __future__ import annotations

from collections.abc import Iterable
from typing import Any


TERRAINS = ("flat", "uphill", "downhill")
PHASE_LABELS = ("first_25", "second_25", "third_25", "last_25")
UPHILL_THRESHOLDS = (10, 15, 20)
DOWNHILL_THRESHOLDS = (10, 15, 20)


def route_structure_features(segments: Iterable[dict[str, Any]]) -> dict[str, object]:
    """Build explainable route-structure features from ordered terrain segments.

    The input is deliberately small and shared by GPX routes and FIT-derived
    historical routes: distance, gain/loss, grade and terrain type.  No raw
    coordinates or activity records are retained in the output.
    """
    rows = [_normalise_segment(segment) for segment in segments]
    rows = [row for row in rows if row["distance_m"] > 0]
    total_distance = sum(row["distance_m"] for row in rows)
    total_gain = sum(row["gain_m"] for row in rows)
    total_loss = sum(row["loss_m"] for row in rows)
    if total_distance <= 0:
        return _empty_features()

    grade_bands: dict[str, float] = {}
    for threshold in UPHILL_THRESHOLDS:
        selected = [row for row in rows if row["grade_pct"] >= threshold]
        grade_bands[f"uphill_{threshold}_distance_share"] = _share(sum(row["distance_m"] for row in selected), total_distance)
        grade_bands[f"uphill_{threshold}_gain_share"] = _share(sum(row["gain_m"] for row in selected), total_gain)
    for threshold in DOWNHILL_THRESHOLDS:
        selected = [row for row in rows if row["grade_pct"] <= -threshold]
        grade_bands[f"downhill_{threshold}_distance_share"] = _share(sum(row["distance_m"] for row in selected), total_distance)
        grade_bands[f"downhill_{threshold}_loss_share"] = _share(sum(row["loss_m"] for row in selected), total_loss)

    runs = _terrain_runs(rows, total_distance)
    uphill_runs = [run for run in runs if run["terrain"] == "uphill"]
    downhill_runs = [run for run in runs if run["terrain"] == "downhill"]
    longest_uphill = _largest_run(uphill_runs)
    longest_downhill = _largest_run(downhill_runs)
    phase_distribution = _phase_distribution(rows, total_distance, total_gain, total_loss)
    transitions = _sequence_features(runs, total_distance, total_gain)
    return {
        "version": 1,
        "grade_bands": {key: round(value, 4) for key, value in grade_bands.items()},
        "continuous": {
            "longest_uphill_distance_km": round(longest_uphill["distance_m"] / 1000.0, 3),
            "longest_uphill_gain_m": round(longest_uphill["gain_m"], 1),
            "longest_uphill_average_grade_pct": round(longest_uphill["average_grade_pct"], 2),
            "longest_uphill_start_progress": round(longest_uphill["start_progress"], 4),
            "longest_uphill_end_progress": round(longest_uphill["end_progress"], 4),
            "longest_downhill_distance_km": round(longest_downhill["distance_m"] / 1000.0, 3),
            "longest_downhill_loss_m": round(longest_downhill["loss_m"], 1),
            "longest_downhill_average_grade_pct": round(longest_downhill["average_grade_pct"], 2),
            "longest_downhill_start_progress": round(longest_downhill["start_progress"], 4),
            "longest_downhill_end_progress": round(longest_downhill["end_progress"], 4),
            "maximum_single_ascent_m": round(max((run["gain_m"] for run in uphill_runs), default=0.0), 1),
            "maximum_single_descent_m": round(max((run["loss_m"] for run in downhill_runs), default=0.0), 1),
        },
        "phase_distribution": phase_distribution,
        "sequence": transitions,
    }


def _normalise_segment(segment: dict[str, Any]) -> dict[str, float | str]:
    distance = max(0.0, float(segment.get("distance", segment.get("distance_m", 0.0)) or 0.0))
    gain = max(0.0, float(segment.get("gain", segment.get("gain_m", 0.0)) or 0.0))
    loss = max(0.0, float(segment.get("loss", segment.get("loss_m", 0.0)) or 0.0))
    terrain = str(segment.get("type", segment.get("terrain", "flat")))
    grade = segment.get("grade", segment.get("grade_pct"))
    if grade is None:
        grade = (gain - loss) / max(distance, 1.0) * 100.0
    return {
        "distance_m": distance,
        "gain_m": gain,
        "loss_m": loss,
        "grade_pct": float(grade),
        "terrain": terrain if terrain in TERRAINS else "flat",
    }


def _terrain_runs(rows: list[dict[str, float | str]], total_distance: float) -> list[dict[str, float | str]]:
    runs: list[dict[str, float | str]] = []
    travelled = 0.0
    for row in rows:
        terrain = str(row["terrain"])
        if not runs or runs[-1]["terrain"] != terrain:
            runs.append({
                "terrain": terrain, "distance_m": 0.0, "gain_m": 0.0, "loss_m": 0.0,
                "grade_distance": 0.0, "start_distance_m": travelled,
            })
        run = runs[-1]
        run["distance_m"] = float(run["distance_m"]) + float(row["distance_m"])
        run["gain_m"] = float(run["gain_m"]) + float(row["gain_m"])
        run["loss_m"] = float(run["loss_m"]) + float(row["loss_m"])
        run["grade_distance"] = float(run["grade_distance"]) + float(row["grade_pct"]) * float(row["distance_m"])
        travelled += float(row["distance_m"])
    for run in runs:
        run["average_grade_pct"] = float(run["grade_distance"]) / max(float(run["distance_m"]), 1.0)
        run["start_progress"] = float(run["start_distance_m"]) / max(total_distance, 1.0)
        run["end_progress"] = (float(run["start_distance_m"]) + float(run["distance_m"])) / max(total_distance, 1.0)
        del run["grade_distance"]
        del run["start_distance_m"]
    return runs


def _largest_run(runs: list[dict[str, float | str]]) -> dict[str, float]:
    if not runs:
        return {
            "distance_m": 0.0, "gain_m": 0.0, "loss_m": 0.0, "average_grade_pct": 0.0,
            "start_progress": 0.0, "end_progress": 0.0,
        }
    selected = max(runs, key=lambda item: float(item["distance_m"]))
    return {
        "distance_m": float(selected["distance_m"]), "gain_m": float(selected["gain_m"]),
        "loss_m": float(selected["loss_m"]), "average_grade_pct": float(selected["average_grade_pct"]),
        "start_progress": float(selected["start_progress"]), "end_progress": float(selected["end_progress"]),
    }


def _phase_distribution(
    rows: list[dict[str, float | str]], total_distance: float, total_gain: float, total_loss: float
) -> dict[str, dict[str, float]]:
    values = {
        label: {
            "distance_m": 0.0, "gain_m": 0.0, "loss_m": 0.0, "hard_uphill_gain_m": 0.0,
            **{terrain: 0.0 for terrain in TERRAINS},
        }
        for label in PHASE_LABELS
    }
    elapsed_distance = 0.0
    for row in rows:
        remaining = float(row["distance_m"])
        row_start = elapsed_distance
        while remaining > 1e-9:
            # A segment may end exactly on a 25% boundary.  Floating-point
            # rounding can otherwise keep it in the preceding phase with a
            # zero-sized remainder, causing an endless loop while building a
            # route summary.
            phase_index = min(3, int(row_start / total_distance * 4.0 + 1e-9))
            phase_end = total_distance * (phase_index + 1) / 4.0
            part = min(remaining, max(0.0, phase_end - row_start))
            if part <= 1e-9:
                row_start = phase_end
                continue
            label = PHASE_LABELS[phase_index]
            ratio = part / max(float(row["distance_m"]), 1e-9)
            values[label]["distance_m"] += part
            values[label]["gain_m"] += float(row["gain_m"]) * ratio
            values[label]["loss_m"] += float(row["loss_m"]) * ratio
            if float(row["grade_pct"]) >= 10.0:
                values[label]["hard_uphill_gain_m"] += float(row["gain_m"]) * ratio
            values[label][str(row["terrain"])] += part
            row_start += part
            remaining -= part
        elapsed_distance += float(row["distance_m"])
    return {
        label: {
            "gain_share": round(_share(item["gain_m"], total_gain), 4),
            "loss_share": round(_share(item["loss_m"], total_loss), 4),
            "hard_uphill_gain_share": round(_share(item["hard_uphill_gain_m"], total_gain), 4),
            **{f"{terrain}_distance_share": round(_share(item[terrain], total_distance), 4) for terrain in TERRAINS},
        }
        for label, item in values.items()
    }


def _sequence_features(runs: list[dict[str, float | str]], total_distance: float, total_gain: float) -> dict[str, float]:
    transition_distance = sum(
        float(left["distance_m"])
        for left, right in zip(runs, runs[1:])
        if left["terrain"] == "uphill" and right["terrain"] == "downhill" and float(right["distance_m"]) >= 1000.0
    )
    uphill_distance = sum(float(run["distance_m"]) for run in runs if run["terrain"] == "uphill")
    late_runs = runs[len(runs) // 2:]
    late_hard_gain = sum(float(run["gain_m"]) for run in late_runs if float(run["average_grade_pct"]) >= 10.0)
    return {
        "uphill_to_long_downhill_transition_share": round(_share(transition_distance, uphill_distance), 4),
        "late_hard_uphill_gain_share": round(_share(late_hard_gain, total_gain), 4),
        "terrain_run_count_per_10km": round(len(runs) / max(total_distance / 10_000.0, 0.1), 3),
    }


def _share(value: float, total: float) -> float:
    return max(0.0, float(value) / float(total)) if total > 0 else 0.0


def _empty_features() -> dict[str, object]:
    return {"version": 1, "grade_bands": {}, "continuous": {}, "phase_distribution": {}, "sequence": {}}
