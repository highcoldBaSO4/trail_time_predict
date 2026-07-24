from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.capability import build_runner_profile
from analysis.route_features import route_structure_features
from predictor.pacing_strategy import match_route_pacing_strategy, route_features, route_similarity_details
from predictor.race_predictor import predict_race
from predictor.report import build_markdown_report


def _route(order: tuple[str, ...]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for index, terrain in enumerate(order):
        if terrain == "uphill":
            rows.append({"distance": 1000.0, "gain": 150.0, "loss": 0.0, "grade": 15.0, "type": terrain, "start_km": float(index), "end_km": float(index + 1)})
        elif terrain == "downhill":
            rows.append({"distance": 1000.0, "gain": 0.0, "loss": 150.0, "grade": -15.0, "type": terrain, "start_km": float(index), "end_km": float(index + 1)})
        else:
            rows.append({"distance": 1000.0, "gain": 0.0, "loss": 0.0, "grade": 0.0, "type": terrain, "start_km": float(index), "end_km": float(index + 1)})
    return rows


def _activity_frame() -> pd.DataFrame:
    distance = np.arange(181, dtype=float) * 25.0
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=181, freq="10s", tz="UTC"),
            "latitude": 30.0,
            "longitude": 120.0 + distance / 96_000.0,
            "distance": distance,
            "altitude": 100.0 + distance * 0.03,
            "heart_rate": 150.0,
            "cadence": 170.0,
            "power": np.nan,
            "temperature": 18.0,
        }
    )


def _strategy_sample(activity: str, route: list[dict[str, float | str]]) -> dict[str, object]:
    features = route_features(route)
    return {
        "activity": activity,
        "activity_type": "trail",
        **features,
        "overall_curve": [1.0, 1.02, 1.06, 1.10],
        "terrain_curves": {terrain: [1.0, 1.02, 1.06, 1.10] for terrain in ("flat", "uphill", "downhill")},
        "strategy_type": "positive_split",
        "confidence": 0.9,
        "model_weight": 1.0,
    }


def test_route_structure_features_capture_grade_runs_and_late_hard_climbing() -> None:
    features = route_structure_features(_route(("flat", "downhill", "uphill", "uphill")))

    assert features["grade_bands"]["uphill_15_distance_share"] == 0.5
    assert features["continuous"]["longest_uphill_distance_km"] == 2.0
    assert features["continuous"]["maximum_single_ascent_m"] == 300.0
    assert features["phase_distribution"]["last_25"]["hard_uphill_gain_share"] == 0.5
    assert features["sequence"]["terrain_run_count_per_10km"] > 0


def test_route_structure_handles_segments_on_quarter_boundaries() -> None:
    features = route_structure_features([
        {"distance": 230.75, "gain": 23.075, "loss": 0.0, "grade": 10.0, "type": "uphill"}
        for _ in range(4)
    ])

    phases = features["phase_distribution"]
    assert sum(values["gain_share"] for values in phases.values()) == 1.0
    assert all(values["uphill_distance_share"] == 0.25 for values in phases.values())


def test_similarity_separates_same_scale_routes_with_different_slope_sequence() -> None:
    target = route_features(_route(("uphill", "uphill", "downhill", "downhill")))
    same_order = _strategy_sample("same.fit", _route(("uphill", "uphill", "downhill", "downhill")))
    reversed_order = _strategy_sample("reversed.fit", _route(("downhill", "downhill", "uphill", "uphill")))

    same_detail = route_similarity_details(target, same_order)
    reversed_detail = route_similarity_details(target, reversed_order)

    assert same_detail["groups"]["scale"] == 1.0
    assert same_detail["groups"]["terrain_sequence"] == 1.0
    assert reversed_detail["groups"]["scale"] == 1.0
    assert reversed_detail["groups"]["terrain_sequence"] < 0.8
    assert reversed_detail["score"] < same_detail["score"]


def test_missing_historical_structure_is_not_treated_as_a_false_mismatch() -> None:
    target = route_features(_route(("uphill", "uphill", "downhill", "downhill")))
    sample = _strategy_sample("legacy.fit", _route(("uphill", "uphill", "downhill", "downhill")))
    sample.pop("route_structure")

    detail = route_similarity_details(target, sample)

    assert detail["score"] > 0.7
    assert set(detail["missing_feature_groups"]) == {"grade_structure", "continuous_slope", "terrain_sequence"}
    assert any("历史活动缺少部分结构特征" in reason for reason in detail["reasons"])


def test_similarity_ignores_unrecognised_large_legacy_route_payloads() -> None:
    target = route_features(_route(("uphill", "uphill", "downhill", "downhill")))
    normal = _strategy_sample("normal.fit", _route(("uphill", "uphill", "downhill", "downhill")))
    legacy = deepcopy(normal)
    legacy["route_structure"]["grade_bands"].update({f"legacy_raw_{index}": index for index in range(5000)})

    expected = route_similarity_details(target, normal)
    actual = route_similarity_details(target, legacy)

    assert actual["score"] == expected["score"]
    assert actual["groups"] == expected["groups"]


def test_uncovered_long_climb_widens_probability_and_is_reported() -> None:
    target_route = _route(("uphill", "uphill", "uphill", "uphill", "downhill", "downhill", "downhill", "downhill"))
    historical_route = _route(("uphill", "flat", "uphill", "flat", "downhill", "flat", "downhill", "flat"))
    profile = build_runner_profile({"training.fit": _activity_frame()})
    profile["pacing_strategy"] = {
        "phase_centers": [0.125, 0.375, 0.625, 0.875],
        "samples": [_strategy_sample("short-climbs.fit", historical_route)],
    }

    match = match_route_pacing_strategy(profile, target_route, estimated_hours=3.0)
    prediction = predict_race(profile, target_route, simulations=1000, seed=71)
    report = build_markdown_report(profile, prediction)
    route_uncertainty = prediction["probability"]["uncertainty"]["route_similarity"]

    assert match["uncertainty"]["terrain_sigma"]["uphill"] > 0
    assert any("连续上坡" in reason for reason in match["uncertainty"]["reasons"])
    assert route_uncertainty["terrain_sigma"]["uphill"] > 0
    assert "相似度维度" in report
    assert "目标路线最长连续上坡超出历史覆盖" in report
