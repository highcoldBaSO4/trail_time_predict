from __future__ import annotations

import io
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.activity_analysis import analyze_activity
from analysis.activity_selection import apply_activity_review, build_activity_review
from analysis.capability import build_runner_profile
from analysis.confidence import calculate_confidence
from analysis.data_quality import diagnose_fit, diagnose_gpx
from analysis.downhill import interpolate_downhill_speed
from analysis.environment import build_environment_profile, relative_altitude_factor, solar_elevation_degrees
from analysis.fatigue import interpolate_fatigue
from analysis.uphill import interpolate_uphill_vam
from config import load_config
from models import CapabilityValue, PredictionResult, RaceCondition, RunnerProfile
from parser.gpx_reader import _terrain_type, build_race_segments, read_gpx, route_summary
from parser import fit_reader
from predictor.race_predictor import format_duration, predict_race
from predictor.report import build_markdown_report


def synthetic_activity(points: int = 120, seconds: int = 10) -> pd.DataFrame:
    distance = np.arange(points, dtype=float) * 25.0
    altitude = 100.0 + distance * 0.08
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=points, freq=f"{seconds}s", tz="UTC"),
            "latitude": 30.0,
            "longitude": 120.0,
            "distance": distance,
            "altitude": altitude,
            "heart_rate": 150.0,
            "cadence": 170.0,
            "power": np.nan,
        }
    )


def test_activity_metrics_and_profile() -> None:
    frame = synthetic_activity()
    summary = analyze_activity(frame, "training.fit")
    assert 2.9 < summary["distance_km"] < 3.1
    assert summary["elevation_gain"] > 200
    profile = build_runner_profile({"training.fit": frame})
    assert profile["sample_count"] == 1
    assert profile["uphill"]["5_percent"] > 0


def test_flat_pace_uses_natural_flat_without_heart_rate() -> None:
    frame = synthetic_activity()
    frame["altitude"] = 100.0
    frame["heart_rate"] = np.nan

    profile = build_runner_profile({"flat.fit": frame})

    assert profile["flat"]["source"] == "discounted_road_only"
    assert profile["flat"]["qualified_segments"] == 1
    assert profile["flat"]["sample_distance_km"] > 2.9
    assert profile["flat"]["aerobic_pace"] == 440.0


def test_flat_pace_weights_trail_and_discounted_road() -> None:
    road = synthetic_activity(seconds=10)
    trail = synthetic_activity(seconds=12)
    for frame in (road, trail):
        frame["altitude"] = 100.0
        frame["heart_rate"] = np.nan

    profile = build_runner_profile({"road.fit": road, "trail.fit": trail})

    assert profile["flat"]["source"] == "trail_70pct_plus_discounted_road_30pct"
    assert profile["flat"]["road"]["median_pace"] == 400.0
    assert profile["flat"]["trail"]["median_pace"] == 480.0
    assert profile["flat"]["aerobic_pace"] == 468.0


def test_activity_review_filters_and_overrides_road_trail_type() -> None:
    activities = {"training.fit": synthetic_activity(), "short.fit": synthetic_activity(points=5)}
    rows = build_activity_review(activities)
    by_name = {row["filename"]: row for row in rows}

    assert by_name["training.fit"]["confirmed_type"] == "road"
    assert by_name["training.fit"]["use_for_model"] is True
    assert by_name["short.fit"]["use_for_model"] is False
    assert "活动时长不足10分钟" in by_name["short.fit"]["quality_issues"]

    by_name["training.fit"]["confirmed_type"] = "trail"
    selected, activity_types = apply_activity_review(activities, list(by_name.values()))
    profile = build_runner_profile(selected, activity_types)

    assert list(selected) == ["training.fit"]
    assert activity_types == {"training.fit": "trail"}
    assert profile["activities"][0]["activity_type"] == "trail"


def test_activity_review_requires_one_selected_activity() -> None:
    activities = {"training.fit": synthetic_activity()}
    rows = build_activity_review(activities)
    rows[0]["use_for_model"] = False

    with pytest.raises(ValueError, match="至少选择一个活动"):
        apply_activity_review(activities, rows)


def test_profile_and_report_expose_four_slope_bands() -> None:
    profile = build_runner_profile({"training.fit": synthetic_activity()})

    assert {"1_percent", "5_percent", "10_percent", "15_percent"} <= set(profile["uphill"])
    assert {"-1_percent", "-5_percent", "-10_percent", "-15_percent"} <= set(profile["downhill"])

    gpx = """<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">
      <trk><trkseg><trkpt lat="0" lon="0"><ele>0</ele></trkpt>
      <trkpt lat="0" lon="0.01"><ele>70</ele></trkpt></trkseg></trk></gpx>"""
    prediction = predict_race(profile, build_race_segments(read_gpx(io.StringIO(gpx)), 100))
    report = build_markdown_report(profile, prediction)
    assert "微坡" in report and "缓坡" in report and "中坡" in report and "陡坡" in report
    assert "微下降" in report and "陡下降" in report
    assert "水平速度" not in report
    assert "| 档位 | 平均坡度 | 等效配速 | VAM | 可信度 | 历史样本 | 累计距离 | 累计高度 |" in report
    assert "m/h" in report
    assert "长时间疲劳衰减" in report
    assert "能力保留比例" in report
    assert "耗时修正倍率" in report
    assert "疲劳因子" in report


def test_gpx_segmentation_prediction_and_report() -> None:
    gpx = """<?xml version="1.0" encoding="UTF-8"?>
    <gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">
      <trk><trkseg>
        <trkpt lat="30.0000" lon="120.0000"><ele>100</ele></trkpt>
        <trkpt lat="30.0000" lon="120.0060"><ele>150</ele></trkpt>
        <trkpt lat="30.0000" lon="120.0120"><ele>100</ele></trkpt>
      </trkseg></trk>
    </gpx>"""
    points = read_gpx(io.StringIO(gpx))
    segments = build_race_segments(points, 100)
    summary = route_summary(segments)
    assert 1.0 < summary["distance_km"] < 1.3
    assert summary["climbs"] == 1
    assert summary["descents"] == 1
    assert {"latitude", "longitude", "elevation", "elevation_available"} <= set(segments[0])

    profile = build_runner_profile({"training.fit": synthetic_activity()})
    prediction = predict_race(profile, segments, aid_minutes=10)
    assert prediction["total_time_seconds"] > prediction["moving_time_seconds"]
    report = build_markdown_report(profile, prediction)
    assert "最终预测" in report
    assert "分段预测" in report


def test_segmentation_groups_a_continuous_slope_as_one_natural_climb() -> None:
    points = [
        {"latitude": 0.0, "longitude": index * (500.0 / 111_194.9266), "elevation": float(index * 30)}
        for index in range(5)
    ]
    segments = build_race_segments(points, 500.0)

    assert len(segments) == 1
    assert segments[0]["type"] == "uphill"
    assert 1.99 < float(segments[0]["distance"]) / 1000.0 < 2.01


def test_fit_reader_converts_path_for_fitparse(monkeypatch, tmp_path: Path) -> None:
    received = {}

    class FakeMessage:
        def get_values(self):
            return {"timestamp": pd.Timestamp("2026-01-01T00:00:00Z"), "distance": 0}

    class FakeFitFile:
        def __init__(self, source, check_crc=True):
            received["source"] = source

        def get_messages(self, message_type):
            return [FakeMessage()] if message_type == "record" else []

    monkeypatch.setattr(fit_reader, "FitFile", FakeFitFile)
    frame = fit_reader.read_fit(tmp_path / "activity.fit")

    assert isinstance(received["source"], str)
    assert len(frame) == 1


def test_fit_reader_keeps_records_before_trailing_parse_error(monkeypatch, tmp_path: Path) -> None:
    class FakeMessage:
        def get_values(self):
            return {"timestamp": pd.Timestamp("2026-01-01T00:00:00Z"), "distance": 0}

    class PartiallyBrokenFitFile:
        def __init__(self, source, check_crc=True):
            pass

        def get_messages(self, message_type):
            yield FakeMessage()
            raise RuntimeError("invalid extension field")

    monkeypatch.setattr(fit_reader, "FitFile", PartiallyBrokenFitFile)
    with pytest.warns(RuntimeWarning, match="已保留异常前"):
        frame = fit_reader.read_fit(tmp_path / "partial.fit")

    assert len(frame) == 1


def test_config_and_typed_capability_model() -> None:
    assert load_config()["confidence"]["default"] == 0.20
    assert load_config()["terrain"]["flat_grade_abs_percent"] == 2.0
    value = CapabilityValue(420, "seconds_per_km", "personal", 0.8, sample_count=4)
    assert value.sample_count == 4


def test_flat_grade_threshold_is_consistent() -> None:
    assert _terrain_type(-2.0) == "flat"
    assert _terrain_type(0.0) == "flat"
    assert _terrain_type(2.0) == "flat"
    assert _terrain_type(-2.01) == "downhill"
    assert _terrain_type(2.01) == "uphill"


def test_continuous_capability_and_fatigue_interpolation() -> None:
    assert interpolate_uphill_vam(10.0, [(5.0, 500.0), (15.0, 400.0)]) == pytest.approx(450.0)
    assert interpolate_downhill_speed(-10.0, [(-15.0, 2.0), (-5.0, 3.0)]) == pytest.approx(2.5)
    assert interpolate_fatigue(4.0, [(3.0, 1.0), (5.0, 0.8)]) == pytest.approx(0.9)


def test_confidence_and_fit_quality_safe_defaults() -> None:
    assert calculate_confidence(0, 0, source="default") == 0.2
    report = diagnose_fit(synthetic_activity())
    assert report["level"] in {"高", "中", "低"}
    assert 0 <= report["score"] <= 1


def test_gpx_quality_reports_multiscale_gain() -> None:
    points = [
        {"latitude": 0.0, "longitude": i * 0.001, "elevation": float(i * 5)}
        for i in range(6)
    ]
    report = diagnose_gpx(points)
    assert report["point_count"] == 6
    assert set(report["resampled_gain_m"]) == {"50", "100", "200"}


def test_solar_position_and_historical_environment_profile() -> None:
    noon = solar_elevation_degrees(datetime(2026, 3, 20, 12, tzinfo=timezone.utc), 0.0, 0.0)
    midnight = solar_elevation_degrees(datetime(2026, 3, 20, 0, tzinfo=timezone.utc), 0.0, 0.0)
    environment = build_environment_profile([synthetic_activity()])

    assert noon > 80
    assert midnight < -80
    assert environment["night"]["source"] == "fit_coordinates"
    assert environment["altitude"]["mean_m"] > 100
    assert relative_altitude_factor(2500, 500) > 1.0


def test_profile_exposes_quality_confidence_and_continuous_curves() -> None:
    profile = build_runner_profile({"training.fit": synthetic_activity()})
    assert profile["schema_version"] == "0.2"
    assert 0 <= profile["flat"]["confidence"] <= 1
    assert len(profile["uphill"]["curve"]) == 4
    assert len(profile["downhill"]["curve"]) == 4
    assert {"flat", "uphill", "downhill"} <= set(profile["fatigue"])
    for terrain in ("flat", "uphill", "downhill"):
        anchor = profile["fatigue"][terrain][0]
        assert anchor == {"hour": 0.0, "factor": 1.0, "sample_count": 0, "source": "anchor", "confidence": None}


def test_runner_profile_typed_round_trip_and_missing_fields() -> None:
    profile = build_runner_profile({"training.fit": synthetic_activity()})
    typed = RunnerProfile.from_profile_dict(profile)
    restored = typed.to_profile_dict()

    assert typed.source_activity_count == 1
    assert restored["flat"]["aerobic_pace"] == profile["flat"]["aerobic_pace"]
    assert restored["uphill"]["curve"] == profile["uphill"]["curve"]
    with pytest.raises(ValueError, match="个人能力画像结构无效"):
        RunnerProfile.from_profile_dict({})


def test_runner_profile_typed_model_upgrades_legacy_curves() -> None:
    legacy = build_runner_profile({"training.fit": synthetic_activity()})
    legacy["uphill"].pop("curve")
    legacy["downhill"].pop("curve")
    for terrain in ("flat", "uphill", "downhill"):
        legacy["fatigue"].pop(terrain)

    upgraded = RunnerProfile.from_profile_dict(legacy).to_profile_dict()

    assert len(upgraded["uphill"]["curve"]) == 4
    assert len(upgraded["downhill"]["curve"]) == 4
    assert upgraded["fatigue"]["flat"][2]["factor"] == legacy["fatigue"]["5h"]
    assert upgraded["uphill"]["curve"][0]["source"] == "legacy"


def test_profile_exposes_duration_capability_layers() -> None:
    profile = build_runner_profile({"training.fit": synthetic_activity()})
    layers = profile["duration_capabilities"]
    assert [layer["name"] for layer in layers] == ["short", "medium", "long", "ultra"]
    assert layers[0]["terrain_source"]["uphill"] == "personal"
    assert layers[-1]["source"] == "fallback"
    assert all("terrain_confidence" in layer for layer in layers)


def test_phase2_prediction_conditions_and_probability_order() -> None:
    profile = build_runner_profile({"training.fit": synthetic_activity()})
    segments = [{"index": 1, "name": "flat_1", "start_km": 0.0, "end_km": 10.0,
                 "distance": 10000.0, "gain": 0.0, "loss": 0.0, "grade": 0.0,
                 "max_grade": 0.0, "type": "flat", "terrain": "平路"}]
    normal = predict_race(profile, segments, simulations=1000, seed=7)
    difficult = predict_race(
        profile, segments, condition=RaceCondition(current_form="poor", temperature_c=30,
                                                   humidity_percent=85, terrain_technical_level=4,
                                                   mud_level=3, night_running_ratio=0.5,
                                                   carried_weight_kg=2, aid_station_minutes=10),
        simulations=1000, seed=7,
    )
    assert difficult["adjusted_moving_time_seconds"] > difficult["standard_moving_time_seconds"]
    assert difficult["median_finish_time_seconds"] > normal["median_finish_time_seconds"]
    assert difficult["optimistic_time_seconds"] <= difficult["median_finish_time_seconds"] <= difficult["conservative_time_seconds"]
    assert difficult["adjustment_breakdown"]["technical"] > 0
    assert difficult["aid_station_time_seconds"] == 600
    report = build_markdown_report(profile, difficult)
    assert "概率区间依据" in report
    assert "疲劳可信度" in report


def test_phase2_monte_carlo_is_reproducible() -> None:
    profile = build_runner_profile({"training.fit": synthetic_activity()})
    segments = [{"distance": 1000.0, "gain": 0.0, "loss": 0.0, "grade": 0.0,
                 "type": "flat", "start_km": 0.0, "end_km": 1.0}]
    first = predict_race(profile, segments, simulations=1000, seed=99)
    second = predict_race(profile, segments, simulations=1000, seed=99)
    assert first["probability"] == second["probability"]
    assert first["probability"]["method"] == "segmented_source_condition_physical_gpx"
    assert set(first["probability"]["uncertainty"]["ability_confidence"]) == {"flat", "uphill", "downhill"}
    assert first["probability"]["uncertainty"]["gpx"]["mode"] == "segment_elevation_grade"
    assert "route_weighted_confidence" in first["probability"]["uncertainty"]


def test_segmented_probability_reflects_route_terrain_confidence() -> None:
    profile = build_runner_profile({"training.fit": synthetic_activity()})
    profile["flat"]["confidence"] = 0.95
    for point in profile["uphill"]["curve"]:
        point["confidence"] = 0.2
    for point in profile["downhill"]["curve"]:
        point["confidence"] = 0.95
    for layer in profile["duration_capabilities"]:
        layer["terrain_confidence"] = {terrain: 0.95 for terrain in ("flat", "uphill", "downhill")}
    for terrain in ("flat", "uphill", "downhill"):
        for point in profile["fatigue"][terrain]:
            if point["confidence"] is not None:
                point["confidence"] = 0.95

    flat_segments = [{"distance": 1000.0, "gain": 0.0, "loss": 0.0, "grade": 0.0,
                      "type": "flat", "start_km": 0.0, "end_km": 1.0}]
    uphill_segments = [{"distance": 1000.0, "gain": 100.0, "loss": 0.0, "grade": 10.0,
                        "type": "uphill", "start_km": 0.0, "end_km": 1.0}]
    flat_prediction = predict_race(profile, flat_segments, simulations=3000, seed=31)
    uphill_prediction = predict_race(profile, uphill_segments, simulations=3000, seed=31)

    assert flat_prediction["probability"]["uncertainty"]["terrain_time_share"]["flat"] == 1.0
    assert uphill_prediction["probability"]["uncertainty"]["terrain_time_share"]["uphill"] == 1.0
    assert uphill_prediction["probability"]["sigma"] > flat_prediction["probability"]["sigma"]
    assert uphill_prediction["confidence"] < flat_prediction["confidence"]


def test_probability_separates_condition_sources_and_gpx_geometry() -> None:
    profile = build_runner_profile({"training.fit": synthetic_activity()})
    segments = [{"distance": 1000.0, "gain": 100.0, "loss": 0.0, "grade": 10.0,
                 "type": "uphill", "start_km": 0.0, "end_km": 1.0,
                 "elevation": 1800.0, "elevation_available": True}]
    prediction = predict_race(
        profile,
        segments,
        condition=RaceCondition(current_form="poor", terrain_technical_level=3, mud_level=2),
        simulations=1000,
        seed=41,
        gpx_quality_score=0.5,
    )
    high_quality = predict_race(
        profile,
        segments,
        condition=RaceCondition(current_form="poor", terrain_technical_level=3, mud_level=2),
        simulations=1000,
        seed=41,
        gpx_quality_score=1.0,
    )
    uncertainty = prediction["probability"]["uncertainty"]

    assert uncertainty["condition_sources"]["form"]["active_time_share"] == 1.0
    assert uncertainty["condition_sources"]["technical"]["effective_sigma"] > 0
    assert uncertainty["condition_sources"]["mud"]["effective_sigma"] > 0
    assert uncertainty["condition_sources"]["night"]["effective_sigma"] == 0
    assert uncertainty["gpx"]["affected_time_share"] == 1.0
    assert uncertainty["route_weighted_confidence"]["gpx_quality"] == 0.5
    assert prediction["probability"]["sigma"] > high_quality["probability"]["sigma"]


def test_segmented_probability_samples_uphill_fatigue_separately() -> None:
    high_confidence = build_runner_profile({"training.fit": synthetic_activity()})
    low_confidence = deepcopy(high_confidence)
    for profile, confidence in ((high_confidence, 0.95), (low_confidence, 0.2)):
        for point in profile["uphill"]["curve"]:
            point["confidence"] = 0.95
        for layer in profile["duration_capabilities"]:
            layer["terrain_confidence"] = {terrain: 0.95 for terrain in ("flat", "uphill", "downhill")}
        for terrain in ("flat", "uphill", "downhill"):
            for point in profile["fatigue"][terrain]:
                if point["confidence"] is not None:
                    point["confidence"] = confidence if terrain == "uphill" else 0.95
    segments = [
        {"distance": 5000.0, "gain": 500.0, "loss": 0.0, "grade": 10.0,
         "type": "uphill", "start_km": index * 5.0, "end_km": (index + 1) * 5.0}
        for index in range(8)
    ]
    stable = predict_race(high_confidence, segments, simulations=3000, seed=37)
    uncertain = predict_race(low_confidence, segments, simulations=3000, seed=37)

    assert uncertain["probability"]["sigma"] > stable["probability"]["sigma"]
    assert uncertain["probability"]["uncertainty"]["fatigue_confidence"]["uphill"][2] == 0.2


def test_race_automatically_applies_night_and_altitude_by_segment() -> None:
    profile = build_runner_profile({"training.fit": synthetic_activity()})
    profile["environment"]["night"]["ratio"] = 0.0
    segments = [{"distance": 1000.0, "gain": 100.0, "loss": 0.0, "grade": 10.0,
                 "type": "uphill", "start_km": 0.0, "end_km": 1.0,
                 "latitude": 0.0, "longitude": 0.0, "elevation": 2500.0,
                 "elevation_available": True}]
    night = predict_race(
        profile,
        segments,
        condition=RaceCondition(race_start_time_utc=datetime(2026, 3, 20, 0, tzinfo=timezone.utc)),
        simulations=1000,
        seed=5,
    )
    day = predict_race(
        profile,
        segments,
        condition=RaceCondition(race_start_time_utc=datetime(2026, 3, 20, 12, tzinfo=timezone.utc)),
        simulations=1000,
        seed=5,
    )

    assert night["segments"][0]["environment"]["night"] is True
    assert day["segments"][0]["environment"]["night"] is False
    assert night["segments"][0]["environment"]["altitude_factor"] > 1.0
    assert night["adjusted_moving_time_seconds"] > day["adjusted_moving_time_seconds"]
    assert night["environment"]["race_night_ratio"] == 1.0
    assert day["environment"]["race_maximum_elevation_m"] == 2500.0
    assert str(night["condition"]["race_start_time_utc"]).endswith("+00:00")
    report = build_markdown_report(profile, night)
    assert "历史环境覆盖" in report
    assert "比赛预计夜间占比" in report
    assert "海拔系数" in report


def test_night_penalty_ignores_flat_but_affects_slopes() -> None:
    from predictor.condition_adjustment import condition_factors

    condition = RaceCondition(night_running_ratio=1.0)

    assert condition_factors(condition, "flat")["night"] == pytest.approx(1.0)
    assert condition_factors(condition, "uphill")["night"] > 1.0
    assert condition_factors(condition, "downhill")["night"] > condition_factors(condition, "uphill")["night"]


def test_historical_night_flat_does_not_reduce_slope_night_penalty() -> None:
    flat = synthetic_activity()
    flat["altitude"] = 100.0
    flat["latitude"] = 0.0
    flat["longitude"] = 0.0
    flat["timestamp"] = pd.date_range("2026-03-20T00:00:00Z", periods=len(flat), freq="10s")
    profile = build_runner_profile({"night_flat.fit": flat})

    assert profile["environment"]["night"]["terrain"]["flat"]["ratio"] == 1.0
    assert profile["environment"]["night"]["terrain"]["uphill"]["ratio"] == 0.0
    segments = [{"distance": 1000.0, "gain": 100.0, "loss": 0.0, "grade": 10.0,
                 "type": "uphill", "start_km": 0.0, "end_km": 1.0,
                 "latitude": 0.0, "longitude": 0.0, "elevation": 100.0,
                 "elevation_available": True}]
    prediction = predict_race(
        profile,
        segments,
        condition=RaceCondition(race_start_time_utc=datetime(2026, 3, 20, 0, tzinfo=timezone.utc)),
        simulations=1000,
        seed=13,
    )

    assert prediction["segments"][0]["environment"]["historical_night_ratio"] == 0.0
    assert prediction["segments"][0]["condition_factors"]["night"] == pytest.approx(1.05)


def test_prediction_result_typed_round_trip_and_order_validation() -> None:
    profile = build_runner_profile({"training.fit": synthetic_activity()})
    segments = [{"distance": 1000.0, "gain": 0.0, "loss": 0.0, "grade": 0.0,
                 "type": "flat", "start_km": 0.0, "end_km": 1.0}]
    prediction = predict_race(profile, segments, simulations=1000, seed=11)
    typed = PredictionResult.from_dict(prediction)

    assert typed.to_dict()["probability"] == prediction["probability"]
    invalid = dict(prediction)
    invalid["optimistic_time_seconds"] = invalid["conservative_time_seconds"] + 1
    with pytest.raises(ValueError, match="P10 <= P50 <= P90"):
        PredictionResult.from_dict(invalid)


def test_negative_time_impact_formats_with_a_single_sign() -> None:
    assert format_duration(-7 * 60) == "-7分钟"
    assert format_duration(-67 * 60) == "-1小时07分钟"
