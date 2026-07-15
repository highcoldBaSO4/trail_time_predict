from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.activity_analysis import analyze_activity
from analysis.capability import build_runner_profile
from analysis.confidence import calculate_confidence
from analysis.data_quality import diagnose_fit, diagnose_gpx
from analysis.downhill import interpolate_downhill_speed
from analysis.fatigue import interpolate_fatigue
from analysis.uphill import interpolate_uphill_vam
from config import load_config
from models import CapabilityValue
from models import RaceCondition
from parser.gpx_reader import build_race_segments, read_gpx, route_summary
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

    profile = build_runner_profile({"training.fit": synthetic_activity()})
    prediction = predict_race(profile, segments, aid_minutes=10)
    assert prediction["total_time_seconds"] > prediction["moving_time_seconds"]
    report = build_markdown_report(profile, prediction)
    assert "最终预测" in report
    assert "分段预测" in report


def test_segmentation_groups_a_continuous_slope_as_one_natural_climb() -> None:
    points = [
        {"latitude": 0.0, "longitude": index * (500.0 / 111_194.9266), "elevation": float(index * 10)}
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
    value = CapabilityValue(420, "seconds_per_km", "personal", 0.8, sample_count=4)
    assert value.sample_count == 4


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


def test_profile_exposes_quality_confidence_and_continuous_curves() -> None:
    profile = build_runner_profile({"training.fit": synthetic_activity()})
    assert profile["schema_version"] == "0.2-phase1"
    assert 0 <= profile["flat"]["confidence"] <= 1
    assert len(profile["uphill"]["curve"]) == 4
    assert len(profile["downhill"]["curve"]) == 4
    assert {"flat", "uphill", "downhill"} <= set(profile["fatigue"])
    for terrain in ("flat", "uphill", "downhill"):
        anchor = profile["fatigue"][terrain][0]
        assert anchor == {"hour": 0.0, "factor": 1.0, "sample_count": 0, "source": "anchor", "confidence": None}


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


def test_phase2_monte_carlo_is_reproducible() -> None:
    profile = build_runner_profile({"training.fit": synthetic_activity()})
    segments = [{"distance": 1000.0, "gain": 0.0, "loss": 0.0, "grade": 0.0,
                 "type": "flat", "start_km": 0.0, "end_km": 1.0}]
    first = predict_race(profile, segments, simulations=1000, seed=99)
    second = predict_race(profile, segments, simulations=1000, seed=99)
    assert first["probability"] == second["probability"]


def test_negative_time_impact_formats_with_a_single_sign() -> None:
    assert format_duration(-7 * 60) == "-7分钟"
    assert format_duration(-67 * 60) == "-1小时07分钟"
