from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.calibration import build_calibration_model, calibrate_prediction_interval
from analysis.capability import build_runner_profile
from predictor.race_predictor import predict_race
from predictor.report import build_markdown_report


def _record(index: int, residual: float = 0.08, terrain_error: float = 8.0) -> dict[str, object]:
    p50 = 10_000.0
    return {
        "target_activity_time": f"2026-0{index + 1}-01T08:00:00+00:00",
        "actual_moving_seconds": p50 * (1.0 + residual),
        "p10_seconds": 9_000.0,
        "p50_seconds": p50,
        "p90_seconds": 11_000.0,
        "route": {
            "distance_km": 25.0,
            "elevation_gain_m": 800.0,
            "elevation_loss_m": 800.0,
            "terrain_share": {"flat": 0.50, "uphill": 0.25, "downhill": 0.25},
        },
        "terrain_errors": {"flat": terrain_error, "uphill": terrain_error, "downhill": terrain_error},
        "data_quality": {"target": 0.95, "route": 0.95},
    }


def _probability() -> dict[str, object]:
    values = np.linspace(9_000.0, 11_000.0, 1000)
    return {
        "p10_seconds": 9_200.0,
        "p50_seconds": 10_000.0,
        "p90_seconds": 10_800.0,
        "samples_seconds": values.tolist(),
        "sigma": 0.05,
    }


def _target_route() -> dict[str, object]:
    return {
        "distance_km": 25.0,
        "elevation_gain_m": 800.0,
        "terrain_share": {"flat": 0.50, "uphill": 0.25, "downhill": 0.25},
    }


def test_fewer_than_three_backtests_only_display_evidence_without_changing_prediction() -> None:
    model = build_calibration_model([_record(0), _record(1)])
    calibrated, detail = calibrate_prediction_interval(
        _probability(), model, _target_route(), {"flat": 0.5, "uphill": 0.25, "downhill": 0.25}
    )

    assert model["enabled"] is False
    assert detail["enabled"] is False
    assert calibrated == _probability()


def test_single_outlier_cannot_activate_calibration() -> None:
    model = build_calibration_model(
        [_record(index, residual=0.005) for index in range(4)] + [_record(5, residual=0.80)]
    )

    assert model["valid_backtest_count"] == 5
    assert model["enabled"] is False
    assert model["status"] == "证据不稳定"


def test_p50_calibration_is_bounded_and_interval_never_narrows() -> None:
    model = build_calibration_model([_record(index) for index in range(5)])
    calibrated, detail = calibrate_prediction_interval(
        _probability(), model, _target_route(), {"flat": 0.5, "uphill": 0.25, "downhill": 0.25},
        {"route_uncertainty": {"additional_global_sigma": 0.03, "terrain_sigma": {}}, "terrain_time_share": {"flat": 0.5}},
    )

    assert model["enabled"] is True
    assert detail["enabled"] is True
    assert 1.0 < float(detail["p50_factor"]) <= np.exp(0.10)
    assert float(calibrated["p50_seconds"]) > float(_probability()["p50_seconds"])
    assert np.log(float(calibrated["p50_seconds"]) / float(calibrated["p10_seconds"])) >= np.log(10_000.0 / 9_200.0)
    assert np.log(float(calibrated["p90_seconds"]) / float(calibrated["p50_seconds"])) >= np.log(10_800.0 / 10_000.0)
    assert "路线相似度" in detail["interval_external_reasons"][0]


def test_terrain_adjustment_needs_extra_evidence_and_is_time_weighted() -> None:
    records = [_record(index, terrain_error=8.0) for index in range(8)]
    for record in records:
        record["terrain_errors"] = {"flat": 8.0, "uphill": 24.0, "downhill": 8.0}
    model = build_calibration_model(records)
    calibrated, detail = calibrate_prediction_interval(
        _probability(), model, _target_route(), {"flat": 0.0, "uphill": 1.0, "downhill": 0.0}
    )

    assert model["terrain"]["uphill"]["enabled"] is True
    assert detail["terrain"]["uphill"]["enabled"] is True
    assert float(detail["terrain_log_adjustment"]) > 0.0
    assert float(calibrated["p50_seconds"]) > float(_probability()["p50_seconds"])


def test_future_prediction_uses_saved_model_and_report_discloses_it() -> None:
    distance = np.arange(181, dtype=float) * 25.0
    import pandas as pd

    activity = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=181, freq="10s", tz="UTC"),
            "latitude": 30.0,
            "longitude": 120.0,
            "distance": distance,
            "altitude": 100.0,
            "heart_rate": 150.0,
            "cadence": 170.0,
            "power": np.nan,
            "temperature": 18.0,
        }
    )
    profile = build_runner_profile({"training.fit": activity})
    segments = [{"distance": 25_000.0, "gain": 0.0, "loss": 0.0, "grade": 0.0,
                 "type": "flat", "start_km": 0.0, "end_km": 25.0}]
    original = predict_race(profile, segments, simulations=1000, seed=95)
    profile["prediction_calibration"] = build_calibration_model([_record(index) for index in range(5)])
    calibrated = predict_race(profile, segments, simulations=1000, seed=95)

    assert calibrated["calibration"]["enabled"] is True
    assert float(calibrated["median_finish_time_seconds"]) > float(original["median_finish_time_seconds"])
    assert float(calibrated["optimistic_time_seconds"]) <= float(calibrated["median_finish_time_seconds"])
    assert float(calibrated["median_finish_time_seconds"]) <= float(calibrated["conservative_time_seconds"])
    assert "历史回测校准" in build_markdown_report(profile, calibrated)
