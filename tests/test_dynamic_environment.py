from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.capability import build_runner_profile
from analysis.environment import solar_elevation_degrees, solar_elevation_degrees_vector
from models import RaceCondition
from predictor.race_predictor import predict_race
from predictor.report import build_markdown_report


def _profile() -> dict[str, object]:
    distance = np.arange(181, dtype=float) * 25.0
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=181, freq="10s", tz="UTC"),
            "latitude": 0.0,
            "longitude": 0.0,
            "distance": distance,
            "altitude": 100.0,
            "heart_rate": 150.0,
            "cadence": 170.0,
            "power": np.nan,
            "temperature": 20.0,
        }
    )
    return build_runner_profile({"training.fit": frame})


def _segments() -> list[dict[str, float | str | bool]]:
    return [
        {
            "distance": 25000.0, "gain": 0.0, "loss": 0.0, "grade": 0.0, "type": "flat",
            "start_km": float(index * 25), "end_km": float((index + 1) * 25),
            "latitude": 0.0, "longitude": 0.0, "elevation": 100.0, "elevation_available": True,
        }
        for index in range(2)
    ]


def test_vectorised_solar_elevation_matches_scalar_calculation() -> None:
    start = datetime(2026, 3, 20, 12, tzinfo=timezone.utc)
    elapsed = np.asarray([0.0, 1800.0, 3600.0])

    actual = solar_elevation_degrees_vector(start, elapsed, 0.0, 0.0)

    expected = [solar_elevation_degrees(start, 0.0, 0.0), solar_elevation_degrees(start.replace(hour=12, minute=30), 0.0, 0.0), solar_elevation_degrees(start.replace(hour=13), 0.0, 0.0)]
    assert actual.tolist() == pytest.approx(expected)


def test_condition_adjusted_arrival_time_converges_and_probability_updates_environment() -> None:
    profile = _profile()
    condition = RaceCondition(
        temperature_c=18.0,
        temperature_peak_c=33.0,
        temperature_peak_hour=2.0,
        temperature_finish_c=22.0,
        humidity_percent=85.0,
        race_start_time_utc=datetime(2026, 3, 20, 16, 30, tzinfo=timezone.utc),
    )

    first = predict_race(profile, _segments(), condition=condition, simulations=1000, seed=81)
    second = predict_race(profile, _segments(), condition=condition, simulations=1000, seed=81)
    dynamic = first["probability"]["uncertainty"]["dynamic_environment"]

    assert first == second
    assert first["duration_match"]["mode"] == "condition_adjusted_arrival_time"
    assert first["duration_match"]["converged"] is True
    assert abs(float(first["duration_match"]["estimated_hours"]) * 3600.0 - float(first["adjusted_moving_time_seconds"])) <= 25.0
    assert dynamic["enabled"] is True
    assert dynamic["sources"]["weather"] == "user_input"
    assert dynamic["sources"]["night"] == "route_confirmed"
    assert float(dynamic["temperature_c_p90"]) > float(dynamic["temperature_c_p10"])
    assert float(dynamic["night_ratio_p90"]) > float(dynamic["night_ratio_p10"])
    report = build_markdown_report(profile, first)
    assert "条件总时长迭代收敛" in report
    assert "动态到达时间环境" in report


def test_unknown_weather_and_night_use_conservative_probability_prior() -> None:
    prediction = predict_race(
        _profile(),
        [{"distance": 5000.0, "gain": 0.0, "loss": 0.0, "grade": 0.0,
          "type": "flat", "start_km": 0.0, "end_km": 5.0}],
        simulations=1000,
        seed=89,
    )
    uncertainty = prediction["probability"]["uncertainty"]

    assert uncertainty["dynamic_environment"]["sources"]["weather"] == "unknown"
    assert uncertainty["dynamic_environment"]["sources"]["night"] == "unknown"
    assert uncertainty["condition_sources"]["weather"]["effective_sigma"] > 0
    assert uncertainty["condition_sources"]["night"]["effective_sigma"] > 0
