from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.ability_file import build_ability_bundle
from analysis.backtest import run_rolling_backtest
from analysis.fatigue import build_fatigue_profile, interpolate_fatigue


def _fatigue_activity(hours: float, late_speed_mps: float = 3.0) -> pd.DataFrame:
    count = int(hours * 60) + 1
    elapsed_h = np.arange(count, dtype=float) / 60.0
    speed = np.where(elapsed_h >= 4.5, late_speed_mps, 3.0)
    seconds = np.full(count, 60.0)
    return pd.DataFrame(
        {
            "moving_interval": True,
            "dt_seconds": seconds,
            "movement_grade_pct": 0.0,
            "movement_speed_mps": speed,
            "dd_m": speed * seconds,
            "delev_m": 0.0,
        }
    )


def _route_activity(start: str, seconds: int = 10) -> pd.DataFrame:
    points = 181
    distance = np.arange(points, dtype=float) * 25.0
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=points, freq=f"{seconds}s", tz="UTC"),
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
    frame.attrs["sport"] = "running"
    return frame


def test_fatigue_nodes_require_activity_coverage_and_extrapolate_tail() -> None:
    profile = build_fatigue_profile([_fatigue_activity(6.0, late_speed_mps=1.5)])
    flat = {float(point["hour"]): point for point in profile["flat"]}

    assert flat[5.0]["observed_activity_count"] == 1
    assert flat[5.0]["source"] == "blended"
    assert flat[8.0]["observed_activity_count"] == 0
    assert flat[8.0]["source"] == "default"
    assert flat[20.0]["source"] == "extrapolated"
    assert interpolate_fatigue(10.0, profile["flat"]) < interpolate_fatigue(8.0, profile["flat"])


def test_rolling_backtest_excludes_target_and_future_and_is_reproducible() -> None:
    activities = {
        "old.fit": _route_activity("2026-01-01"),
        "target.fit": _route_activity("2026-02-01", seconds=11),
        "future.fit": _route_activity("2026-03-01", seconds=12),
    }
    bundle = build_ability_bundle(
        activities,
        {name: "trail" for name in activities},
        reference_time=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )

    first = run_rolling_backtest(bundle, simulations=1000, seed=61)
    second = run_rolling_backtest(bundle, simulations=1000, seed=61)

    assert first == second
    assert first["raw_trajectory_stored"] is False
    assert len(first["records"]) == 2
    assert len(first["skipped"]) == 1
    ablation = first["metrics"]["route_similarity_ablation"]
    assert set(ablation) == {"structural", "legacy", "duration_fallback"}
    assert all(values["count"] == 2 for values in ablation.values())
    assert set(first["route_similarity_ablations"]) == {"structural", "legacy", "duration_fallback"}
    for record in first["records"]:
        assert record["baseline_latest_activity_time"] < record["target_activity_time"]
        assert set(record) >= {"actual_moving_seconds", "p10_seconds", "p50_seconds", "p90_seconds"}
