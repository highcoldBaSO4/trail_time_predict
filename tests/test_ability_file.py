from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.ability_file import (
    build_ability_bundle,
    load_ability_bundle,
    profile_before_activity,
    refresh_ability_bundle,
    serialize_ability_bundle,
    update_ability_bundle,
)
from analysis import ability_file as ability_file_module


def activity(start: str, seconds: int = 10) -> pd.DataFrame:
    points = 121
    distance = np.arange(points, dtype=float) * 25.0
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=points, freq=f"{seconds}s", tz="UTC"),
            "latitude": 30.0,
            "longitude": 120.0,
            "distance": distance,
            "altitude": 100.0 + distance * 0.08,
            "heart_rate": 150.0,
            "cadence": 170.0,
            "power": np.nan,
            "temperature": 18.0,
        }
    )
    frame.attrs["sport"] = "running"
    return frame


def test_ability_bundle_round_trip_preserves_profile_and_activity_evidence() -> None:
    activities = {
        "older.fit": activity("2026-01-01"),
        "newer.fit": activity("2026-02-01", seconds=9),
    }
    hashes = {name: hashlib.sha256(name.encode()).hexdigest() for name in activities}
    bundle = build_ability_bundle(
        activities,
        {name: "trail" for name in activities},
        hashes,
        reference_time=datetime(2026, 2, 15, tzinfo=timezone.utc),
    )

    restored = load_ability_bundle(serialize_ability_bundle(bundle))

    assert restored.supports_update
    assert restored.activity_hashes == hashes
    assert restored.profile["ability_file"]["activity_count"] == 2
    assert restored.profile["recency_weighting"]["method"] == "exponential_half_life"
    assert restored.activities["older.fit"].attrs["sport"] == "running"
    assert str(restored.activities["older.fit"]["timestamp"].dt.tz) == "UTC"
    refreshed = refresh_ability_bundle(
        restored,
        reference_time=datetime(2026, 2, 20, tzinfo=timezone.utc),
    )
    assert refreshed.profile["sample_count"] == 2


def test_update_skips_duplicate_hash_and_adds_new_activity() -> None:
    first = activity("2026-01-01")
    bundle = build_ability_bundle(
        {"first.fit": first},
        {"first.fit": "trail"},
        {"first.fit": "a" * 64},
        reference_time=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )

    updated, skipped = update_ability_bundle(
        bundle,
        {"duplicate.fit": first.copy(), "second.fit": activity("2026-01-20")},
        {"duplicate.fit": "trail", "second.fit": "road"},
        {"duplicate.fit": "a" * 64, "second.fit": "b" * 64},
        reference_time=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )

    assert skipped == ["duplicate.fit"]
    assert set(updated.activities) == {"first.fit", "second.fit"}
    assert updated.profile["sample_count"] == 2


def test_diagnosis_snapshot_excludes_target_and_future_activities() -> None:
    activities = {
        "past.fit": activity("2026-01-01"),
        "target.fit": activity("2026-02-01"),
        "future.fit": activity("2026-03-01"),
    }
    hashes = {"past.fit": "1" * 64, "target.fit": "2" * 64, "future.fit": "3" * 64}
    bundle = build_ability_bundle(
        activities,
        {name: "trail" for name in activities},
        hashes,
        reference_time=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )

    profile, excluded = profile_before_activity(
        bundle,
        activities["target.fit"],
        target_hash=hashes["target.fit"],
    )

    assert profile["sample_count"] == 1
    assert set(excluded) == {"target.fit", "future.fit"}
    assert str(profile["reference_time"]).startswith("2026-02-01")


def test_legacy_runner_profile_is_read_only() -> None:
    bundle = build_ability_bundle(
        {"first.fit": activity("2026-01-01")},
        {"first.fit": "trail"},
        {"first.fit": "a" * 64},
        reference_time=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )

    legacy = load_ability_bundle(json.dumps(bundle.profile, ensure_ascii=False).encode("utf-8"))

    assert legacy.legacy
    assert not legacy.supports_update
    with pytest.raises(ValueError, match="不能增量更新"):
        update_ability_bundle(
            legacy,
            {"new.fit": activity("2026-02-02")},
            {"new.fit": "trail"},
        )


def test_ability_file_config_tolerates_pre_v05_cached_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ability_file_module, "load_config", lambda: {})

    config = ability_file_module._ability_file_config()

    assert config["schema_version"] == "1.0"
    assert config["current_half_life_days"] == 90.0
