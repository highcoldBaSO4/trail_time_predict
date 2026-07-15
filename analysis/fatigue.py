from __future__ import annotations

import numpy as np
import pandas as pd


FATIGUE_WINDOWS = ((0.0, 3.0, "3h"), (3.0, 5.0, "5h"), (5.0, float("inf"), "8h"))


def build_fatigue_profile(activities: list[pd.DataFrame]) -> dict[str, float]:
    """Estimate retained speed by elapsed-time window across long activities."""
    ratios: dict[str, list[float]] = {"3h": [1.0], "5h": [], "8h": []}
    for activity in activities:
        valid = activity[activity["valid_interval"]].copy()
        if valid.empty:
            continue
        valid["elapsed_h"] = valid["dt_seconds"].cumsum() / 3600.0
        baseline = _weighted_speed(valid[valid["elapsed_h"] <= 3.0])
        if not np.isfinite(baseline) or baseline <= 0:
            continue
        for low, high, label in FATIGUE_WINDOWS[1:]:
            sample = valid[(valid["elapsed_h"] > low) & (valid["elapsed_h"] <= high)]
            speed = _weighted_speed(sample)
            if np.isfinite(speed):
                ratios[label].append(float(np.clip(speed / baseline, 0.5, 1.05)))

    defaults = {"3h": 1.0, "5h": 0.90, "8h": 0.80}
    return {
        label: round(float(np.median(values)), 3) if values else default
        for label, default in defaults.items()
        for values in [ratios[label]]
    }


def _weighted_speed(frame: pd.DataFrame) -> float:
    seconds = frame["dt_seconds"].sum()
    return float(frame["dd_m"].sum() / seconds) if seconds > 0 else np.nan

