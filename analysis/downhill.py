from __future__ import annotations

import numpy as np
import pandas as pd


def build_downhill_profile(intervals: pd.DataFrame) -> dict[str, dict[str, float]]:
    profile: dict[str, dict[str, float]] = {}
    grade = intervals["grade_pct"]
    bins = (
        ((grade > -10.0) & (grade < -5.0), "-5_percent"),
        ((grade > -15.0) & (grade <= -10.0), "-10_percent"),
        ((grade <= -15.0), "-15_percent"),
    )
    for grade_mask, label in bins:
        sample = intervals[
            intervals["valid_interval"]
            & grade_mask
            & (intervals["delev_m"] < 0)
        ]
        seconds = float(sample["dt_seconds"].sum())
        if seconds <= 0 or len(sample) < 3:
            continue
        horizontal_speed = float(sample["dd_m"].sum()) / seconds
        vertical_speed = float(-sample["delev_m"].sum()) / seconds * 3600.0
        if np.isfinite(horizontal_speed):
            profile[label] = {
                "speed_mps": round(horizontal_speed, 3),
                "vertical_speed_mph": round(vertical_speed, 1),
            }
    return profile
