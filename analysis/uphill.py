from __future__ import annotations

import numpy as np
import pandas as pd


UPHILL_BINS = ((5.0, 10.0, "5_percent"), (10.0, 15.0, "10_percent"), (15.0, 61.0, "15_percent"))


def build_uphill_profile(intervals: pd.DataFrame) -> dict[str, float]:
    profile: dict[str, float] = {}
    for low, high, label in UPHILL_BINS:
        sample = intervals[
            intervals["valid_interval"]
            & intervals["grade_pct"].between(low, high, inclusive="left")
            & (intervals["delev_m"] > 0)
        ]
        seconds = float(sample["dt_seconds"].sum())
        vam = float(sample["delev_m"].sum()) / seconds * 3600.0 if seconds > 0 else np.nan
        if np.isfinite(vam) and len(sample) >= 3:
            profile[label] = round(vam, 1)
    return profile

