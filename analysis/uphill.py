from __future__ import annotations

import numpy as np
import pandas as pd


UPHILL_BINS = ((5.0, 10.0, "5_percent"), (10.0, 15.0, "10_percent"), (15.0, 61.0, "15_percent"))


def interpolate_uphill_vam(grade: float, points: list[tuple[float, float]]) -> float:
    """Linearly interpolate VAM across grade nodes, clamping at both ends."""
    if not points:
        raise ValueError("上坡能力曲线不能为空")
    ordered = sorted((float(x), float(y)) for x, y in points)
    return float(np.interp(float(grade), [p[0] for p in ordered], [p[1] for p in ordered]))


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

