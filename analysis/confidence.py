from __future__ import annotations

import math
from typing import Any

from config import load_config


def calculate_confidence(
    duration_seconds: float,
    sample_count: int,
    quality_score: float = 1.0,
    variability: float | None = None,
    source: str = "personal",
) -> float:
    """Score evidence coverage on a 0..1 scale using configured rules."""
    rules: dict[str, Any] = load_config()["confidence"]
    if source == "default" or duration_seconds <= 0 or sample_count <= 0:
        return float(rules["default"])
    minutes = duration_seconds / 60.0
    base = float(rules["maximum"])
    for band in rules["duration_minutes"]:
        if band["max"] is None or minutes < float(band["max"]):
            base = float(band["score"])
            break
    count_bonus = min(0.05, math.log1p(sample_count) * 0.012)
    variability_penalty = 0.0 if variability is None else min(0.15, max(0.0, variability) * 0.25)
    result = (base + count_bonus - variability_penalty) * min(1.0, max(0.2, quality_score))
    return round(min(float(rules["maximum"]), max(float(rules["default"]), result)), 3)


def aggregate_quality_score(reports: list[dict[str, Any]]) -> float:
    if not reports:
        return 0.2
    return sum(float(report.get("score", 0.2)) for report in reports) / len(reports)
