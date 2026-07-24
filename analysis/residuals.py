from __future__ import annotations

from collections.abc import Callable
from statistics import median
from typing import Any

import numpy as np


def residual_from_record(record: dict[str, Any]) -> float:
    """Log(actual / P50), the residual used by later calibration stages."""
    return float(np.log(float(record["actual_moving_seconds"]) / max(float(record["p50_seconds"]), 1.0)))


def metric_summary(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not records:
        return _empty_summary()
    errors = np.asarray([float(record["actual_moving_seconds"]) / float(record["p50_seconds"]) - 1.0 for record in records])
    covered = np.asarray([
        float(record["p10_seconds"]) <= float(record["actual_moving_seconds"]) <= float(record["p90_seconds"])
        for record in records
    ])
    p50_below_actual = np.asarray([
        float(record["actual_moving_seconds"]) <= float(record["p50_seconds"])
        for record in records
    ])
    return {
        "count": int(len(records)),
        "signed_mean_error": round(float(errors.mean()), 5),
        "median_absolute_percentage_error": round(float(median(np.abs(errors))), 5),
        "mean_absolute_percentage_error": round(float(np.abs(errors).mean()), 5),
        "p10_p90_coverage": round(float(covered.mean()), 5),
        "p50_quantile_calibration": round(float(p50_below_actual.mean()), 5),
    }


def grouped_metric_summary(
    records: list[dict[str, Any]],
    selector: Callable[[dict[str, Any]], str | None],
) -> dict[str, dict[str, float | int | None]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        label = selector(record)
        if label is not None:
            groups.setdefault(label, []).append(record)
    return {label: metric_summary(items) for label, items in sorted(groups.items())}


def _empty_summary() -> dict[str, float | int | None]:
    return {
        "count": 0,
        "signed_mean_error": None,
        "median_absolute_percentage_error": None,
        "mean_absolute_percentage_error": None,
        "p10_p90_coverage": None,
        "p50_quantile_calibration": None,
    }
