from __future__ import annotations

import numpy as np

from config import load_config


def simulate_finish_times(
    adjusted_moving_seconds: float,
    aid_seconds: float,
    confidence: float,
    gpx_quality_score: float = 1.0,
    simulations: int | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    """Generate reproducible P10/P50/P90 finish-time estimates."""
    config = load_config()["monte_carlo"]
    count = int(config["simulations"] if simulations is None else simulations)
    count = max(int(config["min_simulations"]), min(int(config["max_simulations"]), count))
    confidence = max(0.0, min(1.0, float(confidence)))
    ability_sigma = float(config["sigma_at_confidence_zero"]) * (1.0 - confidence) + float(config["sigma_at_confidence_one"]) * confidence
    gpx_sigma = float(config["gpx_sigma_high_quality"]) + (1.0 - max(0.0, min(1.0, gpx_quality_score))) * (float(config["gpx_sigma_low_quality"]) - float(config["gpx_sigma_high_quality"]))
    rng = np.random.default_rng(int(config["seed"] if seed is None else seed))
    ability = rng.normal(1.0, ability_sigma, count)
    condition = rng.normal(1.0, float(config["condition_sigma"]), count)
    gpx = rng.normal(1.0, gpx_sigma, count)
    samples = np.maximum(adjusted_moving_seconds * ability * condition * gpx + aid_seconds, 1.0)
    p10, p50, p90 = np.percentile(samples, [10, 50, 90])
    return {"p10_seconds": round(float(p10), 1), "p50_seconds": round(float(p50), 1),
            "p90_seconds": round(float(p90), 1), "simulations": count,
            "sigma": round(float(np.std(samples) / np.mean(samples)), 4),
            "samples_seconds": [round(float(value), 1) for value in samples]}
