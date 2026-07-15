from __future__ import annotations

import numpy as np


def duration_match(profile: dict[str, object], estimated_hours: float, terrain: str) -> dict[str, object]:
    """Interpolate sustainable time factor and confidence for target duration."""
    layers = list(profile.get("duration_capabilities", []))
    if not layers:
        return {"factor": 1.0, "confidence": 0.2, "weights": {}, "source": "legacy"}
    centers = np.asarray([float(layer["center_hours"]) for layer in layers])
    target = max(0.0, float(estimated_hours))
    if target <= centers[0]:
        indices, weights = [0], [1.0]
    elif target >= centers[-1]:
        indices, weights = [len(layers) - 1], [1.0]
    else:
        upper = int(np.searchsorted(centers, target))
        lower = upper - 1
        upper_weight = (target - centers[lower]) / (centers[upper] - centers[lower])
        indices, weights = [lower, upper], [1.0 - upper_weight, upper_weight]
    factor = sum(float(layers[index]["time_factors"][terrain]) * weight for index, weight in zip(indices, weights))
    confidence = sum(float(layers[index].get("terrain_confidence", {}).get(terrain, layers[index]["confidence"])) * weight for index, weight in zip(indices, weights))
    labels = {str(layers[index]["label"]): round(weight, 3) for index, weight in zip(indices, weights)}
    source = "personal" if all(layers[index].get("terrain_source", {}).get(terrain, layers[index]["source"]) == "personal" for index in indices) else "fallback"
    return {"factor": round(factor, 4), "confidence": round(confidence, 3), "weights": labels, "source": source}
