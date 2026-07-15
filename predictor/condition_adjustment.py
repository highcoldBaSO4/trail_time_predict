from __future__ import annotations

from config import load_config
from models import RaceCondition


def condition_factors(condition: RaceCondition, terrain: str) -> dict[str, float]:
    """Return explainable time multipliers for one terrain segment."""
    condition = condition.normalized()
    config = load_config()["conditions"]
    form_entry = config["current_form"].get(condition.current_form, config["current_form"]["normal"])
    technical = float(config["technical_factors"][condition.terrain_technical_level][terrain])
    mud = 1.0 + condition.mud_level * float(config["mud_per_level"][terrain])
    night = 1.0 + condition.night_running_ratio * float(config["night_max"][terrain])
    weight = 1.0 + condition.carried_weight_kg * float(config["carried_weight_per_kg"][terrain])
    heat_config = config["heat"]
    heat = 1.0
    if condition.temperature_c is not None:
        heat += max(0.0, condition.temperature_c - float(heat_config["threshold_c"])) * float(heat_config["per_degree"])
    if condition.humidity_percent is not None:
        heat += max(0.0, condition.humidity_percent - float(heat_config["humidity_threshold_percent"])) * float(heat_config["humidity_per_percent"])
    return {
        "form": float(form_entry["factor"]),
        "technical": technical,
        "mud": mud,
        "night": night,
        "altitude": float(condition.altitude_factor),
        "carried_weight": weight,
        "weather": heat,
    }


def combined_condition_factor(condition: RaceCondition, terrain: str) -> float:
    result = 1.0
    for factor in condition_factors(condition, terrain).values():
        result *= factor
    return result
