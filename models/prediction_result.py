from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class PredictionResult:
    standard_moving_time_seconds: float
    adjusted_moving_time_seconds: float
    aid_station_time_seconds: float
    median_finish_time_seconds: float
    optimistic_time_seconds: float
    conservative_time_seconds: float
    confidence: float
    segment_results: list[dict[str, object]] = field(default_factory=list)
    adjustment_breakdown: dict[str, float] = field(default_factory=dict)
    risk_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
