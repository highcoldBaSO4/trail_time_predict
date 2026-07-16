from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field


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
    metadata: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PredictionResult":
        """Build a typed result from the public prediction payload."""
        try:
            result = cls(
                standard_moving_time_seconds=float(data["standard_moving_time_seconds"]),
                adjusted_moving_time_seconds=float(data["adjusted_moving_time_seconds"]),
                aid_station_time_seconds=float(data["aid_station_time_seconds"]),
                median_finish_time_seconds=float(data["median_finish_time_seconds"]),
                optimistic_time_seconds=float(data["optimistic_time_seconds"]),
                conservative_time_seconds=float(data["conservative_time_seconds"]),
                confidence=float(data["confidence"]),
                segment_results=list(data.get("segments", [])),
                adjustment_breakdown={str(key): float(value) for key, value in dict(data.get("adjustment_breakdown", {})).items()},
                risk_notes=[str(note) for note in data.get("risk_notes", [])],
                metadata=deepcopy(data),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"预测结果结构无效：{exc}") from exc
        result.validate()
        return result

    def validate(self) -> None:
        """Enforce ordering and range guarantees required by V0.3."""
        times = (
            self.standard_moving_time_seconds,
            self.adjusted_moving_time_seconds,
            self.aid_station_time_seconds,
            self.optimistic_time_seconds,
            self.median_finish_time_seconds,
            self.conservative_time_seconds,
        )
        if any(value < 0 for value in times):
            raise ValueError("预测时间不能为负数")
        if not self.optimistic_time_seconds <= self.median_finish_time_seconds <= self.conservative_time_seconds:
            raise ValueError("预测区间必须满足 P10 <= P50 <= P90")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("预测可信度必须在0到1之间")

    def to_dict(self) -> dict[str, object]:
        """Return the existing JSON shape after typed validation."""
        self.validate()
        data = deepcopy(self.metadata)
        data.update(
            {
                "standard_moving_time_seconds": self.standard_moving_time_seconds,
                "adjusted_moving_time_seconds": self.adjusted_moving_time_seconds,
                "aid_station_time_seconds": self.aid_station_time_seconds,
                "median_finish_time_seconds": self.median_finish_time_seconds,
                "optimistic_time_seconds": self.optimistic_time_seconds,
                "conservative_time_seconds": self.conservative_time_seconds,
                "confidence": self.confidence,
                "segments": deepcopy(self.segment_results),
                "adjustment_breakdown": dict(self.adjustment_breakdown),
                "risk_notes": list(self.risk_notes),
            }
        )
        return data
