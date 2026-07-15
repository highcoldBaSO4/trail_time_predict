from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class CapabilityValue:
    value: float
    unit: str
    source: str
    confidence: float
    sample_count: int = 0
    sample_duration_seconds: float = 0.0
    sample_distance_m: float = 0.0
    sample_elevation_m: float = 0.0


@dataclass(frozen=True)
class CurvePoint:
    x: float
    value: float
    confidence: float
    sample_count: int = 0
    sample_duration_seconds: float = 0.0
    sample_distance_m: float = 0.0
    sample_elevation_m: float = 0.0
    source: str = "personal"


@dataclass
class RunnerProfile:
    flat_aerobic_pace: CapabilityValue
    flat_threshold_pace: CapabilityValue
    uphill_curve: list[CurvePoint]
    downhill_curve: list[CurvePoint]
    fatigue_curve_flat: list[CurvePoint]
    fatigue_curve_uphill: list[CurvePoint]
    fatigue_curve_downhill: list[CurvePoint]
    source_activity_count: int
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        data = asdict(self)
        data["generated_at"] = self.generated_at.isoformat()
        return data
