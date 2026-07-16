from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
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

    def validate(self) -> None:
        """Validate one scalar capability at the model boundary."""
        if self.value <= 0:
            raise ValueError(f"能力值必须大于0：{self.value}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"能力可信度必须在0到1之间：{self.confidence}")
        if min(self.sample_count, self.sample_duration_seconds, self.sample_distance_m, self.sample_elevation_m) < 0:
            raise ValueError("能力样本覆盖不能为负数")


@dataclass(frozen=True)
class CurvePoint:
    x: float
    value: float
    confidence: float | None
    sample_count: int = 0
    sample_duration_seconds: float = 0.0
    sample_distance_m: float = 0.0
    sample_elevation_m: float = 0.0
    source: str = "personal"
    metadata: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def validate(self) -> None:
        if self.value <= 0:
            raise ValueError(f"曲线能力值必须大于0：{self.value}")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"曲线可信度必须在0到1之间：{self.confidence}")
        if min(self.sample_count, self.sample_duration_seconds, self.sample_distance_m, self.sample_elevation_m) < 0:
            raise ValueError("曲线样本覆盖不能为负数")

    @classmethod
    def from_dict(cls, data: dict[str, Any], x_field: str, value_field: str) -> "CurvePoint":
        return cls(
            x=float(data[x_field]),
            value=float(data[value_field]),
            confidence=None if data.get("confidence") is None else float(data["confidence"]),
            sample_count=int(data.get("sample_count", 0)),
            sample_duration_seconds=float(data.get("sample_duration_seconds", 0.0)),
            sample_distance_m=float(data.get("sample_distance_m", 0.0)),
            sample_elevation_m=float(data.get("sample_elevation_m", 0.0)),
            source=str(data.get("source", "personal")),
            metadata=deepcopy(data),
        )

    def to_dict(self, x_field: str, value_field: str) -> dict[str, Any]:
        data = deepcopy(self.metadata)
        data.update(
            {
                x_field: self.x,
                value_field: self.value,
                "confidence": self.confidence,
                "sample_count": self.sample_count,
                "source": self.source,
            }
        )
        optional_samples = {
            "sample_duration_seconds": self.sample_duration_seconds,
            "sample_distance_m": self.sample_distance_m,
            "sample_elevation_m": self.sample_elevation_m,
        }
        for key, value in optional_samples.items():
            if key in self.metadata or value != 0:
                data[key] = value
        return data


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

    @classmethod
    def from_profile_dict(cls, data: dict[str, Any]) -> "RunnerProfile":
        """Create and validate a typed view of the compatible profile payload."""
        try:
            flat = data["flat"]
            uphill = data["uphill"]
            downhill = data["downhill"]
            fatigue = data["fatigue"]
            uphill_points = uphill.get("curve") or [
                {"grade": grade, "value": float(uphill[key]), "confidence": 0.2, "source": "legacy"}
                for grade, key in zip((3.0, 7.5, 12.5, 18.0), ("1_percent", "5_percent", "10_percent", "15_percent"))
            ]
            downhill_points = downhill.get("curve") or [
                {"grade": grade, "speed_mps": float(downhill[key]["speed_mps"]),
                 "vertical_speed_mph": float(downhill[key].get("vertical_speed_mph", 0.0)),
                 "confidence": 0.2, "source": "legacy"}
                for grade, key in zip((-3.0, -7.5, -12.5, -18.0), ("-1_percent", "-5_percent", "-10_percent", "-15_percent"))
            ]
            legacy_fatigue = [
                {"hour": 0.0, "factor": 1.0, "sample_count": 0, "source": "anchor", "confidence": None},
                {"hour": 3.0, "factor": float(fatigue["3h"]), "confidence": 0.2, "source": "legacy"},
                {"hour": 5.0, "factor": float(fatigue["5h"]), "confidence": 0.2, "source": "legacy"},
                {"hour": 8.0, "factor": float(fatigue["8h"]), "confidence": 0.2, "source": "legacy"},
            ]
            flat_source = str(flat.get("source", "personal"))
            flat_samples = int(flat.get("qualified_segments", 0))
            flat_duration = float(flat.get("sample_duration_seconds", 0.0))
            flat_distance = float(flat.get("sample_distance_km", 0.0)) * 1000.0
            flat_confidence = float(flat.get("confidence", 0.2))
            generated = data.get("generated_at")
            generated_at = datetime.fromisoformat(str(generated)) if generated else datetime.now(timezone.utc)
            profile = cls(
                flat_aerobic_pace=CapabilityValue(float(flat["aerobic_pace"]), "seconds_per_km", flat_source,
                                                   flat_confidence, flat_samples, flat_duration, flat_distance),
                flat_threshold_pace=CapabilityValue(float(flat["threshold_pace"]), "seconds_per_km", flat_source,
                                                    flat_confidence, flat_samples, flat_duration, flat_distance),
                uphill_curve=[CurvePoint.from_dict(point, "grade", "value") for point in uphill_points],
                downhill_curve=[CurvePoint.from_dict(point, "grade", "speed_mps") for point in downhill_points],
                fatigue_curve_flat=[CurvePoint.from_dict(point, "hour", "factor") for point in fatigue.get("flat", legacy_fatigue)],
                fatigue_curve_uphill=[CurvePoint.from_dict(point, "hour", "factor") for point in fatigue.get("uphill", legacy_fatigue)],
                fatigue_curve_downhill=[CurvePoint.from_dict(point, "hour", "factor") for point in fatigue.get("downhill", legacy_fatigue)],
                source_activity_count=int(data["sample_count"]),
                generated_at=generated_at,
                metadata=deepcopy(data),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"个人能力画像结构无效：{exc}") from exc
        profile.validate()
        return profile

    def validate(self) -> None:
        """Validate all core capabilities before prediction or serialization."""
        self.flat_aerobic_pace.validate()
        self.flat_threshold_pace.validate()
        if self.source_activity_count <= 0:
            raise ValueError("个人能力画像至少需要一个来源活动")
        for name, curve in (
            ("上坡", self.uphill_curve),
            ("下坡", self.downhill_curve),
            ("平路疲劳", self.fatigue_curve_flat),
            ("上坡疲劳", self.fatigue_curve_uphill),
            ("下坡疲劳", self.fatigue_curve_downhill),
        ):
            if not curve:
                raise ValueError(f"{name}能力曲线不能为空")
            for point in curve:
                point.validate()

    def to_profile_dict(self) -> dict[str, Any]:
        """Return the existing JSON shape after typed validation."""
        self.validate()
        data = deepcopy(self.metadata)
        data["generated_at"] = self.generated_at.isoformat()
        data["sample_count"] = self.source_activity_count
        data["flat"]["aerobic_pace"] = self.flat_aerobic_pace.value
        data["flat"]["threshold_pace"] = self.flat_threshold_pace.value
        data["uphill"]["curve"] = [point.to_dict("grade", "value") for point in self.uphill_curve]
        data["downhill"]["curve"] = [point.to_dict("grade", "speed_mps") for point in self.downhill_curve]
        data["fatigue"]["flat"] = [point.to_dict("hour", "factor") for point in self.fatigue_curve_flat]
        data["fatigue"]["uphill"] = [point.to_dict("hour", "factor") for point in self.fatigue_curve_uphill]
        data["fatigue"]["downhill"] = [point.to_dict("hour", "factor") for point in self.fatigue_curve_downhill]
        return data

    def to_dict(self) -> dict[str, Any]:
        """Alias used by callers that serialize domain models."""
        return self.to_profile_dict()
