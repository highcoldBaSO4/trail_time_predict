from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field


@dataclass
class PerformanceResult:
    """Typed boundary for one V0.4 activity diagnosis result."""

    activity_name: str
    actual_elapsed_seconds: float
    actual_timer_seconds: float
    actual_moving_seconds: float
    stopped_seconds: float
    paused_seconds: float
    nonmoving_timer_seconds: float
    predicted_moving_seconds: float
    deviation_seconds: float
    deviation_percent: float
    prediction_percentile: float
    confidence: float
    terrain_analysis: dict[str, dict[str, float]] = field(default_factory=dict)
    progress_analysis: dict[str, dict[str, float]] = field(default_factory=dict)
    segment_analysis: list[dict[str, object]] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict, repr=False)

    def validate(self) -> None:
        times = (
            self.actual_elapsed_seconds,
            self.actual_timer_seconds,
            self.actual_moving_seconds,
            self.stopped_seconds,
            self.paused_seconds,
            self.nonmoving_timer_seconds,
            self.predicted_moving_seconds,
        )
        if any(value < 0 for value in times):
            raise ValueError("活动诊断时间不能为负数")
        if self.predicted_moving_seconds <= 0:
            raise ValueError("活动诊断的预测移动时间必须大于0")
        if not 0.0 <= self.prediction_percentile <= 100.0:
            raise ValueError("活动成绩百分位必须在0到100之间")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("活动诊断可信度必须在0到1之间")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        data = deepcopy(self.metadata)
        data.update(
            {
                "schema_version": "0.4",
                "activity_name": self.activity_name,
                "actual_elapsed_seconds": round(self.actual_elapsed_seconds, 1),
                "actual_timer_seconds": round(self.actual_timer_seconds, 1),
                "actual_moving_seconds": round(self.actual_moving_seconds, 1),
                "stopped_seconds": round(self.stopped_seconds, 1),
                "paused_seconds": round(self.paused_seconds, 1),
                "nonmoving_timer_seconds": round(self.nonmoving_timer_seconds, 1),
                "predicted_moving_seconds": round(self.predicted_moving_seconds, 1),
                "deviation_seconds": round(self.deviation_seconds, 1),
                "deviation_percent": round(self.deviation_percent, 2),
                "prediction_percentile": round(self.prediction_percentile, 1),
                "confidence": round(self.confidence, 3),
                "terrain_analysis": deepcopy(self.terrain_analysis),
                "progress_analysis": deepcopy(self.progress_analysis),
                "segments": deepcopy(self.segment_analysis),
            }
        )
        return data
