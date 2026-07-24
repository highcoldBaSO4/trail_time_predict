from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class BacktestResult:
    """Portable, de-identified output of a rolling historical backtest."""

    records: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for record in self.records:
            required = ("target_activity_time", "actual_moving_seconds", "p10_seconds", "p50_seconds", "p90_seconds")
            missing = [name for name in required if name not in record]
            if missing:
                raise ValueError(f"回测记录缺少字段：{', '.join(missing)}")
            actual = float(record["actual_moving_seconds"])
            p10 = float(record["p10_seconds"])
            p50 = float(record["p50_seconds"])
            p90 = float(record["p90_seconds"])
            if actual <= 0 or p10 < 0 or not p10 <= p50 <= p90:
                raise ValueError("回测记录的时间或预测区间无效")
            baseline_latest = record.get("baseline_latest_activity_time")
            if baseline_latest:
                if datetime.fromisoformat(str(baseline_latest)) >= datetime.fromisoformat(str(record["target_activity_time"])):
                    raise ValueError("回测记录包含目标活动或未来活动")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        data = deepcopy(self.metadata)
        data.update(
            {
                "schema_version": "1.0",
                "records": deepcopy(self.records),
                "skipped": deepcopy(self.skipped),
                "metrics": deepcopy(self.metrics),
            }
        )
        return data
