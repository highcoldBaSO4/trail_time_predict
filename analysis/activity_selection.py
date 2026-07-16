from __future__ import annotations

from typing import Any

import pandas as pd

from analysis.activity_analysis import analyze_activity
from analysis.data_quality import diagnose_fit


ACTIVITY_TYPE_LABELS = {"trail": "越野", "road": "路跑"}
LABEL_TO_ACTIVITY_TYPE = {label: value for value, label in ACTIVITY_TYPE_LABELS.items()}


def infer_activity_type(name: str, frame: pd.DataFrame) -> str:
    """Infer the simplified road/trail type from FIT metadata and filename."""
    sub_sport = str(frame.attrs.get("sub_sport") or "").lower()
    lowered = name.lower()
    return "trail" if sub_sport == "trail" or "越野" in name or "trail" in lowered else "road"


def build_activity_review(activities: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Build editable review rows before activities enter the capability model."""
    rows: list[dict[str, Any]] = []
    for name, frame in activities.items():
        summary = analyze_activity(frame, name)
        quality = diagnose_fit(frame)
        inferred = infer_activity_type(name, frame)
        timestamps = pd.to_datetime(frame.get("timestamp"), errors="coerce", utc=True).dropna()
        rows.append(
            {
                "filename": name,
                "date": timestamps.min().date().isoformat() if not timestamps.empty else "—",
                "distance_km": float(summary["distance_km"]),
                "duration_hour": float(summary["duration_hour"]),
                "elevation_gain_m": float(summary["elevation_gain"]),
                "auto_type": inferred,
                "confirmed_type": inferred,
                "quality_level": str(quality["level"]),
                "quality_score": float(quality["score"]),
                "quality_issues": list(quality["issues"]),
                "use_for_model": bool(quality["recommended_for_model"]),
            }
        )
    return rows


def apply_activity_review(
    activities: dict[str, pd.DataFrame], rows: list[dict[str, Any]]
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Return only confirmed activities and their validated road/trail types."""
    selected: dict[str, pd.DataFrame] = {}
    activity_types: dict[str, str] = {}
    for row in rows:
        if not bool(row.get("use_for_model", False)):
            continue
        name = str(row.get("filename", ""))
        if name not in activities:
            raise ValueError(f"活动确认记录不存在对应 FIT：{name}")
        activity_type = str(row.get("confirmed_type", ""))
        if activity_type not in ACTIVITY_TYPE_LABELS:
            raise ValueError(f"不支持的活动类型：{activity_type}")
        selected[name] = activities[name]
        activity_types[name] = activity_type
    if not selected:
        raise ValueError("至少选择一个活动用于能力建模")
    return selected, activity_types
