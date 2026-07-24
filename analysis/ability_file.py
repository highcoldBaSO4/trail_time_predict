from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Callable

import numpy as np
import pandas as pd

from analysis.capability import build_runner_profile
from config import load_config
from models import RunnerProfile


ABILITY_FILE_EXTENSION = "ttp-profile"
ABILITY_FILE_MIME = "application/vnd.trail-time-predict.profile+zip"
MANIFEST_NAME = "manifest.json"
PROFILE_NAME = "profile.json"
DEFAULT_ABILITY_FILE_CONFIG = {
    "schema_version": "1.0",
    "current_half_life_days": 90.0,
    "long_term_half_life_days": 365.0,
    "maximum_activities": 200,
    "maximum_uncompressed_mb": 200,
}


@dataclass
class AbilityBundle:
    profile: dict[str, object]
    activities: dict[str, pd.DataFrame] = field(default_factory=dict)
    activity_types: dict[str, str] = field(default_factory=dict)
    activity_hashes: dict[str, str] = field(default_factory=dict)
    manifest: dict[str, object] = field(default_factory=dict)

    @property
    def supports_update(self) -> bool:
        return bool(self.activities)


def build_ability_bundle(
    activities: dict[str, pd.DataFrame],
    activity_types: dict[str, str],
    activity_hashes: dict[str, str] | None = None,
    reference_time: datetime | None = None,
    progress: Callable[[str], None] | None = None,
) -> AbilityBundle:
    if not activities:
        raise ValueError("至少需要一个活动才能生成个人能力文件")
    missing_types = sorted(set(activities) - set(activity_types))
    if missing_types:
        raise ValueError(f"以下活动缺少类型确认：{', '.join(missing_types)}")

    config = _ability_file_config()
    maximum = int(config["maximum_activities"])
    if len(activities) > maximum:
        raise ValueError(f"个人能力文件最多保存 {maximum} 个活动")
    reference = _normalize_datetime(reference_time)
    hashes = {
        name: (activity_hashes or {}).get(name) or dataframe_fingerprint(name, frame)
        for name, frame in activities.items()
    }
    if len(set(hashes.values())) != len(hashes):
        raise ValueError("活动列表中存在内容重复的 FIT，请移除重复项后重试")

    current_half_life = float(config["current_half_life_days"])
    long_half_life = float(config["long_term_half_life_days"])
    _emit(progress, "按当前能力半衰期生成个人能力画像……")
    profile = build_runner_profile(
        activities,
        activity_types,
        progress=progress,
        reference_time=reference,
        recency_half_life_days=current_half_life,
    )
    _emit(progress, "生成长期能力基线……")
    long_term = build_runner_profile(
        activities,
        activity_types,
        progress=progress,
        reference_time=reference,
        recency_half_life_days=long_half_life,
    )
    profile["long_term_baseline"] = _baseline_summary(long_term)
    profile["ability_file"] = {
        "schema_version": str(config["schema_version"]),
        "reference_time": reference.isoformat(),
        "current_half_life_days": current_half_life,
        "long_term_half_life_days": long_half_life,
        "activity_count": len(activities),
        "stores_activity_evidence": True,
    }
    profile = RunnerProfile.from_profile_dict(profile).to_profile_dict()
    manifest = _build_manifest(activities, activity_types, hashes, reference)
    return AbilityBundle(
        profile=profile,
        activities={name: frame.copy() for name, frame in activities.items()},
        activity_types=dict(activity_types),
        activity_hashes=hashes,
        manifest=manifest,
    )


def update_ability_bundle(
    existing: AbilityBundle,
    new_activities: dict[str, pd.DataFrame],
    new_activity_types: dict[str, str],
    new_activity_hashes: dict[str, str] | None = None,
    reference_time: datetime | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[AbilityBundle, list[str]]:
    if not existing.supports_update:
        raise ValueError("该能力文件不包含活动证据，不能增量更新；请重新上传历史 FIT 创建新版文件")
    combined = {name: frame.copy() for name, frame in existing.activities.items()}
    combined_types = dict(existing.activity_types)
    combined_hashes = dict(existing.activity_hashes)
    known_hashes = set(combined_hashes.values())
    skipped: list[str] = []

    for original_name, frame in new_activities.items():
        digest = (new_activity_hashes or {}).get(original_name) or dataframe_fingerprint(original_name, frame)
        if digest in known_hashes:
            skipped.append(original_name)
            continue
        name = _unique_activity_name(original_name, digest, set(combined))
        combined[name] = frame.copy()
        combined_types[name] = new_activity_types[original_name]
        combined_hashes[name] = digest
        known_hashes.add(digest)

    if not combined:
        raise ValueError("没有可用于更新的活动")
    bundle = build_ability_bundle(
        combined,
        combined_types,
        combined_hashes,
        reference_time=reference_time,
        progress=progress,
    )
    return bundle, skipped


def serialize_ability_bundle(bundle: AbilityBundle) -> bytes:
    if not bundle.supports_update:
        raise ValueError("个人能力文件缺少可更新的活动证据")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr(MANIFEST_NAME, _json_bytes(bundle.manifest))
        archive.writestr(PROFILE_NAME, _json_bytes(bundle.profile))
        for name, frame in bundle.activities.items():
            digest = bundle.activity_hashes[name]
            payload = {
                "name": name,
                "sha256": digest,
                "activity_type": bundle.activity_types[name],
                "attrs": _json_safe(dict(frame.attrs)),
                "frame": json.loads(frame.to_json(orient="split", date_format="iso", date_unit="ms")),
            }
            archive.writestr(f"activities/{digest}.json", _json_bytes(payload))
    return buffer.getvalue()


def load_ability_bundle(source: bytes | bytearray | Any) -> AbilityBundle:
    raw = bytes(source.getvalue()) if hasattr(source, "getvalue") else bytes(source)
    if not raw:
        raise ValueError("个人能力文件为空")
    if not zipfile.is_zipfile(io.BytesIO(raw)):
        raise ValueError("仅支持新版 .ttp-profile 个人能力文件，请重新上传历史 FIT 生成新版文件")

    config = _ability_file_config()
    maximum_bytes = int(float(config["maximum_uncompressed_mb"]) * 1024 * 1024)
    maximum_activities = int(config["maximum_activities"])
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            infos = archive.infolist()
            if len(infos) > maximum_activities + 10:
                raise ValueError("个人能力文件包含过多条目")
            if sum(info.file_size for info in infos) > maximum_bytes:
                raise ValueError("个人能力文件解压后体积超过安全限制")
            for info in infos:
                path = PurePosixPath(info.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("个人能力文件包含不安全路径")
            names = {info.filename for info in infos}
            if MANIFEST_NAME not in names or PROFILE_NAME not in names:
                raise ValueError("个人能力文件缺少 manifest.json 或 profile.json")
            manifest = _read_json(archive.read(MANIFEST_NAME), "能力文件清单")
            if manifest.get("format") != "trail-time-predict-ability":
                raise ValueError("不是受支持的新版 .ttp-profile 个人能力文件")
            profile = _read_json(archive.read(PROFILE_NAME), "个人能力画像")
            _validate_profile(profile)
            activities: dict[str, pd.DataFrame] = {}
            activity_types: dict[str, str] = {}
            activity_hashes: dict[str, str] = {}
            records = manifest.get("activities", [])
            if not isinstance(records, list):
                raise ValueError("个人能力文件的活动清单无效")
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError("个人能力文件的活动记录无效")
                digest = str(record.get("sha256", ""))
                entry = f"activities/{digest}.json"
                if not digest or entry not in names:
                    raise ValueError("个人能力文件缺少活动证据")
                payload = _read_json(archive.read(entry), "活动证据")
                name = str(payload.get("name", record.get("name", "activity.fit")))
                frame = _frame_from_payload(payload)
                activities[name] = frame
                activity_types[name] = str(payload.get("activity_type", record.get("activity_type", "road")))
                activity_hashes[name] = digest
    except zipfile.BadZipFile as exc:
        raise ValueError(f"个人能力文件已损坏：{exc}") from exc
    return AbilityBundle(profile, activities, activity_types, activity_hashes, manifest)


def refresh_ability_bundle(
    bundle: AbilityBundle,
    reference_time: datetime | None = None,
    progress: Callable[[str], None] | None = None,
) -> AbilityBundle:
    if not bundle.supports_update:
        return bundle
    refreshed = build_ability_bundle(
        bundle.activities,
        bundle.activity_types,
        bundle.activity_hashes,
        reference_time=reference_time,
        progress=progress,
    )
    # A calibration model is valid only for exactly the activity evidence it
    # was backtested against.  Keep a matching, portable model when reopening
    # an ability file; new/removed FITs automatically invalidate it.
    calibration = bundle.profile.get("prediction_calibration")
    if isinstance(calibration, dict) and calibration.get("activity_evidence_fingerprint") == _activity_evidence_fingerprint(bundle):
        refreshed.profile["prediction_calibration"] = dict(calibration)
    return refreshed


def profile_before_activity(
    bundle: AbilityBundle,
    target_activity: pd.DataFrame,
    target_hash: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, object], list[str]]:
    if not bundle.supports_update:
        return bundle.profile, []
    target_times = pd.to_datetime(target_activity.get("timestamp"), errors="coerce", utc=True).dropna()
    if target_times.empty:
        raise ValueError("待诊断 FIT 缺少有效活动时间")
    target_start = target_times.min().to_pydatetime()
    selected: dict[str, pd.DataFrame] = {}
    selected_types: dict[str, str] = {}
    selected_hashes: dict[str, str] = {}
    excluded: list[str] = []
    for name, frame in bundle.activities.items():
        digest = bundle.activity_hashes[name]
        times = pd.to_datetime(frame.get("timestamp"), errors="coerce", utc=True).dropna()
        starts_before_target = not times.empty and times.min().to_pydatetime() < target_start
        if (target_hash and digest == target_hash) or not starts_before_target:
            excluded.append(name)
            continue
        selected[name] = frame
        selected_types[name] = bundle.activity_types[name]
        selected_hashes[name] = digest
    if not selected:
        raise ValueError("个人能力文件中没有早于待诊断活动的基准 FIT")
    snapshot = build_ability_bundle(
        selected,
        selected_types,
        selected_hashes,
        reference_time=target_start,
        progress=progress,
    )
    return snapshot.profile, excluded


def activity_hashes_from_uploads(files: list[Any]) -> dict[str, str]:
    return {item.name: hashlib.sha256(item.getvalue()).hexdigest() for item in files}


def dataframe_fingerprint(name: str, frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8", errors="replace"))
    stable_columns = [column for column in ("timestamp", "distance", "altitude") if column in frame]
    digest.update(frame[stable_columns].to_json(orient="split", date_format="iso").encode("utf-8"))
    return digest.hexdigest()


def _activity_evidence_fingerprint(bundle: AbilityBundle) -> str:
    payload = json.dumps(sorted(bundle.activity_hashes.values()), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def ability_bundle_summary(bundle: AbilityBundle) -> dict[str, object]:
    dates: list[pd.Timestamp] = []
    for frame in bundle.activities.values():
        values = pd.to_datetime(frame.get("timestamp"), errors="coerce", utc=True).dropna()
        if not values.empty:
            dates.extend((values.min(), values.max()))
    return {
        "activity_count": len(bundle.activities) or int(bundle.profile.get("sample_count", 0)),
        "earliest_activity": min(dates).isoformat() if dates else None,
        "latest_activity": max(dates).isoformat() if dates else None,
        "generated_at": bundle.profile.get("generated_at"),
        "reference_time": bundle.profile.get("reference_time", bundle.profile.get("generated_at")),
        "supports_update": bundle.supports_update,
        "flat_confidence": float(dict(bundle.profile.get("flat", {})).get("confidence", 0.2)),
    }


def _build_manifest(
    activities: dict[str, pd.DataFrame],
    activity_types: dict[str, str],
    hashes: dict[str, str],
    reference: datetime,
) -> dict[str, object]:
    records = []
    for name, frame in activities.items():
        times = pd.to_datetime(frame.get("timestamp"), errors="coerce", utc=True).dropna()
        records.append(
            {
                "name": name,
                "sha256": hashes[name],
                "activity_type": activity_types[name],
                "start_time": times.min().isoformat() if not times.empty else None,
                "end_time": times.max().isoformat() if not times.empty else None,
                "record_count": len(frame),
            }
        )
    records.sort(key=lambda item: str(item.get("start_time") or ""))
    return {
        "format": "trail-time-predict-ability",
        "schema_version": str(_ability_file_config()["schema_version"]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_time": reference.isoformat(),
        "activity_count": len(records),
        "activities": records,
    }


def _frame_from_payload(payload: dict[str, object]) -> pd.DataFrame:
    frame_data = payload.get("frame")
    if not isinstance(frame_data, dict):
        raise ValueError("个人能力文件中的活动表无效")
    frame = pd.read_json(io.StringIO(json.dumps(frame_data)), orient="split")
    if "timestamp" not in frame:
        raise ValueError("个人能力文件中的活动缺少时间戳")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if frame.empty:
        raise ValueError("个人能力文件中的活动没有有效记录")
    attrs = payload.get("attrs", {})
    if isinstance(attrs, dict):
        frame.attrs.update(attrs)
    return frame


def _validate_profile(profile: object) -> None:
    if not isinstance(profile, dict):
        raise ValueError("个人能力画像必须是 JSON 对象")
    RunnerProfile.from_profile_dict(profile)


def _read_json(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}不是有效的 UTF-8 JSON：{exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label}结构无效")
    return value


def _json_bytes(value: object) -> bytes:
    return json.dumps(_json_safe(value), ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _baseline_summary(profile: dict[str, object]) -> dict[str, object]:
    flat = dict(profile.get("flat", {}))
    uphill = dict(profile.get("uphill", {}))
    downhill = dict(profile.get("downhill", {}))
    return {
        "reference_time": profile.get("reference_time"),
        "recency_weighting": profile.get("recency_weighting"),
        "flat_aerobic_pace": flat.get("aerobic_pace"),
        "flat_threshold_pace": flat.get("threshold_pace"),
        "uphill": {key: uphill.get(key) for key in ("1_percent", "5_percent", "10_percent", "15_percent", "20_percent")},
        "downhill_speed_mps": {
            key: dict(downhill.get(key, {})).get("speed_mps")
            for key in ("-1_percent", "-5_percent", "-10_percent", "-15_percent", "-20_percent")
        },
        "fatigue_stages": profile.get("fatigue_stages", {}),
    }


def _unique_activity_name(name: str, digest: str, existing: set[str]) -> str:
    if name not in existing:
        return name
    stem, dot, suffix = name.rpartition(".")
    base = stem if dot else name
    extension = f".{suffix}" if dot else ""
    candidate = f"{base}-{digest[:8]}{extension}"
    index = 2
    while candidate in existing:
        candidate = f"{base}-{digest[:8]}-{index}{extension}"
        index += 1
    return candidate


def _normalize_datetime(value: datetime | None) -> datetime:
    result = value or datetime.now(timezone.utc)
    if result.tzinfo is None:
        return result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _ability_file_config() -> dict[str, object]:
    """Read V0.5 options while tolerating a cached pre-V0.5 config mapping."""
    configured = load_config().get("ability_file", {})
    if not isinstance(configured, dict):
        configured = {}
    return {**DEFAULT_ABILITY_FILE_CONFIG, **configured}


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
