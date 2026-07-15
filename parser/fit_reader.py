from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Callable
import warnings

import numpy as np
import pandas as pd
from fitparse import FitFile
from fitparse.records import Crc


FIT_COLUMNS = [
    "timestamp",
    "latitude",
    "longitude",
    "distance",
    "altitude",
    "heart_rate",
    "cadence",
    "power",
]

# Some exporters encode event.data as a one-byte uint32. The definition is
# internally inconsistent, but the data itself is one byte and the file CRC is
# otherwise valid. Restrict the workaround to this exact event definition.
MALFORMED_EVENT_DEFINITION = bytes.fromhex(
    "4a 00 00 15 00 05 fd 04 86 00 01 00 01 01 00 04 01 02 03 01 86"
)
FIXED_EVENT_DEFINITION = MALFORMED_EVENT_DEFINITION[:-1] + b"\x02"


def _semicircles_to_degrees(value: object) -> float:
    if value is None:
        return np.nan
    return float(value) * 180.0 / (2**31)


def read_fit(
    source: str | Path | BinaryIO,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Read FIT record messages into a normalized DataFrame.

    Distance and altitude use metres. Missing optional sensor values remain NaN.
    Corrupt files and files without record messages raise ValueError with context.
    """
    rows: list[dict[str, object]] = []
    session_values: dict[str, object] = {}
    source_name = Path(source).name if isinstance(source, (str, Path)) else getattr(source, "name", "上传文件")
    try:
        fit_source, patched = _prepare_fit_source(source)
        if patched:
            warnings.warn(
                f"FIT 文件 {source_name} 含 {patched} 条导出器错误的 event.data 定义；"
                "已在内存中按 uint8 兼容解析，原文件未修改。",
                RuntimeWarning,
                stacklevel=2,
            )
        fit_file = FitFile(fit_source, check_crc=not patched)
        _emit(progress, f"    开始解析 {source_name}")
        for message in fit_file.get_messages("record"):
            values = message.get_values()
            rows.append(
                {
                    "timestamp": values.get("timestamp"),
                    "latitude": _semicircles_to_degrees(values.get("position_lat")),
                    "longitude": _semicircles_to_degrees(values.get("position_long")),
                    "distance": values.get("distance"),
                    "altitude": values.get("enhanced_altitude", values.get("altitude")),
                    "heart_rate": values.get("heart_rate"),
                    "cadence": values.get("cadence"),
                    "power": values.get("power"),
                }
            )
            if len(rows) % 5000 == 0:
                _emit(progress, f"    {source_name}：已解析 {len(rows):,} 条轨迹记录")
        sessions = list(fit_file.get_messages("session"))
        session_values = sessions[-1].get_values() if sessions else {}
    except Exception as exc:  # fitparse exposes several decoder exception types
        if not rows:
            raise ValueError(f"无法解析 FIT 文件 {source_name}: {exc}") from exc
        warnings.warn(
            f"FIT 文件 {source_name} 尾部或扩展字段异常；已保留异常前的 {len(rows)} 条记录。原始错误: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    if not rows:
        raise ValueError("FIT 文件中没有 record 轨迹记录")

    frame = pd.DataFrame(rows, columns=FIT_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    for column in FIT_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )
    if frame.empty:
        raise ValueError("FIT 文件没有有效的时间戳记录")
    frame.attrs["sport"] = session_values.get("sport")
    frame.attrs["sub_sport"] = session_values.get("sub_sport")
    _emit(progress, f"    {source_name}：完成，共 {len(frame):,} 条有效记录")
    return frame


def read_fit_directory(
    directory: str | Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """Read every .fit file in a directory, keyed by filename."""
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"FIT 目录不存在: {path}")
    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".fit")
    if not files:
        raise FileNotFoundError(f"目录中没有 FIT 文件: {path}")
    activities: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    _emit(progress, f"发现 {len(files)} 个 FIT 文件")
    for index, file in enumerate(files, start=1):
        _emit(progress, f"  [{index}/{len(files)}] {file.name}")
        try:
            activities[file.name] = read_fit(file, progress=progress)
        except ValueError as exc:
            failures.append(f"{file.name}: {exc}")
            _emit(progress, f"    跳过：{exc}")
    if failures:
        warnings.warn(
            "以下 FIT 文件无法读取，已跳过：\n- " + "\n- ".join(failures),
            RuntimeWarning,
            stacklevel=2,
        )
    if not activities:
        raise ValueError("没有可用于建模的有效 FIT 文件")
    return activities


def _prepare_fit_source(source: str | Path | BinaryIO) -> tuple[object, int]:
    """Return a fitparse source, applying one verified exporter workaround."""
    raw: bytes | None = None
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_file():
            raw = path.read_bytes()
        else:
            return str(source), 0
    elif hasattr(source, "getvalue"):
        raw = bytes(source.getvalue())

    if raw is None:
        return source, 0
    count = raw.count(MALFORMED_EVENT_DEFINITION)
    if not count:
        return io.BytesIO(raw), 0
    if Crc.calculate(raw) != 0:
        # Do not repair a file whose original CRC is already invalid.
        return io.BytesIO(raw), 0
    return io.BytesIO(raw.replace(MALFORMED_EVENT_DEFINITION, FIXED_EVENT_DEFINITION)), count


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
