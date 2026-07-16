from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TextIO

import gpxpy
import numpy as np

from config import load_config


EARTH_RADIUS_M = 6_371_000.0


def read_gpx(source: str | Path | TextIO) -> list[dict[str, float | None]]:
    """Read GPX tracks/routes into a continuous list of geographic points."""
    try:
        if hasattr(source, "read"):
            gpx = gpxpy.parse(source)
        else:
            with Path(source).open("r", encoding="utf-8-sig") as handle:
                gpx = gpxpy.parse(handle)
    except Exception as exc:
        raise ValueError(f"无法解析 GPX 文件: {exc}") from exc

    points: list[dict[str, float | None]] = []
    for track in gpx.tracks:
        for track_segment in track.segments:
            for point in track_segment.points:
                points.append(_point(point.latitude, point.longitude, point.elevation))
    if not points:
        for route in gpx.routes:
            for point in route.points:
                points.append(_point(point.latitude, point.longitude, point.elevation))
    if len(points) < 2:
        raise ValueError("GPX 文件至少需要两个有效轨迹点")
    return points


def build_race_segments(
    points: list[dict[str, float | None]], segment_distance_m: float = 100.0
) -> list[dict[str, object]]:
    """Detect natural climbs, descents and flats from distance-sampled terrain.

    ``segment_distance_m`` is the grade sampling window, not the final segment
    length. Adjacent samples of the same terrain type become one natural slope.
    Short interruptions up to two sampling windows are bridged.
    """
    if segment_distance_m <= 0:
        raise ValueError("分段距离必须大于 0")
    detail_distance_m = segment_distance_m / max(2, math.ceil(segment_distance_m / 25.0))
    sampled = _sample_route(points, detail_distance_m)
    detail_chunks = _chunks_from_samples(sampled)
    chunks = _resample_chunks(sampled, segment_distance_m)
    groups = group_terrain_chunks(chunks, segment_distance_m)

    segments: list[dict[str, object]] = []
    counters = {"uphill": 0, "downhill": 0, "flat": 0}
    for start_index, end_index, segment_type in groups:
        selected = chunks[start_index:end_index]
        start_m = float(selected[0]["start_m"])
        end_m = float(selected[-1]["end_m"])
        micro = [
            _micro_segment(item)
            for item in detail_chunks
            if float(item["start_m"]) >= start_m - 1e-6 and float(item["end_m"]) <= end_m + 1e-6
        ]
        if not micro:
            micro = [_micro_segment(item) for item in selected]
        distance = sum(float(item["distance"]) for item in micro)
        gain = sum(float(item["gain"]) for item in micro)
        loss = sum(float(item["loss"]) for item in micro)
        grade_values = [float(item["grade"]) for item in micro]
        grade = sum(float(item["grade"]) * float(item["distance"]) for item in micro) / distance
        latitude = sum(float(item["latitude"]) * float(item["distance"]) for item in micro) / distance
        longitude = sum(float(item["longitude"]) * float(item["distance"]) for item in micro) / distance
        elevation = sum(float(item["elevation"]) * float(item["distance"]) for item in micro) / distance
        elevation_available = all(bool(item["elevation_available"]) for item in micro)
        counters[segment_type] += 1
        max_grade = max(grade_values) if segment_type == "uphill" else min(grade_values) if segment_type == "downhill" else max(grade_values, key=abs)
        segments.append(
            {
                "index": len(segments) + 1,
                "name": f"{segment_type}_{counters[segment_type]}",
                "start_km": round(float(selected[0]["start_m"]) / 1000.0, 3),
                "end_km": round(float(selected[-1]["end_m"]) / 1000.0, 3),
                "distance": round(distance, 2),
                "gain": round(gain, 2),
                "loss": round(loss, 2),
                "grade": round(grade, 2),
                "max_grade": round(max_grade, 2),
                "latitude": round(latitude, 6),
                "longitude": round(longitude, 6),
                "elevation": round(elevation, 1),
                "elevation_available": elevation_available,
                "type": segment_type,
                "terrain": _terrain_label(segment_type, grade),
                "micro_segments": micro,
            }
        )
    if not segments:
        raise ValueError("GPX 路线没有可计算的有效距离")
    return segments


def route_summary(segments: list[dict[str, object]]) -> dict[str, float | int]:
    elevations = [
        float(micro[key])
        for item in segments
        for micro in list(item.get("micro_segments", []))
        for key in ("elevation_start", "elevation_end")
        if bool(micro.get("elevation_available", False))
    ]
    if not elevations:
        elevations = [float(item["elevation"]) for item in segments if bool(item.get("elevation_available", False))]
    weighted_distance = sum(float(item["distance"]) for item in segments if bool(item.get("elevation_available", False)))
    return {
        "distance_km": round(sum(float(item["distance"]) for item in segments) / 1000.0, 3),
        "elevation_gain": round(sum(float(item["gain"]) for item in segments), 1),
        "elevation_loss": round(sum(float(item["loss"]) for item in segments), 1),
        "climbs": sum(item["type"] == "uphill" for item in segments),
        "descents": sum(item["type"] == "downhill" for item in segments),
        "micro_segments": sum(len(list(item.get("micro_segments", []))) for item in segments),
        "average_elevation_m": round(
            sum(float(item["elevation"]) * float(item["distance"]) for item in segments if bool(item.get("elevation_available", False)))
            / weighted_distance,
            1,
        ) if weighted_distance > 0 else None,
        "maximum_elevation_m": round(max(elevations), 1) if elevations else None,
    }


def _terrain_chunks(
    points: list[dict[str, float | None]], sample_distance_m: float
) -> list[dict[str, float]]:
    return _chunks_from_samples(_sample_route(points, sample_distance_m))


def _sample_route(
    points: list[dict[str, float | None]], sample_distance_m: float
) -> dict[str, object]:
    cumulative = [0.0]
    elevations: list[float] = []
    for point in points:
        elevation = point.get("elevation")
        elevations.append(np.nan if elevation is None else float(elevation))
    for start, end in zip(points, points[1:]):
        cumulative.append(
            cumulative[-1]
            + haversine_m(
                float(start["latitude"]), float(start["longitude"]),
                float(end["latitude"]), float(end["longitude"]),
            )
        )
    distance_array = np.asarray(cumulative)
    elevation_array = np.asarray(elevations)
    latitude_array = np.asarray([float(point["latitude"]) for point in points])
    longitude_array = np.asarray([float(point["longitude"]) for point in points])
    unique = np.concatenate(([True], np.diff(distance_array) > 1e-6))
    distance_array = distance_array[unique]
    elevation_array = elevation_array[unique]
    latitude_array = latitude_array[unique]
    longitude_array = longitude_array[unique]
    if len(distance_array) < 2 or distance_array[-1] <= 0:
        raise ValueError("GPX 路线没有可计算的有效距离")
    known = np.isfinite(elevation_array)
    elevation_available = bool(known.sum() >= 2)
    if known.sum() < 2:
        elevation_array = np.zeros_like(distance_array)
    else:
        elevation_array = np.interp(distance_array, distance_array[known], elevation_array[known])

    sample_edges = np.arange(0.0, distance_array[-1], sample_distance_m)
    if distance_array[-1] - sample_edges[-1] > 1e-6:
        sample_edges = np.append(sample_edges, distance_array[-1])
    sampled_elevation = np.interp(sample_edges, distance_array, elevation_array)
    if elevation_available and len(sampled_elevation) >= 3:
        padded = np.pad(sampled_elevation, (1, 1), mode="edge")
        sampled_elevation = np.asarray([
            float(np.median(padded[index:index + 3])) for index in range(len(sampled_elevation))
        ])
    sampled_latitude = np.interp(sample_edges, distance_array, latitude_array)
    sampled_longitude = np.interp(sample_edges, distance_array, longitude_array)
    return {
        "distance": sample_edges,
        "elevation": sampled_elevation,
        "latitude": sampled_latitude,
        "longitude": sampled_longitude,
        "elevation_available": elevation_available,
    }


def _resample_chunks(sampled: dict[str, object], sample_distance_m: float) -> list[dict[str, float]]:
    detail_distance = np.asarray(sampled["distance"], dtype=float)
    edges = np.arange(0.0, detail_distance[-1], sample_distance_m)
    if detail_distance[-1] - edges[-1] > 1e-6:
        edges = np.append(edges, detail_distance[-1])
    coarse = {
        "distance": edges,
        "elevation": np.interp(edges, detail_distance, np.asarray(sampled["elevation"], dtype=float)),
        "latitude": np.interp(edges, detail_distance, np.asarray(sampled["latitude"], dtype=float)),
        "longitude": np.interp(edges, detail_distance, np.asarray(sampled["longitude"], dtype=float)),
        "elevation_available": bool(sampled["elevation_available"]),
    }
    return _chunks_from_samples(coarse)


def _chunks_from_samples(sampled: dict[str, object]) -> list[dict[str, float]]:
    sample_edges = np.asarray(sampled["distance"], dtype=float)
    sampled_elevation = np.asarray(sampled["elevation"], dtype=float)
    sampled_latitude = np.asarray(sampled["latitude"], dtype=float)
    sampled_longitude = np.asarray(sampled["longitude"], dtype=float)
    elevation_available = bool(sampled["elevation_available"])

    chunks: list[dict[str, float]] = []
    for start_m, end_m, start_elev, end_elev, start_lat, end_lat, start_lon, end_lon in zip(
        sample_edges,
        sample_edges[1:],
        sampled_elevation,
        sampled_elevation[1:],
        sampled_latitude,
        sampled_latitude[1:],
        sampled_longitude,
        sampled_longitude[1:],
    ):
        distance = float(end_m - start_m)
        elevation_delta = float(end_elev - start_elev)
        if distance > 1e-3:
            chunks.append(
                {
                    "start_m": float(start_m),
                    "end_m": float(end_m),
                    "start_elevation": float(start_elev),
                    "end_elevation": float(end_elev),
                    "distance": distance,
                    "elevation_delta": elevation_delta,
                    "elevation": float((start_elev + end_elev) / 2.0),
                    "elevation_available": elevation_available,
                    "latitude": float((start_lat + end_lat) / 2.0),
                    "longitude": float((start_lon + end_lon) / 2.0),
                    "grade": elevation_delta / distance * 100.0,
                }
            )
    return chunks


def _micro_segment(chunk: dict[str, float]) -> dict[str, object]:
    grade = float(chunk["grade"])
    segment_type = _terrain_type(grade)
    delta = float(chunk["elevation_delta"])
    return {
        "start_km": round(float(chunk["start_m"]) / 1000.0, 4),
        "end_km": round(float(chunk["end_m"]) / 1000.0, 4),
        "distance": round(float(chunk["distance"]), 2),
        "gain": round(max(delta, 0.0), 2),
        "loss": round(max(-delta, 0.0), 2),
        "grade": round(grade, 2),
        "latitude": round(float(chunk["latitude"]), 6),
        "longitude": round(float(chunk["longitude"]), 6),
        "elevation": round(float(chunk["elevation"]), 1),
        "elevation_start": round(float(chunk["start_elevation"]), 1),
        "elevation_end": round(float(chunk["end_elevation"]), 1),
        "elevation_available": bool(chunk["elevation_available"]),
        "type": segment_type,
        "terrain": _terrain_label(segment_type, grade),
    }


def _terrain_type(grade: float) -> str:
    flat_limit = float(load_config()["terrain"]["flat_grade_abs_percent"])
    return "uphill" if grade > flat_limit else "downhill" if grade < -flat_limit else "flat"


def group_terrain_chunks(
    chunks: list[dict[str, float]], sample_distance_m: float
) -> list[tuple[int, int, str]]:
    """Apply the shared natural-slope grouping rules to sampled terrain."""
    labels = [_terrain_type(float(chunk["grade"])) for chunk in chunks]
    labels = _bridge_short_interruptions(labels, chunks, sample_distance_m * 2.0)
    labels = _remove_insignificant_slopes(labels, chunks, sample_distance_m * 2.0)
    return _group_labels(labels)


def _group_labels(labels: list[str]) -> list[tuple[int, int, str]]:
    if not labels:
        return []
    groups: list[tuple[int, int, str]] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            groups.append((start, index, labels[start]))
            start = index
    return groups


def _bridge_short_interruptions(
    labels: list[str], chunks: list[dict[str, float]], max_distance_m: float
) -> list[str]:
    result = labels.copy()
    for start, end, label in _group_labels(labels):
        distance = sum(float(item["distance"]) for item in chunks[start:end])
        if start > 0 and end < len(labels) and distance <= max_distance_m and labels[start - 1] == labels[end]:
            result[start:end] = [labels[start - 1]] * (end - start)
    return result


def _remove_insignificant_slopes(
    labels: list[str], chunks: list[dict[str, float]], max_distance_m: float
) -> list[str]:
    result = labels.copy()
    for start, end, label in _group_labels(labels):
        if label == "flat":
            continue
        selected = chunks[start:end]
        distance = sum(float(item["distance"]) for item in selected)
        vertical = abs(sum(float(item["elevation_delta"]) for item in selected))
        if distance <= max_distance_m + 1e-6 and vertical < 10.0:
            result[start:end] = ["flat"] * (end - start)
    return result


def _terrain_label(segment_type: str, grade: float) -> str:
    magnitude = abs(grade)
    level = (
        "微" if magnitude < 5 else "缓" if magnitude < 10 else
        "中" if magnitude < 15 else "较陡" if magnitude < 20 else "陡"
    )
    return "平路" if segment_type == "flat" else f"{level}{'爬坡' if segment_type == 'uphill' else '下降'}"


def save_segments(segments: list[dict[str, object]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = math.radians(lon2 - lon1)
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(value))


def _point(latitude: float, longitude: float, elevation: float | None) -> dict[str, float | None]:
    return {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "elevation": None if elevation is None else float(elevation),
    }
