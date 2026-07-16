from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from runtime_paths import weather_cache_directory

import numpy as np
import pandas as pd

from config import load_config


HOURLY_VARIABLES = (
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "direct_radiation",
)


def enrich_activity_with_historical_weather(
    activity: pd.DataFrame,
    activity_name: str = "activity",
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Attach hourly reanalysis weather to FIT records by time and route location.

    Explicit ambient temperatures are preserved. Wrist-device temperature is
    never converted into ambient temperature; it remains a relative signal.
    Network or data-source failures safely leave the activity unmodified.
    """
    data = activity.copy()
    config = load_config()["historical_weather"]
    ambient = _numeric_column(data, "temperature")
    if len(data) and ambient.notna().mean() >= float(config["skip_when_ambient_coverage"]):
        data.attrs["historical_weather"] = {"source": "explicit_ambient", "status": "not_required"}
        return data
    representatives = _representative_points(data, config)
    timestamps = pd.to_datetime(data.get("timestamp"), errors="coerce", utc=True)
    if not representatives or timestamps.dropna().empty:
        data.attrs["historical_weather"] = {"source": "unavailable", "status": "missing_fit_location_or_time"}
        _emit(progress, f"    {activity_name}：缺少有效时间或经纬度，无法匹配历史天气")
        return data
    try:
        payload, cache_hit = _weather_payload(representatives, timestamps, config)
        responses = payload if isinstance(payload, list) else [payload]
        if len(responses) != len(representatives):
            raise ValueError("天气服务返回的地点数量与请求不一致")
        _apply_weather(data, representatives, responses, config)
    except (OSError, ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError) as exc:
        data.attrs["historical_weather"] = {
            "source": "unavailable",
            "status": "request_failed",
            "reason": str(exc),
        }
        _emit(progress, f"    {activity_name}：历史天气不可用，已安全降级（{exc}）")
        return data
    matched = int(pd.to_numeric(data["temperature"], errors="coerce").notna().sum())
    data.attrs["temperature_calibration"] = {
        "source": "historical_weather",
        "absolute_temperature_available": matched > 0,
        "model_weight": float(config["model_weight"]),
        "local_exposure_weight_adjustment": True,
    }
    data.attrs["historical_weather"] = {
        "source": "open_meteo_archive",
        "status": "cache" if cache_hit else "downloaded",
        "representative_point_count": len(representatives),
        "matched_record_count": matched,
        "base_model_weight": float(config["model_weight"]),
        "local_exposure_weight_adjustment": True,
        "spatial_resolution_note": "约9–11km再分析网格，山区微气候可能存在偏差",
    }
    _emit(
        progress,
        f"    {activity_name}：历史天气{'读取缓存' if cache_hit else '下载完成'}，"
        f"{len(representatives)} 个代表点，匹配 {matched:,} 条记录",
    )
    return data


def _representative_points(frame: pd.DataFrame, config: dict[str, object]) -> list[dict[str, float]]:
    required = {"latitude", "longitude", "timestamp"}
    if not required <= set(frame.columns):
        return []
    points = frame[["latitude", "longitude", "timestamp"]].copy()
    points["elevation"] = pd.to_numeric(frame.get("altitude"), errors="coerce")
    points["latitude"] = pd.to_numeric(points["latitude"], errors="coerce")
    points["longitude"] = pd.to_numeric(points["longitude"], errors="coerce")
    points = points[
        points["latitude"].between(-90.0, 90.0)
        & points["longitude"].between(-180.0, 180.0)
    ]
    if points.empty:
        return []
    distance_threshold = float(config["representative_distance_km"]) * 1000.0
    elevation_threshold = float(config["representative_elevation_change_m"])
    selected = [points.iloc[0]]
    for _, point in points.iloc[1:].iterrows():
        previous = selected[-1]
        distance = _haversine_m(previous["latitude"], previous["longitude"], point["latitude"], point["longitude"])
        elevation_change = (
            abs(float(point["elevation"]) - float(previous["elevation"]))
            if pd.notna(point["elevation"]) and pd.notna(previous["elevation"]) else 0.0
        )
        if distance >= distance_threshold or elevation_change >= elevation_threshold:
            selected.append(point)
    if not selected[-1].equals(points.iloc[-1]):
        selected.append(points.iloc[-1])
    maximum = int(config["maximum_representative_points"])
    if len(selected) > maximum:
        indices = np.linspace(0, len(selected) - 1, maximum).round().astype(int)
        selected = [selected[index] for index in np.unique(indices)]
    return [
        {
            "latitude": round(float(point["latitude"]), 5),
            "longitude": round(float(point["longitude"]), 5),
            "elevation": float(point["elevation"]) if pd.notna(point["elevation"]) else float("nan"),
        }
        for point in selected
    ]


def _weather_payload(
    representatives: list[dict[str, float]],
    timestamps: pd.Series,
    config: dict[str, object],
) -> tuple[object, bool]:
    valid_time = timestamps.dropna()
    params = {
        "latitude": ",".join(f"{point['latitude']:.5f}" for point in representatives),
        "longitude": ",".join(f"{point['longitude']:.5f}" for point in representatives),
        "elevation": ",".join(
            "nan" if not np.isfinite(point["elevation"]) else f"{point['elevation']:.1f}"
            for point in representatives
        ),
        "start_date": valid_time.min().date().isoformat(),
        "end_date": valid_time.max().date().isoformat(),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "GMT",
        "wind_speed_unit": "ms",
        "cell_selection": "land",
    }
    encoded = urlencode(params)
    cache_dir = weather_cache_directory(str(config["cache_directory"]))
    cache_key = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8")), True
    request = Request(
        f"{config['endpoint']}?{encoded}",
        headers={"User-Agent": "trail-time-predictor/0.3"},
    )
    with urlopen(request, timeout=float(config["timeout_seconds"])) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, dict) and payload.get("error"):
        raise ValueError(str(payload.get("reason", "历史天气服务返回错误")))
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload, False


def _apply_weather(
    frame: pd.DataFrame,
    representatives: list[dict[str, float]],
    responses: list[dict[str, object]],
    config: dict[str, object],
) -> None:
    latitude = pd.to_numeric(frame["latitude"], errors="coerce").to_numpy(dtype=float)
    longitude = pd.to_numeric(frame["longitude"], errors="coerce").to_numpy(dtype=float)
    location_valid = np.isfinite(latitude) & np.isfinite(longitude)
    nearest = np.zeros(len(frame), dtype=int)
    best_distance = np.full(len(frame), np.inf)
    for index, point in enumerate(representatives):
        distance = _approximate_distance(latitude, longitude, point["latitude"], point["longitude"])
        update = distance < best_distance
        nearest[update] = index
        best_distance[update] = distance[update]
    target_time = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    target_seconds = target_time.astype("int64").to_numpy(dtype=float) / 1_000_000_000.0
    mapped = {variable: np.full(len(frame), np.nan) for variable in HOURLY_VARIABLES}
    for index, response in enumerate(responses):
        hourly = dict(response["hourly"])
        weather_time = pd.to_datetime(hourly["time"], errors="coerce", utc=True)
        valid_time = weather_time.notna()
        source_seconds = weather_time[valid_time].astype("int64").to_numpy(dtype=float) / 1_000_000_000.0
        mask = (nearest == index) & location_valid & target_time.notna().to_numpy()
        for variable in HOURLY_VARIABLES:
            values = pd.to_numeric(pd.Series(hourly.get(variable, [])), errors="coerce")[valid_time].to_numpy(dtype=float)
            finite = np.isfinite(source_seconds) & np.isfinite(values)
            if finite.any():
                mapped[variable][mask] = np.interp(target_seconds[mask], source_seconds[finite], values[finite])
    existing_temperature = _numeric_column(frame, "temperature")
    weather_temperature = pd.Series(mapped["temperature_2m"], index=frame.index)
    fill = existing_temperature.isna() & weather_temperature.notna()
    frame["temperature"] = existing_temperature.where(~fill, weather_temperature)
    frame["temperature_weight"] = _numeric_column(frame, "temperature_weight").fillna(0.0)
    exposure_index = _local_heat_exposure_index(frame, mapped, config)
    exposure_config = dict(config.get("local_exposure", {}))
    maximum_reduction = float(exposure_config.get("maximum_weight_reduction", 0.35))
    weather_weight = float(config["model_weight"]) * (1.0 - maximum_reduction * exposure_index)
    frame.loc[fill, "temperature_weight"] = weather_weight[fill.to_numpy()]
    frame["weather_source"] = np.where(fill, "open_meteo_archive", None)
    frame["weather_model_weight"] = np.where(fill, weather_weight, 0.0)
    frame["local_heat_exposure_index"] = np.where(fill, exposure_index, np.nan)
    for variable, values in mapped.items():
        frame[f"weather_{variable}"] = values
    frame["weather_distance_to_representative_m"] = best_distance


def _local_heat_exposure_index(
    frame: pd.DataFrame,
    mapped: dict[str, np.ndarray],
    config: dict[str, object],
) -> np.ndarray:
    """Estimate local heat exposure without treating it as ambient temperature.

    Solar radiation and positive wrist-temperature deviation are only used to
    reduce confidence in the gridded ambient sample. They never shift the
    absolute temperature value used by the capability model.
    """
    exposure_config = dict(config.get("local_exposure", {}))
    solar_reference = max(float(exposure_config.get("solar_reference_wm2", 800.0)), 1.0)
    wrist_reference = max(float(exposure_config.get("wrist_relative_reference_c", 6.0)), 0.1)
    solar_share = float(exposure_config.get("solar_share", 0.60))
    wrist_share = float(exposure_config.get("wrist_share", 0.40))
    share_total = max(solar_share + wrist_share, 1e-9)
    solar_score = np.clip(np.nan_to_num(mapped["shortwave_radiation"], nan=0.0) / solar_reference, 0.0, 1.0)
    wrist_relative = _numeric_column(frame, "device_temperature_relative").to_numpy(dtype=float)
    wrist_score = np.clip(np.nan_to_num(wrist_relative, nan=0.0) / wrist_reference, 0.0, 1.0)
    return np.clip((solar_share * solar_score + wrist_share * wrist_score) / share_total, 0.0, 1.0)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _approximate_distance(
    latitude: np.ndarray, longitude: np.ndarray, target_latitude: float, target_longitude: float
) -> np.ndarray:
    mean_latitude = np.radians((latitude + target_latitude) / 2.0)
    x = np.radians(longitude - target_longitude) * np.cos(mean_latitude)
    y = np.radians(latitude - target_latitude)
    distance = 6_371_000.0 * np.sqrt(x * x + y * y)
    return np.where(np.isfinite(distance), distance, np.inf)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = phi2 - phi1
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * 6_371_000.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
