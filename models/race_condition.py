from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime


@dataclass(frozen=True)
class RaceCondition:
    current_form: str = "normal"
    pacing_strategy: str = "standard"
    temperature_c: float | None = None
    temperature_peak_c: float | None = None
    temperature_peak_hour: float | None = None
    temperature_finish_c: float | None = None
    humidity_percent: float | None = None
    altitude_factor: float = 1.0
    terrain_technical_level: int = 0
    mud_level: int = 0
    night_running_ratio: float = 0.0
    carried_weight_kg: float = 0.0
    aid_station_minutes: float = 0.0
    race_start_time_utc: datetime | None = None

    def normalized(self) -> "RaceCondition":
        """Clamp user-controlled fields to safe model ranges."""
        return RaceCondition(
            current_form=self.current_form,
            pacing_strategy=(
                self.pacing_strategy
                if self.pacing_strategy in {"conservative", "standard", "aggressive"}
                else "standard"
            ),
            temperature_c=self.temperature_c,
            temperature_peak_c=self.temperature_peak_c,
            temperature_peak_hour=None if self.temperature_peak_hour is None else max(0.0, self.temperature_peak_hour),
            temperature_finish_c=self.temperature_finish_c,
            humidity_percent=None if self.humidity_percent is None else min(100.0, max(0.0, self.humidity_percent)),
            altitude_factor=max(0.8, min(1.5, self.altitude_factor)),
            terrain_technical_level=max(-4, min(4, int(self.terrain_technical_level))),
            mud_level=max(-4, min(4, int(self.mud_level))),
            night_running_ratio=max(0.0, min(1.0, self.night_running_ratio)),
            carried_weight_kg=max(0.0, self.carried_weight_kg),
            aid_station_minutes=max(0.0, self.aid_station_minutes),
            race_start_time_utc=self.race_start_time_utc,
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["race_start_time_utc"] = self.race_start_time_utc.isoformat() if self.race_start_time_utc else None
        return data
