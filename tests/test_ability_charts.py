from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui.ability_charts import fatigue_figure, temperature_figure, terrain_ability_figure

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def test_terrain_ability_charts_render_continuous_uphill_and_downhill_curves() -> None:
    profile = {
        "uphill": {"curve": [{"grade": 5.0, "value": 600.0, "source": "personal"}, {"grade": 15.0, "value": 420.0, "source": "default"}]},
        "downhill": {"curve": [{"grade": -15.0, "speed_mps": 2.0, "source": "personal"}, {"grade": -5.0, "speed_mps": 3.0, "source": "default"}]},
    }
    uphill = terrain_ability_figure(profile, "uphill")
    downhill = terrain_ability_figure(profile, "downhill")

    assert len(uphill.axes) == 1
    assert uphill.axes[0].get_ylabel() == "VAM (m/h)"
    assert len(downhill.axes[0].lines) == 1
    assert downhill.axes[0].get_ylabel() == "下坡速度 (km/h)"
    plt.close(uphill)
    plt.close(downhill)


def test_fatigue_and_temperature_charts_render_sources_and_baselines() -> None:
    fatigue = {
        "flat": [{"hour": 0.0, "factor": 1.0, "source": "anchor"}, {"hour": 8.0, "factor": 0.8, "source": "personal"}],
        "uphill": [{"hour": 0.0, "factor": 1.0, "source": "anchor"}, {"hour": 8.0, "factor": 0.75, "source": "default"}],
        "downhill": [{"hour": 0.0, "factor": 1.0, "source": "anchor"}, {"hour": 20.0, "factor": 0.65, "source": "extrapolated"}],
    }
    temperature = {
        "curve": [
            {"temperature_c": 5.0, "time_factor": 1.02, "default_time_factor": 1.02, "source": "default"},
            {"temperature_c": 15.0, "time_factor": 1.0, "default_time_factor": 1.0, "source": "personal_blend"},
            {"temperature_c": 30.0, "time_factor": 1.06, "default_time_factor": 1.05, "source": "personal_blend"},
        ]
    }
    fatigue_chart = fatigue_figure(fatigue)
    temperature_chart = temperature_figure(temperature)

    assert len(fatigue_chart.axes[0].lines) == 6  # three curves plus 100/90/80% guides
    assert len(temperature_chart.axes[0].lines) == 2
    assert temperature_chart.axes[0].get_xlabel() == "环境温度 (℃)"
    plt.close(fatigue_chart)
    plt.close(temperature_chart)
