from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load validated model parameters from YAML."""
    target = Path(path) if path else Path(__file__).with_name("defaults.yaml")
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"无法读取配置文件 {target}: {exc}") from exc
    if not isinstance(data, dict) or "default_profile" not in data or "confidence" not in data:
        raise ValueError(f"配置文件结构无效: {target}")
    return data
