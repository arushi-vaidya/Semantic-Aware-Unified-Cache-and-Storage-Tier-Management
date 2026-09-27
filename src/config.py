from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = {"cache", "tiers", "metadata", "policy", "workload"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Configuration is missing sections: {', '.join(sorted(missing))}")
    base = config_path.parent.parent
    config["tiers"] = {name: str((base / value).resolve()) if not Path(value).is_absolute() else value for name, value in config["tiers"].items()}
    database = Path(config["metadata"]["database"])
    config["metadata"]["database"] = str((base / database).resolve()) if not database.is_absolute() else str(database)
    return config