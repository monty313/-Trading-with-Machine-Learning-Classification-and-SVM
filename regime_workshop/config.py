"""Configuration and safe teaching-condition validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
TIMEFRAME_SETS = {
    "Set 1": ("1m", "15m", "30m"),
    "Set 2": ("5m", "30m", "1h"),
    "Set 3": ("15m", "1h", "4h"),
    "Set 4": ("30m", "4h", "1d"),
}
CONDITION_OPERATORS = {">", ">=", "<", "<=", "==", "!="}


def new_config() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "observation_profile": {"id": "default", "features": []},
        "regimes": [],
    }


def validate_condition(condition: dict[str, Any]) -> None:
    if not isinstance(condition, dict):
        raise ValueError("Each condition must be an object.")
    if not isinstance(condition.get("feature"), str) or not condition["feature"].strip():
        raise ValueError("A condition needs a feature name.")
    if condition.get("operator") not in CONDITION_OPERATORS:
        raise ValueError("Use one of: >, >=, <, <=, ==, !=.")
    value = condition.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Condition values must be finite numbers.")


def validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Configuration schema_version must be {SCHEMA_VERSION}.")
    regimes = config.get("regimes")
    if not isinstance(regimes, list):
        raise ValueError("Regimes must be a list.")
    regime_ids: set[str] = set()
    strategy_ids: set[str] = set()
    for regime in regimes:
        if not isinstance(regime, dict):
            raise ValueError("Each regime must be an object.")
        for field in ("id", "name"):
            if not isinstance(regime.get(field), str) or not regime[field].strip():
                raise ValueError(f"Each regime needs a {field}.")
        if regime["id"] in regime_ids:
            raise ValueError(f"Duplicate regime ID: {regime['id']}.")
        regime_ids.add(regime["id"])
        if not isinstance(regime.get("strategies"), list):
            raise ValueError(f"Regime {regime['id']} needs a strategies list.")
        for strategy in regime["strategies"]:
            if not isinstance(strategy, dict):
                raise ValueError("Each teaching strategy must be an object.")
            if not isinstance(strategy.get("id"), str) or not strategy["id"].strip():
                raise ValueError("Each teaching strategy needs an ID.")
            if strategy["id"] in strategy_ids:
                raise ValueError(f"Duplicate strategy ID: {strategy['id']}.")
            strategy_ids.add(strategy["id"])
            if not isinstance(strategy.get("version"), str) or not strategy["version"].strip():
                raise ValueError(f"Strategy {strategy['id']} needs a version.")
            if strategy.get("timeframe_set") not in TIMEFRAME_SETS:
                raise ValueError(f"Strategy {strategy['id']} must use an official timeframe set.")
            groups = strategy.get("groups")
            if not isinstance(groups, list) or not groups:
                raise ValueError(f"Strategy {strategy['id']} needs at least one condition group.")
            for group in groups:
                if not isinstance(group, dict) or group.get("logic") not in {"all", "any"}:
                    raise ValueError("Condition groups must use 'all' or 'any'.")
                conditions = group.get("conditions")
                if not isinstance(conditions, list) or not conditions:
                    raise ValueError("Each condition group must contain a condition.")
                for condition in conditions:
                    validate_condition(condition)
    profile = config.get("observation_profile")
    if not isinstance(profile, dict) or not isinstance(profile.get("features"), list):
        raise ValueError("Add a separate observation profile with a feature list.")
    if any(not isinstance(name, str) or not name.strip() for name in profile["features"]):
        raise ValueError("Observation feature names must be non-empty text.")
    if len(set(profile["features"])) != len(profile["features"]):
        raise ValueError("Observation feature names must be unique.")


def save_config(config: dict[str, Any], path: str | Path) -> None:
    validate_config(config)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def load_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return new_config()
    config = json.loads(target.read_text(encoding="utf-8"))
    validate_config(config)
    return config
