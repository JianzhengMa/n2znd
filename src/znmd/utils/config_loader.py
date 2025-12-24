"""
YAML configuration utilities for Zhu–Nakamura simulations.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Optional, Union

__all__ = ["load_config", "create_simulator_from_yaml", "normalize_config"]


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("PyYAML is required to load YAML configuration files.") from exc

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {path} must define a mapping at the root level.")
    return data


@lru_cache(maxsize=1)
def _simulator_schema() -> Dict[str, Dict[str, Any]]:
    from ..dynamics.zhu_nakamura import Zhu_Nakamura

    signature = inspect.signature(Zhu_Nakamura.__init__)
    schema: Dict[str, Dict[str, Any]] = {}
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        default = param.default
        schema[name] = {
            "has_default": default is not inspect._empty,
            "type": None if default is inspect._empty else type(default),
        }
    return schema


def _filter_simulator_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    schema = _simulator_schema()
    return {key: value for key, value in config.items() if key in schema}


def _coerce_value(name: str, value: Any) -> Any:
    schema = _simulator_schema()[name]
    expected_type = schema["type"]
    if expected_type is None or isinstance(value, expected_type):
        return value

    def _raise():
        raise ValueError(f"Configuration field '{name}' expects {expected_type.__name__}, got {value!r}")

    if expected_type is bool:
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        _raise()
    elif expected_type is int:
        try:
            return int(value)
        except (TypeError, ValueError):
            _raise()
    elif expected_type is float:
        try:
            return float(value)
        except (TypeError, ValueError):
            _raise()
    elif expected_type is str:
        return str(value)
    return value


def _apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    if not overrides:
        return config
    schema = _simulator_schema()
    for key, value in overrides.items():
        if key not in schema:
            raise KeyError(f"Unknown configuration key '{key}'")
        config[key] = _coerce_value(key, value)
    return config


def _validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    schema = _simulator_schema()
    missing = [
        key for key, meta in schema.items() if not meta["has_default"] and key not in config
    ]
    if missing:
        raise ValueError(f"Missing required configuration fields: {', '.join(missing)}")
    for key in list(config.keys()):
        config[key] = _coerce_value(key, config[key])
    return config


def _prepare_config(raw_config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    filtered = _filter_simulator_kwargs(raw_config)
    if overrides:
        filtered = _apply_overrides(filtered, overrides)
    return _validate_config(filtered)


def load_config(path: Union[str, Path], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load a YAML document and keep only arguments understood by ``Zhu_Nakamura``.

    Parameters
    ----------
    path:
        Path to the YAML file.
    overrides:
        Optional mapping that takes priority over the YAML content.
    """

    config_path = Path(path)
    config = _read_yaml(config_path)
    return _prepare_config(config, overrides)


def create_simulator_from_yaml(
    path: Union[str, Path], overrides: Optional[Dict[str, Any]] = None
):
    """
    Convenience helper that instantiates ``Zhu_Nakamura`` using a YAML file.
    """

    from ..dynamics.zhu_nakamura import Zhu_Nakamura

    config = load_config(path, overrides=overrides)
    return Zhu_Nakamura(**config)


def normalize_config(config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Validate an in-memory configuration mapping (e.g. CLI defaults).
    """

    raw = dict(config)
    return _prepare_config(raw, overrides)
