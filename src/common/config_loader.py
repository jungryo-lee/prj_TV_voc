"""settings.yaml loader for prj_TV_voc.

This module keeps configuration loading separate from pipeline logic.
It reads the YAML config, resolves a few commonly used paths, and exposes
small helper functions for downstream modules.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml


def _default_settings_path() -> Path:
    """Return the default settings.yaml path relative to this module."""
    return Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def _coerce_path(value: str | None, project_root: Path) -> str | None:
    """Resolve a config path to an absolute string when possible."""
    if not value:
        return value

    raw = Path(value)
    if raw.is_absolute():
        return str(raw)

    return str((project_root / raw).resolve())


def _resolve_paths(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Resolve commonly used path fields."""
    config = deepcopy(config)
    path_cfg = config.setdefault("path", {})

    project_root_value = path_cfg.get("project_root")
    if project_root_value:
        project_root = Path(project_root_value)
    else:
        project_root = config_path.parents[1]
        path_cfg["project_root"] = str(project_root)

    path_cfg["settings"] = str(config_path.resolve())
    path_cfg["project_root"] = str(project_root.resolve())
    path_cfg["env"] = _coerce_path(path_cfg.get("env"), project_root)
    path_cfg["prompts"] = _coerce_path(path_cfg.get("prompts"), project_root)

    return config


def _inject_runtime_metadata(
    config: dict[str, Any],
    run_date: str | date | datetime | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Inject normalized runtime metadata for downstream code."""
    config = deepcopy(config)
    runtime_cfg = config.setdefault("runtime", {})

    if run_date is None:
        normalized_run_date = date.today().isoformat()
    elif isinstance(run_date, datetime):
        normalized_run_date = run_date.date().isoformat()
    elif isinstance(run_date, date):
        normalized_run_date = run_date.isoformat()
    else:
        normalized_run_date = str(run_date)

    runtime_cfg["resolved_run_date"] = normalized_run_date

    if run_id is None:
        prefix = runtime_cfg.get("run_id_prefix", "voc")
        run_id = f"{prefix}_{normalized_run_date.replace('-', '')}"

    runtime_cfg["resolved_run_id"] = run_id
    return config


def load_config(
    config_path: str | Path | None = None,
    run_date: str | date | datetime | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Load settings.yaml and return a resolved config dict."""
    resolved_config_path = Path(config_path) if config_path else _default_settings_path()

    with open(resolved_config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config = _resolve_paths(config, resolved_config_path)
    config = _inject_runtime_metadata(config, run_date=run_date, run_id=run_id)
    return config


def get_table_prefix(config: dict[str, Any]) -> str:
    """Return catalog.schema prefix from config."""
    uc_cfg = config.get("unity_catalog", {})
    catalog = uc_cfg.get("catalog", "")
    schema = uc_cfg.get("schema", "")

    if not catalog or not schema:
        raise ValueError("unity_catalog.catalog and unity_catalog.schema are required.")

    return f"{catalog}.{schema}"


def get_output_table(config: dict[str, Any], table_key: str) -> str:
    """Return a managed output table name by key."""
    outputs = config.get("tables", {}).get("outputs", {})
    table_name = outputs.get(table_key)
    if not table_name:
        raise KeyError(f"Unknown output table key: {table_key}")
    return table_name


def get_log_table(config: dict[str, Any], table_key: str) -> str:
    """Return a managed log table name by key."""
    logs = config.get("tables", {}).get("logs", {})
    table_name = logs.get(table_key)
    if not table_name:
        raise KeyError(f"Unknown log table key: {table_key}")
    return table_name


def get_reference_table(config: dict[str, Any], table_key: str) -> str:
    """Return a reference table name by key."""
    refs = config.get("reference", {})
    table_name = refs.get(table_key)
    if not table_name:
        raise KeyError(f"Unknown reference table key: {table_key}")
    return table_name


def get_source_table(config: dict[str, Any], table_key: str) -> str:
    """Return a source table name by key."""
    sources = config.get("source", {})
    table_name = sources.get(table_key)
    if not table_name:
        raise KeyError(f"Unknown source table key: {table_key}")
    return table_name
