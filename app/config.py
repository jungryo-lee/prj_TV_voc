"""Configuration helpers for the Databricks App layer."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = Path(os.environ.get("VOC_SETTINGS_PATH", PROJECT_ROOT / "config" / "settings.yaml"))
DATA_MODE = os.environ.get("VOC_APP_DATA_MODE", "table")
