"""Configuration helpers for the Databricks App layer."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DATA_ROOT = Path(os.environ.get("VOC_APP_DATA_ROOT", PROJECT_ROOT / "app_data"))
EXPORT_ROOT = Path(os.environ.get("VOC_APP_EXPORT_ROOT", APP_DATA_ROOT / "exports"))
INPUT_ROOT = Path(os.environ.get("VOC_APP_INPUT_ROOT", APP_DATA_ROOT / "inputs"))
OUTPUT_ROOT = Path(os.environ.get("VOC_APP_OUTPUT_ROOT", APP_DATA_ROOT / "outputs"))

TOPIC_POOL_EXPORT = EXPORT_ROOT / "topic_pool_current"
SUMMARY_EXPORT = EXPORT_ROOT / "classification_summary"
OTHERS_REVIEW_EXPORT = EXPORT_ROOT / "others_review_candidates"
MANUAL_REVIEW_INPUT = INPUT_ROOT / "manual_review_decisions"
