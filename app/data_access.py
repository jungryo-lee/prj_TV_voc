"""Parquet IO helpers for the Databricks App layer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from app.config import (
    MANUAL_REVIEW_INPUT,
    OTHERS_REVIEW_EXPORT,
    SUMMARY_EXPORT,
    TOPIC_POOL_EXPORT,
)


def _read_parquet_dir(path: Path) -> pd.DataFrame:
    """Read a parquet file or directory if it exists, otherwise return empty."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        parquet_files = sorted(path.glob("*.parquet")) if path.is_dir() else []
        if not parquet_files:
            return pd.DataFrame()
        return pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)


def load_topic_pool() -> pd.DataFrame:
    """Load topic-pool snapshot for dropdown options."""
    return _read_parquet_dir(TOPIC_POOL_EXPORT)


def load_classification_summary() -> pd.DataFrame:
    """Load classification summary snapshot for dashboard charts."""
    return _read_parquet_dir(SUMMARY_EXPORT)


def load_others_review_candidates() -> pd.DataFrame:
    """Load others candidates for human review."""
    return _read_parquet_dir(OTHERS_REVIEW_EXPORT)


def save_manual_review_decisions(review_df: pd.DataFrame) -> str:
    """Save manually edited review decisions as a timestamped parquet file."""
    MANUAL_REVIEW_INPUT.mkdir(parents=True, exist_ok=True)
    saved_at = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_path = MANUAL_REVIEW_INPUT / f"manual_review_decisions_{saved_at}.parquet"
    write_df = review_df.copy()
    write_df["saved_at"] = saved_at
    write_df.to_parquet(output_path, index=False)
    return str(output_path)
