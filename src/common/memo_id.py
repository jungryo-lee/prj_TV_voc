"""Memo normalization and stable memo_id helpers."""

from __future__ import annotations

import hashlib
from typing import Any

from pyspark.sql import DataFrame, functions as F


def normalize_memo_text_for_id(text: Any) -> str:
    """Normalize memo text for stable ID generation.

    Rules:
    - lowercase for case-insensitive matching
    - convert full-width spaces to normal spaces
    - remove most special characters by replacing them with spaces
    - collapse repeated whitespace, tabs, and line breaks
    - preserve token boundaries to avoid over-merging different meanings
    """
    if text is None:
        return ""

    normalized = str(text).replace("　", " ").lower()
    normalized = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in normalized)
    normalized = " ".join(normalized.split())
    return normalized.strip()


def build_memo_id_value(
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    memo: Any,
) -> str:
    """Build a stable Python-side memo_id value."""
    raw = "||".join(
        [
            str(cate_1_depth or "").strip(),
            str(cate_2_depth or "").strip(),
            str(int(sc_measurement)),
            normalize_memo_text_for_id(memo),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_memo_expr(col_name: str) -> F.Column:
    """Spark expression version of memo normalization."""
    return F.trim(
        F.regexp_replace(
            F.regexp_replace(
                F.lower(
                    F.translate(
                        F.coalesce(F.col(col_name).cast("string"), F.lit("")),
                        "　",
                        " ",
                    )
                ),
                r"[^0-9a-zA-Z가-힣\s]",
                " ",
            ),
            r"\s+",
            " ",
        )
    )


def memo_id_expr(
    cate_1_col: str = "cate_1_depth",
    cate_2_col: str = "cate_2_depth",
    sc_col: str = "sc_measurement",
    memo_col: str = "memo",
) -> F.Column:
    """Spark expression for stable memo_id generation."""
    return F.sha2(
        F.concat_ws(
            "||",
            F.coalesce(F.col(cate_1_col).cast("string"), F.lit("")),
            F.coalesce(F.col(cate_2_col).cast("string"), F.lit("")),
            F.coalesce(F.col(sc_col).cast("string"), F.lit("")),
            normalize_memo_expr(memo_col),
        ),
        256,
    )


def with_memo_id(
    df: DataFrame,
    cate_1_col: str = "cate_1_depth",
    cate_2_col: str = "cate_2_depth",
    sc_col: str = "sc_measurement",
    memo_col: str = "memo",
) -> DataFrame:
    """Attach memo_norm and memo_id columns to a Spark DataFrame."""
    out = df
    if "memo_norm" not in out.columns:
        out = out.withColumn("memo_norm", normalize_memo_expr(memo_col))
    if "memo_id" not in out.columns:
        out = out.withColumn(
            "memo_id",
            memo_id_expr(
                cate_1_col=cate_1_col,
                cate_2_col=cate_2_col,
                sc_col=sc_col,
                memo_col=memo_col,
            ),
        )
    return out
