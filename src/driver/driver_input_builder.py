"""Build driver-analysis input from raw VOC review data."""

from __future__ import annotations

import re
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from common.config_loader import get_output_table, get_source_filters, get_source_table


def clean_feature_name(value: Any) -> str:
    """Return a stable feature key for wide regression columns."""
    text = str(value or "").strip()
    text = re.sub(r"[./()]+", "", text)
    text = re.sub(r"[-\s]+", "_", text)
    text = re.sub(r"[^0-9A-Za-z가-힣_]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def _first_existing_column(df: DataFrame, candidates: list[str]) -> str:
    """Return the first configured column that exists in the source dataframe."""
    available = set(df.columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(
        "No entity id column found for driver analysis. "
        f"candidates={candidates}, available_columns={df.columns}"
    )


def _apply_source_filters(df: DataFrame, config: dict[str, Any], table_key: str) -> DataFrame:
    """Apply configured SQL filter snippets to a source dataframe."""
    for filter_sql in get_source_filters(config, table_key):
        df = df.where(F.expr(filter_sql))
    return df


def build_driver_input_df(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    source_table_key: str | None = None,
) -> DataFrame:
    """Create entity/category level sentiment scores for WLS driver analysis."""
    driver_cfg = config.get("driver_analysis", {}) or {}
    resolved_source_key = source_table_key or str(
        driver_cfg.get("source_table_key", "raw_review_table")
    )
    source_table = get_source_table(config, resolved_source_key)
    source_df = spark.table(source_table)
    if bool(driver_cfg.get("apply_source_filters", True)):
        source_df = _apply_source_filters(
            source_df,
            config,
            resolved_source_key,
        )

    entity_col = _first_existing_column(
        source_df,
        list(driver_cfg.get("entity_id_candidates", []) or ["model_id"]),
    )
    sentiment_col = str(driver_cfg.get("sentiment_col", "sc_measurement"))
    category_key_cols = list(
        driver_cfg.get("category_key_cols", []) or ["cate_1_depth", "cate_2_depth"]
    )
    category_label_col = str(driver_cfg.get("category_label_col", "cate_2_depth"))
    segment_col = str(driver_cfg.get("segment_col") or "").strip()
    group_dims = list(driver_cfg.get("group_dims", []) or [])
    min_entity_category_count = int(driver_cfg.get("min_entity_category_count", 1))

    required_cols = [entity_col, sentiment_col, category_label_col, *category_key_cols]
    if segment_col:
        required_cols.append(segment_col)
    missing_cols = [col for col in required_cols if col not in source_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required driver source columns: {missing_cols}")

    existing_group_dims = [col for col in group_dims if col != "all" and col in source_df.columns]
    sentiment_values = driver_cfg.get("sentiment_values")
    if sentiment_values:
        source_df = source_df.where(F.col(sentiment_col).isin([int(v) for v in sentiment_values]))
    segment_values = driver_cfg.get("segment_values")
    if segment_col and segment_values:
        source_df = source_df.where(
            F.col(segment_col).cast("string").isin([str(v) for v in segment_values])
        )

    category_expr = F.concat_ws(
        "||",
        *[F.coalesce(F.col(col).cast("string"), F.lit("")) for col in category_key_cols],
    )

    grouped_cols = [
        F.col(entity_col).cast("string").alias("entity_id"),
        *([F.col(segment_col).cast("string").alias(segment_col)] if segment_col else []),
        *[F.col(col).cast("string").alias(col) for col in category_key_cols],
        F.col(category_label_col).cast("string").alias("category_label"),
        *[F.col(col).cast("string").alias(col) for col in existing_group_dims],
    ]

    selected_df = (
        source_df.select(
            *grouped_cols,
            category_expr.alias("category_key"),
            F.col(sentiment_col).cast("double").alias("sentiment_score"),
        )
        .where(F.col("entity_id").isNotNull())
        .where(F.col("category_key") != "")
        .where(F.col("sentiment_score").isNotNull())
    )

    result_df = (
        selected_df.groupBy(
            "entity_id",
            *([segment_col] if segment_col else []),
            "category_key",
            "category_label",
            *category_key_cols,
            *existing_group_dims,
        )
        .agg(
            F.avg("sentiment_score").cast("double").alias("avg_sc"),
            F.count("*").cast("long").alias("total_count"),
        )
        .where(F.col("total_count") >= F.lit(min_entity_category_count))
        .withColumn("feature_name", F.udf(clean_feature_name, "string")("category_key"))
        .withColumn("source_table", F.lit(source_table))
        .withColumn("created_at", F.current_timestamp())
    )

    return result_df


def save_driver_input(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    mode: str = "overwrite",
    source_table_key: str | None = None,
) -> dict[str, Any]:
    """Build and save the driver input table."""
    table_name = get_output_table(config, "driver_input")
    driver_input_df = build_driver_input_df(
        spark,
        config,
        source_table_key=source_table_key,
    )
    row_count = driver_input_df.count()
    (
        driver_input_df.write.format("delta")
        .mode(mode)
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )
    return {
        "table_name": table_name,
        "row_count": row_count,
    }
