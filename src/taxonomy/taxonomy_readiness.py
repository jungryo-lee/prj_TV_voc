"""Validate source-category changes before running incremental topic labeling.

The pipeline deliberately keeps ``memo_id`` category-aware. This module adds a
separate source identity for impact analysis so a corrected category can be
detected without changing the memo_id contract used by final labels.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from common.category_alias import apply_category_aliases
from common.config_loader import get_output_table, get_source_filters, get_source_table
from common.memo_id import normalize_memo_expr, with_memo_id


GROUP_COLS = ["cate_1_depth", "cate_2_depth", "sc_measurement"]


def _value(config: dict[str, Any], section: str, key: str, default: str = "") -> str:
    return str((config.get(section, {}) or {}).get(key, default) or default)


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("taxonomy_readiness", {}) or {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "source_snapshot_table_key": cfg.get("source_snapshot_table_key", "source_taxonomy_snapshot"),
        "output_table_key": cfg.get("output_table_key", "taxonomy_readiness"),
        "category_move_ratio_threshold": float(cfg.get("category_move_ratio_threshold", 0.05)),
        "sample_others_ratio_threshold": float(cfg.get("sample_others_ratio_threshold", 0.15)),
        "min_topic_pool_count": int(cfg.get("min_topic_pool_count", 3)),
        "allowed_incremental_statuses": list(
            cfg.get("allowed_incremental_statuses", ["reuse", "refresh_prototype"])
        ),
    }


def _run_date(config: dict[str, Any]) -> str:
    return _value(config, "runtime", "resolved_run_date") or datetime.utcnow().date().isoformat()


def _source_identity_expr(columns: list[str]) -> F.Column:
    """Build a category-independent source identity with the best available key."""
    non_blank = lambda column: F.when(F.length(F.trim(F.col(column).cast("string"))) > 0, F.col(column).cast("string"))
    candidates: list[F.Column] = []
    if "review_id" in columns:
        candidates.append(F.concat(F.lit("review_id|"), non_blank("review_id")))
    if "source_id" in columns:
        candidates.append(F.concat(F.lit("source_id|"), non_blank("source_id")))
    if "post_no" in columns:
        candidates.append(
            F.concat(
                F.lit("post_memo|"),
                non_blank("post_no"),
                F.lit("|"),
                normalize_memo_expr("memo"),
            )
        )
    candidates.append(F.concat(F.lit("memo|"), normalize_memo_expr("memo")))
    return F.coalesce(*candidates)


def build_source_taxonomy_snapshot_df(spark: SparkSession, config: dict[str, Any]) -> DataFrame:
    """Create the current category-aware snapshot without changing source rows."""
    source_key = (config.get("ml_classification", {}) or {}).get("source_table_key", "raw_review_table")
    source_df = spark.table(get_source_table(config, source_key))
    for filter_sql in get_source_filters(config, source_key):
        source_df = source_df.where(F.expr(filter_sql))

    source_df = (
        apply_category_aliases(source_df, config)
        .where(F.col("memo").isNotNull())
        .where(F.length(F.trim(F.col("memo").cast("string"))) > 0)
        .where(F.col("sc_measurement").cast("int").isin(1, -1))
        .transform(with_memo_id)
    )
    run_date = _run_date(config)
    return (
        source_df.select(
            _source_identity_expr(source_df.columns).alias("source_memo_key"),
            F.col("cate_1_depth").cast("string"),
            F.col("cate_2_depth").cast("string"),
            F.col("sc_measurement").cast("int"),
            F.col("memo_id").cast("string"),
            F.col("memo_norm").cast("string"),
            F.lit(run_date).alias("snapshot_run_date"),
            F.current_timestamp().alias("created_at"),
        )
        .dropDuplicates(["source_memo_key", "cate_1_depth", "cate_2_depth", "sc_measurement", "memo_id"])
    )


def _save_snapshot(spark: SparkSession, config: dict[str, Any], snapshot_df: DataFrame) -> str:
    table_name = get_output_table(config, _cfg(config)["source_snapshot_table_key"])
    run_date = _run_date(config).replace("'", "''")
    if spark.catalog.tableExists(table_name):
        spark.sql(f"DELETE FROM {table_name} WHERE snapshot_run_date = '{run_date}'")
        snapshot_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table_name)
    else:
        snapshot_df.write.format("delta").mode("overwrite").saveAsTable(table_name)
    return table_name


def _previous_snapshot_df(spark: SparkSession, table_name: str, run_date: str) -> DataFrame | None:
    if not spark.catalog.tableExists(table_name):
        return None
    previous_dates = (
        spark.table(table_name)
        .where(F.col("snapshot_run_date") < run_date)
        .select("snapshot_run_date")
        .dropDuplicates()
        .orderBy(F.col("snapshot_run_date").desc())
        .limit(1)
        .collect()
    )
    if not previous_dates:
        return None
    return spark.table(table_name).where(F.col("snapshot_run_date") == previous_dates[0]["snapshot_run_date"])


def _asset_group_counts(spark: SparkSession, config: dict[str, Any], table_key: str, count_name: str) -> DataFrame:
    table_name = get_output_table(config, table_key)
    if not spark.catalog.tableExists(table_name):
        return spark.createDataFrame([], "cate_1_depth string, cate_2_depth string, sc_measurement int, " + count_name + " long")
    df = spark.table(table_name)
    prompt_version = _value(config, "version", "prompt_version")
    taxonomy_version = _value(config, "version", "taxonomy_version")
    if "prompt_version" in df.columns:
        df = df.where(F.col("prompt_version") == prompt_version)
    if "taxonomy_version" in df.columns:
        df = df.where(F.col("taxonomy_version") == taxonomy_version)
    if "topic" in df.columns:
        count_expression = F.countDistinct("topic")
    elif "pred_topic" in df.columns:
        count_expression = F.countDistinct("pred_topic")
    else:
        # Rule profiles are one record per group and have no topic column.
        count_expression = F.countDistinct(F.lit(1))
    return df.groupBy(*GROUP_COLS).agg(count_expression.alias(count_name))


def _sample_quality_df(spark: SparkSession, config: dict[str, Any]) -> DataFrame:
    table_name = get_output_table(config, "classification_detail")
    if not spark.catalog.tableExists(table_name):
        return spark.createDataFrame([], "cate_1_depth string, cate_2_depth string, sc_measurement int, sample_memo_cnt long, sample_others_ratio double")
    df = spark.table(table_name)
    prompt_version = _value(config, "version", "prompt_version")
    taxonomy_version = _value(config, "version", "taxonomy_version")
    return (
        df.where(F.col("prompt_version") == prompt_version)
        .where(F.col("taxonomy_version") == taxonomy_version)
        .groupBy(*GROUP_COLS)
        .agg(
            F.countDistinct("memo_id").alias("sample_memo_cnt"),
            (
                F.countDistinct(F.when(F.col("pred_topic_type") == "others", F.col("memo_id")))
                / F.countDistinct("memo_id")
            ).alias("sample_others_ratio"),
        )
    )


def build_taxonomy_readiness_df(
    spark: SparkSession,
    config: dict[str, Any],
    snapshot_df: DataFrame,
    previous_snapshot_df: DataFrame | None,
) -> DataFrame:
    """Return an operational decision for each current category/sentiment group."""
    cfg = _cfg(config)
    source_group_df = snapshot_df.groupBy(*GROUP_COLS).agg(
        F.countDistinct("source_memo_key").alias("source_memo_cnt")
    )
    if previous_snapshot_df is None:
        impact_group_df = source_group_df.select(
            *GROUP_COLS,
            F.lit(0).cast("long").alias("category_moved_memo_cnt"),
            F.lit(0.0).alias("category_move_ratio"),
            F.lit("baseline_snapshot").alias("source_change_status"),
        )
    else:
        previous_df = previous_snapshot_df.select(
            F.col("source_memo_key"),
            F.col("cate_1_depth").alias("previous_cate_1_depth"),
            F.col("cate_2_depth").alias("previous_cate_2_depth"),
            F.col("sc_measurement").alias("previous_sc_measurement"),
        )
        changed_df = snapshot_df.join(previous_df, on="source_memo_key", how="left")
        impact_group_df = (
            changed_df.groupBy(*GROUP_COLS)
            .agg(
                F.countDistinct(
                    F.when(
                        F.col("previous_cate_1_depth").isNotNull()
                        & (
                            (F.col("cate_1_depth") != F.col("previous_cate_1_depth"))
                            | (F.col("cate_2_depth") != F.col("previous_cate_2_depth"))
                            | (F.col("sc_measurement") != F.col("previous_sc_measurement"))
                        ),
                        F.col("source_memo_key"),
                    )
                ).alias("category_moved_memo_cnt"),
                F.countDistinct("source_memo_key").alias("source_memo_cnt"),
            )
            .withColumn("category_move_ratio", F.col("category_moved_memo_cnt") / F.col("source_memo_cnt"))
            .withColumn("source_change_status", F.lit("compared_to_previous_snapshot"))
        )

    readiness_df = (
        source_group_df.join(impact_group_df, on=GROUP_COLS, how="left")
        .join(_asset_group_counts(spark, config, "topic_pool", "topic_pool_cnt"), on=GROUP_COLS, how="left")
        .join(_asset_group_counts(spark, config, "rule_profile", "rule_profile_cnt"), on=GROUP_COLS, how="left")
        .join(_asset_group_counts(spark, config, "topic_prototype", "prototype_cnt"), on=GROUP_COLS, how="left")
        .join(_sample_quality_df(spark, config), on=GROUP_COLS, how="left")
        .fillna({
            "category_moved_memo_cnt": 0,
            "category_move_ratio": 0.0,
            "topic_pool_cnt": 0,
            "rule_profile_cnt": 0,
            "prototype_cnt": 0,
            "sample_memo_cnt": 0,
            "sample_others_ratio": 0.0,
        })
        .withColumn(
            "readiness_status",
            F.when(
                (F.col("topic_pool_cnt") < cfg["min_topic_pool_count"]) | (F.col("rule_profile_cnt") == 0),
                F.lit("new_taxonomy"),
            )
            .when(
                (F.col("category_move_ratio") >= cfg["category_move_ratio_threshold"])
                | (F.col("sample_others_ratio") >= cfg["sample_others_ratio_threshold"]),
                F.lit("review_taxonomy"),
            )
            .when((F.col("sample_memo_cnt") == 0) | (F.col("prototype_cnt") == 0), F.lit("refresh_prototype"))
            .otherwise(F.lit("reuse")),
        )
        .withColumn(
            "readiness_reason",
            F.when(F.col("readiness_status") == "new_taxonomy", F.lit("missing_rule_profile_or_topic_pool"))
            .when(F.col("readiness_status") == "review_taxonomy", F.lit("category_move_or_high_sample_others"))
            .when(F.col("readiness_status") == "refresh_prototype", F.lit("missing_sample_or_prototype"))
            .otherwise(F.lit("existing_taxonomy_reusable")),
        )
        .withColumn("incremental_processing_allowed", F.col("readiness_status").isin(cfg["allowed_incremental_statuses"]))
        .select(
            F.lit(_value(config, "version", "prompt_version")).alias("prompt_version"),
            F.lit(_value(config, "version", "taxonomy_version")).alias("taxonomy_version"),
            F.lit(_run_date(config)).alias("run_date"),
            *[F.col(column) for column in GROUP_COLS],
            "source_memo_cnt",
            "category_moved_memo_cnt",
            "category_move_ratio",
            "source_change_status",
            "rule_profile_cnt",
            "topic_pool_cnt",
            "sample_memo_cnt",
            "sample_others_ratio",
            "prototype_cnt",
            "readiness_status",
            "readiness_reason",
            "incremental_processing_allowed",
            F.current_timestamp().alias("created_at"),
        )
    )
    return readiness_df


def _save_readiness(spark: SparkSession, config: dict[str, Any], readiness_df: DataFrame) -> str:
    table_name = get_output_table(config, _cfg(config)["output_table_key"])
    run_date = _run_date(config).replace("'", "''")
    prompt_version = _value(config, "version", "prompt_version").replace("'", "''")
    taxonomy_version = _value(config, "version", "taxonomy_version").replace("'", "''")
    if spark.catalog.tableExists(table_name):
        spark.sql(
            f"""DELETE FROM {table_name}
            WHERE prompt_version = '{prompt_version}'
              AND taxonomy_version = '{taxonomy_version}'
              AND run_date = '{run_date}'"""
        )
        readiness_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table_name)
    else:
        readiness_df.write.format("delta").mode("overwrite").saveAsTable(table_name)
    return table_name


def run_source_taxonomy_readiness_check(spark: SparkSession, config: dict[str, Any]) -> dict[str, Any]:
    """Persist a source snapshot and taxonomy readiness decisions."""
    if not _cfg(config)["enabled"]:
        return {"skipped": True, "reason": "disabled"}
    snapshot_df = build_source_taxonomy_snapshot_df(spark, config)
    snapshot_table = get_output_table(config, _cfg(config)["source_snapshot_table_key"])
    previous_snapshot_df = _previous_snapshot_df(spark, snapshot_table, _run_date(config))
    snapshot_table = _save_snapshot(spark, config, snapshot_df)
    readiness_df = build_taxonomy_readiness_df(spark, config, snapshot_df, previous_snapshot_df)
    readiness_table = _save_readiness(spark, config, readiness_df)
    return {
        "source_snapshot_table": snapshot_table,
        "taxonomy_readiness_table": readiness_table,
        "snapshot_distinct_source_memo_cnt": snapshot_df.select("source_memo_key").distinct().count(),
        "readiness_summary_df": readiness_df.groupBy("readiness_status", "incremental_processing_allowed").count(),
        "readiness_df": readiness_df,
    }
