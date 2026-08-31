"""Operational lifecycle for immutable VOC labels and active topic mappings.

Final memo labels are never rewritten. This module records monthly topic health
and, when a low-share topic persists for consecutive monthly batches, changes
only the active taxonomy mapping used by downstream views.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from common.config_loader import get_output_table


GROUP_COLS = ["cate_1_depth", "cate_2_depth", "sc_measurement"]
MEMO_COLS = GROUP_COLS + ["memo_id"]


def _value(config: dict[str, Any], section: str, key: str, default: str = "") -> str:
    return str((config.get(section, {}) or {}).get(key, default) or default)


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("taxonomy_lifecycle", {}) or {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "health_table_key": cfg.get("health_table_key", "topic_monthly_health"),
        "active_mapping_table_key": cfg.get("active_mapping_table_key", "topic_active_mapping"),
        "new_candidate_table_key": cfg.get("new_candidate_table_key", "topic_new_candidate"),
        "min_group_labeled_memo_count": int(cfg.get("min_group_labeled_memo_count", 2000)),
        "others_ratio_threshold": float(cfg.get("others_ratio_threshold", 0.15)),
        "low_topic_share_threshold": float(cfg.get("low_topic_share_threshold", 0.01)),
        "low_topic_consecutive_months": int(cfg.get("low_topic_consecutive_months", 2)),
        "protected_topics": set(cfg.get("protected_topics", []) or []),
    }


def _run_month(config: dict[str, Any]) -> str:
    raw_value = _value(config, "runtime", "resolved_run_date")
    for parser in (datetime.fromisoformat, lambda value: datetime.strptime(value, "%Y-%m-%d")):
        try:
            parsed = parser(raw_value.replace("Z", "+00:00"))
            return parsed.strftime("%Y-%m-01")
        except (TypeError, ValueError):
            pass
    return date.today().strftime("%Y-%m-01")


def _group_condition_sql(target_groups: list[tuple[str, str, int]]) -> str:
    conditions = []
    for cate_1_depth, cate_2_depth, sc_measurement in target_groups:
        c1 = str(cate_1_depth).replace("'", "''")
        c2 = str(cate_2_depth).replace("'", "''")
        conditions.append(
            f"(cate_1_depth = '{c1}' AND cate_2_depth = '{c2}' "
            f"AND sc_measurement = {int(sc_measurement)})"
        )
    return " OR ".join(conditions) if conditions else "FALSE"


def _latest_final_df(spark: SparkSession, config: dict[str, Any], target_groups: list[tuple[str, str, int]]) -> DataFrame:
    table_name = get_output_table(config, "classification_detail_final")
    prompt_version = _value(config, "version", "prompt_version")
    taxonomy_version = _value(config, "version", "taxonomy_version")
    group_condition = _group_condition_sql(target_groups)
    window = Window.partitionBy(*MEMO_COLS).orderBy(
        F.col("created_at").desc_nulls_last(), F.col("run_id").desc_nulls_last()
    )
    return (
        spark.table(table_name)
        .where(F.col("prompt_version") == prompt_version)
        .where(F.col("taxonomy_version") == taxonomy_version)
        .where(F.expr(group_condition))
        .where(F.col("memo_id").isNotNull())
        .withColumn("_rn", F.row_number().over(window))
        .where(F.col("_rn") == 1)
        .drop("_rn")
    )


def _save_monthly_health(
    spark: SparkSession,
    config: dict[str, Any],
    latest_final_df: DataFrame,
    target_groups: list[tuple[str, str, int]],
) -> tuple[str, DataFrame]:
    cfg = _cfg(config)
    table_name = get_output_table(config, cfg["health_table_key"])
    run_month = _run_month(config)
    prompt_version = _value(config, "version", "prompt_version")
    taxonomy_version = _value(config, "version", "taxonomy_version")

    labeled_df = latest_final_df.where(F.col("pred_topic").isNotNull())
    group_stats_df = labeled_df.groupBy(*GROUP_COLS).agg(
        F.countDistinct("memo_id").alias("final_labeled_memo_cnt"),
        F.countDistinct(
            F.when(
                (F.col("pred_topic_type") == "others") | (F.col("pred_topic") == "기타"),
                F.col("memo_id"),
            )
        ).alias("others_memo_cnt"),
    )
    topic_stats_df = (
        labeled_df.where(F.col("pred_topic_type") == "topic")
        .where(F.col("pred_topic") != "기타")
        .groupBy(*GROUP_COLS, "pred_topic")
        .agg(F.countDistinct("memo_id").alias("topic_memo_cnt"))
    )
    snapshot_df = (
        topic_stats_df.join(group_stats_df, on=GROUP_COLS, how="inner")
        .withColumn("topic_share", F.col("topic_memo_cnt") / F.col("final_labeled_memo_cnt"))
        .withColumn("others_ratio", F.col("others_memo_cnt") / F.col("final_labeled_memo_cnt"))
        .withColumn(
            "low_topic_yn",
            (F.col("final_labeled_memo_cnt") >= cfg["min_group_labeled_memo_count"])
            & (F.col("topic_share") < cfg["low_topic_share_threshold"]),
        )
        .withColumn(
            "new_topic_candidate_check_yn",
            (F.col("final_labeled_memo_cnt") >= cfg["min_group_labeled_memo_count"])
            & (F.col("others_ratio") >= cfg["others_ratio_threshold"]),
        )
        .select(
            F.lit(prompt_version).alias("prompt_version"),
            F.lit(taxonomy_version).alias("taxonomy_version"),
            F.lit(run_month).alias("run_month"),
            *[F.col(column) for column in GROUP_COLS],
            F.col("pred_topic").alias("source_topic"),
            "final_labeled_memo_cnt",
            "topic_memo_cnt",
            "others_memo_cnt",
            "topic_share",
            "others_ratio",
            "low_topic_yn",
            "new_topic_candidate_check_yn",
            F.current_timestamp().alias("updated_at"),
        )
    )

    if spark.catalog.tableExists(table_name):
        spark.sql(
            f"""DELETE FROM {table_name}
            WHERE prompt_version = '{prompt_version.replace("'", "''")}'
              AND taxonomy_version = '{taxonomy_version.replace("'", "''")}'
              AND run_month = '{run_month}'
              AND ({_group_condition_sql(target_groups)})"""
        )
        snapshot_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table_name)
    else:
        snapshot_df.write.format("delta").mode("overwrite").saveAsTable(table_name)
    return table_name, snapshot_df


def _latest_topic_group_df(spark: SparkSession, config: dict[str, Any], target_groups: list[tuple[str, str, int]]) -> DataFrame:
    table_name = get_output_table(config, "topic_group")
    if not spark.catalog.tableExists(table_name):
        return spark.createDataFrame([], "cate_1_depth string, cate_2_depth string, sc_measurement int, source_topic string, source_topic_group string")
    window = Window.partitionBy(*GROUP_COLS, "topic").orderBy(
        F.col("created_at").desc_nulls_last(), F.col("run_id").desc_nulls_last()
    )
    return (
        spark.table(table_name)
        .where(F.expr(_group_condition_sql(target_groups)))
        .withColumn("_rn", F.row_number().over(window))
        .where(F.col("_rn") == 1)
        .select(*GROUP_COLS, F.col("topic").alias("source_topic"), F.col("topic_group").alias("source_topic_group"))
    )


def _active_low_topic_df(
    spark: SparkSession,
    config: dict[str, Any],
    health_table: str,
    target_groups: list[tuple[str, str, int]],
) -> DataFrame:
    cfg = _cfg(config)
    run_month = _run_month(config)
    prompt_version = _value(config, "version", "prompt_version")
    taxonomy_version = _value(config, "version", "taxonomy_version")
    health_df = (
        spark.table(health_table)
        .where(F.col("prompt_version") == prompt_version)
        .where(F.col("taxonomy_version") == taxonomy_version)
        .where(F.expr(_group_condition_sql(target_groups)))
    )
    current_df = health_df.where(F.col("run_month") == run_month).alias("current")
    previous_df = health_df.where(
        F.col("run_month") == F.date_format(F.add_months(F.to_date(F.lit(run_month)), -1), "yyyy-MM-01")
    ).alias("previous")
    join_cols = GROUP_COLS + ["source_topic"]
    result = current_df.join(previous_df, on=join_cols, how="inner").where(
        F.col("current.low_topic_yn") & F.col("previous.low_topic_yn")
    )
    if cfg["protected_topics"]:
        result = result.where(~F.col("source_topic").isin(sorted(cfg["protected_topics"])))
    return result.select(*[F.col(column) for column in join_cols]).dropDuplicates()


def _refresh_active_mapping(
    spark: SparkSession,
    config: dict[str, Any],
    latest_final_df: DataFrame,
    health_table: str,
    target_groups: list[tuple[str, str, int]],
    *,
    created_by: str,
) -> tuple[str, int]:
    cfg = _cfg(config)
    table_name = get_output_table(config, cfg["active_mapping_table_key"])
    prompt_version = _value(config, "version", "prompt_version")
    taxonomy_version = _value(config, "version", "taxonomy_version")
    run_month = _run_month(config)
    group_condition = _group_condition_sql(target_groups)
    source_topics_df = (
        latest_final_df.where(F.col("pred_topic_type") == "topic")
        .where(F.col("pred_topic") != "기타")
        .select(*GROUP_COLS, F.col("pred_topic").alias("source_topic"))
        .dropDuplicates()
        .join(_latest_topic_group_df(spark, config, target_groups), on=GROUP_COLS + ["source_topic"], how="left")
        .fillna({"source_topic_group": "기타"})
    )
    if source_topics_df.limit(1).count() == 0:
        return table_name, 0

    low_topic_df = _active_low_topic_df(spark, config, health_table, target_groups)
    low_topic_count = low_topic_df.count()
    existing_active_df = None
    if spark.catalog.tableExists(table_name):
        existing_active_df = (
            spark.table(table_name)
            .where(F.col("is_active") == F.lit(True))
            .where(F.col("prompt_version") == prompt_version)
            .where(F.col("taxonomy_version") == taxonomy_version)
            .select(
                *GROUP_COLS,
                "source_topic",
                "active_taxonomy_version",
                "active_topic",
                "active_topic_type",
                "active_topic_group",
                "mapping_reason",
            )
        )

    baseline_version = f"{taxonomy_version}_active_base"
    existing_columns = (
        existing_active_df.select(
            *GROUP_COLS,
            "source_topic",
            F.col("active_taxonomy_version").alias("_existing_active_taxonomy_version"),
            F.col("active_topic").alias("_existing_active_topic"),
            F.col("active_topic_type").alias("_existing_active_topic_type"),
            F.col("active_topic_group").alias("_existing_active_topic_group"),
            F.col("mapping_reason").alias("_existing_mapping_reason"),
        )
        if existing_active_df is not None
        else None
    )
    mapping_df = source_topics_df
    if existing_columns is not None:
        mapping_df = mapping_df.join(existing_columns, on=GROUP_COLS + ["source_topic"], how="left")
    else:
        for column in (
            "_existing_active_taxonomy_version",
            "_existing_active_topic",
            "_existing_active_topic_type",
            "_existing_active_topic_group",
            "_existing_mapping_reason",
        ):
            mapping_df = mapping_df.withColumn(column, F.lit(None).cast("string"))

    low_marker_df = low_topic_df.withColumn("_merge_to_others", F.lit(True))
    mapping_df = mapping_df.join(low_marker_df, on=GROUP_COLS + ["source_topic"], how="left")
    new_active_version = f"{taxonomy_version}_active_{run_month[:7].replace('-', '')}"
    mapping_df = mapping_df.select(
        F.lit(prompt_version).alias("prompt_version"),
        F.lit(taxonomy_version).alias("taxonomy_version"),
        F.when(F.col("_merge_to_others"), F.lit(new_active_version))
        .otherwise(F.coalesce(F.col("_existing_active_taxonomy_version"), F.lit(baseline_version)))
        .alias("active_taxonomy_version"),
        *[F.col(column) for column in GROUP_COLS],
        F.col("source_topic"),
        F.lit("topic").alias("source_topic_type"),
        F.when(F.col("_merge_to_others"), F.lit("기타"))
        .otherwise(F.coalesce(F.col("_existing_active_topic"), F.col("source_topic")))
        .alias("active_topic"),
        F.when(F.col("_merge_to_others"), F.lit("others"))
        .otherwise(F.coalesce(F.col("_existing_active_topic_type"), F.lit("topic")))
        .alias("active_topic_type"),
        F.when(F.col("_merge_to_others"), F.lit("기타"))
        .otherwise(F.coalesce(F.col("_existing_active_topic_group"), F.col("source_topic_group")))
        .alias("active_topic_group"),
        F.when(F.col("_merge_to_others"), F.lit("low_share_two_consecutive_months"))
        .otherwise(F.coalesce(F.col("_existing_mapping_reason"), F.lit("initial_active_mapping")))
        .alias("mapping_reason"),
        F.lit(True).alias("is_active"),
        F.lit(run_month).alias("effective_from_month"),
        F.current_timestamp().alias("created_at"),
        F.lit(created_by).alias("created_by"),
    )

    if spark.catalog.tableExists(table_name):
        # Supersede only mappings that are actually written again. This retains
        # historic versions and avoids touching other category groups.
        update_groups = mapping_df.select(*GROUP_COLS).dropDuplicates().collect()
        if update_groups:
            spark.sql(
                f"""UPDATE {table_name} SET is_active = false
                WHERE is_active = true
                  AND prompt_version = '{prompt_version.replace("'", "''")}'
                  AND taxonomy_version = '{taxonomy_version.replace("'", "''")}'
                  AND ({group_condition})"""
            )
        mapping_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table_name)
    else:
        mapping_df.write.format("delta").mode("overwrite").saveAsTable(table_name)
    return table_name, int(low_topic_count)


def _save_new_topic_candidates(
    spark: SparkSession,
    config: dict[str, Any],
    latest_final_df: DataFrame,
    health_table: str,
    target_groups: list[tuple[str, str, int]],
) -> tuple[str, int]:
    """Persist repeated Others patterns only for groups above the 15% threshold.

    This is a candidate queue, not an automatic taxonomy mutation. The active
    taxonomy stays stable until a later taxonomy version explicitly adopts a
    reviewed candidate.
    """
    cfg = _cfg(config)
    table_name = get_output_table(config, cfg["new_candidate_table_key"])
    run_month = _run_month(config)
    prompt_version = _value(config, "version", "prompt_version")
    taxonomy_version = _value(config, "version", "taxonomy_version")
    eligible_groups_df = (
        spark.table(health_table)
        .where(F.col("prompt_version") == prompt_version)
        .where(F.col("taxonomy_version") == taxonomy_version)
        .where(F.col("run_month") == run_month)
        .where(F.col("new_topic_candidate_check_yn"))
        .select(*GROUP_COLS)
        .dropDuplicates()
    )
    others_df = (
        latest_final_df.where(
            (F.col("pred_topic_type") == "others") | (F.col("pred_topic") == "기타")
        )
        .join(eligible_groups_df, on=GROUP_COLS, how="inner")
        .where(F.col("memo_norm").isNotNull())
    )
    candidates_df = (
        others_df.groupBy(*GROUP_COLS, "memo_norm")
        .agg(
            F.countDistinct("memo_id").alias("candidate_distinct_memo_id_cnt"),
            F.first("memo", ignorenulls=True).alias("sample_memo"),
            F.first("match_reason", ignorenulls=True).alias("sample_match_reason"),
        )
        .where(F.col("candidate_distinct_memo_id_cnt") >= cfg["new_topic_candidate_min_distinct_memo_ids"])
        .select(
            F.lit(prompt_version).alias("prompt_version"),
            F.lit(taxonomy_version).alias("taxonomy_version"),
            F.lit(run_month).alias("run_month"),
            *[F.col(column) for column in GROUP_COLS],
            F.col("memo_norm").alias("candidate_pattern"),
            "candidate_distinct_memo_id_cnt",
            "sample_memo",
            "sample_match_reason",
            F.lit("others_ratio_15pct_repeated_pattern").alias("candidate_reason"),
            F.lit("pending_review").alias("status"),
            F.current_timestamp().alias("created_at"),
        )
    )
    if spark.catalog.tableExists(table_name):
        spark.sql(
            f"""DELETE FROM {table_name}
            WHERE prompt_version = '{prompt_version.replace("'", "''")}'
              AND taxonomy_version = '{taxonomy_version.replace("'", "''")}'
              AND run_month = '{run_month}'
              AND ({_group_condition_sql(target_groups)})"""
        )
        if candidates_df.limit(1).count() > 0:
            candidates_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table_name)
    else:
        candidates_df.write.format("delta").mode("overwrite").saveAsTable(table_name)
    return table_name, candidates_df.count()


def refresh_topic_lifecycle(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    target_groups: list[tuple[str, str, int]],
    created_by: str = "topic_lifecycle",
) -> dict[str, Any]:
    """Refresh monthly health and active topic mapping for an incremental batch."""
    cfg = _cfg(config)
    if not cfg["enabled"] or not target_groups:
        return {"skipped": True, "reason": "disabled_or_no_target_groups"}

    latest_final_df = _latest_final_df(spark, config, target_groups)
    health_table, health_snapshot_df = _save_monthly_health(
        spark, config, latest_final_df, target_groups
    )
    mapping_table, merged_topic_count = _refresh_active_mapping(
        spark,
        config,
        latest_final_df,
        health_table,
        target_groups,
        created_by=created_by,
    )
    candidate_table, candidate_count = _save_new_topic_candidates(
        spark,
        config,
        latest_final_df,
        health_table,
        target_groups,
    )
    return {
        "health_table": health_table,
        "active_mapping_table": mapping_table,
        "run_month": _run_month(config),
        "health_topic_rows": health_snapshot_df.count(),
        "merged_to_others_topic_count": merged_topic_count,
        "new_topic_candidate_table": candidate_table,
        "new_topic_candidate_count": candidate_count,
        "new_topic_candidate_rule": {
            "min_group_labeled_memo_count": cfg["min_group_labeled_memo_count"],
            "others_ratio_threshold": cfg["others_ratio_threshold"],
            "min_distinct_memo_ids": cfg["new_topic_candidate_min_distinct_memo_ids"],
        },
    }
