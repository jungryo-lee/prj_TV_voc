"""Writers for taxonomy classification summary outputs."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from common.config_loader import get_output_table


CLASSIFICATION_SUMMARY_WRITE_SCHEMA = T.StructType(
    [
        T.StructField("cate_1_depth", T.StringType(), True),
        T.StructField("cate_2_depth", T.StringType(), True),
        T.StructField("sc_measurement", T.IntegerType(), True),
        T.StructField("pred_topic", T.StringType(), True),
        T.StructField("pred_topic_type", T.StringType(), True),
        T.StructField("memo_cnt", T.IntegerType(), True),
        T.StructField("topic_ratio", T.DoubleType(), True),
        T.StructField("topic_ratio_pct", T.DoubleType(), True),
        T.StructField("group_total_cnt", T.IntegerType(), True),
        T.StructField("overall_cnt", T.IntegerType(), True),
        T.StructField("overall_ratio", T.DoubleType(), True),
        T.StructField("others_cnt", T.IntegerType(), True),
        T.StructField("others_ratio", T.DoubleType(), True),
        T.StructField("llm_used_cnt", T.IntegerType(), True),
        T.StructField("llm_used_ratio", T.DoubleType(), True),
        T.StructField("review_needed_cnt", T.IntegerType(), True),
        T.StructField("review_needed_ratio", T.DoubleType(), True),
        T.StructField("run_id", T.StringType(), True),
        T.StructField("run_date", T.StringType(), True),
        T.StructField("pipeline_stage", T.StringType(), True),
        T.StructField("prompt_version", T.StringType(), True),
        T.StructField("taxonomy_version", T.StringType(), True),
        T.StructField("model_version", T.StringType(), True),
        T.StructField("pipeline_version", T.StringType(), True),
        T.StructField("source_period_start", T.StringType(), True),
        T.StructField("source_period_end", T.StringType(), True),
        T.StructField("is_latest", T.BooleanType(), True),
        T.StructField("created_at", T.StringType(), True),
        T.StructField("created_by", T.StringType(), True),
    ]
)


def _runtime_value(config: dict[str, Any], key: str, default: str = "") -> str:
    """Read resolved runtime metadata with a safe fallback."""
    return str(config.get("runtime", {}).get(key, default) or default)


def _version_value(config: dict[str, Any], key: str, default: str = "") -> str:
    """Read version metadata with a safe fallback."""
    return str(config.get("version", {}).get(key, default) or default)


def build_classification_summary_df(
    detail_df: DataFrame,
    config: dict[str, Any],
    *,
    created_by: str = "codex",
    pipeline_stage: str = "classification_summary",
    is_latest: bool = True,
    source_period_start: str | None = None,
    source_period_end: str | None = None,
) -> DataFrame:
    """Aggregate classification detail rows into topic-level summary rows."""
    run_id = _runtime_value(config, "resolved_run_id")
    run_date = _runtime_value(config, "resolved_run_date")
    prompt_version = _version_value(config, "prompt_version")
    taxonomy_version = _version_value(config, "taxonomy_version")
    pipeline_version = _version_value(config, "pipeline_version")
    created_at = datetime.utcnow().isoformat(timespec="seconds")

    group_keys = [
        "cate_1_depth",
        "cate_2_depth",
        "sc_measurement",
        "run_id",
        "run_date",
        "prompt_version",
        "taxonomy_version",
        "model_version",
        "pipeline_version",
    ]

    group_stats_df = (
        detail_df.groupBy(*group_keys)
        .agg(
            F.count("*").alias("group_total_cnt"),
            F.sum(F.when(F.col("pred_topic_type") == "overall", 1).otherwise(0)).alias(
                "overall_cnt"
            ),
            F.sum(F.when(F.col("pred_topic_type") == "others", 1).otherwise(0)).alias(
                "others_cnt"
            ),
            F.sum(F.when(F.col("llm_used_yn") == True, 1).otherwise(0)).alias(
                "llm_used_cnt"
            ),
            F.sum(F.when(F.col("review_needed_yn") == True, 1).otherwise(0)).alias(
                "review_needed_cnt"
            ),
        )
        .withColumn(
            "overall_ratio",
            F.when(
                F.col("group_total_cnt") > 0,
                F.col("overall_cnt") / F.col("group_total_cnt"),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "others_ratio",
            F.when(
                F.col("group_total_cnt") > 0,
                F.col("others_cnt") / F.col("group_total_cnt"),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "llm_used_ratio",
            F.when(
                F.col("group_total_cnt") > 0,
                F.col("llm_used_cnt") / F.col("group_total_cnt"),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "review_needed_ratio",
            F.when(
                F.col("group_total_cnt") > 0,
                F.col("review_needed_cnt") / F.col("group_total_cnt"),
            ).otherwise(F.lit(0.0)),
        )
    )

    topic_stats_df = (
        detail_df.groupBy(
            *group_keys,
            "pred_topic",
            "pred_topic_type",
        )
        .agg(F.count("*").alias("memo_cnt"))
        .join(group_stats_df, on=group_keys, how="left")
        .withColumn(
            "topic_ratio",
            F.when(
                F.col("group_total_cnt") > 0,
                F.col("memo_cnt") / F.col("group_total_cnt"),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn("topic_ratio_pct", F.col("topic_ratio") * F.lit(100.0))
        .withColumn("pipeline_stage", F.lit(pipeline_stage))
        .withColumn("source_period_start", F.lit(source_period_start))
        .withColumn("source_period_end", F.lit(source_period_end))
        .withColumn("is_latest", F.lit(bool(is_latest)))
        .withColumn("created_at", F.lit(created_at))
        .withColumn("created_by", F.lit(created_by))
    )

    return topic_stats_df.select(
        "cate_1_depth",
        "cate_2_depth",
        "sc_measurement",
        "pred_topic",
        "pred_topic_type",
        F.col("memo_cnt").cast("int").alias("memo_cnt"),
        F.round("topic_ratio", 4).alias("topic_ratio"),
        F.round("topic_ratio_pct", 2).alias("topic_ratio_pct"),
        F.col("group_total_cnt").cast("int").alias("group_total_cnt"),
        F.col("overall_cnt").cast("int").alias("overall_cnt"),
        F.round("overall_ratio", 4).alias("overall_ratio"),
        F.col("others_cnt").cast("int").alias("others_cnt"),
        F.round("others_ratio", 4).alias("others_ratio"),
        F.col("llm_used_cnt").cast("int").alias("llm_used_cnt"),
        F.round("llm_used_ratio", 4).alias("llm_used_ratio"),
        F.col("review_needed_cnt").cast("int").alias("review_needed_cnt"),
        F.round("review_needed_ratio", 4).alias("review_needed_ratio"),
        "run_id",
        "run_date",
        "pipeline_stage",
        "prompt_version",
        "taxonomy_version",
        "model_version",
        "pipeline_version",
        "source_period_start",
        "source_period_end",
        "is_latest",
        "created_at",
        "created_by",
    )


def _delete_existing_group_rows(
    spark: SparkSession,
    table_name: str,
    df: DataFrame,
) -> None:
    """Delete existing rows for the same group/model/version combination."""
    if not spark.catalog.tableExists(table_name):
        return

    keys_df = (
        df.select(
            "cate_1_depth",
            "cate_2_depth",
            "sc_measurement",
            "model_version",
            "prompt_version",
            "taxonomy_version",
        )
        .dropDuplicates()
    )
    keys_df.createOrReplaceTempView("_tmp_classification_summary_keys")

    spark.sql(
        f"""
        DELETE FROM {table_name}
        WHERE EXISTS (
            SELECT 1
            FROM _tmp_classification_summary_keys src
            WHERE {table_name}.cate_1_depth = src.cate_1_depth
              AND {table_name}.cate_2_depth = src.cate_2_depth
              AND {table_name}.sc_measurement = src.sc_measurement
              AND {table_name}.model_version = src.model_version
              AND {table_name}.prompt_version = src.prompt_version
              AND {table_name}.taxonomy_version = src.taxonomy_version
        )
        """
    )


def save_classification_summary(
    spark: SparkSession,
    config: dict[str, Any],
    detail_df: DataFrame,
    *,
    created_by: str = "codex",
    pipeline_stage: str = "classification_summary",
    write_mode: str = "replace_groups",
    is_latest: bool = True,
    source_period_start: str | None = None,
    source_period_end: str | None = None,
) -> str:
    """Aggregate and save classification summary rows to the configured output table."""
    table_name = get_output_table(config, "classification_summary")
    summary_df = build_classification_summary_df(
        detail_df=detail_df,
        config=config,
        created_by=created_by,
        pipeline_stage=pipeline_stage,
        is_latest=is_latest,
        source_period_start=source_period_start,
        source_period_end=source_period_end,
    )

    summary_df = spark.createDataFrame(
        summary_df.collect(),
        schema=CLASSIFICATION_SUMMARY_WRITE_SCHEMA,
    )

    if write_mode == "overwrite":
        summary_df.write.mode("overwrite").format("delta").saveAsTable(table_name)
        return table_name

    if write_mode == "replace_groups":
        _delete_existing_group_rows(
            spark=spark,
            table_name=table_name,
            df=summary_df,
        )
        summary_df.write.mode("append").format("delta").saveAsTable(table_name)
        return table_name

    if write_mode == "append":
        summary_df.write.mode("append").format("delta").saveAsTable(table_name)
        return table_name

    raise ValueError(f"Unsupported write_mode: {write_mode}")
