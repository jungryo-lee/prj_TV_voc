"""Expand memo_id-level classification results back to raw review rows."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from common.config_loader import get_output_table, get_source_table
from common.memo_id import with_memo_id


CLASSIFICATION_FULL_WRITE_SCHEMA = T.StructType(
    [
        T.StructField("memo_id", T.StringType(), False),
        T.StructField("memo", T.StringType(), True),
        T.StructField("memo_norm", T.StringType(), True),
        T.StructField("cate_1_depth", T.StringType(), True),
        T.StructField("cate_2_depth", T.StringType(), True),
        T.StructField("sc_measurement", T.IntegerType(), True),
        T.StructField("year", T.StringType(), True),
        T.StructField("country", T.StringType(), True),
        T.StructField("brand_name", T.StringType(), True),
        T.StructField("device_type", T.StringType(), True),
        T.StructField("pred_topic", T.StringType(), True),
        T.StructField("pred_topic_type", T.StringType(), True),
        T.StructField("classification_stage", T.StringType(), True),
        T.StructField("confidence_score", T.DoubleType(), True),
        T.StructField("candidate_topics_json", T.StringType(), True),
        T.StructField("match_reason", T.StringType(), True),
        T.StructField("llm_used_yn", T.BooleanType(), True),
        T.StructField("review_needed_yn", T.BooleanType(), True),
        T.StructField("classification_run_id", T.StringType(), True),
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


def _clean_text(value: Any) -> str:
    """Collapse whitespace for stable SQL/table payloads."""
    return " ".join(str(value or "").split()).strip()


def _sql_escape(value: str) -> str:
    """Escape a string for SQL literal interpolation."""
    return str(value).replace("'", "''")


def _runtime_value(config: dict[str, Any], key: str, default: str = "") -> str:
    """Read resolved runtime metadata with a safe fallback."""
    return str(config.get("runtime", {}).get(key, default) or default)


def _version_value(config: dict[str, Any], key: str, default: str = "") -> str:
    """Read version metadata with a safe fallback."""
    return str(config.get("version", {}).get(key, default) or default)


def build_raw_review_query(
    config: dict[str, Any],
    *,
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
) -> str:
    """Build raw review query for expansion."""
    source_table = get_source_table(config, "raw_review_table")
    filters = [
        "memo is not null",
        "length(trim(memo)) > 0",
    ]

    if cate_1_depth is not None:
        filters.append(f"cate_1_depth = '{_sql_escape(cate_1_depth)}'")
    if cate_2_depth is not None:
        filters.append(f"cate_2_depth = '{_sql_escape(cate_2_depth)}'")
    if sc_measurement is not None:
        filters.append(f"sc_measurement = {int(sc_measurement)}")

    where_clause = "\n  and ".join(filters)
    return f"""
select
    cate_1_depth,
    cate_2_depth,
    sc_measurement,
    year,
    country,
    brand_name,
    device_type,
    memo
from {source_table}
where {where_clause}
""".strip()


def load_latest_classification_detail_df(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
    model_version: str | None = None,
    prompt_version: str | None = None,
    taxonomy_version: str | None = None,
) -> DataFrame:
    """Load one latest classification row per group/memo_id/version."""
    detail_table = get_output_table(config, "classification_detail")
    df = spark.table(detail_table)

    if cate_1_depth is not None:
        df = df.where(F.col("cate_1_depth") == cate_1_depth)
    if cate_2_depth is not None:
        df = df.where(F.col("cate_2_depth") == cate_2_depth)
    if sc_measurement is not None:
        df = df.where(F.col("sc_measurement") == int(sc_measurement))
    if model_version is not None:
        df = df.where(F.col("model_version") == model_version)
    if prompt_version is not None:
        df = df.where(F.col("prompt_version") == prompt_version)
    if taxonomy_version is not None:
        df = df.where(F.col("taxonomy_version") == taxonomy_version)

    latest_window = Window.partitionBy(
        "cate_1_depth",
        "cate_2_depth",
        "sc_measurement",
        "memo_id",
        "model_version",
        "prompt_version",
        "taxonomy_version",
    ).orderBy(
        F.col("is_latest").desc_nulls_last(),
        F.col("created_at").desc_nulls_last(),
        F.col("run_id").desc_nulls_last(),
    )

    return (
        df.withColumn("_detail_rn", F.row_number().over(latest_window))
        .where(F.col("_detail_rn") == 1)
        .drop("_detail_rn")
    )


def build_classification_full_df(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
    model_version: str | None = None,
    prompt_version: str | None = None,
    taxonomy_version: str | None = None,
    created_by: str = "codex",
    pipeline_stage: str = "classification_full",
    is_latest: bool = True,
    source_period_start: str | None = None,
    source_period_end: str | None = None,
) -> DataFrame:
    """Join memo_id-level classification back to raw review rows."""
    run_id = _runtime_value(config, "resolved_run_id")
    run_date = _runtime_value(config, "resolved_run_date")
    pipeline_version = _version_value(config, "pipeline_version")
    created_at = datetime.utcnow().isoformat(timespec="seconds")

    raw_df = spark.sql(
        build_raw_review_query(
            config,
            cate_1_depth=cate_1_depth,
            cate_2_depth=cate_2_depth,
            sc_measurement=sc_measurement,
        )
    ).transform(with_memo_id)

    detail_df = load_latest_classification_detail_df(
        spark,
        config,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        model_version=model_version,
        prompt_version=prompt_version,
        taxonomy_version=taxonomy_version,
    ).select(
        "memo_id",
        "cate_1_depth",
        "cate_2_depth",
        "sc_measurement",
        "pred_topic",
        "pred_topic_type",
        "classification_stage",
        "confidence_score",
        "candidate_topics_json",
        "match_reason",
        "llm_used_yn",
        "review_needed_yn",
        F.col("run_id").alias("classification_run_id"),
        "prompt_version",
        "taxonomy_version",
        "model_version",
    )

    join_keys = ["cate_1_depth", "cate_2_depth", "sc_measurement", "memo_id"]
    expanded_df = raw_df.join(detail_df, on=join_keys, how="inner")

    return expanded_df.select(
        "memo_id",
        "memo",
        "memo_norm",
        "cate_1_depth",
        "cate_2_depth",
        F.col("sc_measurement").cast("int").alias("sc_measurement"),
        F.col("year").cast("string").alias("year"),
        F.col("country").cast("string").alias("country"),
        F.col("brand_name").cast("string").alias("brand_name"),
        F.col("device_type").cast("string").alias("device_type"),
        "pred_topic",
        "pred_topic_type",
        "classification_stage",
        F.col("confidence_score").cast("double").alias("confidence_score"),
        "candidate_topics_json",
        "match_reason",
        F.col("llm_used_yn").cast("boolean").alias("llm_used_yn"),
        F.col("review_needed_yn").cast("boolean").alias("review_needed_yn"),
        "classification_run_id",
        F.lit(run_id).alias("run_id"),
        F.lit(run_date).alias("run_date"),
        F.lit(pipeline_stage).alias("pipeline_stage"),
        "prompt_version",
        "taxonomy_version",
        "model_version",
        F.lit(pipeline_version).alias("pipeline_version"),
        F.lit(source_period_start).alias("source_period_start"),
        F.lit(source_period_end).alias("source_period_end"),
        F.lit(bool(is_latest)).alias("is_latest"),
        F.lit(created_at).alias("created_at"),
        F.lit(created_by).alias("created_by"),
    )


def _delete_existing_group_rows(
    spark: SparkSession,
    table_name: str,
    df: DataFrame,
) -> None:
    """Delete existing expanded rows for the same group/model/version."""
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
    keys_df.createOrReplaceTempView("_tmp_classification_full_keys")

    spark.sql(
        f"""
        DELETE FROM {table_name}
        WHERE EXISTS (
            SELECT 1
            FROM _tmp_classification_full_keys src
            WHERE {table_name}.cate_1_depth = src.cate_1_depth
              AND {table_name}.cate_2_depth = src.cate_2_depth
              AND {table_name}.sc_measurement = src.sc_measurement
              AND {table_name}.model_version = src.model_version
              AND {table_name}.prompt_version = src.prompt_version
              AND {table_name}.taxonomy_version = src.taxonomy_version
        )
        """
    )


def save_classification_full(
    spark: SparkSession,
    config: dict[str, Any],
    full_df: DataFrame,
    *,
    write_mode: str = "replace_groups",
) -> str:
    """Save expanded raw-row classification output."""
    table_name = get_output_table(config, "classification_full")
    write_df = full_df.select(
        [F.col(field.name).cast(field.dataType).alias(field.name) for field in CLASSIFICATION_FULL_WRITE_SCHEMA]
    )

    if write_mode == "overwrite":
        write_df.write.mode("overwrite").format("delta").saveAsTable(table_name)
        return table_name

    if write_mode == "replace_groups":
        _delete_existing_group_rows(
            spark=spark,
            table_name=table_name,
            df=write_df,
        )
        write_df.write.mode("append").format("delta").saveAsTable(table_name)
        return table_name

    if write_mode == "append":
        write_df.write.mode("append").format("delta").saveAsTable(table_name)
        return table_name

    raise ValueError(f"Unsupported write_mode: {write_mode}")


def expand_and_save_classification_full(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
    model_version: str | None = None,
    prompt_version: str | None = None,
    taxonomy_version: str | None = None,
    created_by: str = "codex",
    write_mode: str = "replace_groups",
    source_period_start: str | None = None,
    source_period_end: str | None = None,
) -> dict[str, Any]:
    """Build, save, and summarize expanded classification rows."""
    full_df = build_classification_full_df(
        spark=spark,
        config=config,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        model_version=model_version,
        prompt_version=prompt_version,
        taxonomy_version=taxonomy_version,
        created_by=created_by,
        source_period_start=source_period_start,
        source_period_end=source_period_end,
    )
    row_count = full_df.count()
    distinct_memo_count = full_df.select("memo_id").dropDuplicates().count()
    table_name = save_classification_full(
        spark=spark,
        config=config,
        full_df=full_df,
        write_mode=write_mode,
    )

    return {
        "table_name": table_name,
        "row_count": int(row_count),
        "distinct_memo_count": int(distinct_memo_count),
        "unmatched_raw_rows": 0,
    }
