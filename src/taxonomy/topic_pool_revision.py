"""Build topic-pool revision proposals from approved review decisions."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from common.config_loader import get_output_table


TOPIC_POOL_REVISION_WRITE_SCHEMA = T.StructType(
    [
        T.StructField("revision_id", T.StringType(), False),
        T.StructField("decision_id", T.StringType(), True),
        T.StructField("revision_type", T.StringType(), True),
        T.StructField("cate_1_depth", T.StringType(), True),
        T.StructField("cate_2_depth", T.StringType(), True),
        T.StructField("sc_measurement", T.IntegerType(), True),
        T.StructField("source_candidate_type", T.StringType(), True),
        T.StructField("source_memo_norm", T.StringType(), True),
        T.StructField("sample_memo", T.StringType(), True),
        T.StructField("approved_action", T.StringType(), True),
        T.StructField("approved_topic", T.StringType(), True),
        T.StructField("proposed_topic", T.StringType(), True),
        T.StructField("proposed_description", T.StringType(), True),
        T.StructField("proposed_representative_memos_json", T.StringType(), True),
        T.StructField("candidate_cnt", T.IntegerType(), True),
        T.StructField("candidate_distinct_memo_id_cnt", T.IntegerType(), True),
        T.StructField("candidate_ratio", T.DoubleType(), True),
        T.StructField("revision_status", T.StringType(), True),
        T.StructField("reviewer", T.StringType(), True),
        T.StructField("review_comment", T.StringType(), True),
        T.StructField("run_id", T.StringType(), True),
        T.StructField("run_date", T.StringType(), True),
        T.StructField("prompt_version", T.StringType(), True),
        T.StructField("taxonomy_version", T.StringType(), True),
        T.StructField("model_version", T.StringType(), True),
        T.StructField("pipeline_version", T.StringType(), True),
        T.StructField("created_at", T.StringType(), True),
        T.StructField("created_by", T.StringType(), True),
    ]
)


def _clean_text(value: Any) -> str:
    """Collapse whitespace for stable table payloads."""
    return " ".join(str(value or "").split()).strip()


def _runtime_value(config: dict[str, Any], key: str, default: str = "") -> str:
    """Read resolved runtime metadata with a safe fallback."""
    return str(config.get("runtime", {}).get(key, default) or default)


def _version_value(config: dict[str, Any], key: str, default: str = "") -> str:
    """Read version metadata with a safe fallback."""
    return str(config.get("version", {}).get(key, default) or default)


def load_approved_review_decisions(
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
    """Load approved decisions that can drive topic-pool revisions."""
    table_name = get_output_table(config, "review_decision")
    df = (
        spark.table(table_name)
        .where(F.col("decision_status") == "approved")
        .where(F.col("approved_action").isin("create_new_topic", "revise_topic_pool"))
    )

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

    return df


def build_topic_pool_revision_df(
    decision_df: DataFrame,
    config: dict[str, Any],
    *,
    created_by: str = "codex",
    revision_status: str = "proposed",
) -> DataFrame:
    """Convert approved decisions into topic-pool revision proposal rows."""
    run_id = _runtime_value(config, "resolved_run_id")
    run_date = _runtime_value(config, "resolved_run_date")
    pipeline_version = _version_value(config, "pipeline_version")
    created_at = datetime.utcnow().isoformat(timespec="seconds")

    proposed_topic = F.when(
        F.length(F.trim(F.col("approved_topic"))) > 0,
        F.col("approved_topic"),
    ).otherwise(F.col("memo_norm"))

    revision_type = F.when(
        F.col("approved_action") == "create_new_topic",
        F.lit("add_topic"),
    ).otherwise(F.lit("revise_topic_pool"))

    revision_id = F.sha2(
        F.concat_ws(
            "||",
            F.coalesce(F.col("decision_id"), F.lit("")),
            F.coalesce(F.col("approved_action"), F.lit("")),
            F.coalesce(proposed_topic, F.lit("")),
            F.coalesce(F.col("model_version"), F.lit("")),
            F.coalesce(F.col("prompt_version"), F.lit("")),
            F.coalesce(F.col("taxonomy_version"), F.lit("")),
        ),
        256,
    )

    return decision_df.select(
        revision_id.alias("revision_id"),
        "decision_id",
        revision_type.alias("revision_type"),
        "cate_1_depth",
        "cate_2_depth",
        F.col("sc_measurement").cast("int").alias("sc_measurement"),
        F.col("candidate_type").alias("source_candidate_type"),
        F.col("memo_norm").alias("source_memo_norm"),
        "sample_memo",
        "approved_action",
        "approved_topic",
        proposed_topic.alias("proposed_topic"),
        F.concat(
            F.lit("Approved refinement candidate from review decision: "),
            F.coalesce(F.col("sample_memo"), F.col("memo_norm"), F.lit("")),
        ).alias("proposed_description"),
        F.to_json(F.array(F.coalesce(F.col("sample_memo"), F.col("memo_norm")))).alias(
            "proposed_representative_memos_json"
        ),
        F.col("candidate_cnt").cast("int").alias("candidate_cnt"),
        F.col("candidate_distinct_memo_id_cnt").cast("int").alias(
            "candidate_distinct_memo_id_cnt"
        ),
        F.col("candidate_ratio").cast("double").alias("candidate_ratio"),
        F.lit(revision_status).alias("revision_status"),
        "reviewer",
        "review_comment",
        F.coalesce(F.col("run_id"), F.lit(run_id)).alias("run_id"),
        F.coalesce(F.col("run_date"), F.lit(run_date)).alias("run_date"),
        "prompt_version",
        "taxonomy_version",
        "model_version",
        F.lit(pipeline_version).alias("pipeline_version"),
        F.lit(created_at).alias("created_at"),
        F.lit(created_by).alias("created_by"),
    )


def _delete_existing_revision_rows(
    spark: SparkSession,
    table_name: str,
    revision_df: DataFrame,
) -> None:
    """Delete existing revision rows by stable revision_id before append."""
    if not spark.catalog.tableExists(table_name):
        return

    revision_df.select("revision_id").dropDuplicates().createOrReplaceTempView(
        "_tmp_topic_pool_revision_ids"
    )

    spark.sql(
        f"""
        DELETE FROM {table_name}
        WHERE EXISTS (
            SELECT 1
            FROM _tmp_topic_pool_revision_ids src
            WHERE {table_name}.revision_id = src.revision_id
        )
        """
    )


def save_topic_pool_revisions(
    spark: SparkSession,
    config: dict[str, Any],
    revision_df: DataFrame,
    *,
    write_mode: str = "replace_revisions",
) -> str:
    """Save topic-pool revision proposal rows."""
    table_name = get_output_table(config, "topic_pool_revision")
    write_df = revision_df.select(
        [
            F.col(field.name).cast(field.dataType).alias(field.name)
            for field in TOPIC_POOL_REVISION_WRITE_SCHEMA
        ]
    )

    if write_mode == "overwrite":
        write_df.write.mode("overwrite").format("delta").saveAsTable(table_name)
        return table_name

    if write_mode == "replace_revisions":
        _delete_existing_revision_rows(
            spark=spark,
            table_name=table_name,
            revision_df=write_df,
        )
        write_df.write.mode("append").format("delta").saveAsTable(table_name)
        return table_name

    if write_mode == "append":
        write_df.write.mode("append").format("delta").saveAsTable(table_name)
        return table_name

    raise ValueError(f"Unsupported write_mode: {write_mode}")


def build_and_save_topic_pool_revisions(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
    model_version: str | None = None,
    prompt_version: str | None = None,
    taxonomy_version: str | None = None,
    write_mode: str = "replace_revisions",
    created_by: str = "codex",
) -> dict[str, Any]:
    """Load approved decisions, build revisions, save them, and summarize."""
    decision_df = load_approved_review_decisions(
        spark,
        config,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        model_version=model_version,
        prompt_version=prompt_version,
        taxonomy_version=taxonomy_version,
    )
    revision_df = build_topic_pool_revision_df(
        decision_df,
        config,
        created_by=created_by,
    )
    revision_count = revision_df.count()
    table_name = save_topic_pool_revisions(
        spark=spark,
        config=config,
        revision_df=revision_df,
        write_mode=write_mode,
    )

    return {
        "table_name": table_name,
        "decision_count": int(decision_df.count()),
        "revision_count": int(revision_count),
    }
