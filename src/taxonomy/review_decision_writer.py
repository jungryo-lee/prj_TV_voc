"""Writers for human review decisions on taxonomy refinement candidates."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.column import Column
from pyspark.sql import functions as F
from pyspark.sql import types as T

from common.config_loader import get_output_table


REVIEW_DECISION_WRITE_SCHEMA = T.StructType(
    [
        T.StructField("decision_id", T.StringType(), False),
        T.StructField("candidate_type", T.StringType(), True),
        T.StructField("cate_1_depth", T.StringType(), True),
        T.StructField("cate_2_depth", T.StringType(), True),
        T.StructField("sc_measurement", T.IntegerType(), True),
        T.StructField("memo_id", T.StringType(), True),
        T.StructField("memo_norm", T.StringType(), True),
        T.StructField("sample_memo", T.StringType(), True),
        T.StructField("current_pred_topic", T.StringType(), True),
        T.StructField("current_pred_topic_type", T.StringType(), True),
        T.StructField("suggested_action", T.StringType(), True),
        T.StructField("suggested_topic", T.StringType(), True),
        T.StructField("suggestion_score", T.DoubleType(), True),
        T.StructField("candidate_cnt", T.IntegerType(), True),
        T.StructField("candidate_distinct_memo_id_cnt", T.IntegerType(), True),
        T.StructField("candidate_ratio", T.DoubleType(), True),
        T.StructField("candidate_evidence_json", T.StringType(), True),
        T.StructField("decision_status", T.StringType(), True),
        T.StructField("approved_action", T.StringType(), True),
        T.StructField("approved_topic", T.StringType(), True),
        T.StructField("reviewer", T.StringType(), True),
        T.StructField("review_comment", T.StringType(), True),
        T.StructField("reviewed_at", T.StringType(), True),
        T.StructField("source_table_key", T.StringType(), True),
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

VALID_DECISION_STATUSES = {
    "pending",
    "approved",
    "rejected",
    "hold",
}

VALID_APPROVED_ACTIONS = {
    "",
    "reassign_existing_topic",
    "create_new_topic",
    "keep_others",
    "revise_topic_pool",
}


def _clean_text(value: Any) -> str:
    """Collapse whitespace for stable table payloads."""
    return " ".join(str(value or "").split()).strip()


def _runtime_value(config: dict[str, Any], key: str, default: str = "") -> str:
    """Read resolved runtime metadata with a safe fallback."""
    return str(config.get("runtime", {}).get(key, default) or default)


def _version_value(config: dict[str, Any], key: str, default: str = "") -> str:
    """Read version metadata with a safe fallback."""
    return str(config.get("version", {}).get(key, default) or default)


def _ensure_column(df: DataFrame, column_name: str, default_value: Any) -> DataFrame:
    """Add a column with a default value when it does not exist."""
    if column_name in df.columns:
        return df
    if isinstance(default_value, Column):
        return df.withColumn(column_name, default_value)
    return df.withColumn(column_name, F.lit(default_value))


def _empty_review_decision_df(spark: SparkSession) -> DataFrame:
    """Return an empty DataFrame with review decision schema."""
    return spark.createDataFrame([], schema=REVIEW_DECISION_WRITE_SCHEMA)


def build_existing_reassignment_decision_df(
    existing_topic_reassignment_df: DataFrame,
    config: dict[str, Any],
    *,
    source_table_key: str = "classification_full",
    decision_status: str = "pending",
    created_by: str = "codex",
) -> DataFrame:
    """Convert existing-topic reassignment candidates into decision rows."""
    if decision_status not in VALID_DECISION_STATUSES:
        raise ValueError(f"Unsupported decision_status: {decision_status}")

    run_id = _runtime_value(config, "resolved_run_id")
    run_date = _runtime_value(config, "resolved_run_date")
    pipeline_version = _version_value(config, "pipeline_version")
    created_at = datetime.utcnow().isoformat(timespec="seconds")

    df = existing_topic_reassignment_df
    if not df.columns:
        return _empty_review_decision_df(existing_topic_reassignment_df.sparkSession)

    df = _ensure_column(df, "candidate_topics_json", "[]")
    df = _ensure_column(df, "suggestion_reason", "")

    evidence_json = F.to_json(
        F.struct(
            F.col("candidate_topics_json").alias("candidate_topics_json"),
            F.col("suggestion_reason").alias("suggestion_reason"),
        )
    )

    decision_id = F.sha2(
        F.concat_ws(
            "||",
            F.lit("existing_topic_reassignment"),
            F.coalesce(F.col("cate_1_depth"), F.lit("")),
            F.coalesce(F.col("cate_2_depth"), F.lit("")),
            F.coalesce(F.col("sc_measurement").cast("string"), F.lit("")),
            F.coalesce(F.col("memo_id"), F.lit("")),
            F.coalesce(F.col("suggested_topic"), F.lit("")),
            F.coalesce(F.col("model_version"), F.lit("")),
            F.coalesce(F.col("prompt_version"), F.lit("")),
            F.coalesce(F.col("taxonomy_version"), F.lit("")),
        ),
        256,
    )

    return df.select(
        decision_id.alias("decision_id"),
        F.lit("existing_topic_reassignment").alias("candidate_type"),
        "cate_1_depth",
        "cate_2_depth",
        F.col("sc_measurement").cast("int").alias("sc_measurement"),
        "memo_id",
        "memo_norm",
        F.col("memo").alias("sample_memo"),
        "current_pred_topic",
        "current_pred_topic_type",
        "suggested_action",
        "suggested_topic",
        F.col("suggestion_score").cast("double").alias("suggestion_score"),
        F.lit(None).cast("int").alias("candidate_cnt"),
        F.lit(None).cast("int").alias("candidate_distinct_memo_id_cnt"),
        F.lit(None).cast("double").alias("candidate_ratio"),
        evidence_json.alias("candidate_evidence_json"),
        F.lit(decision_status).alias("decision_status"),
        F.lit("").alias("approved_action"),
        F.lit("").alias("approved_topic"),
        F.lit("").alias("reviewer"),
        F.lit("").alias("review_comment"),
        F.lit(None).cast("string").alias("reviewed_at"),
        F.lit(source_table_key).alias("source_table_key"),
        F.coalesce(F.col("run_id"), F.lit(run_id)).alias("run_id"),
        F.coalesce(F.col("run_date"), F.lit(run_date)).alias("run_date"),
        "prompt_version",
        "taxonomy_version",
        "model_version",
        F.lit(pipeline_version).alias("pipeline_version"),
        F.lit(created_at).alias("created_at"),
        F.lit(created_by).alias("created_by"),
    )


def build_new_topic_decision_df(
    new_topic_candidate_df: DataFrame,
    config: dict[str, Any],
    *,
    source_table_key: str = "classification_full",
    decision_status: str = "pending",
    created_by: str = "codex",
) -> DataFrame:
    """Convert repeated others-pattern candidates into decision rows."""
    if decision_status not in VALID_DECISION_STATUSES:
        raise ValueError(f"Unsupported decision_status: {decision_status}")

    run_id = _runtime_value(config, "resolved_run_id")
    run_date = _runtime_value(config, "resolved_run_date")
    pipeline_version = _version_value(config, "pipeline_version")
    created_at = datetime.utcnow().isoformat(timespec="seconds")

    df = new_topic_candidate_df
    if not df.columns:
        return _empty_review_decision_df(new_topic_candidate_df.sparkSession)

    df = _ensure_column(df, "match_reason_samples", F.array().cast("array<string>"))
    df = _ensure_column(df, "candidate_reason", "")

    evidence_json = F.to_json(
        F.struct(
            F.col("match_reason_samples").alias("match_reason_samples"),
            F.col("candidate_reason").alias("candidate_reason"),
        )
    )

    decision_id = F.sha2(
        F.concat_ws(
            "||",
            F.lit("new_topic_candidate"),
            F.coalesce(F.col("cate_1_depth"), F.lit("")),
            F.coalesce(F.col("cate_2_depth"), F.lit("")),
            F.coalesce(F.col("sc_measurement").cast("string"), F.lit("")),
            F.coalesce(F.col("memo_norm"), F.lit("")),
            F.coalesce(F.col("model_version"), F.lit("")),
            F.coalesce(F.col("prompt_version"), F.lit("")),
            F.coalesce(F.col("taxonomy_version"), F.lit("")),
        ),
        256,
    )

    return df.select(
        decision_id.alias("decision_id"),
        F.lit("new_topic_candidate").alias("candidate_type"),
        "cate_1_depth",
        "cate_2_depth",
        F.col("sc_measurement").cast("int").alias("sc_measurement"),
        F.lit("").alias("memo_id"),
        "memo_norm",
        F.col("sample_memo").alias("sample_memo"),
        F.lit("기타").alias("current_pred_topic"),
        F.lit("others").alias("current_pred_topic_type"),
        "suggested_action",
        F.lit("").alias("suggested_topic"),
        F.lit(None).cast("double").alias("suggestion_score"),
        F.col("candidate_cnt").cast("int").alias("candidate_cnt"),
        F.col("candidate_distinct_memo_id_cnt").cast("int").alias(
            "candidate_distinct_memo_id_cnt"
        ),
        F.col("candidate_ratio").cast("double").alias("candidate_ratio"),
        evidence_json.alias("candidate_evidence_json"),
        F.lit(decision_status).alias("decision_status"),
        F.lit("").alias("approved_action"),
        F.lit("").alias("approved_topic"),
        F.lit("").alias("reviewer"),
        F.lit("").alias("review_comment"),
        F.lit(None).cast("string").alias("reviewed_at"),
        F.lit(source_table_key).alias("source_table_key"),
        F.coalesce(F.col("run_id"), F.lit(run_id)).alias("run_id"),
        F.coalesce(F.col("run_date"), F.lit(run_date)).alias("run_date"),
        "prompt_version",
        "taxonomy_version",
        "model_version",
        F.lit(pipeline_version).alias("pipeline_version"),
        F.lit(created_at).alias("created_at"),
        F.lit(created_by).alias("created_by"),
    )


def build_review_decision_df(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    existing_topic_reassignment_df: DataFrame | None = None,
    new_topic_candidate_df: DataFrame | None = None,
    source_table_key: str = "classification_full",
    decision_status: str = "pending",
    created_by: str = "codex",
) -> DataFrame:
    """Build one review-decision DataFrame from refinement candidate outputs."""
    frames: list[DataFrame] = []

    if existing_topic_reassignment_df is not None:
        frames.append(
            build_existing_reassignment_decision_df(
                existing_topic_reassignment_df,
                config,
                source_table_key=source_table_key,
                decision_status=decision_status,
                created_by=created_by,
            )
        )

    if new_topic_candidate_df is not None:
        frames.append(
            build_new_topic_decision_df(
                new_topic_candidate_df,
                config,
                source_table_key=source_table_key,
                decision_status=decision_status,
                created_by=created_by,
            )
        )

    if not frames:
        return _empty_review_decision_df(spark)

    result_df = frames[0]
    for frame in frames[1:]:
        result_df = result_df.unionByName(frame)

    return result_df.select(
        [field.name for field in REVIEW_DECISION_WRITE_SCHEMA]
    )


def _delete_existing_decision_rows(
    spark: SparkSession,
    table_name: str,
    decision_df: DataFrame,
) -> None:
    """Delete existing decision rows by stable decision_id before append."""
    if not spark.catalog.tableExists(table_name):
        return

    decision_df.select("decision_id").dropDuplicates().createOrReplaceTempView(
        "_tmp_review_decision_ids"
    )

    spark.sql(
        f"""
        DELETE FROM {table_name}
        WHERE EXISTS (
            SELECT 1
            FROM _tmp_review_decision_ids src
            WHERE {table_name}.decision_id = src.decision_id
        )
        """
    )


def save_review_decisions(
    spark: SparkSession,
    config: dict[str, Any],
    decision_df: DataFrame,
    *,
    write_mode: str = "replace_decisions",
) -> str:
    """Save review decision rows to the configured output table.

    Supported write modes:
    - replace_decisions: delete same decision_id rows, then append
    - append: append rows as-is
    - overwrite: overwrite entire table
    """
    table_name = get_output_table(config, "review_decision")
    write_df = decision_df.select(
        [
            F.col(field.name).cast(field.dataType).alias(field.name)
            for field in REVIEW_DECISION_WRITE_SCHEMA
        ]
    )

    if write_mode == "overwrite":
        write_df.write.mode("overwrite").format("delta").saveAsTable(table_name)
        return table_name

    if write_mode == "replace_decisions":
        _delete_existing_decision_rows(
            spark=spark,
            table_name=table_name,
            decision_df=write_df,
        )
        write_df.write.mode("append").format("delta").saveAsTable(table_name)
        return table_name

    if write_mode == "append":
        write_df.write.mode("append").format("delta").saveAsTable(table_name)
        return table_name

    raise ValueError(f"Unsupported write_mode: {write_mode}")


def update_review_decision_status(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    decision_id: str,
    decision_status: str,
    approved_action: str = "",
    approved_topic: str = "",
    reviewer: str = "",
    review_comment: str = "",
) -> None:
    """Update one decision row after human review."""
    if decision_status not in VALID_DECISION_STATUSES:
        raise ValueError(f"Unsupported decision_status: {decision_status}")
    if approved_action not in VALID_APPROVED_ACTIONS:
        raise ValueError(f"Unsupported approved_action: {approved_action}")

    table_name = get_output_table(config, "review_decision")
    reviewed_at = datetime.utcnow().isoformat(timespec="seconds")

    def escape(value: str) -> str:
        return str(value).replace("'", "''")

    spark.sql(
        f"""
        UPDATE {table_name}
        SET decision_status = '{escape(decision_status)}',
            approved_action = '{escape(approved_action)}',
            approved_topic = '{escape(approved_topic)}',
            reviewer = '{escape(reviewer)}',
            review_comment = '{escape(review_comment)}',
            reviewed_at = '{escape(reviewed_at)}'
        WHERE decision_id = '{escape(decision_id)}'
        """
    )
