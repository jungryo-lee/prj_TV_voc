"""Apply human fallback decisions for rows classified as others."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from common.config_loader import get_output_table
from taxonomy.classification_expander import CLASSIFICATION_FULL_WRITE_SCHEMA
from taxonomy.classification_writer import CLASSIFICATION_DETAIL_WRITE_SCHEMA
from taxonomy.review_decision_writer import REVIEW_DECISION_WRITE_SCHEMA


VALID_INPUT_TABLE_KEYS = {"classification_detail", "classification_full"}
VALID_MANUAL_ACTIONS = {"reassign_existing_topic", "keep_others"}


def _clean_text(value: Any) -> str:
    """Collapse whitespace for stable joins and table payloads."""
    return " ".join(str(value or "").split()).strip()


def _runtime_value(config: dict[str, Any], key: str, default: str = "") -> str:
    """Read resolved runtime metadata with a safe fallback."""
    return str(config.get("runtime", {}).get(key, default) or default)


def _version_value(config: dict[str, Any], key: str, default: str = "") -> str:
    """Read version metadata with a safe fallback."""
    return str(config.get("version", {}).get(key, default) or default)


def _target_schema(input_table_key: str) -> T.StructType:
    """Return the write schema for a classification output table."""
    if input_table_key == "classification_detail":
        return CLASSIFICATION_DETAIL_WRITE_SCHEMA
    if input_table_key == "classification_full":
        return CLASSIFICATION_FULL_WRITE_SCHEMA
    raise ValueError(f"Unsupported input_table_key: {input_table_key}")


def _load_classification_df(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    input_table_key: str,
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
    model_version: str | None = None,
    prompt_version: str | None = None,
    taxonomy_version: str | None = None,
) -> DataFrame:
    """Load classification rows for manual review or fallback application."""
    if input_table_key not in VALID_INPUT_TABLE_KEYS:
        raise ValueError(f"Unsupported input_table_key: {input_table_key}")

    table_name = get_output_table(config, input_table_key)
    df = spark.table(table_name)

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


def load_manual_fallback_candidates(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    input_table_key: str = "classification_full",
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
    model_version: str | None = None,
    prompt_version: str | None = None,
    taxonomy_version: str | None = None,
    max_rows: int = 200,
) -> DataFrame:
    """Load distinct others rows for human fallback review.

    The output is intentionally one row per memo_id so reviewers do not have to
    decide the same normalized memo repeatedly. Applying the decision later will
    still update all raw rows sharing the same memo_id in classification_full.
    """
    df = _load_classification_df(
        spark,
        config,
        input_table_key=input_table_key,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        model_version=model_version,
        prompt_version=prompt_version,
        taxonomy_version=taxonomy_version,
    ).where(F.col("pred_topic_type") == "others")

    latest_window = Window.partitionBy(
        "cate_1_depth",
        "cate_2_depth",
        "sc_measurement",
        "memo_id",
        "model_version",
        "prompt_version",
        "taxonomy_version",
    ).orderBy(
        F.col("created_at").desc_nulls_last(),
        F.col("run_id").desc_nulls_last(),
    )

    return (
        df.withColumn("_rn", F.row_number().over(latest_window))
        .where(F.col("_rn") == 1)
        .drop("_rn")
        .select(
            "cate_1_depth",
            "cate_2_depth",
            "sc_measurement",
            "memo_id",
            "memo_norm",
            F.col("memo").alias("sample_memo"),
            F.col("pred_topic").alias("current_pred_topic"),
            F.col("pred_topic_type").alias("current_pred_topic_type"),
            "match_reason",
            "model_version",
            "prompt_version",
            "taxonomy_version",
            "run_id",
            "run_date",
        )
        .orderBy(F.col("memo_norm").asc())
        .limit(int(max_rows))
    )


def build_manual_fallback_decision_df(
    reviewed_df: DataFrame,
    config: dict[str, Any],
    *,
    source_table_key: str = "classification_full",
    created_by: str = "manual_review",
) -> DataFrame:
    """Convert reviewer-edited rows into approved review decisions.

    Required reviewer columns:
    - approved_action: reassign_existing_topic or keep_others
    - approved_topic: existing topic name when approved_action is reassign_existing_topic
    Optional columns:
    - reviewer
    - review_comment
    """
    run_id = _runtime_value(config, "resolved_run_id")
    run_date = _runtime_value(config, "resolved_run_date")
    pipeline_version = _version_value(config, "pipeline_version")
    created_at = datetime.utcnow().isoformat(timespec="seconds")

    required_columns = {
        "cate_1_depth",
        "cate_2_depth",
        "sc_measurement",
        "memo_id",
        "memo_norm",
        "sample_memo",
        "current_pred_topic",
        "current_pred_topic_type",
        "approved_action",
        "approved_topic",
    }
    missing_columns = sorted(required_columns - set(reviewed_df.columns))
    if missing_columns:
        raise ValueError(f"reviewed_df is missing columns: {missing_columns}")

    df = reviewed_df
    if "reviewer" not in df.columns:
        df = df.withColumn("reviewer", F.lit(""))
    if "review_comment" not in df.columns:
        df = df.withColumn("review_comment", F.lit(""))
    if "model_version" not in df.columns:
        df = df.withColumn("model_version", F.lit(_version_value(config, "model_version")))
    if "prompt_version" not in df.columns:
        df = df.withColumn("prompt_version", F.lit(_version_value(config, "prompt_version")))
    if "taxonomy_version" not in df.columns:
        df = df.withColumn("taxonomy_version", F.lit(_version_value(config, "taxonomy_version")))
    if "run_id" not in df.columns:
        df = df.withColumn("run_id", F.lit(run_id))
    if "run_date" not in df.columns:
        df = df.withColumn("run_date", F.lit(run_date))

    df = df.where(F.col("approved_action").isin(sorted(VALID_MANUAL_ACTIONS)))
    invalid_reassign_df = df.where(
        (F.col("approved_action") == "reassign_existing_topic")
        & (F.length(F.trim(F.col("approved_topic"))) == 0)
    )
    if invalid_reassign_df.limit(1).count() > 0:
        raise ValueError("approved_topic is required for reassign_existing_topic.")

    decision_id = F.sha2(
        F.concat_ws(
            "||",
            F.lit("manual_fallback"),
            F.coalesce(F.col("cate_1_depth"), F.lit("")),
            F.coalesce(F.col("cate_2_depth"), F.lit("")),
            F.coalesce(F.col("sc_measurement").cast("string"), F.lit("")),
            F.coalesce(F.col("memo_id"), F.lit("")),
            F.coalesce(F.col("approved_action"), F.lit("")),
            F.coalesce(F.col("approved_topic"), F.lit("")),
            F.coalesce(F.col("model_version"), F.lit("")),
            F.coalesce(F.col("prompt_version"), F.lit("")),
            F.coalesce(F.col("taxonomy_version"), F.lit("")),
        ),
        256,
    )

    result_df = df.select(
        decision_id.alias("decision_id"),
        F.lit("manual_fallback").alias("candidate_type"),
        "cate_1_depth",
        "cate_2_depth",
        F.col("sc_measurement").cast("int").alias("sc_measurement"),
        "memo_id",
        "memo_norm",
        "sample_memo",
        "current_pred_topic",
        "current_pred_topic_type",
        F.when(
            F.col("approved_action") == "reassign_existing_topic",
            F.lit("reassign_existing_topic"),
        )
        .otherwise(F.lit("keep_others"))
        .alias("suggested_action"),
        F.when(
            F.col("approved_action") == "reassign_existing_topic",
            F.col("approved_topic"),
        )
        .otherwise(F.lit(""))
        .alias("suggested_topic"),
        F.lit(None).cast("double").alias("suggestion_score"),
        F.lit(None).cast("int").alias("candidate_cnt"),
        F.lit(None).cast("int").alias("candidate_distinct_memo_id_cnt"),
        F.lit(None).cast("double").alias("candidate_ratio"),
        F.to_json(
            F.struct(
                F.lit("manual_others_fallback").alias("source"),
                F.col("review_comment").alias("review_comment"),
            )
        ).alias("candidate_evidence_json"),
        F.lit("approved").alias("decision_status"),
        "approved_action",
        "approved_topic",
        "reviewer",
        "review_comment",
        F.lit(created_at).alias("reviewed_at"),
        F.lit(source_table_key).alias("source_table_key"),
        "run_id",
        "run_date",
        "prompt_version",
        "taxonomy_version",
        "model_version",
        F.lit(pipeline_version).alias("pipeline_version"),
        F.lit(created_at).alias("created_at"),
        F.lit(created_by).alias("created_by"),
    )

    return result_df.select(
        [F.col(field.name).cast(field.dataType).alias(field.name) for field in REVIEW_DECISION_WRITE_SCHEMA]
    )


def load_approved_manual_fallback_decisions(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    source_table_key: str = "classification_full",
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
    model_version: str | None = None,
    prompt_version: str | None = None,
    taxonomy_version: str | None = None,
) -> DataFrame:
    """Load approved manual fallback decisions from review_decision."""
    table_name = get_output_table(config, "review_decision")
    df = (
        spark.table(table_name)
        .where(F.col("decision_status") == "approved")
        .where(F.col("approved_action").isin(sorted(VALID_MANUAL_ACTIONS)))
        .where(F.col("source_table_key") == source_table_key)
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

    latest_window = Window.partitionBy(
        "cate_1_depth",
        "cate_2_depth",
        "sc_measurement",
        "memo_id",
        "source_table_key",
        "model_version",
        "prompt_version",
        "taxonomy_version",
    ).orderBy(
        F.col("reviewed_at").desc_nulls_last(),
        F.col("created_at").desc_nulls_last(),
    )

    return (
        df.withColumn("_rn", F.row_number().over(latest_window))
        .where(F.col("_rn") == 1)
        .drop("_rn")
    )


def apply_manual_fallback_decisions(
    classification_df: DataFrame,
    decision_df: DataFrame,
) -> DataFrame:
    """Apply approved manual fallback decisions to a classification DataFrame."""
    if decision_df.limit(1).count() == 0:
        return classification_df

    decision_cols = [
        F.col("cate_1_depth").alias("_d_cate_1_depth"),
        F.col("cate_2_depth").alias("_d_cate_2_depth"),
        F.col("sc_measurement").alias("_d_sc_measurement"),
        F.col("memo_id").alias("_d_memo_id"),
        F.col("approved_action").alias("_approved_action"),
        F.col("approved_topic").alias("_approved_topic"),
        F.col("decision_id").alias("_decision_id"),
        F.col("reviewer").alias("_reviewer"),
        F.col("review_comment").alias("_review_comment"),
    ]
    decisions = decision_df.select(*decision_cols).dropDuplicates(
        ["_d_cate_1_depth", "_d_cate_2_depth", "_d_sc_measurement", "_d_memo_id"]
    )

    joined = classification_df.join(
        decisions,
        on=(
            (classification_df.cate_1_depth == decisions._d_cate_1_depth)
            & (classification_df.cate_2_depth == decisions._d_cate_2_depth)
            & (classification_df.sc_measurement == decisions._d_sc_measurement)
            & (classification_df.memo_id == decisions._d_memo_id)
        ),
        how="left",
    )

    has_decision = F.col("_approved_action").isNotNull()
    is_reassign = F.col("_approved_action") == "reassign_existing_topic"
    is_keep_others = F.col("_approved_action") == "keep_others"
    manual_reason = F.concat_ws(
        " | ",
        F.lit("human_manual_fallback"),
        F.concat(F.lit("decision_id="), F.coalesce(F.col("_decision_id"), F.lit(""))),
        F.concat(F.lit("reviewer="), F.coalesce(F.col("_reviewer"), F.lit(""))),
        F.concat(F.lit("comment="), F.coalesce(F.col("_review_comment"), F.lit(""))),
    )

    return (
        joined.withColumn(
            "pred_topic",
            F.when(is_reassign, F.col("_approved_topic")).otherwise(F.col("pred_topic")),
        )
        .withColumn(
            "pred_topic_type",
            F.when(is_reassign, F.lit("topic"))
            .when(is_keep_others, F.lit("others"))
            .otherwise(F.col("pred_topic_type")),
        )
        .withColumn(
            "classification_stage",
            F.when(is_reassign, F.lit("human_fallback_reassign"))
            .when(is_keep_others, F.lit("human_fallback_keep_others"))
            .otherwise(F.col("classification_stage")),
        )
        .withColumn(
            "confidence_score",
            F.when(has_decision, F.lit(1.0)).otherwise(F.col("confidence_score")),
        )
        .withColumn(
            "review_needed_yn",
            F.when(has_decision, F.lit(False)).otherwise(F.col("review_needed_yn")),
        )
        .withColumn(
            "match_reason",
            F.when(has_decision, manual_reason).otherwise(F.col("match_reason")),
        )
        .drop(
            "_d_cate_1_depth",
            "_d_cate_2_depth",
            "_d_sc_measurement",
            "_d_memo_id",
            "_approved_action",
            "_approved_topic",
            "_decision_id",
            "_reviewer",
            "_review_comment",
        )
    )


def save_classification_with_manual_fallback(
    spark: SparkSession,
    config: dict[str, Any],
    fallback_df: DataFrame,
    *,
    input_table_key: str,
    write_mode: str = "replace_groups",
) -> str:
    """Save fallback-applied classification rows back to the configured table."""
    if input_table_key not in VALID_INPUT_TABLE_KEYS:
        raise ValueError(f"Unsupported input_table_key: {input_table_key}")

    table_name = get_output_table(config, input_table_key)
    schema = _target_schema(input_table_key)
    write_df = fallback_df.select(
        [F.col(field.name).cast(field.dataType).alias(field.name) for field in schema]
    ).cache()
    write_df.count()

    try:
        if write_mode == "overwrite":
            write_df.write.mode("overwrite").format("delta").saveAsTable(table_name)
            return table_name

        if write_mode == "append":
            write_df.write.mode("append").format("delta").saveAsTable(table_name)
            return table_name

        if write_mode == "replace_groups":
            if spark.catalog.tableExists(table_name):
                keys_df = write_df.select(
                    "cate_1_depth",
                    "cate_2_depth",
                    "sc_measurement",
                    "model_version",
                    "prompt_version",
                    "taxonomy_version",
                ).dropDuplicates()
                keys_df.createOrReplaceTempView("_tmp_manual_fallback_keys")

                spark.sql(
                    f"""
                    DELETE FROM {table_name}
                    WHERE EXISTS (
                        SELECT 1
                        FROM _tmp_manual_fallback_keys src
                        WHERE {table_name}.cate_1_depth = src.cate_1_depth
                          AND {table_name}.cate_2_depth = src.cate_2_depth
                          AND {table_name}.sc_measurement = src.sc_measurement
                          AND {table_name}.model_version = src.model_version
                          AND {table_name}.prompt_version = src.prompt_version
                          AND {table_name}.taxonomy_version = src.taxonomy_version
                    )
                    """
                )

            write_df.write.mode("append").format("delta").saveAsTable(table_name)
            return table_name

        raise ValueError(f"Unsupported write_mode: {write_mode}")
    finally:
        write_df.unpersist()


def apply_and_save_manual_fallback(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    input_table_key: str = "classification_full",
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
    model_version: str | None = None,
    prompt_version: str | None = None,
    taxonomy_version: str | None = None,
    write_mode: str = "replace_groups",
) -> dict[str, Any]:
    """Load, apply, save, and summarize approved manual fallback decisions."""
    classification_df = _load_classification_df(
        spark,
        config,
        input_table_key=input_table_key,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        model_version=model_version,
        prompt_version=prompt_version,
        taxonomy_version=taxonomy_version,
    )
    decision_df = load_approved_manual_fallback_decisions(
        spark,
        config,
        source_table_key=input_table_key,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        model_version=model_version,
        prompt_version=prompt_version,
        taxonomy_version=taxonomy_version,
    )
    fallback_df = apply_manual_fallback_decisions(classification_df, decision_df)
    input_row_count = int(classification_df.count())
    approved_decision_count = int(decision_df.count())
    output_row_count = int(fallback_df.count())
    reassigned_row_count = int(
        fallback_df.where(F.col("classification_stage") == "human_fallback_reassign").count()
    )
    kept_others_row_count = int(
        fallback_df.where(F.col("classification_stage") == "human_fallback_keep_others").count()
    )
    table_name = save_classification_with_manual_fallback(
        spark,
        config,
        fallback_df,
        input_table_key=input_table_key,
        write_mode=write_mode,
    )

    return {
        "table_name": table_name,
        "input_table_key": input_table_key,
        "input_row_count": input_row_count,
        "approved_decision_count": approved_decision_count,
        "output_row_count": output_row_count,
        "reassigned_row_count": reassigned_row_count,
        "kept_others_row_count": kept_others_row_count,
    }
