"""Build final Tableau-ready topic classification outputs.

This module consumes the stage-12 ML/GPT-mini classification result and creates:
- memo_id-level final detail table
- raw-review-row-level Tableau table with memo_id and final topic columns attached
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from common.category_alias import apply_category_aliases
from common.config_loader import get_output_table, get_source_filters, get_source_table
from common.memo_id import with_memo_id
from ml.topic_ml_classifier import ML_CLASSIFICATION_SCHEMA


GROUP_KEYS = ["cate_1_depth", "cate_2_depth", "sc_measurement"]
MEMO_KEYS = GROUP_KEYS + ["memo_id"]
FINAL_ACCEPT_STAGES = ["gpt_mini_fallback", "embedding_prototype_auto_accept"]
PENDING_FALLBACK_STAGE = "embedding_prototype_llm_fallback"
PENDING_FALLBACK_TOPIC = "LLM_FALLBACK_REQUIRED"
LOW_VOLUME_CLASSIFICATION_STAGE = "low_volume_group_rule"
LOW_VOLUME_TOPIC_TYPE = "unclassified"


def _version_value(config: dict[str, Any], key: str, default: str = "") -> str:
    """Return version metadata as string."""
    return str((config.get("version", {}) or {}).get(key, default) or default)


def _runtime_value(config: dict[str, Any], key: str, default: str = "") -> str:
    """Return runtime metadata as string."""
    return str((config.get("runtime", {}) or {}).get(key, default) or default)


def _label_model_version(config: dict[str, Any]) -> str:
    """Resolve the model version that produced supervised/sample labels."""
    model_key = (
        (config.get("final_classification", {}) or {}).get("label_model_key")
        or (config.get("ml_classification", {}) or {}).get("label_model_key")
        or (config.get("pipeline", {}) or {}).get("sample_classification_model_key")
        or (config.get("app", {}) or {}).get("model_key", "gpt_55")
    )
    return str(
        ((config.get("llm", {}) or {}).get("models", {}) or {})
        .get(model_key, {})
        .get("model_version", _version_value(config, "model_version"))
    )


def _sql_escape(value: str) -> str:
    """Escape a value for SQL literal usage."""
    return str(value).replace("'", "''")


def _target_sentiments(config: dict[str, Any]) -> list[int]:
    """Return sentiment values eligible for topic classification/final output."""
    taxonomy_cfg = config.get("taxonomy", {}) or {}
    ml_cfg = config.get("ml_classification", {}) or {}
    values = taxonomy_cfg.get("target_sentiments", ml_cfg.get("target_sentiments", [1, -1]))
    return [int(value) for value in values]


def _table_exists(spark: SparkSession, table_name: str) -> bool:
    """Return whether a Spark table exists."""
    try:
        return bool(spark.catalog.tableExists(table_name))
    except Exception:
        return False


def load_existing_final_keys(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    table_key: str = "classification_detail_final",
    table_name: str | None = None,
) -> DataFrame:
    """Load memo_ids already finalized for the active taxonomy version.

    This table is the pipeline's "do not classify again" registry. Once a
    memo_id is present here, later design/ML/fallback batches should skip it
    unless a separate human-review process intentionally changes the label.
    """
    resolved_table = table_name or get_output_table(config, table_key)
    if not _table_exists(spark, resolved_table):
        schema = "cate_1_depth string, cate_2_depth string, sc_measurement int, memo_id string"
        return spark.createDataFrame([], schema)

    return (
        spark.table(resolved_table)
        .where(F.col("prompt_version") == _version_value(config, "prompt_version"))
        .where(F.col("taxonomy_version") == _version_value(config, "taxonomy_version"))
        .select(
            F.col("cate_1_depth").cast("string"),
            F.col("cate_2_depth").cast("string"),
            F.col("sc_measurement").cast("int"),
            F.col("memo_id").cast("string"),
        )
        .dropDuplicates()
    )


def exclude_existing_final_memo_ids(
    df: DataFrame,
    config: dict[str, Any],
    *,
    table_key: str = "classification_detail_final",
) -> DataFrame:
    """Remove rows whose memo_id already has a final label."""
    existing_keys = load_existing_final_keys(
        df.sparkSession,
        config,
        table_key=table_key,
    )
    return df.join(existing_keys, on=MEMO_KEYS, how="left_anti")


def _delete_active_version_rows(
    spark: SparkSession,
    table_name: str,
    config: dict[str, Any],
) -> None:
    """Delete rows for the active prompt/taxonomy version before re-saving."""
    if not _table_exists(spark, table_name):
        return

    prompt_version = _sql_escape(_version_value(config, "prompt_version"))
    taxonomy_version = _sql_escape(_version_value(config, "taxonomy_version"))
    spark.sql(
        f"""
        DELETE FROM {table_name}
        WHERE prompt_version = '{prompt_version}'
          AND taxonomy_version = '{taxonomy_version}'
        """
    )


def load_ml_classification_df(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    input_table_key: str = "ml_classification_detail",
) -> DataFrame:
    """Load active-version stage-12 classification rows."""
    table_name = get_output_table(config, input_table_key)
    return (
        spark.table(table_name)
        .where(F.col("prompt_version") == _version_value(config, "prompt_version"))
        .where(F.col("taxonomy_version") == _version_value(config, "taxonomy_version"))
    )


def load_sample_classification_final_candidate_df(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    input_table_key: str = "classification_detail",
    created_by: str = "final_classification_builder",
) -> DataFrame:
    """Load sample LLM labels as final candidates.

    These rows are high-cost LLM labels already produced during taxonomy design.
    If a memo_id has no later ML/GPT-mini fallback result, preserving the sample
    label in final prevents unnecessary reclassification.
    """
    table_name = get_output_table(config, input_table_key)
    if not _table_exists(spark, table_name):
        return spark.createDataFrame([], ML_CLASSIFICATION_SCHEMA)

    created_at = datetime.utcnow().isoformat(timespec="seconds")
    run_id = _runtime_value(config, "resolved_run_id")
    run_date = _runtime_value(config, "resolved_run_date")
    pipeline_version = _version_value(config, "pipeline_version")

    base_df = (
        spark.table(table_name)
        .where(F.col("prompt_version") == _version_value(config, "prompt_version"))
        .where(F.col("taxonomy_version") == _version_value(config, "taxonomy_version"))
        .where(F.col("model_version") == _label_model_version(config))
        .where(F.col("memo_id").isNotNull())
        .where(F.col("pred_topic").isNotNull())
        .where(F.col("pred_topic") != PENDING_FALLBACK_TOPIC)
    )
    if "is_latest" in base_df.columns:
        base_df = base_df.where(F.coalesce(F.col("is_latest"), F.lit(True)) == F.lit(True))

    return base_df.select(
        F.col("memo_id").cast("string"),
        F.col("memo").cast("string"),
        F.col("memo_norm").cast("string"),
        F.col("cate_1_depth").cast("string"),
        F.col("cate_2_depth").cast("string"),
        F.col("sc_measurement").cast("int"),
        F.col("pred_topic").cast("string"),
        F.col("pred_topic_type").cast("string"),
        F.col("classification_stage").cast("string"),
        F.col("confidence_score").cast("double"),
        F.coalesce(F.col("review_needed_yn"), F.lit(False)).cast("boolean").alias("review_needed_yn"),
        F.coalesce(F.col("llm_used_yn"), F.lit(True)).cast("boolean").alias("llm_used_yn"),
        F.col("match_reason").cast("string"),
        F.col("candidate_topics_json").cast("string"),
        F.lit(None).cast("string").alias("fallback_model_key"),
        F.lit(None).cast("string").alias("fallback_model_version"),
        F.coalesce(F.col("run_id"), F.lit(run_id)).cast("string").alias("run_id"),
        F.coalesce(F.col("run_date"), F.lit(run_date)).cast("string").alias("run_date"),
        F.col("prompt_version").cast("string"),
        F.col("taxonomy_version").cast("string"),
        F.col("model_version").cast("string"),
        F.lit(pipeline_version).cast("string").alias("pipeline_version"),
        F.lit(created_at).cast("string").alias("created_at"),
        F.lit(created_by).cast("string").alias("created_by"),
    )


def summarize_ml_classification_result(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    input_table_key: str = "ml_classification_detail",
) -> dict[str, Any]:
    """Return stage-12 completion summary before finalization."""
    df = load_ml_classification_df(spark, config, input_table_key=input_table_key)
    accepted_df = df.where(F.col("classification_stage").isin(FINAL_ACCEPT_STAGES))
    pending_df = df.where(
        (F.col("classification_stage") == PENDING_FALLBACK_STAGE)
        & (F.col("pred_topic") == PENDING_FALLBACK_TOPIC)
    )
    fallback_done_keys = (
        df.where(F.col("classification_stage") == "gpt_mini_fallback")
        .select(*MEMO_KEYS)
        .dropDuplicates()
    )
    unresolved_pending_df = pending_df.join(
        fallback_done_keys,
        on=MEMO_KEYS,
        how="left_anti",
    )

    return {
        "ml_total_rows": int(df.count()),
        "accepted_final_candidate_rows": int(accepted_df.count()),
        "accepted_final_distinct_memo_ids": int(
            accepted_df.select(*MEMO_KEYS).dropDuplicates().count()
        ),
        "pending_fallback_rows": int(pending_df.count()),
        "unresolved_pending_fallback_rows": int(unresolved_pending_df.count()),
        "gpt_mini_fallback_rows": int(
            df.where(F.col("classification_stage") == "gpt_mini_fallback").count()
        ),
    }


def build_final_classification_detail_df(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    input_table_key: str = "ml_classification_detail",
    created_by: str = "final_classification_builder",
    exclude_existing_final: bool = True,
    include_sample_classification: bool | None = None,
) -> DataFrame:
    """Pick one final classification row per group/memo_id.

    GPT-mini fallback decisions outrank prototype auto-accepted decisions when both
    exist for the same memo_id. Pending fallback placeholders are intentionally
    excluded from the final output.
    """
    df = load_ml_classification_df(spark, config, input_table_key=input_table_key)
    final_candidates = df.where(F.col("classification_stage").isin(FINAL_ACCEPT_STAGES))

    priority_col = (
        F.when(F.col("classification_stage") == "gpt_mini_fallback", F.lit(1))
        .when(F.col("classification_stage") == "embedding_prototype_auto_accept", F.lit(2))
        .otherwise(F.lit(99))
    )
    latest_window = Window.partitionBy(*MEMO_KEYS).orderBy(
        priority_col.asc(),
        F.col("created_at").desc_nulls_last(),
        F.col("run_id").desc_nulls_last(),
    )

    created_at = datetime.utcnow().isoformat(timespec="seconds")
    run_id = _runtime_value(config, "resolved_run_id")
    run_date = _runtime_value(config, "resolved_run_date")
    pipeline_version = _version_value(config, "pipeline_version")

    selected_df = (
        final_candidates.withColumn("_final_rn", F.row_number().over(latest_window))
        .where(F.col("_final_rn") == 1)
        .drop("_final_rn")
    )

    selected_final_df = selected_df.select(
        F.col("memo_id").cast("string"),
        F.col("memo").cast("string"),
        F.col("memo_norm").cast("string"),
        F.col("cate_1_depth").cast("string"),
        F.col("cate_2_depth").cast("string"),
        F.col("sc_measurement").cast("int"),
        F.col("pred_topic").cast("string"),
        F.col("pred_topic_type").cast("string"),
        F.col("classification_stage").cast("string"),
        F.col("confidence_score").cast("double"),
        F.lit(False).cast("boolean").alias("review_needed_yn"),
        F.col("llm_used_yn").cast("boolean"),
        F.col("match_reason").cast("string"),
        F.col("candidate_topics_json").cast("string"),
        F.col("fallback_model_key").cast("string"),
        F.col("fallback_model_version").cast("string"),
        F.lit(run_id).cast("string").alias("run_id"),
        F.lit(run_date).cast("string").alias("run_date"),
        F.col("prompt_version").cast("string"),
        F.col("taxonomy_version").cast("string"),
        F.col("model_version").cast("string"),
        F.lit(pipeline_version).cast("string").alias("pipeline_version"),
        F.lit(created_at).cast("string").alias("created_at"),
        F.lit(created_by).cast("string").alias("created_by"),
    )

    final_cfg = config.get("final_classification", {}) or {}
    resolved_include_sample = (
        bool(final_cfg.get("include_sample_classification", True))
        if include_sample_classification is None
        else bool(include_sample_classification)
    )
    if resolved_include_sample:
        sample_final_df = load_sample_classification_final_candidate_df(
            spark,
            config,
            created_by=created_by,
        )
        sample_only_df = sample_final_df.join(
            selected_final_df.select(*MEMO_KEYS).dropDuplicates(),
            on=MEMO_KEYS,
            how="left_anti",
        )
        selected_final_df = selected_final_df.unionByName(sample_only_df)

    low_volume_df = build_low_volume_unclassified_detail_df(
        spark,
        config,
        created_by=created_by,
    )

    low_volume_only_df = low_volume_df.join(
        selected_final_df.select(*MEMO_KEYS).dropDuplicates(),
        on=MEMO_KEYS,
        how="left_anti",
    )

    output_df = selected_final_df.unionByName(low_volume_only_df)
    if exclude_existing_final:
        output_df = exclude_existing_final_memo_ids(output_df, config)

    return output_df


def build_low_volume_unclassified_detail_df(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    created_by: str = "final_classification_builder",
) -> DataFrame:
    """Create deterministic final rows for groups below the raw-row threshold."""
    final_cfg = config.get("final_classification", {}) or {}
    max_raw_rows = int(final_cfg.get("low_volume_group_max_raw_rows", 100))
    topic_name = str(
        final_cfg.get("low_volume_unclassified_topic", "미분류(리뷰 100개미만)")
    )

    raw_table = get_source_table(config, "raw_review_table")
    raw_df = (
        spark.table(raw_table)
        .where(F.col("memo").isNotNull())
        .where(F.length(F.trim(F.col("memo").cast("string"))) > 0)
        .where(_build_source_filter_condition(config))
        .withColumn("sc_measurement", F.col("sc_measurement").cast("int"))
        .where(F.col("sc_measurement").isNotNull())
    )
    raw_df = apply_category_aliases(raw_df, config)

    target_sentiments = _target_sentiments(config)
    if target_sentiments:
        raw_df = raw_df.where(F.col("sc_measurement").isin(target_sentiments))

    raw_count_df = (
        raw_df.groupBy(*GROUP_KEYS)
        .agg(F.count("*").cast("long").alias("raw_group_rows"))
        .where(F.col("raw_group_rows") < F.lit(max_raw_rows))
    )

    raw_with_id_df = with_memo_id(raw_df)
    low_volume_raw_df = raw_with_id_df.join(raw_count_df, on=GROUP_KEYS, how="inner")

    dedupe_window = Window.partitionBy(*MEMO_KEYS).orderBy(
        F.length(F.col("memo").cast("string")).desc_nulls_last(),
        F.col("memo").asc_nulls_last(),
    )

    created_at = datetime.utcnow().isoformat(timespec="seconds")
    run_id = _runtime_value(config, "resolved_run_id")
    run_date = _runtime_value(config, "resolved_run_date")
    prompt_version = _version_value(config, "prompt_version")
    taxonomy_version = _version_value(config, "taxonomy_version")
    model_version = "rule_low_volume"
    pipeline_version = _version_value(config, "pipeline_version")

    return (
        low_volume_raw_df.withColumn("_rn", F.row_number().over(dedupe_window))
        .where(F.col("_rn") == 1)
        .select(
            F.col("memo_id").cast("string"),
            F.col("memo").cast("string"),
            F.col("memo_norm").cast("string"),
            F.col("cate_1_depth").cast("string"),
            F.col("cate_2_depth").cast("string"),
            F.col("sc_measurement").cast("int"),
            F.lit(topic_name).cast("string").alias("pred_topic"),
            F.lit(LOW_VOLUME_TOPIC_TYPE).cast("string").alias("pred_topic_type"),
            F.lit(LOW_VOLUME_CLASSIFICATION_STAGE).cast("string").alias(
                "classification_stage"
            ),
            F.lit(1.0).cast("double").alias("confidence_score"),
            F.lit(False).cast("boolean").alias("review_needed_yn"),
            F.lit(False).cast("boolean").alias("llm_used_yn"),
            F.concat(
                F.lit("raw group rows < "),
                F.lit(str(max_raw_rows)),
                F.lit("; assigned deterministic low-volume unclassified label"),
            ).cast("string").alias("match_reason"),
            F.lit("[]").cast("string").alias("candidate_topics_json"),
            F.lit(None).cast("string").alias("fallback_model_key"),
            F.lit(None).cast("string").alias("fallback_model_version"),
            F.lit(run_id).cast("string").alias("run_id"),
            F.lit(run_date).cast("string").alias("run_date"),
            F.lit(prompt_version).cast("string").alias("prompt_version"),
            F.lit(taxonomy_version).cast("string").alias("taxonomy_version"),
            F.lit(model_version).cast("string").alias("model_version"),
            F.lit(pipeline_version).cast("string").alias("pipeline_version"),
            F.lit(created_at).cast("string").alias("created_at"),
            F.lit(created_by).cast("string").alias("created_by"),
        )
    )


def save_final_classification_detail(
    spark: SparkSession,
    config: dict[str, Any],
    final_detail_df: DataFrame,
    *,
    output_table_key: str = "classification_detail_final",
    write_mode: str = "append_new_only",
) -> str:
    """Save memo_id-level final classification detail."""
    table_name = get_output_table(config, output_table_key)
    write_df = final_detail_df.select(
        [F.col(field.name).cast(field.dataType).alias(field.name) for field in ML_CLASSIFICATION_SCHEMA.fields]
    )

    if write_mode == "overwrite":
        write_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
        return table_name
    if write_mode == "replace_version":
        _delete_active_version_rows(spark, table_name, config)
        write_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table_name)
        return table_name
    if write_mode == "append_new_only":
        if _table_exists(spark, table_name):
            write_df = exclude_existing_final_memo_ids(
                write_df,
                config,
                table_key=output_table_key,
            )
        if write_df.limit(1).count() == 0:
            return table_name
        write_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table_name)
        return table_name
    if write_mode == "append":
        write_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table_name)
        return table_name

    raise ValueError(f"Unsupported write_mode: {write_mode}")


def _build_source_filter_condition(config: dict[str, Any]) -> F.Column:
    """Build a Spark SQL condition from configured source filters."""
    filters = get_source_filters(config, "raw_review_table")
    condition = F.lit(True)
    for item in filters:
        condition = condition & F.expr(item)
    return condition


def build_tableau_final_df(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    final_detail_table_key: str = "classification_detail_final",
) -> DataFrame:
    """Join final memo classifications back to the original Intellytics rows."""
    raw_table = get_source_table(config, "raw_review_table")
    final_detail_table = get_output_table(config, final_detail_table_key)

    raw_df = (
        spark.table(raw_table)
        .where(F.col("memo").isNotNull())
        .where(F.length(F.trim(F.col("memo").cast("string"))) > 0)
        .where(_build_source_filter_condition(config))
    )
    raw_df = apply_category_aliases(raw_df, config)
    raw_with_id_df = with_memo_id(raw_df)

    final_detail_df = (
        spark.table(final_detail_table)
        .where(F.col("prompt_version") == _version_value(config, "prompt_version"))
        .where(F.col("taxonomy_version") == _version_value(config, "taxonomy_version"))
        .select(
            *MEMO_KEYS,
            F.col("pred_topic").alias("final_pred_topic"),
            F.col("pred_topic_type").alias("final_pred_topic_type"),
            F.col("classification_stage").alias("final_classification_stage"),
            F.col("confidence_score").alias("final_confidence_score"),
            F.col("llm_used_yn").alias("final_llm_used_yn"),
            F.col("match_reason").alias("final_match_reason"),
            F.col("candidate_topics_json").alias("final_candidate_topics_json"),
            F.col("run_id").alias("final_run_id"),
            F.col("run_date").alias("final_run_date"),
            "prompt_version",
            "taxonomy_version",
            "model_version",
            "pipeline_version",
            F.col("created_at").alias("final_created_at"),
        )
    )

    return (
        raw_with_id_df.alias("raw")
        .join(final_detail_df.alias("cls"), on=MEMO_KEYS, how="inner")
        .select(
            "raw.*",
            F.col("cls.final_pred_topic").alias("pred_topic"),
            F.col("cls.final_pred_topic_type").alias("pred_topic_type"),
            F.col("cls.final_classification_stage").alias("classification_stage"),
            F.col("cls.final_confidence_score").alias("confidence_score"),
            F.col("cls.final_llm_used_yn").alias("llm_used_yn"),
            F.col("cls.final_match_reason").alias("match_reason"),
            F.col("cls.final_candidate_topics_json").alias("candidate_topics_json"),
            F.col("cls.final_run_id").alias("classification_run_id"),
            F.col("cls.final_run_date").alias("classification_run_date"),
            F.col("cls.prompt_version"),
            F.col("cls.taxonomy_version"),
            F.col("cls.model_version"),
            F.col("cls.pipeline_version"),
            F.col("cls.final_created_at").alias("classification_created_at"),
        )
    )


def save_tableau_final(
    spark: SparkSession,
    config: dict[str, Any],
    tableau_df: DataFrame,
    *,
    output_table_key: str = "classification_tableau_final",
    write_mode: str = "replace_version",
) -> str:
    """Save the raw-row-level Tableau final table."""
    table_name = get_output_table(config, output_table_key)

    if write_mode == "overwrite":
        tableau_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
        return table_name
    if write_mode == "replace_version":
        _delete_active_version_rows(spark, table_name, config)
        tableau_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table_name)
        return table_name
    if write_mode == "append":
        tableau_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table_name)
        return table_name

    raise ValueError(f"Unsupported write_mode: {write_mode}")


def build_and_save_final_classification_outputs(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    input_table_key: str = "ml_classification_detail",
    final_detail_table_key: str = "classification_detail_final",
    tableau_table_key: str = "classification_tableau_final",
    write_mode: str = "append_new_only",
) -> dict[str, Any]:
    """Build and save both final memo-level and Tableau-ready raw-row outputs."""
    print("[final_classification] validating stage-12 result")
    before_summary = summarize_ml_classification_result(
        spark,
        config,
        input_table_key=input_table_key,
    )
    print("[final_classification] stage-12 summary =", before_summary)

    final_detail_df = build_final_classification_detail_df(
        spark,
        config,
        input_table_key=input_table_key,
    )
    final_detail_count = final_detail_df.count()
    print(f"[final_classification] final detail rows={final_detail_count}")

    final_detail_table = save_final_classification_detail(
        spark,
        config,
        final_detail_df,
        output_table_key=final_detail_table_key,
        write_mode=write_mode,
    )
    print(f"[final_classification] saved final detail table={final_detail_table}")

    tableau_df = build_tableau_final_df(
        spark,
        config,
        final_detail_table_key=final_detail_table_key,
    )
    tableau_count = tableau_df.count()
    tableau_distinct_memo_count = tableau_df.select("memo_id").dropDuplicates().count()
    print(
        "[final_classification] tableau rows="
        f"{tableau_count} distinct_memo_ids={tableau_distinct_memo_count}"
    )

    tableau_table = save_tableau_final(
        spark,
        config,
        tableau_df,
        output_table_key=tableau_table_key,
        write_mode=write_mode,
    )
    print(f"[final_classification] saved tableau table={tableau_table}")

    return {
        **before_summary,
        "final_detail_table": final_detail_table,
        "final_detail_rows": int(final_detail_count),
        "tableau_table": tableau_table,
        "tableau_rows": int(tableau_count),
        "tableau_distinct_memo_ids": int(tableau_distinct_memo_count),
    }
