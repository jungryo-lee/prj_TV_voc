"""Run taxonomy design + classification batch with restart-safe checkpoints."""

from __future__ import annotations

import gc
import json
from datetime import datetime
from typing import Any

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from common.config_loader import get_log_table, get_output_table, load_config
from pipeline.run_taxonomy_design_refresh import load_design_target_groups
from taxonomy.classification_writer import save_classification_details
from taxonomy.rule_profile_generator import generate_rule_profile_for_group
from taxonomy.rule_profile_writer import save_rule_profiles
from taxonomy.topic_classifier import classify_topic_for_group
from taxonomy.topic_pool_generator import generate_topic_pool_for_group
from taxonomy.topic_pool_writer import save_topic_pools


PIPELINE_NAME = "taxonomy_classification_batch"

PROGRESS_SCHEMA = T.StructType(
    [
        T.StructField("checkpoint_key", T.StringType(), False),
        T.StructField("pipeline_name", T.StringType(), False),
        T.StructField("run_id", T.StringType(), True),
        T.StructField("model_key", T.StringType(), True),
        T.StructField("cate_1_depth", T.StringType(), True),
        T.StructField("cate_2_depth", T.StringType(), True),
        T.StructField("sc_measurement", T.IntegerType(), True),
        T.StructField("status", T.StringType(), True),
        T.StructField("step", T.StringType(), True),
        T.StructField("message", T.StringType(), True),
        T.StructField("row_count", T.IntegerType(), True),
        T.StructField("overall_count", T.IntegerType(), True),
        T.StructField("topic_count", T.IntegerType(), True),
        T.StructField("others_count", T.IntegerType(), True),
        T.StructField("ambiguous_count", T.IntegerType(), True),
        T.StructField("llm_used_count", T.IntegerType(), True),
        T.StructField("created_at", T.StringType(), True),
    ]
)


def _now_text() -> str:
    """Return a compact UTC timestamp string."""
    return datetime.utcnow().isoformat(timespec="seconds")


def _checkpoint_key(
    *,
    model_key: str,
    cate_1_depth: str | None,
    cate_2_depth: str | None,
    sc_measurement: int | None,
    limit_group_count: int | None,
) -> str:
    """Build a stable checkpoint key for restart-safe reruns."""
    return "||".join(
        [
            PIPELINE_NAME,
            model_key,
            cate_1_depth or "*",
            cate_2_depth or "*",
            str(sc_measurement) if sc_measurement is not None else "*",
            str(limit_group_count) if limit_group_count is not None else "*",
        ]
    )


def _group_signature(
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
) -> str:
    """Return a stable group signature."""
    return f"{cate_1_depth}||{cate_2_depth}||{int(sc_measurement)}"


def _summarize_classification_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return a compact per-group execution summary."""
    row_count = int(result.get("row_count", 0))
    overall_count = int(result.get("overall_count", 0))
    topic_count = int(result.get("topic_count", 0))
    others_count = int(result.get("others_count", 0))
    ambiguous_count = int(result.get("ambiguous_count", 0))
    llm_used_count = int(result.get("llm_used_count", 0))

    if row_count > 0:
        overall_ratio = round(overall_count / row_count, 4)
        others_ratio = round(others_count / row_count, 4)
        llm_used_ratio = round(llm_used_count / row_count, 4)
    else:
        overall_ratio = 0.0
        others_ratio = 0.0
        llm_used_ratio = 0.0

    return {
        "cate_1_depth": result.get("cate_1_depth"),
        "cate_2_depth": result.get("cate_2_depth"),
        "sc_measurement": int(result.get("sc_measurement", 0)),
        "row_count": row_count,
        "overall_count": overall_count,
        "topic_count": topic_count,
        "others_count": others_count,
        "ambiguous_count": ambiguous_count,
        "llm_used_count": llm_used_count,
        "overall_ratio": overall_ratio,
        "others_ratio": others_ratio,
        "llm_used_ratio": llm_used_ratio,
    }


def _clean_text(value: Any) -> str:
    """Collapse whitespace for stable comparison and payload reuse."""
    return " ".join(str(value or "").split()).strip()


def _version_value(config: dict[str, Any], key: str, default: str = "") -> str:
    """Read version metadata with a safe fallback."""
    return str(config.get("version", {}).get(key, default) or default)


def _classification_model_version(config: dict[str, Any], model_key: str) -> str:
    """Resolve the model_version stored in classification_detail rows."""
    return str(
        config.get("llm", {})
        .get("models", {})
        .get(model_key, {})
        .get("model_version")
        or _version_value(config, "model_version")
        or model_key
    )


def _json_list(value: Any) -> list[Any]:
    """Parse a JSON-list field from a Delta row."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except Exception:
            return [raw]
        return parsed if isinstance(parsed, list) else [parsed]
    return [value]


def _table_exists(spark: SparkSession, table_name: str) -> bool:
    """Return whether a table exists, swallowing catalog lookup edge cases."""
    try:
        return bool(spark.catalog.tableExists(table_name))
    except Exception:
        return False


def _group_version_filter(
    df,
    *,
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    model_version: str,
    prompt_version: str,
    taxonomy_version: str,
):
    """Apply common group/version filters to a Spark DataFrame."""
    return (
        df.where(F.col("cate_1_depth") == cate_1_depth)
        .where(F.col("cate_2_depth") == cate_2_depth)
        .where(F.col("sc_measurement") == int(sc_measurement))
        .where(F.col("model_version") == model_version)
        .where(F.col("prompt_version") == prompt_version)
        .where(F.col("taxonomy_version") == taxonomy_version)
    )


def load_existing_rule_profile_result(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    model_key: str,
) -> dict[str, Any] | None:
    """Load a previously saved rule-profile result for a group/version."""
    table_name = get_output_table(config, "rule_profile")
    if not _table_exists(spark, table_name):
        return None

    prompt_version = _version_value(config, "prompt_version")
    taxonomy_version = _version_value(config, "taxonomy_version")
    rows = (
        _group_version_filter(
            spark.table(table_name),
            cate_1_depth=cate_1_depth,
            cate_2_depth=cate_2_depth,
            sc_measurement=sc_measurement,
            model_version=model_key,
            prompt_version=prompt_version,
            taxonomy_version=taxonomy_version,
        )
        .orderBy(
            F.col("is_latest").desc_nulls_last(),
            F.col("created_at").desc_nulls_last(),
            F.col("run_id").desc_nulls_last(),
        )
        .limit(1)
        .collect()
    )
    if not rows:
        return None

    row = rows[0].asDict(recursive=True)
    return {
        "cate_1_depth": _clean_text(row.get("cate_1_depth")),
        "cate_2_depth": _clean_text(row.get("cate_2_depth")),
        "sc_measurement": int(row.get("sc_measurement")),
        "sample_memo_count": int(row.get("sample_memo_count") or 0),
        "overall_topic_name": _clean_text(row.get("overall_topic_name")),
        "overall_allowed_rule": _clean_text(row.get("overall_allowed_rule")),
        "overall_block_rule": _clean_text(row.get("overall_block_rule")),
        "overall_sentiment_terms": _json_list(row.get("overall_sentiment_terms_json")),
        "feature_hint_terms": _json_list(row.get("feature_hint_terms_json")),
        "reason_signal_terms": _json_list(row.get("reason_signal_terms_json")),
        "non_overall_examples": _json_list(row.get("non_overall_examples_json")),
        "category_seed_used": {
            "category_summary": _clean_text(row.get("category_seed_summary")),
            "has_static_seed": bool(row.get("category_seed_has_static_seed")),
            "static_feature_hint_terms": _json_list(
                row.get("category_seed_static_feature_hint_terms_json")
            ),
            "feature_hint_terms": _json_list(row.get("category_seed_feature_hint_terms_json")),
            "reason_signal_terms": _json_list(row.get("category_seed_reason_signal_terms_json")),
            "overall_sentiment_terms": _json_list(
                row.get("category_seed_overall_sentiment_terms_json")
            ),
            "candidate_topic_labels": _json_list(
                row.get("category_seed_candidate_topic_labels_json")
            ),
            "sample_non_overall_memos": _json_list(
                row.get("category_seed_sample_non_overall_memos_json")
            ),
        },
    }


def load_existing_topic_pool_result(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    model_key: str,
) -> dict[str, Any] | None:
    """Load a previously saved topic-pool result for a group/version."""
    table_name = get_output_table(config, "topic_pool")
    if not _table_exists(spark, table_name):
        return None

    prompt_version = _version_value(config, "prompt_version")
    taxonomy_version = _version_value(config, "taxonomy_version")
    rows = (
        _group_version_filter(
            spark.table(table_name),
            cate_1_depth=cate_1_depth,
            cate_2_depth=cate_2_depth,
            sc_measurement=sc_measurement,
            model_version=model_key,
            prompt_version=prompt_version,
            taxonomy_version=taxonomy_version,
        )
        .orderBy(F.col("topic_order").asc())
        .collect()
    )
    if not rows:
        return None

    row_dicts = [row.asDict(recursive=True) for row in rows]
    first = row_dicts[0]
    topics = [
        {
            "topic": _clean_text(row.get("topic")),
            "description": _clean_text(row.get("description")),
            "representative_memos": _json_list(row.get("representative_memos_json")),
        }
        for row in row_dicts
        if _clean_text(row.get("topic"))
    ]
    return {
        "cate_1_depth": _clean_text(first.get("cate_1_depth")),
        "cate_2_depth": _clean_text(first.get("cate_2_depth")),
        "sc_measurement": int(first.get("sc_measurement")),
        "sample_memo_count": int(first.get("sample_memo_count") or 0),
        "original_sample_memo_count": int(first.get("original_sample_memo_count") or 0),
        "min_final_topics": int(first.get("min_final_topics") or 0),
        "max_final_topics": int(first.get("max_final_topics") or 0),
        "topic_count": int(first.get("topic_count") or len(topics)),
        "meets_min_topic_count": bool(first.get("meets_min_topic_count")),
        "meets_max_topic_count": bool(first.get("meets_max_topic_count")),
        "model_key": model_key,
        "rule_profile_used": {
            "overall_topic_name": _clean_text(first.get("overall_topic_name")),
            "overall_allowed_rule": _clean_text(
                first.get("rule_profile_overall_allowed_rule")
            ),
            "overall_block_rule": _clean_text(first.get("rule_profile_overall_block_rule")),
            "feature_hint_terms": _json_list(
                first.get("rule_profile_feature_hint_terms_json")
            ),
            "reason_signal_terms": _json_list(
                first.get("rule_profile_reason_signal_terms_json")
            ),
            "overall_sentiment_terms": _json_list(
                first.get("rule_profile_overall_sentiment_terms_json")
            ),
            "non_overall_examples": _json_list(
                first.get("rule_profile_non_overall_examples_json")
            ),
        },
        "topics": topics,
    }


def load_existing_classification_summary(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    model_key: str,
) -> dict[str, Any] | None:
    """Return a summary when classification_detail already exists for the group."""
    table_name = get_output_table(config, "classification_detail")
    if not _table_exists(spark, table_name):
        return None

    prompt_version = _version_value(config, "prompt_version")
    taxonomy_version = _version_value(config, "taxonomy_version")
    model_version = _classification_model_version(config, model_key)
    df = _group_version_filter(
        spark.table(table_name),
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        model_version=model_version,
        prompt_version=prompt_version,
        taxonomy_version=taxonomy_version,
    )
    if df.limit(1).count() == 0:
        return None

    summary_row = df.agg(
        F.count("*").alias("row_count"),
        F.sum(F.when(F.col("pred_topic_type") == "overall", 1).otherwise(0)).alias(
            "overall_count"
        ),
        F.sum(F.when(F.col("pred_topic_type") == "topic", 1).otherwise(0)).alias(
            "topic_count"
        ),
        F.sum(F.when(F.col("pred_topic_type") == "others", 1).otherwise(0)).alias(
            "others_count"
        ),
        F.sum(
            F.when(F.col("classification_stage") == "ambiguous_topic", 1).otherwise(0)
        ).alias("ambiguous_count"),
        F.sum(F.when(F.col("llm_used_yn") == True, 1).otherwise(0)).alias(
            "llm_used_count"
        ),
    ).collect()[0].asDict(recursive=True)
    return _summarize_classification_result(
        {
            "cate_1_depth": cate_1_depth,
            "cate_2_depth": cate_2_depth,
            "sc_measurement": int(sc_measurement),
            **summary_row,
        }
    )


def _append_progress_row(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    checkpoint_key: str,
    run_id: str,
    model_key: str,
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    status: str,
    step: str,
    message: str = "",
    summary: dict[str, Any] | None = None,
) -> None:
    """Append one progress/checkpoint row to the configured progress table."""
    table_name = get_log_table(config, "pipeline_progress")
    summary = summary or {}

    row = {
        "checkpoint_key": checkpoint_key,
        "pipeline_name": PIPELINE_NAME,
        "run_id": run_id,
        "model_key": model_key,
        "cate_1_depth": cate_1_depth,
        "cate_2_depth": cate_2_depth,
        "sc_measurement": int(sc_measurement),
        "status": status,
        "step": step,
        "message": message,
        "row_count": int(summary.get("row_count", 0)),
        "overall_count": int(summary.get("overall_count", 0)),
        "topic_count": int(summary.get("topic_count", 0)),
        "others_count": int(summary.get("others_count", 0)),
        "ambiguous_count": int(summary.get("ambiguous_count", 0)),
        "llm_used_count": int(summary.get("llm_used_count", 0)),
        "created_at": _now_text(),
    }

    spark.createDataFrame([row], schema=PROGRESS_SCHEMA).write.mode("append").format(
        "delta"
    ).saveAsTable(table_name)


def _load_completed_signatures(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    checkpoint_key: str,
) -> set[str]:
    """Load completed group signatures for the checkpoint key."""
    table_name = get_log_table(config, "pipeline_progress")
    if not spark.catalog.tableExists(table_name):
        return set()

    rows = (
        spark.table(table_name)
        .where(F.col("checkpoint_key") == checkpoint_key)
        .where(F.col("pipeline_name") == PIPELINE_NAME)
        .where(F.col("status") == "completed")
        .select("cate_1_depth", "cate_2_depth", "sc_measurement")
        .dropDuplicates()
        .collect()
    )
    return {
        _group_signature(
            row["cate_1_depth"],
            row["cate_2_depth"],
            int(row["sc_measurement"]),
        )
        for row in rows
    }


def _clear_checkpoint_rows(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    checkpoint_key: str,
) -> None:
    """Delete checkpoint rows after a successful full run."""
    table_name = get_log_table(config, "pipeline_progress")
    if not spark.catalog.tableExists(table_name):
        return

    safe_key = checkpoint_key.replace("'", "''")
    spark.sql(
        f"""
        DELETE FROM {table_name}
        WHERE checkpoint_key = '{safe_key}'
          AND pipeline_name = '{PIPELINE_NAME}'
        """
    )


def _clear_runtime_memory(spark: SparkSession) -> None:
    """Release Python and Spark-side cached memory between groups / at end."""
    spark.catalog.clearCache()
    gc.collect()


def run_taxonomy_classification_batch(
    spark,
    *,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    model_key: str = "gpt_55",
    limit_group_count: int | None = None,
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
    force_seed_generation: bool = False,
    max_rows_per_group: int | None = None,
    use_llm_fallback: bool = True,
    save_rule_profile: bool = True,
    save_topic_pool: bool = True,
    save_classification_detail: bool = True,
    source_period_start: str | None = None,
    source_period_end: str | None = None,
    resume_from_checkpoint: bool = True,
    cleanup_checkpoint_on_success: bool = True,
    continue_on_group_failure: bool = False,
    print_progress: bool = True,
) -> dict[str, Any]:
    """Run taxonomy design and 1st-pass topic classification for selected groups.

    Operational behavior:
    - persists each group's outputs immediately
    - records checkpoint rows after each successful group
    - resumes by skipping already completed groups with the same checkpoint key
    - can continue to the next group when one group repeatedly fails
    - clears Spark/Python cache after each group and after full completion
    """
    effective_config = config or load_config(config_path)
    run_id = str(effective_config.get("runtime", {}).get("resolved_run_id", ""))
    checkpoint_key = _checkpoint_key(
        model_key=model_key,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        limit_group_count=limit_group_count,
    )

    target_groups = load_design_target_groups(
        spark=spark,
        config=effective_config,
        limit_group_count=limit_group_count,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
    )

    completed_signatures = (
        _load_completed_signatures(
            spark,
            effective_config,
            checkpoint_key=checkpoint_key,
        )
        if resume_from_checkpoint
        else set()
    )

    if print_progress:
        print(
            f"[{PIPELINE_NAME}] start | groups={len(target_groups)} | "
            f"checkpoint_key={checkpoint_key} | completed_checkpoint_groups={len(completed_signatures)}"
        )

    classification_summaries: list[dict[str, Any]] = []
    failed_groups: list[dict[str, Any]] = []
    processed_group_count = 0
    skipped_group_count = 0
    failed_group_count = 0
    saved_tables: dict[str, str] = {}

    for index, group_row in enumerate(target_groups, start=1):
        group_cate_1 = group_row["cate_1_depth"]
        group_cate_2 = group_row["cate_2_depth"]
        group_sc = int(group_row["sc_measurement"])
        signature = _group_signature(group_cate_1, group_cate_2, group_sc)

        existing_classification_summary = (
            load_existing_classification_summary(
                spark,
                effective_config,
                cate_1_depth=group_cate_1,
                cate_2_depth=group_cate_2,
                sc_measurement=group_sc,
                model_key=model_key,
            )
            if resume_from_checkpoint and save_classification_detail
            else None
        )
        if existing_classification_summary is not None:
            classification_summaries.append(existing_classification_summary)
            skipped_group_count += 1
            if print_progress:
                print(
                    f"[{PIPELINE_NAME}] skip {index}/{len(target_groups)} | "
                    f"{group_cate_1} / {group_cate_2} / {group_sc} | "
                    "reason=classification_detail_exists"
                )
            continue

        if signature in completed_signatures:
            skipped_group_count += 1
            if print_progress:
                print(
                    f"[{PIPELINE_NAME}] skip {index}/{len(target_groups)} | "
                    f"{group_cate_1} / {group_cate_2} / {group_sc} | reason=checkpoint_completed"
                )
            continue

        if print_progress:
            print(
                f"[{PIPELINE_NAME}] run {index}/{len(target_groups)} | "
                f"{group_cate_1} / {group_cate_2} / {group_sc}"
            )

        _append_progress_row(
            spark,
            effective_config,
            checkpoint_key=checkpoint_key,
            run_id=run_id,
            model_key=model_key,
            cate_1_depth=group_cate_1,
            cate_2_depth=group_cate_2,
            sc_measurement=group_sc,
            status="running",
            step="group_start",
            message="group execution started",
        )

        try:
            rule_profile_reused = False
            rule_profile_result = (
                load_existing_rule_profile_result(
                    spark,
                    effective_config,
                    cate_1_depth=group_cate_1,
                    cate_2_depth=group_cate_2,
                    sc_measurement=group_sc,
                    model_key=model_key,
                )
                if resume_from_checkpoint
                else None
            )
            if rule_profile_result is not None:
                rule_profile_reused = True
                if print_progress:
                    print("  - reusing existing rule_profile")
            else:
                if print_progress:
                    print("  - generating rule_profile")
                rule_profile_result = generate_rule_profile_for_group(
                    config=effective_config,
                    spark=spark,
                    cate_1_depth=group_cate_1,
                    cate_2_depth=group_cate_2,
                    sc_measurement=group_sc,
                    model_key=model_key,
                    force_seed_generation=force_seed_generation,
                )

            if save_rule_profile and rule_profile_result is not None and not rule_profile_reused:
                saved_tables["rule_profile"] = save_rule_profiles(
                    spark=spark,
                    config=effective_config,
                    results=[rule_profile_result],
                    model_key=model_key,
                    write_mode="replace_groups",
                )
                _append_progress_row(
                    spark,
                    effective_config,
                    checkpoint_key=checkpoint_key,
                    run_id=run_id,
                    model_key=model_key,
                    cate_1_depth=group_cate_1,
                    cate_2_depth=group_cate_2,
                    sc_measurement=group_sc,
                    status="completed",
                    step="rule_profile_done",
                    message="rule_profile available",
                )

            topic_pool_reused = False
            topic_pool_result = (
                load_existing_topic_pool_result(
                    spark,
                    effective_config,
                    cate_1_depth=group_cate_1,
                    cate_2_depth=group_cate_2,
                    sc_measurement=group_sc,
                    model_key=model_key,
                )
                if resume_from_checkpoint
                else None
            )
            if topic_pool_result is not None:
                topic_pool_reused = True
                if print_progress:
                    print("  - reusing existing topic_pool")
            else:
                if print_progress:
                    print("  - generating topic_pool")
                topic_pool_result = generate_topic_pool_for_group(
                    config=effective_config,
                    spark=spark,
                    cate_1_depth=group_cate_1,
                    cate_2_depth=group_cate_2,
                    sc_measurement=group_sc,
                    rule_profile=rule_profile_result,
                    model_key=model_key,
                )

            if save_topic_pool and topic_pool_result is not None and not topic_pool_reused:
                saved_tables["topic_pool"] = save_topic_pools(
                    spark=spark,
                    config=effective_config,
                    results=[topic_pool_result],
                    model_key=model_key,
                    write_mode="replace_groups",
                )
                _append_progress_row(
                    spark,
                    effective_config,
                    checkpoint_key=checkpoint_key,
                    run_id=run_id,
                    model_key=model_key,
                    cate_1_depth=group_cate_1,
                    cate_2_depth=group_cate_2,
                    sc_measurement=group_sc,
                    status="completed",
                    step="topic_pool_done",
                    message="topic_pool available",
                )

            if print_progress:
                print("  - classifying memos")
            classification_result = classify_topic_for_group(
                spark,
                config=effective_config,
                rule_profile=rule_profile_result,
                topic_pool=topic_pool_result,
                cate_1_depth=group_cate_1,
                cate_2_depth=group_cate_2,
                sc_measurement=group_sc,
                model_key=model_key,
                max_rows=max_rows_per_group,
                use_llm_fallback=use_llm_fallback,
            )
            classification_summary = _summarize_classification_result(
                classification_result
            )

            if save_classification_detail:
                if print_progress:
                    print("  - saving classification_detail")
                saved_tables["classification_detail"] = save_classification_details(
                    spark=spark,
                    config=effective_config,
                    results=[classification_result],
                    model_key=model_key,
                    write_mode="replace_groups",
                    source_period_start=source_period_start,
                    source_period_end=source_period_end,
                )

            classification_summaries.append(classification_summary)
            processed_group_count += 1

            _append_progress_row(
                spark,
                effective_config,
                checkpoint_key=checkpoint_key,
                run_id=run_id,
                model_key=model_key,
                cate_1_depth=group_cate_1,
                cate_2_depth=group_cate_2,
                sc_measurement=group_sc,
                status="completed",
                step="group_done",
                message="group execution completed",
                summary=classification_summary,
            )

            if print_progress:
                print(
                    "  - done | "
                    f"rows={classification_summary['row_count']} | "
                    f"overall={classification_summary['overall_count']} | "
                    f"others={classification_summary['others_count']} | "
                    f"llm_used={classification_summary['llm_used_count']}"
                )

        except Exception as error:
            error_message = str(error)
            _append_progress_row(
                spark,
                effective_config,
                checkpoint_key=checkpoint_key,
                run_id=run_id,
                model_key=model_key,
                cate_1_depth=group_cate_1,
                cate_2_depth=group_cate_2,
                sc_measurement=group_sc,
                status="failed",
                step="group_failed",
                message=error_message,
            )
            if print_progress:
                print(
                    f"[{PIPELINE_NAME}] failed | {group_cate_1} / {group_cate_2} / {group_sc} | "
                    f"error={error}"
                )
            _clear_runtime_memory(spark)
            if continue_on_group_failure:
                failed_group_count += 1
                failed_groups.append(
                    {
                        "cate_1_depth": group_cate_1,
                        "cate_2_depth": group_cate_2,
                        "sc_measurement": group_sc,
                        "error_message": error_message,
                    }
                )
                if print_progress:
                    print(
                        f"[{PIPELINE_NAME}] continue | skipping failed group "
                        f"{group_cate_1} / {group_cate_2} / {group_sc}"
                    )
                continue
            raise

        _clear_runtime_memory(spark)

    total_rows = sum(row["row_count"] for row in classification_summaries)
    total_overall = sum(row["overall_count"] for row in classification_summaries)
    total_topic = sum(row["topic_count"] for row in classification_summaries)
    total_others = sum(row["others_count"] for row in classification_summaries)
    total_ambiguous = sum(row["ambiguous_count"] for row in classification_summaries)
    total_llm_used = sum(row["llm_used_count"] for row in classification_summaries)

    if (
        cleanup_checkpoint_on_success
        and failed_group_count == 0
        and processed_group_count + skipped_group_count == len(target_groups)
    ):
        _clear_checkpoint_rows(
            spark,
            effective_config,
            checkpoint_key=checkpoint_key,
        )
        if print_progress:
            print(
                f"[{PIPELINE_NAME}] success | checkpoint cleared | checkpoint_key={checkpoint_key}"
            )

    _clear_runtime_memory(spark)

    if print_progress:
        print(
            f"[{PIPELINE_NAME}] finished | processed={processed_group_count} | "
            f"skipped={skipped_group_count} | failed={failed_group_count} | "
            f"total_rows={total_rows} | total_others={total_others}"
        )

    return {
        "pipeline_name": PIPELINE_NAME,
        "checkpoint_key": checkpoint_key,
        "group_count": len(target_groups),
        "processed_group_count": processed_group_count,
        "skipped_group_count": skipped_group_count,
        "failed_group_count": failed_group_count,
        "classification_count": len(classification_summaries),
        "model_key": model_key,
        "saved_tables": saved_tables,
        "target_groups": target_groups,
        "classification_summaries": classification_summaries,
        "failed_groups": failed_groups,
        "total_row_count": total_rows,
        "total_overall_count": total_overall,
        "total_topic_count": total_topic,
        "total_others_count": total_others,
        "total_ambiguous_count": total_ambiguous,
        "total_llm_used_count": total_llm_used,
    }
