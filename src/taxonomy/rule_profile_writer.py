"""Writers for taxonomy rule-profile outputs."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from common.config_loader import get_output_table


RULE_PROFILE_WRITE_SCHEMA = T.StructType(
    [
        T.StructField("cate_1_depth", T.StringType(), True),
        T.StructField("cate_2_depth", T.StringType(), True),
        T.StructField("sc_measurement", T.IntegerType(), True),
        T.StructField("sample_memo_count", T.IntegerType(), True),
        T.StructField("overall_topic_name", T.StringType(), True),
        T.StructField("overall_allowed_rule", T.StringType(), True),
        T.StructField("overall_block_rule", T.StringType(), True),
        T.StructField("overall_sentiment_terms_json", T.StringType(), True),
        T.StructField("feature_hint_terms_json", T.StringType(), True),
        T.StructField("reason_signal_terms_json", T.StringType(), True),
        T.StructField("non_overall_examples_json", T.StringType(), True),
        T.StructField("category_seed_summary", T.StringType(), True),
        T.StructField("category_seed_has_static_seed", T.BooleanType(), True),
        T.StructField(
            "category_seed_static_feature_hint_terms_json",
            T.StringType(),
            True,
        ),
        T.StructField("category_seed_feature_hint_terms_json", T.StringType(), True),
        T.StructField("category_seed_reason_signal_terms_json", T.StringType(), True),
        T.StructField(
            "category_seed_overall_sentiment_terms_json",
            T.StringType(),
            True,
        ),
        T.StructField(
            "category_seed_candidate_topic_labels_json",
            T.StringType(),
            True,
        ),
        T.StructField(
            "category_seed_sample_non_overall_memos_json",
            T.StringType(),
            True,
        ),
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
    """Collapse whitespace for stable table payloads."""
    return " ".join(str(value or "").split()).strip()


def _json_dumps(values: list[Any] | None) -> str:
    """Serialize list-like payloads as compact JSON strings."""
    return json.dumps(values or [], ensure_ascii=False)


def _runtime_value(config: dict[str, Any], key: str, default: str = "") -> str:
    """Read resolved runtime metadata with a safe fallback."""
    return str(config.get("runtime", {}).get(key, default) or default)


def _version_value(config: dict[str, Any], key: str, default: str = "") -> str:
    """Read version metadata with a safe fallback."""
    return str(config.get("version", {}).get(key, default) or default)


def build_rule_profile_rows(
    results: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    model_key: str = "gpt_55",
    created_by: str = "codex",
    pipeline_stage: str = "rule_profile",
    is_latest: bool = True,
) -> list[dict[str, Any]]:
    """Convert generator outputs into table-ready row dicts."""
    run_id = _runtime_value(config, "resolved_run_id")
    run_date = _runtime_value(config, "resolved_run_date")
    prompt_version = _version_value(config, "prompt_version")
    taxonomy_version = _version_value(config, "taxonomy_version")
    pipeline_version = _version_value(config, "pipeline_version")
    created_at = datetime.utcnow().isoformat(timespec="seconds")

    rows: list[dict[str, Any]] = []

    for result in results:
        category_seed = result.get("category_seed_used") or {}

        rows.append(
            {
                "cate_1_depth": _clean_text(result.get("cate_1_depth")),
                "cate_2_depth": _clean_text(result.get("cate_2_depth")),
                "sc_measurement": int(result.get("sc_measurement")),
                "sample_memo_count": int(result.get("sample_memo_count", 0)),
                "overall_topic_name": _clean_text(result.get("overall_topic_name")),
                "overall_allowed_rule": _clean_text(result.get("overall_allowed_rule")),
                "overall_block_rule": _clean_text(result.get("overall_block_rule")),
                "overall_sentiment_terms_json": _json_dumps(
                    result.get("overall_sentiment_terms")
                ),
                "feature_hint_terms_json": _json_dumps(
                    result.get("feature_hint_terms")
                ),
                "reason_signal_terms_json": _json_dumps(
                    result.get("reason_signal_terms")
                ),
                "non_overall_examples_json": _json_dumps(
                    result.get("non_overall_examples")
                ),
                "category_seed_summary": _clean_text(
                    category_seed.get("category_summary")
                ),
                "category_seed_has_static_seed": bool(
                    category_seed.get("has_static_seed", False)
                ),
                "category_seed_static_feature_hint_terms_json": _json_dumps(
                    category_seed.get("static_feature_hint_terms")
                ),
                "category_seed_feature_hint_terms_json": _json_dumps(
                    category_seed.get("feature_hint_terms")
                ),
                "category_seed_reason_signal_terms_json": _json_dumps(
                    category_seed.get("reason_signal_terms")
                ),
                "category_seed_overall_sentiment_terms_json": _json_dumps(
                    category_seed.get("overall_sentiment_terms")
                ),
                "category_seed_candidate_topic_labels_json": _json_dumps(
                    category_seed.get("candidate_topic_labels")
                ),
                "category_seed_sample_non_overall_memos_json": _json_dumps(
                    category_seed.get("sample_non_overall_memos")
                ),
                "run_id": run_id,
                "run_date": run_date,
                "pipeline_stage": pipeline_stage,
                "prompt_version": prompt_version,
                "taxonomy_version": taxonomy_version,
                "model_version": model_key,
                "pipeline_version": pipeline_version,
                "source_period_start": None,
                "source_period_end": None,
                "is_latest": bool(is_latest),
                "created_at": created_at,
                "created_by": created_by,
            }
        )

    return rows


def build_rule_profile_spark_df(
    spark: SparkSession,
    results: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    model_key: str = "gpt_55",
    created_by: str = "codex",
    pipeline_stage: str = "rule_profile",
    is_latest: bool = True,
) -> DataFrame:
    """Build a Spark DataFrame for rule-profile writes."""
    rows = build_rule_profile_rows(
        results=results,
        config=config,
        model_key=model_key,
        created_by=created_by,
        pipeline_stage=pipeline_stage,
        is_latest=is_latest,
    )
    return spark.createDataFrame(rows, schema=RULE_PROFILE_WRITE_SCHEMA)


def _delete_existing_group_rows(
    spark: SparkSession,
    table_name: str,
    df: DataFrame,
    *,
    model_key: str,
    prompt_version: str,
    taxonomy_version: str,
) -> None:
    """Delete existing rows for the same group/model/version combination."""
    if not spark.catalog.tableExists(table_name):
        return

    keys_df = (
        df.select("cate_1_depth", "cate_2_depth", "sc_measurement")
        .dropDuplicates()
        .withColumn("model_version", F.lit(model_key))
        .withColumn("prompt_version", F.lit(prompt_version))
        .withColumn("taxonomy_version", F.lit(taxonomy_version))
    )
    keys_df.createOrReplaceTempView("_tmp_rule_profile_keys")

    spark.sql(
        f"""
        DELETE FROM {table_name}
        WHERE EXISTS (
            SELECT 1
            FROM _tmp_rule_profile_keys src
            WHERE {table_name}.cate_1_depth = src.cate_1_depth
              AND {table_name}.cate_2_depth = src.cate_2_depth
              AND {table_name}.sc_measurement = src.sc_measurement
              AND {table_name}.model_version = src.model_version
              AND {table_name}.prompt_version = src.prompt_version
              AND {table_name}.taxonomy_version = src.taxonomy_version
        )
        """
    )


def save_rule_profiles(
    spark: SparkSession,
    config: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    model_key: str = "gpt_55",
    created_by: str = "codex",
    pipeline_stage: str = "rule_profile",
    write_mode: str = "replace_groups",
    is_latest: bool = True,
) -> str:
    """Save rule-profile results to the configured output table.

    Supported write modes:
    - replace_groups: delete same group/model/version rows, then append
    - append: append rows as-is
    - overwrite: overwrite entire table
    """
    if not results:
        raise ValueError("results must not be empty.")

    table_name = get_output_table(config, "rule_profile")
    prompt_version = _version_value(config, "prompt_version")
    taxonomy_version = _version_value(config, "taxonomy_version")

    df = build_rule_profile_spark_df(
        spark=spark,
        results=results,
        config=config,
        model_key=model_key,
        created_by=created_by,
        pipeline_stage=pipeline_stage,
        is_latest=is_latest,
    )

    if write_mode == "overwrite":
        df.write.mode("overwrite").format("delta").saveAsTable(table_name)
        return table_name

    if write_mode == "append":
        mode = "append" if spark.catalog.tableExists(table_name) else "overwrite"
        df.write.mode(mode).format("delta").saveAsTable(table_name)
        return table_name

    if write_mode == "replace_groups":
        _delete_existing_group_rows(
            spark,
            table_name,
            df,
            model_key=model_key,
            prompt_version=prompt_version,
            taxonomy_version=taxonomy_version,
        )
        mode = "append" if spark.catalog.tableExists(table_name) else "overwrite"
        df.write.mode(mode).format("delta").saveAsTable(table_name)
        return table_name

    raise ValueError(f"Unsupported write_mode: {write_mode}")
