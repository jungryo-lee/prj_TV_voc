"""Writers for taxonomy classification detail outputs."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from common.config_loader import get_output_table


CLASSIFICATION_DETAIL_WRITE_SCHEMA = T.StructType(
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


def _runtime_value(config: dict[str, Any], key: str, default: str = "") -> str:
    """Read resolved runtime metadata with a safe fallback."""
    return str(config.get("runtime", {}).get(key, default) or default)


def _version_value(config: dict[str, Any], key: str, default: str = "") -> str:
    """Read version metadata with a safe fallback."""
    return str(config.get("version", {}).get(key, default) or default)


def _collect_result_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten classifier result payloads into detail rows."""
    flattened_rows: list[dict[str, Any]] = []

    for result in results:
        if "rows" in result and isinstance(result["rows"], list):
            flattened_rows.extend(result["rows"])
        else:
            flattened_rows.append(result)

    return flattened_rows


def _normalize_candidate_topics_json(value: Any) -> str:
    """Ensure candidate topics are stored as a JSON string."""
    if isinstance(value, str):
        return value
    return json.dumps(value or [], ensure_ascii=False)


def build_classification_detail_rows(
    results: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    model_key: str = "gpt_55",
    created_by: str = "codex",
    pipeline_stage: str = "classification_detail",
    is_latest: bool = True,
    source_period_start: str | None = None,
    source_period_end: str | None = None,
) -> list[dict[str, Any]]:
    """Convert classifier outputs into table-ready detail rows."""
    run_id = _runtime_value(config, "resolved_run_id")
    run_date = _runtime_value(config, "resolved_run_date")
    prompt_version = _version_value(config, "prompt_version")
    taxonomy_version = _version_value(config, "taxonomy_version")
    pipeline_version = _version_value(config, "pipeline_version")
    created_at = datetime.utcnow().isoformat(timespec="seconds")

    rows: list[dict[str, Any]] = []

    for row in _collect_result_rows(results):
        rows.append(
            {
                "memo_id": _clean_text(row.get("memo_id")),
                "memo": row.get("memo"),
                "memo_norm": _clean_text(row.get("memo_norm")),
                "cate_1_depth": _clean_text(row.get("cate_1_depth")),
                "cate_2_depth": _clean_text(row.get("cate_2_depth")),
                "sc_measurement": int(row.get("sc_measurement")),
                "year": _clean_text(row.get("year")),
                "country": _clean_text(row.get("country")),
                "brand_name": _clean_text(row.get("brand_name")),
                "device_type": _clean_text(row.get("device_type")),
                "pred_topic": _clean_text(row.get("pred_topic")),
                "pred_topic_type": _clean_text(row.get("pred_topic_type")),
                "classification_stage": _clean_text(row.get("classification_stage")),
                "confidence_score": row.get("confidence_score"),
                "candidate_topics_json": _normalize_candidate_topics_json(
                    row.get("candidate_topics_json")
                ),
                "match_reason": _clean_text(row.get("match_reason")),
                "llm_used_yn": bool(row.get("llm_used_yn", False)),
                "review_needed_yn": bool(row.get("review_needed_yn", False)),
                "run_id": _clean_text(row.get("run_id") or run_id),
                "run_date": _clean_text(row.get("run_date") or run_date),
                "pipeline_stage": pipeline_stage,
                "prompt_version": _clean_text(
                    row.get("prompt_version") or prompt_version
                ),
                "taxonomy_version": _clean_text(
                    row.get("taxonomy_version") or taxonomy_version
                ),
                "model_version": _clean_text(row.get("model_version") or model_key),
                "pipeline_version": _clean_text(
                    row.get("pipeline_version") or pipeline_version
                ),
                "source_period_start": source_period_start,
                "source_period_end": source_period_end,
                "is_latest": bool(is_latest),
                "created_at": _clean_text(row.get("created_at") or created_at),
                "created_by": created_by,
            }
        )

    return rows


def build_classification_detail_spark_df(
    spark: SparkSession,
    results: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    model_key: str = "gpt_55",
    created_by: str = "codex",
    pipeline_stage: str = "classification_detail",
    is_latest: bool = True,
    source_period_start: str | None = None,
    source_period_end: str | None = None,
) -> DataFrame:
    """Build a Spark DataFrame for classification detail writes."""
    rows = build_classification_detail_rows(
        results=results,
        config=config,
        model_key=model_key,
        created_by=created_by,
        pipeline_stage=pipeline_stage,
        is_latest=is_latest,
        source_period_start=source_period_start,
        source_period_end=source_period_end,
    )
    return spark.createDataFrame(rows, schema=CLASSIFICATION_DETAIL_WRITE_SCHEMA)


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
    keys_df.createOrReplaceTempView("_tmp_classification_detail_keys")

    spark.sql(
        f"""
        DELETE FROM {table_name}
        WHERE (cate_1_depth, cate_2_depth, sc_measurement, model_version, prompt_version, taxonomy_version)
              IN (
                  SELECT cate_1_depth, cate_2_depth, sc_measurement, model_version, prompt_version, taxonomy_version
                  FROM _tmp_classification_detail_keys
              )
        """
    )


def save_classification_details(
    spark: SparkSession,
    config: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    model_key: str = "gpt_55",
    created_by: str = "codex",
    pipeline_stage: str = "classification_detail",
    write_mode: str = "replace_groups",
    is_latest: bool = True,
    source_period_start: str | None = None,
    source_period_end: str | None = None,
) -> str:
    """Save classification detail rows to the configured output table.

    Supported write modes:
    - replace_groups: delete same group/model/version rows, then append
    - append: append rows as-is
    - overwrite: overwrite entire table
    """
    if not results:
        raise ValueError("results must not be empty.")

    table_name = get_output_table(config, "classification_detail")
    prompt_version = _version_value(config, "prompt_version")
    taxonomy_version = _version_value(config, "taxonomy_version")

    df = build_classification_detail_spark_df(
        spark=spark,
        results=results,
        config=config,
        model_key=model_key,
        created_by=created_by,
        pipeline_stage=pipeline_stage,
        is_latest=is_latest,
        source_period_start=source_period_start,
        source_period_end=source_period_end,
    )

    if write_mode == "overwrite":
        df.write.mode("overwrite").format("delta").saveAsTable(table_name)
        return table_name

    if write_mode == "replace_groups":
        _delete_existing_group_rows(
            spark=spark,
            table_name=table_name,
            df=df,
            model_key=model_key,
            prompt_version=prompt_version,
            taxonomy_version=taxonomy_version,
        )
        df.write.mode("append").format("delta").saveAsTable(table_name)
        return table_name

    if write_mode == "append":
        df.write.mode("append").format("delta").saveAsTable(table_name)
        return table_name

    raise ValueError(f"Unsupported write_mode: {write_mode}")
