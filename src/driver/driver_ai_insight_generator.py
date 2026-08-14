"""Generate dashboard-ready AI insights from weighted correlation and WLS outputs.

The module deliberately sends only pre-aggregated statistical evidence to the LLM.
It never asks the model to calculate a statistic or infer causality from raw VOC data.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from common.config_loader import get_output_table
from common.llm_client import get_llm_client


GROUP_COLUMNS = ["segment_col", "segment_value", "group_dim", "group_key"]
OVERVIEW_Y_FEATURE = "__ALL__"

DRIVER_AI_INSIGHT_SCHEMA = StructType(
    [
        StructField("insight_key", StringType(), False),
        StructField("insight_level", StringType(), False),
        StructField("segment_col", StringType(), False),
        StructField("segment_value", StringType(), False),
        StructField("group_dim", StringType(), False),
        StructField("group_key", StringType(), False),
        StructField("y_feature", StringType(), False),
        StructField("analysis_status", StringType(), False),
        StructField("r_squared", DoubleType(), True),
        StructField("adj_r_squared", DoubleType(), True),
        StructField("prob_f", DoubleType(), True),
        StructField("y_obs", LongType(), True),
        StructField("significant_driver_count", LongType(), True),
        StructField("top_models_json", StringType(), True),
        StructField("top_drivers_json", StringType(), True),
        StructField("top_correlations_json", StringType(), True),
        StructField("condition_comparison_json", StringType(), True),
        StructField("core_summary", StringType(), True),
        StructField("detail_insight", StringType(), True),
        StructField("condition_insight", StringType(), True),
        StructField("caution_note", StringType(), True),
        StructField("source_hash", StringType(), False),
        StructField("model_key", StringType(), False),
        StructField("model_version", StringType(), False),
        StructField("model_endpoint", StringType(), False),
        StructField("prompt_version", StringType(), False),
        StructField("run_id", StringType(), True),
        StructField("run_date", StringType(), True),
        StructField("pipeline_stage", StringType(), True),
        StructField("created_at", TimestampType(), True),
        StructField("created_by", StringType(), True),
    ]
)


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("driver_ai_insight", {}) or {}
    return {
        "output_table_key": cfg.get("output_table_key", "driver_ai_insight"),
        "default_model_key": cfg.get("default_model_key", "sonnet_46"),
        "executive_model_key": cfg.get("executive_model_key", "opus_45"),
        "prompt_version": cfg.get("prompt_version", "v1_driver_ai_insight"),
        "min_y_obs": int(cfg.get("min_y_obs", 30)),
        "min_adj_r_squared": float(cfg.get("min_adj_r_squared", 0.10)),
        "max_prob_f": float(cfg.get("max_prob_f", 0.05)),
        "max_models_in_context": int(cfg.get("max_models_in_context", 3)),
        "max_drivers_in_context": int(cfg.get("max_drivers_in_context", 3)),
        "max_correlations_in_context": int(cfg.get("max_correlations_in_context", 3)),
    }


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _source_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()


def _model_config(config: dict[str, Any], model_key: str) -> dict[str, str]:
    model = (config.get("llm", {}).get("models", {}) or {}).get(model_key)
    if not model:
        raise KeyError(f"Unknown LLM model_key: {model_key}")
    return {
        "model_key": model_key,
        "model_version": str(model.get("model_version", model_key)),
        "model_endpoint": str(model["endpoint"]),
    }


def _as_records(df: DataFrame) -> list[dict[str, Any]]:
    return [row.asDict(recursive=True) for row in df.collect()]


def _round_record(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        result[key] = round(float(value), 4) if isinstance(value, float) else value
    return result


def _model_status(model: dict[str, Any] | None, cfg: dict[str, Any]) -> str:
    if not model:
        return "no_model"
    if (
        (model.get("y_obs") or 0) >= cfg["min_y_obs"]
        and (model.get("adj_r_squared") or 0.0) >= cfg["min_adj_r_squared"]
        and (model.get("prob_f") is not None)
        and float(model["prob_f"]) <= cfg["max_prob_f"]
    ):
        return "significant"
    return "weak_model"


def _insight_key(record: dict[str, Any], model_key: str, prompt_version: str) -> str:
    parts = [record[column] for column in GROUP_COLUMNS]
    parts.extend([record["insight_level"], record["y_feature"], model_key, prompt_version])
    return "||".join(str(value) for value in parts)


def _build_prompts(payload: dict[str, Any]) -> tuple[str, str]:
    system_prompt = """You are a cautious TV VOC and statistical-analysis advisor.
Write Korean dashboard insight text based only on the supplied statistical evidence.
Do not claim causality: use expressions such as '연관', '영향 가능성', and '우선 검토'.
Do not invent values, drivers, or comparisons absent from the input.
If analysis_status is weak_model or no_model, explicitly state that regression evidence is insufficient
and use the supplied correlation candidates only as exploratory evidence.
Return JSON only with core_summary, detail_insight, condition_insight, caution_note.
Each value must be a concise Korean string."""
    user_prompt = """Generate an executive-friendly dashboard insight from this evidence.

Required structure:
- core_summary: 1-2 sentences. For significant regression, identify the strongest evidence and up to three common drivers. For weak/no model, state the limitation and up to three correlation alternatives.
- detail_insight: Explain the evidence and practical product-planning implication without causal overclaim.
- condition_insight: Describe notable differences versus peer group keys when comparison evidence exists; otherwise state that comparison is limited.
- caution_note: State the relevant statistical caveat in one sentence.

Evidence JSON:
""" + _json(payload)
    return system_prompt, user_prompt


def _normalize_response(response: dict[str, Any]) -> dict[str, str]:
    def clean(key: str) -> str:
        value = response.get(key, "")
        return str(value).strip() if value is not None else ""

    result = {
        "core_summary": clean("core_summary"),
        "detail_insight": clean("detail_insight"),
        "condition_insight": clean("condition_insight"),
        "caution_note": clean("caution_note"),
    }
    if not result["core_summary"] or not result["detail_insight"]:
        raise ValueError("LLM insight response is missing core_summary or detail_insight.")
    return result


def _load_source_records(
    spark: SparkSession,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    models = _as_records(spark.table(get_output_table(config, "weighted_regression_model")))
    coefs = _as_records(spark.table(get_output_table(config, "weighted_regression")))
    corrs = _as_records(spark.table(get_output_table(config, "weighted_corr")))
    return models, coefs, corrs


def _same_scope(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(str(left.get(column)) == str(right.get(column)) for column in GROUP_COLUMNS)


def _top_drivers(
    coefs: list[dict[str, Any]],
    scope: dict[str, Any],
    y_feature: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in coefs
        if _same_scope(row, scope)
        and row.get("x_feature") != "β₀"
        and (y_feature is None or row.get("y_feature") == y_feature)
        and int(row.get("is_driver") or 0) == 1
    ]
    rows.sort(key=lambda row: (float(row.get("abs_coef") or 0.0), -float(row.get("p_value") or 1.0)), reverse=True)
    return [_round_record(row) for row in rows[:limit]]


def _top_correlations(
    corrs: list[dict[str, Any]],
    scope: dict[str, Any],
    y_feature: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in corrs
        if _same_scope(row, scope) and (y_feature is None or row.get("y_feature") == y_feature)
    ]
    rows.sort(key=lambda row: float(row.get("abs_weighted_corr") or 0.0), reverse=True)
    return [_round_record(row) for row in rows[:limit]]


def _comparison_records(
    models: list[dict[str, Any]],
    scope: dict[str, Any],
    y_feature: str | None,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if scope["group_dim"] == "all":
        return []
    rows = [
        row
        for row in models
        if row.get("segment_col") == scope["segment_col"]
        and row.get("segment_value") == scope["segment_value"]
        and row.get("group_dim") == scope["group_dim"]
        and (y_feature is None or row.get("y_feature") == y_feature)
    ]
    rows.sort(key=lambda row: float(row.get("adj_r_squared") or -1.0), reverse=True)
    return [_round_record(row) for row in rows[: cfg["max_models_in_context"]]]


def _build_contexts(
    models: list[dict[str, Any]],
    coefs: list[dict[str, Any]],
    corrs: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    include_group_overview: bool,
    include_y_feature_insights: bool,
    target_group_dims: list[str] | None,
    target_group_keys: list[str] | None,
    target_y_features: list[str] | None,
) -> list[dict[str, Any]]:
    scopes: dict[tuple[str, ...], dict[str, Any]] = {}
    for model in models:
        scope = {column: str(model.get(column)) for column in GROUP_COLUMNS}
        if target_group_dims and scope["group_dim"] not in target_group_dims:
            continue
        if target_group_keys and scope["group_key"] not in target_group_keys:
            continue
        scopes[tuple(scope[column] for column in GROUP_COLUMNS)] = scope

    contexts: list[dict[str, Any]] = []
    for scope in scopes.values():
        scope_models = [row for row in models if _same_scope(row, scope)]
        scope_models.sort(
            key=lambda row: (
                _model_status(row, cfg) == "significant",
                float(row.get("adj_r_squared") or -1.0),
            ),
            reverse=True,
        )
        if include_group_overview:
            top_models = [_round_record(row) for row in scope_models[: cfg["max_models_in_context"]]]
            primary = scope_models[0] if scope_models else None
            contexts.append(
                {
                    **scope,
                    "insight_level": "group_overview",
                    "y_feature": OVERVIEW_Y_FEATURE,
                    "analysis_status": _model_status(primary, cfg),
                    "r_squared": primary.get("r_squared") if primary else None,
                    "adj_r_squared": primary.get("adj_r_squared") if primary else None,
                    "prob_f": primary.get("prob_f") if primary else None,
                    "y_obs": primary.get("y_obs") if primary else None,
                    "significant_driver_count": len(_top_drivers(coefs, scope, None, 9999)),
                    "top_models": top_models,
                    "top_drivers": _top_drivers(coefs, scope, None, cfg["max_drivers_in_context"]),
                    "top_correlations": _top_correlations(corrs, scope, None, cfg["max_correlations_in_context"]),
                    "condition_comparison": _comparison_records(models, scope, None, cfg),
                }
            )

        if include_y_feature_insights:
            for model in scope_models:
                y_feature = str(model["y_feature"])
                if target_y_features and y_feature not in target_y_features:
                    continue
                contexts.append(
                    {
                        **scope,
                        "insight_level": "y_feature",
                        "y_feature": y_feature,
                        "analysis_status": _model_status(model, cfg),
                        "r_squared": model.get("r_squared"),
                        "adj_r_squared": model.get("adj_r_squared"),
                        "prob_f": model.get("prob_f"),
                        "y_obs": model.get("y_obs"),
                        "significant_driver_count": len(_top_drivers(coefs, scope, y_feature, 9999)),
                        "top_models": [_round_record(model)],
                        "top_drivers": _top_drivers(coefs, scope, y_feature, cfg["max_drivers_in_context"]),
                        "top_correlations": _top_correlations(corrs, scope, y_feature, cfg["max_correlations_in_context"]),
                        "condition_comparison": _comparison_records(models, scope, y_feature, cfg),
                    }
                )
    return contexts


def _existing_hashes(spark: SparkSession, table_name: str) -> dict[str, str]:
    if not spark.catalog.tableExists(table_name):
        return {}
    return {
        row["insight_key"]: row["source_hash"]
        for row in spark.table(table_name).select("insight_key", "source_hash").collect()
    }


def _save_rows(
    spark: SparkSession,
    table_name: str,
    rows: list[dict[str, Any]],
) -> int:
    if not rows:
        return 0
    df = spark.createDataFrame(rows, schema=DRIVER_AI_INSIGHT_SCHEMA)
    if spark.catalog.tableExists(table_name):
        keys = df.select("insight_key").dropDuplicates()
        keys.createOrReplaceTempView("_tmp_driver_ai_insight_keys")
        spark.sql(
            f"""
            DELETE FROM {table_name}
            WHERE EXISTS (
                SELECT 1
                FROM _tmp_driver_ai_insight_keys src
                WHERE {table_name}.insight_key = src.insight_key
            )
            """
        )
        df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table_name)
    else:
        df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
    return len(rows)


def generate_and_save_driver_ai_insights(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    model_key: str | None = None,
    include_group_overview: bool = True,
    include_y_feature_insights: bool = True,
    target_group_dims: list[str] | None = None,
    target_group_keys: list[str] | None = None,
    target_y_features: list[str] | None = None,
    skip_unchanged: bool = True,
    created_by: str = "driver_ai_insight_generator",
) -> dict[str, Any]:
    """Generate reusable insights, skipping unchanged evidence to control LLM cost."""
    cfg = _cfg(config)
    resolved_model_key = model_key or cfg["default_model_key"]
    model_info = _model_config(config, resolved_model_key)
    output_table = get_output_table(config, cfg["output_table_key"])
    models, coefs, corrs = _load_source_records(spark, config)
    contexts = _build_contexts(
        models,
        coefs,
        corrs,
        cfg,
        include_group_overview=include_group_overview,
        include_y_feature_insights=include_y_feature_insights,
        target_group_dims=target_group_dims,
        target_group_keys=target_group_keys,
        target_y_features=target_y_features,
    )
    existing_hashes = _existing_hashes(spark, output_table) if skip_unchanged else {}
    llm = get_llm_client(config=config, model_key=resolved_model_key)
    rows: list[dict[str, Any]] = []
    skipped_count = 0

    for index, context in enumerate(contexts, start=1):
        payload = {
            "scope": {column: context[column] for column in GROUP_COLUMNS},
            "insight_level": context["insight_level"],
            "y_feature": context["y_feature"],
            "analysis_status": context["analysis_status"],
            "model_metrics": {
                key: context[key]
                for key in ["r_squared", "adj_r_squared", "prob_f", "y_obs", "significant_driver_count"]
            },
            "top_models": context["top_models"],
            "top_drivers": context["top_drivers"],
            "top_correlations": context["top_correlations"],
            "condition_comparison": context["condition_comparison"],
        }
        source_hash = _source_hash(payload)
        insight_key = _insight_key(context, resolved_model_key, cfg["prompt_version"])
        if existing_hashes.get(insight_key) == source_hash:
            skipped_count += 1
            continue

        print(
            f"[driver_ai_insight] {index}/{len(contexts)} | "
            f"{context['group_dim']}={context['group_key']} | {context['y_feature']}"
        )
        system_prompt, user_prompt = _build_prompts(payload)
        response = llm.converse_json(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=1600)
        insight = _normalize_response(response)
        rows.append(
            {
                "insight_key": insight_key,
                "insight_level": context["insight_level"],
                **{column: context[column] for column in GROUP_COLUMNS},
                "y_feature": context["y_feature"],
                "analysis_status": context["analysis_status"],
                "r_squared": context["r_squared"],
                "adj_r_squared": context["adj_r_squared"],
                "prob_f": context["prob_f"],
                "y_obs": context["y_obs"],
                "significant_driver_count": context["significant_driver_count"],
                "top_models_json": _json(context["top_models"]),
                "top_drivers_json": _json(context["top_drivers"]),
                "top_correlations_json": _json(context["top_correlations"]),
                "condition_comparison_json": _json(context["condition_comparison"]),
                **insight,
                "source_hash": source_hash,
                **model_info,
                "prompt_version": cfg["prompt_version"],
                "run_id": config.get("runtime", {}).get("resolved_run_id"),
                "run_date": config.get("runtime", {}).get("resolved_run_date"),
                "pipeline_stage": "driver_ai_insight",
                "created_at": datetime.utcnow(),
                "created_by": created_by,
            }
        )

    saved_count = _save_rows(spark, output_table, rows)
    return {
        "output_table": output_table,
        "model_key": resolved_model_key,
        "model_endpoint": model_info["model_endpoint"],
        "candidate_count": len(contexts),
        "generated_count": saved_count,
        "skipped_unchanged_count": skipped_count,
    }
