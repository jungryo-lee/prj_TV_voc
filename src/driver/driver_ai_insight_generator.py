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
        StructField("analysis_mode", StringType(), True),
        StructField("confidence_title", StringType(), True),
        StructField("summary_title", StringType(), True),
        StructField("driver_title", StringType(), True),
        StructField("detail_title", StringType(), True),
        StructField("confidence_summary", StringType(), True),
        StructField("driver_summary", StringType(), True),
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
        "max_group_keys_in_context": int(cfg.get("max_group_keys_in_context", 10)),
        "min_abs_coef": float(cfg.get("min_abs_coef", 0.10)),
        "min_abs_weighted_corr": float(cfg.get("min_abs_weighted_corr", 0.30)),
        "max_coef_p_value": float(cfg.get("max_coef_p_value", 0.05)),
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
    system_prompt = """You are a senior TV product-planning strategist and a rigorous VOC data analyst.
Write concise Korean dashboard insight cards based only on the supplied weighted-correlation and WLS evidence.

Your audience is TV SW/UX product planners who must decide what to investigate, prioritize, and validate in the next product cycle.
Translate statistics into product-planning language, but preserve statistical discipline.

Evidence rules:
- Do not claim causality. Use '연관', '관련성', '우선 검토', or '검증 가설'.
- Do not invent values, drivers, product facts, group differences, or comparisons absent from the input.
- `y_obs` is the number of model-level observations, not respondents, customers, or review counts.
- Never write p≈0. Use a supplied p-value faithfully; use 'p<0.001' only when the supplied value is below 0.001.
- Use '평가 점수' or '평가 경험' instead of '만족도' unless the Y feature explicitly represents satisfaction.
- Treat coefficients and correlations as directional evidence, not effect-size rankings across groups.
- When analysis_status is weak_model or no_model, state that regression evidence is insufficient and treat correlations as exploratory only.

Anti-repetition rules:
- The four cards must have distinct jobs. Never repeat the same full sentence, numeric fact, driver list, or conclusion across cards.
- Mention an X driver in only one card unless a short reference is essential; if referenced again, add new information rather than restating it.
- Do not use generic filler such as '유의미한 인사이트', '다양한 관점', '지속적인 모니터링이 필요', or '향후 확인이 필요' without a concrete object and reason.
- Avoid formulaic openings and repeated endings such as '...입니다' in every card. Use direct, professional dashboard prose.
- Do not restate the card title in its body and do not add headings, bullets, markdown, or preambles.

Card-writing rules:
- Each card is one or two sentences, approximately 90 to 220 Korean characters.
- confidence_summary: explain only evidence strength, coverage, and limitation. Do not list drivers or recommendations.
- core_summary: explain the most decision-relevant relationship pattern. Do not repeat model-quality caveats already in confidence_summary.
- driver_summary: state only common and differentiated X drivers with their relevant Y or group context. Do not give generic actions.
- detail_insight: convert evidence into up to two concrete product-planning priorities and one measurable validation question. Do not repeat the driver list.
- caution_note: one short methodological boundary statement.

Return JSON only with confidence_summary, core_summary, driver_summary, detail_insight, caution_note.
Each value must be a concise Korean string."""
    if payload["analysis_mode"] == "all_relationship":
        requirement = """Mode: overall relationship diagnosis. Every Y feature is in scope; no single Y feature is selected.
- confidence_summary: State the count of statistically usable representative models and evidence quality using adjusted R-squared, prob_f, and y_obs only when present.
- core_summary: Select at most three representative Y models. Describe the portfolio-level relationship pattern, such as whether the evidence is concentrated in usability, speed, content, or control experience. Do not list every model.
- driver_summary: Select at most three X features that recur across Y models. Call a driver '공통' only when appearance_count is at least 2. State which outcome areas it recurs in.
- detail_insight: Propose no more than two cross-product priorities and one validation question that can be checked in VOC, release, or product-quality data. Separate observed evidence from the hypothesis to validate.
- caution_note: State that this is observational association analysis, not causal proof."""
    else:
        requirement = """Mode: condition-group comparison for one selected Y feature.
- confidence_summary: State how many group keys have statistically usable models, identify the evidence coverage, and name the limitation for weak or missing groups. Do not describe drivers.
- core_summary: Describe the selected Y feature's relationship pattern across group keys in one integrated comparison. Highlight only meaningful contrasts supported by the supplied evidence; do not mechanically enumerate every group.
- driver_summary: Separate common drivers from differentiated drivers. A common driver must appear in at least two group keys. For differentiated drivers, name only the relevant group keys and direction, not a generic group-by-group recap.
- detail_insight: Propose one common planning priority, one group-specific investigation priority, and one validation question. The action must name the product experience or operating condition to inspect.
- caution_note: State that coefficient magnitude alone must not be used to claim one group is superior to another."""
    user_prompt = """Generate an executive-friendly dashboard insight from this evidence.

Required structure:
""" + requirement + """

Evidence JSON:
""" + _json(payload)
    return system_prompt, user_prompt


def _normalize_response(response: dict[str, Any]) -> dict[str, str]:
    def clean(key: str) -> str:
        value = response.get(key, "")
        return str(value).strip() if value is not None else ""

    result = {
        "confidence_summary": clean("confidence_summary"),
        "core_summary": clean("core_summary"),
        "driver_summary": clean("driver_summary"),
        "detail_insight": clean("detail_insight"),
        "caution_note": clean("caution_note"),
    }
    if not all(result[key] for key in ["confidence_summary", "core_summary", "driver_summary", "detail_insight"]):
        raise ValueError("LLM insight response is missing a required dashboard card value.")
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


def _is_dashboard_driver(row: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """Apply the same evidence threshold used by the Tableau coefficient view."""
    return (
        row.get("x_feature") != "β₀"
        and float(row.get("abs_coef") or 0.0) >= cfg["min_abs_coef"]
        and row.get("p_value") is not None
        and float(row["p_value"]) <= cfg["max_coef_p_value"]
        and abs(float(row.get("weighted_corr") or 0.0)) >= cfg["min_abs_weighted_corr"]
    )


def _top_drivers(
    coefs: list[dict[str, Any]],
    scope: dict[str, Any],
    y_feature: str | None,
    limit: int,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in coefs
        if _same_scope(row, scope)
        and (y_feature is None or row.get("y_feature") == y_feature)
        and _is_dashboard_driver(row, cfg)
    ]
    rows.sort(key=lambda row: (float(row.get("abs_coef") or 0.0), -float(row.get("p_value") or 1.0)), reverse=True)
    return [_round_record(row) for row in rows[:limit]]


def _top_correlations(
    corrs: list[dict[str, Any]],
    scope: dict[str, Any],
    y_feature: str | None,
    limit: int,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in corrs
        if _same_scope(row, scope)
        and (y_feature is None or row.get("y_feature") == y_feature)
        and float(row.get("abs_weighted_corr") or 0.0) >= cfg["min_abs_weighted_corr"]
    ]
    rows.sort(key=lambda row: float(row.get("abs_weighted_corr") or 0.0), reverse=True)
    return [_round_record(row) for row in rows[:limit]]


def _common_drivers(
    coefs: list[dict[str, Any]],
    scope: dict[str, Any],
    top_y_features: list[str],
    limit: int,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Summarize X features repeated across representative Y models."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in coefs:
        if (
            _same_scope(row, scope)
            and row.get("y_feature") in top_y_features
            and _is_dashboard_driver(row, cfg)
        ):
            grouped.setdefault(str(row["x_feature"]), []).append(row)

    results: list[dict[str, Any]] = []
    for x_feature, rows in grouped.items():
        distinct_y_features = sorted({str(row["y_feature"]) for row in rows})
        representative = max(rows, key=lambda row: float(row.get("abs_coef") or 0.0))
        results.append(
            {
                "x_feature": x_feature,
                "appearance_count": len(distinct_y_features),
                "affected_y_features": distinct_y_features,
                "representative_coef": round(float(representative.get("coef") or 0.0), 4),
                "best_p_value": round(float(min(row.get("p_value") or 1.0 for row in rows)), 6),
            }
        )
    results = [row for row in results if row["appearance_count"] >= 2]
    results.sort(
        key=lambda row: (row["appearance_count"], abs(row["representative_coef"])),
        reverse=True,
    )
    return results[:limit]


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
    current_rows = [row for row in rows if row.get("group_key") == scope["group_key"]]
    peer_rows = [row for row in rows if row.get("group_key") != scope["group_key"]]
    selected_rows = current_rows[:1] + peer_rows[: max(cfg["max_models_in_context"] - 1, 0)]
    return [_round_record(row) for row in selected_rows]


def _card_titles(analysis_mode: str, group_dim: str) -> dict[str, str]:
    if analysis_mode == "all_relationship":
        return {
            "confidence_title": "분석 신뢰도",
            "summary_title": "핵심 요약",
            "driver_title": "공통 Driver",
            "detail_title": "상세·기획 시사점",
        }

    group_label = {
        "brand_name": "브랜드",
        "country_code": "국가",
        "post_year": "연도",
        "unified_device_type": "디바이스 타입",
    }.get(group_dim, group_dim)
    return {
        "confidence_title": f"{group_label}별 분석 신뢰도",
        "summary_title": f"{group_label}별 핵심 요약",
        "driver_title": "공통·차별 Driver",
        "detail_title": f"{group_label}별 시사점",
    }


def _same_group_dimension_scope(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        str(left.get(column)) == str(right.get(column))
        for column in ["segment_col", "segment_value", "group_dim"]
    )


def _group_driver_summary(
    coefs: list[dict[str, Any]],
    scope: dict[str, Any],
    y_feature: str,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Summarize common and differentiated drivers across group keys for one Y."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in coefs:
        if (
            _same_group_dimension_scope(row, scope)
            and row.get("y_feature") == y_feature
            and _is_dashboard_driver(row, cfg)
        ):
            grouped.setdefault(str(row["x_feature"]), []).append(row)

    results: list[dict[str, Any]] = []
    for x_feature, rows in grouped.items():
        group_keys = sorted({str(row["group_key"]) for row in rows})
        positive_group_keys = sorted(
            {str(row["group_key"]) for row in rows if float(row.get("coef") or 0.0) > 0}
        )
        negative_group_keys = sorted(
            {str(row["group_key"]) for row in rows if float(row.get("coef") or 0.0) < 0}
        )
        representative = max(rows, key=lambda row: float(row.get("abs_coef") or 0.0))
        results.append(
            {
                "x_feature": x_feature,
                "appearance_count": len(group_keys),
                "driver_type": "common" if len(group_keys) >= 2 else "differentiated",
                "group_keys": group_keys,
                "positive_group_keys": positive_group_keys,
                "negative_group_keys": negative_group_keys,
                "representative_coef": round(float(representative.get("coef") or 0.0), 4),
                "best_p_value": round(float(min(row.get("p_value") or 1.0 for row in rows)), 6),
                "max_abs_weighted_corr": round(
                    max(abs(float(row.get("weighted_corr") or 0.0)) for row in rows),
                    4,
                ),
            }
        )
    results.sort(
        key=lambda row: (
            row["appearance_count"],
            row["max_abs_weighted_corr"],
            abs(row["representative_coef"]),
        ),
        reverse=True,
    )
    return results[: cfg["max_drivers_in_context"]]


def _group_top_correlations(
    corrs: list[dict[str, Any]],
    scope: dict[str, Any],
    y_feature: str,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in corrs
        if _same_group_dimension_scope(row, scope)
        and row.get("y_feature") == y_feature
        and float(row.get("abs_weighted_corr") or 0.0) >= cfg["min_abs_weighted_corr"]
    ]
    rows.sort(key=lambda row: float(row.get("abs_weighted_corr") or 0.0), reverse=True)
    return [_round_record(row) for row in rows[: cfg["max_correlations_in_context"]]]


def _build_dashboard_contexts(
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
    """Build one all-mode and one group-comparison context per dashboard selection."""
    filtered_models = [
        row
        for row in models
        if (not target_group_dims or str(row.get("group_dim")) in target_group_dims)
        and (not target_group_keys or str(row.get("group_key")) in target_group_keys)
        and (
            str(row.get("group_dim")) == "all"
            or not target_y_features
            or str(row.get("y_feature")) in target_y_features
        )
    ]
    filtered_coefs = [
        row
        for row in coefs
        if (not target_group_dims or str(row.get("group_dim")) in target_group_dims)
        and (not target_group_keys or str(row.get("group_key")) in target_group_keys)
    ]
    filtered_corrs = [
        row
        for row in corrs
        if (not target_group_dims or str(row.get("group_dim")) in target_group_dims)
        and (not target_group_keys or str(row.get("group_key")) in target_group_keys)
    ]
    contexts: list[dict[str, Any]] = []

    if include_group_overview:
        all_scopes: dict[tuple[str, ...], dict[str, Any]] = {}
        for model in filtered_models:
            if str(model.get("group_dim")) != "all":
                continue
            scope = {column: str(model.get(column)) for column in GROUP_COLUMNS}
            all_scopes[tuple(scope[column] for column in GROUP_COLUMNS)] = scope

        for scope in all_scopes.values():
            scope_models = [row for row in filtered_models if _same_scope(row, scope)]
            scope_models.sort(
                key=lambda row: (
                    _model_status(row, cfg) == "significant",
                    float(row.get("adj_r_squared") or -1.0),
                ),
                reverse=True,
            )
            significant_models = [row for row in scope_models if _model_status(row, cfg) == "significant"]
            representative_models = significant_models or scope_models
            top_models = [_round_record(row) for row in representative_models[: cfg["max_models_in_context"]]]
            representative_y_features = [str(row["y_feature"]) for row in representative_models[: cfg["max_models_in_context"]]]
            primary = representative_models[0] if representative_models else None
            contexts.append(
                {
                    **scope,
                    "analysis_mode": "all_relationship",
                    "insight_level": "all_overview",
                    "y_feature": OVERVIEW_Y_FEATURE,
                    "analysis_status": _model_status(primary, cfg),
                    "r_squared": primary.get("r_squared") if primary else None,
                    "adj_r_squared": primary.get("adj_r_squared") if primary else None,
                    "prob_f": primary.get("prob_f") if primary else None,
                    "y_obs": primary.get("y_obs") if primary else None,
                    "model_count": len(scope_models),
                    "significant_model_count": len(significant_models),
                    "significant_driver_count": len(_top_drivers(filtered_coefs, scope, None, 9999, cfg)),
                    "top_models": top_models,
                    "top_drivers": _common_drivers(
                        filtered_coefs,
                        scope,
                        representative_y_features,
                        cfg["max_drivers_in_context"],
                        cfg,
                    ),
                    "top_correlations": _top_correlations(
                        filtered_corrs, scope, None, cfg["max_correlations_in_context"], cfg
                    ),
                    "condition_comparison": [],
                    **_card_titles("all_relationship", scope["group_dim"]),
                }
            )

    if include_y_feature_insights:
        dimension_scopes: dict[tuple[str, str, str], dict[str, Any]] = {}
        for model in filtered_models:
            if str(model.get("group_dim")) == "all":
                continue
            scope = {
                "segment_col": str(model.get("segment_col")),
                "segment_value": str(model.get("segment_value")),
                "group_dim": str(model.get("group_dim")),
            }
            dimension_scopes[tuple(scope.values())] = scope

        for scope in dimension_scopes.values():
            scope_models = [row for row in filtered_models if _same_group_dimension_scope(row, scope)]
            y_features = sorted({str(row["y_feature"]) for row in scope_models})
            for y_feature in y_features:
                y_models = [row for row in scope_models if str(row.get("y_feature")) == y_feature]
                y_models.sort(
                    key=lambda row: (
                        _model_status(row, cfg) == "significant",
                        float(row.get("adj_r_squared") or -1.0),
                    ),
                    reverse=True,
                )
                significant_models = [row for row in y_models if _model_status(row, cfg) == "significant"]
                primary = (significant_models or y_models or [None])[0]
                comparison_models = [
                    _round_record({**row, "analysis_status": _model_status(row, cfg)})
                    for row in y_models[: cfg["max_group_keys_in_context"]]
                ]
                group_drivers = _group_driver_summary(filtered_coefs, scope, y_feature, cfg)
                contexts.append(
                    {
                        **scope,
                        "group_key": "__ALL_GROUP_KEYS__",
                        "analysis_mode": "group_comparison",
                        "insight_level": "group_comparison",
                        "y_feature": y_feature,
                        "analysis_status": _model_status(primary, cfg),
                        "r_squared": primary.get("r_squared") if primary else None,
                        "adj_r_squared": primary.get("adj_r_squared") if primary else None,
                        "prob_f": primary.get("prob_f") if primary else None,
                        "y_obs": primary.get("y_obs") if primary else None,
                        "model_count": len(y_models),
                        "significant_model_count": len(significant_models),
                        "significant_driver_count": len(group_drivers),
                        "top_models": comparison_models,
                        "top_drivers": group_drivers,
                        "top_correlations": _group_top_correlations(
                            filtered_corrs, scope, y_feature, cfg
                        ),
                        "condition_comparison": comparison_models,
                        **_card_titles("group_comparison", scope["group_dim"]),
                    }
                )
    return contexts


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
            representative_y_features = [str(row["y_feature"]) for row in scope_models[: cfg["max_models_in_context"]]]
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
                    "significant_driver_count": len(_top_drivers(coefs, scope, None, 9999, cfg)),
                    "top_models": top_models,
                    "top_drivers": _common_drivers(
                        coefs,
                        scope,
                        representative_y_features,
                        cfg["max_drivers_in_context"],
                        cfg,
                    ),
                    "top_correlations": _top_correlations(
                        corrs, scope, None, cfg["max_correlations_in_context"], cfg
                    ),
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
                        "significant_driver_count": len(_top_drivers(coefs, scope, y_feature, 9999, cfg)),
                        "top_models": [_round_record(model)],
                        "top_drivers": _top_drivers(
                            coefs, scope, y_feature, cfg["max_drivers_in_context"], cfg
                        ),
                        "top_correlations": _top_correlations(
                            corrs, scope, y_feature, cfg["max_correlations_in_context"], cfg
                        ),
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
    contexts = _build_dashboard_contexts(
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
            "analysis_mode": context["analysis_mode"],
            "insight_level": context["insight_level"],
            "y_feature": context["y_feature"],
            "analysis_status": context["analysis_status"],
            "card_titles": {
                key: context[key]
                for key in ["confidence_title", "summary_title", "driver_title", "detail_title"]
            },
            "model_metrics": {
                key: context[key]
                for key in [
                    "r_squared",
                    "adj_r_squared",
                    "prob_f",
                    "y_obs",
                    "model_count",
                    "significant_model_count",
                    "significant_driver_count",
                ]
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
                "analysis_mode": context["analysis_mode"],
                "confidence_title": context["confidence_title"],
                "summary_title": context["summary_title"],
                "driver_title": context["driver_title"],
                "detail_title": context["detail_title"],
                "r_squared": context["r_squared"],
                "adj_r_squared": context["adj_r_squared"],
                "prob_f": context["prob_f"],
                "y_obs": context["y_obs"],
                "significant_driver_count": context["significant_driver_count"],
                "top_models_json": _json(context["top_models"]),
                "top_drivers_json": _json(context["top_drivers"]),
                "top_correlations_json": _json(context["top_correlations"]),
                "condition_comparison_json": _json(context["condition_comparison"]),
                "confidence_summary": insight["confidence_summary"],
                **insight,
                # Kept for existing dashboard consumers; the new driver card
                # should use driver_summary directly.
                "condition_insight": insight["driver_summary"],
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
