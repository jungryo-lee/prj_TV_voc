"""Create staged, Tableau-ready AI cards from weighted-correlation and WLS outputs.

Statistical judgement is deterministic: code sets analysis confidence and drivers
before the LLM writes the core summary and the planning implication.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from typing import Any

from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType, TimestampType

from common.config_loader import get_output_table
from common.llm_client import get_llm_client


INSIGHT_SCHEMA = StructType([
    StructField("group_dim", StringType(), False),
    StructField("y_feature", StringType(), False),
    StructField("group_key", StringType(), False),
    StructField("analysis_route", StringType(), False),
    StructField("analysis_confidence", StringType(), False),
    StructField("core_summary", StringType(), False),
    StructField("implication", StringType(), False),
    StructField("source_hash", StringType(), False),
    StructField("model_key", StringType(), False),
    StructField("prompt_version", StringType(), False),
    StructField("created_at", TimestampType(), False),
])

DRIVER_SCHEMA = StructType([
    StructField("group_dim", StringType(), False),
    StructField("y_feature", StringType(), False),
    StructField("driver_type", StringType(), False),
    StructField("group_key", StringType(), False),
    StructField("analysis_route", StringType(), False),
    StructField("driver", StringType(), False),
    StructField("source_hash", StringType(), False),
    StructField("created_at", TimestampType(), False),
])

GENERATION_LOG_SCHEMA = StructType([
    StructField("group_dim", StringType(), False),
    StructField("y_feature", StringType(), False),
    StructField("group_key", StringType(), False),
    StructField("source_hash", StringType(), False),
    StructField("generation_status", StringType(), False),
    StructField("error_message", StringType(), True),
    StructField("model_key", StringType(), False),
    StructField("prompt_version", StringType(), False),
    StructField("attempted_at", TimestampType(), False),
])


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    values = config.get("driver_dashboard_cards", {}) or {}
    return {
        "model_key": values.get("model_key", "sonnet_46"),
        "prompt_version": values.get("prompt_version", "v2_staged_driver_dashboard_cards"),
        "min_y_obs": int(values.get("min_y_obs", 30)),
        "min_adj_r_squared": float(values.get("min_adj_r_squared", 0.30)),
        "max_model_p_value": float(values.get("max_model_p_value", 0.05)),
        "max_condition_number": float(values.get("max_condition_number", 30)),
        "min_abs_coef": float(values.get("min_abs_coef", 0.10)),
        "max_coef_p_value": float(values.get("max_coef_p_value", 0.05)),
        "min_abs_weighted_corr": float(values.get("min_abs_weighted_corr", 0.30)),
        "max_items": int(values.get("max_items_per_card", 3)),
        "reuse_unchanged": bool(values.get("reuse_unchanged", True)),
    }


def _records(spark: SparkSession, table: str) -> list[dict[str, Any]]:
    return [row.asDict(recursive=True) for row in spark.table(table).collect()]


def _scope(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return tuple(str(row.get(key)) for key in ["segment_col", "segment_value", "group_dim", "group_key", "y_feature"])


def _digest(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _float(value: Any, default: float = 0.0) -> float:
    return default if value is None else float(value)


def _valid_model(model: dict[str, Any], cfg: dict[str, Any]) -> bool:
    return (
        _float(model.get("y_obs")) >= cfg["min_y_obs"]
        and _float(model.get("adj_r_squared")) >= cfg["min_adj_r_squared"]
        and model.get("prob_f") is not None
        and _float(model.get("prob_f")) <= cfg["max_model_p_value"]
        and (model.get("cond_no") is None or _float(model.get("cond_no")) <= cfg["max_condition_number"])
    )


def _valid_coef(row: dict[str, Any], cfg: dict[str, Any]) -> bool:
    return (
        row.get("x_feature") != "const"
        and abs(_float(row.get("coef"))) >= cfg["min_abs_coef"]
        and row.get("p_value") is not None
        and _float(row.get("p_value")) <= cfg["max_coef_p_value"]
        and abs(_float(row.get("weighted_corr"))) >= cfg["min_abs_weighted_corr"]
    )


def _route(model: dict[str, Any], correlations: list[dict[str, Any]], cfg: dict[str, Any]) -> str:
    if _valid_model(model, cfg):
        return "회귀+상관 활용"
    if any(abs(_float(row.get("weighted_corr"))) >= cfg["min_abs_weighted_corr"] for row in correlations):
        return "상관 기반 탐색"
    return "해석 보류"


def _bullet(items: list[str], fallback: str) -> str:
    return "\n".join(f"• {item}" for item in items) if items else f"• {fallback}"


def _confidence_text(model: dict[str, Any], correlations: list[dict[str, Any]], route: str) -> str:
    y_obs = int(_float(model.get("y_obs")))
    r2 = _float(model.get("adj_r_squared"))
    p_value = _float(model.get("prob_f"))
    cond_no = _float(model.get("cond_no"))
    if route == "회귀+상관 활용":
        return f"회귀 활용 가능: 유효 관측치 {y_obs}건, Adj. R² {r2:.2f}, 모델 p-value {p_value:.3f}, 조건수 {cond_no:.1f}로 기준을 충족함."
    max_corr = max((abs(_float(row.get("weighted_corr"))) for row in correlations), default=0.0)
    if route == "상관 기반 탐색":
        return f"회귀 Driver 확정 보류: 유효 관측치 {y_obs}건, Adj. R² {r2:.2f}, 모델 p-value {p_value:.3f}. 최대 |가중상관| {max_corr:.2f}의 탐색 후보만 활용함."
    return f"해석 보류: 유효 관측치 {y_obs}건과 모델 적합도(Adj. R² {r2:.2f})가 확정 Driver 판단 기준에 미달함."


def _driver_rows(models: list[dict[str, Any]], coefs: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Select common and differentiated drivers by the agreed fixed rules."""
    valid_scopes = {_scope(model) for model in models if _valid_model(model, cfg)}
    by_dimension_y: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    valid_groups: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for model in models:
        if _valid_model(model, cfg):
            segment_col, segment_value, group_dim, group_key, y_feature = _scope(model)
            # All mode is one portfolio card across every Y feature.
            key = (segment_col, segment_value, group_dim, "all" if group_dim == "all" else y_feature)
            valid_groups[key].add(group_key)
            by_dimension_y[key]
    for coef in coefs:
        if _scope(coef) in valid_scopes and _valid_coef(coef, cfg):
            segment_col, segment_value, group_dim, _group_key, y_feature = _scope(coef)
            normalized_y = "all" if group_dim == "all" else y_feature
            by_dimension_y[(segment_col, segment_value, group_dim, normalized_y)].append(coef)

    output: list[dict[str, Any]] = []
    for key, candidates in by_dimension_y.items():
        _segment_col, _segment_value, group_dim, y_feature = key
        groups = sorted(valid_groups[key])
        by_feature: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            by_feature[str(row["x_feature"])].append(row)
        needed = 2 if group_dim == "all" else max(2, math.ceil(len(groups) * 0.6))
        common: dict[str, list[dict[str, Any]]] = {}
        for feature, values in by_feature.items():
            repeated = {str(row["y_feature"] if group_dim == "all" else row["group_key"]) for row in values}
            signs = {1 if _float(row["coef"]) >= 0 else -1 for row in values}
            if len(repeated) >= needed and len(signs) == 1:
                common[feature] = values

        def describe(feature: str, values: list[dict[str, Any]]) -> str:
            avg_coef = sum(_float(row["coef"]) for row in values) / len(values)
            max_corr = max(abs(_float(row.get("weighted_corr"))) for row in values)
            direction = "긍정" if avg_coef >= 0 else "부정"
            return f"{feature}: {direction} 방향, 평균 coef {avg_coef:.2f}, 최대 가중상관 {max_corr:.2f}"

        ranked = sorted(common.items(), key=lambda item: (len(item[1]), abs(sum(_float(row["coef"]) for row in item[1]) / len(item[1]))), reverse=True)
        output.append({
            "group_dim": group_dim, "y_feature": "all" if group_dim == "all" else y_feature,
            "driver_type": "공통", "group_key": "-", "analysis_route": "회귀+상관 활용",
            "driver": _bullet([describe(feature, values) for feature, values in ranked[:cfg["max_items"]]], "유효 회귀 기준의 반복 Driver가 제한적임"),
            "source_hash": _digest({"candidates": candidates, "common": common}), "created_at": datetime.utcnow(),
        })
        if group_dim == "all":
            continue
        for group_key in groups:
            differential = [row for row in candidates if str(row["group_key"]) == group_key and str(row["x_feature"]) not in common]
            differential.sort(key=lambda row: abs(_float(row.get("coef"))), reverse=True)
            items = [
                f"{row['x_feature']}: {'긍정' if _float(row['coef']) >= 0 else '부정'} 방향, coef {_float(row['coef']):.2f}, 가중상관 {_float(row.get('weighted_corr')):.2f}"
                for row in differential[:cfg["max_items"]]
            ]
            output.append({
                "group_dim": group_dim, "y_feature": y_feature, "driver_type": "차별", "group_key": group_key,
                "analysis_route": "회귀+상관 활용", "driver": _bullet(items, "공통 패턴 외 유의한 차별 Driver가 제한적임"),
                "source_hash": _digest(differential), "created_at": datetime.utcnow(),
            })
    return output


def _summary_prompt(payload: dict[str, Any]) -> tuple[str, str]:
    system = """You are a rigorous TV product-planning analyst. Write concise Korean Tableau card text only from supplied evidence.
Never invent statistics, drivers, causality, product facts, or comparisons. Correlation is exploratory association, not causality.
The analysis route, analysis confidence, and driver cards are fixed by code: do not override them.
Do not repeat the group name because Tableau displays it separately. Use up to three Korean bullet lines beginning with '•'.
Return JSON only: {\"core_summary\": \"...\", \"implication\": \"...\"}."""
    user = f"""Create two non-overlapping cards for the selected scope.
Analysis route: {payload['analysis_route']}
Analysis confidence (fixed): {payload['analysis_confidence']}
Common drivers (fixed rules): {payload['common_driver']}
Differentiated drivers (fixed rules): {payload['differentiated_driver']}

Core summary: for 회귀+상관 활용, summarize at most three strongest supported patterns. For 혼합 활용, separate regression-usable patterns from exploratory patterns. For 상관 기반 탐색, state that regression drivers are not confirmed and use correlation only as a validation candidate. For 해석 보류, state that reliable interpretation is not possible.
Implication: give up to two TV product-planning checks and one validation question only when regression is usable. Otherwise request validation or additional data without stating a priority or causal driver.

Raw statistical evidence:\n{json.dumps(payload['evidence'], ensure_ascii=False, default=str)}"""
    return system, user


def _existing_by_hash(spark: SparkSession, table: str) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    if not spark.catalog.tableExists(table):
        return {}
    return {(str(row["group_dim"]), str(row["y_feature"]), str(row["group_key"]), str(row["source_hash"])): row for row in _records(spark, table)}


def _save(spark: SparkSession, rows: list[dict[str, Any]], schema: StructType, table: str) -> int:
    df = spark.createDataFrame(rows, schema=schema)
    count = df.count()
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
    return count


def _append_generation_log(spark: SparkSession, rows: list[dict[str, Any]], table: str) -> int:
    """Persist execution audit rows without making failed text visible in Tableau."""
    if not rows:
        return 0
    df = spark.createDataFrame(rows, schema=GENERATION_LOG_SCHEMA)
    count = df.count()
    df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table)
    return count


def _upsert_insights(spark: SparkSession, rows: list[dict[str, Any]], table: str) -> int:
    """Update only successful scopes so a failed refresh cannot remove prior cards."""
    if not rows:
        return 0
    df = spark.createDataFrame(rows, schema=INSIGHT_SCHEMA)
    count = df.count()
    if not spark.catalog.tableExists(table):
        df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
        return count

    from delta.tables import DeltaTable

    target = DeltaTable.forName(spark, table)
    key_condition = " AND ".join(
        f"target.{key} = source.{key}"
        for key in ("group_dim", "y_feature", "group_key")
    )
    (
        target.alias("target")
        .merge(df.alias("source"), key_condition)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
    return count


def generate_dashboard_cards(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    target_group_dims: list[str] | None = None,
    target_y_features: list[str] | None = None,
    target_group_keys: list[str] | None = None,
    max_profiles: int | None = None,
) -> dict[str, Any]:
    """Create insight and driver marts with deterministic-first, LLM-second processing."""
    cfg = _cfg(config)
    model_info = config["llm"]["models"][cfg["model_key"]]
    models = _records(spark, get_output_table(config, "weighted_regression_model"))
    coefs = _records(spark, get_output_table(config, "weighted_regression"))
    correlations = _records(spark, get_output_table(config, "weighted_corr"))
    insight_table = get_output_table(config, "driver_card_insight")
    driver_table = get_output_table(config, "driver_card_driver")
    generation_log_table = get_output_table(config, "driver_card_generation_log")

    drivers = _driver_rows(models, coefs, cfg)
    driver_lookup = {(row["group_dim"], row["y_feature"], row["driver_type"], row["group_key"]): row["driver"] for row in drivers}
    corr_by_scope: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    coef_by_scope: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in correlations:
        corr_by_scope[_scope(row)].append(row)
    for row in coefs:
        coef_by_scope[_scope(row)].append(row)

    profiles: list[tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], str, list[dict[str, Any]]]] = []
    for model in models:
        if str(model.get("group_dim")) == "all":
            continue
        group_dim, y_feature, group_key = str(model["group_dim"]), str(model["y_feature"]), str(model["group_key"])
        if target_group_dims and group_dim not in target_group_dims:
            continue
        if target_y_features and y_feature not in target_y_features:
            continue
        if target_group_keys and group_key not in target_group_keys:
            continue
        scope = _scope(model)
        peers = [row for row in models if str(row.get("segment_col")) == str(model.get("segment_col")) and str(row.get("segment_value")) == str(model.get("segment_value")) and str(row.get("group_dim")) == group_dim and str(row.get("y_feature")) == y_feature]
        profiles.append((model, corr_by_scope[scope], coef_by_scope[scope], _route(model, corr_by_scope[scope], cfg), peers))

    if not target_group_dims or "all" in target_group_dims:
        all_models = [row for row in models if str(row.get("group_dim")) == "all"]
        if all_models:
            representative = max(all_models, key=lambda row: _float(row.get("adj_r_squared"), -1.0))
            all_corr = [row for row in correlations if str(row.get("group_dim")) == "all"]
            valid_count = sum(_valid_model(row, cfg) for row in all_models)
            route = "회귀+상관 활용" if valid_count == len(all_models) else ("혼합 활용" if valid_count else ("상관 기반 탐색" if all_corr else "해석 보류"))
            profiles.append((representative, all_corr, [row for row in coefs if str(row.get("group_dim")) == "all"], route, all_models))

    profiles.sort(key=lambda item: (str(item[0].get("group_dim")), str(item[0].get("y_feature")), str(item[0].get("group_key"))))
    if max_profiles is not None:
        profiles = profiles[:max_profiles]

    existing = _existing_by_hash(spark, insight_table) if cfg["reuse_unchanged"] else {}
    llm = get_llm_client(config=config, model_key=cfg["model_key"])
    insight_rows: list[dict[str, Any]] = []
    generation_log_rows: list[dict[str, Any]] = []
    reused_count = 0
    failed_count = 0
    for model, scope_corr, scope_coef, route, peers in profiles:
        group_dim = str(model["group_dim"])
        y_feature = "all" if group_dim == "all" else str(model["y_feature"])
        group_key = "all" if group_dim == "all" else str(model["group_key"])
        if route == "혼합 활용":
            valid_count = sum(_valid_model(peer, cfg) for peer in peers)
            confidence = (
                f"혼합 활용: 전체 {len(peers)}개 Y 모델 중 {valid_count}개가 회귀 활용 기준을 충족함. "
                "기준 미달 모델은 상관 기반 탐색 후보로만 해석함."
            )
        else:
            confidence = _confidence_text(model, scope_corr, route)
        common_driver = driver_lookup.get((group_dim, y_feature, "공통", "-"), "• 유효 회귀 기준의 공통 Driver가 제한적임")
        differential_driver = "-" if group_dim == "all" else driver_lookup.get((group_dim, y_feature, "차별", group_key), "• 차별 Driver 없음")
        evidence = {
            "selected_model": model,
            "representative_models": sorted(peers, key=lambda row: _float(row.get("adj_r_squared"), -1.0), reverse=True)[:cfg["max_items"]],
            "top_coefficients": sorted([row for row in scope_coef if _valid_coef(row, cfg)], key=lambda row: abs(_float(row.get("coef"))), reverse=True)[:cfg["max_items"]],
            "top_correlations": sorted(scope_corr, key=lambda row: abs(_float(row.get("weighted_corr"))), reverse=True)[:cfg["max_items"]],
        }
        payload = {"analysis_route": route, "analysis_confidence": confidence, "common_driver": common_driver, "differentiated_driver": differential_driver, "evidence": evidence, "prompt_version": cfg["prompt_version"]}
        source_hash = _digest(payload)
        cached = existing.get((group_dim, y_feature, group_key, source_hash))
        if cached:
            insight_rows.append({field.name: cached[field.name] for field in INSIGHT_SCHEMA.fields})
            reused_count += 1
            continue
        system_prompt, user_prompt = _summary_prompt(payload)
        try:
            response = llm.converse_json(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=900)
            core_summary, implication = str(response["core_summary"]), str(response["implication"])
        except Exception as exc:
            failed_count += 1
            generation_log_rows.append({
                "group_dim": group_dim,
                "y_feature": y_feature,
                "group_key": group_key,
                "source_hash": source_hash,
                "generation_status": "failed",
                "error_message": repr(exc)[:4000],
                "model_key": cfg["model_key"],
                "prompt_version": cfg["prompt_version"],
                "attempted_at": datetime.utcnow(),
            })
            print(f"[driver_dashboard_cards] LLM failed; retry on next run | {group_dim} / {y_feature} / {group_key} | {exc!r}")
            continue
        generation_log_rows.append({
            "group_dim": group_dim,
            "y_feature": y_feature,
            "group_key": group_key,
            "source_hash": source_hash,
            "generation_status": "success",
            "error_message": None,
            "model_key": cfg["model_key"],
            "prompt_version": cfg["prompt_version"],
            "attempted_at": datetime.utcnow(),
        })
        insight_rows.append({
            "group_dim": group_dim, "y_feature": y_feature, "group_key": group_key, "analysis_route": route,
            "analysis_confidence": confidence, "core_summary": core_summary, "implication": implication,
            "source_hash": source_hash, "model_key": cfg["model_key"], "prompt_version": cfg["prompt_version"], "created_at": datetime.utcnow(),
        })

    tables = {"insight": insight_table, "driver": driver_table, "generation_log": generation_log_table}
    counts = {
        "insight_upserted": _upsert_insights(spark, insight_rows, insight_table),
        "driver": _save(spark, drivers, DRIVER_SCHEMA, driver_table),
        "generation_log": _append_generation_log(spark, generation_log_rows, generation_log_table),
    }
    return {
        "tables": tables,
        "counts": counts,
        "profile_count": len(profiles),
        "reused_unchanged_insight_count": reused_count,
        "llm_attempted_insight_count": len(profiles) - reused_count,
        "llm_generated_success_count": len(profiles) - reused_count - failed_count,
        "llm_failed_insight_count": failed_count,
        "model_endpoint": model_info["endpoint"],
    }
