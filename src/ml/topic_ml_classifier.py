"""Embedding/prototype based topic classification for unclassified VOC memos."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from common.config_loader import get_output_table
from ml.topic_prototype import classify_by_topic_prototype
from taxonomy.topic_classifier import apply_llm_fallback, rescue_others_from_match_reason


ML_CLASSIFICATION_SCHEMA = T.StructType(
    [
        T.StructField("memo_id", T.StringType(), False),
        T.StructField("memo", T.StringType(), True),
        T.StructField("memo_norm", T.StringType(), True),
        T.StructField("cate_1_depth", T.StringType(), True),
        T.StructField("cate_2_depth", T.StringType(), True),
        T.StructField("sc_measurement", T.IntegerType(), True),
        T.StructField("pred_topic", T.StringType(), True),
        T.StructField("pred_topic_type", T.StringType(), True),
        T.StructField("classification_stage", T.StringType(), True),
        T.StructField("confidence_score", T.DoubleType(), True),
        T.StructField("review_needed_yn", T.BooleanType(), True),
        T.StructField("llm_used_yn", T.BooleanType(), True),
        T.StructField("match_reason", T.StringType(), True),
        T.StructField("candidate_topics_json", T.StringType(), True),
        T.StructField("fallback_model_key", T.StringType(), True),
        T.StructField("fallback_model_version", T.StringType(), True),
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


FALLBACK_QUEUE_SCHEMA = T.StructType(
    [
        T.StructField("memo_id", T.StringType(), False),
        T.StructField("memo", T.StringType(), True),
        T.StructField("memo_norm", T.StringType(), True),
        T.StructField("cate_1_depth", T.StringType(), True),
        T.StructField("cate_2_depth", T.StringType(), True),
        T.StructField("sc_measurement", T.IntegerType(), True),
        T.StructField("prototype_topic", T.StringType(), True),
        T.StructField("prototype_topic_type", T.StringType(), True),
        T.StructField("prototype_confidence_score", T.DoubleType(), True),
        T.StructField("candidate_topics_json", T.StringType(), True),
        T.StructField("fallback_reason", T.StringType(), True),
        T.StructField("fallback_model_key", T.StringType(), True),
        T.StructField("fallback_model_version", T.StringType(), True),
        T.StructField("status", T.StringType(), True),
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


GROUP_COLS = ["cate_1_depth", "cate_2_depth", "sc_measurement"]


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    """Return ML classification settings with safe defaults."""
    cfg = config.get("ml_classification", {}) or {}
    return {
        "query_embedding_table_key": cfg.get(
            "query_embedding_table_key", "memo_embedding_unclassified"
        ),
        "prototype_table_key": cfg.get("prototype_table_key", "topic_prototype"),
        "output_table_key": cfg.get("output_table_key", "ml_classification_detail"),
        "llm_fallback_queue_table_key": cfg.get(
            "llm_fallback_queue_table_key", "llm_fallback_queue"
        ),
        "embedding_model": cfg.get("embedding_model", "databricks-bge-large-en"),
        "fallback_model_key": cfg.get("fallback_model_key", "gpt_mini"),
        "auto_accept_threshold": float(cfg.get("auto_accept_threshold", 0.80)),
        "llm_fallback_threshold": float(cfg.get("llm_fallback_threshold", 0.0)),
        "top_k": int((config.get("topic_prototype", {}) or {}).get("top_k", 3)),
    }


def _version_value(config: dict[str, Any], key: str, default: str = "") -> str:
    """Return version metadata as string."""
    return str((config.get("version", {}) or {}).get(key, default) or default)


def _runtime_value(config: dict[str, Any], key: str, default: str = "") -> str:
    """Return runtime metadata as string."""
    return str((config.get("runtime", {}) or {}).get(key, default) or default)


def _classification_model_version(config: dict[str, Any]) -> str:
    """Resolve the LLM model version used by existing sample classifications."""
    app_model_key = (config.get("app", {}) or {}).get("model_key", "gpt_55")
    return str(
        ((config.get("llm", {}) or {}).get("models", {}) or {})
        .get(app_model_key, {})
        .get("model_version", _version_value(config, "model_version"))
    )


def _fallback_model_version(config: dict[str, Any], model_key: str) -> str:
    """Resolve fallback LLM model version."""
    return str(
        ((config.get("llm", {}) or {}).get("models", {}) or {})
        .get(model_key, {})
        .get("model_version", model_key)
    )


def _safe_json_loads(value: Any, default: Any) -> Any:
    """Parse JSON while tolerating null or malformed values."""
    if value is None:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _table_exists(spark: SparkSession, table_name: str) -> bool:
    """Return whether a Spark table exists."""
    try:
        spark.table(table_name).limit(1).count()
        return True
    except Exception:
        return False


def load_query_embedding_df(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    input_table_key: str | None = None,
    embedding_model: str | None = None,
) -> DataFrame:
    """Load unclassified memo embeddings to classify."""
    cfg = _cfg(config)
    table_name = get_output_table(config, input_table_key or cfg["query_embedding_table_key"])
    resolved_embedding_model = embedding_model or cfg["embedding_model"]
    return (
        spark.table(table_name)
        .where(F.col("embedding_model") == resolved_embedding_model)
        .where(F.col("prompt_version") == _version_value(config, "prompt_version"))
        .where(F.col("taxonomy_version") == _version_value(config, "taxonomy_version"))
        .where(F.col("embedding").isNotNull())
    )


def _load_latest_rule_profile(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
) -> dict[str, Any]:
    """Load the latest rule profile for one group."""
    table_name = get_output_table(config, "rule_profile")
    model_key = (config.get("app", {}) or {}).get("model_key", "gpt_55")
    rows = (
        spark.table(table_name)
        .where(F.col("cate_1_depth") == cate_1_depth)
        .where(F.col("cate_2_depth") == cate_2_depth)
        .where(F.col("sc_measurement") == int(sc_measurement))
        .where(F.col("prompt_version") == _version_value(config, "prompt_version"))
        .where(F.col("taxonomy_version") == _version_value(config, "taxonomy_version"))
        .where(F.col("model_version") == model_key)
        .orderBy(F.col("created_at").desc_nulls_last())
        .limit(1)
        .collect()
    )
    if not rows:
        raise ValueError(
            "rule_profile not found for "
            f"{cate_1_depth} / {cate_2_depth} / {sc_measurement}"
        )

    row = rows[0].asDict(recursive=True)
    return {
        "overall_topic_name": row.get("overall_topic_name"),
        "overall_allowed_rule": row.get("overall_allowed_rule"),
        "overall_block_rule": row.get("overall_block_rule"),
        "overall_sentiment_terms": _safe_json_loads(
            row.get("overall_sentiment_terms_json"), []
        ),
        "feature_hint_terms": _safe_json_loads(row.get("feature_hint_terms_json"), []),
        "reason_signal_terms": _safe_json_loads(row.get("reason_signal_terms_json"), []),
        "non_overall_examples": _safe_json_loads(
            row.get("non_overall_examples_json"), []
        ),
    }


def _load_latest_topic_pool(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
) -> dict[str, Any]:
    """Load the latest topic pool rows for one group."""
    table_name = get_output_table(config, "topic_pool")
    model_key = (config.get("app", {}) or {}).get("model_key", "gpt_55")
    rows = (
        spark.table(table_name)
        .where(F.col("cate_1_depth") == cate_1_depth)
        .where(F.col("cate_2_depth") == cate_2_depth)
        .where(F.col("sc_measurement") == int(sc_measurement))
        .where(F.col("prompt_version") == _version_value(config, "prompt_version"))
        .where(F.col("taxonomy_version") == _version_value(config, "taxonomy_version"))
        .where(F.col("model_version") == model_key)
        .orderBy(F.col("created_at").desc_nulls_last(), F.col("topic_order").asc())
        .collect()
    )
    topics = []
    for row in rows:
        item = row.asDict(recursive=True)
        topics.append(
            {
                "topic": item.get("topic"),
                "description": item.get("description"),
                "representative_memos": _safe_json_loads(
                    item.get("representative_memos_json"), []
                ),
            }
        )
    if not topics:
        raise ValueError(
            "topic_pool not found for "
            f"{cate_1_depth} / {cate_2_depth} / {sc_measurement}"
        )
    return {"topics": topics}


def _candidate_topics_from_json(value: Any) -> list[dict[str, Any]]:
    """Convert prototype candidate JSON into LLM shortlist hints."""
    candidates = _safe_json_loads(value, [])
    rows: list[dict[str, Any]] = []
    if not isinstance(candidates, list):
        return rows
    for item in candidates:
        if not isinstance(item, dict):
            continue
        topic = item.get("prototype_topic") or item.get("topic")
        if not topic:
            continue
        rows.append(
            {
                "topic": topic,
                "pred_topic": topic,
                "pred_topic_type": item.get("prototype_topic_type"),
                "score": item.get("similarity_score"),
                "match_reason": "prototype_candidate",
            }
        )
    return rows


def load_topic_prototype_df(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    prototype_table_key: str | None = None,
    embedding_model: str | None = None,
) -> DataFrame:
    """Load topic prototypes for the current taxonomy version."""
    cfg = _cfg(config)
    table_name = get_output_table(config, prototype_table_key or cfg["prototype_table_key"])
    resolved_embedding_model = embedding_model or cfg["embedding_model"]
    return (
        spark.table(table_name)
        .where(F.col("embedding_model") == resolved_embedding_model)
        .where(F.col("prompt_version") == _version_value(config, "prompt_version"))
        .where(F.col("taxonomy_version") == _version_value(config, "taxonomy_version"))
        .where(F.col("model_version") == _classification_model_version(config))
        .where(F.col("prototype_embedding").isNotNull())
    )


def filter_existing_ml_classification(
    spark: SparkSession,
    config: dict[str, Any],
    query_df: DataFrame,
    *,
    output_table_key: str | None = None,
) -> DataFrame:
    """Skip query embeddings already classified by this ML stage."""
    cfg = _cfg(config)
    table_name = get_output_table(config, output_table_key or cfg["output_table_key"])
    if not _table_exists(spark, table_name):
        return query_df

    existing_keys = (
        spark.table(table_name)
        .where(F.col("prompt_version") == _version_value(config, "prompt_version"))
        .where(F.col("taxonomy_version") == _version_value(config, "taxonomy_version"))
        .where(F.col("model_version") == _classification_model_version(config))
        .select(*(GROUP_COLS + ["memo_id"]))
        .dropDuplicates()
    )
    return query_df.join(existing_keys, on=GROUP_COLS + ["memo_id"], how="left_anti")


def build_ml_classification_df(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    query_embedding_df: DataFrame | None = None,
    prototype_df: DataFrame | None = None,
    skip_existing: bool = True,
    limit_rows: int | None = None,
    created_by: str = "topic_ml_classifier",
) -> DataFrame:
    """Classify unclassified memo embeddings by nearest topic prototype."""
    cfg = _cfg(config)
    query_df = (
        query_embedding_df
        if query_embedding_df is not None
        else load_query_embedding_df(spark, config)
    )
    if skip_existing:
        query_df = filter_existing_ml_classification(spark, config, query_df)
    if limit_rows is not None:
        query_df = query_df.limit(int(limit_rows))

    proto_df = (
        prototype_df
        if prototype_df is not None
        else load_topic_prototype_df(spark, config)
    )
    routed_df = classify_by_topic_prototype(
        query_df,
        proto_df,
        top_k=cfg["top_k"],
        auto_accept_threshold=cfg["auto_accept_threshold"],
        llm_fallback_threshold=cfg["llm_fallback_threshold"],
    )

    fallback_model_key = cfg["fallback_model_key"]
    fallback_model_version = _fallback_model_version(config, fallback_model_key)
    created_at = datetime.utcnow().isoformat(timespec="seconds")
    run_id = _runtime_value(config, "resolved_run_id")
    run_date = _runtime_value(config, "resolved_run_date")
    prompt_version = _version_value(config, "prompt_version")
    taxonomy_version = _version_value(config, "taxonomy_version")
    model_version = _classification_model_version(config)
    pipeline_version = _version_value(config, "pipeline_version")

    accepted = F.col("prototype_route") == "auto_accept"
    return routed_df.select(
        F.col("memo_id").cast("string"),
        F.col("memo").cast("string"),
        F.col("memo_norm").cast("string"),
        F.col("cate_1_depth").cast("string"),
        F.col("cate_2_depth").cast("string"),
        F.col("sc_measurement").cast("int"),
        F.when(accepted, F.col("prototype_topic")).otherwise(F.lit("LLM_FALLBACK_REQUIRED"))
        .cast("string")
        .alias("pred_topic"),
        F.when(accepted, F.col("prototype_topic_type")).otherwise(F.lit("llm_fallback"))
        .cast("string")
        .alias("pred_topic_type"),
        F.when(accepted, F.lit("embedding_prototype_auto_accept"))
        .otherwise(F.lit("embedding_prototype_llm_fallback"))
        .cast("string")
        .alias("classification_stage"),
        F.col("similarity_score").cast("double").alias("confidence_score"),
        (~accepted).cast("boolean").alias("review_needed_yn"),
        F.lit(False).cast("boolean").alias("llm_used_yn"),
        F.concat(
            F.lit("prototype_similarity="),
            F.round(F.col("similarity_score"), 4).cast("string"),
            F.lit("; threshold="),
            F.lit(str(cfg["auto_accept_threshold"])),
        ).alias("match_reason"),
        F.col("prototype_candidates_json").cast("string").alias("candidate_topics_json"),
        F.lit(fallback_model_key).cast("string").alias("fallback_model_key"),
        F.lit(fallback_model_version).cast("string").alias("fallback_model_version"),
        F.lit(run_id).cast("string").alias("run_id"),
        F.lit(run_date).cast("string").alias("run_date"),
        F.lit(prompt_version).cast("string").alias("prompt_version"),
        F.lit(taxonomy_version).cast("string").alias("taxonomy_version"),
        F.lit(model_version).cast("string").alias("model_version"),
        F.lit(pipeline_version).cast("string").alias("pipeline_version"),
        F.lit(created_at).cast("string").alias("created_at"),
        F.lit(created_by).cast("string").alias("created_by"),
    )


def build_llm_fallback_queue_df(
    ml_classification_df: DataFrame,
    config: dict[str, Any],
    *,
    created_by: str = "topic_ml_classifier",
) -> DataFrame:
    """Build a queue of low-confidence rows to classify with GPT mini later."""
    cfg = _cfg(config)
    fallback_model_key = cfg["fallback_model_key"]
    fallback_model_version = _fallback_model_version(config, fallback_model_key)
    created_at = datetime.utcnow().isoformat(timespec="seconds")

    return (
        ml_classification_df.where(
            F.col("classification_stage") == "embedding_prototype_llm_fallback"
        )
        .select(
            F.col("memo_id").cast("string"),
            F.col("memo").cast("string"),
            F.col("memo_norm").cast("string"),
            F.col("cate_1_depth").cast("string"),
            F.col("cate_2_depth").cast("string"),
            F.col("sc_measurement").cast("int"),
            F.lit(None).cast("string").alias("prototype_topic"),
            F.lit(None).cast("string").alias("prototype_topic_type"),
            F.col("confidence_score").cast("double").alias("prototype_confidence_score"),
            F.col("candidate_topics_json").cast("string"),
            F.lit("prototype_confidence_below_threshold").cast("string").alias(
                "fallback_reason"
            ),
            F.lit(fallback_model_key).cast("string").alias("fallback_model_key"),
            F.lit(fallback_model_version).cast("string").alias("fallback_model_version"),
            F.lit("pending").cast("string").alias("status"),
            F.col("run_id").cast("string"),
            F.col("run_date").cast("string"),
            F.col("prompt_version").cast("string"),
            F.col("taxonomy_version").cast("string"),
            F.col("model_version").cast("string"),
            F.col("pipeline_version").cast("string"),
            F.lit(created_at).cast("string").alias("created_at"),
            F.lit(created_by).cast("string").alias("created_by"),
        )
    )


def save_ml_classification(
    ml_df: DataFrame,
    config: dict[str, Any],
    *,
    output_table_key: str | None = None,
    mode: str = "append",
) -> str:
    """Save ML/prototype classification output."""
    cfg = _cfg(config)
    table_name = get_output_table(config, output_table_key or cfg["output_table_key"])
    (
        ml_df.select([field.name for field in ML_CLASSIFICATION_SCHEMA.fields])
        .write.format("delta")
        .mode(mode)
        .option("mergeSchema", "true")
        .saveAsTable(table_name)
    )
    return table_name


def save_llm_fallback_queue(
    queue_df: DataFrame,
    config: dict[str, Any],
    *,
    output_table_key: str | None = None,
    mode: str = "append",
) -> str:
    """Save low-confidence rows for controlled GPT mini fallback."""
    cfg = _cfg(config)
    table_name = get_output_table(
        config, output_table_key or cfg["llm_fallback_queue_table_key"]
    )
    (
        queue_df.select([field.name for field in FALLBACK_QUEUE_SCHEMA.fields])
        .write.format("delta")
        .mode(mode)
        .option("mergeSchema", "true")
        .saveAsTable(table_name)
    )
    return table_name


def classify_and_save_unclassified_memos(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    limit_rows: int | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Run prototype classification and save accepted labels plus fallback queue."""
    ml_df = build_ml_classification_df(
        spark,
        config,
        skip_existing=skip_existing,
        limit_rows=limit_rows,
    )
    total_count = ml_df.count()
    auto_accept_count = ml_df.where(
        F.col("classification_stage") == "embedding_prototype_auto_accept"
    ).count()
    fallback_count = total_count - auto_accept_count
    print(
        "[topic_ml_classifier] classified "
        f"rows={total_count} auto_accept={auto_accept_count} fallback={fallback_count}"
    )

    if total_count == 0:
        return {
            "classification_table": get_output_table(
                config, _cfg(config)["output_table_key"]
            ),
            "fallback_queue_table": get_output_table(
                config, _cfg(config)["llm_fallback_queue_table_key"]
            ),
            "classification_count": 0,
            "auto_accept_count": 0,
            "fallback_count": 0,
            "saved": False,
        }

    classification_table = save_ml_classification(ml_df, config)
    queue_df = build_llm_fallback_queue_df(ml_df, config)
    queue_count = queue_df.count()
    fallback_queue_table = save_llm_fallback_queue(queue_df, config)
    print(
        "[topic_ml_classifier] saved "
        f"classification_table={classification_table} fallback_queue_rows={queue_count}"
    )

    return {
        "classification_table": classification_table,
        "fallback_queue_table": fallback_queue_table,
        "classification_count": total_count,
        "auto_accept_count": auto_accept_count,
        "fallback_count": fallback_count,
        "saved": True,
    }


def load_pending_llm_fallback_queue(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    queue_table_key: str | None = None,
    limit_rows: int | None = None,
) -> DataFrame:
    """Load pending fallback rows for controlled GPT mini classification."""
    cfg = _cfg(config)
    table_name = get_output_table(
        config, queue_table_key or cfg["llm_fallback_queue_table_key"]
    )
    df = (
        spark.table(table_name)
        .where(F.col("status") == "pending")
        .where(F.col("prompt_version") == _version_value(config, "prompt_version"))
        .where(F.col("taxonomy_version") == _version_value(config, "taxonomy_version"))
        .orderBy(
            F.col("prototype_confidence_score").desc_nulls_last(),
            F.col("cate_1_depth"),
            F.col("cate_2_depth"),
            F.col("sc_measurement"),
            F.col("memo_id"),
        )
    )
    if limit_rows is not None:
        df = df.limit(int(limit_rows))
    return df


def classify_llm_fallback_queue_df(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    queue_df: DataFrame | None = None,
    model_key: str | None = None,
    limit_rows: int | None = None,
    created_by: str = "gpt_mini_fallback",
) -> DataFrame:
    """Classify low-confidence fallback queue rows with GPT mini."""
    cfg = _cfg(config)
    resolved_model_key = model_key or cfg["fallback_model_key"]
    resolved_model_version = _fallback_model_version(config, resolved_model_key)
    source_df = (
        queue_df
        if queue_df is not None
        else load_pending_llm_fallback_queue(spark, config, limit_rows=limit_rows)
    )
    source_rows = [row.asDict(recursive=True) for row in source_df.collect()]
    if not source_rows:
        return spark.createDataFrame([], ML_CLASSIFICATION_SCHEMA)

    prompt_version = _version_value(config, "prompt_version")
    taxonomy_version = _version_value(config, "taxonomy_version")
    model_version = _classification_model_version(config)
    pipeline_version = _version_value(config, "pipeline_version")
    run_id = _runtime_value(config, "resolved_run_id")
    run_date = _runtime_value(config, "resolved_run_date")
    created_at = datetime.utcnow().isoformat(timespec="seconds")

    group_cache: dict[tuple[str, str, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    output_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(source_rows, start=1):
        cate_1_depth = str(row.get("cate_1_depth") or "")
        cate_2_depth = str(row.get("cate_2_depth") or "")
        sc_measurement = int(row.get("sc_measurement") or 0)
        group_key = (cate_1_depth, cate_2_depth, sc_measurement)

        if group_key not in group_cache:
            group_cache[group_key] = (
                _load_latest_rule_profile(
                    spark,
                    config,
                    cate_1_depth=cate_1_depth,
                    cate_2_depth=cate_2_depth,
                    sc_measurement=sc_measurement,
                ),
                _load_latest_topic_pool(
                    spark,
                    config,
                    cate_1_depth=cate_1_depth,
                    cate_2_depth=cate_2_depth,
                    sc_measurement=sc_measurement,
                ),
            )

        rule_profile, topic_pool = group_cache[group_key]
        candidate_topics = _candidate_topics_from_json(row.get("candidate_topics_json"))
        print(
            "[gpt_mini_fallback] "
            f"{idx}/{len(source_rows)} | {cate_1_depth} | {cate_2_depth} | {sc_measurement}"
        )

        decision = apply_llm_fallback(
            str(row.get("memo") or row.get("memo_norm") or ""),
            cate_1_depth=cate_1_depth,
            cate_2_depth=cate_2_depth,
            sc_measurement=sc_measurement,
            rule_profile=rule_profile,
            topic_pool=topic_pool,
            candidate_topics=candidate_topics,
            config=config,
            model_key=resolved_model_key,
        )
        decision = rescue_others_from_match_reason(decision, topic_pool=topic_pool)

        output_rows.append(
            {
                "memo_id": str(row.get("memo_id") or ""),
                "memo": row.get("memo"),
                "memo_norm": row.get("memo_norm"),
                "cate_1_depth": cate_1_depth,
                "cate_2_depth": cate_2_depth,
                "sc_measurement": sc_measurement,
                "pred_topic": decision.get("pred_topic"),
                "pred_topic_type": decision.get("pred_topic_type"),
                "classification_stage": "gpt_mini_fallback",
                "confidence_score": None,
                "review_needed_yn": bool(decision.get("review_needed_yn", False)),
                "llm_used_yn": True,
                "match_reason": decision.get("match_reason"),
                "candidate_topics_json": row.get("candidate_topics_json"),
                "fallback_model_key": resolved_model_key,
                "fallback_model_version": resolved_model_version,
                "run_id": str(row.get("run_id") or run_id),
                "run_date": str(row.get("run_date") or run_date),
                "prompt_version": prompt_version,
                "taxonomy_version": taxonomy_version,
                "model_version": model_version,
                "pipeline_version": pipeline_version,
                "created_at": created_at,
                "created_by": created_by,
            }
        )

    return spark.createDataFrame(output_rows, ML_CLASSIFICATION_SCHEMA)


def classify_and_save_llm_fallback_queue(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    model_key: str | None = None,
    limit_rows: int | None = None,
) -> dict[str, Any]:
    """Classify pending fallback queue rows with GPT mini and append results."""
    fallback_df = classify_llm_fallback_queue_df(
        spark,
        config,
        model_key=model_key,
        limit_rows=limit_rows,
    )
    fallback_count = fallback_df.count()
    print(f"[gpt_mini_fallback] classified rows={fallback_count}")

    if fallback_count == 0:
        return {
            "classification_table": get_output_table(
                config, _cfg(config)["output_table_key"]
            ),
            "fallback_count": 0,
            "saved": False,
        }

    table_name = save_ml_classification(fallback_df, config, mode="append")
    return {
        "classification_table": table_name,
        "fallback_count": fallback_count,
        "saved": True,
    }
