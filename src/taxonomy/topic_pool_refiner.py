"""Build refinement candidates from others / low-confidence classifications."""

from __future__ import annotations

import json
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from common.config_loader import get_output_table
from taxonomy.others_analyzer import load_classification_input_df
from taxonomy.topic_classifier import build_topic_candidates, normalize_topic_pool


EXISTING_TOPIC_REASSIGNMENT_SCHEMA = T.StructType(
    [
        T.StructField("cate_1_depth", T.StringType(), True),
        T.StructField("cate_2_depth", T.StringType(), True),
        T.StructField("sc_measurement", T.IntegerType(), True),
        T.StructField("memo_id", T.StringType(), True),
        T.StructField("memo", T.StringType(), True),
        T.StructField("memo_norm", T.StringType(), True),
        T.StructField("current_pred_topic", T.StringType(), True),
        T.StructField("current_pred_topic_type", T.StringType(), True),
        T.StructField("suggested_topic", T.StringType(), True),
        T.StructField("suggested_action", T.StringType(), True),
        T.StructField("suggestion_score", T.DoubleType(), True),
        T.StructField("candidate_topics_json", T.StringType(), True),
        T.StructField("suggestion_reason", T.StringType(), True),
        T.StructField("model_version", T.StringType(), True),
        T.StructField("prompt_version", T.StringType(), True),
        T.StructField("taxonomy_version", T.StringType(), True),
        T.StructField("run_id", T.StringType(), True),
        T.StructField("run_date", T.StringType(), True),
    ]
)


def _clean_text(value: Any) -> str:
    """Collapse whitespace for stable matching and output."""
    return " ".join(str(value or "").split()).strip()


def _json_loads_list(value: Any) -> list[str]:
    """Parse a JSON list string into a list of strings."""
    if isinstance(value, list):
        return [_clean_text(item) for item in value if _clean_text(item)]
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except Exception:
        return [_clean_text(value)]
    if not isinstance(parsed, list):
        return [_clean_text(parsed)]
    return [_clean_text(item) for item in parsed if _clean_text(item)]


def _json_dumps(value: Any) -> str:
    """Serialize values as UTF-8 JSON."""
    return json.dumps(value or [], ensure_ascii=False)


def get_topic_pool_refinement_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return effective defaults for topic-pool refinement."""
    classification_cfg = config.get("classification", {})

    return {
        "input_table_key": classification_cfg.get(
            "refinement_input_table_key",
            "classification_full",
        ),
        "max_analysis_memos_per_group": int(
            classification_cfg.get("refinement_max_analysis_memos_per_group", 300)
        ),
        "existing_topic_min_score": float(
            classification_cfg.get("refinement_existing_topic_min_score", 1.4)
        ),
        "existing_topic_min_margin": float(
            classification_cfg.get("refinement_existing_topic_min_margin", 0.25)
        ),
        "new_topic_candidate_min_count": int(
            classification_cfg.get("refinement_new_topic_candidate_min_count", 5)
        ),
        "new_topic_candidate_min_ratio": float(
            classification_cfg.get("refinement_new_topic_candidate_min_ratio", 0.02)
        ),
        "max_candidate_rows": int(
            classification_cfg.get("refinement_max_candidate_rows", 200)
        ),
    }


def load_latest_topic_pool_rows(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
    model_version: str | None = None,
    prompt_version: str | None = None,
    taxonomy_version: str | None = None,
) -> list[dict[str, Any]]:
    """Load latest topic-pool rows as Python dictionaries."""
    topic_pool_table = get_output_table(config, "topic_pool")
    df = spark.table(topic_pool_table)

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
        "topic",
        "model_version",
        "prompt_version",
        "taxonomy_version",
    ).orderBy(
        F.col("is_latest").desc_nulls_last(),
        F.col("created_at").desc_nulls_last(),
        F.col("run_id").desc_nulls_last(),
    )

    rows = (
        df.withColumn("_topic_rn", F.row_number().over(latest_window))
        .where(F.col("_topic_rn") == 1)
        .drop("_topic_rn")
        .orderBy(
            "cate_1_depth",
            "cate_2_depth",
            "sc_measurement",
            "topic_order",
            "topic",
        )
        .collect()
    )

    return [row.asDict(recursive=True) for row in rows]


def build_topic_pool_by_group(topic_pool_rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Convert topic-pool table rows into classifier topic_pool payloads."""
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}

    for row in topic_pool_rows:
        key = (
            _clean_text(row.get("cate_1_depth")),
            _clean_text(row.get("cate_2_depth")),
            int(row.get("sc_measurement")),
        )
        grouped.setdefault(key, []).append(
            {
                "topic": _clean_text(row.get("topic")),
                "description": _clean_text(row.get("description")),
                "representative_memos": _json_loads_list(
                    row.get("representative_memos_json")
                ),
            }
        )

    return {
        key: normalize_topic_pool({"topics": topics})
        for key, topics in grouped.items()
    }


def load_refinement_input_df(
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
    run_id: str | None = None,
) -> DataFrame:
    """Load classification rows used for refinement candidate generation."""
    return load_classification_input_df(
        spark,
        config,
        input_table_key=input_table_key,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        model_version=model_version,
        prompt_version=prompt_version,
        taxonomy_version=taxonomy_version,
        run_id=run_id,
    )


def build_existing_topic_reassignment_df(
    spark: SparkSession,
    detail_df: DataFrame,
    topic_pool_by_group: dict[tuple[str, str, int], dict[str, Any]],
    *,
    min_score: float,
    min_margin: float,
    max_analysis_memos_per_group: int,
) -> DataFrame:
    """Find others rows that look close to an existing topic."""
    group_window = Window.partitionBy(
        "cate_1_depth",
        "cate_2_depth",
        "sc_measurement",
    ).orderBy(F.md5(F.coalesce(F.col("memo_id"), F.lit(""))))

    review_rows = (
        detail_df.where(F.col("pred_topic_type") == "others")
        .dropDuplicates(["cate_1_depth", "cate_2_depth", "sc_measurement", "memo_id"])
        .withColumn("_rn", F.row_number().over(group_window))
        .where(F.col("_rn") <= int(max_analysis_memos_per_group))
        .drop("_rn")
        .collect()
    )

    output_rows: list[dict[str, Any]] = []
    empty_rule_profile = {
        "overall_topic_name": "",
        "feature_hint_terms": [],
        "reason_signal_terms": [],
        "overall_sentiment_terms": [],
    }

    for row in review_rows:
        row_dict = row.asDict(recursive=True)
        key = (
            _clean_text(row_dict.get("cate_1_depth")),
            _clean_text(row_dict.get("cate_2_depth")),
            int(row_dict.get("sc_measurement")),
        )
        topic_pool = topic_pool_by_group.get(key)
        if not topic_pool:
            continue

        candidates = build_topic_candidates(
            _clean_text(row_dict.get("memo")),
            topic_pool,
            empty_rule_profile,
        )
        if not candidates:
            continue

        top1 = candidates[0]
        top1_score = float(top1.get("score", 0.0))
        top2_score = float(candidates[1].get("score", 0.0)) if len(candidates) > 1 else 0.0
        margin = top1_score - top2_score

        if top1_score < float(min_score) or margin < float(min_margin):
            continue

        output_rows.append(
            {
                "cate_1_depth": key[0],
                "cate_2_depth": key[1],
                "sc_measurement": key[2],
                "memo_id": _clean_text(row_dict.get("memo_id")),
                "memo": row_dict.get("memo"),
                "memo_norm": _clean_text(row_dict.get("memo_norm")),
                "current_pred_topic": _clean_text(row_dict.get("pred_topic")),
                "current_pred_topic_type": _clean_text(row_dict.get("pred_topic_type")),
                "suggested_topic": _clean_text(top1.get("topic")),
                "suggested_action": "reassign_existing_topic",
                "suggestion_score": top1_score,
                "candidate_topics_json": _json_dumps(candidates[:5]),
                "suggestion_reason": f"heuristic_topic_score={top1_score:.4f};margin={margin:.4f}",
                "model_version": _clean_text(row_dict.get("model_version")),
                "prompt_version": _clean_text(row_dict.get("prompt_version")),
                "taxonomy_version": _clean_text(row_dict.get("taxonomy_version")),
                "run_id": _clean_text(row_dict.get("run_id")),
                "run_date": _clean_text(row_dict.get("run_date")),
            }
        )

    return spark.createDataFrame(output_rows, schema=EXISTING_TOPIC_REASSIGNMENT_SCHEMA)


def build_new_topic_candidate_df(
    detail_df: DataFrame,
    *,
    candidate_min_count: int,
    candidate_min_ratio: float,
    max_candidate_rows: int,
) -> DataFrame:
    """Surface repeated others patterns as possible new topic candidates."""
    others_df = detail_df.where(F.col("pred_topic_type") == "others")
    group_keys = [
        "cate_1_depth",
        "cate_2_depth",
        "sc_measurement",
        "model_version",
        "prompt_version",
        "taxonomy_version",
        "run_id",
        "run_date",
    ]

    group_total_df = others_df.groupBy(*group_keys).agg(
        F.count("*").alias("others_group_total_cnt")
    )

    return (
        others_df.groupBy(*group_keys, "memo_norm")
        .agg(
            F.count("*").alias("candidate_cnt"),
            F.countDistinct("memo_id").alias("candidate_distinct_memo_id_cnt"),
            F.first("memo").alias("sample_memo"),
            F.collect_set("match_reason").alias("match_reason_samples"),
        )
        .join(group_total_df, on=group_keys, how="left")
        .withColumn(
            "candidate_ratio",
            F.when(
                F.col("others_group_total_cnt") > 0,
                F.col("candidate_cnt") / F.col("others_group_total_cnt"),
            ).otherwise(F.lit(0.0)),
        )
        .where(F.col("candidate_cnt") >= int(candidate_min_count))
        .where(F.col("candidate_ratio") >= float(candidate_min_ratio))
        .withColumn("suggested_action", F.lit("consider_new_topic"))
        .withColumn("candidate_reason", F.lit("repeated_others_pattern"))
        .orderBy(
            F.col("candidate_cnt").desc(),
            F.col("candidate_ratio").desc(),
            F.col("cate_1_depth").asc(),
            F.col("cate_2_depth").asc(),
        )
        .limit(int(max_candidate_rows))
    )


def refine_topic_pool_candidates(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    input_table_key: str | None = None,
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
    model_version: str | None = None,
    prompt_version: str | None = None,
    taxonomy_version: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build topic-pool refinement candidates without mutating saved taxonomy."""
    refinement_cfg = get_topic_pool_refinement_config(config)
    effective_input_table_key = input_table_key or refinement_cfg["input_table_key"]

    detail_df = load_refinement_input_df(
        spark,
        config,
        input_table_key=effective_input_table_key,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        model_version=model_version,
        prompt_version=prompt_version,
        taxonomy_version=taxonomy_version,
        run_id=run_id,
    )
    others_df = detail_df.where(F.col("pred_topic_type") == "others")

    topic_pool_rows = load_latest_topic_pool_rows(
        spark,
        config,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        model_version=model_version,
        prompt_version=prompt_version,
        taxonomy_version=taxonomy_version,
    )
    topic_pool_by_group = build_topic_pool_by_group(topic_pool_rows)

    existing_topic_reassignment_df = build_existing_topic_reassignment_df(
        spark,
        detail_df,
        topic_pool_by_group,
        min_score=refinement_cfg["existing_topic_min_score"],
        min_margin=refinement_cfg["existing_topic_min_margin"],
        max_analysis_memos_per_group=refinement_cfg["max_analysis_memos_per_group"],
    )
    new_topic_candidate_df = build_new_topic_candidate_df(
        detail_df,
        candidate_min_count=refinement_cfg["new_topic_candidate_min_count"],
        candidate_min_ratio=refinement_cfg["new_topic_candidate_min_ratio"],
        max_candidate_rows=refinement_cfg["max_candidate_rows"],
    )

    group_summary_df = (
        detail_df.groupBy("cate_1_depth", "cate_2_depth", "sc_measurement")
        .agg(
            F.count("*").alias("row_cnt"),
            F.countDistinct("memo_id").alias("distinct_memo_id_cnt"),
            F.sum(F.when(F.col("pred_topic_type") == "others", 1).otherwise(0)).alias(
                "others_cnt"
            ),
        )
        .withColumn(
            "others_ratio",
            F.when(F.col("row_cnt") > 0, F.col("others_cnt") / F.col("row_cnt")).otherwise(F.lit(0.0)),
        )
        .orderBy(F.col("others_ratio").desc(), F.col("row_cnt").desc())
    )

    return {
        "detail_df": detail_df,
        "others_df": others_df,
        "group_summary_df": group_summary_df,
        "existing_topic_reassignment_df": existing_topic_reassignment_df,
        "new_topic_candidate_df": new_topic_candidate_df,
        "topic_pool_group_count": len(topic_pool_by_group),
        "refinement_config": refinement_cfg,
        "input_table_key": effective_input_table_key,
    }
