"""Analyze 'others' classification results and surface new-topic candidates."""

from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from common.config_loader import get_output_table


def get_others_analysis_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return effective defaults for others/new-topic analysis."""
    classification_cfg = config.get("classification", {})
    taxonomy_cfg = config.get("taxonomy", {})

    return {
        "candidate_min_count": int(classification_cfg.get("others_candidate_min_count", 5)),
        "candidate_min_ratio": float(
            classification_cfg.get("others_candidate_min_ratio", 0.01)
        ),
        "max_candidate_rows": int(
            classification_cfg.get("others_candidate_max_rows", 200)
        ),
        "review_required_default": bool(
            taxonomy_cfg.get("review_required_default", True)
        ),
    }


def load_classification_detail_df(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
    model_version: str | None = None,
    prompt_version: str | None = None,
    taxonomy_version: str | None = None,
    run_id: str | None = None,
) -> DataFrame:
    """Load classification_detail rows with optional filters."""
    table_name = get_output_table(config, "classification_detail")
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
    if run_id is not None:
        df = df.where(F.col("run_id") == run_id)

    return df


def load_others_detail_df(
    spark: SparkSession,
    config: dict[str, Any],
    **filters: Any,
) -> DataFrame:
    """Load only 'others' rows from classification_detail."""
    return load_classification_detail_df(
        spark,
        config,
        **filters,
    ).where(F.col("pred_topic_type") == "others")


def build_others_group_summary_df(others_df: DataFrame) -> DataFrame:
    """Summarize others volume by taxonomy group."""
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

    total_df = (
        others_df.groupBy(*group_keys)
        .agg(
            F.count("*").alias("others_cnt"),
            F.countDistinct("memo_norm").alias("others_distinct_memo_cnt"),
            F.countDistinct("memo_id").alias("others_distinct_memo_id_cnt"),
            F.sum(F.when(F.col("llm_used_yn") == True, 1).otherwise(0)).alias(
                "others_llm_used_cnt"
            ),
            F.sum(F.when(F.col("review_needed_yn") == True, 1).otherwise(0)).alias(
                "others_review_needed_cnt"
            ),
        )
        .withColumn(
            "others_llm_used_ratio",
            F.when(
                F.col("others_cnt") > 0,
                F.col("others_llm_used_cnt") / F.col("others_cnt"),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "others_review_needed_ratio",
            F.when(
                F.col("others_cnt") > 0,
                F.col("others_review_needed_cnt") / F.col("others_cnt"),
            ).otherwise(F.lit(0.0)),
        )
    )

    return total_df.orderBy(
        F.col("others_cnt").desc(),
        F.col("cate_1_depth").asc(),
        F.col("cate_2_depth").asc(),
    )


def build_new_topic_candidate_df(
    detail_df: DataFrame,
    *,
    candidate_min_count: int,
    candidate_min_ratio: float,
    max_candidate_rows: int,
) -> DataFrame:
    """Build repeated-pattern candidates from others rows.

    Current strategy:
    - work only on pred_topic_type='others'
    - group by memo_norm within each taxonomy group
    - surface repeated patterns above min count / min ratio
    """
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

    group_total_df = (
        others_df.groupBy(*group_keys)
        .agg(F.count("*").alias("others_group_total_cnt"))
    )

    candidate_df = (
        others_df.groupBy(*group_keys, "memo_norm")
        .agg(
            F.count("*").alias("candidate_cnt"),
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
        .withColumn(
            "candidate_reason",
            F.lit("repeated_others_pattern"),
        )
        .orderBy(
            F.col("candidate_cnt").desc(),
            F.col("candidate_ratio").desc(),
            F.col("cate_1_depth").asc(),
            F.col("cate_2_depth").asc(),
        )
        .limit(int(max_candidate_rows))
    )

    return candidate_df


def analyze_others(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
    model_version: str | None = None,
    prompt_version: str | None = None,
    taxonomy_version: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run others analysis and return both summary and candidate DataFrames."""
    analysis_cfg = get_others_analysis_config(config)

    detail_df = load_classification_detail_df(
        spark,
        config,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        model_version=model_version,
        prompt_version=prompt_version,
        taxonomy_version=taxonomy_version,
        run_id=run_id,
    )
    others_df = detail_df.where(F.col("pred_topic_type") == "others")

    others_group_summary_df = build_others_group_summary_df(others_df)
    new_topic_candidate_df = build_new_topic_candidate_df(
        detail_df,
        candidate_min_count=analysis_cfg["candidate_min_count"],
        candidate_min_ratio=analysis_cfg["candidate_min_ratio"],
        max_candidate_rows=analysis_cfg["max_candidate_rows"],
    )

    return {
        "detail_df": detail_df,
        "others_df": others_df,
        "others_group_summary_df": others_group_summary_df,
        "new_topic_candidate_df": new_topic_candidate_df,
        "analysis_config": analysis_cfg,
    }
