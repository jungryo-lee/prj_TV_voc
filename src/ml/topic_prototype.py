"""Topic prototype utilities for embedding-based VOC topic classification."""

from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from common.config_loader import get_output_table


GROUP_COLS = ["cate_1_depth", "cate_2_depth", "sc_measurement"]
VERSION_COLS = ["prompt_version", "taxonomy_version", "model_version"]
TOPIC_COLS = GROUP_COLS + ["pred_topic", "pred_topic_type"]


def _prototype_cfg(config: dict[str, Any]) -> dict[str, Any]:
    """Return topic prototype config with safe defaults."""
    cfg = config.get("topic_prototype", {}) or {}
    return {
        "input_table_key": cfg.get("input_table_key", "memo_embedding"),
        "output_table_key": cfg.get("output_table_key", "topic_prototype"),
        "embedding_model": cfg.get("embedding_model"),
        "min_topic_memo_count": int(cfg.get("min_topic_memo_count", 3)),
        "auto_accept_threshold": float(cfg.get("auto_accept_threshold", 0.82)),
        "llm_fallback_threshold": float(cfg.get("llm_fallback_threshold", 0.70)),
        "top_k": int(cfg.get("top_k", 3)),
    }


def _embedding_model_filter(config: dict[str, Any], embedding_model: str | None = None) -> str | None:
    """Resolve embedding-model filter value."""
    return (
        embedding_model
        or _prototype_cfg(config).get("embedding_model")
        or (config.get("memo_embedding", {}) or {}).get("model_path")
        or (config.get("path", {}) or {}).get("embedding_model")
    )


def load_memo_embedding_df(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    input_table_key: str | None = None,
    embedding_model: str | None = None,
    prompt_version: str | None = None,
    taxonomy_version: str | None = None,
    model_version: str | None = None,
) -> DataFrame:
    """Load memo embedding rows for prototype analysis."""
    cfg = _prototype_cfg(config)
    table_name = get_output_table(config, input_table_key or cfg["input_table_key"])
    df = spark.table(table_name).where(F.col("embedding").isNotNull())

    resolved_embedding_model = _embedding_model_filter(config, embedding_model)
    if resolved_embedding_model:
        df = df.where(F.col("embedding_model") == resolved_embedding_model)

    if prompt_version:
        df = df.where(F.col("prompt_version") == prompt_version)
    if taxonomy_version:
        df = df.where(F.col("taxonomy_version") == taxonomy_version)
    if model_version:
        df = df.where(F.col("model_version") == model_version)

    return df


def summarize_embedding_table(embedding_df: DataFrame) -> dict[str, DataFrame]:
    """Return quality summary DataFrames for memo embedding rows."""
    base_df = embedding_df.withColumn("embedding_dim_actual", F.size("embedding"))

    overall_df = base_df.agg(
        F.count("*").alias("row_cnt"),
        F.countDistinct("memo_id").alias("distinct_memo_id_cnt"),
        F.sum(F.when(F.col("embedding").isNull(), 1).otherwise(0)).alias("null_embedding_cnt"),
        F.countDistinct("embedding_dim_actual").alias("embedding_dim_type_cnt"),
        F.min("embedding_dim_actual").alias("min_embedding_dim"),
        F.max("embedding_dim_actual").alias("max_embedding_dim"),
    )

    topic_df = (
        base_df.groupBy(*TOPIC_COLS)
        .agg(
            F.count("*").alias("memo_cnt"),
            F.countDistinct("memo_id").alias("distinct_memo_id_cnt"),
            F.avg("confidence_score").alias("avg_label_confidence"),
            F.min("embedding_dim_actual").alias("min_embedding_dim"),
            F.max("embedding_dim_actual").alias("max_embedding_dim"),
        )
        .orderBy(F.desc("memo_cnt"))
    )

    duplicate_df = (
        base_df.groupBy(*(GROUP_COLS + VERSION_COLS + ["memo_id"]))
        .agg(F.count("*").alias("row_cnt"))
        .where(F.col("row_cnt") > 1)
        .orderBy(F.desc("row_cnt"))
    )

    sparse_topic_df = topic_df.where(F.col("distinct_memo_id_cnt") < 3)

    return {
        "overall_df": overall_df,
        "topic_df": topic_df,
        "duplicate_df": duplicate_df,
        "sparse_topic_df": sparse_topic_df,
    }


def build_topic_prototype_df(
    embedding_df: DataFrame,
    *,
    min_topic_memo_count: int = 3,
) -> DataFrame:
    """Build one normalized mean embedding per topic group."""
    exploded_df = (
        embedding_df.where(F.col("embedding").isNotNull()).select(
            *(TOPIC_COLS + VERSION_COLS),
            "embedding_model",
            "embedding_dim",
            "memo_id",
            F.posexplode("embedding").alias("embedding_pos", "embedding_value"),
        )
    )

    topic_meta_df = (
        embedding_df.groupBy(*(TOPIC_COLS + VERSION_COLS + ["embedding_model", "embedding_dim"]))
        .agg(
            F.count("*").alias("prototype_memo_cnt"),
            F.countDistinct("memo_id").alias("prototype_distinct_memo_id_cnt"),
            F.avg("confidence_score").alias("avg_label_confidence"),
        )
        .where(F.col("prototype_distinct_memo_id_cnt") >= int(min_topic_memo_count))
    )

    mean_values_df = (
        exploded_df.join(
            topic_meta_df.select(*(TOPIC_COLS + VERSION_COLS + ["embedding_model", "embedding_dim"])),
            on=TOPIC_COLS + VERSION_COLS + ["embedding_model", "embedding_dim"],
            how="inner",
        )
        .groupBy(*(TOPIC_COLS + VERSION_COLS + ["embedding_model", "embedding_dim", "embedding_pos"]))
        .agg(F.avg("embedding_value").alias("mean_embedding_value"))
    )

    prototype_df = (
        mean_values_df.groupBy(*(TOPIC_COLS + VERSION_COLS + ["embedding_model", "embedding_dim"]))
        .agg(
            F.array_sort(
                F.collect_list(
                    F.struct(
                        F.col("embedding_pos"),
                        F.col("mean_embedding_value"),
                    )
                )
            ).alias("_ordered_embedding")
        )
        .withColumn(
            "_raw_prototype_embedding",
            F.expr("transform(_ordered_embedding, x -> cast(x.mean_embedding_value as double))"),
        )
        .withColumn(
            "_prototype_norm",
            F.sqrt(
                F.aggregate(
                    F.col("_raw_prototype_embedding"),
                    F.lit(0.0),
                    lambda acc, x: acc + x * x,
                )
            ),
        )
        .withColumn(
            "prototype_embedding",
            F.expr(
                "transform(_raw_prototype_embedding, x -> cast(x / nullif(_prototype_norm, 0.0) as float))"
            ),
        )
        .drop("_ordered_embedding", "_raw_prototype_embedding", "_prototype_norm")
    )

    return (
        prototype_df.join(
            topic_meta_df,
            on=TOPIC_COLS + VERSION_COLS + ["embedding_model", "embedding_dim"],
            how="inner",
        )
        .withColumn("created_at", F.current_timestamp())
        .withColumn("created_by", F.lit("topic_prototype"))
    )


def save_topic_prototypes(
    prototype_df: DataFrame,
    config: dict[str, Any],
    *,
    output_table_key: str | None = None,
    mode: str = "overwrite",
) -> str:
    """Save topic prototype rows to a Delta table."""
    cfg = _prototype_cfg(config)
    table_name = get_output_table(config, output_table_key or cfg["output_table_key"])
    (
        prototype_df.write.format("delta")
        .mode(mode)
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )
    return table_name


def _similarity_expr(query_col: str = "q.embedding", prototype_col: str = "p.prototype_embedding") -> str:
    """Return SQL expression for dot-product similarity."""
    return (
        f"aggregate(zip_with({query_col}, {prototype_col}, "
        "(x, y) -> cast(x as double) * cast(y as double)), "
        "cast(0.0 as double), (acc, x) -> acc + x)"
    )


def classify_by_topic_prototype(
    query_embedding_df: DataFrame,
    prototype_df: DataFrame,
    *,
    top_k: int = 3,
    auto_accept_threshold: float = 0.82,
    llm_fallback_threshold: float = 0.70,
) -> DataFrame:
    """Classify query embeddings by nearest topic prototype within each group."""
    q = query_embedding_df.alias("q")
    p = prototype_df.alias("p")
    join_cond = (
        (F.col("q.cate_1_depth") == F.col("p.cate_1_depth"))
        & (F.col("q.cate_2_depth") == F.col("p.cate_2_depth"))
        & (F.col("q.sc_measurement") == F.col("p.sc_measurement"))
        & (F.col("q.embedding_model") == F.col("p.embedding_model"))
        & (F.size(F.col("q.embedding")) == F.col("p.embedding_dim"))
    )

    scored_df = (
        q.join(p, join_cond, "inner")
        .select(
            F.col("q.memo_id"),
            F.col("q.memo"),
            F.col("q.memo_norm"),
            F.col("q.cate_1_depth"),
            F.col("q.cate_2_depth"),
            F.col("q.sc_measurement"),
            F.col("q.pred_topic").alias("true_topic"),
            F.col("q.pred_topic_type").alias("true_topic_type"),
            F.col("p.pred_topic").alias("prototype_topic"),
            F.col("p.pred_topic_type").alias("prototype_topic_type"),
            F.col("p.prototype_distinct_memo_id_cnt"),
            F.expr(_similarity_expr()).alias("similarity_score"),
        )
    )

    from pyspark.sql.window import Window

    window = Window.partitionBy(
        "memo_id", "cate_1_depth", "cate_2_depth", "sc_measurement"
    ).orderBy(F.desc("similarity_score"), F.desc("prototype_distinct_memo_id_cnt"))

    ranked_df = scored_df.withColumn("prototype_rank", F.row_number().over(window))

    top1_df = ranked_df.where(F.col("prototype_rank") == 1).withColumn(
        "prototype_route",
        F.when(F.col("similarity_score") >= float(auto_accept_threshold), F.lit("auto_accept"))
        .when(F.col("similarity_score") >= float(llm_fallback_threshold), F.lit("llm_fallback"))
        .otherwise(F.lit("review_required")),
    )

    topk_df = (
        ranked_df.where(F.col("prototype_rank") <= int(top_k))
        .groupBy("memo_id", "cate_1_depth", "cate_2_depth", "sc_measurement")
        .agg(
            F.to_json(
                F.array_sort(
                    F.collect_list(
                        F.struct(
                            F.col("prototype_rank"),
                            F.col("prototype_topic"),
                            F.col("prototype_topic_type"),
                            F.col("similarity_score"),
                        )
                    )
                )
            ).alias("prototype_candidates_json")
        )
    )

    return top1_df.join(
        topk_df,
        on=["memo_id", "cate_1_depth", "cate_2_depth", "sc_measurement"],
        how="left",
    )


def evaluate_prototype_baseline(classified_df: DataFrame) -> dict[str, DataFrame]:
    """Return evaluation tables for prototype predictions with known labels."""
    eval_df = classified_df.withColumn(
        "is_correct_topic",
        F.col("true_topic") == F.col("prototype_topic"),
    )

    overall_df = eval_df.agg(
        F.count("*").alias("eval_cnt"),
        F.avg(F.col("is_correct_topic").cast("double")).alias("top1_accuracy"),
        F.avg("similarity_score").alias("avg_similarity_score"),
        F.sum(F.when(F.col("prototype_route") == "auto_accept", 1).otherwise(0)).alias("auto_accept_cnt"),
        F.sum(F.when(F.col("prototype_route") == "llm_fallback", 1).otherwise(0)).alias("llm_fallback_cnt"),
        F.sum(F.when(F.col("prototype_route") == "review_required", 1).otherwise(0)).alias("review_required_cnt"),
    )

    group_df = (
        eval_df.groupBy(*GROUP_COLS)
        .agg(
            F.count("*").alias("eval_cnt"),
            F.avg(F.col("is_correct_topic").cast("double")).alias("top1_accuracy"),
            F.avg("similarity_score").alias("avg_similarity_score"),
            F.sum(F.when(F.col("prototype_route") == "auto_accept", 1).otherwise(0)).alias("auto_accept_cnt"),
            F.sum(F.when(F.col("prototype_route") == "llm_fallback", 1).otherwise(0)).alias("llm_fallback_cnt"),
            F.sum(F.when(F.col("prototype_route") == "review_required", 1).otherwise(0)).alias("review_required_cnt"),
        )
        .orderBy(F.asc("top1_accuracy"), F.desc("eval_cnt"))
    )

    topic_df = (
        eval_df.groupBy(*(TOPIC_COLS + ["prototype_topic"]))
        .agg(
            F.count("*").alias("eval_cnt"),
            F.avg(F.col("is_correct_topic").cast("double")).alias("top1_accuracy"),
            F.avg("similarity_score").alias("avg_similarity_score"),
        )
        .orderBy(F.asc("top1_accuracy"), F.desc("eval_cnt"))
    )

    return {
        "overall_df": overall_df,
        "group_df": group_df,
        "topic_df": topic_df,
    }


def evaluate_intra_inter_topic_similarity(
    embedding_df: DataFrame,
    *,
    sample_per_topic: int = 20,
) -> DataFrame:
    """Estimate same-topic vs different-topic similarity within each category group."""
    from pyspark.sql.window import Window

    sample_window = Window.partitionBy(*TOPIC_COLS).orderBy(F.rand(seed=42))
    sampled_df = (
        embedding_df.withColumn("_sample_rank", F.row_number().over(sample_window))
        .where(F.col("_sample_rank") <= int(sample_per_topic))
        .drop("_sample_rank")
    )

    left_df = sampled_df.alias("l")
    right_df = sampled_df.alias("r")
    pair_df = (
        left_df.join(
            right_df,
            (F.col("l.cate_1_depth") == F.col("r.cate_1_depth"))
            & (F.col("l.cate_2_depth") == F.col("r.cate_2_depth"))
            & (F.col("l.sc_measurement") == F.col("r.sc_measurement"))
            & (F.col("l.memo_id") < F.col("r.memo_id")),
            "inner",
        )
        .select(
            F.col("l.cate_1_depth").alias("cate_1_depth"),
            F.col("l.cate_2_depth").alias("cate_2_depth"),
            F.col("l.sc_measurement").alias("sc_measurement"),
            (F.col("l.pred_topic") == F.col("r.pred_topic")).alias("same_topic_yn"),
            F.expr(_similarity_expr("l.embedding", "r.embedding")).alias("similarity_score"),
        )
    )

    return (
        pair_df.groupBy(*GROUP_COLS, "same_topic_yn")
        .agg(
            F.count("*").alias("pair_cnt"),
            F.avg("similarity_score").alias("avg_similarity_score"),
            F.expr("percentile_approx(similarity_score, 0.5)").alias("median_similarity_score"),
        )
        .orderBy(*GROUP_COLS, F.desc("same_topic_yn"))
    )
