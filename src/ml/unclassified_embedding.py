"""Build embeddings for raw memos not yet covered by sample classification."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from common.category_alias import cate_2_alias_sql
from common.config_loader import (
    build_source_filter_sql,
    get_output_table,
    get_source_table,
)
from common.memo_id import with_memo_id
from ml.memo_embedding import MEMO_EMBEDDING_SCHEMA, _align_embedding_schema_for_delta


GROUP_COLS = ["cate_1_depth", "cate_2_depth", "sc_measurement"]
VERSION_COLS = ["prompt_version", "taxonomy_version", "model_version"]


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    """Return ML classification config with stable defaults."""
    ml_cfg = config.get("ml_classification", {}) or {}
    taxonomy_cfg = config.get("taxonomy", {}) or {}
    return {
        "source_table_key": ml_cfg.get("source_table_key", "raw_review_table"),
        "classification_table_key": ml_cfg.get(
            "classification_table_key", "classification_detail"
        ),
        "prototype_table_key": ml_cfg.get("prototype_table_key", "topic_prototype"),
        "query_embedding_table_key": ml_cfg.get(
            "query_embedding_table_key", "memo_embedding_unclassified"
        ),
        "embedding_model": ml_cfg.get("embedding_model", "databricks-bge-large-en"),
        "target_sentiments": ml_cfg.get(
            "target_sentiments", taxonomy_cfg.get("target_sentiments", [1, -1])
        ),
        "exclude_existing_classification": bool(
            ml_cfg.get("exclude_existing_classification", True)
        ),
        "text_col": ml_cfg.get("text_col", "memo_norm"),
        "limit_rows": ml_cfg.get("limit_rows"),
        "limit_rows_per_group": ml_cfg.get("limit_rows_per_group"),
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


def _table_exists(spark: SparkSession, table_name: str) -> bool:
    """Return whether a Spark table exists."""
    try:
        spark.table(table_name).limit(1).count()
        return True
    except Exception:
        return False


def load_completed_topic_groups(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    prototype_table_key: str | None = None,
) -> DataFrame:
    """Load groups that already have topic prototypes and can classify new memos."""
    cfg = _cfg(config)
    table_name = get_output_table(config, prototype_table_key or cfg["prototype_table_key"])
    prompt_version = _version_value(config, "prompt_version")
    taxonomy_version = _version_value(config, "taxonomy_version")
    model_version = _classification_model_version(config)

    return (
        spark.table(table_name)
        .where(F.col("prompt_version") == prompt_version)
        .where(F.col("taxonomy_version") == taxonomy_version)
        .where(F.col("model_version") == model_version)
        .select(*GROUP_COLS)
        .dropDuplicates()
    )


def load_raw_unclassified_memo_df(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    limit_rows: int | None = None,
    limit_rows_per_group: int | None = None,
) -> DataFrame:
    """Load one raw memo per memo_id that is not in existing classification_detail."""
    cfg = _cfg(config)
    source_table_key = cfg["source_table_key"]
    source_table = get_source_table(config, source_table_key)
    source_filter_sql = build_source_filter_sql(config, source_table_key)
    sentiments = ", ".join(str(int(value)) for value in cfg["target_sentiments"])
    cate_2_expr = cate_2_alias_sql(config)

    raw_df = spark.sql(
        f"""
        select
            cate_1_depth,
            {cate_2_expr} as cate_2_depth,
            cast(sc_measurement as int) as sc_measurement,
            memo
        from {source_table}
        where memo is not null
          and length(trim(memo)) > 0
          and sc_measurement in ({sentiments})
          {source_filter_sql}
        """
    ).transform(with_memo_id)

    completed_groups = load_completed_topic_groups(spark, config)
    raw_df = raw_df.join(completed_groups, on=GROUP_COLS, how="inner")

    if cfg["exclude_existing_classification"]:
        classified_table = get_output_table(config, cfg["classification_table_key"])
        classified_keys = (
            spark.table(classified_table)
            .where(F.col("prompt_version") == _version_value(config, "prompt_version"))
            .where(F.col("taxonomy_version") == _version_value(config, "taxonomy_version"))
            .where(F.col("model_version") == _classification_model_version(config))
            .select(*(GROUP_COLS + ["memo_id"]))
            .dropDuplicates()
        )
        raw_df = raw_df.join(classified_keys, on=GROUP_COLS + ["memo_id"], how="left_anti")

    representative_window = Window.partitionBy(*(GROUP_COLS + ["memo_id"])).orderBy(
        F.length(F.col("memo")).asc(),
        F.col("memo").asc(),
    )
    deduped_df = (
        raw_df.withColumn("_rn", F.row_number().over(representative_window))
        .where(F.col("_rn") == 1)
        .drop("_rn")
    )

    effective_limit_per_group = (
        limit_rows_per_group
        if limit_rows_per_group is not None
        else cfg["limit_rows_per_group"]
    )
    if effective_limit_per_group is not None:
        balanced_window = Window.partitionBy(*GROUP_COLS).orderBy(
            F.md5(
                F.concat_ws(
                    "||",
                    F.coalesce(F.col("memo_id"), F.lit("")),
                    F.coalesce(F.col("cate_1_depth"), F.lit("")),
                    F.coalesce(F.col("cate_2_depth"), F.lit("")),
                    F.coalesce(F.col("sc_measurement").cast("string"), F.lit("")),
                    F.lit("ml_unclassified_balanced_sample"),
                )
            )
        )
        deduped_df = (
            deduped_df.withColumn("_group_sample_rn", F.row_number().over(balanced_window))
            .where(F.col("_group_sample_rn") <= int(effective_limit_per_group))
            .drop("_group_sample_rn")
        )

    effective_limit = limit_rows if limit_rows is not None else cfg["limit_rows"]
    if effective_limit is not None:
        deduped_df = deduped_df.limit(int(effective_limit))

    return deduped_df


def filter_existing_query_embeddings(
    spark: SparkSession,
    config: dict[str, Any],
    query_df: DataFrame,
    *,
    output_table_key: str | None = None,
    embedding_model: str | None = None,
) -> DataFrame:
    """Remove raw memo rows that already have query embeddings."""
    cfg = _cfg(config)
    resolved_output_key = output_table_key or cfg["query_embedding_table_key"]
    table_name = get_output_table(config, resolved_output_key)
    resolved_embedding_model = embedding_model or cfg["embedding_model"]

    if not _table_exists(spark, table_name):
        return query_df

    existing_keys = (
        spark.table(table_name)
        .where(F.col("embedding_model") == resolved_embedding_model)
        .where(F.col("prompt_version") == _version_value(config, "prompt_version"))
        .where(F.col("taxonomy_version") == _version_value(config, "taxonomy_version"))
        .select(*(GROUP_COLS + ["memo_id"]))
        .dropDuplicates()
    )
    return query_df.join(existing_keys, on=GROUP_COLS + ["memo_id"], how="left_anti")


def build_query_embedding_df(
    spark: SparkSession,
    config: dict[str, Any],
    query_df: DataFrame,
    *,
    embedding_model: str | None = None,
    created_by: str = "unclassified_embedding",
) -> DataFrame:
    """Attach Databricks AI Query embeddings to raw unclassified memo rows."""
    cfg = _cfg(config)
    resolved_embedding_model = embedding_model or cfg["embedding_model"]
    text_col = cfg["text_col"]
    created_at = datetime.utcnow().isoformat(timespec="seconds")
    run_id = _runtime_value(config, "resolved_run_id")
    run_date = _runtime_value(config, "resolved_run_date")
    prompt_version = _version_value(config, "prompt_version")
    taxonomy_version = _version_value(config, "taxonomy_version")
    model_version = _classification_model_version(config)
    pipeline_version = _version_value(config, "pipeline_version")

    source_df = query_df.withColumn(
        "embedding_text",
        F.coalesce(F.col(text_col).cast("string"), F.col("memo").cast("string")),
    )

    embedded_df = source_df.withColumn(
        "_raw_embedding",
        F.expr(
            f"ai_query('{resolved_embedding_model}', embedding_text, returnType => 'ARRAY<FLOAT>')"
        ),
    ).withColumn(
        "_embedding_norm",
        F.sqrt(
            F.aggregate(
                F.col("_raw_embedding"),
                F.lit(0.0),
                lambda acc, x: acc + x.cast("double") * x.cast("double"),
            )
        ),
    ).withColumn(
        "embedding",
        F.when(
            F.col("_embedding_norm") > F.lit(0.0),
            F.transform(
                F.col("_raw_embedding"),
                lambda x: (x.cast("double") / F.col("_embedding_norm")).cast("float"),
            ),
        ).otherwise(F.col("_raw_embedding")),
    ).withColumn("embedding_dim", F.size("embedding"))

    return embedded_df.select(
        F.col("memo_id").cast("string"),
        F.col("memo").cast("string"),
        F.col("memo_norm").cast("string"),
        F.col("cate_1_depth").cast("string"),
        F.col("cate_2_depth").cast("string"),
        F.col("sc_measurement").cast("int"),
        F.lit(None).cast("string").alias("pred_topic"),
        F.lit(None).cast("string").alias("pred_topic_type"),
        F.lit("raw_unclassified").cast("string").alias("classification_stage"),
        F.lit(None).cast("double").alias("confidence_score"),
        F.lit(True).cast("boolean").alias("review_needed_yn"),
        F.lit(False).cast("boolean").alias("llm_used_yn"),
        F.col("embedding_text").cast("string"),
        F.col("embedding").cast("array<float>"),
        F.lit(resolved_embedding_model).cast("string").alias("embedding_model"),
        F.col("embedding_dim").cast("int"),
        F.lit(True).cast("boolean").alias("embedding_normalized"),
        F.lit("raw_unclassified").cast("string").alias("label_source_table_key"),
        F.lit(run_id).cast("string").alias("run_id"),
        F.lit(run_date).cast("string").alias("run_date"),
        F.lit(prompt_version).cast("string").alias("prompt_version"),
        F.lit(taxonomy_version).cast("string").alias("taxonomy_version"),
        F.lit(model_version).cast("string").alias("model_version"),
        F.lit(pipeline_version).cast("string").alias("pipeline_version"),
        F.lit(created_at).cast("string").alias("created_at"),
        F.lit(created_by).cast("string").alias("created_by"),
    )


def save_query_embeddings(
    embedding_df: DataFrame,
    config: dict[str, Any],
    *,
    output_table_key: str | None = None,
    mode: str = "append",
) -> str:
    """Save raw unclassified memo embeddings."""
    cfg = _cfg(config)
    table_name = get_output_table(config, output_table_key or cfg["query_embedding_table_key"])
    aligned_df = _align_embedding_schema_for_delta(embedding_df, table_name)
    (
        aligned_df.select([field.name for field in MEMO_EMBEDDING_SCHEMA.fields])
        .write.format("delta")
        .mode(mode)
        .option("mergeSchema", "true")
        .saveAsTable(table_name)
    )
    return table_name


def build_and_save_unclassified_embeddings(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    output_table_key: str | None = None,
    embedding_model: str | None = None,
    limit_rows: int | None = None,
    limit_rows_per_group: int | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Build and save embeddings for unclassified raw memos."""
    cfg = _cfg(config)
    resolved_output_key = output_table_key or cfg["query_embedding_table_key"]
    resolved_embedding_model = embedding_model or cfg["embedding_model"]

    query_df = load_raw_unclassified_memo_df(
        spark,
        config,
        limit_rows=limit_rows,
        limit_rows_per_group=limit_rows_per_group,
    )
    raw_target_count = query_df.count()
    print(f"[unclassified_embedding] raw target rows={raw_target_count}")

    target_df = query_df
    if skip_existing:
        target_df = filter_existing_query_embeddings(
            spark,
            config,
            query_df,
            output_table_key=resolved_output_key,
            embedding_model=resolved_embedding_model,
        )
    target_count = target_df.count()
    print(f"[unclassified_embedding] embedding target rows={target_count}")

    if target_count == 0:
        return {
            "output_table": get_output_table(config, resolved_output_key),
            "raw_target_count": raw_target_count,
            "target_count": 0,
            "embedding_count": 0,
            "saved": False,
        }

    embedding_df = build_query_embedding_df(
        spark,
        config,
        target_df,
        embedding_model=resolved_embedding_model,
    )
    embedding_count = embedding_df.count()
    table_name = save_query_embeddings(
        embedding_df,
        config,
        output_table_key=resolved_output_key,
    )
    print(f"[unclassified_embedding] saved table={table_name} rows={embedding_count}")
    return {
        "output_table": table_name,
        "raw_target_count": raw_target_count,
        "target_count": target_count,
        "embedding_count": embedding_count,
        "saved": True,
    }
