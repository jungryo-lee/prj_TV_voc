"""Build memo-level embeddings from approved taxonomy classification labels."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from common.config_loader import get_output_table
from common.embedder import DEFAULT_EMBEDDING_MODEL_PATH, Embedder


MEMO_EMBEDDING_SCHEMA = T.StructType(
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
        T.StructField("embedding_text", T.StringType(), True),
        T.StructField("embedding", T.ArrayType(T.FloatType()), True),
        T.StructField("embedding_model", T.StringType(), True),
        T.StructField("embedding_dim", T.IntegerType(), True),
        T.StructField("embedding_normalized", T.BooleanType(), True),
        T.StructField("label_source_table_key", T.StringType(), True),
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


def _embedding_cfg(config: dict[str, Any]) -> dict[str, Any]:
    """Return memo embedding config with safe defaults."""
    cfg = config.get("memo_embedding", {}) or {}
    path_cfg = config.get("path", {}) or {}
    return {
        "input_table_key": cfg.get("input_table_key", "classification_detail_final"),
        "output_table_key": cfg.get("output_table_key", "memo_embedding"),
        "embedding_model": cfg.get("embedding_model", "databricks-bge-large-en"),
        "model_path": (
            cfg.get("model_path")
            or path_cfg.get("embedding_model")
            or DEFAULT_EMBEDDING_MODEL_PATH
        ),
        "batch_size": int(cfg.get("batch_size", 64)),
        "label_min_confidence_score": float(
            cfg.get(
                "heuristic_label_min_confidence_score",
                cfg.get("label_min_confidence_score", 0.6),
            )
        ),
        "ml_auto_label_min_confidence_score": float(
            cfg.get("ml_auto_label_min_confidence_score", 0.9)
        ),
        "include_pred_topic_types": list(
            cfg.get("include_pred_topic_types", ["topic", "overall"])
        ),
        "exclude_review_needed": bool(cfg.get("exclude_review_needed", True)),
        "text_col": str(cfg.get("text_col", "memo_norm")),
    }


def _version_value(config: dict[str, Any], key: str, default: str = "") -> str:
    """Read version metadata with a safe fallback."""
    return str(config.get("version", {}).get(key, default) or default)


def _runtime_value(config: dict[str, Any], key: str, default: str = "") -> str:
    """Read resolved runtime metadata with a safe fallback."""
    return str(config.get("runtime", {}).get(key, default) or default)


def _classification_model_version(config: dict[str, Any]) -> str:
    """Resolve the model version that produced supervised topic labels."""
    model_key = (
        (config.get("memo_embedding", {}) or {}).get("label_model_key")
        or (config.get("ml_classification", {}) or {}).get("label_model_key")
        or (config.get("pipeline", {}) or {}).get("sample_classification_model_key")
        or (config.get("app", {}) or {}).get("model_key", "gpt_55")
    )
    return str(
        ((config.get("llm", {}) or {}).get("models", {}) or {})
        .get(model_key, {})
        .get("model_version", _version_value(config, "model_version"))
    )


def _table_exists(spark: SparkSession, table_name: str) -> bool:
    """Return whether a table exists."""
    try:
        spark.table(table_name).limit(1).count()
        return True
    except Exception:
        return False


def _align_embedding_schema_for_delta(
    embedding_df: DataFrame,
    table_name: str,
) -> DataFrame:
    """Cast write columns to the existing Delta table schema when possible.

    Delta cannot merge `array<float>` into an existing `array<double>` column
    with the same name. It also cannot merge string timestamps into an existing
    timestamp column. This helper keeps appends compatible with the table shape
    that was created first.
    """
    spark = embedding_df.sparkSession
    if not _table_exists(spark, table_name):
        return embedding_df

    existing_schema = spark.table(table_name).schema
    aligned_df = embedding_df
    for field in existing_schema.fields:
        if field.name not in aligned_df.columns:
            continue
        aligned_df = aligned_df.withColumn(
            field.name,
            F.col(field.name).cast(field.dataType),
        )

    return aligned_df


def _trusted_label_condition(
    min_confidence_score: float,
    *,
    ml_auto_min_confidence_score: float = 0.9,
) -> F.Column:
    """Return stage-aware condition for embedding label inclusion.

    Operational teacher labels are intentionally conservative:
    - sample LLM labels produced during taxonomy design
    - GPT-mini fallback labels
    - prototype ML labels only when confidence is high enough

    This prevents low-confidence auto labels from being fed back into the
    prototype model and amplifying early classification errors.
    """
    stage_col = F.coalesce(F.col("classification_stage"), F.lit(""))
    confidence_col = F.coalesce(F.col("confidence_score"), F.lit(0.0))
    llm_used_col = F.coalesce(F.col("llm_used_yn"), F.lit(False))

    sample_llm_label = (
        llm_used_col
        & stage_col.isin(
            "llm_fallback",
            "llm_reason_recovered",
            "sample_llm_fallback",
            "sample_gpt_mini_fallback",
        )
    )
    gpt_mini_fallback_label = stage_col.isin(
        "gpt_mini_fallback",
        "llm_fallback",
        "llm_reason_recovered",
    )
    high_confidence_ml_label = (
        (stage_col == F.lit("embedding_prototype_auto_accept"))
        & (confidence_col >= float(ml_auto_min_confidence_score))
    )

    # Backward-compatible scored sample labels are allowed only at the old
    # heuristic threshold. Operational ML labels use the stricter 0.9 threshold.
    legacy_scored_sample_label = (
        stage_col.isin("heuristic_topic_match", "ambiguous_candidate_match")
        & llm_used_col
        & (confidence_col >= float(min_confidence_score))
    )

    return (
        sample_llm_label
        | gpt_mini_fallback_label
        | high_confidence_ml_label
        | legacy_scored_sample_label
    )


def load_labeled_memo_df(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    input_table_key: str | None = None,
    model_version: str | None = None,
    prompt_version: str | None = None,
    taxonomy_version: str | None = None,
    include_pred_topic_types: Iterable[str] | None = None,
    min_confidence_score: float | None = None,
    exclude_review_needed: bool | None = None,
    group_filter_sql: str | None = None,
) -> DataFrame:
    """Load one labeled row per group/memo_id from classification output.

    The returned rows are intended to become the supervised/reference set for
    later ML/DL topic classification. By default, uncertain rows that still need
    review and `others` rows are excluded.
    """
    cfg = _embedding_cfg(config)
    resolved_input_key = input_table_key or cfg["input_table_key"]
    table_name = get_output_table(config, resolved_input_key)
    resolved_prompt_version = prompt_version or _version_value(config, "prompt_version")
    resolved_taxonomy_version = taxonomy_version or _version_value(config, "taxonomy_version")
    resolved_model_version = model_version or _classification_model_version(config)
    resolved_topic_types = list(include_pred_topic_types or cfg["include_pred_topic_types"])
    resolved_min_confidence = (
        cfg["label_min_confidence_score"]
        if min_confidence_score is None
        else float(min_confidence_score)
    )
    resolved_exclude_review_needed = (
        cfg["exclude_review_needed"]
        if exclude_review_needed is None
        else bool(exclude_review_needed)
    )

    base_df = (
        spark.table(table_name)
        .where(F.col("prompt_version") == resolved_prompt_version)
        .where(F.col("taxonomy_version") == resolved_taxonomy_version)
        .where(F.col("model_version") == resolved_model_version)
        .where(F.col("memo_id").isNotNull())
        .where(F.col("pred_topic").isNotNull())
        .where(F.col("pred_topic") != "기타")
        .where(F.col("pred_topic_type").isin(resolved_topic_types))
        .where(
            _trusted_label_condition(
                resolved_min_confidence,
                ml_auto_min_confidence_score=cfg["ml_auto_label_min_confidence_score"],
            )
        )
    )

    if "is_latest" in base_df.columns:
        base_df = base_df.where(F.coalesce(F.col("is_latest"), F.lit(True)) == F.lit(True))
    if resolved_exclude_review_needed and "review_needed_yn" in base_df.columns:
        base_df = base_df.where(F.coalesce(F.col("review_needed_yn"), F.lit(False)) == F.lit(False))
    if group_filter_sql:
        base_df = base_df.where(F.expr(group_filter_sql))

    window = Window.partitionBy(
        "cate_1_depth",
        "cate_2_depth",
        "sc_measurement",
        "memo_id",
        "prompt_version",
        "taxonomy_version",
        "model_version",
    ).orderBy(F.col("created_at").desc_nulls_last(), F.col("run_id").desc_nulls_last())

    return (
        base_df.withColumn("_rn", F.row_number().over(window))
        .where(F.col("_rn") == 1)
        .drop("_rn")
    )


def filter_unembedded_memo_df(
    spark: SparkSession,
    config: dict[str, Any],
    labeled_df: DataFrame,
    *,
    output_table_key: str | None = None,
    embedding_model: str | None = None,
) -> DataFrame:
    """Remove rows that already have embeddings in the output table."""
    cfg = _embedding_cfg(config)
    resolved_output_key = output_table_key or cfg["output_table_key"]
    table_name = get_output_table(config, resolved_output_key)
    resolved_model = embedding_model or cfg["model_path"]

    if not _table_exists(spark, table_name):
        return labeled_df

    existing_keys = (
        spark.table(table_name)
        .where(F.col("embedding_model") == resolved_model)
        .select(
            "cate_1_depth",
            "cate_2_depth",
            "sc_measurement",
            "memo_id",
            "pred_topic",
            "pred_topic_type",
            "prompt_version",
            "taxonomy_version",
            "model_version",
        )
        .dropDuplicates()
    )

    join_keys = [
        "cate_1_depth",
        "cate_2_depth",
        "sc_measurement",
        "memo_id",
        "pred_topic",
        "pred_topic_type",
        "prompt_version",
        "taxonomy_version",
        "model_version",
    ]
    return labeled_df.join(existing_keys, on=join_keys, how="left_anti")


def _batched(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    """Yield fixed-size batches."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_memo_embedding_spark_df(
    spark: SparkSession,
    labeled_df: DataFrame,
    config: dict[str, Any],
    *,
    embedder: Embedder | None = None,
    model_path: str | None = None,
    batch_size: int | None = None,
    text_col: str | None = None,
    label_source_table_key: str | None = None,
    created_by: str = "codex",
    limit_rows: int | None = None,
    show_progress: bool = True,
) -> DataFrame:
    """Encode labeled memo rows and return a Spark DataFrame."""
    cfg = _embedding_cfg(config)
    resolved_model_path = model_path or cfg["model_path"]
    resolved_batch_size = int(batch_size or cfg["batch_size"])
    resolved_text_col = text_col or cfg["text_col"]
    resolved_label_source_table_key = label_source_table_key or cfg["input_table_key"]
    model = embedder or Embedder(resolved_model_path)
    run_id = _runtime_value(config, "resolved_run_id")
    run_date = _runtime_value(config, "resolved_run_date")
    pipeline_version = _version_value(config, "pipeline_version")
    created_at = datetime.utcnow().isoformat(timespec="seconds")

    selected_cols = [
        "memo_id",
        "memo",
        "memo_norm",
        "cate_1_depth",
        "cate_2_depth",
        "sc_measurement",
        "pred_topic",
        "pred_topic_type",
        "classification_stage",
        "confidence_score",
        "review_needed_yn",
        "llm_used_yn",
        "run_id",
        "run_date",
        "prompt_version",
        "taxonomy_version",
        "model_version",
    ]
    available_cols = [col for col in selected_cols if col in labeled_df.columns]
    source_df = labeled_df.select(*available_cols)
    if limit_rows is not None:
        source_df = source_df.limit(int(limit_rows))

    source_rows = [row.asDict(recursive=True) for row in source_df.collect()]
    if not source_rows:
        return spark.createDataFrame([], MEMO_EMBEDDING_SCHEMA)

    output_rows: list[dict[str, Any]] = []
    embedding_dim: int | None = None
    for batch_no, batch_rows in enumerate(_batched(source_rows, resolved_batch_size), start=1):
        texts = [
            str(row.get(resolved_text_col) or row.get("memo") or "").strip()
            for row in batch_rows
        ]
        embeddings = model.encode(
            texts,
            batch_size=resolved_batch_size,
            show_progress=show_progress,
        )
        embedding_dim = int(embeddings.shape[1]) if embeddings.ndim == 2 else model.embedding_dim()
        print(
            f"[memo_embedding] encoded batch={batch_no} rows={len(batch_rows)} dim={embedding_dim}"
        )

        for row, vector, text in zip(batch_rows, embeddings, texts):
            output_rows.append(
                {
                    "memo_id": str(row.get("memo_id") or ""),
                    "memo": row.get("memo"),
                    "memo_norm": row.get("memo_norm"),
                    "cate_1_depth": row.get("cate_1_depth"),
                    "cate_2_depth": row.get("cate_2_depth"),
                    "sc_measurement": int(row.get("sc_measurement") or 0),
                    "pred_topic": row.get("pred_topic"),
                    "pred_topic_type": row.get("pred_topic_type"),
                    "classification_stage": row.get("classification_stage"),
                    "confidence_score": (
                        float(row["confidence_score"])
                        if row.get("confidence_score") is not None
                        else None
                    ),
                    "review_needed_yn": bool(row.get("review_needed_yn", False)),
                    "llm_used_yn": bool(row.get("llm_used_yn", False)),
                    "embedding_text": text,
                    "embedding": [float(value) for value in vector.tolist()],
                    "embedding_model": resolved_model_path,
                    "embedding_dim": embedding_dim,
                    "embedding_normalized": True,
                    "label_source_table_key": resolved_label_source_table_key,
                    "run_id": str(row.get("run_id") or run_id),
                    "run_date": str(row.get("run_date") or run_date),
                    "prompt_version": str(row.get("prompt_version") or _version_value(config, "prompt_version")),
                    "taxonomy_version": str(row.get("taxonomy_version") or _version_value(config, "taxonomy_version")),
                    "model_version": str(row.get("model_version") or _version_value(config, "model_version")),
                    "pipeline_version": pipeline_version,
                    "created_at": created_at,
                    "created_by": created_by,
                }
            )

    return spark.createDataFrame(output_rows, MEMO_EMBEDDING_SCHEMA)


def save_memo_embeddings(
    embedding_df: DataFrame,
    config: dict[str, Any],
    *,
    output_table_key: str | None = None,
    mode: str = "append",
) -> str:
    """Save memo embeddings to the configured Delta table."""
    cfg = _embedding_cfg(config)
    resolved_output_key = output_table_key or cfg["output_table_key"]
    table_name = get_output_table(config, resolved_output_key)
    aligned_df = _align_embedding_schema_for_delta(embedding_df, table_name)

    (
        aligned_df.select([field.name for field in MEMO_EMBEDDING_SCHEMA.fields])
        .write.format("delta")
        .mode(mode)
        .option("mergeSchema", "true")
        .saveAsTable(table_name)
    )
    return table_name


def build_memo_embedding_ai_query_df(
    spark: SparkSession,
    labeled_df: DataFrame,
    config: dict[str, Any],
    *,
    embedding_model: str | None = None,
    text_col: str | None = None,
    label_source_table_key: str | None = None,
    created_by: str = "memo_embedding_ai_query",
    limit_rows: int | None = None,
) -> DataFrame:
    """Attach Databricks AI Query embeddings to labeled memo rows.

    This path keeps labeled sample embeddings compatible with the operational
    unclassified embedding stage, which also uses a Databricks serving endpoint.
    """
    cfg = _embedding_cfg(config)
    resolved_embedding_model = embedding_model or cfg["embedding_model"]
    resolved_text_col = text_col or cfg["text_col"]
    resolved_label_source_table_key = label_source_table_key or cfg["input_table_key"]
    run_id = _runtime_value(config, "resolved_run_id")
    run_date = _runtime_value(config, "resolved_run_date")
    pipeline_version = _version_value(config, "pipeline_version")
    created_at = datetime.utcnow().isoformat(timespec="seconds")

    selected_cols = [
        "memo_id",
        "memo",
        "memo_norm",
        "cate_1_depth",
        "cate_2_depth",
        "sc_measurement",
        "pred_topic",
        "pred_topic_type",
        "classification_stage",
        "confidence_score",
        "review_needed_yn",
        "llm_used_yn",
        "run_id",
        "run_date",
        "prompt_version",
        "taxonomy_version",
        "model_version",
    ]
    available_cols = [col for col in selected_cols if col in labeled_df.columns]
    source_df = labeled_df.select(*available_cols)
    if limit_rows is not None:
        source_df = source_df.limit(int(limit_rows))

    source_df = source_df.withColumn(
        "embedding_text",
        F.coalesce(
            F.col(resolved_text_col).cast("string"),
            F.col("memo").cast("string"),
        ),
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
        F.col("pred_topic").cast("string"),
        F.col("pred_topic_type").cast("string"),
        F.col("classification_stage").cast("string"),
        F.col("confidence_score").cast("double"),
        F.col("review_needed_yn").cast("boolean"),
        F.col("llm_used_yn").cast("boolean"),
        F.col("embedding_text").cast("string"),
        F.col("embedding").cast("array<float>"),
        F.lit(resolved_embedding_model).cast("string").alias("embedding_model"),
        F.col("embedding_dim").cast("int"),
        F.lit(True).cast("boolean").alias("embedding_normalized"),
        F.lit(resolved_label_source_table_key).cast("string").alias("label_source_table_key"),
        F.coalesce(F.col("run_id").cast("string"), F.lit(run_id)).alias("run_id"),
        F.coalesce(F.col("run_date").cast("string"), F.lit(run_date)).alias("run_date"),
        F.coalesce(
            F.col("prompt_version").cast("string"),
            F.lit(_version_value(config, "prompt_version")),
        ).alias("prompt_version"),
        F.coalesce(
            F.col("taxonomy_version").cast("string"),
            F.lit(_version_value(config, "taxonomy_version")),
        ).alias("taxonomy_version"),
        F.coalesce(
            F.col("model_version").cast("string"),
            F.lit(_version_value(config, "model_version")),
        ).alias("model_version"),
        F.lit(pipeline_version).cast("string").alias("pipeline_version"),
        F.lit(created_at).cast("string").alias("created_at"),
        F.lit(created_by).cast("string").alias("created_by"),
    )


def build_and_save_memo_embeddings_ai_query(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    input_table_key: str | None = None,
    output_table_key: str | None = None,
    embedding_model: str | None = None,
    min_confidence_score: float | None = None,
    limit_rows: int | None = None,
    skip_existing: bool = True,
    group_filter_sql: str | None = None,
    created_by: str = "memo_embedding_ai_query",
) -> dict[str, Any]:
    """Load trusted labeled rows, embed with Databricks AI Query, and save."""
    cfg = _embedding_cfg(config)
    resolved_input_key = input_table_key or cfg["input_table_key"]
    resolved_output_key = output_table_key or cfg["output_table_key"]
    resolved_embedding_model = embedding_model or cfg["embedding_model"]

    labeled_df = load_labeled_memo_df(
        spark,
        config,
        input_table_key=resolved_input_key,
        min_confidence_score=min_confidence_score,
        group_filter_sql=group_filter_sql,
    )
    labeled_count = labeled_df.count()
    print(f"[memo_embedding_ai_query] labeled rows={labeled_count}")

    target_df = labeled_df
    if skip_existing:
        target_df = filter_unembedded_memo_df(
            spark,
            config,
            labeled_df,
            output_table_key=resolved_output_key,
            embedding_model=resolved_embedding_model,
        )
    target_count = target_df.count()
    print(
        "[memo_embedding_ai_query] "
        f"target rows={target_count} skip_existing={skip_existing}"
    )

    if target_count == 0:
        return {
            "input_table_key": resolved_input_key,
            "output_table_key": resolved_output_key,
            "output_table": get_output_table(config, resolved_output_key),
            "labeled_count": labeled_count,
            "target_count": target_count,
            "embedding_count": 0,
            "saved": False,
        }

    embedding_df = build_memo_embedding_ai_query_df(
        spark,
        target_df,
        config,
        embedding_model=resolved_embedding_model,
        label_source_table_key=resolved_input_key,
        created_by=created_by,
        limit_rows=limit_rows,
    )
    embedding_count = embedding_df.count()
    table_name = save_memo_embeddings(
        embedding_df,
        config,
        output_table_key=resolved_output_key,
        mode="append",
    )
    print(f"[memo_embedding_ai_query] saved table={table_name} rows={embedding_count}")
    return {
        "input_table_key": resolved_input_key,
        "output_table_key": resolved_output_key,
        "output_table": table_name,
        "labeled_count": labeled_count,
        "target_count": target_count,
        "embedding_count": embedding_count,
        "saved": True,
    }


def build_and_save_memo_embeddings(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    input_table_key: str | None = None,
    output_table_key: str | None = None,
    model_path: str | None = None,
    batch_size: int | None = None,
    min_confidence_score: float | None = None,
    limit_rows: int | None = None,
    skip_existing: bool = True,
    created_by: str = "codex",
) -> dict[str, Any]:
    """Load labeled memo rows, encode missing rows, and save embeddings."""
    cfg = _embedding_cfg(config)
    resolved_input_key = input_table_key or cfg["input_table_key"]
    resolved_output_key = output_table_key or cfg["output_table_key"]
    resolved_model_path = model_path or cfg["model_path"]

    labeled_df = load_labeled_memo_df(
        spark,
        config,
        input_table_key=resolved_input_key,
        min_confidence_score=min_confidence_score,
    )
    labeled_count = labeled_df.count()
    print(f"[memo_embedding] labeled rows={labeled_count}")

    target_df = labeled_df
    if skip_existing:
        target_df = filter_unembedded_memo_df(
            spark,
            config,
            labeled_df,
            output_table_key=resolved_output_key,
            embedding_model=resolved_model_path,
        )
    target_count = target_df.count()
    print(f"[memo_embedding] target rows={target_count} skip_existing={skip_existing}")

    embedding_df = build_memo_embedding_spark_df(
        spark,
        target_df,
        config,
        model_path=resolved_model_path,
        batch_size=batch_size,
        label_source_table_key=resolved_input_key,
        created_by=created_by,
        limit_rows=limit_rows,
    )
    embedding_count = embedding_df.count()
    if embedding_count == 0:
        return {
            "input_table_key": resolved_input_key,
            "output_table_key": resolved_output_key,
            "output_table": get_output_table(config, resolved_output_key),
            "labeled_count": labeled_count,
            "target_count": target_count,
            "embedding_count": 0,
            "saved": False,
        }

    table_name = save_memo_embeddings(
        embedding_df,
        config,
        output_table_key=resolved_output_key,
        mode="append",
    )
    print(f"[memo_embedding] saved table={table_name} rows={embedding_count}")
    return {
        "input_table_key": resolved_input_key,
        "output_table_key": resolved_output_key,
        "output_table": table_name,
        "labeled_count": labeled_count,
        "target_count": target_count,
        "embedding_count": embedding_count,
        "saved": True,
    }
