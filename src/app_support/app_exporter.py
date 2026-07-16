"""Export pipeline tables to Parquet snapshots for Databricks Apps."""

from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from common.config_loader import get_output_table


def _app_data_root(config: dict[str, Any]) -> str:
    """Return the app-data root path used by Databricks Apps."""
    return str(
        config.get("app", {}).get("data_root")
        or f"{config.get('path', {}).get('project_root', '.')}/app_data"
    ).rstrip("/")


def _model_version(config: dict[str, Any], model_key: str) -> str:
    """Resolve a model version from settings."""
    return str(config["llm"]["models"][model_key]["model_version"])


def _version_value(config: dict[str, Any], key: str) -> str:
    """Resolve a pipeline version value from settings."""
    return str(config.get("version", {}).get(key, ""))


def _filter_target(
    df: DataFrame,
    *,
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
    model_version: str | None = None,
    prompt_version: str | None = None,
    taxonomy_version: str | None = None,
) -> DataFrame:
    """Apply optional common target filters."""
    if cate_1_depth is not None:
        df = df.where(F.col("cate_1_depth") == cate_1_depth)
    if cate_2_depth is not None:
        df = df.where(F.col("cate_2_depth") == cate_2_depth)
    if sc_measurement is not None:
        df = df.where(F.col("sc_measurement") == int(sc_measurement))
    if model_version is not None and "model_version" in df.columns:
        df = df.where(F.col("model_version") == model_version)
    if prompt_version is not None and "prompt_version" in df.columns:
        df = df.where(F.col("prompt_version") == prompt_version)
    if taxonomy_version is not None and "taxonomy_version" in df.columns:
        df = df.where(F.col("taxonomy_version") == taxonomy_version)
    return df


def _write_snapshot(df: DataFrame, path: str, *, mode: str = "overwrite") -> str:
    """Write an app-facing Parquet snapshot."""
    df.write.mode(mode).parquet(path)
    return path


def export_topic_pool_snapshot(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
    model_key: str = "gpt_55",
    mode: str = "overwrite",
) -> dict[str, Any]:
    """Export the current topic pool for app dropdowns."""
    model_version = _model_version(config, model_key)
    prompt_version = _version_value(config, "prompt_version")
    taxonomy_version = _version_value(config, "taxonomy_version")
    source_table = get_output_table(config, "topic_pool")
    output_path = f"{_app_data_root(config)}/exports/topic_pool_current"

    df = _filter_target(
        spark.table(source_table),
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        model_version=model_version,
        prompt_version=prompt_version,
        taxonomy_version=taxonomy_version,
    ).select(
        "cate_1_depth",
        "cate_2_depth",
        "sc_measurement",
        "topic_order",
        "topic",
        "description",
        "model_version",
        "prompt_version",
        "taxonomy_version",
        "created_at",
    )

    row_count = df.count()
    return {
        "name": "topic_pool_current",
        "path": _write_snapshot(df, output_path, mode=mode),
        "row_count": int(row_count),
    }


def export_classification_summary_snapshot(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
    model_key: str = "gpt_55",
    mode: str = "overwrite",
) -> dict[str, Any]:
    """Export topic distribution summary for app dashboards."""
    model_version = _model_version(config, model_key)
    prompt_version = _version_value(config, "prompt_version")
    taxonomy_version = _version_value(config, "taxonomy_version")
    source_table = get_output_table(config, "classification_full")
    output_path = f"{_app_data_root(config)}/exports/classification_summary"

    base_df = _filter_target(
        spark.table(source_table),
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        model_version=model_version,
        prompt_version=prompt_version,
        taxonomy_version=taxonomy_version,
    )

    summary_df = (
        base_df.groupBy(
            "cate_1_depth",
            "cate_2_depth",
            "sc_measurement",
            "pred_topic",
            "pred_topic_type",
        )
        .agg(
            F.count("*").alias("row_cnt"),
            F.countDistinct("memo_id").alias("memo_id_cnt"),
        )
        .withColumn(
            "row_ratio",
            F.col("row_cnt")
            / F.sum("row_cnt").over(
                Window.partitionBy("cate_1_depth", "cate_2_depth", "sc_measurement")
            ),
        )
        .orderBy(F.col("row_cnt").desc())
    )

    row_count = summary_df.count()
    return {
        "name": "classification_summary",
        "path": _write_snapshot(summary_df, output_path, mode=mode),
        "row_count": int(row_count),
    }


def export_others_review_candidates_snapshot(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
    model_key: str = "gpt_55",
    max_rows: int = 500,
    mode: str = "overwrite",
) -> dict[str, Any]:
    """Export distinct others rows for human app review."""
    from taxonomy.manual_fallback_applier import load_manual_fallback_candidates

    model_version = _model_version(config, model_key)
    prompt_version = _version_value(config, "prompt_version")
    taxonomy_version = _version_value(config, "taxonomy_version")
    output_path = f"{_app_data_root(config)}/exports/others_review_candidates"

    df = load_manual_fallback_candidates(
        spark,
        config,
        input_table_key="classification_full",
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        model_version=model_version,
        prompt_version=prompt_version,
        taxonomy_version=taxonomy_version,
        max_rows=max_rows,
    )

    row_count = df.count()
    return {
        "name": "others_review_candidates",
        "path": _write_snapshot(df, output_path, mode=mode),
        "row_count": int(row_count),
    }


def export_app_snapshots(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
    model_key: str = "gpt_55",
    max_review_rows: int = 500,
    mode: str = "overwrite",
) -> dict[str, Any]:
    """Export all current app-facing Parquet snapshots."""
    outputs = [
        export_topic_pool_snapshot(
            spark,
            config,
            cate_1_depth=cate_1_depth,
            cate_2_depth=cate_2_depth,
            sc_measurement=sc_measurement,
            model_key=model_key,
            mode=mode,
        ),
        export_classification_summary_snapshot(
            spark,
            config,
            cate_1_depth=cate_1_depth,
            cate_2_depth=cate_2_depth,
            sc_measurement=sc_measurement,
            model_key=model_key,
            mode=mode,
        ),
        export_others_review_candidates_snapshot(
            spark,
            config,
            cate_1_depth=cate_1_depth,
            cate_2_depth=cate_2_depth,
            sc_measurement=sc_measurement,
            model_key=model_key,
            max_rows=max_review_rows,
            mode=mode,
        ),
    ]
    return {
        "app_data_root": _app_data_root(config),
        "outputs": outputs,
    }
