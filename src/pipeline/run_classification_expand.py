"""Run raw-row expansion for memo_id-level taxonomy classifications."""

from __future__ import annotations

from typing import Any

from pyspark.sql import SparkSession

from common.config_loader import load_config
from taxonomy.classification_expander import expand_and_save_classification_full


def run_classification_expand(
    spark: SparkSession,
    *,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
    model_version: str | None = None,
    prompt_version: str | None = None,
    taxonomy_version: str | None = None,
    write_mode: str = "replace_groups",
    source_period_start: str | None = None,
    source_period_end: str | None = None,
    print_progress: bool = True,
) -> dict[str, Any]:
    """Expand classification_detail rows to raw review-row level."""
    effective_config = config or load_config(config_path)

    if print_progress:
        print(
            "[classification_expand] start | "
            f"cate_1_depth={cate_1_depth or '*'} | "
            f"cate_2_depth={cate_2_depth or '*'} | "
            f"sc_measurement={sc_measurement if sc_measurement is not None else '*'}"
        )

    result = expand_and_save_classification_full(
        spark=spark,
        config=effective_config,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        model_version=model_version,
        prompt_version=prompt_version,
        taxonomy_version=taxonomy_version,
        write_mode=write_mode,
        source_period_start=source_period_start,
        source_period_end=source_period_end,
    )

    if print_progress:
        print(
            "[classification_expand] finished | "
            f"table={result['table_name']} | "
            f"rows={result['row_count']} | "
            f"distinct_memo_ids={result['distinct_memo_count']}"
        )

    return result
