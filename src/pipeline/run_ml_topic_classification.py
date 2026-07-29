"""Run embedding/prototype ML topic classification for unclassified VOC memos."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common.config_loader import load_config
from ml.topic_ml_classifier import (
    classify_and_save_llm_fallback_queue,
    classify_and_save_unclassified_memos,
)
from ml.unclassified_embedding import build_and_save_unclassified_embeddings


def run_ml_topic_classification(
    spark: Any,
    config: dict[str, Any],
    *,
    run_embedding: bool = True,
    run_classification: bool = True,
    run_llm_fallback: bool = False,
    limit_rows: int | None = None,
    limit_rows_per_group: int | None = None,
    fallback_limit_rows: int | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Run unclassified memo embedding and prototype classification."""
    result: dict[str, Any] = {
        "embedding": None,
        "classification": None,
        "llm_fallback": None,
    }

    if run_embedding:
        print("[ml_topic_classification] step=embedding start")
        result["embedding"] = build_and_save_unclassified_embeddings(
            spark,
            config,
            limit_rows=limit_rows,
            limit_rows_per_group=limit_rows_per_group,
            skip_existing=skip_existing,
        )
        print(f"[ml_topic_classification] step=embedding finished | {result['embedding']}")

    if run_classification:
        print("[ml_topic_classification] step=classification start")
        result["classification"] = classify_and_save_unclassified_memos(
            spark,
            config,
            limit_rows=limit_rows,
            skip_existing=skip_existing,
        )
        print(
            "[ml_topic_classification] step=classification finished | "
            f"{result['classification']}"
        )

    if run_llm_fallback:
        print("[ml_topic_classification] step=gpt_mini_fallback start")
        result["llm_fallback"] = classify_and_save_llm_fallback_queue(
            spark,
            config,
            limit_rows=fallback_limit_rows,
        )
        print(
            "[ml_topic_classification] step=gpt_mini_fallback finished | "
            f"{result['llm_fallback']}"
        )

    return result


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[2] / "config" / "settings_intellytics.yaml"
        ),
        help="Path to settings YAML.",
    )
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--limit-rows-per-group", type=int, default=None)
    parser.add_argument("--fallback-limit-rows", type=int, default=None)
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--skip-classification", action="store_true")
    parser.add_argument("--run-llm-fallback", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run from a Databricks notebook or job context."""
    args = parse_args()

    try:
        spark  # type: ignore[name-defined]
    except NameError as exc:  # pragma: no cover - Databricks runtime path
        raise RuntimeError("This runner expects a Databricks/Spark session named `spark`.") from exc

    config = load_config(args.config)
    result = run_ml_topic_classification(
        spark,  # type: ignore[name-defined]
        config,
        run_embedding=not args.skip_embedding,
        run_classification=not args.skip_classification,
        run_llm_fallback=args.run_llm_fallback,
        limit_rows=args.limit_rows,
        limit_rows_per_group=args.limit_rows_per_group,
        fallback_limit_rows=args.fallback_limit_rows,
        skip_existing=not args.no_skip_existing,
    )
    print(result)


if __name__ == "__main__":
    main()
