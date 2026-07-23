"""Run memo embedding build from labeled taxonomy classification rows."""

from __future__ import annotations

import argparse
from pathlib import Path

from common.config_loader import load_config
from ml.memo_embedding import build_and_save_memo_embeddings


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[2] / "config" / "settings_intellytics.yaml"),
        help="Path to settings YAML.",
    )
    parser.add_argument("--input-table-key", default=None)
    parser.add_argument("--output-table-key", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--min-confidence-score", type=float, default=None)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--no-skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the memo embedding pipeline."""
    args = parse_args()

    try:
        spark  # type: ignore[name-defined]
    except NameError as exc:  # pragma: no cover - Databricks runtime path
        raise RuntimeError("This runner expects a Databricks/Spark session named `spark`.") from exc

    config = load_config(args.config)
    result = build_and_save_memo_embeddings(
        spark,  # type: ignore[name-defined]
        config,
        input_table_key=args.input_table_key,
        output_table_key=args.output_table_key,
        model_path=args.model_path,
        batch_size=args.batch_size,
        min_confidence_score=args.min_confidence_score,
        limit_rows=args.limit_rows,
        skip_existing=not args.no_skip_existing,
    )
    print(result)


if __name__ == "__main__":
    main()
