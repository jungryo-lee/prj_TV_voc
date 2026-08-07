"""Category alias helpers used before taxonomy grouping/classification."""

from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame, functions as F


def _sql_escape(value: Any) -> str:
    """Escape a value for SQL literal usage."""
    return str(value).replace("'", "''")


def get_category_alias_rules(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return enabled category alias rules from config."""
    taxonomy_cfg = config.get("taxonomy", {}) or {}
    rules = taxonomy_cfg.get("category_alias_rules", []) or []
    return [rule for rule in rules if rule and rule.get("enabled", True)]


def cate_2_alias_sql(config: dict[str, Any], *, cate_1_col: str = "cate_1_depth", cate_2_col: str = "cate_2_depth") -> str:
    """Build a SQL CASE expression that normalizes cate_2_depth aliases."""
    expr = cate_2_col
    for rule in get_category_alias_rules(config):
        cate_1 = _sql_escape(rule.get("cate_1_depth", ""))
        target = _sql_escape(rule.get("target_cate_2_depth", ""))
        sources = [_sql_escape(value) for value in rule.get("source_cate_2_depths", [])]
        if not cate_1 or not target or not sources:
            continue
        source_sql = ", ".join(f"'{value}'" for value in sources)
        expr = (
            "case "
            f"when {cate_1_col} = '{cate_1}' and {cate_2_col} in ({source_sql}) "
            f"then '{target}' "
            f"else {expr} end"
        )
    return expr


def apply_category_aliases(
    df: DataFrame,
    config: dict[str, Any],
    *,
    cate_1_col: str = "cate_1_depth",
    cate_2_col: str = "cate_2_depth",
) -> DataFrame:
    """Normalize category columns for pipeline logic without touching source data."""
    out = df
    for rule in get_category_alias_rules(config):
        cate_1 = rule.get("cate_1_depth")
        target = rule.get("target_cate_2_depth")
        sources = rule.get("source_cate_2_depths", []) or []
        if not cate_1 or not target or not sources:
            continue
        out = out.withColumn(
            cate_2_col,
            F.when(
                (F.col(cate_1_col) == F.lit(str(cate_1)))
                & F.col(cate_2_col).isin([str(value) for value in sources]),
                F.lit(str(target)),
            ).otherwise(F.col(cate_2_col)),
        )
    return out
