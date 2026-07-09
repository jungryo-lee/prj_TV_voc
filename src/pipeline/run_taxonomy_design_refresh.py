"""Run taxonomy design refresh for rule-profile and topic-pool stages."""

from __future__ import annotations

from typing import Any

from common.config_loader import load_config
from taxonomy.group_sampler import load_target_groups
from taxonomy.rule_profile_generator import generate_rule_profile_for_group
from taxonomy.rule_profile_writer import save_rule_profiles
from taxonomy.topic_pool_generator import generate_topic_pool_for_group
from taxonomy.topic_pool_writer import save_topic_pools


def _matches_target_filters(
    group_row: dict[str, Any],
    *,
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
) -> bool:
    """Return whether a group row matches the optional target filters."""
    if cate_1_depth and group_row["cate_1_depth"] != cate_1_depth:
        return False
    if cate_2_depth and group_row["cate_2_depth"] != cate_2_depth:
        return False
    if sc_measurement is not None and int(group_row["sc_measurement"]) != int(
        sc_measurement
    ):
        return False
    return True


def load_design_target_groups(
    spark,
    config: dict[str, Any],
    *,
    limit_group_count: int | None = None,
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
) -> list[dict[str, Any]]:
    """Load design target groups and apply optional filters."""
    group_df = load_target_groups(
        spark=spark,
        config=config,
        limit_group_count=None,
    )
    group_rows = [row.asDict(recursive=True) for row in group_df.toLocalIterator()]
    filtered_rows = [
        row
        for row in group_rows
        if _matches_target_filters(
            row,
            cate_1_depth=cate_1_depth,
            cate_2_depth=cate_2_depth,
            sc_measurement=sc_measurement,
        )
    ]

    if limit_group_count is not None:
        return filtered_rows[: int(limit_group_count)]

    return filtered_rows


def run_taxonomy_design_refresh(
    spark,
    *,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    model_key: str = "gpt_55",
    limit_group_count: int | None = None,
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
    force_seed_generation: bool = False,
    save_rule_profile: bool = True,
    save_topic_pool: bool = True,
) -> dict[str, Any]:
    """Run rule-profile and topic-pool refresh for selected taxonomy groups."""
    effective_config = config or load_config(config_path)
    target_groups = load_design_target_groups(
        spark=spark,
        config=effective_config,
        limit_group_count=limit_group_count,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
    )

    rule_profile_results: list[dict[str, Any]] = []
    topic_pool_results: list[dict[str, Any]] = []

    for group_row in target_groups:
        rule_profile_result = generate_rule_profile_for_group(
            config=effective_config,
            spark=spark,
            cate_1_depth=group_row["cate_1_depth"],
            cate_2_depth=group_row["cate_2_depth"],
            sc_measurement=int(group_row["sc_measurement"]),
            model_key=model_key,
            force_seed_generation=force_seed_generation,
        )
        rule_profile_results.append(rule_profile_result)

        topic_pool_result = generate_topic_pool_for_group(
            config=effective_config,
            spark=spark,
            cate_1_depth=group_row["cate_1_depth"],
            cate_2_depth=group_row["cate_2_depth"],
            sc_measurement=int(group_row["sc_measurement"]),
            rule_profile=rule_profile_result,
            model_key=model_key,
        )
        topic_pool_results.append(topic_pool_result)

    saved_tables: dict[str, str] = {}

    if save_rule_profile and rule_profile_results:
        saved_tables["rule_profile"] = save_rule_profiles(
            spark=spark,
            config=effective_config,
            results=rule_profile_results,
            model_key=model_key,
            write_mode="replace_groups",
        )

    if save_topic_pool and topic_pool_results:
        saved_tables["topic_pool"] = save_topic_pools(
            spark=spark,
            config=effective_config,
            results=topic_pool_results,
            model_key=model_key,
            write_mode="replace_groups",
        )

    return {
        "group_count": len(target_groups),
        "rule_profile_count": len(rule_profile_results),
        "topic_pool_count": len(topic_pool_results),
        "model_key": model_key,
        "saved_tables": saved_tables,
        "target_groups": target_groups,
        "rule_profile_results": rule_profile_results,
        "topic_pool_results": topic_pool_results,
    }
