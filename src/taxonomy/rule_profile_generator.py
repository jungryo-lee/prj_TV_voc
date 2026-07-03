"""Rule-profile generation orchestration for taxonomy groups."""

from __future__ import annotations

from typing import Any

from common.llm_client import DatabricksLLMClient, get_llm_client
from taxonomy.category_pattern_generator import get_effective_category_pattern_seed
from taxonomy.group_sampler import collect_rule_profile_sample_memos
from taxonomy.prompt_builder import build_rule_profile_messages, overall_topic_name


def _clean_text(value: Any) -> str:
    """Collapse whitespace for stable downstream payloads."""
    return " ".join(str(value or "").split()).strip()


def _clean_term_list(
    values: list[Any] | None,
    *,
    max_items: int,
) -> list[str]:
    """Normalize list-like LLM output into unique compact strings."""
    if not values:
        return []

    seen: set[str] = set()
    cleaned: list[str] = []

    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(text)
        if len(cleaned) >= max_items:
            break

    return cleaned


def merge_category_seed_into_rule_profile_messages(
    messages: list[dict[str, str]],
    category_seed: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Append category-seed guidance to the base rule-profile prompt."""
    if not category_seed:
        return messages

    seed_lines = [
        "",
        "Bootstrap category seed guidance:",
    ]

    if category_seed.get("has_static_seed"):
        seed_lines.append("- This category already has known static feature seeds.")
    else:
        seed_lines.append("- This category does not have curated static seeds yet.")

    if category_seed.get("category_summary"):
        seed_lines.append(f"- category_summary: {category_seed['category_summary']}")

    if category_seed.get("static_feature_hint_terms"):
        seed_lines.append(
            f"- static_feature_hint_terms: {category_seed['static_feature_hint_terms']}"
        )

    for key in [
        "feature_hint_terms",
        "reason_signal_terms",
        "overall_sentiment_terms",
        "candidate_topic_labels",
        "sample_non_overall_memos",
    ]:
        values = category_seed.get(key)
        if values:
            seed_lines.append(f"- {key}: {values}")

    merged = list(messages)
    merged[0] = {
        **merged[0],
        "content": merged[0]["content"] + "\n" + "\n".join(seed_lines),
    }
    return merged


def normalize_rule_profile_output(
    payload: dict[str, Any],
    *,
    sc_measurement: int,
) -> dict[str, Any]:
    """Normalize raw LLM JSON into a stable rule-profile schema."""
    return {
        "overall_topic_name": overall_topic_name(sc_measurement),
        "overall_allowed_rule": _clean_text(payload.get("overall_allowed_rule")),
        "overall_block_rule": _clean_text(payload.get("overall_block_rule")),
        "overall_sentiment_terms": _clean_term_list(
            payload.get("overall_sentiment_terms"),
            max_items=40,
        ),
        "feature_hint_terms": _clean_term_list(
            payload.get("feature_hint_terms"),
            max_items=100,
        ),
        "reason_signal_terms": _clean_term_list(
            payload.get("reason_signal_terms"),
            max_items=100,
        ),
        "non_overall_examples": _clean_term_list(
            payload.get("non_overall_examples"),
            max_items=20,
        ),
    }


def build_rule_profile_result(
    *,
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    sample_memos: list[str],
    category_seed: dict[str, Any],
    llm_client: DatabricksLLMClient,
) -> dict[str, Any]:
    """Build one normalized rule-profile result from sampled memos."""
    messages = build_rule_profile_messages(
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        sample_memos=sample_memos,
    )
    messages = merge_category_seed_into_rule_profile_messages(
        messages,
        category_seed=category_seed,
    )

    raw_payload = llm_client.converse_json(
        system_prompt=messages[0]["content"],
        user_prompt=messages[1]["content"],
    )
    normalized = normalize_rule_profile_output(
        raw_payload,
        sc_measurement=sc_measurement,
    )

    return {
        "cate_1_depth": cate_1_depth,
        "cate_2_depth": cate_2_depth,
        "sc_measurement": int(sc_measurement),
        "sample_memo_count": len(sample_memos),
        "category_seed_used": category_seed,
        **normalized,
    }


def generate_rule_profile_for_group(
    *,
    config: dict[str, Any],
    spark,
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    llm_client: DatabricksLLMClient | None = None,
    sample_memos: list[str] | None = None,
    sample_seed: str | None = None,
    model_key: str | None = None,
    force_seed_generation: bool = False,
) -> dict[str, Any]:
    """Generate a rule profile for one category/sentiment group."""
    client = llm_client or get_llm_client(config=config, model_key=model_key)

    if sample_memos is None:
        sample_memos = collect_rule_profile_sample_memos(
            spark=spark,
            config=config,
            cate_1_depth=cate_1_depth,
            cate_2_depth=cate_2_depth,
            sc_measurement=sc_measurement,
            sample_seed=sample_seed or "seed_20260420",
        )

    category_seed = get_effective_category_pattern_seed(
        config=config,
        spark=spark,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        llm_client=client,
        sample_memos=sample_memos,
        sample_seed=sample_seed,
        model_key=model_key,
        force_generate=force_seed_generation,
    )

    return build_rule_profile_result(
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        sample_memos=sample_memos,
        category_seed=category_seed,
        llm_client=client,
    )


def generate_rule_profiles_for_groups(
    *,
    config: dict[str, Any],
    spark,
    group_rows: list[dict[str, Any]],
    llm_client: DatabricksLLMClient | None = None,
    sample_seed: str | None = None,
    model_key: str | None = None,
    force_seed_generation: bool = False,
) -> list[dict[str, Any]]:
    """Generate rule profiles for multiple group rows."""
    client = llm_client or get_llm_client(config=config, model_key=model_key)
    results: list[dict[str, Any]] = []

    for group_row in group_rows:
        results.append(
            generate_rule_profile_for_group(
                config=config,
                spark=spark,
                cate_1_depth=group_row["cate_1_depth"],
                cate_2_depth=group_row["cate_2_depth"],
                sc_measurement=int(group_row["sc_measurement"]),
                llm_client=client,
                sample_seed=sample_seed,
                model_key=model_key,
                force_seed_generation=force_seed_generation,
            )
        )

    return results
