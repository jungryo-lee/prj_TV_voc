"""Dynamic category-pattern seed generation for taxonomy bootstrapping."""

from __future__ import annotations

from typing import Any

from common.llm_client import DatabricksLLMClient, get_llm_client
from taxonomy.group_sampler import collect_rule_profile_sample_memos
from taxonomy.prompt_builder import (
    build_category_pattern_seed_messages,
    get_static_category_feature_patterns,
    has_static_category_patterns,
)


def _clean_term_list(
    values: list[Any] | None,
    *,
    max_items: int,
) -> list[str]:
    """Normalize list-like LLM outputs into compact unique string lists."""
    if not values:
        return []

    seen: set[str] = set()
    cleaned: list[str] = []

    for value in values:
        text = " ".join(str(value).split()).strip()
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


def normalize_category_pattern_seed_output(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Normalize raw LLM JSON into a stable category seed schema."""
    return {
        "category_summary": " ".join(str(payload.get("category_summary", "")).split()).strip(),
        "feature_hint_terms": _clean_term_list(
            payload.get("feature_hint_terms"),
            max_items=80,
        ),
        "reason_signal_terms": _clean_term_list(
            payload.get("reason_signal_terms"),
            max_items=80,
        ),
        "overall_sentiment_terms": _clean_term_list(
            payload.get("overall_sentiment_terms"),
            max_items=40,
        ),
        "candidate_topic_labels": _clean_term_list(
            payload.get("candidate_topic_labels"),
            max_items=20,
        ),
        "sample_non_overall_memos": _clean_term_list(
            payload.get("sample_non_overall_memos"),
            max_items=12,
        ),
    }


def build_static_category_pattern_seed(
    cate_1_depth: str,
    cate_2_depth: str,
) -> dict[str, Any]:
    """Build a minimal seed payload from existing static category patterns."""
    static_terms = get_static_category_feature_patterns(cate_1_depth, cate_2_depth)
    return {
        "category_summary": "",
        "feature_hint_terms": list(static_terms),
        "reason_signal_terms": [],
        "overall_sentiment_terms": [],
        "candidate_topic_labels": [],
        "sample_non_overall_memos": [],
    }


def generate_category_pattern_seed(
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
) -> dict[str, Any]:
    """Generate dynamic category seed terms for a category group."""
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

    messages = build_category_pattern_seed_messages(
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        sample_memos=sample_memos,
    )
    raw_payload = client.converse_json(
        system_prompt=messages[0]["content"],
        user_prompt=messages[1]["content"],
    )
    normalized = normalize_category_pattern_seed_output(raw_payload)

    return {
        "cate_1_depth": cate_1_depth,
        "cate_2_depth": cate_2_depth,
        "sc_measurement": int(sc_measurement),
        "sample_memo_count": len(sample_memos),
        "has_static_seed": has_static_category_patterns(cate_1_depth, cate_2_depth),
        "static_feature_hint_terms": get_static_category_feature_patterns(
            cate_1_depth,
            cate_2_depth,
        ),
        **normalized,
    }


def get_effective_category_pattern_seed(
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
    force_generate: bool = False,
) -> dict[str, Any]:
    """Return static seed when available, otherwise generate a dynamic seed."""
    if has_static_category_patterns(cate_1_depth, cate_2_depth) and not force_generate:
        return {
            "cate_1_depth": cate_1_depth,
            "cate_2_depth": cate_2_depth,
            "sc_measurement": int(sc_measurement),
            "sample_memo_count": 0 if sample_memos is None else len(sample_memos),
            "has_static_seed": True,
            "static_feature_hint_terms": get_static_category_feature_patterns(
                cate_1_depth,
                cate_2_depth,
            ),
            **build_static_category_pattern_seed(cate_1_depth, cate_2_depth),
        }

    return generate_category_pattern_seed(
        config=config,
        spark=spark,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        llm_client=llm_client,
        sample_memos=sample_memos,
        sample_seed=sample_seed,
        model_key=model_key,
    )
