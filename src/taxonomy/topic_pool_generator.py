"""Topic-pool generation orchestration for taxonomy groups."""

from __future__ import annotations

import json
from typing import Any

from common.llm_client import DatabricksLLMClient, get_llm_client
from taxonomy.group_sampler import collect_diverse_topic_pool_prompt_memos
from taxonomy.prompt_builder import build_topic_pool_messages, overall_topic_name


def _clean_text(value: Any) -> str:
    """Collapse whitespace for stable downstream payloads."""
    return " ".join(str(value or "").split()).strip()


def _clean_term_list(
    values: list[Any] | None,
    *,
    max_items: int,
) -> list[str]:
    """Normalize list-like values into unique compact strings."""
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


def get_topic_pool_prompt_memo_limit(
    config: dict[str, Any],
    *,
    model_key: str | None,
) -> int:
    """Return the configured prompt memo cap for topic-pool generation."""
    topic_pool_cfg = config.get("topic_pool", {})
    max_by_model = topic_pool_cfg.get("max_prompt_memos_by_model", {}) or {}

    if model_key and model_key in max_by_model:
        return int(max_by_model[model_key])

    if "max_prompt_memos_default" in topic_pool_cfg:
        return int(topic_pool_cfg["max_prompt_memos_default"])

    return int(config.get("taxonomy", {}).get("max_sample_rows", 1000))


def get_topic_pool_topic_count_limits(
    config: dict[str, Any],
    *,
    cate_1_depth: str,
    cate_2_depth: str,
) -> tuple[int, int]:
    """Return min/max topic count with optional category-level overrides."""
    taxonomy_cfg = config.get("taxonomy", {})
    topic_pool_cfg = config.get("topic_pool", {})
    default_min = int(taxonomy_cfg.get("min_final_topics", 7))
    default_max = int(taxonomy_cfg.get("max_final_topics", 17))

    override_key = f"{cate_1_depth}||{cate_2_depth}"
    overrides = topic_pool_cfg.get("topic_count_overrides", {}) or {}
    override = overrides.get(override_key, {}) or {}

    min_topics = int(override.get("min_final_topics", default_min))
    max_topics = int(override.get("max_final_topics", default_max))
    return min_topics, max_topics


def _parse_json_list_field(value: Any) -> list[str]:
    """Parse a JSON string/list payload into a compact string list."""
    if value is None:
        return []
    if isinstance(value, list):
        return _clean_term_list(value, max_items=200)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except Exception:
            return [raw]
        if isinstance(parsed, list):
            return _clean_term_list(parsed, max_items=200)
        return [_clean_text(parsed)]
    return [_clean_text(value)]


def normalize_rule_profile_for_topic_pool(
    rule_profile: dict[str, Any],
    *,
    sc_measurement: int,
) -> dict[str, Any]:
    """Normalize rule-profile inputs for topic-pool prompt injection."""
    return {
        "overall_topic_name": _clean_text(
            rule_profile.get("overall_topic_name") or overall_topic_name(sc_measurement)
        ),
        "overall_allowed_rule": _clean_text(rule_profile.get("overall_allowed_rule")),
        "overall_block_rule": _clean_text(rule_profile.get("overall_block_rule")),
        "overall_sentiment_terms": _parse_json_list_field(
            rule_profile.get("overall_sentiment_terms")
            or rule_profile.get("overall_sentiment_terms_json")
        ),
        "feature_hint_terms": _parse_json_list_field(
            rule_profile.get("feature_hint_terms")
            or rule_profile.get("feature_hint_terms_json")
        ),
        "reason_signal_terms": _parse_json_list_field(
            rule_profile.get("reason_signal_terms")
            or rule_profile.get("reason_signal_terms_json")
        ),
        "non_overall_examples": _parse_json_list_field(
            rule_profile.get("non_overall_examples")
            or rule_profile.get("non_overall_examples_json")
        ),
    }


def normalize_topic_pool_output(
    payload: dict[str, Any],
    *,
    overall_topic: str,
    min_final_topics: int,
    max_final_topics: int,
) -> dict[str, Any]:
    """Normalize raw LLM JSON into a stable topic-pool schema."""
    topics = payload.get("topics") or []
    normalized_topics: list[dict[str, Any]] = []
    seen_topics: set[str] = set()

    for topic_row in topics:
        if not isinstance(topic_row, dict):
            continue
        topic_name = _clean_text(topic_row.get("topic"))
        if not topic_name:
            continue
        topic_key = topic_name.lower()
        if topic_key in seen_topics:
            continue
        seen_topics.add(topic_key)
        normalized_topics.append(
            {
                "topic": topic_name,
                "description": _clean_text(topic_row.get("description")),
                "representative_memos": _clean_term_list(
                    topic_row.get("representative_memos"),
                    max_items=5,
                ),
            }
        )
        if len(normalized_topics) >= int(max_final_topics):
            break

    if overall_topic and overall_topic.lower() not in seen_topics:
        normalized_topics.insert(
            0,
            {
                "topic": overall_topic,
                "description": "특별한 구체 사유 없이 전반적 감정만 표현한 리뷰",
                "representative_memos": [],
            },
        )
        normalized_topics = normalized_topics[: int(max_final_topics)]

    return {
        "topics": normalized_topics,
        "topic_count": len(normalized_topics),
        "meets_min_topic_count": len(normalized_topics) >= int(min_final_topics),
        "meets_max_topic_count": len(normalized_topics) <= int(max_final_topics),
    }


def build_topic_pool_result(
    *,
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    sample_memos: list[str],
    rule_profile: dict[str, Any],
    llm_client: DatabricksLLMClient,
    config: dict[str, Any],
    model_key: str | None = None,
    original_sample_memo_count: int | None = None,
) -> dict[str, Any]:
    """Build one normalized topic-pool result from sampled memos."""
    topic_pool_cfg = config.get("topic_pool", {})
    min_final_topics, max_final_topics = get_topic_pool_topic_count_limits(
        config,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
    )
    normalized_rule_profile = normalize_rule_profile_for_topic_pool(
        rule_profile,
        sc_measurement=sc_measurement,
    )

    messages = build_topic_pool_messages(
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        sample_memos=sample_memos,
        rule_profile=normalized_rule_profile,
        min_final_topics=min_final_topics,
        max_final_topics=max_final_topics,
    )
    raw_payload = llm_client.converse_json(
        system_prompt=messages[0]["content"],
        user_prompt=messages[1]["content"],
    )
    normalized = normalize_topic_pool_output(
        raw_payload,
        overall_topic=normalized_rule_profile["overall_topic_name"],
        min_final_topics=min_final_topics,
        max_final_topics=max_final_topics,
    )
    if topic_pool_cfg.get("raise_on_invalid_topic_count", True):
        if not normalized["meets_min_topic_count"]:
            raise ValueError(
                f"Generated topic count {normalized['topic_count']} is below the configured "
                f"minimum {min_final_topics} for {cate_1_depth} / {cate_2_depth}."
            )
        if not normalized["meets_max_topic_count"]:
            raise ValueError(
                f"Generated topic count {normalized['topic_count']} exceeds the configured "
                f"maximum {max_final_topics} for {cate_1_depth} / {cate_2_depth}."
            )

    return {
        "cate_1_depth": cate_1_depth,
        "cate_2_depth": cate_2_depth,
        "sc_measurement": int(sc_measurement),
        "sample_memo_count": len(sample_memos),
        "min_final_topics": int(min_final_topics),
        "max_final_topics": int(max_final_topics),
        "original_sample_memo_count": (
            int(original_sample_memo_count)
            if original_sample_memo_count is not None
            else len(sample_memos)
        ),
        "model_key": model_key or llm_client.model_key,
        "rule_profile_used": normalized_rule_profile,
        **normalized,
    }


def generate_topic_pool_for_group(
    *,
    config: dict[str, Any],
    spark,
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    rule_profile: dict[str, Any],
    llm_client: DatabricksLLMClient | None = None,
    sample_memos: list[str] | None = None,
    sample_seed: str | None = None,
    model_key: str | None = None,
) -> dict[str, Any]:
    """Generate a topic pool for one category/sentiment group."""
    client = llm_client or get_llm_client(config=config, model_key=model_key)
    selected_model_key = model_key or client.model_key
    prompt_memo_limit = get_topic_pool_prompt_memo_limit(
        config,
        model_key=selected_model_key,
    )

    if sample_memos is None:
        sample_memos = collect_diverse_topic_pool_prompt_memos(
            spark=spark,
            config=config,
            cate_1_depth=cate_1_depth,
            cate_2_depth=cate_2_depth,
            sc_measurement=sc_measurement,
            max_prompt_rows=prompt_memo_limit,
            sample_seed=sample_seed or "seed_20260420",
        )

    original_sample_memo_count = len(sample_memos)

    return build_topic_pool_result(
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        sample_memos=sample_memos,
        rule_profile=rule_profile,
        llm_client=client,
        config=config,
        model_key=selected_model_key,
        original_sample_memo_count=original_sample_memo_count,
    )


def generate_topic_pools_for_groups(
    *,
    config: dict[str, Any],
    spark,
    group_rows: list[dict[str, Any]],
    rule_profile_map: dict[tuple[str, str, int], dict[str, Any]],
    llm_client: DatabricksLLMClient | None = None,
    sample_seed: str | None = None,
    model_key: str | None = None,
) -> list[dict[str, Any]]:
    """Generate topic pools for multiple group rows."""
    client = llm_client or get_llm_client(config=config, model_key=model_key)
    results: list[dict[str, Any]] = []

    for group_row in group_rows:
        key = (
            group_row["cate_1_depth"],
            group_row["cate_2_depth"],
            int(group_row["sc_measurement"]),
        )
        if key not in rule_profile_map:
            continue

        results.append(
            generate_topic_pool_for_group(
                config=config,
                spark=spark,
                cate_1_depth=group_row["cate_1_depth"],
                cate_2_depth=group_row["cate_2_depth"],
                sc_measurement=int(group_row["sc_measurement"]),
                rule_profile=rule_profile_map[key],
                llm_client=client,
                sample_seed=sample_seed,
                model_key=model_key,
            )
        )

    return results
