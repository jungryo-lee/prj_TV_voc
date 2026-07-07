"""Topic classification engine driven by rule-profile and topic-pool outputs."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pyspark.sql import DataFrame, SparkSession, types as T
from pyspark.sql import functions as F

from common.config_loader import get_source_table, load_config
from common.llm_client import get_llm_client
from common.memo_id import with_memo_id
from taxonomy.prompt_builder import overall_topic_name


DEFAULT_CLASSIFICATION_SEED = "seed_20260707"

CLASSIFICATION_RESULT_SCHEMA = T.StructType(
    [
        T.StructField("memo_id", T.StringType(), False),
        T.StructField("memo", T.StringType(), True),
        T.StructField("memo_norm", T.StringType(), True),
        T.StructField("cate_1_depth", T.StringType(), True),
        T.StructField("cate_2_depth", T.StringType(), True),
        T.StructField("sc_measurement", T.IntegerType(), True),
        T.StructField("year", T.StringType(), True),
        T.StructField("country", T.StringType(), True),
        T.StructField("brand_name", T.StringType(), True),
        T.StructField("device_type", T.StringType(), True),
        T.StructField("pred_topic", T.StringType(), True),
        T.StructField("pred_topic_type", T.StringType(), True),
        T.StructField("classification_stage", T.StringType(), True),
        T.StructField("confidence_score", T.DoubleType(), True),
        T.StructField("candidate_topics_json", T.StringType(), True),
        T.StructField("match_reason", T.StringType(), True),
        T.StructField("llm_used_yn", T.BooleanType(), True),
        T.StructField("review_needed_yn", T.BooleanType(), True),
        T.StructField("run_id", T.StringType(), True),
        T.StructField("run_date", T.StringType(), True),
        T.StructField("prompt_version", T.StringType(), True),
        T.StructField("taxonomy_version", T.StringType(), True),
        T.StructField("model_version", T.StringType(), True),
        T.StructField("pipeline_version", T.StringType(), True),
        T.StructField("created_at", T.StringType(), True),
    ]
)


def _sql_escape(value: str) -> str:
    """Escape a string for simple SQL literal interpolation."""
    return str(value).replace("'", "''")


def _clean_text(value: Any) -> str:
    """Collapse whitespace for stable downstream matching."""
    return " ".join(str(value or "").split()).strip()


def _clean_term_list(values: list[Any] | None, *, max_items: int = 200) -> list[str]:
    """Normalize list-like inputs into unique compact strings."""
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


def _parse_json_list_field(value: Any) -> list[str]:
    """Parse JSON-string/list payloads into compact string lists."""
    if value is None:
        return []
    if isinstance(value, list):
        return _clean_term_list(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except Exception:
            return [raw]
        if isinstance(parsed, list):
            return _clean_term_list(parsed)
        return [_clean_text(parsed)]
    return [_clean_text(value)]


def _tokenize(text: str) -> set[str]:
    """Tokenize normalized text into a small lexical set."""
    return {token for token in _clean_text(text).lower().split() if token}


def _contains_any_term(text: str, terms: list[str]) -> bool:
    """Return whether any term is found in text using compact contains logic."""
    normalized = _clean_text(text).lower()
    return any(term.lower() in normalized for term in terms if _clean_text(term))


def _safe_json_dumps(value: Any) -> str:
    """Serialize Python values into a UTF-8 safe JSON string."""
    return json.dumps(value, ensure_ascii=False)


def get_classification_stage_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return effective classification settings with stable defaults."""
    taxonomy_cfg = config.get("taxonomy", {})
    classification_cfg = config.get("classification", {})

    return {
        "max_sample_rows_per_group": int(
            classification_cfg.get(
                "max_sample_rows_per_group",
                taxonomy_cfg.get("max_sample_rows", 1000),
            )
        ),
        "overall_max_text_length": int(
            classification_cfg.get("overall_max_text_length", 40)
        ),
        "topic_match_min_score": float(
            classification_cfg.get("topic_match_min_score", 0.35)
        ),
        "topic_match_min_margin": float(
            classification_cfg.get("topic_match_min_margin", 0.08)
        ),
        "review_required_default": bool(
            taxonomy_cfg.get("review_required_default", True)
        ),
        "classify_batch_size": int(classification_cfg.get("classify_batch_size", 25)),
    }


def build_classification_target_query(
    config: dict[str, Any],
    *,
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
) -> str:
    """Build the SQL used to load one classification target group."""
    source_table = get_source_table(config, "raw_review_table")
    cate_1 = _sql_escape(cate_1_depth)
    cate_2 = _sql_escape(cate_2_depth)

    return f"""
select
    cate_1_depth,
    cate_2_depth,
    sc_measurement,
    year,
    country,
    brand_name,
    device_type,
    memo
from {source_table}
where cate_1_depth = '{cate_1}'
  and cate_2_depth = '{cate_2}'
  and sc_measurement = {int(sc_measurement)}
  and memo is not null
  and length(trim(memo)) > 0
""".strip()


def prepare_classification_df(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    max_rows: int | None = None,
    sample_seed: str = DEFAULT_CLASSIFICATION_SEED,
) -> DataFrame:
    """Load and normalize target memos for one classification group."""
    stage_cfg = get_classification_stage_config(config)
    effective_max_rows = int(max_rows or stage_cfg["max_sample_rows_per_group"])

    base_df = spark.sql(
        build_classification_target_query(
            config,
            cate_1_depth=cate_1_depth,
            cate_2_depth=cate_2_depth,
            sc_measurement=sc_measurement,
        )
    ).transform(with_memo_id)

    sampled_df = (
        base_df.withColumn(
            "_sample_key",
            F.md5(
                F.concat(
                    F.coalesce(F.col("memo_id"), F.lit("")),
                    F.lit("||"),
                    F.lit(sample_seed),
                )
            ),
        )
        .orderBy("_sample_key")
        .drop("_sample_key")
    )

    if effective_max_rows > 0:
        sampled_df = sampled_df.limit(effective_max_rows)

    return sampled_df


def normalize_rule_profile(
    rule_profile: dict[str, Any],
    *,
    sc_measurement: int,
) -> dict[str, Any]:
    """Normalize stored rule-profile payloads into a classification shape."""
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


def normalize_topic_pool(topic_pool: dict[str, Any]) -> dict[str, Any]:
    """Normalize stored topic-pool payloads into stable topic rows."""
    raw_topics = topic_pool.get("topics") or []
    normalized_topics: list[dict[str, Any]] = []

    for row in raw_topics:
        if not isinstance(row, dict):
            continue
        topic_name = _clean_text(row.get("topic"))
        if not topic_name:
            continue
        normalized_topics.append(
            {
                "topic": topic_name,
                "description": _clean_text(row.get("description")),
                "representative_memos": _clean_term_list(
                    row.get("representative_memos"),
                    max_items=10,
                ),
            }
        )

    return {"topics": normalized_topics}


def apply_overall_rules(
    memo_text: str,
    rule_profile: dict[str, Any],
    *,
    overall_max_text_length: int,
) -> dict[str, Any]:
    """Evaluate whether a memo should be classified as overall sentiment."""
    memo_clean = _clean_text(memo_text)
    memo_lower = memo_clean.lower()
    feature_terms = rule_profile.get("feature_hint_terms", [])
    reason_terms = rule_profile.get("reason_signal_terms", [])
    overall_terms = rule_profile.get("overall_sentiment_terms", [])

    has_overall_term = _contains_any_term(memo_lower, overall_terms)
    has_feature_term = _contains_any_term(memo_lower, feature_terms)
    has_reason_term = _contains_any_term(memo_lower, reason_terms)
    is_short_text = len(memo_clean) <= int(overall_max_text_length)

    allowed = has_overall_term and is_short_text and not has_feature_term and not has_reason_term
    blocked = has_feature_term or has_reason_term

    if allowed:
        return {
            "is_overall": True,
            "stage": "rule_overall",
            "pred_topic": rule_profile.get("overall_topic_name"),
            "pred_topic_type": "overall",
            "confidence_score": 0.99,
            "match_reason": "pure_sentiment_short_memo",
        }

    if blocked and has_overall_term:
        return {
            "is_overall": False,
            "stage": "rule_overall_blocked",
            "pred_topic": "",
            "pred_topic_type": "",
            "confidence_score": 0.0,
            "match_reason": "feature_or_reason_detected",
        }

    return {
        "is_overall": False,
        "stage": "rule_non_overall",
        "pred_topic": "",
        "pred_topic_type": "",
        "confidence_score": 0.0,
        "match_reason": "no_overall_rule_match",
    }


def build_topic_candidates(
    memo_text: str,
    topic_pool: dict[str, Any],
    rule_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build lexical topic candidates for one memo."""
    memo_clean = _clean_text(memo_text)
    memo_tokens = _tokenize(memo_clean)
    candidates: list[dict[str, Any]] = []

    for topic_row in topic_pool.get("topics", []):
        topic_name = topic_row["topic"]
        if topic_name == rule_profile.get("overall_topic_name"):
            continue

        topic_terms = _tokenize(topic_name)
        description_terms = _tokenize(topic_row.get("description", ""))
        representative_terms = _tokenize(
            " ".join(topic_row.get("representative_memos", []))
        )

        topic_overlap = len(memo_tokens & topic_terms)
        description_overlap = len(memo_tokens & description_terms)
        representative_overlap = len(memo_tokens & representative_terms)

        feature_bonus = 0.0
        for feature_term in rule_profile.get("feature_hint_terms", []):
            normalized_feature = feature_term.lower()
            if normalized_feature and normalized_feature in memo_clean.lower():
                if normalized_feature in topic_name.lower():
                    feature_bonus += 0.25
                elif normalized_feature in topic_row.get("description", "").lower():
                    feature_bonus += 0.10

        score = (
            (0.50 * topic_overlap)
            + (0.30 * description_overlap)
            + (0.20 * representative_overlap)
            + feature_bonus
        )

        if score <= 0:
            continue

        candidates.append(
            {
                "topic": topic_name,
                "score": round(float(score), 4),
                "topic_overlap": int(topic_overlap),
                "description_overlap": int(description_overlap),
                "representative_overlap": int(representative_overlap),
                "feature_bonus": round(float(feature_bonus), 4),
            }
        )

    candidates.sort(key=lambda row: (-row["score"], row["topic"]))
    return candidates


def resolve_topic_decision(
    candidates: list[dict[str, Any]],
    *,
    min_score: float,
    min_margin: float,
    review_required_default: bool,
) -> dict[str, Any]:
    """Resolve topic candidates into final topic / ambiguous / others."""
    if not candidates:
        return {
            "pred_topic": "기타",
            "pred_topic_type": "others",
            "classification_stage": "forced_others",
            "confidence_score": 0.0,
            "review_needed_yn": review_required_default,
            "match_reason": "no_topic_candidate",
        }

    top1 = candidates[0]
    top2_score = float(candidates[1]["score"]) if len(candidates) > 1 else 0.0
    top1_score = float(top1["score"])
    margin = top1_score - top2_score

    if top1_score >= float(min_score) and margin >= float(min_margin):
        return {
            "pred_topic": top1["topic"],
            "pred_topic_type": "topic",
            "classification_stage": "heuristic_topic_match",
            "confidence_score": top1_score,
            "review_needed_yn": False,
            "match_reason": f"top_score={top1_score:.4f};margin={margin:.4f}",
        }

    return {
        "pred_topic": "",
        "pred_topic_type": "ambiguous",
        "classification_stage": "ambiguous_candidate_match",
        "confidence_score": top1_score,
        "review_needed_yn": True,
        "match_reason": f"top_score={top1_score:.4f};margin={margin:.4f}",
    }


def apply_llm_fallback(
    memo_text: str,
    *,
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    rule_profile: dict[str, Any],
    topic_pool: dict[str, Any],
    candidate_topics: list[dict[str, Any]],
    config: dict[str, Any],
    model_key: str | None = None,
) -> dict[str, Any]:
    """Use LLM only for hard ambiguous cases."""
    llm_client = get_llm_client(config=config, model_key=model_key)
    candidate_names = [row["topic"] for row in candidate_topics[:5]]

    system_prompt = """
You are classifying one VOC memo into a topic taxonomy.
Return strict JSON only with keys:
- pred_topic
- pred_topic_type
- match_reason
""".strip()

    user_prompt = f"""
[Group]
- cate_1_depth: {cate_1_depth}
- cate_2_depth: {cate_2_depth}
- sc_measurement: {int(sc_measurement)}

[Overall topic]
- overall_topic_name: {rule_profile.get("overall_topic_name", "")}
- overall_allowed_rule: {rule_profile.get("overall_allowed_rule", "")}
- overall_block_rule: {rule_profile.get("overall_block_rule", "")}

[Available topics]
{_safe_json_dumps(topic_pool.get("topics", []))}

[Top candidates]
{_safe_json_dumps(candidate_topics[:5])}

[Memo]
{memo_text}

Rules:
- Use pred_topic_type among overall, topic, others.
- If none fits clearly, return pred_topic as 기타 and pred_topic_type as others.
- If you pick topic, it must be one of the available topics.
- Return JSON only.
""".strip()

    payload = llm_client.converse_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    pred_topic = _clean_text(payload.get("pred_topic"))
    pred_topic_type = _clean_text(payload.get("pred_topic_type")).lower()
    match_reason = _clean_text(payload.get("match_reason"))

    valid_topics = {row["topic"] for row in topic_pool.get("topics", [])}
    if pred_topic not in valid_topics and pred_topic != "기타":
        pred_topic = "기타"
        pred_topic_type = "others"
        match_reason = "llm_returned_unknown_topic"

    if pred_topic_type not in {"overall", "topic", "others"}:
        pred_topic_type = "others"

    return {
        "pred_topic": pred_topic or "기타",
        "pred_topic_type": pred_topic_type or "others",
        "classification_stage": "llm_fallback",
        "confidence_score": None,
        "review_needed_yn": pred_topic == "기타",
        "match_reason": match_reason or "llm_fallback_applied",
        "llm_used_yn": True,
    }


def normalize_classification_result(
    base_row: dict[str, Any],
    *,
    decision: dict[str, Any],
    candidate_topics: list[dict[str, Any]],
    config: dict[str, Any],
    model_key: str | None,
    llm_used_yn: bool,
) -> dict[str, Any]:
    """Build one stable classification result row."""
    version_cfg = config.get("version", {})
    runtime_cfg = config.get("runtime", {})
    llm_cfg = config.get("llm", {})
    model_cfg = llm_cfg.get("models", {}).get(model_key or llm_cfg.get("default_model_key"), {})

    return {
        "memo_id": base_row["memo_id"],
        "memo": base_row.get("memo"),
        "memo_norm": base_row.get("memo_norm"),
        "cate_1_depth": base_row.get("cate_1_depth"),
        "cate_2_depth": base_row.get("cate_2_depth"),
        "sc_measurement": int(base_row.get("sc_measurement")),
        "year": _clean_text(base_row.get("year")),
        "country": _clean_text(base_row.get("country")),
        "brand_name": _clean_text(base_row.get("brand_name")),
        "device_type": _clean_text(base_row.get("device_type")),
        "pred_topic": decision.get("pred_topic") or "기타",
        "pred_topic_type": decision.get("pred_topic_type") or "others",
        "classification_stage": decision.get("classification_stage"),
        "confidence_score": decision.get("confidence_score"),
        "candidate_topics_json": _safe_json_dumps(candidate_topics[:5]),
        "match_reason": decision.get("match_reason"),
        "llm_used_yn": bool(llm_used_yn),
        "review_needed_yn": bool(decision.get("review_needed_yn")),
        "run_id": runtime_cfg.get("resolved_run_id"),
        "run_date": runtime_cfg.get("resolved_run_date"),
        "prompt_version": version_cfg.get("prompt_version"),
        "taxonomy_version": version_cfg.get("taxonomy_version"),
        "model_version": model_cfg.get(
            "model_version",
            version_cfg.get("model_version"),
        ),
        "pipeline_version": version_cfg.get("pipeline_version"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def classify_topic_for_group(
    spark: SparkSession,
    *,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    rule_profile: dict[str, Any],
    topic_pool: dict[str, Any],
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    model_key: str | None = None,
    max_rows: int | None = None,
    sample_seed: str = DEFAULT_CLASSIFICATION_SEED,
    use_llm_fallback: bool = False,
) -> dict[str, Any]:
    """Classify memos for one taxonomy group using current design artifacts."""
    effective_config = config or load_config(config_path)
    stage_cfg = get_classification_stage_config(effective_config)
    normalized_rule_profile = normalize_rule_profile(
        rule_profile,
        sc_measurement=sc_measurement,
    )
    normalized_topic_pool = normalize_topic_pool(topic_pool)
    target_df = prepare_classification_df(
        spark,
        effective_config,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        max_rows=max_rows,
        sample_seed=sample_seed,
    )

    result_rows: list[dict[str, Any]] = []
    overall_count = 0
    topic_count = 0
    others_count = 0
    ambiguous_count = 0
    llm_used_count = 0

    for row in target_df.toLocalIterator():
        base_row = row.asDict(recursive=True)
        memo_text = _clean_text(base_row.get("memo"))

        overall_decision = apply_overall_rules(
            memo_text,
            normalized_rule_profile,
            overall_max_text_length=stage_cfg["overall_max_text_length"],
        )
        llm_used_yn = False

        if overall_decision["is_overall"]:
            decision = {
                "pred_topic": overall_decision["pred_topic"],
                "pred_topic_type": overall_decision["pred_topic_type"],
                "classification_stage": overall_decision["stage"],
                "confidence_score": overall_decision["confidence_score"],
                "review_needed_yn": False,
                "match_reason": overall_decision["match_reason"],
            }
            candidate_topics: list[dict[str, Any]] = []
        else:
            candidate_topics = build_topic_candidates(
                memo_text,
                normalized_topic_pool,
                normalized_rule_profile,
            )
            decision = resolve_topic_decision(
                candidate_topics,
                min_score=stage_cfg["topic_match_min_score"],
                min_margin=stage_cfg["topic_match_min_margin"],
                review_required_default=stage_cfg["review_required_default"],
            )

            if use_llm_fallback and decision["pred_topic_type"] == "ambiguous":
                decision = apply_llm_fallback(
                    memo_text,
                    cate_1_depth=cate_1_depth,
                    cate_2_depth=cate_2_depth,
                    sc_measurement=sc_measurement,
                    rule_profile=normalized_rule_profile,
                    topic_pool=normalized_topic_pool,
                    candidate_topics=candidate_topics,
                    config=effective_config,
                    model_key=model_key,
                )
                llm_used_yn = True

        result_row = normalize_classification_result(
            base_row,
            decision=decision,
            candidate_topics=candidate_topics,
            config=effective_config,
            model_key=model_key,
            llm_used_yn=llm_used_yn,
        )
        result_rows.append(result_row)

        pred_type = result_row["pred_topic_type"]
        if pred_type == "overall":
            overall_count += 1
        elif pred_type == "topic":
            topic_count += 1
        elif pred_type == "others":
            others_count += 1
        else:
            ambiguous_count += 1

        if llm_used_yn:
            llm_used_count += 1

    result_df = spark.createDataFrame(result_rows, schema=CLASSIFICATION_RESULT_SCHEMA)

    return {
        "cate_1_depth": cate_1_depth,
        "cate_2_depth": cate_2_depth,
        "sc_measurement": int(sc_measurement),
        "row_count": len(result_rows),
        "overall_count": overall_count,
        "topic_count": topic_count,
        "others_count": others_count,
        "ambiguous_count": ambiguous_count,
        "llm_used_count": llm_used_count,
        "model_key": model_key or effective_config.get("llm", {}).get("default_model_key"),
        "result_df": result_df,
        "rows": result_rows,
    }


def classify_topics_for_groups(
    spark: SparkSession,
    *,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    group_payloads: list[dict[str, Any]],
    model_key: str | None = None,
    max_rows: int | None = None,
    use_llm_fallback: bool = False,
) -> dict[str, Any]:
    """Run classification for multiple groups and union the results."""
    effective_config = config or load_config(config_path)
    all_rows: list[dict[str, Any]] = []
    group_results: list[dict[str, Any]] = []

    for payload in group_payloads:
        group_result = classify_topic_for_group(
            spark,
            config=effective_config,
            rule_profile=payload["rule_profile"],
            topic_pool=payload["topic_pool"],
            cate_1_depth=payload["cate_1_depth"],
            cate_2_depth=payload["cate_2_depth"],
            sc_measurement=int(payload["sc_measurement"]),
            model_key=model_key,
            max_rows=max_rows,
            use_llm_fallback=use_llm_fallback,
        )
        group_results.append(
            {key: value for key, value in group_result.items() if key != "result_df" and key != "rows"}
        )
        all_rows.extend(group_result["rows"])

    result_df = spark.createDataFrame(all_rows, schema=CLASSIFICATION_RESULT_SCHEMA)

    return {
        "group_count": len(group_results),
        "row_count": len(all_rows),
        "result_df": result_df,
        "group_results": group_results,
        "rows": all_rows,
    }
