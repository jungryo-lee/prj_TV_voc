"""Topic classification engine driven by rule-profile and topic-pool outputs."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from pyspark.sql import DataFrame, SparkSession, types as T
from pyspark.sql import functions as F

from common.config_loader import get_source_table, load_config
from common.llm_client import get_llm_client
from common.memo_id import with_memo_id
from taxonomy.prompt_builder import overall_topic_name


DEFAULT_CLASSIFICATION_SEED = "seed_20260707"

CATEGORY_NEUTRAL_TARGET_TERMS: dict[str, list[str]] = {
    "07-06. 리모컨 사용성": [
        "리모컨",
        "리모콘",
        "매직리모컨",
        "매직 리모컨",
        "스마트 리모컨",
        "remote",
        "remote control",
        "smart remote",
        "magic remote",
        "control",
        "controls",
        "controller",
    ]
}

GENERIC_NEUTRAL_TERM_HINTS: dict[str, list[str]] = {
    "app": ["app", "apps", "application", "applications", "ott", "streaming"],
    "컨텐츠": ["컨텐츠", "콘텐츠", "content", "contents"],
    "채널": ["채널", "channel", "channels"],
    "메뉴": ["메뉴", "menu", "menus"],
    "ui": ["ui", "interface", "screen", "home screen"],
    "sw": ["software", "sw", "os", "system", "firmware", "platform"],
    "게임": ["game", "gaming", "games", "게임"],
    "음성": ["voice", "speech", "audio", "음성"],
    "모바일": ["mobile", "phone", "smartphone", "tablet", "모바일"],
    "iot": ["iot", "home iot", "smart home", "device connection", "연동"],
    "광고": ["ad", "ads", "advertisement", "advertising", "광고"],
    "전반적": ["overall", "general", "전반", "전체"],
}

GENERIC_OVERALL_REASON_HINTS: list[str] = [
    "easy",
    "easier",
    "intuitive",
    "simple",
    "convenient",
    "fast",
    "faster",
    "responsive",
    "backlit",
    "solar",
    "usb-c",
    "voice",
    "shortcut",
    "button",
    "layout",
    "menu",
    "search",
    "navigation",
    "update",
    "bug",
    "lag",
    "many apps",
    "few buttons",
    "controls all devices",
    "all devices",
    "set top box",
    "easy to use",
    "직관적",
    "편리",
    "간편",
    "빠르",
    "반응",
    "버튼",
    "검색",
    "탐색",
    "업데이트",
]

GENERIC_OVERALL_POSITIVE_PATTERNS: list[str] = [
    "great {target}",
    "good {target}",
    "{target} is great",
    "{target} is good",
    "{target} is awesome",
    "{target} is excellent",
    "love the {target}",
]

TOPIC_DIRECT_SIGNAL_HINTS: dict[str, list[str]] = {
    "앱 바로가기 버튼": [
        "app",
        "apps",
        "application",
        "applications",
        "ott",
        "netflix",
        "prime",
        "prime video",
        "youtube",
        "disney",
        "shortcut",
        "quick button",
        "dedicated button",
        "direct button",
        "바로가기",
        "전용 버튼",
        "앱 버튼",
        "넷플릭스 버튼",
        "유튜브 버튼",
    ],
    "통합 리모컨/외부기기 제어": [
        "one remote",
        "single remote",
        "all my devices",
        "all devices",
        "everything",
        "external device",
        "external devices",
        "set top box",
        "decoder",
        "virgin box",
        "ziggobox",
        "dvd player",
        "sound bar",
        "amazon fire tv",
        "외부기기",
        "셋톱박스",
        "통합",
        "하나의 리모컨",
        "모든 기기",
    ],
    "음성 제어/마이크": [
        "voice",
        "voice control",
        "voice command",
        "voice button",
        "voice search",
        "microphone",
        "mic",
        "alexa",
        "google assistant",
        "음성",
        "마이크",
        "음성 버튼",
        "음성검색",
    ],
    "충전식/태양광 배터리": [
        "solar",
        "solar cell",
        "solar powered",
        "solar charging",
        "usb-c",
        "usb charged",
        "rechargeable",
        "battery",
        "charge",
        "charging",
        "태양광",
        "충전",
        "충전식",
        "배터리",
    ],
    "포인터/마우스 조작": [
        "pointer",
        "cursor",
        "mouse",
        "motion sensor",
        "motion remote",
        "air mouse",
        "pointer remote",
        "포인터",
        "커서",
        "마우스",
        "모션",
    ],
    "쉬운 조작/직관적 탐색": [
        "easy to use",
        "easy for me",
        "easy",
        "intuitive",
        "simple",
        "convenient",
        "navigate",
        "navigation",
        "easy access",
        "straightforward",
        "직관적",
        "편리",
        "간편",
        "쉽다",
        "조작이 쉽",
        "메뉴 이동",
        "탐색",
    ],
    "빠른 반응/원활한 작동": [
        "responsive",
        "response",
        "fast",
        "quick",
        "works well",
        "works perfectly",
        "reacts quickly",
        "반응",
        "빠르",
        "원활",
        "잘 작동",
    ],
    "버튼 구성/레이아웃": [
        "button layout",
        "buttons",
        "button",
        "number keys",
        "setup",
        "set up",
        "layout",
        "rubbery button",
        "버튼",
        "버튼 구성",
        "버튼 배열",
        "숫자버튼",
        "레이아웃",
    ],
    "그립감/크기/무게": [
        "fits perfectly in the hand",
        "in the hand",
        "small",
        "minimal",
        "slim",
        "light",
        "lightweight",
        "handy",
        "grip",
        "가볍",
        "작다",
        "슬림",
        "그립",
        "손에",
    ],
}

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


def _extract_label_tokens(label: str) -> list[str]:
    """Extract compact lexical tokens from a category/topic label."""
    cleaned = re.sub(r"^\d+\-\d+\.\s*", "", _clean_text(label))
    parts = re.split(r"[\/\(\)\-\_,\s]+", cleaned)
    tokens = [part.strip() for part in parts if part.strip()]
    return tokens


def _expand_generic_neutral_terms(tokens: list[str]) -> list[str]:
    """Expand category label tokens into generic neutral target terms."""
    expanded: list[str] = []
    for token in tokens:
        lowered = token.lower()
        expanded.append(token)
        for generic_key, values in GENERIC_NEUTRAL_TERM_HINTS.items():
            if generic_key in lowered or generic_key in token:
                expanded.extend(values)
    return _clean_term_list(expanded, max_items=50)


def build_neutral_target_terms(
    cate_1_depth: str,
    cate_2_depth: str,
    *,
    rule_profile: dict[str, Any] | None = None,
    topic_pool: dict[str, Any] | None = None,
) -> list[str]:
    """Build neutral category target terms from category labels and artifacts.

    These terms represent the *target object* of the category and should not
    block overall sentiment by themselves.
    """
    terms: list[str] = []
    terms.extend(CATEGORY_NEUTRAL_TARGET_TERMS.get(cate_2_depth, []))
    terms.extend(_extract_label_tokens(cate_1_depth))
    terms.extend(_extract_label_tokens(cate_2_depth))
    terms.extend(
        _expand_generic_neutral_terms(
            _extract_label_tokens(cate_1_depth) + _extract_label_tokens(cate_2_depth)
        )
    )

    if rule_profile:
        feature_terms = rule_profile.get("feature_hint_terms", [])[:30]
        for term in feature_terms:
            cleaned = _clean_text(term)
            if cleaned and len(cleaned.split()) <= 2:
                terms.append(cleaned)

    if topic_pool:
        for topic_row in (topic_pool.get("topics") or [])[:20]:
            terms.extend(_extract_label_tokens(topic_row.get("topic", "")))

    return _clean_term_list(terms, max_items=80)


def build_dynamic_overall_examples(
    neutral_terms: list[str],
    *,
    sentiment_terms: list[str] | None = None,
) -> list[str]:
    """Build category-aware overall positive examples for the LLM prompt."""
    target_terms = [term for term in neutral_terms if len(term) >= 2][:6]
    examples: list[str] = []
    for target in target_terms:
        lowered = target.lower()
        if lowered in {"app", "apps", "application", "applications"}:
            examples.extend(["great apps", "good apps", "apps are great"])
        elif lowered in {"channel", "channels", "채널"}:
            examples.extend(["great channels", "good channels"])
        elif lowered in {"content", "contents", "컨텐츠", "콘텐츠"}:
            examples.extend(["great content", "good content"])

        for pattern in GENERIC_OVERALL_POSITIVE_PATTERNS:
            examples.append(pattern.format(target=target))

    if sentiment_terms:
        for sentiment in sentiment_terms[:5]:
            cleaned = _clean_text(sentiment)
            if cleaned:
                examples.append(cleaned)

    return _clean_term_list(examples, max_items=20)


def _strip_neutral_target_terms(text: str, neutral_terms: list[str]) -> str:
    """Remove neutral category target nouns before reason/feature checks."""
    normalized = f" {_clean_text(text).lower()} "
    for term in sorted((_clean_text(v).lower() for v in neutral_terms if _clean_text(v)), key=len, reverse=True):
        normalized = normalized.replace(f" {term} ", " ")
    return " ".join(normalized.split())


def get_topic_direct_signal_hints(topic_name: str) -> list[str]:
    """Return direct signal terms that strongly imply a topic."""
    return TOPIC_DIRECT_SIGNAL_HINTS.get(topic_name, [])


def build_topic_direct_signal_hints(
    topic_name: str,
    *,
    topic_description: str = "",
    representative_memos: list[str] | None = None,
) -> list[str]:
    """Build topic signal hints from static hints plus topic artifacts."""
    hints: list[str] = []
    hints.extend(get_topic_direct_signal_hints(topic_name))
    hints.extend(_extract_label_tokens(topic_name))

    description_tokens = _extract_label_tokens(topic_description)
    hints.extend(_expand_generic_neutral_terms(description_tokens))
    hints.extend(description_tokens)

    for memo in (representative_memos or [])[:5]:
        memo_tokens = _extract_label_tokens(memo)
        hints.extend([token for token in memo_tokens if len(token) >= 4][:5])

    return _clean_term_list(hints, max_items=40)


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
        "candidate_topic_limit": int(classification_cfg.get("candidate_topic_limit", 5)),
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
    cate_1_depth: str,
    cate_2_depth: str,
    topic_pool: dict[str, Any] | None = None,
    overall_max_text_length: int,
) -> dict[str, Any]:
    """Evaluate whether a memo should be classified as overall sentiment."""
    memo_clean = _clean_text(memo_text)
    memo_lower = memo_clean.lower()
    neutral_terms = build_neutral_target_terms(
        cate_1_depth,
        cate_2_depth,
        rule_profile=rule_profile,
        topic_pool=topic_pool,
    )
    memo_without_target = _strip_neutral_target_terms(memo_lower, neutral_terms)
    feature_terms = rule_profile.get("feature_hint_terms", [])
    reason_terms = _clean_term_list(
        (rule_profile.get("reason_signal_terms", []) or []) + GENERIC_OVERALL_REASON_HINTS,
        max_items=200,
    )
    overall_terms = rule_profile.get("overall_sentiment_terms", [])

    has_overall_term = _contains_any_term(memo_lower, overall_terms)
    has_feature_term = _contains_any_term(memo_without_target, feature_terms)
    has_reason_term = _contains_any_term(memo_without_target, reason_terms)
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
    """Build heuristic topic candidates for one memo.

    This stage is intentionally conservative: it should produce a useful shortlist
    for downstream LLM selection, not aggressively force a final topic.
    """
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

        direct_signal_bonus = 0.0
        for feature_term in rule_profile.get("feature_hint_terms", []):
            normalized_feature = feature_term.lower()
            if normalized_feature and normalized_feature in memo_clean.lower():
                if normalized_feature in topic_name.lower():
                    direct_signal_bonus += 1.25
                elif normalized_feature in topic_row.get("description", "").lower():
                    direct_signal_bonus += 0.85

        topic_signal_hits: list[str] = []
        for hint in build_topic_direct_signal_hints(
            topic_name,
            topic_description=topic_row.get("description", ""),
            representative_memos=topic_row.get("representative_memos", []),
        ):
            normalized_hint = hint.lower()
            if normalized_hint and normalized_hint in memo_clean.lower():
                topic_signal_hits.append(hint)
                direct_signal_bonus += 2.0 if " " in normalized_hint else 1.4

        score = (
            (1.20 * topic_overlap)
            + (0.85 * description_overlap)
            + (0.05 * representative_overlap)
            + direct_signal_bonus
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
                "direct_signal_bonus": round(float(direct_signal_bonus), 4),
                "topic_signal_hits": topic_signal_hits[:5],
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
    """Use LLM to pick the final topic from a heuristic shortlist."""
    llm_client = get_llm_client(config=config, model_key=model_key)
    shortlist = candidate_topics[: int(get_classification_stage_config(config)["candidate_topic_limit"])]
    candidate_names = [row["topic"] for row in shortlist]
    available_topics = topic_pool.get("topics", [])
    if candidate_names:
        allowed_topic_rows = [
            row for row in available_topics if row.get("topic") in set(candidate_names)
        ]
    else:
        allowed_topic_rows = [
            row
            for row in available_topics
            if row.get("topic") != rule_profile.get("overall_topic_name")
        ]

    neutral_terms = build_neutral_target_terms(
        cate_1_depth,
        cate_2_depth,
        rule_profile=rule_profile,
        topic_pool=topic_pool,
    )
    overall_examples = build_dynamic_overall_examples(
        neutral_terms,
        sentiment_terms=rule_profile.get("overall_sentiment_terms", []),
    )

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

[Allowed topics for final selection]
{_safe_json_dumps(allowed_topic_rows)}

[Top candidates]
{_safe_json_dumps(shortlist)}

[Neutral category target terms]
{_safe_json_dumps(neutral_terms[:20])}

[Memo]
{memo_text}

Rules:
- Use pred_topic_type among overall, topic, others.
- If none fits clearly, return pred_topic as 기타 and pred_topic_type as others.
- If you pick topic, it must be one of the allowed topics above.
- Mentioning the category target itself alone does not block overall.
- If the memo only says the target object is good/bad/great/awesome without a concrete reason,
  classify it as overall.
- Do not use overall when the memo gives a specific reason such as easy, intuitive,
  voice, app button, Netflix, Prime Video, backlit, solar, USB-C, one remote,
  set top box, external device control, cursor, pointer, typing, navigation.
- Use overall for short pure praise/complaint about the category target itself when there is no concrete reason.
- Overall positive examples:
{_safe_json_dumps(overall_examples)}
- Topic priority hints:
  If the memo mentions Netflix, Prime Video, YouTube, OTT, app shortcut, direct app access,
  or dedicated app buttons, prefer 앱 바로가기 버튼.
  If the memo mentions one remote, single remote, all devices, set top box, decoder,
  virgin box, DVD player, sound bar, Fire TV, or combining remotes, prefer 통합 리모컨/외부기기 제어.
  If the memo mentions voice button, voice command, Alexa, microphone, or voice search,
  prefer 음성 제어/마이크.
  If the memo mentions solar, USB-C, charging, rechargeable, or battery, prefer 충전식/태양광 배터리.
  If the memo mentions backlight, lighting in the dark, or illuminated buttons, prefer 백라이트/야간 사용.
  If both app-button and button-layout cues are present, choose 앱 바로가기 버튼 when the
  user's point is fast access to apps/services; choose 버튼 구성/레이아웃 only when the
  point is button count, spacing, arrangement, or redundancy itself.
  If both one-remote and easy-to-use cues are present, choose 통합 리모컨/외부기기 제어 when
  the main point is unified control across devices; choose 쉬운 조작/직관적 탐색 only when
  the main point is ease/intuition without multi-device control.
- Prefer the most specific topic when the memo mentions a concrete function or feature.
- Return JSON only.
""".strip()

    payload = llm_client.converse_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    pred_topic = _clean_text(payload.get("pred_topic"))
    pred_topic_type = _clean_text(payload.get("pred_topic_type")).lower()
    match_reason = _clean_text(payload.get("match_reason"))

    valid_topics = {row["topic"] for row in allowed_topic_rows}
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


def rescue_others_from_match_reason(
    decision: dict[str, Any],
    *,
    topic_pool: dict[str, Any],
) -> dict[str, Any]:
    """Promote 'others' to a real topic when match_reason names one valid topic.

    This is a conservative post-process:
    - applies only to llm_fallback -> others
    - requires exactly one topic name from the actual topic_pool in match_reason
    """
    pred_topic_type = _clean_text(decision.get("pred_topic_type")).lower()
    classification_stage = _clean_text(decision.get("classification_stage"))
    match_reason = _clean_text(decision.get("match_reason"))

    if pred_topic_type != "others":
        return decision
    if classification_stage != "llm_fallback":
        return decision
    if not match_reason:
        return decision

    valid_topics = [
        _clean_text(row.get("topic"))
        for row in (topic_pool.get("topics") or [])
        if _clean_text(row.get("topic")) and _clean_text(row.get("topic")) != "기타"
    ]
    matched_topics = [topic for topic in valid_topics if topic in match_reason]

    if len(matched_topics) != 1:
        return decision

    rescued_topic = matched_topics[0]
    rescued = dict(decision)
    rescued["pred_topic"] = rescued_topic
    rescued["pred_topic_type"] = "topic"
    rescued["classification_stage"] = "llm_reason_recovered"
    rescued["review_needed_yn"] = False
    rescued["match_reason"] = (
        match_reason + f" | rescued_from_match_reason={rescued_topic}"
    )
    return rescued


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
    use_llm_fallback: bool = True,
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
            cate_1_depth=cate_1_depth,
            cate_2_depth=cate_2_depth,
            topic_pool=normalized_topic_pool,
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
            if use_llm_fallback:
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
                decision = rescue_others_from_match_reason(
                    decision,
                    topic_pool=normalized_topic_pool,
                )
                llm_used_yn = True
            else:
                decision = resolve_topic_decision(
                    candidate_topics,
                    min_score=stage_cfg["topic_match_min_score"],
                    min_margin=stage_cfg["topic_match_min_margin"],
                    review_required_default=stage_cfg["review_required_default"],
                )

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
    use_llm_fallback: bool = True,
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
