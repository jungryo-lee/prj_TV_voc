"""Prompt builders for taxonomy rule-profile and topic-pool generation."""

from __future__ import annotations

import json
import re
from typing import Any


COMMON_FEATURE_PATTERNS = [
    "quality",
    "picture",
    "image",
    "visual",
    "sound",
    "audio",
    "video",
    "screen",
    "display",
    "tv",
    "setup",
    "install",
    "installation",
    "manual",
    "guide",
    "weight",
    "stand",
    "mount",
    "wall",
    "cable",
    "port",
    "hdmi",
    "bluetooth",
    "wifi",
    "wireless",
    "connect",
    "connection",
    "compatibility",
    "os",
    "software",
    "menu",
    "ui",
    "app",
    "apps",
    "channel",
    "content",
    "voice",
    "game",
    "gaming",
    "iot",
    "mobile",
    "brand",
    "service",
    "support",
    "warranty",
    "energy",
    "efficiency",
    "price",
    "value",
    "design",
    "color",
    "material",
    "finish",
    "heat",
    "durability",
    "panel",
    "glare",
    "angle",
    "brightness",
    "contrast",
    "resolution",
    "sharp",
    "clarity",
    "bass",
    "surround",
    "화질",
    "음질",
    "사운드",
    "화면",
    "디스플레이",
    "설치",
    "세팅",
    "매뉴얼",
    "가이드",
    "무게",
    "스탠드",
    "벽걸이",
    "선정리",
    "단자",
    "블루투스",
    "와이파이",
    "연결",
    "호환",
    "소프트웨어",
    "메뉴",
    "앱",
    "채널",
    "콘텐츠",
    "음성",
    "게임",
    "iot",
    "모바일",
    "브랜드",
    "서비스",
    "보증",
    "에너지",
    "효율",
    "가격",
    "가성비",
    "디자인",
    "색상",
    "소재",
    "마감",
    "발열",
    "내구성",
    "패널",
    "반사",
    "시야각",
    "밝기",
    "명암",
    "해상도",
    "선명",
    "저음",
    "서라운드",
]


CATEGORY_FEATURE_PATTERNS = {
    ("01. 사이즈", "01-01. TV 사이즈"): [
        "size",
        "inch",
        "inches",
        "big",
        "large",
        "small",
        "fit",
        "fits",
        "사이즈",
        "크기",
        "인치",
        "대형",
        "소형",
        "공간",
        "잘맞",
    ],
    ("02. 화질", "02-01. 선명도"): [
        "sharp",
        "sharpness",
        "clear",
        "clarity",
        "crisp",
        "blur",
        "blurry",
        "선명",
        "선명도",
        "또렷",
        "흐림",
        "블러",
    ],
    ("02. 화질", "02-02. 컬러"): [
        "color",
        "colour",
        "vivid",
        "vibrant",
        "lifelike",
        "saturation",
        "컬러",
        "색감",
        "생생",
        "채도",
    ],
    ("02. 화질", "02-03. 밝기"): [
        "bright",
        "brightness",
        "dim",
        "luminous",
        "밝기",
        "밝음",
        "어두움",
    ],
    ("02. 화질", "02-04. 명암비"): [
        "contrast",
        "black level",
        "black",
        "white balance",
        "명암",
        "명암비",
        "블랙",
        "검정",
    ],
    ("02. 화질", "02-05. 해상도"): [
        "resolution",
        "4k",
        "uhd",
        "hdr",
        "detail",
        "pixel",
        "해상도",
        "4k",
        "uhd",
        "hdr",
        "디테일",
        "픽셀",
    ],
    ("02. 화질", "02-06. 움직이는 영상 표현"): [
        "motion",
        "smooth",
        "blur",
        "judder",
        "stutter",
        "fast scene",
        "움직임",
        "모션",
        "부드러움",
        "잔상",
        "버벅",
    ],
    ("02. 화질", "02-08. 시야각"): [
        "angle",
        "viewing angle",
        "off angle",
        "시야각",
        "각도",
    ],
    ("02. 화질", "02-09. 반사율"): [
        "glare",
        "reflection",
        "reflective",
        "anti glare",
        "반사",
        "반사율",
        "빛반사",
        "눈부심",
    ],
    ("02. 화질", "02-10. 화질세팅"): [
        "setting",
        "mode",
        "calibration",
        "preset",
        "설정",
        "세팅",
        "모드",
        "보정",
    ],
    ("02. 화질", "02-10. 화질세팅_(1)화질 모드"): [
        "picture mode",
        "mode",
        "preset",
        "화질모드",
        "모드",
        "프리셋",
    ],
    ("02. 화질", "02-20. 전반적 화질"): [
        "picture",
        "image",
        "visual",
        "screen quality",
        "화질",
        "화면",
        "영상",
    ],
    ("03. 음질", "03-01. 출력"): [
        "volume",
        "loud",
        "output",
        "speaker power",
        "출력",
        "볼륨",
        "소리크기",
    ],
    ("03. 음질", "03-02. 선명한 음질"): [
        "clear sound",
        "clarity",
        "crisp audio",
        "선명",
        "맑음",
        "또렷",
    ],
    ("03. 음질", "03-03. 음질 세팅"): [
        "sound mode",
        "setting",
        "equalizer",
        "eq",
        "설정",
        "세팅",
        "모드",
        "eq",
    ],
    ("03. 음질", "03-04. 서라운드"): [
        "surround",
        "immersive",
        "spatial",
        "서라운드",
        "공간감",
    ],
    ("03. 음질", "03-05. 저음/베이스"): [
        "bass",
        "low end",
        "deep sound",
        "저음",
        "베이스",
    ],
    ("03. 음질", "03-20. 전반적 음질"): [
        "sound",
        "audio",
        "speaker",
        "소리",
        "음질",
        "사운드",
    ],
    ("04. 디자인", "04-01. 옆면 두께"): [
        "thin",
        "thickness",
        "slim",
        "얇음",
        "두께",
        "슬림",
    ],
    ("04. 디자인", "04-02. 베젤(프레임) 두께"): [
        "bezel",
        "frame",
        "베젤",
        "프레임",
    ],
    ("04. 디자인", "04-03. 스탠드 높이/형태"): [
        "stand",
        "height",
        "base",
        "스탠드",
        "높이",
        "받침",
    ],
    ("04. 디자인", "04-04. 벽걸이 디자인"): [
        "wall mount",
        "mount",
        "flush",
        "벽걸이",
        "마운트",
    ],
    ("04. 디자인", "04-05. 소재"): [
        "material",
        "texture",
        "소재",
        "재질",
    ],
    ("04. 디자인", "04-06. 색상"): [
        "color",
        "colour",
        "색상",
        "색",
    ],
    ("04. 디자인", "04-07. 마감"): [
        "finish",
        "build quality",
        "마감",
        "완성도",
    ],
    ("04. 디자인", "04-08. 후면부 디자인"): [
        "rear",
        "back design",
        "back panel",
        "후면",
        "뒷면",
    ],
    ("04. 디자인", "04-20. 전반적 디자인"): [
        "beautiful",
        "stylish",
        "look",
        "appearance",
        "예쁨",
        "외관",
        "디자인",
    ],
    ("05. 설치/세팅", "05-01. 세팅"): [
        "setup",
        "set up",
        "configure",
        "설치",
        "세팅",
        "설정",
    ],
    ("05. 설치/세팅", "05-02. 매뉴얼/가이드"): [
        "manual",
        "guide",
        "instruction",
        "매뉴얼",
        "가이드",
        "설명서",
    ],
    ("05. 설치/세팅", "05-03. 무게"): [
        "weight",
        "heavy",
        "light",
        "무게",
        "무겁",
        "가볍",
    ],
    ("05. 설치/세팅", "05-04. 선처리"): [
        "cable",
        "wire",
        "cable management",
        "선정리",
        "케이블",
        "선",
    ],
    ("05. 설치/세팅", "05-05. 각도 조절(벽걸이)"): [
        "angle",
        "tilt",
        "swivel",
        "각도",
        "틸트",
        "회전",
    ],
    ("05. 설치/세팅", "05-06. 벽걸이 설치용이성"): [
        "wall mount",
        "mounting",
        "벽걸이",
        "설치",
    ],
    ("05. 설치/세팅", "05-07. 스탠드 설치용이성"): [
        "stand assembly",
        "stand install",
        "스탠드",
        "조립",
    ],
    ("05. 설치/세팅", "05-20. 전반적 설치용이성"): [
        "install",
        "installation",
        "easy setup",
        "설치",
        "세팅",
        "조립",
    ],
    ("06. 연결성", "06-01. 연결기기 호환성"): [
        "compatible",
        "compatibility",
        "device",
        "devices",
        "호환",
        "호환성",
        "기기",
    ],
    ("06. 연결성", "06-02. 무선 연결성"): [
        "wifi",
        "wireless",
        "bluetooth",
        "pairing",
        "와이파이",
        "무선",
        "블루투스",
        "페어링",
    ],
    ("06. 연결성", "06-03. 연결단자 지원/개수"): [
        "hdmi",
        "usb",
        "port",
        "ports",
        "단자",
        "포트",
        "hdmi",
        "usb",
    ],
    ("06. 연결성", "06-04. (편리한)단자 위치"): [
        "port location",
        "easy access",
        "단자위치",
        "접근",
    ],
    ("06. 연결성", "06-20. 전반적 연결성"): [
        "connect",
        "connection",
        "연결",
        "연결성",
    ],
    ("07. 스마트 사용성", "07-01. 채널/컨텐츠 APP"): [
        "app",
        "apps",
        "channel",
        "content",
        "streaming",
        "ott",
        "앱",
        "채널",
        "콘텐츠",
        "스트리밍",
        "ott",
    ],
    ("07. 스마트 사용성", "07-02. 구동성/구동속도"): [
        "fast",
        "slow",
        "lag",
        "speed",
        "loading",
        "속도",
        "빠름",
        "느림",
        "로딩",
        "렉",
    ],
    ("07. 스마트 사용성", "07-02. 구동성/구동속도_(1)TV 전반"): [
        "fast",
        "slow",
        "lag",
        "speed",
        "tv response",
        "속도",
        "빠름",
        "느림",
        "반응",
        "렉",
    ],
    ("07. 스마트 사용성", "07-03. 메뉴 구성/UI"): [
        "menu",
        "ui",
        "interface",
        "navigation",
        "메뉴",
        "ui",
        "인터페이스",
        "탐색",
    ],
    ("07. 스마트 사용성", "07-04. SW/OS"): [
        "os",
        "software",
        "update",
        "bug",
        "os",
        "소프트웨어",
        "업데이트",
        "버그",
    ],
    ("07. 스마트 사용성", "07-05. 컨텐츠 탐색 사용성"): [
        "search",
        "browse",
        "discover",
        "탐색",
        "검색",
        "브라우징",
    ],
    ("07. 스마트 사용성", "07-06. 리모컨 사용성"): [
        "remote",
        "control",
        "button",
        "layout",
        "pointer",
        "backlight",
        "easy",
        "controller",
        "리모컨",
        "조작",
        "버튼",
        "레이아웃",
        "포인터",
        "백라이트",
        "편리",
    ],
    ("07. 스마트 사용성", "07-07. 리모컨 디자인"): [
        "remote design",
        "remote look",
        "리모컨 디자인",
        "외관",
    ],
    ("07. 스마트 사용성", "07-08. 음성 인식/조작"): [
        "voice",
        "speech",
        "assistant",
        "음성",
        "음성인식",
        "음성조작",
    ],
    ("07. 스마트 사용성", "07-09. 게임 기능"): [
        "game",
        "gaming",
        "latency",
        "게임",
        "게이밍",
        "지연",
    ],
    ("07. 스마트 사용성", "07-10. 부가 기능"): [
        "feature",
        "features",
        "extra function",
        "기능",
        "부가기능",
    ],
    ("07. 스마트 사용성", "07-11. 홈 IoT"): [
        "iot",
        "smart home",
        "iot",
        "스마트홈",
    ],
    ("07. 스마트 사용성", "07-12. 모바일 연동"): [
        "mobile",
        "phone",
        "cast",
        "mirror",
        "모바일",
        "휴대폰",
        "캐스트",
        "미러링",
    ],
    ("07. 스마트 사용성", "07-13. 광고"): [
        "ad",
        "ads",
        "advertisement",
        "광고",
    ],
    ("07. 스마트 사용성", "07-20. 전반적 스마트 사용성"): [
        "smart",
        "usability",
        "easy to use",
        "스마트",
        "사용성",
        "편리",
    ],
    ("08. 내구성", "08-01. A/S"): [
        "service",
        "support",
        "repair",
        "warranty",
        "as",
        "서비스",
        "지원",
        "수리",
        "보증",
    ],
    ("08. 내구성", "08-02. 품질보증기간"): [
        "warranty",
        "guarantee",
        "보증",
        "보증기간",
    ],
    ("08. 내구성", "08-03. 잔상"): [
        "ghosting",
        "burn in",
        "image retention",
        "잔상",
        "번인",
    ],
    ("08. 내구성", "08-04. 패널 내구성"): [
        "panel",
        "durability",
        "screen life",
        "패널",
        "내구성",
    ],
    ("08. 내구성", "08-05. 발열"): [
        "heat",
        "heating",
        "hot",
        "발열",
        "뜨거움",
    ],
    ("08. 내구성", "08-20. 전반적 내구성"): [
        "durable",
        "reliable",
        "lasting",
        "내구성",
        "튼튼",
        "오래감",
    ],
    ("09. 친환경", "09-01. 에너지 효율"): [
        "energy",
        "efficient",
        "efficiency",
        "power saving",
        "에너지",
        "효율",
        "절전",
    ],
    ("09. 친환경", "09-02. 친환경 소재"): [
        "eco",
        "eco-friendly",
        "material",
        "친환경",
        "소재",
    ],
    ("10. 가격", "10-01. 가격/가격대비 가치"): [
        "price",
        "value",
        "worth",
        "deal",
        "budget",
        "affordable",
        "가격",
        "가성비",
        "값어치",
        "딜",
        "저렴",
    ],
    ("11. 브랜드", "11-01. 브랜드"): [
        "brand",
        "samsung",
        "lg",
        "sony",
        "tcl",
        "hisense",
        "브랜드",
        "삼성",
        "엘지",
        "소니",
    ],
    ("12. 품질 불량", "12-01. 화질 불량"): [
        "defect",
        "screen issue",
        "dead pixel",
        "dark",
        "blurry",
        "불량",
        "화질불량",
        "데드픽셀",
        "암부",
        "흐림",
    ],
    ("12. 품질 불량", "12-02. 제품 불량"): [
        "defective",
        "broken",
        "fault",
        "problem",
        "불량",
        "고장",
        "문제",
    ],
    ("12. 품질 불량", "12-03. 설치 불량"): [
        "installation issue",
        "bad install",
        "설치불량",
        "설치문제",
    ],
    ("12. 품질 불량", "12-04. 기능 불량"): [
        "malfunction",
        "doesn't work",
        "failed",
        "not working",
        "기능불량",
        "작동안함",
        "먹통",
        "고장",
    ],
    ("12. 품질 불량", "12-05. 기타 불량"): [
        "issue",
        "problem",
        "fault",
        "불량",
        "문제",
        "이슈",
    ],
    ("13. 전반적 평가", "13-01. 전반적 평가"): [
        "overall",
        "generally",
        "satisfied",
        "happy",
        "purchase",
        "product",
        "전반적",
        "전체적",
        "만족",
        "구매",
        "제품",
    ],
}


def clean_text(value: Any) -> str:
    """Normalize whitespace for prompt-friendly text."""
    return "" if value is None else re.sub(r"\s+", " ", str(value)).strip()


def _compact_prompt_terms(
    values: list[str] | None,
    *,
    max_items: int,
) -> list[str]:
    """Return a compact unique list for prompt injection."""
    if not values:
        return []

    seen: set[str] = set()
    compacted: list[str] = []

    for value in values:
        text = clean_text(value)
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        compacted.append(text)
        if len(compacted) >= max_items:
            break

    return compacted


def sc_label(sc_measurement: int) -> str:
    """Return a Korean sentiment label for the current group."""
    if int(sc_measurement) == 1:
        return "긍정"
    if int(sc_measurement) == -1:
        return "부정"
    return "기타"


def overall_topic_name(sc_measurement: int) -> str:
    """Return the fixed overall fallback topic name."""
    if int(sc_measurement) == 1:
        return "전반적 긍정"
    if int(sc_measurement) == -1:
        return "전반적 부정"
    return "전반적 평가"


def get_feature_patterns(cate_1_depth: str, cate_2_depth: str) -> list[str]:
    """Return base feature hints for a category."""
    return COMMON_FEATURE_PATTERNS + CATEGORY_FEATURE_PATTERNS.get(
        (cate_1_depth, cate_2_depth),
        [],
    )


def get_static_category_feature_patterns(
    cate_1_depth: str,
    cate_2_depth: str,
) -> list[str]:
    """Return only the category-specific static patterns."""
    return CATEGORY_FEATURE_PATTERNS.get((cate_1_depth, cate_2_depth), [])


def get_rule_profile_prompt_feature_hints(
    cate_1_depth: str,
    cate_2_depth: str,
    *,
    max_items: int = 40,
) -> list[str]:
    """Return a compact feature-hint list for rule-profile prompts."""
    static_patterns = get_static_category_feature_patterns(cate_1_depth, cate_2_depth)
    if static_patterns:
        return static_patterns[:max_items]
    return COMMON_FEATURE_PATTERNS[:max_items]


def has_static_category_patterns(cate_1_depth: str, cate_2_depth: str) -> bool:
    """Return whether static category patterns exist for the category key."""
    return (cate_1_depth, cate_2_depth) in CATEGORY_FEATURE_PATTERNS


def build_rule_profile_messages(
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    sample_memos: list[str],
) -> list[dict[str, str]]:
    """Build messages for rule-profile generation."""
    polarity_label = sc_label(sc_measurement)
    overall_name = overall_topic_name(sc_measurement)
    feature_patterns = get_rule_profile_prompt_feature_hints(
        cate_1_depth,
        cate_2_depth,
        max_items=40,
    )

    system = f"""
You are a VOC rule designer for TV review topic classification.

Category:
- cate_1_depth = {cate_1_depth}
- cate_2_depth = {cate_2_depth}
- polarity = {polarity_label}
- fixed overall fallback topic = {overall_name}

Goal:
- Build classification guidance so that only pure sentiment-only memos become {overall_name}.
- If a memo contains even a short but usable reason, feature, attribute, symptom, usage context, or target object, it must not be treated as {overall_name}.

Definition of allowed overall memo:
- Very short sentiment-only evaluation.
- Contains emotional judgment such as good, nice, bad, love it, hate it, 만족, 별로, 좋다, 나쁘다.
- Does not mention any feature, reason, object, symptom, function, or usage context.

Definition of blocked overall memo:
- Mentions why the sentiment happened.
- Mentions an attribute or function even in one word, such as fast, slow, bright, dark, heavy, laggy, clear, blurry, loud, quiet, stable, unstable.
- Mentions category-specific target objects or functions.

Category-specific feature hints:
{json.dumps(feature_patterns[:120], ensure_ascii=False)}

Instructions:
- Focus on what should block {overall_name}, not on broad sentiment words.
- feature_hint_terms should capture category objects, functions, or targets, including synonyms and likely variants.
- reason_signal_terms should capture short property or symptom expressions that imply a specific reason.
- overall_sentiment_terms should contain pure sentiment expressions only.
- non_overall_examples must be short memo-like examples that look brief but should still avoid {overall_name}.
- Return Korean outputs when possible, but include English terms if they are useful real-world variants.
- Do not return markdown.
- Do not return code fences.
- Return JSON only.

Return schema:
{{
  "overall_allowed_rule": "",
  "overall_block_rule": "",
  "overall_sentiment_terms": [""],
  "feature_hint_terms": [""],
  "reason_signal_terms": [""],
  "non_overall_examples": [""]
}}
"""

    user = "Review memos:\n" + "\n".join(
        f"- {clean_text(memo)}" for memo in sample_memos if clean_text(memo)
    )

    return [
        {"role": "system", "content": clean_text(system)},
        {"role": "user", "content": clean_text(user)},
    ]


def build_category_pattern_seed_messages(
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    sample_memos: list[str],
) -> list[dict[str, str]]:
    """Build messages for generating category seed patterns for new categories."""
    polarity_label = sc_label(sc_measurement)
    common_patterns = COMMON_FEATURE_PATTERNS[:120]

    system = f"""
You are a VOC taxonomy bootstrap designer for TV review topic classification.

Category:
- cate_1_depth = {cate_1_depth}
- cate_2_depth = {cate_2_depth}
- polarity = {polarity_label}

Goal:
- Infer reusable seed terms for a category that does not yet have a curated pattern dictionary.
- Focus on terms that help detect specific reasons, feature mentions, and category targets.
- Distinguish pure sentiment words from feature/object/reason words.

Reference common feature vocabulary:
{json.dumps(common_patterns, ensure_ascii=False)}

Instructions:
- feature_hint_terms:
  Capture product objects, functions, interfaces, devices, targets, and likely synonyms.
- reason_signal_terms:
  Capture short attributes, symptoms, states, or issue expressions that imply a concrete reason.
- overall_sentiment_terms:
  Capture pure sentiment-only words that do not indicate a specific reason by themselves.
- candidate_topic_labels:
  Suggest short Korean topic labels that seem plausible for this category.
- Keep terms concise and deduplicated.
- Prefer terms grounded in the sample memos, but add high-confidence synonyms if they are operationally useful.
- Do not return markdown.
- Do not return code fences.
- Return JSON only.

Return schema:
{{
  "category_summary": "",
  "feature_hint_terms": [""],
  "reason_signal_terms": [""],
  "overall_sentiment_terms": [""],
  "candidate_topic_labels": [""],
  "sample_non_overall_memos": [""]
}}
"""

    user = "Review memos:\n" + "\n".join(
        f"- {clean_text(memo)}" for memo in sample_memos if clean_text(memo)
    )

    return [
        {"role": "system", "content": clean_text(system)},
        {"role": "user", "content": clean_text(user)},
    ]


def build_topic_pool_messages(
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    sample_memos: list[str],
    rule_profile: dict[str, Any],
    min_final_topics: int,
    max_final_topics: int,
) -> list[dict[str, str]]:
    """Build messages for topic-pool generation."""
    overall_name = overall_topic_name(sc_measurement)
    feature_patterns = get_rule_profile_prompt_feature_hints(
        cate_1_depth,
        cate_2_depth,
        max_items=25,
    )

    overall_allowed_rule = clean_text(rule_profile.get("overall_allowed_rule"))
    overall_block_rule = clean_text(rule_profile.get("overall_block_rule"))
    overall_sentiment_terms = _compact_prompt_terms(
        rule_profile.get("overall_sentiment_terms", []) or [],
        max_items=15,
    )
    feature_hint_terms = _compact_prompt_terms(
        rule_profile.get("feature_hint_terms", []) or [],
        max_items=25,
    )
    reason_signal_terms = _compact_prompt_terms(
        rule_profile.get("reason_signal_terms", []) or [],
        max_items=25,
    )
    non_overall_examples = _compact_prompt_terms(
        rule_profile.get("non_overall_examples", []) or [],
        max_items=8,
    )

    system = f"""
You are a VOC taxonomy designer for TV review topic classification.

Category:
- cate_1_depth = {cate_1_depth}
- cate_2_depth = {cate_2_depth}
- polarity = {sc_label(sc_measurement)}
- fixed overall fallback topic = {overall_name}

Rules:
- Final topic count must be between {int(min_final_topics)} and {int(max_final_topics)}.
- One mandatory topic must be "{overall_name}".
- "{overall_name}" is only for pure sentiment-only memos with no usable reason.
- If a memo contains a feature, attribute, symptom, object, usage context, or short reason, it belongs to a specific topic, not "{overall_name}".
- Topic labels must be Korean.
- Topic labels should be concise and operationally useful.
- Avoid duplicate or near-synonym topics.
- Prefer function/issue/attribute-based topics over vague sentiment buckets.
- Avoid overly fine-grained topics that would be too small to operate monthly.

Rule profile guidance:
- overall_allowed_rule: {overall_allowed_rule}
- overall_block_rule: {overall_block_rule}
- overall_sentiment_terms: {json.dumps(overall_sentiment_terms, ensure_ascii=False)}
- feature_hint_terms: {json.dumps(feature_hint_terms, ensure_ascii=False)}
- reason_signal_terms: {json.dumps(reason_signal_terms, ensure_ascii=False)}
- non_overall_examples: {json.dumps(non_overall_examples, ensure_ascii=False)}
- category_feature_hints: {json.dumps(feature_patterns, ensure_ascii=False)}

Output rules:
- Include the mandatory overall topic exactly once.
- Create topics that are broad enough for monthly operation but specific enough for root-cause analysis.
- Merge near-synonyms instead of splitting them.
- Use representative memos only as short supporting examples.

Return JSON only:
{{
  "topics": [
    {{
      "topic": "",
      "description": "",
      "representative_memos": [""]
    }}
  ]
}}
"""

    user = "Sample memos:\n" + "\n".join(
        f"- {clean_text(memo)}" for memo in sample_memos if clean_text(memo)
    )

    return [
        {"role": "system", "content": clean_text(system)},
        {"role": "user", "content": clean_text(user)},
    ]
