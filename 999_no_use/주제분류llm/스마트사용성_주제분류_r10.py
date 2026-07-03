# Databricks notebook source
# DBTITLE 1,1) Config and Imports
# ============================================================
# Round 10: 스마트 사용성 전수 LLM 주제분류
# ============================================================
# - cate_1_depth = '07. 스마트 사용성' 전체 하위 그룹 대상
# - 전수 분류 (샘플링 안 함, 미분류 메모 전체)
# - 200자 초과 메모 → 1문장 요약 후 분류
# - 기타 병합 없이 LLM 토픽 그대로 유지
# - 기존 로직 동일: round10 전용 DETAIL 테이블 생성
# - Integration 테이블에 llm_round=r10 머지
# - category_topic_version_log 에 버전 기록
# ============================================================

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

# --------------------------------------------------
# LLM Endpoint
# --------------------------------------------------
ENDPOINT = "databricks-gpt-5-4-mini"

# --------------------------------------------------
# DB / Tables
# --------------------------------------------------
SAVE_DB = "sandbox.z_jungryo_lee"
SOURCE_TABLE = "sandbox.t_online_voc_analysis.buzzmetrix"
RAW_SOURCE_TABLE = "kic_data_ods.buzzmetrix.buzzmetrix"
PROMPT_VERSION = "_v2"

SAMPLING_ROUND = 11
ROUND_LABEL = f"r{SAMPLING_ROUND}"

RULE_PROFILE_TABLE = f"{SAVE_DB}.category_topic_rule_profile"
TOPIC_POOL_TABLE   = f"{SAVE_DB}.category_topic_catalog"

# Round 10 전용 DETAIL 테이블 (기존 로직과 동일하게 회차별 테이블 생성)
DETAIL_TABLE = f"{SAVE_DB}.category_topic_detail{PROMPT_VERSION}_round{SAMPLING_ROUND}"
SUMMARY_TABLE = f"{SAVE_DB}.category_topic_summary{PROMPT_VERSION}_round{SAMPLING_ROUND}"

# Integration (merge 대상)
V1_DETAIL_TABLE = f"{SAVE_DB}.category_topic_detail_integration"
FINAL_OUTPUT_TABLE = "sandbox.t_online_voc_analysis.voc_llm_topic_classification"
SOURCE_RECLASSIFY_SCOPE_VIEW = f"tmp_smart_source_scope_{SAMPLING_ROUND}"

# Checkpoint
PROGRESS_TABLE = f"{SAVE_DB}.smart_full_round{SAMPLING_ROUND}_progress"
VERSION_TRACKING_TABLE = f"{SAVE_DB}.category_topic_version_log"

# --------------------------------------------------
# 대상 필터
# --------------------------------------------------
TARGET_CATE_1 = "07. 스마트 사용성"

# --------------------------------------------------
# Thresholds
# --------------------------------------------------
CLASSIFY_BATCH_SIZE  = 25
MAX_TOKENS           = 2200
MAX_RETRIES          = 3
BACKOFF_BASE         = 1.8
RATE_LIMIT_SECONDS   = 0.4
MEMO_SUMMARIZE_THRESHOLD = 200
SUMMARIZE_BATCH_SIZE = 20
CHUNK_SIZE = 500  # 500건마다 checkpoint 저장

print(f"[CONFIG] Round {SAMPLING_ROUND}: 스마트 사용성 전수분류")
print(f"  TARGET       = {TARGET_CATE_1}")
print(f"  ENDPOINT     = {ENDPOINT}")
print(f"  DETAIL_TABLE = {DETAIL_TABLE}")
print(f"  INTEGRATION  = {V1_DETAIL_TABLE}")
print(f"  ROUND_LABEL  = {ROUND_LABEL}")
print(f"  CHUNK_SIZE   = {CHUNK_SIZE}")

# COMMAND ----------

# DBTITLE 1,2) Helpers
# ============================================================
# 2) Helpers
# ============================================================

def clean_text(x: Any) -> str:
    return "" if x is None else re.sub(r"\s+", " ", str(x)).strip()


def normalize_memo_text(x: Any) -> str:
    if x is None:
        return ""
    text = str(x).replace("　", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_json(text: str) -> Dict[str, Any]:
    text = clean_text(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            candidate = re.sub(r",\s*}", "}", candidate)
            candidate = re.sub(r",\s*]", "]", candidate)
            return json.loads(candidate)
    raise ValueError(f"Invalid JSON: {text[:1000]}")


def chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def sc_label(sc_measurement: int) -> str:
    return {1: "긍정", -1: "부정"}.get(sc_measurement, "기타")


def overall_label(sc_measurement: int) -> str:
    return {1: "전반적 긍정", -1: "전반적 부정"}.get(sc_measurement, "전반적 평가")


def call_llm(messages: List[Dict[str, str]], max_tokens: int = MAX_TOKENS) -> Dict[str, Any]:
    from mlflow.deployments import get_deploy_client
    client = get_deploy_client("databricks")
    payload = {"messages": messages, "temperature": 0.0, "max_tokens": max_tokens}
    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.predict(endpoint=ENDPOINT, inputs=payload)
            if isinstance(resp, dict):
                if "choices" in resp and resp["choices"]:
                    return extract_json(resp["choices"][0]["message"]["content"])
                if "predictions" in resp and resp["predictions"]:
                    pred0 = resp["predictions"][0]
                    if isinstance(pred0, dict) and "content" in pred0:
                        return extract_json(pred0["content"])
                    if isinstance(pred0, str):
                        return extract_json(pred0)
                if "content" in resp:
                    return extract_json(resp["content"])
            if isinstance(resp, str):
                return extract_json(resp)
            raise ValueError(f"Unexpected response: {resp}")
        except Exception as exc:
            last_error = exc
            print(f"[LLM ERROR] attempt={attempt+1}/{MAX_RETRIES}, error={repr(exc)}")
            time.sleep(BACKOFF_BASE ** attempt)
    raise RuntimeError(f"LLM call failed: {repr(last_error)}")


def append_table(df, table_name: str) -> None:
    if spark.catalog.tableExists(table_name):
        df.write.mode("append").option("mergeSchema", "true").format("delta").saveAsTable(table_name)
    else:
        df.write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(table_name)


def _source_filter_sql(
    c1: str,
    c2: Optional[str] = None,
    sc: Optional[int] = None,
    llm_round_is_null: Optional[bool] = None,
) -> str:
    filters = [
        f"cate_1_depth = '{c1}'",
        "memo IS NOT NULL",
        "LENGTH(TRIM(memo)) > 0",
    ]
    if c2 is not None:
        filters.append(f"cate_2_depth = '{c2}'")
    if sc is None:
        filters.append("sc_measurement IN (1, -1)")
    else:
        filters.append(f"sc_measurement = {int(sc)}")
    if llm_round_is_null is True:
        filters.append("llm_round IS NULL")
    elif llm_round_is_null is False:
        filters.append("llm_round IS NOT NULL")
    return " AND ".join(filters)


def _prior_round_labels() -> List[str]:
    return [f"r{i}" for i in range(0, SAMPLING_ROUND)]


def build_memo_id_value(c1: str, c2: str, sc: int, memo: Any) -> str:
    raw = "||".join([
        clean_text(c1),
        clean_text(c2),
        str(int(sc)),
        normalize_memo_text(memo),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def with_memo_id(df):
    return df.withColumn(
        "memo_id",
        F.sha2(
            F.concat_ws(
                "||",
                F.coalesce(F.col("cate_1_depth").cast("string"), F.lit("")),
                F.coalesce(F.col("cate_2_depth").cast("string"), F.lit("")),
                F.coalesce(F.col("sc_measurement").cast("string"), F.lit("")),
                F.trim(
                    F.regexp_replace(
                        F.translate(F.coalesce(F.col("memo").cast("string"), F.lit("")), "　", " "),
                        "\\s+",
                        " ",
                    )
                ),
            ),
            256,
        ),
    )


def exclude_existing_source_rows(source_df, table_name: str, where_sql: str):
    table_cols = spark.table(table_name).columns
    if "memo_id" in table_cols:
        existing_memo_ids = spark.sql(f"""
            SELECT DISTINCT memo_id
            FROM {table_name}
            WHERE {where_sql}
              AND memo_id IS NOT NULL
        """)
        source_df = source_df.join(existing_memo_ids, on="memo_id", how="left_anti")

        existing_legacy_memos = spark.sql(f"""
            SELECT DISTINCT memo
            FROM {table_name}
            WHERE {where_sql}
              AND memo_id IS NULL
        """)
        return source_df.join(existing_legacy_memos, on="memo", how="left_anti")

    existing_legacy_memos = spark.sql(f"""
        SELECT DISTINCT memo
        FROM {table_name}
        WHERE {where_sql}
    """)
    return source_df.join(existing_legacy_memos, on="memo", how="left_anti")


def dedupe_merge_source(df, key_columns: List[str]):
    duplicate_keys_df = (
        df.groupBy(*key_columns)
        .count()
        .where(F.col("count") > 1)
    )

    duplicate_key_count = duplicate_keys_df.count()
    if duplicate_key_count == 0:
        return df

    duplicate_row_count = (
        duplicate_keys_df
        .agg(F.sum("count").alias("duplicate_rows"))
        .collect()[0]["duplicate_rows"]
    )
    print(
        f"  [DEDUP] merge source duplicates detected: "
        f"{duplicate_key_count:,} keys / {duplicate_row_count:,} rows"
    )

    order_exprs = []
    for col_name in ["topic_rev", "main_topic", "llm_round", "topic", "description", "_row_id"]:
        if col_name in df.columns:
            order_exprs.append(F.col(col_name).isNotNull().cast("int").desc())
            order_exprs.append(F.col(col_name).desc_nulls_last())

    if not order_exprs:
        order_exprs.append(F.lit(1))

    dedup_window = Window.partitionBy(*key_columns).orderBy(*order_exprs)

    deduped_df = (
        df.withColumn("_merge_rn", F.row_number().over(dedup_window))
        .where(F.col("_merge_rn") == 1)
        .drop("_merge_rn")
    )

    deduped_count = deduped_df.count()
    print(f"  [DEDUP] source rows after key dedup: {deduped_count:,}")
    return deduped_df


def merge_table_by_keys(df, table_name: str, key_columns: List[str]) -> None:
    df = dedupe_merge_source(df, key_columns)
    temp_view = f"tmp_merge_{uuid.uuid4().hex}"
    df.createOrReplaceTempView(temp_view)
    on_clause = " AND ".join([f"t.`{col}` <=> s.`{col}`" for col in key_columns])
    spark.sql(f"""
        MERGE INTO {table_name} t
        USING {temp_view} s
        ON {on_clause}
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)
    spark.catalog.dropTempView(temp_view)


# Progress
PROGRESS_STRUCT = T.StructType([
    T.StructField("cate_2_depth", T.StringType(), True),
    T.StructField("sc_measurement", T.IntegerType(), True),
    T.StructField("status", T.StringType(), True),
    T.StructField("classified_count", T.IntegerType(), True),
    T.StructField("event_ts", T.TimestampType(), True),
])

def log_group_done(c2, sc, count):
    row = [(clean_text(c2), int(sc), "done", int(count), pd.Timestamp.now().to_pydatetime())]
    append_table(spark.createDataFrame(row, schema=PROGRESS_STRUCT), PROGRESS_TABLE)


def build_llm_round_status_snapshot(
    source_table: str,
    source_filter_sql: str,
    classified_table: Optional[str] = None,
):
    base_df = spark.sql(f"""
        SELECT
            cate_2_depth,
            sc_measurement,
            COUNT(DISTINCT memo) AS total_memos
        FROM {source_table}
        WHERE {source_filter_sql}
        GROUP BY cate_2_depth, sc_measurement
    """)

    if classified_table is None:
        classified_df = spark.sql(f"""
            SELECT
                cate_2_depth,
                sc_measurement,
                COUNT(DISTINCT CASE WHEN llm_round IS NOT NULL THEN memo END) AS classified_memos,
                COUNT(DISTINCT CASE WHEN llm_round IS NULL THEN memo END) AS target_memos
            FROM {source_table}
            WHERE {source_filter_sql}
            GROUP BY cate_2_depth, sc_measurement
        """)
    elif spark.catalog.tableExists(classified_table):
        classified_df = spark.sql(f"""
            SELECT
                cate_2_depth,
                sc_measurement,
                COUNT(DISTINCT memo) AS classified_memos
            FROM {classified_table}
            WHERE cate_1_depth = '{TARGET_CATE_1}'
              AND sc_measurement IN (1, -1)
            GROUP BY cate_2_depth, sc_measurement
        """)
    else:
        classified_df = spark.createDataFrame(
            [],
            "cate_2_depth string, sc_measurement int, classified_memos long",
        )

    status_df = (
        base_df.alias("b")
        .join(classified_df.alias("c"), on=["cate_2_depth", "sc_measurement"], how="left")
        .fillna(0, subset=["classified_memos"])
    )

    if "target_memos" not in status_df.columns:
        status_df = status_df.withColumn("target_memos", F.col("total_memos") - F.col("classified_memos"))

    status_df = (
        status_df
        .withColumn("target_pct", F.round(F.col("target_memos") / F.col("total_memos") * 100, 1))
        .withColumn("classified_pct", F.round(F.col("classified_memos") / F.col("total_memos") * 100, 1))
        .orderBy("cate_2_depth", "sc_measurement")
    )
    return status_df


def print_llm_round_status_snapshot(title: str, status_df):
    status_pdf = status_df.toPandas()
    print(f"\n[{title}]")
    print("세부카테고리별 전체 대비 미분류(llm_round is null) / 분류완료(llm_round is not null)")
    if status_pdf.empty:
        print("  스마트 사용성 대상 현황 데이터가 없습니다.")
        return

    display(spark.createDataFrame(status_pdf))

    total_memos = int(status_pdf["total_memos"].sum())
    target_memos = int(status_pdf["target_memos"].sum())
    classified_memos = int(status_pdf["classified_memos"].sum())
    target_pct = round(target_memos / total_memos * 100, 1) if total_memos else 0.0
    classified_pct = round(classified_memos / total_memos * 100, 1) if total_memos else 0.0

    print(
        f"  전체 {total_memos:,}건 | 미분류 {target_memos:,}건 ({target_pct:.1f}%)"
        f" | 분류완료 {classified_memos:,}건 ({classified_pct:.1f}%)"
    )


def split_publishable_vs_reclassify(df):
    reclassify_condition = (
        F.col("llm_round").isNotNull()
        & F.col("topic_rev").isNull()
        & F.col("topic").isNotNull()
        & (F.length(F.trim(F.col("topic"))) > 0)
        & (~F.col("topic").isin("기타", "오분류"))
    )
    publishable_df = df.where(~reclassify_condition)
    reclassify_df = df.where(reclassify_condition)
    return publishable_df, reclassify_df


def create_source_reclassify_scope_view():
    source_scope_df = (
        spark.table(SOURCE_TABLE)
        .where(
            (F.col("cate_1_depth") == TARGET_CATE_1)
            & F.col("sc_measurement").isin(1, -1)
            & F.col("memo").isNotNull()
            & (F.length(F.trim(F.col("memo"))) > 0)
        )
        .select("cate_1_depth", "cate_2_depth", "sc_measurement", "memo", "topic", "llm_round")
        .dropDuplicates()
        .transform(with_memo_id)
    )

    raw_source_scope_df = (
        spark.table(RAW_SOURCE_TABLE)
        .where(
            (F.col("cate_1_depth") == TARGET_CATE_1)
            & F.col("sc_measurement").isin(1, -1)
            & F.col("memo").isNotNull()
            & (F.length(F.trim(F.col("memo"))) > 0)
        )
        .select("cate_1_depth", "cate_2_depth", "sc_measurement", "memo")
        .dropDuplicates()
        .withColumn("llm_round", F.lit(None).cast("string"))
        .transform(with_memo_id)
    )

    if spark.catalog.tableExists(FINAL_OUTPUT_TABLE):
        final_scope_df = spark.table(FINAL_OUTPUT_TABLE)
        if "memo_id" not in final_scope_df.columns:
            final_scope_df = with_memo_id(final_scope_df)
        final_scope_df = (
            final_scope_df
            .where(
                (F.col("cate_1_depth") == TARGET_CATE_1)
                & F.col("sc_measurement").isin(1, -1)
                & F.col("memo").isNotNull()
                & (F.length(F.trim(F.col("memo"))) > 0)
            )
            .select("memo_id")
            .dropDuplicates()
        )
    else:
        final_scope_df = spark.createDataFrame([], "memo_id string")

    missing_final_df = raw_source_scope_df.join(final_scope_df, on="memo_id", how="left_anti")

    reclassify_scope_df = (
        source_scope_df
        .where(
            F.col("llm_round").isNull()
            | F.col("topic").isNull()
            | (F.length(F.trim(F.col("topic"))) == 0)
        )
        .unionByName(missing_final_df)
        .dropDuplicates(["memo_id"])
    )

    missing_final_count = missing_final_df.count()
    reclassify_scope_count = reclassify_scope_df.count()
    recovered_non_null_count = (
        reclassify_scope_df
        .where(F.col("llm_round").isNotNull())
        .count()
    )
    recovered_null_topic_count = (
        reclassify_scope_df
        .where(
            F.col("llm_round").isNotNull()
            & (F.col("topic").isNull() | (F.length(F.trim(F.col("topic"))) == 0))
        )
        .count()
    )

    reclassify_scope_df.createOrReplaceTempView(SOURCE_RECLASSIFY_SCOPE_VIEW)
    print(f"[SOURCE SYNC] final output missing in buzz scope: {missing_final_count:,}건")
    print(f"[SOURCE SYNC] recovered despite llm_round not null: {recovered_non_null_count:,}건")
    print(f"[SOURCE SYNC] recovered because topic is null: {recovered_null_topic_count:,}건")
    print(f"[SOURCE SYNC] classification scope view: {SOURCE_RECLASSIFY_SCOPE_VIEW} ({reclassify_scope_count:,}건)")

print("[OK] Helpers")

# COMMAND ----------

# DBTITLE 1,3) Memo Summarization
# ============================================================
# 3) Memo Summarization (200자 초과 → 1문장 요약)
# ============================================================

def build_summarize_messages(batch_rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    system = """You are a Korean text summarizer for TV product reviews.

Task: Summarize each memo into exactly ONE concise Korean sentence.
Rules:
- Keep the core opinion/evaluation/complaint intact.
- Remove unnecessary repetition, filler words, and irrelevant details.
- The summary must preserve the sentiment (positive/negative) and the main subject.
- Output in Korean only.
- Do NOT add any interpretation or opinion not present in the original.

Return only JSON:
{
  "results": [{"row_id": "", "summary": ""}]
}
"""
    user = (
        "Memos to summarize:\n"
        + json.dumps(
            [{"row_id": str(r["_row_id"]), "memo": clean_text(r["memo"])} for r in batch_rows],
            ensure_ascii=False,
        )
    )
    return [
        {"role": "system", "content": clean_text(system)},
        {"role": "user", "content": clean_text(user)},
    ]


def summarize_memos_batch(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    long_rows = [r for r in rows if len(clean_text(r["memo"])) > MEMO_SUMMARIZE_THRESHOLD]
    if not long_rows:
        return {}
    print(f"    [요약] {len(long_rows)}/{len(rows)}건 요약 시작")
    summary_map = {}
    for batch_no, batch in enumerate(chunk_list(long_rows, SUMMARIZE_BATCH_SIZE), start=1):
        try:
            result = call_llm(build_summarize_messages(batch), max_tokens=MAX_TOKENS)
            for item in result.get("results", []):
                if isinstance(item, dict) and item.get("row_id") and item.get("summary"):
                    summary_map[str(item["row_id"])] = clean_text(item["summary"])
        except Exception as exc:
            print(f"    [요약 ERROR] batch={batch_no}, {repr(exc)}")
            for r in batch:
                summary_map[str(r["_row_id"])] = clean_text(r["memo"])[:200]
        time.sleep(RATE_LIMIT_SECONDS)
    return summary_map


def prepare_rows_for_classification(rows):
    summary_map = summarize_memos_batch(rows)
    classify_rows = []
    for r in rows:
        new_row = dict(r)
        if str(r["_row_id"]) in summary_map:
            new_row["memo_for_llm"] = summary_map[str(r["_row_id"])]
        else:
            new_row["memo_for_llm"] = r["memo"]
        classify_rows.append(new_row)
    return classify_rows, summary_map

print("[OK] Memo Summarization")

# COMMAND ----------

# DBTITLE 1,4) Load Target Groups
# ============================================================
# 4) 대상 그룹 로드 (07. 스마트 사용성 하위 전체)
# ============================================================

create_source_reclassify_scope_view()

status_source_df = build_llm_round_status_snapshot(
    source_table=SOURCE_TABLE,
    source_filter_sql=_source_filter_sql(TARGET_CATE_1),
)
print_llm_round_status_snapshot(
    title=f"LLM ROUND STATUS BEFORE RUN | {SOURCE_TABLE}",
    status_df=status_source_df,
)

groups_df = spark.sql(f"""
    SELECT cate_2_depth, sc_measurement, COUNT(DISTINCT memo) as total_memos
    FROM {SOURCE_RECLASSIFY_SCOPE_VIEW}
    GROUP BY cate_2_depth, sc_measurement
    ORDER BY cate_2_depth, sc_measurement
""")

# PROGRESS_TABLE은 참고용으로만 보고, 실제 대상 그룹은 현재 재분류 스코프 기준으로 모두 다시 계산
done_groups = set()
if spark.catalog.tableExists(PROGRESS_TABLE):
    for r in spark.table(PROGRESS_TABLE).where("status = 'done'").collect():
        done_groups.add((r["cate_2_depth"], int(r["sc_measurement"])))

group_rows = [r.asDict(recursive=True) for r in groups_df.toLocalIterator()]

print(f"[LOAD] 전체 그룹: {groups_df.count()}개, PROGRESS 완료기록: {len(done_groups)}개, 현재 대기: {len(group_rows)}개")
for g in group_rows[:10]:
    print(f"  {g['cate_2_depth']} / sc={g['sc_measurement']} / {g['total_memos']:,}건")

# COMMAND ----------

# DBTITLE 1,5) Classification Functions
# ============================================================
# 5) Classification Functions (v5 원본과 동일)
#    - Category Feature Patterns (전체)
#    - has_specific_reason_for_category / should_be_overall_for_category
#    - is_pure_overall_sentiment (전반적 그룹 rule pre-filter)
#    - build_classify_messages (기타 + 오분류 포함)
#    - build_batch_reclassify_messages (Phase 2)
#    - classify_batch_and_merge (Phase 0요약 + Phase 1분류 + Phase 2 Overall재분류)
# ============================================================

# --------------------------------------------------
# Feature Patterns
# --------------------------------------------------
COMMON_FEATURE_PATTERNS = [
    "quality", "picture", "image", "visual", "sound", "audio", "video", "screen", "display", "tv",
    "setup", "install", "installation", "manual", "guide", "weight", "stand", "mount", "wall",
    "cable", "port", "hdmi", "bluetooth", "wifi", "wireless", "connect", "connection", "compatibility",
    "os", "software", "menu", "ui", "app", "apps", "channel", "content", "voice", "game", "gaming",
    "iot", "mobile", "brand", "service", "support", "warranty", "energy", "efficiency", "price",
    "value", "design", "color", "material", "finish", "heat", "durability", "panel", "glare",
    "angle", "brightness", "contrast", "resolution", "sharp", "clarity", "bass", "surround",
    "화질", "음질", "사운드", "화면", "디스플레이", "설치", "세팅", "매뉴얼", "가이드", "무게",
    "스탠드", "벽걸이", "선정리", "단자", "블루투스", "와이파이", "연결", "호환", "소프트웨어",
    "메뉴", "ui", "앱", "채널", "콘텐츠", "음성", "게임", "iot", "모바일", "브랜드", "서비스",
    "보증", "에너지", "효율", "가격", "가성비", "디자인", "색상", "소재", "마감", "발열",
    "내구성", "패널", "반사", "시야각", "밝기", "명암", "해상도", "선명", "저음", "서라운드"
]

CATEGORY_FEATURE_PATTERNS = {
    ("07. 스마트 사용성", "07-01. 채널/컨텐츠 APP"): [
        "app", "apps", "channel", "content", "streaming", "ott",
        "앱", "채널", "콘텐츠", "스트리밍", "ott"
    ],
    ("07. 스마트 사용성", "07-02. 구동성/구동속도"): [
        "fast", "slow", "lag", "speed", "loading",
        "속도", "빠름", "느림", "로딩", "렉"
    ],
    ("07. 스마트 사용성", "07-03. 메뉴 구성/UI"): [
        "menu", "ui", "interface", "navigation",
        "메뉴", "ui", "인터페이스", "탐색"
    ],
    ("07. 스마트 사용성", "07-04. SW/OS"): [
        "os", "software", "update", "bug",
        "os", "소프트웨어", "업데이트", "버그"
    ],
    ("07. 스마트 사용성", "07-05. 컨텐츠 탐색 사용성"): [
        "search", "browse", "discover",
        "탐색", "검색", "브라우징"
    ],
    ("07. 스마트 사용성", "07-06. 리모컨 사용성"): [
        "remote", "control", "button", "layout", "pointer", "backlight", "easy",
        "리모컨", "조작", "버튼", "레이아웃", "포인터", "백라이트", "편리"
    ],
    ("07. 스마트 사용성", "07-07. 리모컨 디자인"): [
        "remote design", "remote look",
        "리모컨 디자인", "외관"
    ],
    ("07. 스마트 사용성", "07-08. 음성 인식/조작"): [
        "voice", "speech", "assistant",
        "음성", "음성인식", "음성조작"
    ],
    ("07. 스마트 사용성", "07-09. 게임 기능"): [
        "game", "gaming", "latency",
        "게임", "게이밍", "지연"
    ],
    ("07. 스마트 사용성", "07-10. 부가 기능"): [
        "feature", "features", "extra function",
        "기능", "부가기능"
    ],
    ("07. 스마트 사용성", "07-11. 홈 IoT"): [
        "iot", "smart home",
        "iot", "스마트홈"
    ],
    ("07. 스마트 사용성", "07-12. 모바일 연동"): [
        "mobile", "phone", "cast", "mirror",
        "모바일", "휴대폰", "캐스트", "미러링"
    ],
    ("07. 스마트 사용성", "07-13. 광고"): [
        "ad", "ads", "advertisement",
        "광고"
    ],
    ("07. 스마트 사용성", "07-20. 전반적 스마트 사용성"): [
        "smart", "usability", "easy to use",
        "스마트", "사용성", "편리"
    ],
}


def get_feature_patterns(cate_1_depth: str, cate_2_depth: str) -> List[str]:
    return COMMON_FEATURE_PATTERNS + CATEGORY_FEATURE_PATTERNS.get((cate_1_depth, cate_2_depth), [])


# --------------------------------------------------
# Overall Detection Helpers
# --------------------------------------------------
def has_specific_reason_for_category(text: str, clue_keywords: List[str], cate_1_depth: str, cate_2_depth: str) -> bool:
    memo = clean_text(text).lower()
    if any(clean_text(k).lower() in memo for k in clue_keywords if clean_text(k)):
        return True
    feature_patterns = get_feature_patterns(cate_1_depth, cate_2_depth)
    return any(clean_text(p).lower() in memo for p in feature_patterns if clean_text(p))


def should_be_overall_for_category(text: str, clue_keywords: List[str], generic_terms: List[str], cate_1_depth: str, cate_2_depth: str) -> bool:
    memo = clean_text(text).lower()
    if not memo:
        return False
    if len(memo) > 14:
        return False
    if has_specific_reason_for_category(memo, clue_keywords, cate_1_depth, cate_2_depth):
        return False
    if len(re.findall(r"[A-Za-z\uac00-\ud7a3]+", memo)) > 3:
        return False
    return any(clean_text(t).lower() in memo for t in generic_terms if clean_text(t))


# --------------------------------------------------
# Overall Sentiment Detection ("전반적" 그룹 전용 rule pre-filter)
# --------------------------------------------------
SENTIMENT_EXPRESSIONS = {
    "좋아요", "좋아", "좋음", "좋다", "좋습니다", "좋네요", "좋았", "좋은",
    "최고", "만족", "대만족", "훌륭", "완벽", "추천", "강추", "굿",
    "괜찮", "맘에", "마음에", "만족스러", "괴다",
    "별로", "나쁨", "나빠", "나쁜", "불만", "실망", "후회", "최악",
    "싫어", "아쉬", "그저그래", "그냥", "보통",
    "너무", "정말", "아주", "매우", "진짜",
    "good", "great", "excellent", "awesome", "amazing", "perfect",
    "wonderful", "fantastic", "love", "loved", "best", "nice",
    "superb", "outstanding", "incredible", "brilliant",
    "bad", "terrible", "awful", "horrible", "worst", "poor",
    "suck", "sucks", "disappointed", "disappointing", "hate", "ugly",
    "useless", "okay", "ok", "fine", "not bad", "rubbish", "crap",
    "very", "really", "so", "super", "quite", "absolutely", "totally",
    "highly", "extremely", "pretty", "just",
    "gut", "toll", "super", "perfekt", "schlecht", "ausgezeichnet", "sehr",
    "bueno", "malo", "excelente", "genial", "perfecto", "muy",
}

FILLER_WORDS = {
    "it", "is", "was", "the", "a", "an", "i", "my", "this", "that",
    "its", "im", "am", "are", "be", "been", "being",
    "not", "no", "yes", "yeah", "yep", "nope",
    "에요", "이에요", "입니다", "해요", "요", "네요",
    "다", "은", "는", "이", "가", "를", "을", "도", "만",
    "데", "에", "서", "로",
}

FEATURE_EXCLUSION_WORDS = [
    "크", "작", "커", "작아", "big", "small", "large", "tiny", "huge", "사이즈", "인치",
    "밝", "어두", "밝아", "어두워", "bright", "dark", "dim",
    "시끄러", "조용", "loud", "quiet", "silent",
    "빠르", "느리", "fast", "slow",
    "무겁", "가벼", "heavy", "light",
    "선명", "흐릿", "sharp", "clear", "blurry",
    "얇", "두꺼", "thin", "thick",
    "색감", "색상", "color", "colour",
    "화질", "음질", "화면", "소리", "사운드", "디자인", "설치",
    "리모컨", "앱", "게임", "연결", "wifi", "블루투스", "hdmi",
    "패널", "베젤", "스탠드", "가격", "배송", "보증",
    "picture", "sound", "audio", "screen", "display", "remote",
    "price", "delivery", "install", "panel", "bezel",
]


def is_pure_overall_sentiment(memo: str) -> bool:
    if not memo:
        return False
    text = memo.strip()
    if len(text) >= 25:
        return False
    text_lower = text.lower()
    for w in FEATURE_EXCLUSION_WORDS:
        if w.lower() in text_lower:
            return False
    tokens = re.findall(r"[a-z\uac00-\ud7a3]+", text_lower)
    if not tokens:
        return False
    has_sentiment = False
    for tok in tokens:
        if tok in SENTIMENT_EXPRESSIONS:
            has_sentiment = True
        elif tok in FILLER_WORDS:
            continue
        else:
            return False
    return has_sentiment


# --------------------------------------------------
# rule_map / topic_pool 로드
# --------------------------------------------------
rule_map = {
    (r["cate_1_depth"], r["cate_2_depth"], int(r["sc_measurement"])): r.asDict(recursive=True)
    for r in spark.table(RULE_PROFILE_TABLE).toLocalIterator()
}

topic_pool_group_map = {}
for row in spark.table(TOPIC_POOL_TABLE).toLocalIterator():
    key = (row["cate_1_depth"], row["cate_2_depth"], int(row["sc_measurement"]))
    reps = []
    try:
        reps = json.loads(row["representative_memos_json"]) if row["representative_memos_json"] else []
    except Exception:
        pass
    topic_pool_group_map.setdefault(key, []).append({
        "topic": row["topic"],
        "description": row["description"],
        "representative_memos": reps,
    })


# --------------------------------------------------
# build_classify_messages (기타 + 오분류 포함)
# --------------------------------------------------
def build_classify_messages(batch_rows, topic_pool_payload, rule_row, c1, c2, sc):
    clue_keywords = json.loads(rule_row["clue_keywords_json"]) if rule_row["clue_keywords_json"] else []
    non_overall_examples = json.loads(rule_row["non_overall_examples_json"]) if rule_row["non_overall_examples_json"] else []
    category_patterns = get_feature_patterns(c1, c2)

    topic_info = []
    for t in topic_pool_payload:
        info = {"topic": t["topic"], "description": t["description"]}
        rep_memos = t.get("representative_memos", [])
        if rep_memos:
            info["examples"] = rep_memos[:3]
        topic_info.append(info)
    topic_info.append({"topic": "기타", "description": "어느 주제에도 속하지 않는 메모"})
    topic_info.append({"topic": "오분류", "description": "감성 불일치 또는 카테고리 불일치 메모"})

    system = f"""
You are a VOC classifier for TV review topic classification.

Classify each memo into exactly one topic from the fixed topic list.
Every memo must belong to exactly one topic.

Rules:
- The task is to identify WHY the writer evaluated {c2} as {sc_label(sc)}.
- Overall topic is '{clean_text(rule_row["overall_topic_label"])}'.
- {clean_text(rule_row["overall_usage_rule"])}
- {clean_text(rule_row["specific_reason_rule"])}
- clue keywords:
  {json.dumps(clue_keywords, ensure_ascii=False)}
- category fallback feature patterns:
  {json.dumps(category_patterns[:60], ensure_ascii=False)}
- non-overall examples:
  {json.dumps(non_overall_examples, ensure_ascii=False)}
- Do not invent any new topic.
- explanation must be a short Korean sentence.

Return only JSON:
{{
  "results": [
    {{"row_id": "", "topic": "", "explanation": ""}}
  ]
}}
"""
    user = (
        "Fixed topics (with description and example memos):\n"
        + json.dumps(topic_info, ensure_ascii=False)
        + "\n\nMemos:\n"
        + json.dumps(
            [{"row_id": str(r["_row_id"]), "memo": clean_text(r.get("memo_for_llm", r["memo"]))} for r in batch_rows],
            ensure_ascii=False)
    )
    return [
        {"role": "system", "content": clean_text(system)},
        {"role": "user", "content": clean_text(user)},
    ]


# --------------------------------------------------
# build_batch_reclassify_messages (Phase 2: Overall 재분류)
# --------------------------------------------------
def build_batch_reclassify_messages(batch_rows, topic_pool_payload, rule_row, c1, c2):
    overall_topic = clean_text(rule_row["overall_topic_label"])
    specific_topic_payload = [
        {"topic": t["topic"], "description": t["description"]}
        for t in topic_pool_payload if clean_text(t["topic"]) != overall_topic
    ]
    clue_keywords = json.loads(rule_row["clue_keywords_json"]) if rule_row["clue_keywords_json"] else []
    category_patterns = get_feature_patterns(c1, c2)

    system = f"""
You are a VOC classifier for TV review topic classification.

These memos were incorrectly over-generalized as '{overall_topic}'.
Reclassify each memo using only the non-general topics below.

Rules:
- Choose exactly one non-general topic for each memo.
- Each memo contains a specific reason and must not remain as '{overall_topic}'.
- clue keywords:
  {json.dumps(clue_keywords, ensure_ascii=False)}
- category fallback feature patterns:
  {json.dumps(category_patterns[:60], ensure_ascii=False)}
- explanation must be a short Korean sentence.

Return only JSON:
{{
  "results": [
    {{"row_id": "", "topic": "", "explanation": ""}}
  ]
}}
"""
    user = (
        "Allowed non-general topics:\n"
        + json.dumps(specific_topic_payload, ensure_ascii=False)
        + "\n\nMemos:\n"
        + json.dumps(
            [{"row_id": str(r["_row_id"]), "memo": clean_text(r.get("memo_for_llm", r["memo"]))} for r in batch_rows],
            ensure_ascii=False)
    )
    return [
        {"role": "system", "content": clean_text(system)},
        {"role": "user", "content": clean_text(user)},
    ]


# --------------------------------------------------
# classify_batch_and_merge (Phase 0 + 1 + 2, v4 원본 동일)
# --------------------------------------------------
def classify_batch_and_merge(
    source_rows: List[Dict[str, Any]],
    topic_payload: List[Dict[str, Any]],
    rule_row: Dict[str, Any],
    c1: str, c2: str, sc: int,
) -> List[Dict[str, Any]]:
    clue_keywords = json.loads(rule_row["clue_keywords_json"]) if rule_row["clue_keywords_json"] else []
    generic_terms = json.loads(rule_row["generic_terms_json"]) if rule_row["generic_terms_json"] else []
    overall_topic = clean_text(rule_row["overall_topic_label"])
    total_batches = (len(source_rows) + CLASSIFY_BATCH_SIZE - 1) // CLASSIFY_BATCH_SIZE

    # Phase 0: 메모 요약
    classify_rows, summary_map = prepare_rows_for_classification(source_rows)
    summarized_count = len(summary_map)
    if summarized_count > 0:
        print(f"    [v4 요약] {summarized_count}건 메모 요약 완료")

    # Phase 1: 초기 배치 분류 (요약본 사용)
    classified_rows = []
    for batch_no, batch in enumerate(chunk_list(classify_rows, CLASSIFY_BATCH_SIZE), start=1):
        batch_start_ts = time.time()
        print(f"    [BATCH] {batch_no}/{total_batches}, rows={len(batch)}")
        try:
            batch_result = call_llm(
                build_classify_messages(batch, topic_payload, rule_row, c1, c2, sc)
            )
            result_map = {
                str(item.get("row_id")): item
                for item in batch_result.get("results", [])
                if isinstance(item, dict)
            }
        except Exception as exc:
            print(f"    [BATCH ERROR] {batch_no}/{total_batches}: {repr(exc)}")
            result_map = {}
        for row in batch:
            mapped = result_map.get(str(row["_row_id"]), {})
            classified_rows.append({
                "_row_id": row["_row_id"],
                "cate_1_depth": c1, "cate_2_depth": c2, "sc_measurement": sc,
                "memo": row["memo"],
                "memo_for_llm": row.get("memo_for_llm", row["memo"]),
                "memo_id": row.get("memo_id"),
                "topic_raw": clean_text(mapped.get("topic")),
                "explanation_raw": clean_text(mapped.get("explanation")),
            })
        print(f"    [BATCH DONE] {batch_no}/{total_batches}, {round(time.time() - batch_start_ts, 2)}s")
        time.sleep(RATE_LIMIT_SECONDS)

    # Phase 2: Overall 재분류
    reclass_candidates = [
        cr for cr in classified_rows
        if cr["topic_raw"] == overall_topic
        and has_specific_reason_for_category(cr.get("memo_for_llm", cr["memo"]), clue_keywords, c1, c2)
        and not should_be_overall_for_category(cr.get("memo_for_llm", cr["memo"]), clue_keywords, generic_terms, c1, c2)
    ]
    if reclass_candidates:
        reclass_total = len(reclass_candidates)
        reclass_batches = (reclass_total + CLASSIFY_BATCH_SIZE - 1) // CLASSIFY_BATCH_SIZE
        print(f"    [RECLASS] {reclass_total} rows -> {reclass_batches} batch(es)")
        reclass_result_map = {}
        for rb_no, rb in enumerate(chunk_list(reclass_candidates, CLASSIFY_BATCH_SIZE), start=1):
            try:
                reclass_result = call_llm(
                    build_batch_reclassify_messages(rb, topic_payload, rule_row, c1, c2)
                )
                for item in reclass_result.get("results", []):
                    if isinstance(item, dict) and item.get("row_id"):
                        reclass_result_map[str(item["row_id"])] = item
            except Exception as exc:
                print(f"    [RECLASS ERROR] batch {rb_no}: {repr(exc)}")
            time.sleep(RATE_LIMIT_SECONDS)
        applied = 0
        for cr in classified_rows:
            mapped = reclass_result_map.get(str(cr["_row_id"]))
            if mapped:
                retry_topic = clean_text(mapped.get("topic"))
                if retry_topic and retry_topic != overall_topic:
                    cr["topic_raw"] = retry_topic
                    cr["explanation_raw"] = clean_text(mapped.get("explanation")) or cr["explanation_raw"]
                    applied += 1
        print(f"    [RECLASS DONE] {applied}/{reclass_total} reclassified")

    # Output: LLM 선정 토픽 그대로 사용 (기타 병합 없음) + memo_summary
    topic_desc_map = {t["topic"]: t["description"] for t in topic_payload}
    original_memo_map = {str(r["_row_id"]): r["memo"] for r in source_rows}
    memo_id_map = {str(r["_row_id"]): r.get("memo_id") for r in source_rows}

    final_rows = []
    for rd in classified_rows:
        raw_topic = clean_text(rd["topic_raw"])
        raw_expl = clean_text(rd["explanation_raw"])
        final_topic = raw_topic if raw_topic else "기타"
        final_explanation = raw_expl or topic_desc_map.get(final_topic, "")
        row_id_str = str(rd["_row_id"])
        original_memo = original_memo_map.get(row_id_str, rd["memo"])
        memo_summary = summary_map.get(row_id_str) or ""
        final_rows.append({
            "_row_id": rd["_row_id"],
            "memo_id": memo_id_map.get(row_id_str) or build_memo_id_value(c1, c2, sc, original_memo),
            "cate_1_depth": c1, "cate_2_depth": c2, "sc_measurement": sc,
            "memo": original_memo,
            "memo_summary": memo_summary,
            "topic": final_topic,
            "description": final_explanation,
        })
    return final_rows


print(f"[OK] Classification Functions (v5 원본 동일)")
print(f"  rule_map: {len(rule_map)} groups")
print(f"  topic_pool: {len(topic_pool_group_map)} groups")

# COMMAND ----------

# DBTITLE 1,6) 전수 분류 메인 루프
# ============================================================
# 6) 전수 분류 메인 루프 (v5 원본 동일)
#    - 그룹별 전체 미분류 메모 대상 (샘플링 없음)
#    - "전반적" 그룹: is_pure_overall_sentiment 규칙 자동분류 + 나머지 LLM
#    - 일반 그룹: 전체 LLM 분류 (Phase 0+1+2)
#    - CHUNK_SIZE(500건) 단위 checkpoint 저장
# ============================================================

pipeline_start_ts = time.time()
total_groups = len(group_rows)
processed = 0
skipped = 0

print(f"\n{'='*60}")
print(f"[START] Round {SAMPLING_ROUND} 전수 분류 - {total_groups}개 그룹")
print(f"  DETAIL_TABLE = {DETAIL_TABLE}")
print(f"  MEMO_SUMMARIZE_THRESHOLD = {MEMO_SUMMARIZE_THRESHOLD}자")
print(f"{'='*60}\n")

for idx, g in enumerate(group_rows, start=1):
    c1 = TARGET_CATE_1
    c2 = g["cate_2_depth"]
    sc = int(g["sc_measurement"])
    key = (c1, c2, sc)

    if key not in rule_map:
        skipped += 1
        print(f"[SKIP] {idx}/{total_groups} {c2} sc={sc} - rule profile 없음")
        continue
    if key not in topic_pool_group_map:
        skipped += 1
        print(f"[SKIP] {idx}/{total_groups} {c2} sc={sc} - topic pool 없음")
        continue

    group_start_ts = time.time()
    rule_row = rule_map[key]
    topic_payload = topic_pool_group_map[key]
    is_overall_group = "전반적" in c2

    # 미분류 메모 로드
    # - source llm_round is null 이거나
    # - source에는 있지만 최종 산출물에 없어 재분류 대상으로 복구된 memo
    # - 현재 round DETAIL_TABLE에 저장된 memo는 제외
    source_df = with_memo_id(spark.sql(f"""
        SELECT DISTINCT cate_1_depth, cate_2_depth, sc_measurement, memo, memo_id
        FROM {SOURCE_RECLASSIFY_SCOPE_VIEW}
        WHERE cate_1_depth = '{c1}'
          AND cate_2_depth = '{c2}'
          AND sc_measurement = {sc}
    """))

    if spark.catalog.tableExists(DETAIL_TABLE):
        source_df = exclude_existing_source_rows(
            source_df,
            DETAIL_TABLE,
            f"cate_1_depth='{c1}' AND cate_2_depth='{c2}' AND sc_measurement={sc}",
        )

    remaining_count = source_df.count()
    if remaining_count == 0:
        skipped += 1
        print(f"[DONE] {idx}/{total_groups} {c2} sc={sc} - 미분류 0건")
        log_group_done(c2, sc, 0)
        continue

    print(f"\n[GROUP] {idx}/{total_groups} | {c2} | sc={sc} | 미분류 {remaining_count:,}건 | overall={is_overall_group}")

    try:
        llm_rows = []

        if is_overall_group:
            # ==================================================
            # A-1. 전반적 그룹: rule pre-filter 후 LLM
            # ==================================================
            all_memos_df = source_df.withColumn("_row_id", F.monotonically_increasing_id())
            total_count = all_memos_df.count()

            is_overall_udf = F.udf(is_pure_overall_sentiment, T.BooleanType())
            tagged_df = all_memos_df.withColumn("_is_overall", is_overall_udf(F.col("memo")))

            overall_df = tagged_df.where(F.col("_is_overall") == True).select(
                "cate_1_depth", "cate_2_depth", "sc_measurement", "memo", "memo_id", "_row_id"
            )
            non_overall_df = tagged_df.where(F.col("_is_overall") == False).select(
                "cate_1_depth", "cate_2_depth", "sc_measurement", "memo", "memo_id", "_row_id"
            )

            overall_count = overall_df.count()
            non_overall_count = non_overall_df.count()
            print(f"  [전반적 PRE-FILTER] total={total_count}, overall={overall_count}, non_overall={non_overall_count}")

            # 순수 감정 표현 → rule 기반 자동 분류 (LLM 호출 없이)
            overall_topic_name = clean_text(rule_row["overall_topic_label"])
            for r in overall_df.toLocalIterator():
                llm_rows.append({
                    "_row_id": r["_row_id"], "cate_1_depth": c1, "cate_2_depth": c2,
                    "sc_measurement": sc, "memo_id": r["memo_id"], "memo": r["memo"],
                    "memo_summary": "",
                    "topic": overall_topic_name,
                    "description": "순수 감정 표현 (rule 기반 자동 분류)",
                })

            # non-overall 메모: LLM 분류 (전수)
            if non_overall_count > 0:
                source_rows = [r.asDict(recursive=True) for r in non_overall_df.toLocalIterator()]
                for chunk_no, chunk in enumerate(chunk_list(source_rows, CHUNK_SIZE), start=1):
                    print(f"  [CHUNK {chunk_no}] {len(chunk)}건 LLM 분류...")
                    chunk_results = classify_batch_and_merge(
                        chunk, topic_payload, rule_row, c1, c2, sc
                    )
                    llm_rows += chunk_results
                    # 체크포인트 저장
                    if chunk_results:
                        append_table(spark.createDataFrame(pd.DataFrame(chunk_results)), DETAIL_TABLE)
                        print(f"  [CHUNK {chunk_no} SAVED] {len(chunk_results)}건")

            # overall 자동분류 결과도 저장
            if overall_count > 0:
                overall_rows_to_save = [r for r in llm_rows if r.get("description") == "순수 감정 표현 (rule 기반 자동 분류)"]
                if overall_rows_to_save:
                    append_table(spark.createDataFrame(pd.DataFrame(overall_rows_to_save)), DETAIL_TABLE)
                    print(f"  [OVERALL SAVED] {len(overall_rows_to_save)}건 rule 기반 자동분류")

        else:
            # ==================================================
            # A-2. 일반 그룹: 전체 메모 LLM 분류 (CHUNK 단위)
            # ==================================================
            all_rows = [
                r.asDict(recursive=True)
                for r in source_df.withColumn("_row_id", F.monotonically_increasing_id()).toLocalIterator()
            ]
            for chunk_no, chunk in enumerate(chunk_list(all_rows, CHUNK_SIZE), start=1):
                chunk_start = time.time()
                print(f"  [CHUNK {chunk_no}] {len(chunk)}건 LLM 분류...")
                try:
                    results = classify_batch_and_merge(
                        chunk, topic_payload, rule_row, c1, c2, sc
                    )
                    if results:
                        for row in results:
                            row.setdefault("memo_summary", "")
                        append_table(spark.createDataFrame(pd.DataFrame(results)), DETAIL_TABLE)
                        llm_rows += results
                        print(f"  [CHUNK {chunk_no} DONE] {len(results)}건 ({round(time.time()-chunk_start,1)}s) | 누적: {len(llm_rows):,}")
                except Exception as exc:
                    print(f"  [CHUNK {chunk_no} FAILED] {repr(exc)}")
                    continue

        # ==================================================
        # B. 그룹 완료
        # ==================================================
        processed += 1
        log_group_done(c2, sc, len(llm_rows))
        elapsed = round(time.time() - group_start_ts, 2)
        total_elapsed = round(time.time() - pipeline_start_ts)
        remaining = total_groups - idx
        avg_per_group = total_elapsed / processed if processed else 0
        eta = round(remaining * avg_per_group)

        print(f"  [SAVE] LLM={len(llm_rows)} -> {DETAIL_TABLE}")
        print(f"  [DONE] {idx}/{total_groups} | {elapsed}s")
        print(f"  [CHECKPOINT] processed={processed}, skipped={skipped}, elapsed={total_elapsed}s, ETA≈{eta}s")

    except Exception as exc:
        print(f"  [FAILED] {idx}/{total_groups} | {c2} | {repr(exc)[:200]}")

# --------------------------------------------------
# Pipeline 완료 요약
# --------------------------------------------------
total_elapsed = round(time.time() - pipeline_start_ts)
print(f"\n{'='*60}")
print(f"[PIPELINE COMPLETE] Round {SAMPLING_ROUND}")
print(f"  processed={processed}, skipped={skipped}, total_elapsed={total_elapsed}s")
if spark.catalog.tableExists(DETAIL_TABLE):
    total_detail = spark.table(DETAIL_TABLE).count()
    print(f"  {DETAIL_TABLE} total = {total_detail:,}")
print(f"{'='*60}")
print(f"\n\u2192 \ub2e4\uc74c \ub2e8\uacc4: Cell 7 \uc2e4\ud589 (catalog \uc870\uc778 + integration \uba38\uc9c0)")

# COMMAND ----------

# DBTITLE 1,6-1) 진행현황 확인
# ============================================================
# 6-1) 진행현황 확인
#      - 그룹별 전체 메모 vs 분류완료 메모 vs 잔여
#      - 500건 단위로 어디까지 완료되었는지 표시
#      - 실행 중 / 실행 후 언제든 이 셀만 실행하면 현황 확인 가능
# ============================================================

print(f"{'='*70}")
print(f"[Round {SAMPLING_ROUND}] 스마트 사용성 전수분류 진행현황")
print(f"{'='*70}\n")

# 1. 전체 대상 그룹별 소스 메모 수
all_groups = spark.sql(f"""
    SELECT cate_2_depth, sc_measurement, COUNT(DISTINCT memo) as total_memos
    FROM {SOURCE_TABLE}
    WHERE {_source_filter_sql(TARGET_CATE_1)}
    GROUP BY cate_2_depth, sc_measurement
    ORDER BY cate_2_depth, sc_measurement
""").toPandas()

# 2. DETAIL_TABLE에서 분류 완료 건수
if spark.catalog.tableExists(DETAIL_TABLE):
    classified = spark.sql(f"""
        SELECT cate_2_depth, sc_measurement, COUNT(*) as classified_count
        FROM {DETAIL_TABLE}
        WHERE cate_1_depth = '{TARGET_CATE_1}'
        GROUP BY cate_2_depth, sc_measurement
    """).toPandas()
else:
    classified = pd.DataFrame(columns=["cate_2_depth", "sc_measurement", "classified_count"])

# 3. 병합
import pandas as pd
status_df = all_groups.merge(classified, on=["cate_2_depth", "sc_measurement"], how="left")
status_df["classified_count"] = status_df["classified_count"].fillna(0).astype(int)
status_df["remaining"] = status_df["total_memos"] - status_df["classified_count"]
status_df["progress_pct"] = (status_df["classified_count"] / status_df["total_memos"] * 100).round(1)
status_df["chunks_done"] = (status_df["classified_count"] / CHUNK_SIZE).apply(lambda x: int(x))
status_df["chunks_total"] = (status_df["total_memos"] / CHUNK_SIZE).apply(lambda x: int(x) + (1 if x % 1 > 0 else 0))

# 4. 출력
total_memos_all = status_df["total_memos"].sum()
total_classified = status_df["classified_count"].sum()
total_remaining = status_df["remaining"].sum()

print(f"[전체 요약]")
print(f"  대상 그룹: {len(status_df)}개")
print(f"  전체 메모: {total_memos_all:,}건")
print(f"  분류완료: {total_classified:,}건 ({total_classified/total_memos_all*100:.1f}%)")
print(f"  잔여:     {total_remaining:,}건")
print(f"  CHUNK_SIZE: {CHUNK_SIZE}건\n")

print(f"{'─'*70}")
print(f"{'cate_2_depth':<35} {'sc':>3} {'total':>7} {'done':>7} {'remain':>7} {'%':>6} {'chunks':>10}")
print(f"{'─'*70}")

for _, row in status_df.iterrows():
    sc_str = "+" if row["sc_measurement"] == 1 else "-"
    chunks_str = f"{row['chunks_done']}/{row['chunks_total']}"
    done_marker = " ✅" if row["remaining"] == 0 else ""
    print(
        f"{row['cate_2_depth']:<35} {sc_str:>3} "
        f"{row['total_memos']:>7,} {row['classified_count']:>7,} {row['remaining']:>7,} "
        f"{row['progress_pct']:>5.1f}% {chunks_str:>10}{done_marker}"
    )

print(f"{'─'*70}")

# 5. PROGRESS_TABLE 완료 기록
if spark.catalog.tableExists(PROGRESS_TABLE):
    progress_df = spark.table(PROGRESS_TABLE).orderBy("event_ts", ascending=False)
    done_count = progress_df.where("status = 'done'").count()
    print(f"\n[PROGRESS_TABLE] 완료 기록: {done_count}건")
    display(progress_df.limit(10))
else:
    print(f"\n[PROGRESS_TABLE] 아직 생성되지 않음")

# COMMAND ----------

# DBTITLE 1,7) Integration Merge
# ============================================================
# 7) Round 테이블에 catalog_v2 조인 후 Integration에 합치기
#    1. DETAIL_TABLE(round10) + catalog_v2 조인 → topic_rev, main_topic 추가
#    2. 전반적 긍정/부정 NULL 매핑
#    3. llm_round 컬럼 추가 (r10)
#    4. 보강된 round 데이터를 integration 테이블에 머지
#    5. memo_summary는 DETAIL_TABLE에만 유지, integration에는 미포함
# ============================================================

INTEGRATION_TABLE = V1_DETAIL_TABLE
CATALOG_V2_TABLE = f"{SAVE_DB}.category_topic_catalog_v2"

if not spark.catalog.tableExists(DETAIL_TABLE):
    print(f"[SKIP] {DETAIL_TABLE} 없음 - 분류 실행 후 재실행")
else:
    print(f"[STEP 1] {DETAIL_TABLE} + {CATALOG_V2_TABLE} 조인")

    # 1. Round DETAIL_TABLE 로드
    round_df = spark.table(DETAIL_TABLE)
    for col in ["topic_rev", "main_topic", "llm_round"]:
        if col in round_df.columns:
            round_df = round_df.drop(col)

    # 2. catalog_v2 조인용 DF
    catalog_df = (
        spark.table(CATALOG_V2_TABLE)
        .select("cate_1_depth", "cate_2_depth", "sc_measurement", "topic", "topic_rev", "main_topic")
        .dropDuplicates(["cate_1_depth", "cate_2_depth", "sc_measurement", "topic"])
    )

    # 3. LEFT JOIN
    enriched_df = round_df.join(
        catalog_df,
        on=["cate_1_depth", "cate_2_depth", "sc_measurement", "topic"],
        how="left"
    )

    # 4. 전반적 긍정/부정 NULL 매핑
    enriched_df = enriched_df.withColumn(
        "topic_rev",
        F.when(
            F.col("topic_rev").isNull() & F.col("topic").isin("전반적 긍정", "전반적 부정"),
            F.col("topic")
        ).otherwise(F.col("topic_rev"))
    ).withColumn(
        "main_topic",
        F.when(
            F.col("main_topic").isNull() & F.col("topic").isin("전반적 긍정", "전반적 부정"),
            F.col("topic")
        ).otherwise(F.col("main_topic"))
    )

    # 5. llm_round
    enriched_df = enriched_df.withColumn("llm_round", F.lit(ROUND_LABEL))

    publishable_detail_df, reclassify_detail_df = split_publishable_vs_reclassify(enriched_df)

    round_count = enriched_df.count()
    null_count = enriched_df.where(F.col("topic_rev").isNull()).count()
    publishable_count = publishable_detail_df.count()
    reclassify_count = reclassify_detail_df.count()
    print(f"  round rows: {round_count:,}")
    print(f"  topic_rev NULL: {null_count:,}")
    print(f"  publishable rows: {publishable_count:,}")
    print(f"  reclassify target rows: {reclassify_count:,}")
    print(f"  llm_round: {ROUND_LABEL}")

    # 6. DETAIL_TABLE 덮어쓰기
    #    - topic_rev NULL + 비-기타/오분류는 재분류 대상으로 제외
    publishable_detail_df.write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(DETAIL_TABLE)
    print(f"  [SAVE] {DETAIL_TABLE} 저장 완료")

    # ============================================================
    # STEP 2: Integration 테이블에 머지 (memo_summary 제외, 동일 memo는 round10 결과로 덮어쓰기)
    # ============================================================
    print(f"\n[STEP 2] {INTEGRATION_TABLE}에 머지 (memo_summary 제외, 동일 key 덮어쓰기)")

    enriched_for_integration = (
        publishable_detail_df.drop("memo_summary")
        if "memo_summary" in publishable_detail_df.columns
        else publishable_detail_df
    )

    if spark.catalog.tableExists(INTEGRATION_TABLE):
        integ_df = spark.table(INTEGRATION_TABLE)
        for col in ["topic_rev", "main_topic", "llm_round", "memo_id"]:
            if col not in integ_df.columns:
                spark.sql(f"ALTER TABLE {INTEGRATION_TABLE} ADD COLUMN {col} STRING")
                print(f"  [컨럼 추가] {col}")

        v1_before = spark.table(INTEGRATION_TABLE).count()
        print(f"  integration 기존: {v1_before:,}건")

        target_columns = spark.table(INTEGRATION_TABLE).columns
        merge_keys = ["memo_id"] if "memo_id" in target_columns else ["memo", "cate_1_depth", "cate_2_depth", "sc_measurement"]
        merge_rows = enriched_for_integration

        for col_name in target_columns:
            if col_name not in merge_rows.columns:
                merge_rows = merge_rows.withColumn(col_name, F.lit(None).cast("string"))
        merge_rows = merge_rows.select(*target_columns)
        merge_rows = dedupe_merge_source(merge_rows, merge_keys)

        merge_count = merge_rows.count()
        print(f"  [ROUND{SAMPLING_ROUND} 반영 대상] {merge_count:,}건")

        if merge_count > 0:
            merge_table_by_keys(merge_rows, INTEGRATION_TABLE, merge_keys)
            print(f"  [MERGE] {merge_count:,}건 반영 (llm_round={ROUND_LABEL})")
        else:
            print(f"  [SKIP] 반영할 round 데이터 없음")

        v1_after = spark.table(INTEGRATION_TABLE).count()
        print(f"\n  integration: {v1_before:,} -> {v1_after:,} ({v1_after - v1_before:+,})")
    else:
        enriched_for_integration.write.mode("overwrite").format("delta").saveAsTable(INTEGRATION_TABLE)
        print(f"  [CREATE] {INTEGRATION_TABLE} 생성 ({enriched_for_integration.count():,}건)")

    print(f"\n[ALL DONE] Round{SAMPLING_ROUND} catalog 조인 + integration 머지 완료")

# COMMAND ----------

# DBTITLE 1,8) Summary Table
# ============================================================
# 8) Summary Table
# ============================================================

if spark.catalog.tableExists(DETAIL_TABLE):
    detail_df = spark.table(DETAIL_TABLE).where(f"cate_1_depth = '{TARGET_CATE_1}'")

    summary_df = (
        detail_df
        .groupBy("cate_1_depth", "cate_2_depth", "sc_measurement", "topic")
        .agg(F.count("*").alias("review_count"))
        .withColumn(
            "group_total",
            F.sum("review_count").over(
                Window.partitionBy("cate_1_depth", "cate_2_depth", "sc_measurement")
            ),
        )
        .withColumn("review_share_pct", F.round(F.col("review_count") / F.col("group_total") * 100, 2))
        .orderBy("cate_2_depth", "sc_measurement", F.desc("review_count"))
    )

    summary_df.write.mode("overwrite").format("delta").saveAsTable(SUMMARY_TABLE)
    print(f"[SAVE] {SUMMARY_TABLE}")
    display(summary_df.limit(50))
else:
    print("[SKIP] DETAIL_TABLE 없음")

# COMMAND ----------

# DBTITLE 1,9) Version Tracking Log
# ============================================================
# 9) Version Tracking Log
# ============================================================

from datetime import datetime

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {VERSION_TRACKING_TABLE} (
        version STRING,
        round_number INT,
        change_description STRING,
        change_detail STRING,
        prompt_version STRING,
        endpoint STRING,
        detail_table STRING,
        created_at TIMESTAMP,
        created_by STRING
    ) USING DELTA
""")

existing = spark.sql(f"SELECT version, round_number FROM {VERSION_TRACKING_TABLE}").collect()
existing_keys = {(r["version"], r["round_number"]) for r in existing}

if ("v5", SAMPLING_ROUND) not in existing_keys:
    record = [{
        "version": "v5",
        "round_number": SAMPLING_ROUND,
        "change_description": f"Round {SAMPLING_ROUND}: 스마트 사용성(07) 전수분류",
        "change_detail": (
            f"cate_1_depth='07. 스마트 사용성' 전체 하위그룹 전수분류. "
            f"샘플링 없이 전체 미분류 메모 대상. "
            f"DETAIL 테이블: {DETAIL_TABLE}. "
            f"동일 v4 로직 (200자 초과 요약 + 기타/오분류 특수토픽). "
            f"catalog_v2 조인 후 topic_rev/main_topic 추가. "
            f"Integration 테이블에 llm_round={ROUND_LABEL}로 머지. "
            f"voc_llm_topic_classification에 적재. "
            f"500건 chunk checkpoint."
        ),
        "prompt_version": PROMPT_VERSION,
        "endpoint": ENDPOINT,
        "detail_table": DETAIL_TABLE,
        "created_at": datetime.now(),
        "created_by": "jungryo.lee@lge.com",
    }]
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
    schema = StructType([
        StructField("version", StringType(), True),
        StructField("round_number", IntegerType(), True),
        StructField("change_description", StringType(), True),
        StructField("change_detail", StringType(), True),
        StructField("prompt_version", StringType(), True),
        StructField("endpoint", StringType(), True),
        StructField("detail_table", StringType(), True),
        StructField("created_at", TimestampType(), True),
        StructField("created_by", StringType(), True),
    ])
    spark.createDataFrame(record, schema=schema).write.mode("append").format("delta").saveAsTable(VERSION_TRACKING_TABLE)
    print(f"[VERSION LOG] Round {SAMPLING_ROUND} 기록 추가")
else:
    print(f"[VERSION LOG] (v5, round {SAMPLING_ROUND}) 이미 존재")

display(spark.table(VERSION_TRACKING_TABLE).orderBy("round_number"))

# COMMAND ----------

# DBTITLE 1,10) Export to voc_llm_topic_classification
# ============================================================
# 10) 최종 결과 적재: voc_llm_topic_classification
#     - 해당 회차 결과를 data_created_dt 추가하여 적재
#     - 대상 테이블: sandbox.t_online_voc_analysis.voc_llm_topic_classification
#     - 적재 방식: append (회차별 누적)
# ============================================================

print(f"[STEP] 회차 {SAMPLING_ROUND} 결과 → {FINAL_OUTPUT_TABLE} 적재")
print(f"  소스: {DETAIL_TABLE}")

# 1. 현재 회차 DETAIL_TABLE 로드 (catalog 조인 완료된 상태)
if not spark.catalog.tableExists(DETAIL_TABLE):
    raise RuntimeError(f"[ERROR] {DETAIL_TABLE} 존재하지 않음. Cell 7 실행 후 진행하세요.")

round_df = spark.table(DETAIL_TABLE)
print(f"  회차 데이터: {round_df.count():,}건")

# 2. data_created_dt 컨럼 추가 (DATE 타입)
output_df = round_df.withColumn("data_created_dt", F.current_date())

# 3. 최종 적재 컨럼 선택 (memo_summary 제외)
final_columns = [
    "cate_1_depth", "cate_2_depth", "sc_measurement",
    "topic", "_row_id", "memo_id", "memo", "description",
    "topic_rev", "main_topic", "llm_round",
    "data_created_dt"
]
available_cols = [c for c in final_columns if c in output_df.columns]
output_df = output_df.select(*available_cols)

print(f"  적재 컨럼: {available_cols}")
print(f"  적재 건수: {output_df.count():,}")

# 4. 테이블 존재 여부에 따라 merge / create
if spark.catalog.tableExists(FINAL_OUTPUT_TABLE):
    existing_df = spark.table(FINAL_OUTPUT_TABLE)
    existing_count = existing_df.count()
    print(f"  기존 테이블: {existing_count:,}건")

    if "memo_id" not in existing_df.columns:
        spark.sql(f"ALTER TABLE {FINAL_OUTPUT_TABLE} ADD COLUMN memo_id STRING")
        existing_df = spark.table(FINAL_OUTPUT_TABLE)
        print("  [컬럼 추가] memo_id")

    merge_keys = ["memo_id"] if "memo_id" in existing_df.columns else ["memo", "cate_1_depth", "cate_2_depth", "sc_measurement"]
    output_df = dedupe_merge_source(output_df, merge_keys)
    merge_count = output_df.count()
    print(f"  [ROUND{SAMPLING_ROUND} 반영 대상] {merge_count:,}건")

    merge_table_by_keys(output_df, FINAL_OUTPUT_TABLE, merge_keys)
    final_count = spark.table(FINAL_OUTPUT_TABLE).count()
    print(f"  [MERGE] {existing_count:,} → {final_count:,} ({final_count - existing_count:+,})")
else:
    output_df.write.mode("overwrite").format("delta").saveAsTable(FINAL_OUTPUT_TABLE)
    print(f"  [CREATE] {FINAL_OUTPUT_TABLE} 생성 ({output_df.count():,}건)")

print(f"\n✅ 적재 완료: {FINAL_OUTPUT_TABLE}")
print(f"   data_created_dt = current_date()")
print(f"   llm_round = {ROUND_LABEL}")

# 검증
display(
    spark.table(FINAL_OUTPUT_TABLE)
    .groupBy("llm_round")
    .agg(F.count("*").alias("row_count"), F.min("data_created_dt").alias("min_dt"), F.max("data_created_dt").alias("max_dt"))
    .orderBy("llm_round")
)

post_run_status_df = build_llm_round_status_snapshot(
    source_table=SOURCE_TABLE,
    source_filter_sql=_source_filter_sql(TARGET_CATE_1),
    classified_table=FINAL_OUTPUT_TABLE,
)
print_llm_round_status_snapshot(
    title=f"LLM ROUND STATUS AFTER RUN | source={SOURCE_TABLE} | classified={FINAL_OUTPUT_TABLE}",
    status_df=post_run_status_df,
)
