# Databricks notebook source
# =========================
# Widgets & Config
# =========================
dbutils.widgets.text("snapshot_quarter", "2026Q1")
dbutils.widgets.text("prompt_version", "v2.0.0")           # 프롬프트/로직 변경 시 갱신
dbutils.widgets.text("endpoint", "databricks-gpt-5-2")     # Databricks LLM 엔드포인트
dbutils.widgets.text("test_mode", "false")
dbutils.widgets.text("test_max_groups", "60")
dbutils.widgets.text("rate_limit_seconds", "0.6")
dbutils.widgets.text("max_tokens", "1800")
dbutils.widgets.text("run_scope", "PROD")                  # TEST | PROD

# 드라이버 임계값 파라미터 (요청 반영)
dbutils.widgets.text("pval_max", "0.10")                   # p_value < pval_max
dbutils.widgets.text("abs_coef_min", "0.00")               # |coef| >= abs_coef_min
dbutils.widgets.text("abs_coef_max", "0.07")               # |coef| <= abs_coef_max

# 처리 대상 group_dim 제한(선택) - 콤마로 나열, 빈값이면 전부
dbutils.widgets.text("group_dims_to_run", "all, brand_name, country, d_type, year")

SNAPSHOT_QUARTER   = dbutils.widgets.get("snapshot_quarter")
PROMPT_VERSION     = dbutils.widgets.get("prompt_version")
ENDPOINT           = dbutils.widgets.get("endpoint")
TEST_MODE          = dbutils.widgets.get("test_mode").lower() == "true"
TEST_MAX_GROUPS    = int(dbutils.widgets.get("test_max_groups"))
RATE_LIMIT_SECONDS = float(dbutils.widgets.get("rate_limit_seconds"))
MAX_TOKENS         = int(dbutils.widgets.get("max_tokens"))
RUN_SCOPE          = dbutils.widgets.get("run_scope").upper()

PVAL_MAX      = float(dbutils.widgets.get("pval_max"))
ABS_COEF_MIN  = float(dbutils.widgets.get("abs_coef_min"))
ABS_COEF_MAX  = float(dbutils.widgets.get("abs_coef_max"))

GDIMS_TO_RUN  = [g.strip() for g in dbutils.widgets.get("group_dims_to_run").split(",") if g.strip()]

print(f"[CONFIG] quarter={SNAPSHOT_QUARTER}, prompt_version={PROMPT_VERSION}, endpoint={ENDPOINT}, run_scope={RUN_SCOPE}")
print(f"[CONFIG] thresholds: pval<{PVAL_MAX}, |coef|∈[{ABS_COEF_MIN},{ABS_COEF_MAX}]")
print(f"[CONFIG] dims: {GDIMS_TO_RUN}, test_mode={TEST_MODE} (limit={TEST_MAX_GROUPS})")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window as W
from pyspark.sql.types import *
from datetime import datetime
import time, json, traceback

model_summary_sdf_org = spark.table("sandbox.z_jungryo_lee.voc_wls_model_summary")
coef_summary_sdf_org  = spark.table("sandbox.z_jungryo_lee.voc_wls_coef_summary")

# 선택한 group_dim만 대상으로 제한
if GDIMS_TO_RUN:
    model_summary_sdf = model_summary_sdf_org.filter(F.col("group_dim").isin(GDIMS_TO_RUN))
    coef_summary_sdf  = coef_summary_sdf_org.filter(F.col("group_dim").isin(GDIMS_TO_RUN))

print("[INFO] model_summary rows:", model_summary_sdf.count())
print("[INFO] coef_summary rows :", coef_summary_sdf.count())

# COMMAND ----------

drivers_sdf = (
    coef_summary_sdf
      .withColumn("abs_coef", F.abs(F.col("coef")))
      .filter( (F.col("p_value") < F.lit(PVAL_MAX)) &
               (F.col("abs_coef") >= F.lit(ABS_COEF_MIN)) &
               (F.col("abs_coef") <= F.lit(ABS_COEF_MAX)) )
)

w_abs_desc = W.partitionBy("group_dim","group_key","y_feature").orderBy(F.col("abs_coef").desc())
drivers_ranked_sdf = drivers_sdf.withColumn("rank_abs", F.dense_rank().over(w_abs_desc)).cache()

# COMMAND ----------

# 모델 요약 집계
ms_agg_sdf = (
    model_summary_sdf
      .groupBy("group_dim","y_feature")
      .agg(F.collect_list(F.struct(
            F.col("group_key"),
            F.col("r_squared"),
            F.col("adj_r_squared"),
            F.col("f_statistic"),
            F.col("prob_f").alias("model_p_value"),
            F.col("log_likelihood"),
            F.col("aic"),
            F.col("bic"),
            F.col("y_obs"),
            F.col("cond_no")
      )).alias("model_stats"))
)

# 드라이버 전량 집계
drivers_agg_sdf = (
    drivers_ranked_sdf
      .select("group_dim","y_feature","group_key","x_feature","coef","p_value","t_value","x_obs","abs_coef")
      .groupBy("group_dim","y_feature")
      .agg(F.collect_list(F.struct(
            F.col("group_key"),
            F.col("x_feature"),
            F.col("coef"),
            F.col("p_value"),
            F.col("t_value"),
            F.col("x_obs"),
            F.col("abs_coef")
      )).alias("drivers_filtered"))
)

payload_base_sdf = ms_agg_sdf.join(drivers_agg_sdf, ["group_dim","y_feature"], "left").cache()

total_groups = payload_base_sdf.count()
print("[INFO] payload base groups:", total_groups)

if TEST_MODE and total_groups > TEST_MAX_GROUPS:
    payload_base_sdf = payload_base_sdf.limit(TEST_MAX_GROUPS).cache()
    print(f"[INFO] TEST_MODE → limited to {TEST_MAX_GROUPS}")

# COMMAND ----------

# 원본 계수(필터 전)도 비교할 수 있게 준비(원하면 필터 전/후 선택 가능)
coef_for_compare = (
    coef_summary_sdf
    .select("group_dim","group_key","y_feature","x_feature","coef")
    .cache()
)

# x_feature별 group_key 차이 통계
spread_stats_sdf = (
    coef_for_compare
    .groupBy("group_dim","y_feature","x_feature")
    .agg(
        F.min("coef").alias("coef_min"),
        F.max("coef").alias("coef_max"),
        F.avg("coef").alias("coef_mean"),
        F.stddev("coef").alias("coef_std")
    )
    .withColumn("coef_spread", F.col("coef_max") - F.col("coef_min"))
)

# group_key별(원본) coef 목록까지 함께 제공 (LLM이 비교에 활용)
by_key_list_sdf = (
    coef_for_compare
    .groupBy("group_dim","y_feature","x_feature")
    .agg(F.collect_list(F.struct(F.col("group_key"),F.col("coef"))).alias("by_key_coef_list"))
)

# 조합
comparative_agg_sdf = (
    spread_stats_sdf
    .join(by_key_list_sdf, ["group_dim","y_feature","x_feature"], "left")
    .groupBy("group_dim","y_feature")
    .agg(F.collect_list(F.struct(
        F.col("x_feature"),
        F.col("coef_min"),
        F.col("coef_max"),
        F.col("coef_mean"),
        F.col("coef_std"),
        F.col("coef_spread"),
        F.col("by_key_coef_list")
    )).alias("x_feature_diff_stats"))
)

# COMMAND ----------

# y 별 모델 요약(ALL만)
all_ms = (
    model_summary_sdf
    .filter(F.col("group_dim")=="all")
    .groupBy("y_feature")
    .agg(
        F.first("r_squared").alias("r_squared"),
        F.first("adj_r_squared").alias("adj_r_squared"),
        F.first("f_statistic").alias("f_stat"),
        F.first("prob_f").alias("model_p_value"),
        F.first("y_obs").alias("review_cnt"),
    )
)

# y 별 유의 드라이버 리스트(필터 적용 후)
all_drivers = (
    drivers_ranked_sdf
    .filter(F.col("group_dim")=="all")
    .groupBy("y_feature")
    .agg(F.collect_list(F.struct("x_feature","coef","abs_coef")).alias("drivers"))
)

# y 비교용 페이로드: y_strength, driver_freq 등
y_strength_sdf = (
    all_ms.join(all_drivers, ["y_feature"], "left")
    .withColumn("n_drivers", F.size(F.col("drivers")))
    .orderBy(F.desc("adj_r_squared"))
)

# 드라이버 빈도(어떤 x가 여러 y에서 반복적으로 중요?)
driver_freq_sdf = (
    drivers_ranked_sdf
    .filter(F.col("group_dim")=="all")
    .groupBy("x_feature")
    .agg(F.countDistinct("y_feature").alias("appear_y_cnt"),
         F.avg("abs_coef").alias("avg_abs_coef"))
    .orderBy(F.desc("appear_y_cnt"), F.desc("avg_abs_coef"))
)

# collect for Python payload build (ALL 집계는 하나의 특별 레코드로 생성)
y_strength_rows = y_strength_sdf.collect()
driver_freq_rows = driver_freq_sdf.collect()

# COMMAND ----------

SYSTEM_PROMPT_OLD  = """You are a senior product-planning analyst for TV products.
Return ONLY strict JSON. No preface, no markdown.
When group_dim='all', also compare across y features (model strength, repeated drivers).
When group_dim in {'brand','country','d_type'}, focus on comparative differences across group_keys, strongest/weakest, and negative signals.
Use concise, decision-oriented language in Korean.
"""

SYSTEM_PROMPT = """
You are a senior product-planning analyst for TV products.
Return ONLY strict JSON. No preface, no markdown.

General rules:
- Use concise, decision-oriented Korean.
- Base all interpretations on coefficients, R², and review volume.
- Avoid abstract expressions (e.g., '이미지 개선 필요').
- Write short paragraphs with clear line breaks where needed.

Comparative insight rules (applies to ALL group_dim values):
- Always compare across group_keys within the selected group_dim.
- Identify:
  • Strongest vs weakest group_keys
  • Drivers unique to specific group_keys
  • Drivers repeatedly important across multiple group_keys
  • Negative or underperforming signals

Formatting rules:
- Use sentence-level line breaks to improve readability.
- Each insight paragraph should focus on ONE idea.

Case handling:
- When group_dim='all':
  • Additionally compare across y_features (model strength, repeated drivers).
"""

LLM_SCHEMA_JSON_OLD = """
{
  "headline": "One-line summary",
  "summary": "2~4 sentence product-planning insight",
  "top_takeaways": ["key point 1", "key point 2", "key point 3"],
  "risk_signal": "risk or warning interpretation",
  "recommended_actions": ["action 1", "action 2", "action 3"],
  "top_positive_drivers": [{"group_key": "LG", "x_feature": "UI Navigation", "coef": 0.82}],
  "top_negative_drivers": [{"group_key": "KR", "x_feature": "Remote Control", "coef": -0.41}],
  "model_comment": "interpretation of R² and review counts"
}
""".strip()

LLM_SCHEMA_JSON = """
{
  "headline": "One-line summary",
  "summary": "2~4 sentence product-planning insight",
  "top_takeaways": ["key point 1", "key point 2", "key point 3"],
  "risk_signal": "risk or warning interpretation",
  "recommended_actions": ["action 1", "action 2", "action 3"],
  "top_positive_drivers": [
    {"group_key": "A", "x_feature": "UI Navigation", "coef": 0.82}
  ],
  "top_negative_drivers": [
    {"group_key": "B", "x_feature": "Remote Control", "coef": -0.41}
  ],
  "model_comment": "interpretation of R² and review counts",

  "comparative_insight": {
    "group_strength_overview": [
      {
        "group_key": "A",
        "model_strength": "high | mid | low",
        "interpretation": "해당 그룹의 전반적 만족도 설명 특성 요약"
      }
    ],
    "differentiating_drivers": [
      {
        "group_key": "A",
        "x_feature": "UI Navigation",
        "coef": 0.82,
        "interpretation": "다른 그룹 대비 차별적으로 작용하는 요인"
      }
    ],
    "common_key_drivers": [
      {
        "x_feature": "Picture Quality",
        "interpretation": "다수 그룹에서 반복적으로 중요하게 작용"
      }
    ],
    "underperforming_groups": [
      {
        "group_key": "B",
        "x_feature": "Remote Control",
        "coef": -0.41,
        "interpretation": "상대적 약점 또는 리스크 신호"
      }
    ]
  }
}
"""
def case_hint(group_dim: str) -> str:
    if group_dim == "all":
        return "overall + cross-y comparison"
    return "comparative across group_keys"

def build_user_prompt(group_dim: str, thresholds: dict):
  
    return f"""
You will receive a JSON payload and must return ONLY strict JSON
following the schema below.

Case:
- group_dim: {group_dim}
- Focus on comparative insights across group_keys.

Comparative Insight Instructions:
- Always compare group_keys within the given group_dim.
- Explicitly state:
  • Which group_keys are stronger or weaker
  • Which drivers are unique vs common
  • Where negative or underperforming signals appear
- Write insights in short paragraphs with clear line breaks.
- Avoid long compound sentences.

Thresholds (for reference only):
{json.dumps(thresholds, ensure_ascii=False)}

Output JSON schema (strict):
{LLM_SCHEMA_JSON}
""".strip()

#     return f"""
# You will receive a JSON payload and must return ONLY strict JSON in the schema below.
# Case: {case_hint(group_dim)}

# Thresholds (for your reference):
# {json.dumps(thresholds, ensure_ascii=False)}

# Output JSON schema (strict):
# {LLM_SCHEMA_JSON}
# """.strip()

MAX_RETRIES   = 4
BACKOFF_BASE  = 1.8

def _rows_to_pylist(lst):
    if not lst: return []
    return [dict(x.asDict()) if hasattr(x, "asDict") else dict(x) for x in lst]

def call_ai_query(endpoint: str, request_text: str, temperature: float = 0.0, max_tokens: int = 1800):
    df = spark.createDataFrame([(request_text,)], ["p"])
    out = (
        df.select(
            F.expr(f"""
                ai_query(
                    endpoint => '{endpoint}',
                    request  => p,
                    modelParameters => named_struct(
                        'temperature', {float(temperature)},
                        'max_tokens',  {int(max_tokens)}
                    )
                )
            """).alias("json_out")
        ).first()
    )
    return out["json_out"] if out else None

def call_ai_query_with_retry(endpoint: str, system_prompt: str, user_prompt: str, payload_json: str):
    last_err = None
    for attempt in range(1, MAX_RETRIES+1):
        try:
            req = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}\n\n[PAYLOAD]\n{payload_json}"
            raw = call_ai_query(endpoint, req, temperature=0.0, max_tokens=MAX_TOKENS)
            if not raw:
                raise RuntimeError("Empty response from ai_query")

            clean = raw.replace("```json","").replace("```","").strip()
            data = json.loads(clean)  # strict
            time.sleep(RATE_LIMIT_SECONDS)  # rate limit
            return data, clean
        except Exception as e:
            wait = BACKOFF_BASE ** (attempt - 1)
            print(f"[WARN] ai_query failed (attempt {attempt}/{MAX_RETRIES}): {repr(e)} → wait {wait:.1f}s")
            time.sleep(wait)
            last_err = e
    raise RuntimeError(f"ai_query failed after {MAX_RETRIES} retries: {repr(last_err)}")

# COMMAND ----------

# (group_dim, y_feature) 단위 기본 집계 로우
base_rows = payload_base_sdf.collect()
print(f"[INFO] generation targets: {len(base_rows)}")

results = []
failed  = []

thresholds_meta = {"p_value_max": PVAL_MAX, "abs_coef_min": ABS_COEF_MIN, "abs_coef_max": ABS_COEF_MAX}

# 비교 통계 조인 준비(필요한 조합만 dict로 빠르게 접근)
comp_rows = { (r["group_dim"], r["y_feature"]): r for r in comparative_agg_sdf.collect() }

def sort_drivers(drivers):
    if not drivers: return []
    return sorted(drivers, key=lambda d: abs(d.get("coef", 0.0)), reverse=True)

for i, r in enumerate(base_rows, 1):
    try:
        group_dim = r["group_dim"]
        y_feature = r["y_feature"]
        model_stats = _rows_to_pylist(r["model_stats"]) if r["model_stats"] else []
        drivers_filtered = sort_drivers(_rows_to_pylist(r["drivers_filtered"]) if r["drivers_filtered"] else [])

        payload = {
            "meta": {
                "snapshot_quarter": SNAPSHOT_QUARTER,
                "prompt_version": PROMPT_VERSION,
                "endpoint": ENDPOINT,
                "run_scope": RUN_SCOPE,
                "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "driver_thresholds": thresholds_meta,
                "case": "overall" if group_dim == "all" else "comparative"
            },
            "context": {"group_dim": group_dim, "y_feature": y_feature},
            "model_summary": model_stats,
            "drivers_filtered": drivers_filtered
        }

        # 비교 통계(brand/country/d_type)
        if group_dim in {"brand","country","d_type"}:
            comp = comp_rows.get((group_dim, y_feature))
            if comp:
                payload["x_feature_diff_stats"] = _rows_to_pylist(comp["x_feature_diff_stats"])
            else:
                payload["x_feature_diff_stats"] = []

        # ---- LLM 호출
        user_prompt = build_user_prompt(group_dim, thresholds_meta)
        payload_str = json.dumps(payload, ensure_ascii=False)
        llm_json, raw = call_ai_query_with_retry(
            endpoint=ENDPOINT,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            payload_json=payload_str
        )

        results.append({
            "snapshot_quarter": SNAPSHOT_QUARTER,
            "prompt_version": PROMPT_VERSION,
            "group_dim": group_dim,
            "y_feature": y_feature,
            "headline": llm_json.get("headline",""),
            "summary": llm_json.get("summary",""),
            "top_takeaways_json": json.dumps(llm_json.get("top_takeaways",[]), ensure_ascii=False),
            "risk_signal": llm_json.get("risk_signal",""),
            "recommended_actions_json": json.dumps(llm_json.get("recommended_actions",[]), ensure_ascii=False),
            "top_positive_drivers_json": json.dumps(llm_json.get("top_positive_drivers",[]), ensure_ascii=False),
            "top_negative_drivers_json": json.dumps(llm_json.get("top_negative_drivers",[]), ensure_ascii=False),
            "model_comment": llm_json.get("model_comment",""),
            "payload_json": payload_str,
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        })

        if i % 5 == 0 or i == len(base_rows):
            print(f"[INFO] progress {i}/{len(base_rows)}")

    except Exception as e:
        print(f"[ERROR] ({r['group_dim']}, {r['y_feature']}) failed: {repr(e)}")
        traceback.print_exc()
        failed.append({
            "snapshot_quarter": SNAPSHOT_QUARTER,
            "prompt_version": PROMPT_VERSION,
            "group_dim": r["group_dim"],
            "y_feature": r["y_feature"],
            "error": repr(e),
            "payload_json": json.dumps({
                "context": {"group_dim": r["group_dim"], "y_feature": r["y_feature"]}
            }, ensure_ascii=False),
            "failed_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        })

print(f"[INFO] individual results: success={len(results)}, failed={len(failed)}")

# COMMAND ----------

# group_dim='all' 전용: y간 비교용 특별 레코드 생성 (y_feature='_ALL_')
try:
    y_strength_py = [dict(r.asDict()) for r in y_strength_rows] if y_strength_rows else []
    driver_freq_py = [dict(r.asDict()) for r in driver_freq_rows] if driver_freq_rows else []

    all_payload = {
        "meta": {
            "snapshot_quarter": SNAPSHOT_QUARTER,
            "prompt_version": PROMPT_VERSION,
            "endpoint": ENDPOINT,
            "run_scope": RUN_SCOPE,
            "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "driver_thresholds": thresholds_meta,
            "case": "overall_y_comparison"
        },
        "context": {"group_dim": "all", "y_feature": "_ALL_"},
        "y_strength": y_strength_py,               # 각 y의 adj_r2, y_obs, review_cnt, 드라이버 수
        "driver_reuse_stats": driver_freq_py       # 여러 y에서 반복 등장하는 x의 빈도/평균 효과
    }

    user_prompt_all = f"""
You will receive a JSON payload that summarizes multiple y features under group_dim='all'.
Return ONLY strict JSON in the schema below. Provide cross-y comparisons (which y has stronger model strength, common drivers, gaps).
{LLM_SCHEMA_JSON}
"""
    all_payload_str = json.dumps(all_payload, ensure_ascii=False)
    llm_json_all, raw_all = call_ai_query_with_retry(
        endpoint=ENDPOINT,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt_all,
        payload_json=all_payload_str
    )

    results.append({
        "snapshot_quarter": SNAPSHOT_QUARTER,
        "prompt_version": PROMPT_VERSION,
        "group_dim": "all",
        "y_feature": "_ALL_",
        "headline": llm_json_all.get("headline",""),
        "summary": llm_json_all.get("summary",""),
        "top_takeaways_json": json.dumps(llm_json_all.get("top_takeaways",[]), ensure_ascii=False),
        "risk_signal": llm_json_all.get("risk_signal",""),
        "recommended_actions_json": json.dumps(llm_json_all.get("recommended_actions",[]), ensure_ascii=False),
        "top_positive_drivers_json": json.dumps(llm_json_all.get("top_positive_drivers",[]), ensure_ascii=False),
        "top_negative_drivers_json": json.dumps(llm_json_all.get("top_negative_drivers",[]), ensure_ascii=False),
        "model_comment": llm_json_all.get("model_comment",""),
        "payload_json": all_payload_str,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    })
    print("[INFO] appended ALL-y comparison record.")

except Exception as e:
    print("[WARN] ALL-y comparison generation failed:", repr(e))

# COMMAND ----------

import json

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# COMMAND ----------

import json

# 1. 파일 열기 (읽기 모드 'r', 인코딩 'utf-8')
with open("results.json", "r", encoding="utf-8") as f:
    # 2. json.load를 사용하여 파일 내용을 파이썬 객체(리스트)로 변환
    loaded_results = json.load(f)

# 결과 확인
print(type(loaded_results)) # <class 'list'> 출력됨
print(loaded_results)

# COMMAND ----------

# 타겟/실패 테이블 생성
spark.sql(
  """
  CREATE TABLE IF NOT EXISTS sandbox.z_jungryo_lee.voc_wls_dashboard_insight (
    snapshot_quarter STRING,
    prompt_version STRING,
    group_dim STRING,
    y_feature STRING,
    headline STRING,
    summary STRING,
    top_takeaways_json STRING,
    risk_signal STRING,
    recommended_actions_json STRING,
    top_positive_drivers_json STRING,
    top_negative_drivers_json STRING,
    model_comment STRING,
    payload_json STRING,
    generated_at TIMESTAMP
  ) USING delta
"""
)

spark.sql(
    """
CREATE TABLE IF NOT EXISTS sandbox.z_jungryo_lee.voc_wls_dashboard_insight_failed (
  snapshot_quarter STRING,
  prompt_version STRING,
  group_dim STRING,
  y_feature STRING,
  error STRING,
  payload_json STRING,
  failed_at TIMESTAMP
) USING delta
""")

from pyspark.sql import Row

if results:
    out_sdf = spark.createDataFrame([Row(**rec) for rec in results])
    out_sdf.createOrReplaceTempView("insight_updates")

    spark.sql("""
    MERGE INTO sandbox.z_jungryo_lee.voc_wls_dashboard_insight AS t
    USING insight_updates AS s
    ON  t.snapshot_quarter = s.snapshot_quarter
    AND t.prompt_version   = s.prompt_version
    AND t.group_dim        = s.group_dim
    AND t.y_feature        = s.y_feature
    WHEN MATCHED THEN UPDATE SET
      t.headline                  = s.headline,
      t.summary                   = s.summary,
      t.top_takeaways_json        = s.top_takeaways_json,
      t.risk_signal               = s.risk_signal,
      t.recommended_actions_json  = s.recommended_actions_json,
      t.top_positive_drivers_json = s.top_positive_drivers_json,
      t.top_negative_drivers_json = s.top_negative_drivers_json,
      t.model_comment             = s.model_comment,
      t.payload_json              = s.payload_json,
      t.generated_at              = s.generated_at
    WHEN NOT MATCHED THEN INSERT *
    """)
    print(f"[DONE] MERGE upserted: {out_sdf.count()} rows")

if failed:
    failed_sdf = spark.createDataFrame([Row(**rec) for rec in failed])
    failed_sdf.write.mode("append").saveAsTable("sandbox.z_jungryo_lee.voc_wls_dashboard_insight_failed")
    print(f"[WARN] failed logged:", failed_sdf.count())

# COMMAND ----------


