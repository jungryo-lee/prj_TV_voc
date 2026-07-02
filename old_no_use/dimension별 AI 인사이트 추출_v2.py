# Databricks notebook source
# Databricks notebook source
# =========================
# Widgets & Config
# =========================
dbutils.widgets.text("snapshot_quarter", "2026Q1")
dbutils.widgets.text("prompt_version", "v3.1.0")
dbutils.widgets.text("endpoint", "databricks-gpt-5-2")
dbutils.widgets.text("test_mode", "false")
dbutils.widgets.text("test_max_groups", "60")
dbutils.widgets.text("rate_limit_seconds", "0.6")
dbutils.widgets.text("max_tokens", "1800")
dbutils.widgets.text("run_scope", "PROD")

dbutils.widgets.text("pval_max", "0.10")
dbutils.widgets.text("abs_coef_min", "0.00")
dbutils.widgets.text("abs_coef_max", "0.07")

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

THRESHOLDS_META = {
    "p_value_max": PVAL_MAX,
    "abs_coef_min": ABS_COEF_MIN,
    "abs_coef_max": ABS_COEF_MAX
}

print(f"[CONFIG] quarter={SNAPSHOT_QUARTER}, prompt_version={PROMPT_VERSION}, endpoint={ENDPOINT}, run_scope={RUN_SCOPE}")
print(f"[CONFIG] thresholds: pval<{PVAL_MAX}, |coef|∈[{ABS_COEF_MIN},{ABS_COEF_MAX}]")
print(f"[CONFIG] dims: {GDIMS_TO_RUN}, test_mode={TEST_MODE} (limit={TEST_MAX_GROUPS})")


# COMMAND ----------

# COMMAND ----------
from pyspark.sql import functions as F
from pyspark.sql import Window as W
from pyspark.sql import Row
from datetime import datetime
import time
import json
import traceback

MODEL_SUMMARY_TABLE = "sandbox.z_jungryo_lee.voc_wls_model_summary"
COEF_SUMMARY_TABLE  = "sandbox.z_jungryo_lee.voc_wls_coef_summary"

TARGET_TABLE = "sandbox.z_jungryo_lee.voc_wls_dashboard_insight_v3"
FAILED_TABLE = "sandbox.z_jungryo_lee.voc_wls_dashboard_insight_failed_v3"

MAX_RETRIES  = 4
BACKOFF_BASE = 1.8

# COMMAND ----------

# COMMAND ----------
SYSTEM_PROMPT = """
You are a senior product-planning analyst for TV products.
Return ONLY strict JSON. No preface, no markdown.

Language and style rules:
- Write all outputs in Korean.
- Every sentence must use a formal business tone ending in '~습니다', '~입니다', or equivalent formal style.
- Do not use casual, conversational, or vague wording.
- Keep each sentence concise and decision-oriented.

Analysis rules:
- Base all interpretations only on coefficients, p-values, R², adjusted R², and review volume.
- Always compare group_keys within the given group_dim.
- Focus only on the following:
1. headline
2. model confidence reason
3. overall summary
4. one-line insight for each group_key
5. key drivers for each group_key
6. distinctive points versus other group_keys
7. strongest/weakest group_keys
8. common and unique drivers

Important:
- model_confidence.level will be calculated outside the model.
- You must still write model_confidence.reason based on adjusted R², model p-value, and review volume.

Do not produce:
- generic recommendations
- filler text
- abstract branding language
- markdown
""".strip()

LLM_SCHEMA_JSON = """
{
"headline": "한 줄 핵심 요약",
"model_confidence": {
    "reason": "R², p-value, review volume 기반 신뢰도 설명"
},
"overall_summary": "이 y_feature에 대한 전체 비교 요약",
"group_key_insights": [
    {
    "group_key": "A",
    "one_line_insight": "해당 group_key 핵심 해석 한 줄",
    "key_drivers": [
        {
        "x_feature": "UI Navigation",
        "coef": 0.052,
        "direction": "positive",
        "comment": "핵심 변수 해석"
        }
    ],
    "distinctive_points": [
        "다른 group_key 대비 상대적으로 두드러지는 차이점입니다."
    ]
    }
],
"cross_group_comparison": {
    "strongest_group_keys": ["A"],
    "weakest_group_keys": ["B"],
    "common_drivers": [
    {
        "x_feature": "Picture Quality",
        "comment": "여러 group_key에서 반복적으로 중요하게 작용합니다."
    }
    ],
    "unique_drivers": [
    {
        "group_key": "A",
        "x_feature": "UI Navigation",
        "comment": "해당 group_key에서만 상대적으로 두드러집니다."
    }
    ]
}
}
""".strip()

def build_user_prompt(group_dim: str, y_feature: str, thresholds: dict) -> str:
    return f"""
You will receive a JSON payload for one y_feature.

Context:
- group_dim: {group_dim}
- y_feature: {y_feature}

Required output:
- headline: one-line summary
- model_confidence:
- reason only
- overall_summary: short comparison summary
- group_key_insights:
- one_line_insight
- key_drivers
- distinctive_points
- cross_group_comparison:
- strongest_group_keys
- weakest_group_keys
- common_drivers
- unique_drivers

Important writing constraint:
- Every Korean sentence must be in formal business tone such as '~습니다' or '~입니다'.
- Do not mix styles.

Important logic constraint:
- Do not output model_confidence.level.
- model_confidence.reason must explain confidence only from adjusted R², model p-value, and review volume.

Thresholds:
{json.dumps(thresholds, ensure_ascii=False)}

Return ONLY strict JSON following this schema:
{LLM_SCHEMA_JSON}
""".strip()

# COMMAND ----------

# COMMAND ----------
def rows_to_pylist(lst):
    if not lst:
        return []
    return [dict(x.asDict()) if hasattr(x, "asDict") else dict(x) for x in lst]

def sort_drivers(drivers):
    if not drivers:
        return []
    return sorted(drivers, key=lambda d: abs(d.get("coef", 0.0)), reverse=True)

def json_dumps_safe(obj):
    return json.dumps(obj, ensure_ascii=False)

def safe_float(v, default=None):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default

def safe_int(v, default=None):
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default

# COMMAND ----------
def compute_confidence_level(model_stats: list) -> dict:
    """
    group_dim + y_feature 단위에 포함된 여러 group_key의 모델 통계를 보고
    보수적으로 최악값 기준으로 confidence를 계산합니다.
    """
    if not model_stats:
        return {
            "level": "low",
            "score_basis": {
                "min_adj_r_squared": None,
                "max_model_p_value": None,
                "min_y_obs": None
            }
        }

    adj_r2_list = [safe_float(x.get("adj_r_squared")) for x in model_stats if safe_float(x.get("adj_r_squared")) is not None]
    pval_list   = [safe_float(x.get("model_p_value")) for x in model_stats if safe_float(x.get("model_p_value")) is not None]
    yobs_list   = [safe_int(x.get("y_obs")) for x in model_stats if safe_int(x.get("y_obs")) is not None]

    min_adj_r2 = min(adj_r2_list) if adj_r2_list else None
    max_pval   = max(pval_list) if pval_list else None
    min_y_obs  = min(yobs_list) if yobs_list else None

    if min_adj_r2 is None or max_pval is None or min_y_obs is None:
        level = "low"
    elif min_adj_r2 >= 0.60 and max_pval <= 0.05 and min_y_obs >= 30:
        level = "high"
    elif min_adj_r2 >= 0.30 and max_pval <= 0.10 and min_y_obs >= 10:
        level = "mid"
    else:
        level = "low"

    return {
        "level": level,
        "score_basis": {
            "min_adj_r_squared": min_adj_r2,
            "max_model_p_value": max_pval,
            "min_y_obs": min_y_obs
        }
    }

# COMMAND ----------

# COMMAND ----------
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
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}\n\n[PAYLOAD]\n{payload_json}"
            raw = call_ai_query(endpoint, req, temperature=0.0, max_tokens=MAX_TOKENS)
            if not raw:
                raise RuntimeError("Empty response from ai_query")

            clean = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)
            time.sleep(RATE_LIMIT_SECONDS)
            return data, clean

        except Exception as e:
            wait = BACKOFF_BASE ** (attempt - 1)
            print(f"[WARN] ai_query failed (attempt {attempt}/{MAX_RETRIES}): {repr(e)} -> wait {wait:.1f}s")
            time.sleep(wait)
            last_err = e

    raise RuntimeError(f"ai_query failed after {MAX_RETRIES} retries: {repr(last_err)}")

# COMMAND ----------

# COMMAND ----------
model_summary_sdf_org = spark.table(MODEL_SUMMARY_TABLE)
coef_summary_sdf_org  = spark.table(COEF_SUMMARY_TABLE)

if GDIMS_TO_RUN:
    model_summary_sdf = model_summary_sdf_org.filter(F.col("group_dim").isin(GDIMS_TO_RUN))
    coef_summary_sdf  = coef_summary_sdf_org.filter(F.col("group_dim").isin(GDIMS_TO_RUN))
else:
    model_summary_sdf = model_summary_sdf_org
    coef_summary_sdf  = coef_summary_sdf_org

print("[INFO] model_summary rows:", model_summary_sdf.count())
print("[INFO] coef_summary rows :", coef_summary_sdf.count())

# COMMAND ----------
drivers_sdf = (
    coef_summary_sdf
    .withColumn("abs_coef", F.abs(F.col("coef")))
    .filter(
        (F.col("p_value") < F.lit(PVAL_MAX)) &
        (F.col("abs_coef") >= F.lit(ABS_COEF_MIN)) &
        (F.col("abs_coef") <= F.lit(ABS_COEF_MAX))
    )
)

w_abs_desc = W.partitionBy("group_dim", "group_key", "y_feature").orderBy(F.col("abs_coef").desc())

drivers_ranked_sdf = (
    drivers_sdf
    .withColumn("rank_abs", F.dense_rank().over(w_abs_desc))
    .cache()
)


# COMMAND ----------

# COMMAND ----------
ms_agg_sdf = (
    model_summary_sdf
    .groupBy("group_dim", "y_feature")
    .agg(
        F.collect_list(
            F.struct(
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
            )
        ).alias("model_stats")
    )
)

drivers_agg_sdf = (
    drivers_ranked_sdf
    .select(
        "group_dim", "y_feature", "group_key", "x_feature",
        "coef", "p_value", "t_value", "x_obs", "abs_coef"
    )
    .groupBy("group_dim", "y_feature")
    .agg(
        F.collect_list(
            F.struct(
                F.col("group_key"),
                F.col("x_feature"),
                F.col("coef"),
                F.col("p_value"),
                F.col("t_value"),
                F.col("x_obs"),
                F.col("abs_coef")
            )
        ).alias("drivers_filtered")
    )
)

payload_base_sdf = (
    ms_agg_sdf
    .join(drivers_agg_sdf, ["group_dim", "y_feature"], "left")
    .cache()
)

total_groups = payload_base_sdf.count()
print("[INFO] payload base groups:", total_groups)

if TEST_MODE and total_groups > TEST_MAX_GROUPS:
    payload_base_sdf = payload_base_sdf.limit(TEST_MAX_GROUPS).cache()
    print(f"[INFO] TEST_MODE -> limited to {TEST_MAX_GROUPS}")

# COMMAND ----------

# COMMAND ----------
coef_for_compare = (
    coef_summary_sdf
    .select("group_dim", "group_key", "y_feature", "x_feature", "coef")
    .cache()
)

spread_stats_sdf = (
    coef_for_compare
    .groupBy("group_dim", "y_feature", "x_feature")
    .agg(
        F.min("coef").alias("coef_min"),
        F.max("coef").alias("coef_max"),
        F.avg("coef").alias("coef_mean"),
        F.stddev("coef").alias("coef_std")
    )
    .withColumn("coef_spread", F.col("coef_max") - F.col("coef_min"))
)

by_key_list_sdf = (
    coef_for_compare
    .groupBy("group_dim", "y_feature", "x_feature")
    .agg(
        F.collect_list(
            F.struct(F.col("group_key"), F.col("coef"))
        ).alias("by_key_coef_list")
    )
)

comparative_agg_sdf = (
    spread_stats_sdf
    .join(by_key_list_sdf, ["group_dim", "y_feature", "x_feature"], "left")
    .groupBy("group_dim", "y_feature")
    .agg(
        F.collect_list(
            F.struct(
                F.col("x_feature"),
                F.col("coef_min"),
                F.col("coef_max"),
                F.col("coef_mean"),
                F.col("coef_std"),
                F.col("coef_spread"),
                F.col("by_key_coef_list")
            )
        ).alias("x_feature_diff_stats")
    )
)

comp_rows = {(r["group_dim"], r["y_feature"]): r for r in comparative_agg_sdf.collect()}

# COMMAND ----------

# COMMAND ----------
def build_payload(base_row, comp_rows_dict):
    group_dim = base_row["group_dim"]
    y_feature = base_row["y_feature"]

    model_summary = rows_to_pylist(base_row["model_stats"])
    confidence_meta = compute_confidence_level(model_summary)

    payload = {
        "meta": {
            "snapshot_quarter": SNAPSHOT_QUARTER,
            "prompt_version": PROMPT_VERSION,
            "endpoint": ENDPOINT,
            "run_scope": RUN_SCOPE,
            "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "driver_thresholds": THRESHOLDS_META,
            "confidence_rule_basis": confidence_meta["score_basis"]
        },
        "context": {
            "group_dim": group_dim,
            "y_feature": y_feature
        },
        "model_summary": model_summary,
        "drivers_filtered": sort_drivers(rows_to_pylist(base_row["drivers_filtered"]))
    }

    comp = comp_rows_dict.get((group_dim, y_feature))
    payload["x_feature_diff_stats"] = rows_to_pylist(comp["x_feature_diff_stats"]) if comp else []

    return payload

def llm_json_to_result_record(group_dim, y_feature, llm_json, payload_str, computed_confidence):
    model_confidence = llm_json.get("model_confidence", {}) or {}

    return {
        "snapshot_quarter": SNAPSHOT_QUARTER,
        "prompt_version": PROMPT_VERSION,
        "group_dim": group_dim,
        "y_feature": y_feature,
        "headline": llm_json.get("headline", ""),
        "overall_summary": llm_json.get("overall_summary", ""),
        "model_confidence_level": computed_confidence.get("level", "low"),
        "model_confidence_reason": model_confidence.get("reason", ""),
        "group_key_insights_json": json_dumps_safe(llm_json.get("group_key_insights", [])),
        "cross_group_comparison_json": json_dumps_safe(llm_json.get("cross_group_comparison", {})),
        "raw_llm_json": json_dumps_safe(llm_json),
        "payload_json": payload_str,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    }

# COMMAND ----------
base_rows = payload_base_sdf.collect()
print(f"[INFO] generation targets: {len(base_rows)}")

results = []
failed  = []

for i, r in enumerate(base_rows, 1):
    group_dim = r["group_dim"]
    y_feature = r["y_feature"]

    try:
        payload = build_payload(r, comp_rows)
        payload_str = json.dumps(payload, ensure_ascii=False)

        computed_confidence = compute_confidence_level(payload["model_summary"])

        user_prompt = build_user_prompt(
            group_dim=group_dim,
            y_feature=y_feature,
            thresholds=THRESHOLDS_META
        )

        llm_json, raw = call_ai_query_with_retry(
            endpoint=ENDPOINT,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            payload_json=payload_str
        )

        llm_json["model_confidence"] = llm_json.get("model_confidence", {}) or {}
        llm_json["model_confidence"]["level"] = computed_confidence["level"]

        results.append(
            llm_json_to_result_record(
                group_dim=group_dim,
                y_feature=y_feature,
                llm_json=llm_json,
                payload_str=payload_str,
                computed_confidence=computed_confidence
            )
        )

        if i % 5 == 0 or i == len(base_rows):
            print(f"[INFO] progress {i}/{len(base_rows)}")

    except Exception as e:
        print(f"[ERROR] ({group_dim}, {y_feature}) failed: {repr(e)}")
        traceback.print_exc()

        failed.append({
            "snapshot_quarter": SNAPSHOT_QUARTER,
            "prompt_version": PROMPT_VERSION,
            "group_dim": group_dim,
            "y_feature": y_feature,
            "error": repr(e),
            "payload_json": json.dumps(
                {"context": {"group_dim": group_dim, "y_feature": y_feature}},
                ensure_ascii=False
            ),
            "failed_at": datetime.utcnow()
        })

print(f"[INFO] individual results: success={len(results)}, failed={len(failed)}")

# COMMAND ----------
with open("results_v3.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("[INFO] saved local file: results_v3.json")

# COMMAND ----------
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
snapshot_quarter STRING,
prompt_version STRING,
group_dim STRING,
y_feature STRING,
headline STRING,
overall_summary STRING,
model_confidence_level STRING,
model_confidence_reason STRING,
group_key_insights_json STRING,
cross_group_comparison_json STRING,
raw_llm_json STRING,
payload_json STRING,
generated_at TIMESTAMP
) USING delta
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {FAILED_TABLE} (
snapshot_quarter STRING,
prompt_version STRING,
group_dim STRING,
y_feature STRING,
error STRING,
payload_json STRING,
failed_at TIMESTAMP
) USING delta
""")

#

# COMMAND ----------

# COMMAND ----------
if results:
    out_sdf = spark.createDataFrame([Row(**rec) for rec in results])
    out_sdf.createOrReplaceTempView("insight_updates_v3")

    spark.sql(f"""
        MERGE INTO {TARGET_TABLE} AS t
        USING insight_updates_v3 AS s
        ON  t.snapshot_quarter = s.snapshot_quarter
        AND t.prompt_version   = s.prompt_version
        AND t.group_dim        = s.group_dim
        AND t.y_feature        = s.y_feature
        WHEN MATCHED THEN UPDATE SET
        t.headline                    = s.headline,
        t.overall_summary             = s.overall_summary,
        t.model_confidence_level      = s.model_confidence_level,
        t.model_confidence_reason     = s.model_confidence_reason,
        t.group_key_insights_json     = s.group_key_insights_json,
        t.cross_group_comparison_json = s.cross_group_comparison_json,
        t.raw_llm_json                = s.raw_llm_json,
        t.payload_json                = s.payload_json,
        t.generated_at                = s.generated_at
        WHEN NOT MATCHED THEN INSERT (
        snapshot_quarter,
        prompt_version,
        group_dim,
        y_feature,
        headline,
        overall_summary,
        model_confidence_level,
        model_confidence_reason,
        group_key_insights_json,
        cross_group_comparison_json,
        raw_llm_json,
        payload_json,
        generated_at
        )
        VALUES (
        s.snapshot_quarter,
        s.prompt_version,
        s.group_dim,
        s.y_feature,
        s.headline,
        s.overall_summary,
        s.model_confidence_level,
        s.model_confidence_reason,
        s.group_key_insights_json,
        s.cross_group_comparison_json,
        s.raw_llm_json,
        s.payload_json,
        s.generated_at
        )
    """)

    print(f"[INFO] merged success rows: {out_sdf.count()}")

if failed:
    failed_sdf = spark.createDataFrame([Row(**rec) for rec in failed])
    failed_sdf.write.mode("append").saveAsTable(FAILED_TABLE)
    print(f"[WARN] failed logged: {failed_sdf.count()}")

# COMMAND ----------
display(
    spark.table(TARGET_TABLE)
        .filter(F.col("snapshot_quarter") == SNAPSHOT_QUARTER)
        .filter(F.col("prompt_version") == PROMPT_VERSION)
        .orderBy(F.desc("generated_at"))
)


# COMMAND ----------

# COMMAND ----------
retry_failed_sdf = (
    spark.table(FAILED_TABLE)
    .filter(F.col("snapshot_quarter") == SNAPSHOT_QUARTER)
    .filter(F.col("prompt_version") == PROMPT_VERSION)
    .select("snapshot_quarter", "prompt_version", "group_dim", "y_feature", "error", "failed_at")
    .dropDuplicates(["snapshot_quarter", "prompt_version", "group_dim", "y_feature"])
)

print(f"[INFO] retry targets: {retry_failed_sdf.count()}")
display(retry_failed_sdf.orderBy(F.desc("failed_at")))

# COMMAND ----------
retry_keys = {
    (r["group_dim"], r["y_feature"])
    for r in retry_failed_sdf.select("group_dim", "y_feature").collect()
}

base_rows_retry = [
    r for r in payload_base_sdf.collect()
    if (r["group_dim"], r["y_feature"]) in retry_keys
]

print(f"[INFO] retry base rows: {len(base_rows_retry)}")

# COMMAND ----------
retry_results = []
retry_failed = []

for i, r in enumerate(base_rows_retry, 1):
    group_dim = r["group_dim"]
    y_feature = r["y_feature"]

    try:
        payload = build_payload(r, comp_rows)
        payload_str = json.dumps(payload, ensure_ascii=False)

        computed_confidence = compute_confidence_level(payload["model_summary"])

        user_prompt = build_user_prompt(
            group_dim=group_dim,
            y_feature=y_feature,
            thresholds=THRESHOLDS_META
        )

        llm_json, raw = call_ai_query_with_retry(
            endpoint=ENDPOINT,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            payload_json=payload_str
        )

        llm_json["model_confidence"] = llm_json.get("model_confidence", {}) or {}
        llm_json["model_confidence"]["level"] = computed_confidence["level"]

        retry_results.append(
            llm_json_to_result_record(
                group_dim=group_dim,
                y_feature=y_feature,
                llm_json=llm_json,
                payload_str=payload_str,
                computed_confidence=computed_confidence
            )
        )

        if i % 5 == 0 or i == len(base_rows_retry):
            print(f"[INFO] retry progress {i}/{len(base_rows_retry)}")

    except Exception as e:
        print(f"[ERROR] retry failed ({group_dim}, {y_feature}): {repr(e)}")
        traceback.print_exc()

        retry_failed.append({
            "snapshot_quarter": SNAPSHOT_QUARTER,
            "prompt_version": PROMPT_VERSION,
            "group_dim": group_dim,
            "y_feature": y_feature,
            "error": repr(e),
            "payload_json": json.dumps(
                {"context": {"group_dim": group_dim, "y_feature": y_feature}},
                ensure_ascii=False
            ),
            "failed_at": datetime.utcnow()
        })

print(f"[INFO] retry result: success={len(retry_results)}, failed={len(retry_failed)}")

# COMMAND ----------
if retry_results:
    retry_out_sdf = (
        spark.createDataFrame([Row(**rec) for rec in retry_results])
        .select(
            F.col("snapshot_quarter").cast("string").alias("snapshot_quarter"),
            F.col("prompt_version").cast("string").alias("prompt_version"),
            F.col("group_dim").cast("string").alias("group_dim"),
            F.col("y_feature").cast("string").alias("y_feature"),
            F.col("headline").cast("string").alias("headline"),
            F.col("overall_summary").cast("string").alias("overall_summary"),
            F.col("model_confidence_level").cast("string").alias("model_confidence_level"),
            F.col("model_confidence_reason").cast("string").alias("model_confidence_reason"),
            F.col("group_key_insights_json").cast("string").alias("group_key_insights_json"),
            F.col("cross_group_comparison_json").cast("string").alias("cross_group_comparison_json"),
            F.col("raw_llm_json").cast("string").alias("raw_llm_json"),
            F.col("payload_json").cast("string").alias("payload_json"),
            F.to_timestamp("generated_at").alias("generated_at")
        )
    )

    retry_out_sdf.createOrReplaceTempView("insight_updates_v3_retry")

    spark.sql(f"""
        MERGE INTO {TARGET_TABLE} AS t
        USING insight_updates_v3_retry AS s
        ON  t.snapshot_quarter = s.snapshot_quarter
        AND t.prompt_version   = s.prompt_version
        AND t.group_dim        = s.group_dim
        AND t.y_feature        = s.y_feature
        WHEN MATCHED THEN UPDATE SET
        t.headline                    = s.headline,
        t.overall_summary             = s.overall_summary,
        t.model_confidence_level      = s.model_confidence_level,
        t.model_confidence_reason     = s.model_confidence_reason,
        t.group_key_insights_json     = s.group_key_insights_json,
        t.cross_group_comparison_json = s.cross_group_comparison_json,
        t.raw_llm_json                = s.raw_llm_json,
        t.payload_json                = s.payload_json,
        t.generated_at                = s.generated_at
        WHEN NOT MATCHED THEN INSERT (
        snapshot_quarter,
        prompt_version,
        group_dim,
        y_feature,
        headline,
        overall_summary,
        model_confidence_level,
        model_confidence_reason,
        group_key_insights_json,
        cross_group_comparison_json,
        raw_llm_json,
        payload_json,
        generated_at
        )
        VALUES (
        s.snapshot_quarter,
        s.prompt_version,
        s.group_dim,
        s.y_feature,
        s.headline,
        s.overall_summary,
        s.model_confidence_level,
        s.model_confidence_reason,
        s.group_key_insights_json,
        s.cross_group_comparison_json,
        s.raw_llm_json,
        s.payload_json,
        s.generated_at
        )
    """)

    print(f"[INFO] retried rows merged: {retry_out_sdf.count()}")
else:
    print("[INFO] no retry success rows")

# COMMAND ----------
F.col("snapshot_quarter").cast("string").alias("snapshot_quarter"),
F.col("prompt_version").cast("string").alias("prompt_version"),
F.col("group_dim").cast("string").alias("group_dim"),
F.col("y_feature").cast("string").alias("y_feature"),
F.col("error").cast("string").alias("error"),
F.col("payload_json").cast("string").alias("payload_json"),
F.col("failed_at").cast("timestamp").alias("failed_at")
).write.mode("append").saveAsTable(FAILED_TABLE)

print(f"[WARN] retry failed logged: {len(retry_failed)}")

# COMMAND ----------
display(
    spark.table(TARGET_TABLE)
        .filter(F.col("snapshot_quarter") == SNAPSHOT_QUARTER)
        .filter(F.col("prompt_version") == PROMPT_VERSION)
        .orderBy(F.desc("generated_at"))
)

# COMMAND ----------

# COMMAND ----------
dbutils.widgets.text("target_y_feature", "리모컨 사용성")
TARGET_Y_FEATURE = dbutils.widgets.get("target_y_feature").strip()

print(f"[CONFIG] target_y_feature={TARGET_Y_FEATURE}")


# COMMAND ----------

# COMMAND ----------
import re
import json
import time
import traceback
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql import Row, Window as W

TARGET_TABLE = "sandbox.z_jungryo_lee.voc_wls_dashboard_insight_v4"
FAILED_TABLE = "sandbox.z_jungryo_lee.voc_wls_dashboard_insight_failed_v4"


# COMMAND ----------

# COMMAND ----------
def clean_feature_name(name: str) -> str:
    if not name:
        return ""
    s = str(name)

    # 앞쪽 코드성 prefix 제거: 07_02_, 01-, 03.04_ 같은 패턴
    s = re.sub(r"^[0-9]+([_./-][0-9]+)*[_./-]*", "", s)

    # 남은 앞쪽 특수문자 정리
    s = re.sub(r"^[^A-Za-z가-힣]+", "", s)

    # 언더바 정리
    s = s.replace("_", " ").strip()
    return s

def cleanse_driver_list(drivers: list) -> list:
    out = []
    for d in drivers or []:
        x = dict(d)
        x["x_feature"] = clean_feature_name(x.get("x_feature"))
        out.append(x)
    return out

def cleanse_model_summary(model_summary: list) -> list:
    return [dict(x) for x in (model_summary or [])]

def rows_to_pylist(lst):
    if not lst:
        return []
    return [dict(x.asDict()) if hasattr(x, "asDict") else dict(x) for x in lst]

def sort_drivers(drivers):
    if not drivers:
        return []
    return sorted(drivers, key=lambda d: abs(float(d.get("coef", 0.0) or 0.0)), reverse=True)

def json_dumps_safe(obj):
    return json.dumps(obj, ensure_ascii=False)


# COMMAND ----------

# COMMAND ----------
SYSTEM_PROMPT = """
You are a senior product strategy planner for TV products.

Return ONLY strict JSON. No markdown. No explanation outside JSON.

Audience:
- Product planning leaders
- Market strategy managers
- Executives reviewing dashboard summaries

Writing objective:
- Turn model outputs into planning insight, not analyst commentary.
- The reader should immediately understand what value proposition matters by market and how planning priorities should differ.

Language rules:
- Write all outputs in Korean.
- Use concise, executive-friendly Korean.
- Slogan-like phrasing is allowed if it is clear and strategically meaningful.
- Avoid statistical narration unless needed for credibility.
- When mentioning adjusted R-squared, keep the English term exactly as "adjusted R-squared".
- Do not expose raw variable codes such as numeric prefixes or special-character prefixes in x_feature names.

Reasoning rules:
- Use only the provided coefficients, p-values, adjusted R-squared, model p-value, and review volume as evidence.
- Do not invent business facts beyond the payload.
- Translate numeric evidence into planning meaning:
  - what customers appear to value
  - what value proposition is central in each market
  - what product planning emphasis should differ across markets
- Prefer implications and positioning over technical metric descriptions.
- Still stay faithful to the evidence.

Output design rules:
- overall_core_message:
  One sentence.
  This should read like an executive headline for the whole comparison.
- planning_summary:
  2 to 3 sentences.
  Explain what the cross-market pattern means for planning priorities.
- country_insight_cards:
  One object per group_key.
  Each object should contain:
  - country_core: one short and intuitive sentence about what that market treats as the core value of the remote experience
  - summary_comment: one or two sentences explaining the planning meaning
  - key_drivers: 1 to 3 drivers with short comments
  - special_points: distinct points versus other markets
- strategy_takeaways:
  2 to 4 concise statements for planning or portfolio prioritization.
- confidence_note:
  One short sentence using adjusted R-squared, model p-value, and review volume.
  Do not mention confidence level labels like high/mid/low.

Schema:
{
  "overall_core_message": "전체 비교 한 줄 핵심",
  "planning_summary": "기획/전략 관점의 요약 멘트",
  "confidence_note": "adjusted R-squared, model p-value, review volume 기반 신뢰도 메모",
  "country_insight_cards": [
    {
      "group_key": "Brazil",
      "country_core": "브라질 고객은 리모컨을 연결 경험의 중심으로 인식합니다.",
      "summary_comment": "이 시장에서는 음성 제어와 IoT 연동 계열 가치가 만족도를 설명하는 핵심 축입니다.",
      "key_drivers": [
        {
          "x_feature": "Voice Control",
          "direction": "positive",
          "comment": "음성 기반 조작 편의성이 핵심 동인입니다."
        }
      ],
      "special_points": [
        "연결성과 확장성이 다른 시장보다 더 중요한 차별 포인트입니다."
      ]
    }
  ],
  "strategy_takeaways": [
    "시장별로 리모컨의 핵심 가치 정의를 다르게 가져가야 합니다."
  ]
}
""".strip()

def build_user_prompt(group_dim: str, y_feature: str, thresholds: dict) -> str:
    return f"""
You will receive a JSON payload for one y_feature comparison.

Context:
- group_dim: {group_dim}
- y_feature: {y_feature}

Threshold metadata:
{json.dumps(thresholds, ensure_ascii=False)}

Important:
- This output is for dashboard insight used by planners and strategists.
- Make the writing intuitive and decision-friendly.
- Do not expose technical code-like variable prefixes in prose.
- Use adjusted R-squared exactly in English if referenced.
- Return ONLY strict JSON following the required schema.
""".strip()

