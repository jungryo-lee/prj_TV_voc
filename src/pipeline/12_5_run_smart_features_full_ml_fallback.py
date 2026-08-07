# Databricks notebook source
# MAGIC %md
# MAGIC # 12.5. Smart Features Full ML Classification + GPT Mini Fallback
# MAGIC
# MAGIC `Smart Features & User Experience (UX)` 카테고리만 대상으로 원천 전체 memo_id를 prototype ML로 분류하고, 저신뢰 건은 GPT mini fallback으로 확정합니다.
# MAGIC
# MAGIC - 입력: `raw_review_table` (`settings_intellytics.yaml` 기준 work view)
# MAGIC - 대상: `cate_1_depth = Smart Features & User Experience (UX)`, `is_lifestyle = 'N'`, `sc_measurement in (1, -1)`
# MAGIC - 전제: 10번 source view, 11번 sample classification, 11.5 embedding/prototype 생성 완료
# MAGIC - 출력: `memo_embedding_unclassified`, `ml_classification_detail`, `llm_fallback_queue`
# MAGIC - 재실행: 기존 embedding / ML / fallback 결과가 있는 memo_id는 중복 처리하지 않음

# COMMAND ----------

import copy
import sys
import importlib

from pyspark.sql import functions as F

PROJECT_ROOT = "/Workspace/Users/jungryo.lee@lge.com/prj_TV_voc"
SRC_ROOT = f"{PROJECT_ROOT}/src"

if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

import common.config_loader as config_loader
import ml.unclassified_embedding as unclassified_embedding
import ml.topic_ml_classifier as topic_ml_classifier

importlib.reload(config_loader)
importlib.reload(unclassified_embedding)
importlib.reload(topic_ml_classifier)

from common.config_loader import load_config, get_output_table, get_reference_table, get_source_table
from ml.unclassified_embedding import build_and_save_unclassified_embeddings
from ml.topic_ml_classifier import (
    build_ml_classification_df,
    save_ml_classification,
    load_query_embedding_df,
    load_topic_prototype_df,
    load_fallback_required_ml_df,
    build_llm_fallback_queue_df,
    save_llm_fallback_queue,
    classify_llm_fallback_queue_df,
)

base_config = load_config(f"{PROJECT_ROOT}/config/settings_intellytics.yaml")
config = copy.deepcopy(base_config)

print("settings =", config["path"]["settings"])
print("source =", get_source_table(config, "raw_review_table"))
print("embedding_model =", config["ml_classification"]["embedding_model"])
print("fallback_model =", config["ml_classification"]["fallback_model_key"])

# COMMAND ----------

# 실행 옵션
TARGET_CATE_1_DEPTH = "Smart Features & User Experience (UX)"
TARGET_SC_MEASUREMENTS = [1, -1]

# raw 전체 memo_id 대상. 비용/시간 안전 검증이 필요하면 숫자로 줄여 실행하세요.
LIMIT_ROWS = None
LIMIT_ROWS_PER_GROUP = None

RUN_EMBEDDING = True
RUN_ML_CLASSIFICATION = True
RUN_GPT_MINI_FALLBACK = True

# fallback 전체 실행. 비용 안전 검증이 필요하면 숫자로 줄여 실행하세요.
FALLBACK_LIMIT_ROWS = None

SKIP_EXISTING = True

source_key = config["ml_classification"].get("source_table_key", "raw_review_table")
config.setdefault("source", {}).setdefault("filters", {}).setdefault(source_key, [])
config["source"]["filters"][source_key].append(
    f"cate_1_depth = '{TARGET_CATE_1_DEPTH}'"
)
config["ml_classification"]["target_sentiments"] = TARGET_SC_MEASUREMENTS
config["ml_classification"]["limit_rows"] = LIMIT_ROWS
config["ml_classification"]["limit_rows_per_group"] = LIMIT_ROWS_PER_GROUP

MODEL_KEY = config["app"].get("model_key", "gpt_55")
MODEL_VERSION = config["llm"]["models"][MODEL_KEY]["model_version"]
PROMPT_VERSION = config["version"]["prompt_version"]
TAXONOMY_VERSION = config["version"]["taxonomy_version"]
EMBEDDING_MODEL = config["ml_classification"].get(
    "embedding_model", "databricks-bge-large-en"
)

print(
    {
        "target_cate_1_depth": TARGET_CATE_1_DEPTH,
        "target_sc_measurements": TARGET_SC_MEASUREMENTS,
        "limit_rows": LIMIT_ROWS,
        "limit_rows_per_group": LIMIT_ROWS_PER_GROUP,
        "run_embedding": RUN_EMBEDDING,
        "run_ml_classification": RUN_ML_CLASSIFICATION,
        "run_gpt_mini_fallback": RUN_GPT_MINI_FALLBACK,
        "fallback_limit_rows": FALLBACK_LIMIT_ROWS,
        "skip_existing": SKIP_EXISTING,
        "model_version": MODEL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "embedding_model": EMBEDDING_MODEL,
    }
)

# COMMAND ----------

# 사전 점검: source / prototype / 기존 ML 처리 현황
source_table = get_source_table(config, "raw_review_table")
prototype_table = get_output_table(config, "topic_prototype")
embedding_table = get_output_table(config, "memo_embedding_unclassified")
ml_table = get_output_table(config, "ml_classification_detail")
queue_table = get_output_table(config, "llm_fallback_queue")
category_mapping_table = get_reference_table(config, "category_mapping_table")

source_summary_df = spark.sql(
    f"""
    SELECT
      cate_1_depth,
      cate_2_depth,
      CAST(sc_measurement AS INT) AS sc_measurement,
      COUNT(*) AS raw_rows,
      COUNT(DISTINCT memo) AS distinct_memo_text_cnt
    FROM {source_table}
    WHERE cate_1_depth = '{TARGET_CATE_1_DEPTH}'
      AND is_lifestyle = 'N'
      AND CAST(sc_measurement AS INT) IN ({", ".join(str(v) for v in TARGET_SC_MEASUREMENTS)})
      AND memo IS NOT NULL
      AND LENGTH(TRIM(CAST(memo AS STRING))) > 0
    GROUP BY cate_1_depth, cate_2_depth, CAST(sc_measurement AS INT)
    ORDER BY cate_2_depth, sc_measurement
    """
)

display(source_summary_df)

prototype_summary_df = (
    spark.table(prototype_table)
    .where(F.col("cate_1_depth") == TARGET_CATE_1_DEPTH)
    .where(F.col("prompt_version") == PROMPT_VERSION)
    .where(F.col("taxonomy_version") == TAXONOMY_VERSION)
    .where(F.col("model_version") == MODEL_VERSION)
    .where(F.col("embedding_model") == EMBEDDING_MODEL)
    .groupBy("cate_1_depth", "cate_2_depth", "sc_measurement")
    .agg(
        F.count("*").alias("prototype_topic_cnt"),
        F.sum("prototype_distinct_memo_id_cnt").alias("prototype_label_memo_cnt"),
    )
    .orderBy("cate_2_depth", "sc_measurement")
)

display(prototype_summary_df)

# COMMAND ----------

# 1. Smart Features 원천 전체 memo_id embedding 생성
embedding_result = {"skipped": True}

if RUN_EMBEDDING:
    embedding_result = build_and_save_unclassified_embeddings(
        spark,
        config,
        embedding_model=EMBEDDING_MODEL,
        limit_rows=LIMIT_ROWS,
        limit_rows_per_group=LIMIT_ROWS_PER_GROUP,
        skip_existing=SKIP_EXISTING,
    )

embedding_result

# COMMAND ----------

# 2. Smart Features prototype ML 분류
ml_result = {"skipped": True}

if RUN_ML_CLASSIFICATION:
    query_embedding_df = (
        load_query_embedding_df(
            spark,
            config,
            embedding_model=EMBEDDING_MODEL,
        )
        .where(F.col("cate_1_depth") == TARGET_CATE_1_DEPTH)
        .where(F.col("sc_measurement").isin(TARGET_SC_MEASUREMENTS))
    )

    prototype_df = (
        load_topic_prototype_df(
            spark,
            config,
            embedding_model=EMBEDDING_MODEL,
        )
        .where(F.col("cate_1_depth") == TARGET_CATE_1_DEPTH)
        .where(F.col("sc_measurement").isin(TARGET_SC_MEASUREMENTS))
    )

    print("query_embedding_rows =", query_embedding_df.count())
    print("prototype_rows =", prototype_df.count())

    ml_df = build_ml_classification_df(
        spark,
        config,
        query_embedding_df=query_embedding_df,
        prototype_df=prototype_df,
        skip_existing=SKIP_EXISTING,
        limit_rows=LIMIT_ROWS,
        created_by="smart_features_full_ml_classifier",
    )

    ml_count = ml_df.count()
    auto_accept_count = ml_df.where(
        F.col("classification_stage") == "embedding_prototype_auto_accept"
    ).count()
    fallback_required_count = ml_count - auto_accept_count

    if ml_count > 0:
        saved_ml_table = save_ml_classification(
            ml_df,
            config,
            output_table_key="ml_classification_detail",
            mode="append",
        )
    else:
        saved_ml_table = ml_table

    ml_result = {
        "saved_ml_table": saved_ml_table,
        "ml_count": ml_count,
        "auto_accept_count": auto_accept_count,
        "fallback_required_count": fallback_required_count,
    }

ml_result

# COMMAND ----------

# 3. Smart Features fallback queue를 Smart Features 대상만 재구성
fallback_queue_result = {"skipped": True}

fallback_required_df = (
    load_fallback_required_ml_df(
        spark,
        config,
        input_table_key="ml_classification_detail",
    )
    .where(F.col("cate_1_depth") == TARGET_CATE_1_DEPTH)
    .where(F.col("sc_measurement").isin(TARGET_SC_MEASUREMENTS))
)

existing_fallback_keys = (
    spark.table(ml_table)
    .where(F.col("prompt_version") == PROMPT_VERSION)
    .where(F.col("taxonomy_version") == TAXONOMY_VERSION)
    .where(F.col("model_version") == MODEL_VERSION)
    .where(F.col("classification_stage") == "gpt_mini_fallback")
    .where(F.col("cate_1_depth") == TARGET_CATE_1_DEPTH)
    .where(F.col("sc_measurement").isin(TARGET_SC_MEASUREMENTS))
    .select("cate_1_depth", "cate_2_depth", "sc_measurement", "memo_id")
    .dropDuplicates()
)

fallback_required_df = fallback_required_df.join(
    existing_fallback_keys,
    on=["cate_1_depth", "cate_2_depth", "sc_measurement", "memo_id"],
    how="left_anti",
)

fallback_required_count = fallback_required_df.count()
print("smart_features_fallback_required_count =", fallback_required_count)

if spark.catalog.tableExists(queue_table):
    spark.sql(
        f"""
        DELETE FROM {queue_table}
        WHERE prompt_version = '{PROMPT_VERSION}'
          AND taxonomy_version = '{TAXONOMY_VERSION}'
          AND model_version = '{MODEL_VERSION}'
          AND status = 'pending'
          AND cate_1_depth = '{TARGET_CATE_1_DEPTH}'
          AND CAST(sc_measurement AS INT) IN ({", ".join(str(v) for v in TARGET_SC_MEASUREMENTS)})
        """
    )

queue_df = build_llm_fallback_queue_df(
    fallback_required_df,
    config,
    created_by="smart_features_full_fallback_queue",
)

queue_count = queue_df.count()
if queue_count > 0:
    saved_queue_table = save_llm_fallback_queue(
        queue_df,
        config,
        output_table_key="llm_fallback_queue",
        mode="append",
    )
else:
    saved_queue_table = queue_table

fallback_queue_result = {
    "saved_queue_table": saved_queue_table,
    "fallback_required_count": fallback_required_count,
    "queue_count": queue_count,
}

fallback_queue_result

# COMMAND ----------

# 4. Smart Features GPT mini fallback 실행
fallback_result = {"skipped": True}

if RUN_GPT_MINI_FALLBACK:
    pending_queue_df = (
        spark.table(queue_table)
        .where(F.col("prompt_version") == PROMPT_VERSION)
        .where(F.col("taxonomy_version") == TAXONOMY_VERSION)
        .where(F.col("model_version") == MODEL_VERSION)
        .where(F.col("status") == "pending")
        .where(F.col("cate_1_depth") == TARGET_CATE_1_DEPTH)
        .where(F.col("sc_measurement").isin(TARGET_SC_MEASUREMENTS))
        .orderBy(
            F.col("prototype_confidence_score").desc_nulls_last(),
            F.col("cate_2_depth"),
            F.col("sc_measurement"),
            F.col("memo_id"),
        )
    )

    if FALLBACK_LIMIT_ROWS is not None:
        pending_queue_df = pending_queue_df.limit(int(FALLBACK_LIMIT_ROWS))

    pending_count = pending_queue_df.count()
    print("smart_features_pending_fallback_count =", pending_count)

    fallback_df = classify_llm_fallback_queue_df(
        spark,
        config,
        queue_df=pending_queue_df,
        model_key=config["ml_classification"].get("fallback_model_key", "gpt_mini"),
        limit_rows=None,
        created_by="smart_features_full_gpt_mini_fallback",
    )

    fallback_count = fallback_df.count()
    if fallback_count > 0:
        saved_fallback_table = save_ml_classification(
            fallback_df,
            config,
            output_table_key="ml_classification_detail",
            mode="append",
        )
    else:
        saved_fallback_table = ml_table

    fallback_result = {
        "saved_fallback_table": saved_fallback_table,
        "pending_count": pending_count,
        "fallback_count": fallback_count,
    }

fallback_result

# COMMAND ----------

# 5. 결과 요약: 13번 finalization 전에 unresolved fallback이 남았는지 확인
ml_df = (
    spark.table(ml_table)
    .where(F.col("prompt_version") == PROMPT_VERSION)
    .where(F.col("taxonomy_version") == TAXONOMY_VERSION)
    .where(F.col("model_version") == MODEL_VERSION)
    .where(F.col("cate_1_depth") == TARGET_CATE_1_DEPTH)
    .where(F.col("sc_measurement").isin(TARGET_SC_MEASUREMENTS))
)

fallback_done_keys = (
    ml_df.where(F.col("classification_stage") == "gpt_mini_fallback")
    .select("cate_1_depth", "cate_2_depth", "sc_measurement", "memo_id")
    .dropDuplicates()
)

pending_unresolved_df = (
    ml_df.where(F.col("classification_stage") == "embedding_prototype_llm_fallback")
    .join(
        fallback_done_keys,
        on=["cate_1_depth", "cate_2_depth", "sc_measurement", "memo_id"],
        how="left_anti",
    )
)

summary_df = (
    ml_df.groupBy("cate_1_depth", "cate_2_depth", "sc_measurement", "classification_stage")
    .agg(
        F.count("*").alias("row_cnt"),
        F.countDistinct("memo_id").alias("distinct_memo_id_cnt"),
        F.avg("confidence_score").alias("avg_confidence_score"),
    )
    .orderBy("cate_2_depth", "sc_measurement", "classification_stage")
)

display(summary_df)

display(
    spark.createDataFrame(
        [
            {
                "target_cate_1_depth": TARGET_CATE_1_DEPTH,
                "ml_total_rows": ml_df.count(),
                "ml_distinct_memo_id_cnt": ml_df.select("memo_id").dropDuplicates().count(),
                "auto_accept_rows": ml_df.where(
                    F.col("classification_stage") == "embedding_prototype_auto_accept"
                ).count(),
                "pending_fallback_rows": ml_df.where(
                    F.col("classification_stage") == "embedding_prototype_llm_fallback"
                ).count(),
                "gpt_mini_fallback_rows": ml_df.where(
                    F.col("classification_stage") == "gpt_mini_fallback"
                ).count(),
                "unresolved_pending_fallback_rows": pending_unresolved_df.count(),
            }
        ]
    )
)
