# Databricks notebook source
# MAGIC %md
# MAGIC # 12_1. Smart Features Incremental Classification
# MAGIC
# MAGIC `Smart Features & User Experience (UX)` 카테고리만 대상으로 원천 전체 memo_id를 prototype ML로 분류하고, 저신뢰 건은 GPT mini fallback으로 확정합니다.
# MAGIC
# MAGIC - 입력: `raw_review_table` (`settings_intellytics.yaml` 기준 work view)
# MAGIC - 대상: `cate_1_depth = Smart Features & User Experience (UX)`, `is_lifestyle = 'N'`, `sc_measurement in (1, -1)`
# MAGIC - 전제: 10_1 source view refresh 완료
# MAGIC - 포함: Smart 부가 기능 alias 그룹 topic/sample/prototype 준비
# MAGIC - 출력: `memo_embedding_unclassified`, `ml_classification_detail`, `llm_fallback_queue`
# MAGIC - 재실행: `classification_detail_final`에 이미 있는 memo_id는 중복 처리하지 않음

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
import ml.memo_embedding as memo_embedding
import ml.topic_prototype as topic_prototype
import pipeline.run_taxonomy_classification_batch as taxonomy_batch

importlib.reload(config_loader)
importlib.reload(unclassified_embedding)
importlib.reload(topic_ml_classifier)
importlib.reload(memo_embedding)
importlib.reload(topic_prototype)
importlib.reload(taxonomy_batch)

from common.config_loader import load_config, get_output_table, get_reference_table, get_source_table
from pipeline.run_taxonomy_classification_batch import run_taxonomy_classification_batch
from ml.unclassified_embedding import build_and_save_unclassified_embeddings
from ml.memo_embedding import build_and_save_memo_embeddings_ai_query
from ml.topic_prototype import load_memo_embedding_df, build_topic_prototype_df, save_topic_prototypes
from ml.final_classification_builder import load_existing_final_keys
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
SMART_ALIAS_TARGET_CATE_2_DEPTH = "Accessibility Features"
SMART_ALIAS_SOURCE_CATE_2_DEPTHS = [
    "Accessibility Features",
    "Ambient & Gallery Mode",
    "Multi View & Screen Split",
    "Program Guide (EPG)",
    "Recording & Utility Features",
]

# raw 전체 memo_id 대상. 비용/시간 안전 검증이 필요하면 숫자로 줄여 실행하세요.
LIMIT_ROWS = None
LIMIT_ROWS_PER_GROUP = None

RUN_SMART_ALIAS_DESIGN_SAMPLE = True
RUN_SMART_ALIAS_PROTOTYPE_IF_MISSING = True
RUN_EMBEDDING = True
RUN_ML_CLASSIFICATION = True
RUN_GPT_MINI_FALLBACK = True

DESIGN_MODEL_KEY = "gpt_55"
SAMPLE_LABEL_MODEL_KEY = "gpt_mini"
SAMPLE_CLASSIFICATION_ROWS_PER_GROUP = 100
DESIGN_SAMPLE_POOL_ROWS = 500
PROMPT_MEMO_ROWS = 200

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
config["ml_classification"]["taxonomy_design_model_key"] = DESIGN_MODEL_KEY
config["ml_classification"]["fallback_model_key"] = "gpt_mini"
config["app"]["model_key"] = SAMPLE_LABEL_MODEL_KEY

MODEL_KEY = config["app"].get("model_key", SAMPLE_LABEL_MODEL_KEY)
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
        "run_smart_alias_design_sample": RUN_SMART_ALIAS_DESIGN_SAMPLE,
        "run_smart_alias_prototype_if_missing": RUN_SMART_ALIAS_PROTOTYPE_IF_MISSING,
        "fallback_limit_rows": FALLBACK_LIMIT_ROWS,
        "skip_existing": SKIP_EXISTING,
        "design_model_key": DESIGN_MODEL_KEY,
        "sample_label_model_key": SAMPLE_LABEL_MODEL_KEY,
        "model_version": MODEL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "embedding_model": EMBEDDING_MODEL,
    }
)

# COMMAND ----------

# 0. Smart 부가 기능 alias 그룹 topic/sample/prototype 준비
# 기존 산출물을 삭제하지 않습니다. topic/sample/prototype이 이미 있으면 재사용하고, 없는 부분만 생성합니다.
alias_result = {"skipped": True}

if RUN_SMART_ALIAS_DESIGN_SAMPLE:
    alias_config = copy.deepcopy(base_config)
    alias_source_key = alias_config["ml_classification"].get("source_table_key", "raw_review_table")
    escaped_alias_sources = ", ".join(
        "'" + item.replace("'", "''") + "'" for item in SMART_ALIAS_SOURCE_CATE_2_DEPTHS
    )
    alias_config.setdefault("source", {}).setdefault("filters", {}).setdefault(alias_source_key, [])
    alias_config["source"]["filters"][alias_source_key].append(
        f"cate_1_depth = '{TARGET_CATE_1_DEPTH}' AND cate_2_depth IN ({escaped_alias_sources})"
    )
    alias_config.setdefault("taxonomy", {})["max_sample_rows"] = DESIGN_SAMPLE_POOL_ROWS
    alias_config.setdefault("taxonomy", {})["max_rule_sample_rows"] = DESIGN_SAMPLE_POOL_ROWS
    alias_config.setdefault("rule_profile", {})["max_prompt_memos_default"] = PROMPT_MEMO_ROWS
    alias_config.setdefault("topic_pool", {})["max_prompt_memos_default"] = PROMPT_MEMO_ROWS
    alias_config.setdefault("rule_profile", {}).setdefault("max_prompt_memos_by_model", {})[DESIGN_MODEL_KEY] = PROMPT_MEMO_ROWS
    alias_config.setdefault("topic_pool", {}).setdefault("max_prompt_memos_by_model", {})[DESIGN_MODEL_KEY] = PROMPT_MEMO_ROWS
    alias_config.setdefault("pipeline", {})["sample_classification_model_key"] = SAMPLE_LABEL_MODEL_KEY
    alias_config["pipeline"]["run_sample_classification_in_design_batch"] = True
    alias_config.setdefault("ml_classification", {})["taxonomy_design_model_key"] = DESIGN_MODEL_KEY
    alias_config["ml_classification"]["fallback_model_key"] = "gpt_mini"
    alias_config.setdefault("app", {})["model_key"] = SAMPLE_LABEL_MODEL_KEY

    alias_result = run_taxonomy_classification_batch(
        spark,
        config=alias_config,
        model_key=DESIGN_MODEL_KEY,
        limit_group_count=None,
        max_rows_per_group=SAMPLE_CLASSIFICATION_ROWS_PER_GROUP,
        use_llm_fallback=True,
        save_rule_profile=True,
        save_topic_pool=True,
        save_classification_detail=True,
        run_sample_classification=True,
        sample_classification_model_key=SAMPLE_LABEL_MODEL_KEY,
        resume_from_checkpoint=True,
        cleanup_checkpoint_on_success=True,
        continue_on_group_failure=True,
        print_progress=True,
    )

alias_result

# COMMAND ----------

# 0-1. Smart 부가 기능 alias 샘플 라벨 embedding + prototype 생성
alias_prototype_result = {"skipped": True}

if RUN_SMART_ALIAS_PROTOTYPE_IF_MISSING:
    prototype_table = get_output_table(config, "topic_prototype")
    existing_alias_prototype_count = 0
    if spark.catalog.tableExists(prototype_table):
        existing_alias_prototype_count = (
            spark.table(prototype_table)
            .where(F.col("prompt_version") == PROMPT_VERSION)
            .where(F.col("taxonomy_version") == TAXONOMY_VERSION)
            .where(F.col("model_version") == MODEL_VERSION)
            .where(F.col("embedding_model") == EMBEDDING_MODEL)
            .where(F.col("cate_1_depth") == TARGET_CATE_1_DEPTH)
            .where(F.col("cate_2_depth") == SMART_ALIAS_TARGET_CATE_2_DEPTH)
            .where(F.col("sc_measurement").isin(TARGET_SC_MEASUREMENTS))
            .count()
        )

    if existing_alias_prototype_count > 0:
        alias_prototype_result = {
            "skipped": True,
            "reason": "alias prototype already exists",
            "existing_alias_prototype_count": existing_alias_prototype_count,
        }
    else:
        embedding_result = build_and_save_memo_embeddings_ai_query(
            spark,
            config,
            input_table_key="classification_detail",
            output_table_key="memo_embedding",
            embedding_model=EMBEDDING_MODEL,
            limit_rows=None,
            skip_existing=True,
            created_by="smart_alias_labeled_embedding",
        )

        embedding_df = (
            load_memo_embedding_df(
                spark,
                config,
                input_table_key="memo_embedding",
                embedding_model=EMBEDDING_MODEL,
                prompt_version=PROMPT_VERSION,
                taxonomy_version=TAXONOMY_VERSION,
                model_version=MODEL_VERSION,
            )
            .where(F.col("cate_1_depth") == TARGET_CATE_1_DEPTH)
            .where(F.col("cate_2_depth") == SMART_ALIAS_TARGET_CATE_2_DEPTH)
            .where(F.col("sc_measurement").isin(TARGET_SC_MEASUREMENTS))
        )

        prototype_df = build_topic_prototype_df(
            embedding_df,
            min_topic_memo_count=config["topic_prototype"].get("min_topic_memo_count", 3),
        )
        prototype_count = prototype_df.count()
        if prototype_count > 0:
            prototype_saved_table = save_topic_prototypes(
                prototype_df,
                config,
                output_table_key="topic_prototype",
                mode="append",
            )
        else:
            prototype_saved_table = prototype_table

        alias_prototype_result = {
            "embedding_result": embedding_result,
            "prototype_count": prototype_count,
            "prototype_table": prototype_saved_table,
        }

alias_prototype_result

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

final_done_keys = load_existing_final_keys(spark, config)
fallback_required_df = fallback_required_df.join(
    final_done_keys,
    on=["cate_1_depth", "cate_2_depth", "sc_measurement", "memo_id"],
    how="left_anti",
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

queue_df = build_llm_fallback_queue_df(
    fallback_required_df,
    config,
    created_by="smart_features_full_fallback_queue",
)

if spark.catalog.tableExists(queue_table):
    existing_queue_keys = (
        spark.table(queue_table)
        .where(F.col("prompt_version") == PROMPT_VERSION)
        .where(F.col("taxonomy_version") == TAXONOMY_VERSION)
        .where(F.col("model_version") == MODEL_VERSION)
        .where(F.col("status") == "pending")
        .where(F.col("cate_1_depth") == TARGET_CATE_1_DEPTH)
        .where(F.col("sc_measurement").isin(TARGET_SC_MEASUREMENTS))
        .select("cate_1_depth", "cate_2_depth", "sc_measurement", "memo_id")
        .dropDuplicates()
    )
    queue_df = queue_df.join(
        existing_queue_keys,
        on=["cate_1_depth", "cate_2_depth", "sc_measurement", "memo_id"],
        how="left_anti",
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

    final_done_keys = load_existing_final_keys(spark, config)
    pending_queue_df = pending_queue_df.join(
        final_done_keys,
        on=["cate_1_depth", "cate_2_depth", "sc_measurement", "memo_id"],
        how="left_anti",
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
