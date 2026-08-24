# Databricks notebook source
# MAGIC %md
# MAGIC # 17_0. Run Non-Smart Incremental Classification
# MAGIC 
# MAGIC 비스마트 카테고리 전체를 대상으로, 아직 최종 분류되지 않은 memo_id만 주제분류를 점진 실행합니다.
# MAGIC 
# MAGIC - 주제 설계: `gpt_55`로 rule profile / topic pool 생성 또는 기존 결과 재사용
# MAGIC - 샘플 라벨: topic pool 기준 그룹별 100건을 `gpt_mini`로 분류
# MAGIC - Prototype: 샘플 라벨 embedding 기반 topic prototype 생성
# MAGIC - 전수/증분 분류: `classification_detail_final`에 아직 없는 memo_id 중 그룹별 최대 500건 분류
# MAGIC - 저신뢰 fallback: GPT mini로 fallback queue 분류
# MAGIC - 최종화: 신규 memo_id만 final detail에 append하고, topic group 생성
# MAGIC 
# MAGIC Smart Features 전수/부가 기능 alias 처리는 12_1 노트북에서 담당합니다.

# COMMAND ----------
import sys
import copy
import importlib

from pyspark.sql import functions as F
from pyspark.sql.window import Window

PROJECT_ROOT = "/Workspace/Users/jungryo.lee@lge.com/prj_TV_voc"
SRC_ROOT = f"{PROJECT_ROOT}/src"

if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

import common.config_loader as config_loader
import pipeline.run_taxonomy_classification_batch as taxonomy_batch
import ml.memo_embedding as memo_embedding
import ml.topic_prototype as topic_prototype
import pipeline.run_ml_topic_classification as ml_batch
import ml.topic_ml_classifier as topic_ml_classifier
import ml.final_classification_builder as final_builder
import taxonomy.topic_group_generator as topic_group_generator

importlib.reload(config_loader)
importlib.reload(taxonomy_batch)
importlib.reload(memo_embedding)
importlib.reload(topic_prototype)
importlib.reload(ml_batch)
importlib.reload(topic_ml_classifier)
importlib.reload(final_builder)
importlib.reload(topic_group_generator)

from common.config_loader import load_config, get_output_table, get_reference_table, get_source_table, get_log_table
from pipeline.run_taxonomy_classification_batch import run_taxonomy_classification_batch
from ml.memo_embedding import build_and_save_memo_embeddings_ai_query
from ml.topic_prototype import load_memo_embedding_df, build_topic_prototype_df, save_topic_prototypes
from pipeline.run_ml_topic_classification import run_ml_topic_classification
from ml.topic_ml_classifier import (
    load_pending_llm_fallback_queue,
    classify_llm_fallback_queue_df,
    save_ml_classification,
)
from ml.final_classification_builder import (
    build_final_classification_detail_df,
    load_existing_final_keys,
    save_final_classification_detail,
)
from taxonomy.topic_group_generator import generate_and_save_topic_groups

base_config = load_config(f"{PROJECT_ROOT}/config/settings_intellytics.yaml")
config = copy.deepcopy(base_config)

print("source =", get_source_table(config, "raw_review_table"))
print("classification_detail =", get_output_table(config, "classification_detail"))
print("topic_prototype =", get_output_table(config, "topic_prototype"))
print("ml_classification_detail =", get_output_table(config, "ml_classification_detail"))

# COMMAND ----------
# 실행 옵션
# 셀을 반복 실행해도 source filter가 누적되지 않도록 매번 원본 설정에서 새로 시작합니다.
config = copy.deepcopy(base_config)

EXCLUDED_CATE_1_DEPTHS = ["Smart Features & User Experience (UX)"]

# 주제 생성용 원천 sample pool 상한입니다. 실제 prompt에는 아래 PROMPT_MEMO_ROWS만큼 diverse sample을 넣습니다.
DESIGN_SAMPLE_POOL_ROWS = 500
PROMPT_MEMO_ROWS = 200

# topic pool 기준 샘플 주제분류 건수입니다.
SAMPLE_CLASSIFICATION_ROWS_PER_GROUP = 100

# ML/prototype 분류는 아직 처리되지 않은 memo_id 중 그룹별 최대 500건씩 증분 처리합니다.
ML_ROWS_PER_GROUP = 500

DESIGN_MODEL_KEY = "gpt_55"
SAMPLE_LABEL_MODEL_KEY = "gpt_mini"
FALLBACK_MODEL_KEY = "gpt_mini"
EMBEDDING_MODEL = config["ml_classification"].get("embedding_model", "databricks-bge-large-en")

# 필요 시 비용 안전장치로 LIMIT_GROUP_COUNT를 숫자로 줄여 먼저 검증하세요. 전체 실행은 None.
LIMIT_GROUP_COUNT = None

RUN_11_DESIGN_AND_SAMPLE = True
RUN_11_5_EMBEDDING_AND_PROTOTYPE = True
RUN_12_ML = True
RUN_GPT_MINI_FALLBACK = True
RUN_13_FINALIZE = True
RUN_14_TOPIC_GROUPING = True

SKIP_EXISTING = True
CONTINUE_ON_GROUP_FAILURE = True
RESET_FAILED_CHECKPOINT_ROWS = False

PROMPT_VERSION = config["version"]["prompt_version"]
TAXONOMY_VERSION = config["version"]["taxonomy_version"]
LABEL_MODEL_VERSION = config["llm"]["models"][SAMPLE_LABEL_MODEL_KEY]["model_version"]

source_key = config["ml_classification"].get("source_table_key", "raw_review_table")
escaped_excluded = ", ".join("'" + item.replace("'", "''") + "'" for item in EXCLUDED_CATE_1_DEPTHS)
sc_sql = "1, -1"

# raw source filter: 비스마트 전체만 포함합니다.
scope_filter = f"""
(
  cate_1_depth NOT IN ({escaped_excluded})
)
""".strip()

# output table filter: 비스마트 전체만 포함합니다.
output_scope_sql = f"""
(
  cate_1_depth NOT IN ({escaped_excluded})
)
""".strip()

config.setdefault("source", {}).setdefault("filters", {}).setdefault(source_key, [])
config["source"]["filters"][source_key].append(scope_filter)

config.setdefault("taxonomy", {})["max_sample_rows"] = DESIGN_SAMPLE_POOL_ROWS
config.setdefault("taxonomy", {})["max_rule_sample_rows"] = DESIGN_SAMPLE_POOL_ROWS
config.setdefault("rule_profile", {})["max_prompt_memos_default"] = PROMPT_MEMO_ROWS
config.setdefault("topic_pool", {})["max_prompt_memos_default"] = PROMPT_MEMO_ROWS
config.setdefault("rule_profile", {}).setdefault("max_prompt_memos_by_model", {})[DESIGN_MODEL_KEY] = PROMPT_MEMO_ROWS
config.setdefault("topic_pool", {}).setdefault("max_prompt_memos_by_model", {})[DESIGN_MODEL_KEY] = PROMPT_MEMO_ROWS
config.setdefault("pipeline", {})["sample_classification_model_key"] = SAMPLE_LABEL_MODEL_KEY
config["pipeline"]["run_sample_classification_in_design_batch"] = True
config.setdefault("ml_classification", {})["limit_rows_per_group"] = ML_ROWS_PER_GROUP
config["ml_classification"]["taxonomy_design_model_key"] = DESIGN_MODEL_KEY
config["ml_classification"]["fallback_model_key"] = FALLBACK_MODEL_KEY

# 샘플 라벨/prototype/ML 결과는 gpt_mini model_version 기준으로 이어집니다.
config.setdefault("app", {})["model_key"] = SAMPLE_LABEL_MODEL_KEY

print({
    "excluded_cate_1_depths": EXCLUDED_CATE_1_DEPTHS,
    "design_sample_pool_rows": DESIGN_SAMPLE_POOL_ROWS,
    "prompt_memo_rows": PROMPT_MEMO_ROWS,
    "sample_classification_rows_per_group": SAMPLE_CLASSIFICATION_ROWS_PER_GROUP,
    "ml_rows_per_group": ML_ROWS_PER_GROUP,
    "design_model_key": DESIGN_MODEL_KEY,
    "sample_label_model_key": SAMPLE_LABEL_MODEL_KEY,
    "label_model_version": LABEL_MODEL_VERSION,
    "fallback_model_key": FALLBACK_MODEL_KEY,
    "limit_group_count": LIMIT_GROUP_COUNT,
    "source_filters": config["source"]["filters"][source_key],
})

# COMMAND ----------
# 0. 실행 전 대상 그룹/기존 처리 현황 확인
raw_table = get_source_table(config, "raw_review_table")
classification_detail_table = get_output_table(config, "classification_detail")
prototype_table = get_output_table(config, "topic_prototype")
ml_table = get_output_table(config, "ml_classification_detail")
final_detail_table = get_output_table(config, "classification_detail_final")
topic_group_table = get_output_table(config, "topic_group")
category_mapping_table = get_reference_table(config, "category_mapping_table")

print("[mode] Non-Smart incremental only. Existing final memo_ids will be reused and skipped.")

raw_alias_expr = "cate_2_depth"

status_sql = f"""
WITH raw_group AS (
  SELECT cate_1_depth, {raw_alias_expr} AS cate_2_depth, CAST(sc_measurement AS INT) AS sc_measurement,
         COUNT(*) AS raw_rows
  FROM {raw_table}
  WHERE memo IS NOT NULL
    AND LENGTH(TRIM(CAST(memo AS STRING))) > 0
    AND sc_measurement IN ({sc_sql})
    AND is_lifestyle = 'N'
    AND {scope_filter}
  GROUP BY cate_1_depth, {raw_alias_expr}, CAST(sc_measurement AS INT)
), sample_detail AS (
  SELECT cate_1_depth, cate_2_depth, sc_measurement,
         COUNT(DISTINCT memo_id) AS sample_detail_memo_cnt
  FROM {classification_detail_table}
  WHERE prompt_version = '{PROMPT_VERSION}'
    AND taxonomy_version = '{TAXONOMY_VERSION}'
    AND model_version = '{LABEL_MODEL_VERSION}'
    AND {output_scope_sql}
  GROUP BY cate_1_depth, cate_2_depth, sc_measurement
), prototype AS (
  SELECT cate_1_depth, cate_2_depth, sc_measurement,
         COUNT(*) AS prototype_topic_cnt
  FROM {prototype_table}
  WHERE prompt_version = '{PROMPT_VERSION}'
    AND taxonomy_version = '{TAXONOMY_VERSION}'
    AND model_version = '{LABEL_MODEL_VERSION}'
    AND embedding_model = '{EMBEDDING_MODEL}'
    AND {output_scope_sql}
  GROUP BY cate_1_depth, cate_2_depth, sc_measurement
), ml_done AS (
  SELECT cate_1_depth, cate_2_depth, sc_measurement,
         COUNT(DISTINCT memo_id) AS ml_done_memo_cnt
  FROM {ml_table}
  WHERE prompt_version = '{PROMPT_VERSION}'
    AND taxonomy_version = '{TAXONOMY_VERSION}'
    AND model_version = '{LABEL_MODEL_VERSION}'
    AND {output_scope_sql}
  GROUP BY cate_1_depth, cate_2_depth, sc_measurement
), final_done AS (
  SELECT cate_1_depth, cate_2_depth, sc_measurement,
         COUNT(DISTINCT memo_id) AS final_done_memo_cnt
  FROM {final_detail_table}
  WHERE prompt_version = '{PROMPT_VERSION}'
    AND taxonomy_version = '{TAXONOMY_VERSION}'
    AND {output_scope_sql}
  GROUP BY cate_1_depth, cate_2_depth, sc_measurement
), topic_group_done AS (
  SELECT cate_1_depth, cate_2_depth, sc_measurement,
         COUNT(DISTINCT topic_group) AS topic_group_cnt
  FROM {topic_group_table}
  WHERE {output_scope_sql}
  GROUP BY cate_1_depth, cate_2_depth, sc_measurement
)
SELECT
  COALESCE(m.cate_1_depth_kor, r.cate_1_depth) AS cate_1_depth_kor,
  r.cate_1_depth,
  COALESCE(m.cate_2_depth_kor, r.cate_2_depth) AS cate_2_depth_kor,
  r.cate_2_depth,
  r.sc_measurement,
  r.raw_rows,
  COALESCE(s.sample_detail_memo_cnt, 0) AS sample_detail_memo_cnt,
  COALESCE(p.prototype_topic_cnt, 0) AS prototype_topic_cnt,
  COALESCE(md.ml_done_memo_cnt, 0) AS ml_done_memo_cnt,
  COALESCE(fd.final_done_memo_cnt, 0) AS final_done_memo_cnt,
  COALESCE(tg.topic_group_cnt, 0) AS topic_group_cnt,
  CASE WHEN COALESCE(s.sample_detail_memo_cnt, 0) > 0 THEN 1 ELSE 0 END AS has_sample_detail,
  CASE WHEN COALESCE(p.prototype_topic_cnt, 0) > 0 THEN 1 ELSE 0 END AS has_prototype,
  CASE WHEN COALESCE(md.ml_done_memo_cnt, 0) >= {ML_ROWS_PER_GROUP} THEN 1 ELSE 0 END AS has_first_500_ml,
  CASE WHEN COALESCE(fd.final_done_memo_cnt, 0) > 0 THEN 1 ELSE 0 END AS has_final,
  CASE WHEN COALESCE(tg.topic_group_cnt, 0) > 0 THEN 1 ELSE 0 END AS has_topic_group
FROM raw_group r
LEFT JOIN sample_detail s USING (cate_1_depth, cate_2_depth, sc_measurement)
LEFT JOIN prototype p USING (cate_1_depth, cate_2_depth, sc_measurement)
LEFT JOIN ml_done md USING (cate_1_depth, cate_2_depth, sc_measurement)
LEFT JOIN final_done fd USING (cate_1_depth, cate_2_depth, sc_measurement)
LEFT JOIN topic_group_done tg USING (cate_1_depth, cate_2_depth, sc_measurement)
LEFT JOIN {category_mapping_table} m
  ON r.cate_1_depth = m.cate_1_depth
 AND r.cate_2_depth = m.cate_2_depth
ORDER BY has_final, has_topic_group, has_prototype, has_sample_detail, has_first_500_ml,
         r.cate_1_depth, r.cate_2_depth, r.sc_measurement
"""

display(spark.sql(status_sql))

# COMMAND ----------
# 1. 주제 생성 + 샘플 100건 주제분류
# 이미 sample classification_detail이 있는 그룹은 skip되고, topic_pool만 있는 그룹도 재사용됩니다.
if RUN_11_DESIGN_AND_SAMPLE:
    design_sample_result = run_taxonomy_classification_batch(
        spark,
        config=config,
        model_key=DESIGN_MODEL_KEY,
        limit_group_count=LIMIT_GROUP_COUNT,
        max_rows_per_group=SAMPLE_CLASSIFICATION_ROWS_PER_GROUP,
        use_llm_fallback=True,
        save_rule_profile=True,
        save_topic_pool=True,
        save_classification_detail=True,
        run_sample_classification=True,
        sample_classification_model_key=SAMPLE_LABEL_MODEL_KEY,
        resume_from_checkpoint=True,
        reset_failed_checkpoint_rows=RESET_FAILED_CHECKPOINT_ROWS,
        cleanup_checkpoint_on_success=True,
        continue_on_group_failure=CONTINUE_ON_GROUP_FAILURE,
        print_progress=True,
    )
else:
    design_sample_result = {"skipped": True}

design_sample_result

# COMMAND ----------
# 2. 샘플 라벨 embedding + topic prototype 생성
if RUN_11_5_EMBEDDING_AND_PROTOTYPE:
    embedding_result = build_and_save_memo_embeddings_ai_query(
        spark,
        config,
        input_table_key="classification_detail",
        output_table_key="memo_embedding",
        embedding_model=EMBEDDING_MODEL,
        limit_rows=None,
        skip_existing=SKIP_EXISTING,
        created_by="non_smart_labeled_embedding",
    )

    embedding_df = (
        load_memo_embedding_df(
            spark,
            config,
            input_table_key="memo_embedding",
            embedding_model=EMBEDDING_MODEL,
            prompt_version=PROMPT_VERSION,
            taxonomy_version=TAXONOMY_VERSION,
            model_version=LABEL_MODEL_VERSION,
        )
        .where(F.expr(output_scope_sql))
    )

    dedupe_window = Window.partitionBy(
        "cate_1_depth",
        "cate_2_depth",
        "sc_measurement",
        "memo_id",
        "prompt_version",
        "taxonomy_version",
        "model_version",
    ).orderBy(F.col("created_at").desc_nulls_last(), F.col("run_id").desc_nulls_last())

    embedding_df = (
        embedding_df.withColumn("_rn", F.row_number().over(dedupe_window))
        .where(F.col("_rn") == 1)
        .drop("_rn")
    )

    prototype_df = build_topic_prototype_df(
        embedding_df,
        min_topic_memo_count=config["topic_prototype"].get("min_topic_memo_count", 3),
    )

    prototype_table = get_output_table(config, "topic_prototype")
    if spark.catalog.tableExists(prototype_table):
        existing_prototype_groups = (
            spark.table(prototype_table)
            .where(F.col("prompt_version") == PROMPT_VERSION)
            .where(F.col("taxonomy_version") == TAXONOMY_VERSION)
            .where(F.col("model_version") == LABEL_MODEL_VERSION)
            .where(F.col("embedding_model") == EMBEDDING_MODEL)
            .where(F.expr(output_scope_sql))
            .select("cate_1_depth", "cate_2_depth", "sc_measurement")
            .dropDuplicates()
        )
        prototype_df = prototype_df.join(
            existing_prototype_groups,
            on=["cate_1_depth", "cate_2_depth", "sc_measurement"],
            how="left_anti",
        )
        prototype_save_mode = "append"
    else:
        prototype_save_mode = "overwrite"

    prototype_count = prototype_df.count()
    if prototype_count > 0:
        prototype_saved_table = save_topic_prototypes(
            prototype_df,
            config,
            output_table_key="topic_prototype",
            mode=prototype_save_mode,
        )
    else:
        prototype_saved_table = prototype_table

    prototype_result = {
        "embedding_result": embedding_result,
        "new_prototype_count": prototype_count,
        "prototype_table": prototype_saved_table,
    }
else:
    prototype_result = {"skipped": True}

prototype_result

# COMMAND ----------
# 3. 아직 처리되지 않은 memo_id 중 그룹별 최대 500건 ML/prototype 분류
if RUN_12_ML:
    ml_result = run_ml_topic_classification(
        spark,
        config,
        run_embedding=True,
        run_classification=True,
        run_llm_fallback=False,
        limit_rows=None,
        limit_rows_per_group=ML_ROWS_PER_GROUP,
        fallback_limit_rows=None,
        skip_existing=SKIP_EXISTING,
    )
else:
    ml_result = {"skipped": True}

ml_result

# COMMAND ----------
# 4. GPT mini fallback 분류
# 이미 gpt_mini_fallback 결과가 있는 memo_id는 제외합니다.
if RUN_GPT_MINI_FALLBACK:
    queue_df = (
        load_pending_llm_fallback_queue(spark, config, limit_rows=None)
        .where(F.expr(output_scope_sql))
    )

    final_done_keys = load_existing_final_keys(spark, config)
    queue_df = queue_df.join(
        final_done_keys,
        on=["cate_1_depth", "cate_2_depth", "sc_measurement", "memo_id"],
        how="left_anti",
    )

    existing_fallback_keys = (
        spark.table(get_output_table(config, "ml_classification_detail"))
        .where(F.col("prompt_version") == PROMPT_VERSION)
        .where(F.col("taxonomy_version") == TAXONOMY_VERSION)
        .where(F.col("model_version") == LABEL_MODEL_VERSION)
        .where(F.col("classification_stage") == "gpt_mini_fallback")
        .select("cate_1_depth", "cate_2_depth", "sc_measurement", "memo_id")
        .dropDuplicates()
    )

    queue_df = queue_df.join(
        existing_fallback_keys,
        on=["cate_1_depth", "cate_2_depth", "sc_measurement", "memo_id"],
        how="left_anti",
    )

    pending_fallback_count = queue_df.count()
    print("pending_fallback_count =", pending_fallback_count)

    fallback_df = classify_llm_fallback_queue_df(
        spark,
        config,
        queue_df=queue_df,
        model_key=FALLBACK_MODEL_KEY,
        created_by="non_smart_gpt_mini_fallback",
    )
    fallback_count = fallback_df.count()

    if fallback_count > 0:
        fallback_saved_table = save_ml_classification(
            fallback_df,
            config,
            output_table_key="ml_classification_detail",
            mode="append",
        )
    else:
        fallback_saved_table = get_output_table(config, "ml_classification_detail")

    fallback_result = {
        "pending_fallback_count": pending_fallback_count,
        "fallback_count": fallback_count,
        "fallback_saved_table": fallback_saved_table,
    }
else:
    fallback_result = {"skipped": True}

fallback_result

# COMMAND ----------
# 5. 최종 detail 생성: ML auto accept + GPT mini fallback + 저용량 그룹 rule을 통합합니다.
# 전체 버전을 replace하지 않고, 신규 memo_id만 append합니다.
if RUN_13_FINALIZE:
    final_detail_df = build_final_classification_detail_df(
        spark,
        config,
        input_table_key="ml_classification_detail",
    ).where(F.expr(output_scope_sql))

    final_detail_rows = final_detail_df.count()
    final_detail_distinct_memo_ids = final_detail_df.select("memo_id").dropDuplicates().count()
    final_detail_table = get_output_table(config, "classification_detail_final")

    if final_detail_rows > 0:
        final_detail_table = save_final_classification_detail(
            spark,
            config,
            final_detail_df,
            output_table_key="classification_detail_final",
            write_mode="append_new_only",
        )
        final_saved_mode = "append_new_only"
    else:
        final_saved_mode = "no_rows"

    final_result = {
        "final_detail_table": final_detail_table,
        "final_detail_rows": final_detail_rows,
        "final_detail_distinct_memo_ids": final_detail_distinct_memo_ids,
        "final_saved_mode": final_saved_mode,
    }
else:
    final_result = {"skipped": True}

final_result

# COMMAND ----------
# 6. 주제 그룹핑: 이미 그룹핑된 그룹은 skip하고, 신규 topic_pool 그룹만 GPT-5-5로 그룹핑합니다.
if RUN_14_TOPIC_GROUPING:
    topic_group_result = generate_and_save_topic_groups(
        spark,
        config,
        limit_groups=None,
        model_key=DESIGN_MODEL_KEY,
        skip_existing=True,
        write_mode="replace_groups",
    )
else:
    topic_group_result = {"skipped": True}

topic_group_result

# COMMAND ----------
# 7. 실행 후 그룹별 처리 현황 확인
display(spark.sql(status_sql))

final_detail_table = get_output_table(config, "classification_detail_final")
display(
    spark.table(final_detail_table)
    .where(F.col("prompt_version") == PROMPT_VERSION)
    .where(F.col("taxonomy_version") == TAXONOMY_VERSION)
    .where(F.expr(output_scope_sql))
    .groupBy("cate_1_depth", "cate_2_depth", "sc_measurement", "classification_stage", "pred_topic_type")
    .agg(
        F.count("*").alias("final_cnt"),
        F.countDistinct("memo_id").alias("final_distinct_memo_id_cnt"),
    )
    .orderBy("cate_1_depth", "cate_2_depth", "sc_measurement", F.desc("final_cnt"))
)
