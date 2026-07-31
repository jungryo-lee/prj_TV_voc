# Databricks notebook source
# MAGIC %md
# MAGIC # 16. Driver Analysis - Weighted Correlation / WLS Regression
# MAGIC
# MAGIC Intellytics VOC 원천 데이터를 기준으로 Driver 분석용 입력 테이블을 만들고,
# MAGIC weighted correlation 및 WLS regression 산출물을 생성합니다.

# COMMAND ----------

import sys
import importlib

from pyspark.sql import functions as F

PROJECT_ROOT = "/Workspace/Users/jungryo.lee@lge.com/prj_TV_voc"
SRC_ROOT = f"{PROJECT_ROOT}/src"

if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

import common.config_loader as config_loader
import driver.driver_input_builder as driver_input_builder
import driver.weighted_regression as weighted_regression

importlib.reload(config_loader)
importlib.reload(driver_input_builder)
importlib.reload(weighted_regression)

from common.config_loader import get_output_table, load_config
from driver.driver_input_builder import save_driver_input
from driver.weighted_regression import run_weighted_regression

CONFIG_FILE_NAME = "settings_intellytics.yaml"
config = load_config(f"{PROJECT_ROOT}/config/{CONFIG_FILE_NAME}")

DRIVER_INPUT_TABLE = get_output_table(config, "driver_input")
WEIGHTED_CORR_TABLE = get_output_table(config, "weighted_corr")
WEIGHTED_REGRESSION_TABLE = get_output_table(config, "weighted_regression")
WEIGHTED_REGRESSION_MODEL_TABLE = get_output_table(config, "weighted_regression_model")
WEIGHTED_NETWORK_EDGES_TABLE = get_output_table(config, "weighted_network_edges")
DRIVER_SELECTION_TABLE = get_output_table(config, "driver_selection")

print(
    {
        "config": CONFIG_FILE_NAME,
        "driver_input": DRIVER_INPUT_TABLE,
        "weighted_corr": WEIGHTED_CORR_TABLE,
        "weighted_regression": WEIGHTED_REGRESSION_TABLE,
        "weighted_regression_model": WEIGHTED_REGRESSION_MODEL_TABLE,
        "weighted_network_edges": WEIGHTED_NETWORK_EDGES_TABLE,
        "driver_selection": DRIVER_SELECTION_TABLE,
        "driver_analysis": config.get("driver_analysis", {}),
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build Driver Input

# COMMAND ----------

driver_input_result = save_driver_input(
    spark,
    config,
    mode="overwrite",
)

driver_input_result

# COMMAND ----------

display(
    spark.table(DRIVER_INPUT_TABLE)
    .groupBy("is_lifestyle", "cate_1_depth", "cate_2_depth")
    .agg(
        F.count("*").alias("entity_category_rows"),
        F.countDistinct("entity_id").alias("post_cnt"),
        F.sum("total_count").alias("review_cnt"),
        F.avg("avg_sc").alias("avg_sc"),
    )
    .orderBy("cate_1_depth", "cate_2_depth")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Run Weighted Correlation / WLS

# COMMAND ----------

regression_result = run_weighted_regression(
    spark,
    config,
    input_table_key="driver_input",
    mode="overwrite",
)

regression_result

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Check Outputs

# COMMAND ----------

for table_name in [
    WEIGHTED_CORR_TABLE,
    WEIGHTED_REGRESSION_TABLE,
    WEIGHTED_REGRESSION_MODEL_TABLE,
    WEIGHTED_NETWORK_EDGES_TABLE,
    DRIVER_SELECTION_TABLE,
]:
    print(table_name, spark.table(table_name).count())

# COMMAND ----------

display(
    spark.table(WEIGHTED_REGRESSION_MODEL_TABLE)
    .orderBy("segment_value", F.col("adj_r_squared").desc_nulls_last())
    .limit(50)
)

# COMMAND ----------

display(
    spark.table(DRIVER_SELECTION_TABLE)
    .orderBy("segment_value", "group_dim", "group_key", "y_feature", "driver_rank")
    .limit(200)
)
