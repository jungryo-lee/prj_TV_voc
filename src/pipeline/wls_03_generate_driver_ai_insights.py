# Databricks notebook source
# MAGIC %md
# MAGIC # WLS 03. Generate Driver AI Insights
# MAGIC
# MAGIC Weighted correlation/WLS output tables are condensed into dashboard-ready AI insights.
# MAGIC - `all`: every Y feature is summarized as one overall relationship diagnosis.
# MAGIC - group dimensions: one selected Y feature is summarized as a group-key comparison.
# MAGIC Existing rows with the same statistical evidence hash are reused without another LLM call.

# COMMAND ----------

import importlib
import sys

from pyspark.sql import functions as F

PROJECT_ROOT = "/Workspace/Users/jungryo.lee@lge.com/prj_TV_voc"
SRC_ROOT = f"{PROJECT_ROOT}/src"

if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

import common.config_loader as config_loader
import driver.driver_ai_insight_generator as driver_ai_insight_generator

importlib.reload(config_loader)
importlib.reload(driver_ai_insight_generator)

from common.config_loader import get_output_table, load_config
from driver.driver_ai_insight_generator import generate_and_save_driver_ai_insights

CONFIG_FILE_NAME = "settings_intellytics.yaml"
config = load_config(f"{PROJECT_ROOT}/config/{CONFIG_FILE_NAME}")

AI_INSIGHT_TABLE = get_output_table(config, "driver_ai_insight")
print({"config": CONFIG_FILE_NAME, "driver_ai_insight": AI_INSIGHT_TABLE})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Execution Options

# COMMAND ----------

# `sonnet_46` is the operating default. Use `opus_45` only for selected executive-facing refreshes.
MODEL_KEY = "sonnet_46"

# Generate `all` overview plus group-dimension/Y-feature comparison cards.
# Restrict this list only for a targeted re-generation.
TARGET_GROUP_DIMS = None
TARGET_GROUP_KEYS = None
TARGET_Y_FEATURES = None

INCLUDE_ALL_OVERVIEW = True
INCLUDE_GROUP_COMPARISONS = True
SKIP_UNCHANGED = True

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Generate or Reuse Insights

# COMMAND ----------

result = generate_and_save_driver_ai_insights(
    spark,
    config,
    model_key=MODEL_KEY,
    include_group_overview=INCLUDE_ALL_OVERVIEW,
    include_y_feature_insights=INCLUDE_GROUP_COMPARISONS,
    target_group_dims=TARGET_GROUP_DIMS,
    target_group_keys=TARGET_GROUP_KEYS,
    target_y_features=TARGET_Y_FEATURES,
    skip_unchanged=SKIP_UNCHANGED,
)
result

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Validate Dashboard Output

# COMMAND ----------

display(
    spark.table(AI_INSIGHT_TABLE)
    .orderBy("group_dim", "group_key", "insight_level", "y_feature", F.col("created_at").desc())
    .select(
        "analysis_mode",
        "group_dim",
        "group_key",
        "insight_level",
        "y_feature",
        "analysis_status",
        "confidence_title",
        "summary_title",
        "driver_title",
        "detail_title",
        "adj_r_squared",
        "y_obs",
        "confidence_summary",
        "core_summary",
        "driver_summary",
        "detail_insight",
        "model_endpoint",
        "created_at",
    )
)
