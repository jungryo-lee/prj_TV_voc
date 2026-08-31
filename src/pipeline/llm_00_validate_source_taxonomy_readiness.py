# Databricks notebook source
# MAGIC %md
# MAGIC # LLM 00. Validate Source and Taxonomy Readiness
# MAGIC
# MAGIC Run this before incremental topic classification whenever the ODS source
# MAGIC may have category corrections. It keeps category-aware `memo_id` unchanged,
# MAGIC compares a category-independent source identity with the prior snapshot,
# MAGIC and assigns one operating status per category/sentiment group:
# MAGIC
# MAGIC - `reuse`: use existing taxonomy and prototype
# MAGIC - `refresh_prototype`: taxonomy is reusable, but sample/prototype needs refresh
# MAGIC - `review_taxonomy`: category movement or sample Others ratio requires review
# MAGIC - `new_taxonomy`: rule profile or topic pool is missing

# COMMAND ----------
import importlib
import sys

PROJECT_ROOT = "/Workspace/Users/jungryo.lee@lge.com/prj_TV_voc"
SRC_ROOT = f"{PROJECT_ROOT}/src"
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

import common.config_loader as config_loader
import taxonomy.taxonomy_readiness as taxonomy_readiness

importlib.reload(config_loader)
importlib.reload(taxonomy_readiness)

from common.config_loader import load_config, get_output_table
from taxonomy.taxonomy_readiness import run_source_taxonomy_readiness_check

config = load_config(f"{PROJECT_ROOT}/config/settings_intellytics.yaml")
result = run_source_taxonomy_readiness_check(spark, config)

print("source_snapshot_table =", result["source_snapshot_table"])
print("taxonomy_readiness_table =", result["taxonomy_readiness_table"])
print("snapshot_distinct_source_memo_cnt =", result["snapshot_distinct_source_memo_cnt"])
display(result["readiness_summary_df"])

# COMMAND ----------
readiness_table = get_output_table(config, "taxonomy_readiness")
display(
    spark.table(readiness_table)
    .where("run_date = '{}'".format(config["runtime"]["resolved_run_date"]))
    .orderBy("readiness_status", "cate_1_depth", "cate_2_depth", "sc_measurement")
)
