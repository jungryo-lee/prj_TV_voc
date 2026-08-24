"""Create the raw-row-preserving final topic-classification view for Tableau."""

from __future__ import annotations

from typing import Any

from pyspark.sql import SparkSession

from common.category_alias import cate_2_alias_sql
from common.config_loader import get_output_table, get_source_table


def _incentive_review_filter_sql(alias: str) -> str:
    """Return the final-view eligibility rule shared with the operating pipeline."""
    return f"UPPER(TRIM(CAST({alias}.incentive_review AS STRING))) = 'N'"


def _final_view_source_projection_sql(config: dict[str, Any], alias: str) -> str:
    """Project the fixed Delta Sharing source schema in its published order."""
    columns = (config.get("source", {}) or {}).get("final_tableau_source_columns") or []
    if not columns:
        raise ValueError(
            "source.final_tableau_source_columns must define the fixed shared-view schema."
        )
    return ",\n  ".join(f"{alias}.`{column}`" for column in columns)


def _normalized_category_sql(alias: str) -> tuple[str, str]:
    cate_1 = (
        f"TRIM(REGEXP_REPLACE(TRIM(CAST({alias}.cate_1_depth AS STRING)), "
        "'^[0-9]+[.] ', ''))"
    )
    cate_2 = (
        f"TRIM(REGEXP_REPLACE(REGEXP_REPLACE(TRIM(CAST({alias}.cate_2_depth AS STRING)), "
        "'^[0-9]+-[0-9]+[.] ', ''), '^[0-9]+[.] ', ''))"
    )
    return cate_1, cate_2


def _memo_norm_sql(alias: str) -> str:
    """Return Spark SQL matching ``common.memo_id.normalize_memo_expr`` exactly."""
    return (
        "TRIM(REGEXP_REPLACE(REGEXP_REPLACE("
        f"LOWER(TRANSLATE(COALESCE(CAST({alias}.memo AS STRING), ''), '　', ' ')), "
        # Spark SQL needs two backslashes in the SQL literal for regex \s.
        "'[^0-9a-zA-Z가-힣\\\\s]', ' '), '\\\\s+', ' '))"
    )


def _source_with_keys_sql(config: dict[str, Any], *, source_table_key: str) -> str:
    """Return eligible source rows with the exact keys used by the final-view join."""
    source_table = get_source_table(config, source_table_key)
    cate_1_sql, cate_2_sql = _normalized_category_sql("r")
    aliased_cate_2_sql = cate_2_alias_sql(
        config,
        cate_1_col=cate_1_sql,
        cate_2_col=cate_2_sql,
    )
    memo_norm_sql = _memo_norm_sql("r")
    incentive_review_filter_sql = _incentive_review_filter_sql("r")

    return f"""SELECT
  r.*,
  CASE WHEN {incentive_review_filter_sql} THEN TRUE ELSE FALSE END AS _topic_pipeline_eligible,
  {cate_1_sql} AS _join_cate_1_depth,
  {aliased_cate_2_sql} AS _join_cate_2_depth,
  SHA2(
    CONCAT_WS(
      '||',
      COALESCE(CAST({cate_1_sql} AS STRING), ''),
      COALESCE(CAST({aliased_cate_2_sql} AS STRING), ''),
      COALESCE(CAST(CAST(r.sc_measurement AS INT) AS STRING), ''),
      {memo_norm_sql}
    ),
    256
  ) AS _join_memo_id
FROM {source_table} r
WHERE {incentive_review_filter_sql}"""


def _final_latest_sql(final_detail_table: str, *, exists: bool) -> str:
    """Return the latest immutable label for each classified memo key."""
    if not exists:
        return _empty_final_latest_sql()

    return f"""SELECT *
FROM (
  SELECT
    f.cate_1_depth,
    f.cate_2_depth,
    f.sc_measurement,
    f.memo_id,
    f.pred_topic,
    f.pred_topic_type,
    f.created_at,
    ROW_NUMBER() OVER (
      PARTITION BY f.cate_1_depth, f.cate_2_depth, f.sc_measurement, f.memo_id
      ORDER BY f.created_at DESC, f.run_id DESC
    ) AS _rn
  FROM {final_detail_table} f
) ranked
WHERE _rn = 1"""


def build_final_classification_view_sql(
    config: dict[str, Any],
    *,
    source_table_key: str = "raw_ods_table",
    final_view_key: str = "final_tableau_view",
    final_detail_table_key: str = "classification_detail_final",
    topic_group_table_key: str = "topic_group",
    final_detail_exists: bool = True,
    topic_group_exists: bool = True,
) -> str:
    """Return SQL for an incentive-eligible source view with two added label columns."""
    source_table = get_source_table(config, source_table_key)
    final_view = get_source_table(config, final_view_key)
    final_detail_table = get_output_table(config, final_detail_table_key)
    topic_group_table = get_output_table(config, topic_group_table_key)

    source_with_keys_sql = _source_with_keys_sql(
        config,
        source_table_key=source_table_key,
    )
    final_latest_sql = _final_latest_sql(
        final_detail_table,
        exists=final_detail_exists,
    )
    topic_group_latest_sql = f"""SELECT *
  FROM (
    SELECT
      g.cate_1_depth,
      g.cate_2_depth,
      g.sc_measurement,
      g.topic,
      g.topic_group,
      g.created_at,
      ROW_NUMBER() OVER (
        PARTITION BY g.cate_1_depth, g.cate_2_depth, g.sc_measurement, g.topic
        ORDER BY g.created_at DESC, g.run_id DESC
      ) AS _rn
    FROM {topic_group_table} g
  ) ranked
  WHERE _rn = 1""" if topic_group_exists else _empty_topic_group_sql()
    source_projection_sql = _final_view_source_projection_sql(config, "s")

    return f"""
CREATE OR REPLACE VIEW {final_view} AS
WITH source_with_keys AS (
  {source_with_keys_sql}
), final_latest AS (
  {final_latest_sql}
), topic_group_latest AS (
  {topic_group_latest_sql}
)
SELECT
  {source_projection_sql},
  f.pred_topic,
  CASE
    WHEN f.pred_topic IS NULL THEN NULL
    WHEN f.pred_topic_type IN ('overall', 'others', 'unclassified') THEN '기타'
    ELSE COALESCE(g.topic_group, '기타')
  END AS topic_group
FROM source_with_keys s
LEFT JOIN final_latest f
  ON s._topic_pipeline_eligible
 AND s._join_cate_1_depth = f.cate_1_depth
 AND s._join_cate_2_depth = f.cate_2_depth
 AND CAST(s.sc_measurement AS INT) = CAST(f.sc_measurement AS INT)
 AND s._join_memo_id = f.memo_id
LEFT JOIN topic_group_latest g
  ON f.cate_1_depth = g.cate_1_depth
 AND f.cate_2_depth = g.cate_2_depth
 AND CAST(f.sc_measurement AS INT) = CAST(g.sc_measurement AS INT)
 AND f.pred_topic = g.topic
""".strip()


def build_final_mapping_coverage_report(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    source_table_key: str = "raw_ods_table",
    final_detail_table_key: str = "classification_detail_final",
) -> dict[str, Any]:
    """Measure reusable final labels before refreshing source-derived views.

    The report never changes a view or table. It uses the same category cleanup,
    alias rules, and memo ID expression as the final Tableau view, so its match
    rate is the expected label coverage after a refresh.
    """
    source_table = get_source_table(config, source_table_key)
    final_detail_table = get_output_table(config, final_detail_table_key)
    source_with_keys_sql = _source_with_keys_sql(
        config,
        source_table_key=source_table_key,
    )
    final_latest_sql = _final_latest_sql(
        final_detail_table,
        exists=spark.catalog.tableExists(final_detail_table),
    )

    common_cte = f"""
WITH source_with_keys AS (
  {source_with_keys_sql}
), source_keys AS (
  SELECT DISTINCT
    _join_cate_1_depth,
    _join_cate_2_depth,
    CAST(sc_measurement AS INT) AS sc_measurement,
    _join_memo_id
  FROM source_with_keys
), final_latest AS (
  {final_latest_sql}
), final_keys AS (
  SELECT DISTINCT
    cate_1_depth,
    cate_2_depth,
    CAST(sc_measurement AS INT) AS sc_measurement,
    memo_id
  FROM final_latest
  WHERE pred_topic IS NOT NULL
), matched_keys AS (
  SELECT s.*
  FROM source_keys s
  INNER JOIN final_keys f
    ON s._join_cate_1_depth = f.cate_1_depth
   AND s._join_cate_2_depth = f.cate_2_depth
   AND s.sc_measurement = f.sc_measurement
   AND s._join_memo_id = f.memo_id
), matched_source_rows AS (
  SELECT s.*
  FROM source_with_keys s
  INNER JOIN matched_keys m
    ON s._join_cate_1_depth = m._join_cate_1_depth
   AND s._join_cate_2_depth = m._join_cate_2_depth
   AND CAST(s.sc_measurement AS INT) = m.sc_measurement
   AND s._join_memo_id = m._join_memo_id
)
"""

    summary_df = spark.sql(
        common_cte
        + f"""
SELECT
  '{source_table}' AS source_table,
  '{final_detail_table}' AS final_detail_table,
  (SELECT COUNT(*) FROM source_with_keys) AS source_raw_rows,
  (SELECT COUNT(*) FROM source_keys) AS source_distinct_memo_id_cnt,
  (SELECT COUNT(*) FROM final_keys) AS historical_final_distinct_memo_id_cnt,
  (SELECT COUNT(*) FROM matched_keys) AS reusable_distinct_memo_id_cnt,
  (SELECT COUNT(*) FROM matched_source_rows) AS reusable_raw_rows,
  (SELECT COUNT(*) FROM source_keys) - (SELECT COUNT(*) FROM matched_keys)
    AS new_or_unclassified_distinct_memo_id_cnt,
  (SELECT COUNT(*) FROM source_with_keys) - (SELECT COUNT(*) FROM matched_source_rows)
    AS new_or_unclassified_raw_rows,
  ROUND(
    100.0 * (SELECT COUNT(*) FROM matched_keys)
      / NULLIF((SELECT COUNT(*) FROM source_keys), 0),
    2
  ) AS source_distinct_memo_reuse_ratio_pct,
  ROUND(
    100.0 * (SELECT COUNT(*) FROM matched_source_rows)
      / NULLIF((SELECT COUNT(*) FROM source_with_keys), 0),
    2
  ) AS source_raw_row_coverage_ratio_pct,
  (SELECT COUNT(*) FROM source_with_keys
   WHERE is_lifestyle = 'N'
     AND _topic_pipeline_eligible
     AND CAST(sc_measurement AS INT) IN (-1, 1))
    AS pipeline_eligible_raw_rows,
  (SELECT COUNT(DISTINCT _join_memo_id) FROM source_with_keys
   WHERE is_lifestyle = 'N'
     AND _topic_pipeline_eligible
     AND CAST(sc_measurement AS INT) IN (-1, 1))
    AS pipeline_eligible_distinct_memo_id_cnt,
  (SELECT COUNT(*) FROM matched_source_rows
   WHERE is_lifestyle = 'N'
     AND _topic_pipeline_eligible
     AND CAST(sc_measurement AS INT) IN (-1, 1))
    AS reusable_eligible_raw_rows,
  (SELECT COUNT(DISTINCT _join_memo_id) FROM matched_source_rows
   WHERE is_lifestyle = 'N'
     AND _topic_pipeline_eligible
     AND CAST(sc_measurement AS INT) IN (-1, 1))
    AS reusable_eligible_distinct_memo_id_cnt,
  ROUND(
    100.0 * (SELECT COUNT(*) FROM matched_source_rows
             WHERE is_lifestyle = 'N'
               AND _topic_pipeline_eligible
               AND CAST(sc_measurement AS INT) IN (-1, 1))
      / NULLIF((SELECT COUNT(*) FROM source_with_keys
                WHERE is_lifestyle = 'N'
                  AND _topic_pipeline_eligible
                  AND CAST(sc_measurement AS INT) IN (-1, 1)), 0),
    2
  ) AS pipeline_eligible_raw_row_coverage_ratio_pct
"""
    )

    group_df = spark.sql(
        common_cte
        + """
SELECT
  s._join_cate_1_depth AS cate_1_depth,
  s._join_cate_2_depth AS cate_2_depth,
  CAST(s.sc_measurement AS INT) AS sc_measurement,
  COUNT(*) AS source_raw_rows,
  COUNT(DISTINCT s._join_memo_id) AS source_distinct_memo_id_cnt,
  COUNT(DISTINCT CASE
    WHEN s.is_lifestyle = 'N' AND CAST(s.sc_measurement AS INT) IN (-1, 1)
     AND s._topic_pipeline_eligible
    THEN s._join_memo_id
  END) AS pipeline_eligible_distinct_memo_id_cnt,
  COUNT(DISTINCT m._join_memo_id) AS reusable_distinct_memo_id_cnt,
  COUNT(m._join_memo_id) AS reusable_raw_rows,
  COUNT(DISTINCT s._join_memo_id) - COUNT(DISTINCT m._join_memo_id)
    AS new_or_unclassified_distinct_memo_id_cnt,
  ROUND(
    100.0 * COUNT(DISTINCT m._join_memo_id)
      / NULLIF(COUNT(DISTINCT s._join_memo_id), 0),
    2
  ) AS source_distinct_memo_reuse_ratio_pct,
  ROUND(
    100.0 * COUNT(m._join_memo_id) / NULLIF(COUNT(*), 0),
    2
  ) AS source_raw_row_coverage_ratio_pct
FROM source_with_keys s
LEFT JOIN matched_keys m
  ON s._join_cate_1_depth = m._join_cate_1_depth
 AND s._join_cate_2_depth = m._join_cate_2_depth
 AND CAST(s.sc_measurement AS INT) = m.sc_measurement
 AND s._join_memo_id = m._join_memo_id
GROUP BY
  s._join_cate_1_depth,
  s._join_cate_2_depth,
  CAST(s.sc_measurement AS INT)
ORDER BY source_distinct_memo_reuse_ratio_pct ASC, source_raw_rows DESC
"""
    )

    return {
        "source_table": source_table,
        "final_detail_table": final_detail_table,
        "summary_df": summary_df,
        "group_df": group_df,
    }


def _empty_final_latest_sql() -> str:
    """Return a typed empty CTE when no classification result exists yet."""
    return """SELECT
  CAST(NULL AS STRING) AS cate_1_depth,
  CAST(NULL AS STRING) AS cate_2_depth,
  CAST(NULL AS INT) AS sc_measurement,
  CAST(NULL AS STRING) AS memo_id,
  CAST(NULL AS STRING) AS pred_topic,
  CAST(NULL AS STRING) AS pred_topic_type
WHERE 1 = 0"""


def _empty_topic_group_sql() -> str:
    """Return a typed empty CTE when no topic-group mapping exists yet."""
    return """SELECT
  CAST(NULL AS STRING) AS cate_1_depth,
  CAST(NULL AS STRING) AS cate_2_depth,
  CAST(NULL AS INT) AS sc_measurement,
  CAST(NULL AS STRING) AS topic,
  CAST(NULL AS STRING) AS topic_group
WHERE 1 = 0"""


def create_or_replace_final_classification_view(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    replace_existing_table_with_view: bool = True,
) -> dict[str, Any]:
    """Create the final view, replacing a legacy object at the target only when allowed."""
    final_view = get_source_table(config, "final_tableau_view")
    source_table = get_source_table(config, "raw_ods_table")
    final_detail_table = get_output_table(config, "classification_detail_final")
    topic_group_table = get_output_table(config, "topic_group")
    create_sql = build_final_classification_view_sql(
        config,
        final_detail_exists=spark.catalog.tableExists(final_detail_table),
        topic_group_exists=spark.catalog.tableExists(topic_group_table),
    )
    try:
        object_type = spark.catalog.getTable(final_view).tableType.upper()
    except Exception:
        object_type = None

    # Delta Sharing blocks DROP/CREATE OR REPLACE for an already shared view.
    # ALTER VIEW changes only its definition and keeps the shared object intact.
    if object_type == "VIEW":
        create_prefix = f"CREATE OR REPLACE VIEW {final_view} AS"
        alter_prefix = f"ALTER VIEW {final_view} AS"
        alter_sql = create_sql.replace(create_prefix, alter_prefix, 1)
        try:
            spark.sql(alter_sql)
        except Exception as error:
            raise RuntimeError(
                f"Failed to alter final Tableau view {final_view}. "
                "The shared view was left unchanged."
            ) from error
    else:
        try:
            spark.sql(create_sql)
        except Exception as initial_error:
            if not replace_existing_table_with_view:
                raise

            # A legacy managed table can be replaced only when it is not shared.
            try:
                object_type = spark.catalog.getTable(final_view).tableType.upper()
            except Exception:
                object_type = None

            if object_type in {"MANAGED", "EXTERNAL"}:
                spark.sql(f"DROP TABLE IF EXISTS {final_view}")
                spark.sql(create_sql)
            else:
                raise RuntimeError(
                    f"Failed to create final Tableau view {final_view}. "
                    f"Existing object type could not be replaced: {object_type!r}."
                ) from initial_error

    eligible_source_count = spark.sql(
        f"""SELECT COUNT(*) AS row_count
        FROM {source_table} r
        WHERE {_incentive_review_filter_sql('r')}"""
    ).first()["row_count"]
    final_row_count = spark.table(final_view).count()
    if eligible_source_count != final_row_count:
        raise ValueError(
            "Final view row count must match the incentive-eligible ODS source: "
            f"eligible_source={eligible_source_count}, final={final_row_count}"
        )
    return {
        "source_table": source_table,
        "final_view": final_view,
        "source_row_count": eligible_source_count,
        "final_row_count": final_row_count,
    }
