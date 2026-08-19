"""Create the raw-row-preserving final topic-classification view for Tableau."""

from __future__ import annotations

from typing import Any

from pyspark.sql import SparkSession

from common.category_alias import cate_2_alias_sql
from common.config_loader import get_output_table, get_source_table


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
    """Return SQL for an ODS-row-preserving final view with two added label columns."""
    source_table = get_source_table(config, source_table_key)
    final_view = get_source_table(config, final_view_key)
    final_detail_table = get_output_table(config, final_detail_table_key)
    topic_group_table = get_output_table(config, topic_group_table_key)

    cate_1_sql, cate_2_sql = _normalized_category_sql("r")
    aliased_cate_2_sql = cate_2_alias_sql(
        config,
        cate_1_col=cate_1_sql,
        cate_2_col=cate_2_sql,
    )
    memo_norm_sql = _memo_norm_sql("r")
    final_latest_sql = f"""SELECT *
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
  WHERE _rn = 1""" if final_detail_exists else _empty_final_latest_sql()
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

    return f"""
CREATE OR REPLACE VIEW {final_view} AS
WITH source_with_keys AS (
  SELECT
    r.*,
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
), final_latest AS (
  {final_latest_sql}
), topic_group_latest AS (
  {topic_group_latest_sql}
)
SELECT
  s.* EXCEPT (_join_cate_1_depth, _join_cate_2_depth, _join_memo_id),
  f.pred_topic,
  CASE
    WHEN f.pred_topic IS NULL THEN NULL
    WHEN f.pred_topic_type IN ('overall', 'others', 'unclassified') THEN '기타'
    ELSE COALESCE(g.topic_group, '기타')
  END AS topic_group
FROM source_with_keys s
LEFT JOIN final_latest f
  ON s._join_cate_1_depth = f.cate_1_depth
 AND s._join_cate_2_depth = f.cate_2_depth
 AND CAST(s.sc_measurement AS INT) = CAST(f.sc_measurement AS INT)
 AND s._join_memo_id = f.memo_id
LEFT JOIN topic_group_latest g
  ON f.cate_1_depth = g.cate_1_depth
 AND f.cate_2_depth = g.cate_2_depth
 AND CAST(f.sc_measurement AS INT) = CAST(g.sc_measurement AS INT)
 AND f.pred_topic = g.topic
""".strip()


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

    source_row_count = spark.table(source_table).count()
    final_row_count = spark.table(final_view).count()
    if source_row_count != final_row_count:
        raise ValueError(
            "Final view row count must match the ODS source: "
            f"source={source_row_count}, final={final_row_count}"
        )
    return {
        "source_table": source_table,
        "final_view": final_view,
        "source_row_count": source_row_count,
        "final_row_count": final_row_count,
    }
