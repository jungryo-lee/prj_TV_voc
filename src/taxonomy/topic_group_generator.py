"""Generate and apply higher-level topic groups for Tableau reporting."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from common.config_loader import get_output_table
from common.llm_client import get_llm_client


TOPIC_GROUP_SCHEMA = T.StructType(
    [
        T.StructField("cate_1_depth", T.StringType(), True),
        T.StructField("cate_2_depth", T.StringType(), True),
        T.StructField("sc_measurement", T.IntegerType(), True),
        T.StructField("topic", T.StringType(), True),
        T.StructField("topic_description", T.StringType(), True),
        T.StructField("topic_order", T.IntegerType(), True),
        T.StructField("topic_group", T.StringType(), True),
        T.StructField("topic_group_order", T.IntegerType(), True),
        T.StructField("topic_group_description", T.StringType(), True),
        T.StructField("grouping_reason", T.StringType(), True),
        T.StructField("is_special_group", T.BooleanType(), True),
        T.StructField("run_id", T.StringType(), True),
        T.StructField("run_date", T.StringType(), True),
        T.StructField("pipeline_stage", T.StringType(), True),
        T.StructField("prompt_version", T.StringType(), True),
        T.StructField("taxonomy_version", T.StringType(), True),
        T.StructField("model_version", T.StringType(), True),
        T.StructField("pipeline_version", T.StringType(), True),
        T.StructField("is_latest", T.BooleanType(), True),
        T.StructField("created_at", T.StringType(), True),
        T.StructField("created_by", T.StringType(), True),
    ]
)


GROUP_KEYS = ["cate_1_depth", "cate_2_depth", "sc_measurement"]
SPECIAL_TOPICS = {
    "",
    "기타",
    "미분류",
    "미분류(리뷰 100개미만)",
    "전반적 긍정",
    "전반적 부정",
    "LLM_FALLBACK_REQUIRED",
}


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    """Return topic group settings with safe defaults."""
    cfg = config.get("topic_group", {}) or {}
    return {
        "input_table_key": cfg.get("input_table_key", "topic_pool"),
        "output_table_key": cfg.get("output_table_key", "topic_group"),
        "model_key": cfg.get("model_key", "gpt_55"),
        "min_groups": int(cfg.get("min_groups", 3)),
        "max_groups": int(cfg.get("max_groups", 5)),
        "special_group_name": str(cfg.get("special_group_name", "기타")),
    }


def _runtime_value(config: dict[str, Any], key: str, default: str = "") -> str:
    return str((config.get("runtime", {}) or {}).get(key, default) or default)


def _version_value(config: dict[str, Any], key: str, default: str = "") -> str:
    return str((config.get("version", {}) or {}).get(key, default) or default)


def _model_version(config: dict[str, Any], model_key: str) -> str:
    return str(
        ((config.get("llm", {}) or {}).get("models", {}) or {})
        .get(model_key, {})
        .get("model_version", model_key)
    )


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _is_special_topic(topic: str) -> bool:
    normalized = _clean_text(topic)
    return normalized in SPECIAL_TOPICS or normalized.startswith("전반적 ")


def load_latest_topic_pool_for_group(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
) -> list[dict[str, Any]]:
    """Load latest topic pool rows for one category/sentiment group."""
    cfg = _cfg(config)
    table_name = get_output_table(config, cfg["input_table_key"])
    df = (
        spark.table(table_name)
        .where(F.col("cate_1_depth") == cate_1_depth)
        .where(F.col("cate_2_depth") == cate_2_depth)
        .where(F.col("sc_measurement") == int(sc_measurement))
        .where(F.col("prompt_version") == _version_value(config, "prompt_version"))
        .where(F.col("taxonomy_version") == _version_value(config, "taxonomy_version"))
    )

    latest_window = Window.partitionBy(
        "cate_1_depth",
        "cate_2_depth",
        "sc_measurement",
        "topic",
        "prompt_version",
        "taxonomy_version",
    ).orderBy(
        F.col("is_latest").desc_nulls_last(),
        F.col("created_at").desc_nulls_last(),
        F.col("run_id").desc_nulls_last(),
    )

    rows = (
        df.withColumn("_rn", F.row_number().over(latest_window))
        .where(F.col("_rn") == 1)
        .drop("_rn")
        .orderBy(F.col("topic_order").asc_nulls_last(), F.col("topic").asc())
        .collect()
    )
    return [row.asDict(recursive=True) for row in rows]


def _build_topic_group_prompt(
    *,
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    topics: list[dict[str, Any]],
    min_groups: int,
    max_groups: int,
) -> tuple[str, str]:
    """Build GPT prompt for topic grouping."""
    topic_lines = []
    for idx, row in enumerate(topics, start=1):
        topic_lines.append(
            {
                "topic_no": idx,
                "topic": row["topic"],
                "description": row.get("description", ""),
            }
        )

    system_prompt = """
You are a senior VOC taxonomy architect.
Group detailed VOC topics into a small number of higher-level reporting groups.
Return JSON only. Do not include markdown.
""".strip()

    user_prompt = f"""
Category group:
- cate_1_depth: {cate_1_depth}
- cate_2_depth: {cate_2_depth}
- sc_measurement: {sc_measurement}

Task:
- Group the given detailed topics into {min_groups} to {max_groups} higher-level topic groups.
- Use topic names and descriptions semantically; do not group only by identical words.
- Each detailed topic must belong to exactly one topic_group.
- topic_group names should be concise Korean labels suitable for Tableau/reporting.
- Do not create a group for overall/others/unclassified here. Those are handled separately as "기타".
- Avoid generic group names unless necessary.

Detailed topics:
{json.dumps(topic_lines, ensure_ascii=False, indent=2)}

Return exactly this JSON shape:
{{
  "topic_groups": [
    {{
      "topic_group": "상위 그룹명",
      "description": "이 그룹에 묶인 공통 의미",
      "topics": [
        {{
          "topic": "원본 topic명",
          "reason": "왜 이 그룹에 속하는지"
        }}
      ]
    }}
  ]
}}
""".strip()
    return system_prompt, user_prompt


def _normalize_grouping_response(
    response: dict[str, Any],
    topics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate and flatten GPT topic-group response."""
    topic_by_name = {_clean_text(row.get("topic")): row for row in topics}
    assigned: set[str] = set()
    rows: list[dict[str, Any]] = []
    groups = response.get("topic_groups") or []
    if not isinstance(groups, list):
        groups = []

    for group_order, group in enumerate(groups, start=1):
        if not isinstance(group, dict):
            continue
        topic_group = _clean_text(group.get("topic_group"))
        description = _clean_text(group.get("description"))
        if not topic_group:
            continue
        for item in group.get("topics") or []:
            if not isinstance(item, dict):
                continue
            topic = _clean_text(item.get("topic"))
            if topic not in topic_by_name or topic in assigned:
                continue
            source = topic_by_name[topic]
            assigned.add(topic)
            rows.append(
                {
                    "topic": topic,
                    "topic_description": _clean_text(source.get("description")),
                    "topic_order": int(source.get("topic_order") or 0),
                    "topic_group": topic_group,
                    "topic_group_order": int(group_order),
                    "topic_group_description": description,
                    "grouping_reason": _clean_text(item.get("reason")),
                    "is_special_group": False,
                }
            )

    # Keep the pipeline complete even if GPT misses a topic.
    fallback_order = len(groups) + 1
    for topic, source in topic_by_name.items():
        if topic in assigned:
            continue
        rows.append(
            {
                "topic": topic,
                "topic_description": _clean_text(source.get("description")),
                "topic_order": int(source.get("topic_order") or 0),
                "topic_group": "기타",
                "topic_group_order": int(fallback_order),
                "topic_group_description": "LLM grouping에서 누락되어 기타로 보정된 주제",
                "grouping_reason": "missing_from_llm_grouping_response",
                "is_special_group": True,
            }
        )

    return rows


def generate_topic_group_for_group(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    model_key: str | None = None,
) -> dict[str, Any]:
    """Generate topic groups for one category/sentiment group using GPT-5-5."""
    cfg = _cfg(config)
    resolved_model_key = model_key or cfg["model_key"]
    special_group_name = cfg["special_group_name"]
    topic_rows = load_latest_topic_pool_for_group(
        spark,
        config,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
    )
    if not topic_rows:
        raise ValueError(
            f"topic_pool not found: {cate_1_depth} / {cate_2_depth} / {sc_measurement}"
        )

    normal_topics = [
        {
            "topic": _clean_text(row.get("topic")),
            "description": _clean_text(row.get("description")),
            "topic_order": int(row.get("topic_order") or 0),
        }
        for row in topic_rows
        if not _is_special_topic(_clean_text(row.get("topic")))
    ]
    special_topics = [
        {
            "topic": _clean_text(row.get("topic")),
            "topic_description": _clean_text(row.get("description")),
            "topic_order": int(row.get("topic_order") or 0),
            "topic_group": special_group_name,
            "topic_group_order": 999,
            "topic_group_description": "전반적/기타/미분류성 주제를 통합 관리하는 그룹",
            "grouping_reason": "special_topic_for_reporting",
            "is_special_group": True,
        }
        for row in topic_rows
        if _is_special_topic(_clean_text(row.get("topic")))
    ]

    if len(normal_topics) <= 1:
        grouped_topics = [
            {
                **row,
                "topic_group": _clean_text(row["topic"]) or special_group_name,
                "topic_group_order": 1,
                "topic_group_description": _clean_text(row.get("description")),
                "grouping_reason": "single_topic_group",
                "is_special_group": False,
            }
            for row in normal_topics
        ]
    else:
        system_prompt, user_prompt = _build_topic_group_prompt(
            cate_1_depth=cate_1_depth,
            cate_2_depth=cate_2_depth,
            sc_measurement=sc_measurement,
            topics=normal_topics,
            min_groups=min(cfg["min_groups"], len(normal_topics)),
            max_groups=min(cfg["max_groups"], len(normal_topics)),
        )
        llm = get_llm_client(config=config, model_key=resolved_model_key)
        response = llm.converse_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1800,
        )
        grouped_topics = _normalize_grouping_response(response, normal_topics)

    return {
        "cate_1_depth": cate_1_depth,
        "cate_2_depth": cate_2_depth,
        "sc_measurement": int(sc_measurement),
        "model_key": resolved_model_key,
        "topics": grouped_topics + special_topics,
    }


def build_topic_group_rows(
    results: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    created_by: str = "topic_group_generator",
    pipeline_stage: str = "topic_group",
    is_latest: bool = True,
) -> list[dict[str, Any]]:
    """Convert topic group results into table-ready rows."""
    run_id = _runtime_value(config, "resolved_run_id")
    run_date = _runtime_value(config, "resolved_run_date")
    prompt_version = _version_value(config, "prompt_version")
    taxonomy_version = _version_value(config, "taxonomy_version")
    pipeline_version = _version_value(config, "pipeline_version")
    created_at = datetime.utcnow().isoformat(timespec="seconds")

    rows: list[dict[str, Any]] = []
    for result in results:
        model_key = str(result.get("model_key") or _cfg(config)["model_key"])
        model_version = _model_version(config, model_key)
        for topic in result.get("topics") or []:
            rows.append(
                {
                    "cate_1_depth": _clean_text(result.get("cate_1_depth")),
                    "cate_2_depth": _clean_text(result.get("cate_2_depth")),
                    "sc_measurement": int(result.get("sc_measurement")),
                    "topic": _clean_text(topic.get("topic")),
                    "topic_description": _clean_text(topic.get("topic_description")),
                    "topic_order": int(topic.get("topic_order") or 0),
                    "topic_group": _clean_text(topic.get("topic_group")),
                    "topic_group_order": int(topic.get("topic_group_order") or 999),
                    "topic_group_description": _clean_text(
                        topic.get("topic_group_description")
                    ),
                    "grouping_reason": _clean_text(topic.get("grouping_reason")),
                    "is_special_group": bool(topic.get("is_special_group", False)),
                    "run_id": run_id,
                    "run_date": run_date,
                    "pipeline_stage": pipeline_stage,
                    "prompt_version": prompt_version,
                    "taxonomy_version": taxonomy_version,
                    "model_version": model_version,
                    "pipeline_version": pipeline_version,
                    "is_latest": bool(is_latest),
                    "created_at": created_at,
                    "created_by": created_by,
                }
            )
    return rows


def build_topic_group_spark_df(
    spark: SparkSession,
    results: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    created_by: str = "topic_group_generator",
) -> DataFrame:
    """Build Spark DataFrame for topic group writes."""
    rows = build_topic_group_rows(results, config, created_by=created_by)
    return spark.createDataFrame(rows, schema=TOPIC_GROUP_SCHEMA)


def _delete_existing_group_rows(
    spark: SparkSession,
    table_name: str,
    df: DataFrame,
) -> None:
    """Delete topic group rows for the same group/version/model."""
    if not spark.catalog.tableExists(table_name):
        return
    keys_df = (
        df.select(
            *GROUP_KEYS,
            "prompt_version",
            "taxonomy_version",
            "model_version",
        )
        .dropDuplicates()
    )
    keys_df.createOrReplaceTempView("_tmp_topic_group_keys")
    spark.sql(
        f"""
        DELETE FROM {table_name}
        WHERE EXISTS (
            SELECT 1
            FROM _tmp_topic_group_keys src
            WHERE {table_name}.cate_1_depth = src.cate_1_depth
              AND {table_name}.cate_2_depth = src.cate_2_depth
              AND {table_name}.sc_measurement = src.sc_measurement
              AND {table_name}.prompt_version = src.prompt_version
              AND {table_name}.taxonomy_version = src.taxonomy_version
              AND {table_name}.model_version = src.model_version
        )
        """
    )


def save_topic_groups(
    spark: SparkSession,
    config: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    output_table_key: str | None = None,
    write_mode: str = "replace_groups",
) -> str:
    """Save topic group mapping rows."""
    if not results:
        raise ValueError("results must not be empty.")
    cfg = _cfg(config)
    table_name = get_output_table(config, output_table_key or cfg["output_table_key"])
    df = build_topic_group_spark_df(spark, results, config)

    if write_mode == "overwrite":
        df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
        return table_name
    if write_mode == "replace_groups":
        _delete_existing_group_rows(spark, table_name, df)
        df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table_name)
        return table_name
    if write_mode == "append":
        df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table_name)
        return table_name
    raise ValueError(f"Unsupported write_mode: {write_mode}")


def load_latest_topic_group_df(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    topic_group_table_key: str = "topic_group",
    model_key: str | None = None,
) -> DataFrame:
    """Load one latest topic_group row per group/topic for the active version."""
    cfg = _cfg(config)
    table_name = get_output_table(config, topic_group_table_key or cfg["output_table_key"])
    resolved_model_key = model_key or cfg["model_key"]
    resolved_model_version = _model_version(config, resolved_model_key)

    df = (
        spark.table(table_name)
        .where(F.col("prompt_version") == _version_value(config, "prompt_version"))
        .where(F.col("taxonomy_version") == _version_value(config, "taxonomy_version"))
        .where(F.col("model_version") == resolved_model_version)
    )

    latest_window = Window.partitionBy(
        *GROUP_KEYS,
        "topic",
        "prompt_version",
        "taxonomy_version",
        "model_version",
    ).orderBy(
        F.col("is_latest").desc_nulls_last(),
        F.col("created_at").desc_nulls_last(),
        F.col("run_id").desc_nulls_last(),
    )

    return (
        df.withColumn("_topic_group_rn", F.row_number().over(latest_window))
        .where(F.col("_topic_group_rn") == 1)
        .drop("_topic_group_rn")
    )


def _filter_existing_topic_group_keys(
    spark: SparkSession,
    config: dict[str, Any],
    groups: list[tuple[str, str, int]],
    *,
    model_key: str | None = None,
) -> list[tuple[str, str, int]]:
    """Return only groups without an active topic_group mapping."""
    if not groups:
        return groups

    table_name = get_output_table(config, _cfg(config)["output_table_key"])
    if not spark.catalog.tableExists(table_name):
        return groups

    existing_rows = (
        load_latest_topic_group_df(spark, config, model_key=model_key)
        .select(*GROUP_KEYS)
        .dropDuplicates()
        .collect()
    )
    existing = {
        (
            str(row["cate_1_depth"]),
            str(row["cate_2_depth"]),
            int(row["sc_measurement"]),
        )
        for row in existing_rows
    }
    return [group for group in groups if group not in existing]


def load_topic_pool_groups(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
) -> list[tuple[str, str, int]]:
    """List topic-pool groups available for grouping."""
    table_name = get_output_table(config, _cfg(config)["input_table_key"])
    df = (
        spark.table(table_name)
        .where(F.col("prompt_version") == _version_value(config, "prompt_version"))
        .where(F.col("taxonomy_version") == _version_value(config, "taxonomy_version"))
    )
    if cate_1_depth is not None:
        df = df.where(F.col("cate_1_depth") == cate_1_depth)
    if cate_2_depth is not None:
        df = df.where(F.col("cate_2_depth") == cate_2_depth)
    if sc_measurement is not None:
        df = df.where(F.col("sc_measurement") == int(sc_measurement))

    rows = df.select(*GROUP_KEYS).dropDuplicates().orderBy(*GROUP_KEYS).collect()
    return [
        (str(row["cate_1_depth"]), str(row["cate_2_depth"]), int(row["sc_measurement"]))
        for row in rows
    ]


def generate_and_save_topic_groups(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    cate_1_depth: str | None = None,
    cate_2_depth: str | None = None,
    sc_measurement: int | None = None,
    limit_groups: int | None = None,
    model_key: str | None = None,
    skip_existing: bool = True,
    write_mode: str = "replace_groups",
) -> dict[str, Any]:
    """Generate topic groups for selected topic-pool groups and save them."""
    all_groups = load_topic_pool_groups(
        spark,
        config,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
    )
    groups = (
        _filter_existing_topic_group_keys(
            spark,
            config,
            all_groups,
            model_key=model_key,
        )
        if skip_existing
        else all_groups
    )
    if limit_groups is not None:
        groups = groups[: int(limit_groups)]

    results = []
    for idx, (cate1, cate2, sc) in enumerate(groups, start=1):
        print(f"[topic_group] {idx}/{len(groups)} | {cate1} | {cate2} | {sc}")
        results.append(
            generate_topic_group_for_group(
                spark,
                config,
                cate_1_depth=cate1,
                cate_2_depth=cate2,
                sc_measurement=sc,
                model_key=model_key,
            )
        )

    table_name = save_topic_groups(
        spark,
        config,
        results,
        write_mode=write_mode,
    ) if results else get_output_table(config, _cfg(config)["output_table_key"])

    return {
        "table_name": table_name,
        "available_group_count": len(all_groups),
        "skipped_existing_group_count": len(all_groups) - len(groups) if skip_existing else 0,
        "group_count": len(groups),
        "topic_group_row_count": sum(len(result.get("topics") or []) for result in results),
    }


def build_tableau_grouped_final_df(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    tableau_table_key: str = "classification_tableau_final",
    topic_group_table_key: str = "topic_group",
) -> DataFrame:
    """Attach topic_group columns to the final Tableau classification table."""
    special_group_name = _cfg(config)["special_group_name"]
    tableau_table = get_output_table(config, tableau_table_key)
    topic_group_table = get_output_table(config, topic_group_table_key)

    tableau_df = (
        spark.table(tableau_table)
        .where(F.col("prompt_version") == _version_value(config, "prompt_version"))
        .where(F.col("taxonomy_version") == _version_value(config, "taxonomy_version"))
    )
    group_df = (
        load_latest_topic_group_df(
            spark,
            config,
            topic_group_table_key=topic_group_table_key,
        )
        .select(
            *GROUP_KEYS,
            F.col("topic").alias("_group_topic"),
            "topic_group",
            "topic_group_order",
            "topic_group_description",
            "grouping_reason",
            "is_special_group",
        )
    )

    t = tableau_df.alias("t")
    g = group_df.alias("g")
    joined_df = t.join(
        g,
        on=[
            F.col("t.cate_1_depth") == F.col("g.cate_1_depth"),
            F.col("t.cate_2_depth") == F.col("g.cate_2_depth"),
            F.col("t.sc_measurement") == F.col("g.sc_measurement"),
            F.col("t.pred_topic") == F.col("g._group_topic"),
        ],
        how="left",
    )

    return joined_df.select(
        *[F.col(f"t.`{col}`") for col in tableau_df.columns],
        F.when(
            F.col("t.pred_topic_type").isin("overall", "others", "llm_fallback")
            | F.col("t.pred_topic").isin(list(SPECIAL_TOPICS))
            | F.col("g.topic_group").isNull(),
            F.lit(special_group_name),
        )
        .otherwise(F.col("g.topic_group"))
        .alias("topic_group"),
        F.when(
            F.col("t.pred_topic_type").isin("overall", "others", "llm_fallback")
            | F.col("t.pred_topic").isin(list(SPECIAL_TOPICS))
            | F.col("g.topic_group").isNull(),
            F.lit(999),
        )
        .otherwise(F.col("g.topic_group_order"))
        .alias("topic_group_order"),
        F.coalesce(
            F.col("g.topic_group_description"),
            F.lit("전반적/기타/미분류 또는 매핑되지 않은 주제"),
        ).alias("topic_group_description"),
        F.coalesce(F.col("g.grouping_reason"), F.lit("special_or_unmapped_topic")).alias(
            "topic_grouping_reason"
        ),
        F.when(
            F.col("t.pred_topic_type").isin("overall", "others", "llm_fallback")
            | F.col("t.pred_topic").isin(list(SPECIAL_TOPICS))
            | F.col("g.is_special_group").isNull(),
            F.lit(True),
        )
        .otherwise(F.col("g.is_special_group"))
        .alias("is_special_topic_group"),
    )


def save_tableau_grouped_final(
    spark: SparkSession,
    config: dict[str, Any],
    grouped_df: DataFrame,
    *,
    output_table_key: str = "classification_tableau_grouped_final",
    write_mode: str = "replace_version",
) -> str:
    """Save Tableau final table with topic_group columns."""
    table_name = get_output_table(config, output_table_key)
    if write_mode == "overwrite":
        grouped_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
        return table_name
    if write_mode == "replace_version" and spark.catalog.tableExists(table_name):
        prompt_version = _version_value(config, "prompt_version").replace("'", "''")
        taxonomy_version = _version_value(config, "taxonomy_version").replace("'", "''")
        spark.sql(
            f"""
            DELETE FROM {table_name}
            WHERE prompt_version = '{prompt_version}'
              AND taxonomy_version = '{taxonomy_version}'
            """
        )
    grouped_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table_name)
    return table_name


def build_and_save_tableau_grouped_final(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    write_mode: str = "replace_version",
) -> dict[str, Any]:
    """Build and save grouped Tableau output."""
    grouped_df = build_tableau_grouped_final_df(spark, config)
    row_count = grouped_df.count()
    table_name = save_tableau_grouped_final(
        spark,
        config,
        grouped_df,
        write_mode=write_mode,
    )
    return {"table_name": table_name, "row_count": int(row_count)}
