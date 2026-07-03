"""Group extraction and memo sampling for rule-profile generation."""

from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from common.config_loader import get_source_table
from common.memo_id import with_memo_id


DEFAULT_SAMPLE_SEED = "seed_20260420"


def _sql_escape(value: str) -> str:
    """Escape a string for simple SQL literal interpolation."""
    return str(value).replace("'", "''")


def get_rule_profile_stage_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return merged taxonomy + rule-profile settings for this stage."""
    taxonomy_cfg = config.get("taxonomy", {})
    rule_profile_cfg = config.get("rule_profile", {})

    return {
        "target_sentiments": taxonomy_cfg.get("target_sentiments", [1, -1]),
        "exclude_category_prefix": taxonomy_cfg.get("exclude_category_prefix", "***"),
        "min_group_size": int(taxonomy_cfg.get("min_group_size", 100)),
        "max_sample_rows": int(taxonomy_cfg.get("max_sample_rows", 1000)),
        "max_rule_sample_rows": int(taxonomy_cfg.get("max_rule_sample_rows", 800)),
        "limit_group_count": rule_profile_cfg.get("limit_group_count"),
    }


def build_group_query(config: dict[str, Any]) -> str:
    """Build the SQL used to load raw rows for eligible rule-profile groups."""
    source_table = get_source_table(config, "raw_review_table")
    stage_cfg = get_rule_profile_stage_config(config)

    exclude_prefix = _sql_escape(stage_cfg["exclude_category_prefix"])
    sentiments = ", ".join(str(int(v)) for v in stage_cfg["target_sentiments"])

    return f"""
select
    cate_1_depth,
    cate_2_depth,
    sc_measurement,
    memo
from {source_table}
where cate_1_depth not like '{exclude_prefix}%'
  and sc_measurement in ({sentiments})
  and memo is not null
  and length(trim(memo)) > 0
""".strip()


def _dedupe_by_memo_id(df: DataFrame) -> DataFrame:
    """Keep one representative row per memo_id."""
    window = Window.partitionBy("memo_id").orderBy(F.col("memo").asc())
    return (
        df.transform(with_memo_id)
        .withColumn("_memo_rn", F.row_number().over(window))
        .where(F.col("_memo_rn") == 1)
        .drop("_memo_rn")
    )


def load_target_groups(
    spark: SparkSession,
    config: dict[str, Any],
    limit_group_count: int | None = None,
) -> DataFrame:
    """Load eligible groups for rule-profile generation."""
    stage_cfg = get_rule_profile_stage_config(config)
    source_df = spark.sql(build_group_query(config))
    deduped_df = _dedupe_by_memo_id(source_df)
    group_df = (
        deduped_df.groupBy("cate_1_depth", "cate_2_depth", "sc_measurement")
        .agg(F.count("*").alias("group_total_cnt"))
        .where(F.col("group_total_cnt") >= int(stage_cfg["min_group_size"]))
        .orderBy("cate_1_depth", "cate_2_depth", "sc_measurement")
    )

    effective_limit = (
        limit_group_count
        if limit_group_count is not None
        else stage_cfg.get("limit_group_count")
    )
    if effective_limit is not None:
        group_df = group_df.limit(int(effective_limit))

    return group_df


def build_group_sample_query(
    config: dict[str, Any],
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    max_rows: int,
    sample_seed: str = DEFAULT_SAMPLE_SEED,
) -> str:
    """Build SQL used to load raw rows for one category/sentiment group."""
    source_table = get_source_table(config, "raw_review_table")
    cate_1 = _sql_escape(cate_1_depth)
    cate_2 = _sql_escape(cate_2_depth)

    return f"""
select
    cate_1_depth,
    cate_2_depth,
    sc_measurement,
    year,
    country,
    brand_name,
    d_type,
    memo
from {source_table}
where cate_1_depth = '{cate_1}'
  and cate_2_depth = '{cate_2}'
  and sc_measurement = {int(sc_measurement)}
  and memo is not null
  and length(trim(memo)) > 0
""".strip()


def load_group_sample_df(
    spark: SparkSession,
    config: dict[str, Any],
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    max_rows: int,
    sample_seed: str = DEFAULT_SAMPLE_SEED,
) -> DataFrame:
    """Load a reproducible memo sample DataFrame for one group."""
    query = build_group_sample_query(
        config=config,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        max_rows=max_rows,
        sample_seed=sample_seed,
    )
    seed = _sql_escape(sample_seed)
    base_df = spark.sql(query)
    deduped_df = _dedupe_by_memo_id(base_df)

    sampled_df = (
        deduped_df.withColumn(
            "_sample_rn",
            F.row_number().over(
                Window.orderBy(
                    F.md5(
                        F.concat(
                            F.coalesce(F.col("memo_id"), F.lit("")),
                            F.lit("||"),
                            F.lit(cate_1_depth),
                            F.lit("||"),
                            F.lit(cate_2_depth),
                            F.lit("||"),
                            F.lit(str(int(sc_measurement))),
                            F.lit("||"),
                            F.lit(seed),
                        )
                    )
                )
            ),
        )
        .where(F.col("_sample_rn") <= int(max_rows))
        .orderBy("_sample_rn")
        .drop("_sample_rn")
    )
    return sampled_df


def load_diverse_prompt_sample_df(
    spark: SparkSession,
    config: dict[str, Any],
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    max_prompt_rows: int,
    sample_seed: str = DEFAULT_SAMPLE_SEED,
) -> DataFrame:
    """Load a diverse prompt sample mixed across metadata and memo lengths."""
    stage_cfg = get_rule_profile_stage_config(config)
    base_df = load_group_sample_df(
        spark=spark,
        config=config,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        max_rows=int(stage_cfg["max_rule_sample_rows"]),
        sample_seed=sample_seed,
    )

    enriched_df = (
        base_df.withColumn(
            "_year_group",
            F.coalesce(F.col("year").cast("string"), F.lit("unknown_year")),
        )
        .withColumn(
            "_country_group",
            F.coalesce(F.col("country").cast("string"), F.lit("unknown_country")),
        )
        .withColumn(
            "_brand_group",
            F.coalesce(F.col("brand_name").cast("string"), F.lit("unknown_brand")),
        )
        .withColumn(
            "_dtype_group",
            F.coalesce(F.col("d_type").cast("string"), F.lit("unknown_dtype")),
        )
        .withColumn("_memo_len", F.length(F.coalesce(F.col("memo_norm"), F.col("memo"))))
        .withColumn(
            "_memo_len_bucket",
            F.when(F.col("_memo_len") <= 25, F.lit("short"))
            .when(F.col("_memo_len") <= 80, F.lit("medium"))
            .otherwise(F.lit("long")),
        )
        .withColumn(
            "_stratum_key",
            F.concat_ws(
                "||",
                F.col("_year_group"),
                F.col("_country_group"),
                F.col("_brand_group"),
                F.col("_dtype_group"),
                F.col("_memo_len_bucket"),
            ),
        )
    )

    within_stratum_window = Window.partitionBy("_stratum_key").orderBy(
        F.md5(
            F.concat(
                F.coalesce(F.col("memo_id"), F.lit("")),
                F.lit("||"),
                F.lit(sample_seed),
            )
        )
    )
    round_robin_window = Window.orderBy(
        F.col("_within_stratum_rn").asc(),
        F.md5(
            F.concat(
                F.col("_stratum_key"),
                F.lit("||"),
                F.coalesce(F.col("memo_id"), F.lit("")),
                F.lit("||"),
                F.lit(sample_seed),
            )
        ).asc(),
    )

    return (
        enriched_df.withColumn(
            "_within_stratum_rn",
            F.row_number().over(within_stratum_window),
        )
        .withColumn("_prompt_rn", F.row_number().over(round_robin_window))
        .where(F.col("_prompt_rn") <= int(max_prompt_rows))
        .orderBy("_prompt_rn")
        .drop(
            "_year_group",
            "_country_group",
            "_brand_group",
            "_dtype_group",
            "_memo_len",
            "_memo_len_bucket",
            "_stratum_key",
            "_within_stratum_rn",
            "_prompt_rn",
        )
    )


def collect_group_sample_memos(
    spark: SparkSession,
    config: dict[str, Any],
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    max_rows: int,
    sample_seed: str = DEFAULT_SAMPLE_SEED,
) -> list[str]:
    """Collect sampled memos as a Python list."""
    sample_df = load_group_sample_df(
        spark=spark,
        config=config,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        max_rows=max_rows,
        sample_seed=sample_seed,
    )
    return [row["memo"] for row in sample_df.toLocalIterator()]


def collect_rule_profile_sample_memos(
    spark: SparkSession,
    config: dict[str, Any],
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    sample_seed: str = DEFAULT_SAMPLE_SEED,
) -> list[str]:
    """Collect memos sized for rule-profile generation."""
    stage_cfg = get_rule_profile_stage_config(config)
    return collect_group_sample_memos(
        spark=spark,
        config=config,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        max_rows=int(stage_cfg["max_rule_sample_rows"]),
        sample_seed=sample_seed,
    )


def collect_diverse_rule_profile_prompt_memos(
    spark: SparkSession,
    config: dict[str, Any],
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    max_prompt_rows: int,
    sample_seed: str = DEFAULT_SAMPLE_SEED,
) -> list[str]:
    """Collect a prompt-sized memo list with mixed metadata/length coverage."""
    sample_df = load_diverse_prompt_sample_df(
        spark=spark,
        config=config,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        max_prompt_rows=max_prompt_rows,
        sample_seed=sample_seed,
    )
    return [row["memo"] for row in sample_df.toLocalIterator()]


def collect_topic_pool_sample_memos(
    spark: SparkSession,
    config: dict[str, Any],
    cate_1_depth: str,
    cate_2_depth: str,
    sc_measurement: int,
    sample_seed: str = DEFAULT_SAMPLE_SEED,
) -> list[str]:
    """Collect memos sized for topic-pool generation."""
    stage_cfg = get_rule_profile_stage_config(config)
    return collect_group_sample_memos(
        spark=spark,
        config=config,
        cate_1_depth=cate_1_depth,
        cate_2_depth=cate_2_depth,
        sc_measurement=sc_measurement,
        max_rows=int(stage_cfg["max_sample_rows"]),
        sample_seed=sample_seed,
    )
