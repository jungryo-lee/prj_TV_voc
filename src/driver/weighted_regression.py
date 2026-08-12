"""Weighted correlation and WLS regression for VOC driver analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from common.config_loader import get_output_table


MODEL_SCHEMA = StructType(
    [
        StructField("segment_col", StringType(), False),
        StructField("segment_value", StringType(), False),
        StructField("group_dim", StringType(), False),
        StructField("group_key", StringType(), False),
        StructField("y_feature", StringType(), False),
        StructField("y_obs", LongType(), True),
        StructField("x_feature_count", LongType(), True),
        StructField("r_squared", DoubleType(), True),
        StructField("adj_r_squared", DoubleType(), True),
        StructField("f_statistic", DoubleType(), True),
        StructField("prob_f", DoubleType(), True),
        StructField("log_likelihood", DoubleType(), True),
        StructField("aic", DoubleType(), True),
        StructField("bic", DoubleType(), True),
        StructField("cond_no", DoubleType(), True),
        StructField("data_created_dt", TimestampType(), True),
    ]
)

COEF_SCHEMA = StructType(
    [
        StructField("segment_col", StringType(), False),
        StructField("segment_value", StringType(), False),
        StructField("group_dim", StringType(), False),
        StructField("group_key", StringType(), False),
        StructField("y_feature", StringType(), False),
        StructField("x_feature", StringType(), False),
        StructField("coef", DoubleType(), True),
        StructField("p_value", DoubleType(), True),
        StructField("t_value", DoubleType(), True),
        StructField("y_obs", LongType(), True),
        StructField("x_obs", LongType(), True),
        StructField("abs_coef", DoubleType(), True),
        StructField("driver_rank", LongType(), True),
        StructField("is_driver", IntegerType(), True),
        StructField("weighted_corr", DoubleType(), True),
        StructField("r_squared", DoubleType(), True),
        StructField("adj_r_squared", DoubleType(), True),
        StructField("data_created_dt", TimestampType(), True),
    ]
)

CORR_SCHEMA = StructType(
    [
        StructField("segment_col", StringType(), False),
        StructField("segment_value", StringType(), False),
        StructField("group_dim", StringType(), False),
        StructField("group_key", StringType(), False),
        StructField("y_feature", StringType(), False),
        StructField("x_feature", StringType(), False),
        StructField("weighted_corr", DoubleType(), True),
        StructField("abs_weighted_corr", DoubleType(), True),
        StructField("selected_for_regression", BooleanType(), True),
        StructField("data_created_dt", TimestampType(), True),
    ]
)

DRIVER_SELECTION_SCHEMA = StructType(
    [
        StructField("segment_col", StringType(), False),
        StructField("segment_value", StringType(), False),
        StructField("group_dim", StringType(), False),
        StructField("group_key", StringType(), False),
        StructField("y_feature", StringType(), False),
        StructField("x_feature", StringType(), False),
        StructField("coef", DoubleType(), True),
        StructField("p_value", DoubleType(), True),
        StructField("t_value", DoubleType(), True),
        StructField("weighted_corr", DoubleType(), True),
        StructField("y_obs", LongType(), True),
        StructField("x_obs", LongType(), True),
        StructField("abs_coef", DoubleType(), True),
        StructField("driver_rank", LongType(), True),
        StructField("is_driver", IntegerType(), True),
        StructField("data_created_dt", TimestampType(), True),
    ]
)

NETWORK_EDGE_SCHEMA = StructType(
    [
        StructField("segment_col", StringType(), False),
        StructField("segment_value", StringType(), False),
        StructField("group_dim", StringType(), False),
        StructField("group_key", StringType(), False),
        StructField("y_feature", StringType(), False),
        StructField("x_feature", StringType(), False),
        StructField("coef_sign", StringType(), True),
        StructField("corr_abs", DoubleType(), True),
        StructField("edge_weight", DoubleType(), True),
        StructField("r_squared", DoubleType(), True),
        StructField("y_obs", LongType(), True),
        StructField("is_significant", IntegerType(), True),
        StructField("data_created_dt", TimestampType(), True),
    ]
)


def weighted_corr(x: pd.Series, y: pd.Series, w: pd.Series) -> float:
    """Calculate weighted Pearson correlation."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    w_arr = np.asarray(w, dtype=float)

    valid = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(w_arr) & (w_arr > 0)
    if not valid.any():
        return float("nan")

    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    w_arr = w_arr[valid]

    w_sum = np.sum(w_arr)
    if w_sum <= 0:
        return float("nan")

    mx = np.sum(w_arr * x_arr) / w_sum
    my = np.sum(w_arr * y_arr) / w_sum
    cov = np.sum(w_arr * (x_arr - mx) * (y_arr - my))
    var_x = np.sum(w_arr * (x_arr - mx) ** 2)
    var_y = np.sum(w_arr * (y_arr - my) ** 2)
    if var_x <= 0 or var_y <= 0:
        return float("nan")
    return float(cov / np.sqrt(var_x * var_y))


def _weight_series(count_series: pd.Series, method: str) -> pd.Series:
    """Return WLS weights from review counts."""
    counts = count_series.fillna(0).clip(lower=0)
    if method == "total_count":
        return counts
    return np.sqrt(counts)


def _build_wide_frames(input_pdf: pd.DataFrame, group_dims: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return wide score frame and entity metadata frame."""
    score_df = input_pdf.pivot_table(
        index="model_id",
        columns="feature_name",
        values="avg_sc",
        aggfunc="mean",
    ).add_suffix("_score")
    count_df = input_pdf.pivot_table(
        index="model_id",
        columns="feature_name",
        values="total_count",
        aggfunc="sum",
    ).add_suffix("_count")
    wide_df = pd.concat([score_df, count_df], axis=1).fillna(0)

    meta_cols = ["model_id", *[col for col in group_dims if col != "all" and col in input_pdf.columns]]
    meta_df = input_pdf[meta_cols].drop_duplicates("model_id").set_index("model_id")
    return wide_df, meta_df


def _calculate_weighted_corrs(
    pdf: pd.DataFrame,
    y_feature: str,
    x_features: list[str],
    *,
    weight_method: str,
) -> list[tuple[str, float]]:
    """Calculate weighted correlation for every available X feature."""
    y_score_col = f"{y_feature}_score"
    y_count_col = f"{y_feature}_count"
    if y_score_col not in pdf.columns or y_count_col not in pdf.columns:
        return []

    y = pdf[y_score_col]
    weights = _weight_series(pdf[y_count_col], weight_method)
    corr_values: list[tuple[str, float]] = []
    for x_feature in x_features:
        if x_feature == y_feature:
            continue
        x_score_col = f"{x_feature}_score"
        if x_score_col not in pdf.columns:
            continue
        corr = weighted_corr(pdf[x_score_col], y, weights)
        if pd.notna(corr):
            corr_values.append((x_feature, corr))
    return sorted(corr_values, key=lambda item: abs(item[1]), reverse=True)


def _select_x_by_weighted_corr(
    corr_values: list[tuple[str, float]],
    *,
    corr_threshold: float,
) -> list[tuple[str, float]]:
    """Select regression candidates by absolute weighted correlation."""
    return [
        (x_feature, corr)
        for x_feature, corr in corr_values
        if pd.notna(corr) and abs(corr) >= corr_threshold
    ]


def _fit_wls(
    pdf: pd.DataFrame,
    y_feature: str,
    selected_x: list[tuple[str, float]],
    *,
    min_group_obs: int,
    weight_method: str,
):
    """Fit one WLS model, returning None when the design matrix is not usable."""
    y_score_col = f"{y_feature}_score"
    y_count_col = f"{y_feature}_count"
    filtered_pdf = pdf[(pdf[y_count_col] > 0) & pdf[y_score_col].notna()].copy()
    if len(filtered_pdf) < min_group_obs:
        return None

    final_x = [x for x, _ in selected_x if f"{x}_score" in filtered_pdf.columns]
    if not final_x:
        return None
    if len(filtered_pdf) <= len(final_x) + 1:
        final_x = final_x[: max(1, len(filtered_pdf) - 2)]

    x_cols = [f"{x}_score" for x in final_x]
    x_df = filtered_pdf[x_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    if x_df.nunique(dropna=False).le(1).all():
        return None

    X = sm.add_constant(x_df, has_constant="add")
    y = pd.to_numeric(filtered_pdf[y_score_col], errors="coerce")
    weights = _weight_series(filtered_pdf[y_count_col], weight_method)
    try:
        return sm.WLS(y, X, weights=weights).fit()
    except Exception:
        return None


def run_weighted_regression(
    spark: SparkSession,
    config: dict[str, Any],
    *,
    input_table_key: str = "driver_input",
    mode: str = "overwrite",
) -> dict[str, Any]:
    """Run weighted correlation/WLS and save driver-analysis output tables."""
    driver_cfg = config.get("driver_analysis", {}) or {}
    input_table = get_output_table(config, input_table_key)
    input_pdf = spark.table(input_table).toPandas()
    if input_pdf.empty:
        raise ValueError(f"No rows found in driver input table: {input_table}")

    group_dims = list(driver_cfg.get("group_dims", []) or ["all"])
    corr_threshold = float(driver_cfg.get("corr_threshold", 0.1))
    pvalue_max = float(driver_cfg.get("pvalue_max", 0.1))
    abs_coef_threshold = float(driver_cfg.get("abs_coef_threshold", 0.1))
    weight_method = str(driver_cfg.get("weight_method", "sqrt_total_count"))
    min_group_obs = int(driver_cfg.get("min_group_obs", 20))
    segment_col = str(driver_cfg.get("segment_col") or "").strip()
    configured_segment_values = [str(value) for value in driver_cfg.get("segment_values", []) or []]

    features = sorted(str(value) for value in input_pdf["feature_name"].dropna().unique())
    if segment_col and segment_col in input_pdf.columns:
        if configured_segment_values:
            segment_values = configured_segment_values
        else:
            segment_values = sorted(str(value) for value in input_pdf[segment_col].dropna().unique())
        segment_items = [(segment_col, value) for value in segment_values]
    else:
        segment_items = [("all", "all")]

    model_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    network_rows: list[dict[str, Any]] = []

    for segment_name, segment_value in segment_items:
        if segment_name == "all":
            segment_pdf = input_pdf.copy()
        else:
            segment_pdf = input_pdf[input_pdf[segment_name].astype(str) == segment_value].copy()
        if segment_pdf.empty:
            continue

        wide_df, meta_df = _build_wide_frames(segment_pdf, group_dims)

        for group_dim in group_dims:
            if group_dim == "all":
                group_keys = ["all"]
            elif group_dim in meta_df.columns:
                group_keys = sorted(str(value) for value in meta_df[group_dim].dropna().unique())
            else:
                continue

            for group_key in group_keys:
                if group_dim == "all":
                    group_pdf = wide_df.copy()
                else:
                    model_ids = meta_df.index[meta_df[group_dim].astype(str) == group_key]
                    group_pdf = wide_df.loc[wide_df.index.isin(model_ids)].copy()

                if group_pdf.empty:
                    continue

                for y_feature in features:
                    all_corr_values = _calculate_weighted_corrs(
                        group_pdf,
                        y_feature,
                        features,
                        weight_method=weight_method,
                    )
                    selected_x = _select_x_by_weighted_corr(
                        all_corr_values,
                        corr_threshold=corr_threshold,
                    )
                    selected_feature_set = {x_feature for x_feature, _ in selected_x}
                    for x_feature, corr in all_corr_values:
                        corr_rows.append(
                            {
                                "segment_col": segment_name,
                                "segment_value": segment_value,
                                "group_dim": group_dim,
                                "group_key": group_key,
                                "y_feature": y_feature,
                                "x_feature": x_feature,
                                "weighted_corr": float(corr),
                                "abs_weighted_corr": float(abs(corr)),
                                "selected_for_regression": x_feature in selected_feature_set,
                            }
                        )

                    model = _fit_wls(
                        group_pdf,
                        y_feature,
                        selected_x,
                        min_group_obs=min_group_obs,
                        weight_method=weight_method,
                    )
                    if model is None:
                        continue

                    selected_feature_names = [x for x, _ in selected_x]
                    y_score_col = f"{y_feature}_score"
                    y_count_col = f"{y_feature}_count"
                    model_base_pdf = group_pdf[
                        (group_pdf[y_count_col] > 0) & group_pdf[y_score_col].notna()
                    ]
                    y_obs = int(model.nobs)
                    model_rows.append(
                        {
                            "segment_col": segment_name,
                            "segment_value": segment_value,
                            "group_dim": group_dim,
                            "group_key": group_key,
                            "y_feature": y_feature,
                            "y_obs": int(model.nobs),
                            "x_feature_count": int(len(selected_feature_names)),
                            "r_squared": float(model.rsquared),
                            "adj_r_squared": float(model.rsquared_adj),
                            "f_statistic": float(model.fvalue) if pd.notna(model.fvalue) else None,
                            "prob_f": float(model.f_pvalue) if pd.notna(model.f_pvalue) else None,
                            "log_likelihood": float(model.llf),
                            "aic": float(model.aic),
                            "bic": float(model.bic),
                            "cond_no": float(model.condition_number),
                        }
                    )

                    coef_rows.append(
                        {
                            "segment_col": segment_name,
                            "segment_value": segment_value,
                            "group_dim": group_dim,
                            "group_key": group_key,
                            "y_feature": y_feature,
                            "x_feature": "β₀",
                            "coef": float(model.params.get("const"))
                            if pd.notna(model.params.get("const"))
                            else None,
                            "p_value": float(model.pvalues.get("const"))
                            if pd.notna(model.pvalues.get("const"))
                            else None,
                            "t_value": float(model.tvalues.get("const"))
                            if pd.notna(model.tvalues.get("const"))
                            else None,
                            "y_obs": y_obs,
                            "x_obs": None,
                            "abs_coef": abs(float(model.params.get("const")))
                            if pd.notna(model.params.get("const"))
                            else None,
                            "driver_rank": None,
                            "is_driver": 0,
                            "weighted_corr": None,
                            "r_squared": float(model.rsquared),
                            "adj_r_squared": float(model.rsquared_adj),
                        }
                    )
                    corr_by_feature = dict(selected_x)
                    for x_feature in selected_feature_names:
                        param_name = f"{x_feature}_score"
                        x_count_col = f"{x_feature}_count"
                        x_obs = (
                            int((model_base_pdf[x_count_col] > 0).sum())
                            if x_count_col in model_base_pdf.columns
                            else None
                        )
                        coef_value = (
                            float(model.params.get(param_name))
                            if pd.notna(model.params.get(param_name))
                            else None
                        )
                        p_value = (
                            float(model.pvalues.get(param_name))
                            if pd.notna(model.pvalues.get(param_name))
                            else None
                        )
                        weighted_corr_value = (
                            float(corr_by_feature.get(x_feature))
                            if pd.notna(corr_by_feature.get(x_feature))
                            else None
                        )
                        abs_coef_value = abs(coef_value) if coef_value is not None else None
                        is_driver = int(
                            p_value is not None
                            and p_value < pvalue_max
                            and abs_coef_value is not None
                            and abs_coef_value >= abs_coef_threshold
                        )
                        coef_rows.append(
                            {
                                "segment_col": segment_name,
                                "segment_value": segment_value,
                                "group_dim": group_dim,
                                "group_key": group_key,
                                "y_feature": y_feature,
                                "x_feature": x_feature,
                                "coef": coef_value,
                                "p_value": p_value,
                                "t_value": float(model.tvalues.get(param_name))
                                if pd.notna(model.tvalues.get(param_name))
                                else None,
                                "y_obs": y_obs,
                                "x_obs": x_obs,
                                "abs_coef": abs_coef_value,
                                "driver_rank": None,
                                "is_driver": is_driver,
                                "weighted_corr": weighted_corr_value,
                                "r_squared": float(model.rsquared),
                                "adj_r_squared": float(model.rsquared_adj),
                            }
                        )
                        network_rows.append(
                            {
                                "segment_col": segment_name,
                                "segment_value": segment_value,
                                "group_dim": group_dim,
                                "group_key": group_key,
                                "y_feature": y_feature,
                                "x_feature": x_feature,
                                "coef_sign": "positive"
                                if coef_value is not None and coef_value >= 0
                                else "negative",
                                "corr_abs": abs(weighted_corr_value)
                                if weighted_corr_value is not None
                                else None,
                                "edge_weight": abs_coef_value,
                                "r_squared": float(model.rsquared),
                                "y_obs": int(model.nobs),
                                "is_significant": is_driver,
                            }
                        )

    model_df = spark.createDataFrame(model_rows, schema=MODEL_SCHEMA)
    coef_df = spark.createDataFrame(coef_rows, schema=COEF_SCHEMA)
    corr_df = spark.createDataFrame(corr_rows, schema=CORR_SCHEMA)
    network_df = spark.createDataFrame(network_rows, schema=NETWORK_EDGE_SCHEMA)

    data_created_dt = F.current_timestamp()
    model_df = model_df.withColumn(
        "data_created_dt",
        F.coalesce(F.col("data_created_dt"), data_created_dt),
    )
    coef_df = coef_df.withColumn(
        "data_created_dt",
        F.coalesce(F.col("data_created_dt"), data_created_dt),
    )
    corr_df = corr_df.withColumn(
        "data_created_dt",
        F.coalesce(F.col("data_created_dt"), data_created_dt),
    )
    network_df = network_df.withColumn(
        "data_created_dt",
        F.coalesce(F.col("data_created_dt"), data_created_dt),
    )

    driver_selection_df = (
        coef_df.where(F.col("x_feature") != "β₀")
        .where(F.col("p_value") <= F.lit(pvalue_max))
        .where(F.col("abs_coef") >= F.lit(abs_coef_threshold))
        .withColumn(
            "driver_rank",
            F.row_number().over(
                Window.partitionBy(
                    "segment_col",
                    "segment_value",
                    "group_dim",
                    "group_key",
                    "y_feature",
                ).orderBy(F.col("abs_coef").desc())
            ),
        )
        .withColumn("is_driver", F.lit(1))
    )

    coef_rank_window = Window.partitionBy(
        "segment_col",
        "segment_value",
        "group_dim",
        "group_key",
        "y_feature",
    ).orderBy(F.col("abs_coef").desc_nulls_last(), F.col("x_feature").asc())
    coef_df = coef_df.withColumn(
        "driver_rank",
        F.when(F.col("x_feature") == "β₀", F.lit(None).cast("bigint")).otherwise(
            F.row_number().over(coef_rank_window)
        ),
    ).withColumn(
        "is_driver",
        F.when(
            (F.col("x_feature") != "β₀")
            & (F.col("p_value") < F.lit(pvalue_max))
            & (F.col("abs_coef") >= F.lit(abs_coef_threshold)),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )

    table_map = {
        "weighted_corr": (corr_df, get_output_table(config, "weighted_corr")),
        "weighted_regression": (coef_df, get_output_table(config, "weighted_regression")),
        "weighted_network_edges": (
            network_df,
            get_output_table(config, "weighted_network_edges"),
        ),
        "driver_selection": (
            driver_selection_df.select([field.name for field in DRIVER_SELECTION_SCHEMA.fields]),
            get_output_table(config, "driver_selection"),
        ),
    }
    model_table = get_output_table(config, "weighted_regression_model")
    table_map["weighted_regression_model"] = (model_df, model_table)

    counts: dict[str, int] = {}
    for key, (df, table_name) in table_map.items():
        counts[key] = df.count()
        (
            df.write.format("delta")
            .mode(mode)
            .option("overwriteSchema", "true")
            .saveAsTable(table_name)
        )

    return {
        "input_table": input_table,
        "weighted_corr_table": get_output_table(config, "weighted_corr"),
        "weighted_regression_table": get_output_table(config, "weighted_regression"),
        "weighted_regression_model_table": model_table,
        "weighted_network_edges_table": get_output_table(config, "weighted_network_edges"),
        "driver_selection_table": get_output_table(config, "driver_selection"),
        "counts": counts,
    }
