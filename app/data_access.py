"""Databricks SQL table access helpers for the Databricks App layer."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import yaml
from databricks import sql
from databricks.sdk.core import Config, oauth_service_principal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = Path(
    os.environ.get("VOC_SETTINGS_PATH", PROJECT_ROOT / "config" / "settings_intellytics.yaml")
)


def _load_settings() -> dict[str, Any]:
    """Load project settings for app table names and defaults."""
    with SETTINGS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


SETTINGS = _load_settings()


def _output_table(key: str) -> str:
    """Return a configured output table name."""
    return str(SETTINGS["tables"]["outputs"][key])


def _reference_table(key: str) -> str:
    """Return a configured reference table name."""
    return str(SETTINGS["reference"][key])


def _is_blank_setting(value: Any) -> bool:
    """Return whether an app setting should be treated as unset."""
    if value is None:
        return True
    return str(value).strip().lower() in {"", "none", "null", "*"}


def _app_setting(key: str, default: Any = None) -> Any:
    """Read an app setting with environment variable override support."""
    env_key = f"VOC_APP_{key.upper()}"
    if env_key in os.environ and not _is_blank_setting(os.environ[env_key]):
        return os.environ[env_key]
    return SETTINGS.get("app", {}).get(key, default)


def _model_key() -> str:
    """Resolve the active app model key."""
    return str(_app_setting("model_key", "gpt_55"))


def _model_version() -> str:
    """Resolve the active app model version."""
    model_key = _model_key()
    return str(SETTINGS["llm"]["models"][model_key]["model_version"])


def _classification_table_key() -> str:
    """Resolve which classification output the app should read."""
    table_key = _app_setting("classification_table_key", "classification_full")
    if _is_blank_setting(table_key):
        return "classification_full"
    return str(table_key)


def _classification_table_name() -> str:
    """Return the configured classification table for summary and review."""
    table_key = _classification_table_key()
    return _output_table(table_key)


def _version_value(key: str) -> str:
    """Resolve version metadata from settings."""
    return str(SETTINGS.get("version", {}).get(key, ""))


def _target_filter(alias: str = "", *, model_version: str | None = None) -> str:
    """Build the default app target filter clause."""
    prefix = f"{alias}." if alias else ""
    resolved_model_version = model_version if model_version is not None else _model_version()
    conditions = [
        f"{prefix}model_version = '{_sql_escape(resolved_model_version)}'",
        f"{prefix}prompt_version = '{_sql_escape(_version_value('prompt_version'))}'",
        f"{prefix}taxonomy_version = '{_sql_escape(_version_value('taxonomy_version'))}'",
    ]

    cate_1_depth = _app_setting("target_cate_1_depth")
    cate_2_depth = _app_setting("target_cate_2_depth")
    sc_measurement = _app_setting("target_sc_measurement")

    if not _is_blank_setting(cate_1_depth):
        conditions.append(f"{prefix}cate_1_depth = '{_sql_escape(str(cate_1_depth))}'")
    if not _is_blank_setting(cate_2_depth):
        conditions.append(f"{prefix}cate_2_depth = '{_sql_escape(str(cate_2_depth))}'")
    if not _is_blank_setting(sc_measurement):
        conditions.append(f"{prefix}sc_measurement = {int(sc_measurement)}")

    return " AND ".join(conditions)


def _sql_escape(value: str) -> str:
    """Escape a string for SQL literal interpolation."""
    return str(value or "").replace("'", "''")


def _server_hostname() -> str:
    """Resolve Databricks SQL server hostname."""
    host = (
        os.environ.get("DATABRICKS_SERVER_HOSTNAME")
        or os.environ.get("DATABRICKS_HOST")
        or str(_app_setting("server_hostname", ""))
    )
    return host.replace("https://", "").replace("http://", "").strip("/")


def _http_path() -> str:
    """Resolve Databricks SQL warehouse HTTP path."""
    return (
        os.environ.get("DATABRICKS_HTTP_PATH")
        or os.environ.get("DATABRICKS_WAREHOUSE_HTTP_PATH")
        or str(_app_setting("warehouse_http_path", ""))
    )


def _access_token() -> str:
    """Resolve Databricks auth token."""
    return os.environ.get("DATABRICKS_TOKEN") or str(_app_setting("access_token", ""))


def _oauth_credentials_provider(server_hostname: str):
    """Build OAuth credentials provider from Databricks Apps service principal env."""
    config = Config(
        host=f"https://{server_hostname}",
        client_id=os.environ.get("DATABRICKS_CLIENT_ID"),
        client_secret=os.environ.get("DATABRICKS_CLIENT_SECRET"),
    )
    return oauth_service_principal(config)


@contextmanager
def _connect() -> Iterator[Any]:
    """Open a Databricks SQL connection."""
    server_hostname = _server_hostname()
    http_path = _http_path()
    access_token = _access_token()
    client_id = os.environ.get("DATABRICKS_CLIENT_ID")
    client_secret = os.environ.get("DATABRICKS_CLIENT_SECRET")

    if not server_hostname:
        raise RuntimeError("Databricks SQL hostname is missing. Set DATABRICKS_HOST.")
    if not http_path:
        raise RuntimeError("Databricks SQL warehouse HTTP path is missing. Set DATABRICKS_HTTP_PATH.")

    if access_token:
        connection = sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=access_token,
        )
    elif client_id and client_secret:
        connection = sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            credentials_provider=lambda: _oauth_credentials_provider(server_hostname),
        )
    else:
        raise RuntimeError(
            "Databricks auth is missing. Set DATABRICKS_TOKEN or use Databricks Apps OAuth env."
        )
    try:
        yield connection
    finally:
        connection.close()


def _query_df(query: str) -> pd.DataFrame:
    """Execute a SQL query and return pandas DataFrame."""
    with _connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()


def _count_query(table_name: str, where_clause: str | None = None) -> int | None:
    """Return table count for diagnostics."""
    query = f"SELECT COUNT(*) AS cnt FROM {table_name}"
    if where_clause:
        query = f"{query} WHERE {where_clause}"
    df = _query_df(query)
    if df.empty:
        return None
    return int(df.iloc[0]["cnt"])


def _execute_sql(statement: str) -> None:
    """Execute one SQL statement."""
    with _connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(statement)


def load_app_diagnostics() -> pd.DataFrame:
    """Load app settings and row-count diagnostics for troubleshooting."""
    classification_table = _classification_table_name()
    topic_pool_table = _output_table("topic_pool")
    review_decision_table = _output_table("review_decision")
    category_mapping_table = _reference_table("category_mapping_table")

    rows = [
        {
            "check": "settings_path",
            "target": str(SETTINGS_PATH),
            "status": "ok" if SETTINGS_PATH.exists() else "missing",
            "value": "",
        },
        {
            "check": "classification_table_key",
            "target": "app.classification_table_key",
            "status": "ok",
            "value": _classification_table_key(),
        },
        {
            "check": "model_versions",
            "target": "topic_pool / classification",
            "status": "ok",
            "value": f"topic_pool={_model_key()}, classification={_model_version()}",
        },
        {
            "check": "target_filter",
            "target": "classification",
            "status": "ok",
            "value": _target_filter(),
        },
        {
            "check": "target_filter",
            "target": "topic_pool",
            "status": "ok",
            "value": _target_filter(model_version=_model_key()),
        },
    ]

    table_checks = [
        (
            "classification_total",
            classification_table,
            None,
        ),
        (
            "classification_app_filtered",
            classification_table,
            _target_filter(),
        ),
        (
            "topic_pool_total",
            topic_pool_table,
            None,
        ),
        (
            "topic_pool_app_filtered",
            topic_pool_table,
            _target_filter(model_version=_model_key()),
        ),
        (
            "review_decision_total",
            review_decision_table,
            None,
        ),
        (
            "category_mapping_total",
            category_mapping_table,
            None,
        ),
    ]

    for check_name, table_name, where_clause in table_checks:
        try:
            count_value = _count_query(table_name, where_clause)
            rows.append(
                {
                    "check": check_name,
                    "target": table_name,
                    "status": "ok",
                    "value": str(count_value),
                }
            )
        except Exception as exc:  # pragma: no cover - displayed in Databricks Apps
            rows.append(
                {
                    "check": check_name,
                    "target": table_name,
                    "status": "error",
                    "value": repr(exc),
                }
            )

    return pd.DataFrame(rows)


def load_topic_pool() -> pd.DataFrame:
    """Load topic-pool rows from the configured sandbox table."""
    table_name = _output_table("topic_pool")
    mapping_table = _reference_table("category_mapping_table")
    query = f"""
        SELECT
            t.cate_1_depth,
            COALESCE(m.cate_1_depth_kor, t.cate_1_depth) AS cate_1_depth_kor,
            t.cate_2_depth,
            COALESCE(m.cate_2_depth_kor, t.cate_2_depth) AS cate_2_depth_kor,
            t.sc_measurement,
            t.topic_order,
            t.topic,
            t.description,
            t.model_version,
            t.prompt_version,
            t.taxonomy_version,
            t.created_at
        FROM {table_name} t
        LEFT JOIN {mapping_table} m
          ON t.cate_1_depth = m.cate_1_depth
         AND t.cate_2_depth = m.cate_2_depth
        WHERE {_target_filter("t", model_version=_model_key())}
        ORDER BY t.topic_order ASC
        LIMIT 500
    """
    return _query_df(query)


def load_classification_summary() -> pd.DataFrame:
    """Load topic distribution summary from the configured classification table."""
    table_name = _classification_table_name()
    mapping_table = _reference_table("category_mapping_table")
    query = f"""
        SELECT
            c.cate_1_depth,
            COALESCE(m.cate_1_depth_kor, c.cate_1_depth) AS cate_1_depth_kor,
            c.cate_2_depth,
            COALESCE(m.cate_2_depth_kor, c.cate_2_depth) AS cate_2_depth_kor,
            c.sc_measurement,
            c.pred_topic,
            c.pred_topic_type,
            COUNT(*) AS row_cnt,
            COUNT(DISTINCT c.memo_id) AS memo_id_cnt
        FROM {table_name} c
        LEFT JOIN {mapping_table} m
          ON c.cate_1_depth = m.cate_1_depth
         AND c.cate_2_depth = m.cate_2_depth
        WHERE {_target_filter("c")}
        GROUP BY
            c.cate_1_depth,
            COALESCE(m.cate_1_depth_kor, c.cate_1_depth),
            c.cate_2_depth,
            COALESCE(m.cate_2_depth_kor, c.cate_2_depth),
            c.sc_measurement,
            c.pred_topic,
            c.pred_topic_type
        ORDER BY row_cnt DESC
        LIMIT 500
    """
    df = _query_df(query)
    if df.empty or "row_cnt" not in df.columns:
        return df
    total = float(df["row_cnt"].sum())
    df["row_ratio"] = df["row_cnt"] / total if total > 0 else 0.0
    return df


def load_others_review_candidates() -> pd.DataFrame:
    """Load distinct others rows for human review from the configured classification table."""
    table_name = _classification_table_name()
    mapping_table = _reference_table("category_mapping_table")
    max_rows = int(_app_setting("max_review_rows", 300))
    query = f"""
        SELECT
            c.cate_1_depth,
            COALESCE(m.cate_1_depth_kor, c.cate_1_depth) AS cate_1_depth_kor,
            c.cate_2_depth,
            COALESCE(m.cate_2_depth_kor, c.cate_2_depth) AS cate_2_depth_kor,
            c.sc_measurement,
            c.memo_id,
            c.memo_norm,
            c.memo AS sample_memo,
            c.pred_topic AS current_pred_topic,
            c.pred_topic_type AS current_pred_topic_type,
            c.match_reason,
            c.model_version,
            c.prompt_version,
            c.taxonomy_version,
            c.run_id,
            c.run_date
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY cate_1_depth, cate_2_depth, sc_measurement, memo_id,
                                 model_version, prompt_version, taxonomy_version
                    ORDER BY created_at DESC NULLS LAST, run_id DESC NULLS LAST
                ) AS rn
            FROM {table_name}
            WHERE {_target_filter()}
              AND pred_topic_type = 'others'
        ) c
        LEFT JOIN {mapping_table} m
          ON c.cate_1_depth = m.cate_1_depth
         AND c.cate_2_depth = m.cate_2_depth
        WHERE c.rn = 1
        ORDER BY c.memo_norm ASC
        LIMIT {max_rows}
    """
    return _query_df(query)


def save_manual_review_decisions(review_df: pd.DataFrame) -> str:
    """Save manually edited review decisions directly to review_decision table."""
    if review_df.empty:
        return "no rows"

    decision_table = _output_table("review_decision")
    created_at = datetime.utcnow().isoformat(timespec="seconds")
    values: list[str] = []

    for row in review_df.to_dict("records"):
        approved_topic = str(row.get("approved_topic") or "").strip()
        approved_action = (
            "reassign_existing_topic"
            if approved_topic
            else "keep_others"
        )
        memo_id = str(row.get("memo_id") or "").strip()
        if not memo_id:
            continue

        decision_seed = "||".join(
            [
                "manual_fallback",
                str(row.get("cate_1_depth") or ""),
                str(row.get("cate_2_depth") or ""),
                str(row.get("sc_measurement") or ""),
                memo_id,
                approved_action,
                approved_topic,
                str(row.get("model_version") or _model_version()),
                str(row.get("prompt_version") or _version_value("prompt_version")),
                str(row.get("taxonomy_version") or _version_value("taxonomy_version")),
            ]
        )
        decision_id = hashlib.sha256(decision_seed.encode("utf-8")).hexdigest()
        suggested_topic = approved_topic if approved_action == "reassign_existing_topic" else ""
        evidence_json = json.dumps(
            {
                "source": "databricks_app",
                "review_comment": str(row.get("review_comment") or ""),
            },
            ensure_ascii=False,
        )

        values.append(
            "("
            f"'{_sql_escape(decision_id)}', "
            "'manual_fallback', "
            f"'{_sql_escape(str(row.get('cate_1_depth') or ''))}', "
            f"'{_sql_escape(str(row.get('cate_2_depth') or ''))}', "
            f"{int(row.get('sc_measurement') or 0)}, "
            f"'{_sql_escape(memo_id)}', "
            f"'{_sql_escape(str(row.get('memo_norm') or ''))}', "
            f"'{_sql_escape(str(row.get('sample_memo') or ''))}', "
            f"'{_sql_escape(str(row.get('current_pred_topic') or '기타'))}', "
            f"'{_sql_escape(str(row.get('current_pred_topic_type') or 'others'))}', "
            f"'{_sql_escape(approved_action)}', "
            f"'{_sql_escape(suggested_topic)}', "
            "NULL, NULL, NULL, NULL, "
            f"'{_sql_escape(evidence_json)}', "
            "'approved', "
            f"'{_sql_escape(approved_action)}', "
            f"'{_sql_escape(approved_topic)}', "
            f"'{_sql_escape(str(row.get('reviewer') or 'databricks_app'))}', "
            f"'{_sql_escape(str(row.get('review_comment') or ''))}', "
            f"'{_sql_escape(created_at)}', "
            f"'{_sql_escape(_classification_table_key())}', "
            f"'{_sql_escape(str(row.get('run_id') or ''))}', "
            f"'{_sql_escape(str(row.get('run_date') or ''))}', "
            f"'{_sql_escape(str(row.get('prompt_version') or _version_value('prompt_version')))}', "
            f"'{_sql_escape(str(row.get('taxonomy_version') or _version_value('taxonomy_version')))}', "
            f"'{_sql_escape(str(row.get('model_version') or _model_version()))}', "
            f"'{_sql_escape(_version_value('pipeline_version'))}', "
            f"'{_sql_escape(created_at)}', "
            "'databricks_app'"
            ")"
        )

    if not values:
        return "no valid rows"

    values_sql = ",\n".join(values)
    statement = f"""
        INSERT INTO {decision_table} (
            decision_id,
            candidate_type,
            cate_1_depth,
            cate_2_depth,
            sc_measurement,
            memo_id,
            memo_norm,
            sample_memo,
            current_pred_topic,
            current_pred_topic_type,
            suggested_action,
            suggested_topic,
            suggestion_score,
            candidate_cnt,
            candidate_distinct_memo_id_cnt,
            candidate_ratio,
            candidate_evidence_json,
            decision_status,
            approved_action,
            approved_topic,
            reviewer,
            review_comment,
            reviewed_at,
            source_table_key,
            run_id,
            run_date,
            prompt_version,
            taxonomy_version,
            model_version,
            pipeline_version,
            created_at,
            created_by
        )
        VALUES {values_sql}
    """
    _execute_sql(statement)
    return f"{decision_table}: inserted {len(values)} rows"
