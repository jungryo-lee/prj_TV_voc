"""Databricks Apps entrypoint for the TV VOC review console."""

from __future__ import annotations

import os
from numbers import Number

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dash_table, dcc, html

from app.data_access import (
    load_app_diagnostics,
    load_classification_summary,
    load_others_review_candidates,
    load_topic_pool,
    save_manual_review_decisions,
)


dash_app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
server = dash_app.server


def _empty_message(message: str) -> dbc.Alert:
    """Return a compact empty-state message."""
    return dbc.Alert(message, color="secondary", className="my-2")


def _summary_chart(summary_df: pd.DataFrame):
    """Build the topic distribution chart."""
    if summary_df.empty:
        return px.bar(pd.DataFrame({"pred_topic": [], "row_cnt": []}), x="pred_topic", y="row_cnt")

    plot_df = summary_df.sort_values("row_cnt", ascending=False).head(20)
    return px.bar(
        plot_df,
        x="pred_topic",
        y="row_cnt",
        color="pred_topic_type",
        template="simple_white",
        labels={"pred_topic": "Topic", "row_cnt": "Rows", "pred_topic_type": "Type"},
    )


def _visible_columns(df: pd.DataFrame, *, hidden: set[str] | None = None) -> list[dict[str, str]]:
    """Return DataTable columns excluding internal fields."""
    hidden = hidden or set()
    return [{"name": col, "id": col} for col in df.columns if col not in hidden]


def _filter_value(value) -> str:
    """Return a Dash DataTable filter-query literal."""
    if pd.isna(value):
        return '""'
    if isinstance(value, Number) and not isinstance(value, bool):
        return str(int(value)) if float(value).is_integer() else str(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _group_filter_query(cate_1_depth, cate_2_depth, sc_measurement) -> str:
    """Build a row-level group filter for DataTable dropdowns."""
    return (
        f"{{cate_1_depth}} = {_filter_value(cate_1_depth)} && "
        f"{{cate_2_depth}} = {_filter_value(cate_2_depth)} && "
        f"{{sc_measurement}} = {_filter_value(sc_measurement)}"
    )


def _topic_dropdown_conditional(topic_pool_df: pd.DataFrame) -> list[dict]:
    """Return row-group-specific topic dropdown options."""
    required_cols = {"cate_1_depth", "cate_2_depth", "sc_measurement", "topic"}
    if topic_pool_df.empty or not required_cols.issubset(topic_pool_df.columns):
        return []

    dropdown_rules: list[dict] = []
    group_cols = ["cate_1_depth", "cate_2_depth", "sc_measurement"]
    for group_values, group_df in topic_pool_df.groupby(group_cols, dropna=False):
        cate_1_depth, cate_2_depth, sc_measurement = group_values
        options = [{"label": "기타 유지", "value": ""}]
        for topic in group_df["topic"].dropna().astype(str).drop_duplicates().sort_values():
            options.append({"label": topic, "value": topic})

        dropdown_rules.append(
            {
                "if": {
                    "column_id": "approved_topic",
                    "filter_query": _group_filter_query(
                        cate_1_depth,
                        cate_2_depth,
                        sc_measurement,
                    ),
                },
                "options": options,
            }
        )

    return dropdown_rules


def _topic_tooltip_data(review_df: pd.DataFrame, topic_pool_df: pd.DataFrame) -> list[dict]:
    """Return tooltip text with available topics for each review row."""
    required_cols = {"cate_1_depth", "cate_2_depth", "sc_measurement", "topic"}
    if topic_pool_df.empty or not required_cols.issubset(topic_pool_df.columns):
        return [{} for _ in range(len(review_df))]

    group_topics: dict[tuple[str, str, str], str] = {}
    group_cols = ["cate_1_depth", "cate_2_depth", "sc_measurement"]
    for group_values, group_df in topic_pool_df.groupby(group_cols, dropna=False):
        topics = group_df["topic"].dropna().astype(str).drop_duplicates().sort_values()
        topic_text = "\n".join(f"- {topic}" for topic in topics)
        group_topics[tuple(str(value) for value in group_values)] = topic_text

    tooltip_rows: list[dict] = []
    for row in review_df.to_dict("records"):
        group_key = (
            str(row.get("cate_1_depth")),
            str(row.get("cate_2_depth")),
            str(row.get("sc_measurement")),
        )
        topic_text = group_topics.get(group_key, "선택 가능한 주제가 없습니다.")
        tooltip_rows.append(
            {
                "approved_topic": {
                    "value": f"이 그룹의 선택 가능 주제:\n{topic_text}",
                    "type": "markdown",
                }
            }
        )
    return tooltip_rows


dash_app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1("TV VOC AI Review Console", className="mt-4"),
                        html.P(
                            "sandbox 테이블을 직접 조회하여 분류 현황과 기타 리뷰 검토를 수행하는 Databricks Apps 화면입니다.",
                            className="text-muted",
                        ),
                    ],
                    width=12,
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Alert(
                        [
                            html.Strong("Data source: "),
                            html.Code("Databricks SQL tables"),
                        ],
                        color="light",
                    ),
                    width=12,
                )
            ]
        ),
        dcc.Tabs(
            id="main-tabs",
            value="summary",
            children=[
                dcc.Tab(label="분류 현황", value="summary"),
                dcc.Tab(label="기타 리뷰 검토", value="others-review"),
                dcc.Tab(label="Topic Pool", value="topic-pool"),
                dcc.Tab(label="진단", value="diagnostics"),
            ],
        ),
        html.Div(id="tab-content", className="pt-3"),
    ],
    fluid=True,
)


@dash_app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab_value: str):
    """Render each app tab from current Databricks SQL tables."""
    if tab_value == "diagnostics":
        diagnostics_df = load_app_diagnostics()
        return dash_table.DataTable(
            data=diagnostics_df.to_dict("records"),
            columns=[{"name": col, "id": col} for col in diagnostics_df.columns],
            page_size=20,
            style_table={"overflowX": "auto"},
            style_cell={
                "fontFamily": "sans-serif",
                "fontSize": 13,
                "textAlign": "left",
                "whiteSpace": "normal",
                "height": "auto",
            },
        )

    if tab_value == "summary":
        summary_df = load_classification_summary()
        if summary_df.empty:
            return _empty_message(
                "설정된 분류 결과 테이블에서 조회된 분류 현황이 없습니다. 배치 실행 결과와 App SQL 접속 설정을 확인하세요."
            )

        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("Total Rows", className="card-title"),
                                        html.H3(f"{int(summary_df['row_cnt'].sum()):,}"),
                                    ]
                                )
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("Others Rows", className="card-title"),
                                        html.H3(
                                            f"{int(summary_df.loc[summary_df['pred_topic_type'] == 'others', 'row_cnt'].sum()):,}"
                                        ),
                                    ]
                                )
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("Topics", className="card-title"),
                                        html.H3(f"{summary_df['pred_topic'].nunique():,}"),
                                    ]
                                )
                            ),
                            md=4,
                        ),
                    ],
                    className="g-3",
                ),
                dcc.Graph(figure=_summary_chart(summary_df), className="mt-3"),
                dash_table.DataTable(
                    data=summary_df.head(200).to_dict("records"),
                    columns=_visible_columns(summary_df, hidden={"cate_1_depth", "cate_2_depth"}),
                    page_size=20,
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={"fontFamily": "sans-serif", "fontSize": 13, "textAlign": "left"},
                ),
            ],
            fluid=True,
        )

    if tab_value == "topic-pool":
        topic_pool_df = load_topic_pool()
        if topic_pool_df.empty:
            return _empty_message("topic_pool 테이블에서 조회된 주제 목록이 없습니다.")

        return dash_table.DataTable(
            data=topic_pool_df.to_dict("records"),
            columns=_visible_columns(topic_pool_df, hidden={"cate_1_depth", "cate_2_depth"}),
            page_size=20,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"fontFamily": "sans-serif", "fontSize": 13, "textAlign": "left"},
        )

    topic_pool_df = load_topic_pool()
    others_df = load_others_review_candidates()
    topic_dropdown_conditional = _topic_dropdown_conditional(topic_pool_df)
    if others_df.empty:
        return _empty_message("설정된 분류 결과 테이블에서 조회된 기타 리뷰 후보가 없습니다.")

    review_df = others_df.copy()
    review_df["approved_topic"] = ""
    review_df["approved_action"] = "keep_others"
    review_df["review_comment"] = ""

    visible_columns = [
        col
        for col in [
            "memo_id",
            "sample_memo",
            "cate_1_depth_kor",
            "cate_2_depth_kor",
            "sc_measurement",
            "current_pred_topic",
            "match_reason",
            "approved_topic",
            "approved_action",
            "review_comment",
        ]
        if col in review_df.columns
    ]
    table_data_columns = [
        col
        for col in visible_columns + ["cate_1_depth", "cate_2_depth"]
        if col in review_df.columns
    ]

    return dbc.Container(
        [
            dbc.Alert(
                "approved_topic을 선택하면 기존 topic으로 재배치됩니다. 비워두면 기타 유지입니다.",
                color="info",
            ),
            dash_table.DataTable(
                id="manual-review-table",
                data=review_df[table_data_columns].head(300).to_dict("records"),
                columns=[
                    {
                        "name": col,
                        "id": col,
                        "editable": col in {"approved_topic", "approved_action", "review_comment"},
                        "presentation": "dropdown" if col in {"approved_topic", "approved_action"} else "input",
                    }
                    for col in visible_columns
                    + [
                        col
                        for col in ["cate_1_depth", "cate_2_depth"]
                        if col in table_data_columns
                    ]
                ],
                hidden_columns=["cate_1_depth", "cate_2_depth"],
                dropdown={
                    "approved_topic": {"options": [{"label": "기타 유지", "value": ""}]},
                    "approved_action": {
                        "options": [
                            {"label": "기존 topic으로 재배치", "value": "reassign_existing_topic"},
                            {"label": "기타 유지", "value": "keep_others"},
                        ]
                    },
                },
                dropdown_conditional=topic_dropdown_conditional,
                tooltip_data=_topic_tooltip_data(review_df[table_data_columns].head(300), topic_pool_df),
                tooltip_duration=None,
                editable=True,
                page_size=20,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={
                    "fontFamily": "sans-serif",
                    "fontSize": 13,
                    "textAlign": "left",
                    "whiteSpace": "normal",
                    "height": "auto",
                },
            ),
            dbc.Button("검토 결과 테이블 저장", id="save-review-button", color="primary", className="mt-3"),
            html.Div(id="save-review-status", className="mt-3"),
        ],
        fluid=True,
    )


@dash_app.callback(
    Output("save-review-status", "children"),
    Input("save-review-button", "n_clicks"),
    State("manual-review-table", "data"),
    prevent_initial_call=True,
)
def save_review_rows(_n_clicks: int, table_rows: list[dict]):
    """Persist manually edited review decisions to review_decision table."""
    if not table_rows:
        return dbc.Alert("저장할 검토 결과가 없습니다.", color="warning")

    review_df = pd.DataFrame(table_rows)
    review_df = review_df[
        review_df.apply(
            lambda row: bool(str(row.get("approved_topic") or "").strip())
            or bool(str(row.get("review_comment") or "").strip()),
            axis=1,
        )
    ].copy()
    if review_df.empty:
        return dbc.Alert("변경되었거나 코멘트가 입력된 검토 결과가 없습니다.", color="warning")

    review_df["approved_action"] = review_df.apply(
        lambda row: "reassign_existing_topic"
        if str(row.get("approved_topic") or "").strip()
        else "keep_others",
        axis=1,
    )
    saved_result = save_manual_review_decisions(review_df)
    return dbc.Alert(f"검토 결과 저장 완료: {saved_result}", color="success")


if __name__ == "__main__":
    dash_app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        debug=os.environ.get("DASH_DEBUG", "false").lower() == "true",
    )
