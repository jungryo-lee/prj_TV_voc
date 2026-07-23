"""Databricks Apps entrypoint for the TV VOC review console."""

from __future__ import annotations

import os

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dash_table, dcc, html
from dash import callback_context

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


def _topic_distribution_chart(
    summary_df: pd.DataFrame,
    *,
    sc_measurement: int,
    title: str,
):
    """Build a descending topic-distribution bar chart for one sentiment group."""
    chart_df = summary_df.copy()
    if chart_df.empty:
        return px.bar(
            pd.DataFrame({"pred_topic": [], "row_cnt": []}),
            x="pred_topic",
            y="row_cnt",
            title=title,
            template="simple_white",
        )

    chart_df = chart_df[chart_df["sc_measurement"].astype(str) == str(sc_measurement)]
    chart_df = chart_df.sort_values("row_cnt", ascending=False).head(25)
    fig = px.bar(
        chart_df,
        x="pred_topic",
        y="row_cnt",
        color="pred_topic_type",
        title=title,
        template="simple_white",
        labels={"pred_topic": "Topic", "row_cnt": "Rows", "pred_topic_type": "Type"},
    )
    fig.update_layout(xaxis_tickangle=-35, margin={"l": 40, "r": 20, "t": 60, "b": 120})
    return fig


def _filter_summary_records(
    records: list[dict],
    *,
    cate_1_depth_kor: str | None = None,
    cate_2_depth_kor: str | None = None,
) -> list[dict]:
    """Apply summary-tab category filters."""
    filtered = records or []
    if cate_1_depth_kor:
        filtered = [row for row in filtered if str(row.get("cate_1_depth_kor") or "") == str(cate_1_depth_kor)]
    if cate_2_depth_kor:
        filtered = [row for row in filtered if str(row.get("cate_2_depth_kor") or "") == str(cate_2_depth_kor)]
    return filtered


def _visible_columns(df: pd.DataFrame, *, hidden: set[str] | None = None) -> list[dict[str, str]]:
    """Return DataTable columns excluding internal fields."""
    hidden = hidden or set()
    return [{"name": col, "id": col} for col in df.columns if col not in hidden]


def _dropdown_options(values) -> list[dict[str, str]]:
    """Return single-select dropdown options with a total option."""
    options = [{"label": "전체", "value": ""}]
    clean_values = sorted({str(value) for value in values if not pd.isna(value) and str(value).strip()})
    options.extend({"label": value, "value": value} for value in clean_values)
    return options


def _filter_review_records(
    records: list[dict],
    *,
    cate_1_depth_kor: str | None = None,
    cate_2_depth_kor: str | None = None,
    sc_measurement: str | int | None = None,
    topic: str | None = None,
) -> list[dict]:
    """Apply review-tab filters to row records."""
    filtered = records or []
    if cate_1_depth_kor:
        filtered = [row for row in filtered if str(row.get("cate_1_depth_kor") or "") == str(cate_1_depth_kor)]
    if cate_2_depth_kor:
        filtered = [row for row in filtered if str(row.get("cate_2_depth_kor") or "") == str(cate_2_depth_kor)]
    if sc_measurement not in (None, ""):
        filtered = [row for row in filtered if str(row.get("sc_measurement") or "") == str(sc_measurement)]
    if topic:
        filtered = [row for row in filtered if str(row.get("current_pred_topic") or "") == str(topic)]
    return filtered


def _row_identity(row: dict) -> str:
    """Return stable row identity for saved-row removal."""
    return "||".join(
        [
            str(row.get("cate_1_depth") or ""),
            str(row.get("cate_2_depth") or ""),
            str(row.get("sc_measurement") or ""),
            str(row.get("memo_id") or ""),
            str(row.get("model_version") or ""),
            str(row.get("prompt_version") or ""),
            str(row.get("taxonomy_version") or ""),
        ]
    )


def _group_key(row: dict) -> str:
    """Return stable group key for topic-option lookup."""
    return "||".join(
        [
            str(row.get("cate_1_depth") or ""),
            str(row.get("cate_2_depth") or ""),
            str(row.get("sc_measurement") or ""),
        ]
    )


def _topic_options_by_group(topic_pool_df: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    """Return topic options grouped by source category keys."""
    required_cols = {"cate_1_depth", "cate_2_depth", "sc_measurement", "topic"}
    if topic_pool_df.empty or not required_cols.issubset(topic_pool_df.columns):
        return {}

    topic_options: dict[str, list[dict[str, str]]] = {}
    group_cols = ["cate_1_depth", "cate_2_depth", "sc_measurement"]
    for group_values, group_df in topic_pool_df.groupby(group_cols, dropna=False):
        group_key = "||".join(str(value) for value in group_values)
        options = [{"label": "기타 유지", "value": ""}]
        for topic in group_df["topic"].dropna().astype(str).drop_duplicates().sort_values():
            options.append({"label": topic, "value": topic})
        topic_options[group_key] = options

    return topic_options


def _review_table_columns() -> list[dict[str, str]]:
    """Return columns for the left-side review candidate table."""
    visible_columns = [
        "sample_memo",
        "cate_1_depth_kor",
        "cate_2_depth_kor",
        "sc_measurement",
        "current_pred_topic",
        "match_reason",
    ]
    return [{"name": col, "id": col} for col in visible_columns]


def _detail_topic_options(row: dict | None, topic_options_by_group: dict | None) -> list[dict[str, str]]:
    """Return topic dropdown options for one selected review row."""
    if not row:
        return []
    topic_options_by_group = topic_options_by_group or {}
    options = topic_options_by_group.get(_group_key(row), [])
    return [option for option in options if option.get("value")]


def _selected_review_row(table_rows: list[dict] | None, selected_rows: list[int] | None) -> dict | None:
    """Return the currently selected review row from the left table."""
    if not table_rows:
        return None
    selected_index = selected_rows[0] if selected_rows else 0
    if selected_index < 0 or selected_index >= len(table_rows):
        selected_index = 0
    return table_rows[selected_index]


def _review_detail_panel(row: dict | None) -> html.Div:
    """Render the right-side manual review detail panel."""
    if not row:
        return html.Div(
            dbc.Alert("검토할 리뷰를 왼쪽 테이블에서 선택하세요.", color="secondary"),
            className="h-100",
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Div("선택 리뷰", className="text-muted small"),
                    html.H5(str(row.get("sample_memo") or ""), className="mb-3"),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("카테고리", className="text-muted small"),
                            html.Div(str(row.get("cate_1_depth_kor") or "")),
                            html.Div(str(row.get("cate_2_depth_kor") or ""), className="fw-semibold"),
                        ],
                        md=8,
                    ),
                    dbc.Col(
                        [
                            html.Div("감성", className="text-muted small"),
                            html.Div(str(row.get("sc_measurement") or "")),
                        ],
                        md=4,
                    ),
                ],
                className="mb-3",
            ),
            html.Div("현재 분류", className="text-muted small"),
            html.Div(
                [
                    dbc.Badge(str(row.get("current_pred_topic_type") or ""), color="secondary", className="me-2"),
                    html.Span(str(row.get("current_pred_topic") or "")),
                ],
                className="mb-3",
            ),
            html.Div("분류 사유", className="text-muted small"),
            html.Div(
                str(row.get("match_reason") or ""),
                className="border rounded p-2 bg-light mb-3",
                style={"whiteSpace": "pre-wrap", "maxHeight": "220px", "overflowY": "auto"},
            ),
            html.Div("저장 방식", className="text-muted small"),
            html.P(
                "주제를 선택하고 '기존 Topic으로 확정'을 누르면 reassign_existing_topic으로 저장됩니다. "
                "'기타 유지'를 누르면 keep_others로 저장됩니다.",
                className="small text-muted",
            ),
        ]
    )


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
                dcc.Tab(label="리뷰 검토", value="others-review"),
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

        summary_records = summary_df.to_dict("records")
        return dbc.Container(
            [
                dcc.Store(id="summary-data-store", data=summary_records),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("cate_1_depth_kor", className="small text-muted"),
                                dcc.Dropdown(
                                    id="summary-filter-cate1",
                                    options=_dropdown_options(summary_df.get("cate_1_depth_kor", [])),
                                    value="",
                                    clearable=False,
                                ),
                            ],
                            md=6,
                        ),
                        dbc.Col(
                            [
                                html.Label("cate_2_depth_kor", className="small text-muted"),
                                dcc.Dropdown(
                                    id="summary-filter-cate2",
                                    options=_dropdown_options(summary_df.get("cate_2_depth_kor", [])),
                                    value="",
                                    clearable=False,
                                ),
                            ],
                            md=6,
                        ),
                    ],
                    className="g-2 mb-3",
                ),
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
                dcc.Graph(
                    id="summary-positive-chart",
                    figure=_topic_distribution_chart(summary_df, sc_measurement=1, title="긍정 주제분류 분포"),
                    className="mt-3",
                ),
                dcc.Graph(
                    id="summary-negative-chart",
                    figure=_topic_distribution_chart(summary_df, sc_measurement=-1, title="부정 주제분류 분포"),
                    className="mt-3",
                ),
                dash_table.DataTable(
                    id="summary-detail-table",
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
    topic_options_by_group = _topic_options_by_group(topic_pool_df)
    if others_df.empty:
        return _empty_message("설정된 분류 결과 테이블에서 조회된 기타 리뷰 후보가 없습니다.")

    review_df = others_df.copy()
    review_df["approved_topic"] = ""
    review_df["review_comment"] = ""
    review_records = review_df.to_dict("records")
    initial_review_row = review_records[0] if review_records else None
    initial_topic_options = _detail_topic_options(initial_review_row, topic_options_by_group)

    return dbc.Container(
        [
            dcc.Store(id="topic-options-store", data=topic_options_by_group),
            dcc.Store(id="review-data-store", data=review_records),
            dbc.Alert(
                "왼쪽에서 검토할 리뷰를 선택하고, 오른쪽 패널에서 주제를 선택하거나 기타 유지를 확정하세요.",
                color="info",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("cate_1_depth_kor", className="small text-muted"),
                            dcc.Dropdown(
                                id="review-filter-cate1",
                                options=_dropdown_options(review_df.get("cate_1_depth_kor", [])),
                                value="",
                                clearable=False,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("cate_2_depth_kor", className="small text-muted"),
                            dcc.Dropdown(
                                id="review-filter-cate2",
                                options=_dropdown_options(review_df.get("cate_2_depth_kor", [])),
                                value="",
                                clearable=False,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("sc_measurement", className="small text-muted"),
                            dcc.Dropdown(
                                id="review-filter-sc",
                                options=_dropdown_options(review_df.get("sc_measurement", [])),
                                value="",
                                clearable=False,
                            ),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        [
                            html.Label("topic", className="small text-muted"),
                            dcc.Dropdown(
                                id="review-filter-topic",
                                options=_dropdown_options(review_df.get("current_pred_topic", [])),
                                value="",
                                clearable=False,
                            ),
                        ],
                        md=2,
                    ),
                ],
                className="g-2 mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dash_table.DataTable(
                            id="manual-review-table",
                            data=review_records[:300],
                            columns=_review_table_columns(),
                            row_selectable="single",
                            selected_rows=[0] if review_records else [],
                            page_size=12,
                            sort_action="native",
                            filter_action="native",
                            style_table={"overflowX": "auto", "maxHeight": "720px", "overflowY": "auto"},
                            style_cell={
                                "fontFamily": "sans-serif",
                                "fontSize": 13,
                                "textAlign": "left",
                                "whiteSpace": "normal",
                                "height": "auto",
                                "maxWidth": "360px",
                            },
                            style_data_conditional=[
                                {
                                    "if": {"state": "selected"},
                                    "backgroundColor": "#e8f1ff",
                                    "border": "1px solid #2f6fed",
                                }
                            ],
                        ),
                        md=7,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div(
                                        _review_detail_panel(initial_review_row),
                                        id="review-detail-panel",
                                    ),
                                    html.Label("확정 Topic", className="small text-muted mt-2"),
                                    dcc.Dropdown(
                                        id="detail-approved-topic",
                                        options=initial_topic_options,
                                        value="",
                                        placeholder="기존 Topic 중 하나를 선택",
                                        clearable=True,
                                    ),
                                    html.Label("검토 코멘트", className="small text-muted mt-3"),
                                    dcc.Textarea(
                                        id="review-comment-input",
                                        value="",
                                        placeholder="선택 사항입니다.",
                                        style={"width": "100%", "height": "90px"},
                                    ),
                                    dbc.ButtonGroup(
                                        [
                                            dbc.Button(
                                                "기존 Topic으로 확정",
                                                id="approve-topic-button",
                                                color="primary",
                                            ),
                                            dbc.Button(
                                                "기타 유지",
                                                id="keep-others-button",
                                                color="secondary",
                                            ),
                                        ],
                                        className="mt-3 w-100",
                                    ),
                                    html.Div(id="save-review-status", className="mt-3"),
                                ]
                            ),
                            className="h-100",
                        ),
                        md=5,
                    ),
                ],
                className="g-3",
            ),
        ],
        fluid=True,
    )


@dash_app.callback(
    Output("summary-filter-cate2", "options"),
    Output("summary-filter-cate2", "value"),
    Output("summary-positive-chart", "figure"),
    Output("summary-negative-chart", "figure"),
    Output("summary-detail-table", "data"),
    Input("summary-filter-cate1", "value"),
    Input("summary-filter-cate2", "value"),
    State("summary-data-store", "data"),
    prevent_initial_call=True,
)
def update_summary_view(cate_1_depth_kor, cate_2_depth_kor, summary_records):
    """Update summary filters and positive/negative charts."""
    summary_records = summary_records or []
    cate1_rows = _filter_summary_records(
        summary_records,
        cate_1_depth_kor=cate_1_depth_kor,
    )
    cate2_options = _dropdown_options(
        [row.get("cate_2_depth_kor") for row in cate1_rows]
    )
    allowed_cate2_values = {option["value"] for option in cate2_options}
    resolved_cate2 = cate_2_depth_kor if cate_2_depth_kor in allowed_cate2_values else ""

    filtered_rows = _filter_summary_records(
        summary_records,
        cate_1_depth_kor=cate_1_depth_kor,
        cate_2_depth_kor=resolved_cate2,
    )
    filtered_df = pd.DataFrame(filtered_rows)

    positive_fig = _topic_distribution_chart(
        filtered_df,
        sc_measurement=1,
        title="긍정 주제분류 분포",
    )
    negative_fig = _topic_distribution_chart(
        filtered_df,
        sc_measurement=-1,
        title="부정 주제분류 분포",
    )
    table_rows = filtered_df.sort_values("row_cnt", ascending=False).head(200).to_dict("records") if not filtered_df.empty else []
    return cate2_options, resolved_cate2, positive_fig, negative_fig, table_rows


@dash_app.callback(
    Output("manual-review-table", "data"),
    Output("manual-review-table", "selected_rows"),
    Input("review-filter-cate1", "value"),
    Input("review-filter-cate2", "value"),
    Input("review-filter-sc", "value"),
    Input("review-filter-topic", "value"),
    State("review-data-store", "data"),
    prevent_initial_call=True,
)
def filter_review_table(
    cate_1_depth_kor,
    cate_2_depth_kor,
    sc_measurement,
    topic,
    stored_rows,
):
    """Filter review candidates in the left table."""
    stored_rows = stored_rows or []
    filtered_rows = _filter_review_records(
        stored_rows,
        cate_1_depth_kor=cate_1_depth_kor,
        cate_2_depth_kor=cate_2_depth_kor,
        sc_measurement=sc_measurement,
        topic=topic,
    )[:300]
    selected_rows = [0] if filtered_rows else []
    return filtered_rows, selected_rows


@dash_app.callback(
    Output("review-detail-panel", "children"),
    Output("detail-approved-topic", "options"),
    Output("detail-approved-topic", "value"),
    Output("review-comment-input", "value"),
    Input("manual-review-table", "data"),
    Input("manual-review-table", "selected_rows"),
    State("topic-options-store", "data"),
    prevent_initial_call=True,
)
def update_review_detail(table_rows, selected_rows, topic_options_by_group):
    """Update the right-side detail panel for the selected review row."""
    row = _selected_review_row(table_rows, selected_rows)
    topic_options = _detail_topic_options(row, topic_options_by_group)
    return _review_detail_panel(row), topic_options, "", ""


@dash_app.callback(
    Output("review-data-store", "data"),
    Output("manual-review-table", "data", allow_duplicate=True),
    Output("manual-review-table", "selected_rows", allow_duplicate=True),
    Output("save-review-status", "children"),
    Input("approve-topic-button", "n_clicks"),
    Input("keep-others-button", "n_clicks"),
    State("manual-review-table", "data"),
    State("manual-review-table", "selected_rows"),
    State("review-data-store", "data"),
    State("detail-approved-topic", "value"),
    State("review-comment-input", "value"),
    State("review-filter-cate1", "value"),
    State("review-filter-cate2", "value"),
    State("review-filter-sc", "value"),
    State("review-filter-topic", "value"),
    prevent_initial_call=True,
)
def save_selected_review_decision(
    _approve_clicks,
    _keep_clicks,
    table_rows,
    selected_rows,
    stored_rows,
    approved_topic,
    review_comment,
    cate_1_depth_kor,
    cate_2_depth_kor,
    sc_measurement,
    topic,
):
    """Save one selected review decision and remove it from the pending list."""
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
    selected_row = _selected_review_row(table_rows, selected_rows)
    stored_rows = stored_rows or []

    if not selected_row:
        return stored_rows, table_rows or [], [], dbc.Alert("선택된 리뷰가 없습니다.", color="warning")

    row_to_save = dict(selected_row)
    row_to_save["review_comment"] = review_comment or ""

    if triggered == "approve-topic-button":
        approved_topic = str(approved_topic or "").strip()
        if not approved_topic:
            return (
                stored_rows,
                table_rows or [],
                selected_rows or [],
                dbc.Alert("확정할 기존 Topic을 먼저 선택하세요.", color="warning"),
            )
        row_to_save["approved_topic"] = approved_topic
    elif triggered == "keep-others-button":
        row_to_save["approved_topic"] = ""
    else:
        return stored_rows, table_rows or [], selected_rows or [], ""

    saved_result = save_manual_review_decisions(pd.DataFrame([row_to_save]))
    saved_id = _row_identity(row_to_save)
    remaining_rows = [
        row for row in stored_rows if _row_identity(row) != saved_id
    ]
    filtered_rows = _filter_review_records(
        remaining_rows,
        cate_1_depth_kor=cate_1_depth_kor,
        cate_2_depth_kor=cate_2_depth_kor,
        sc_measurement=sc_measurement,
        topic=topic,
    )[:300]
    next_selected_rows = [0] if filtered_rows else []
    action_label = "기존 Topic 확정" if row_to_save.get("approved_topic") else "기타 유지"
    status = dbc.Alert(
        f"{action_label} 저장 완료 | {saved_result}",
        color="success",
    )
    return remaining_rows, filtered_rows, next_selected_rows, status


if __name__ == "__main__":
    dash_app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        debug=os.environ.get("DASH_DEBUG", "false").lower() == "true",
    )
