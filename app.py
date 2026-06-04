import os
import json
import shutil
import uuid
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from groq import Groq
from dotenv import load_dotenv

from src.data_loader import (
    LIVE_DEMO_BASELINE_PATH,
    LIVE_TRANSACTIONS_PATH,
    SHOPIFY_CACHE_PATH,
    derive_latest_action_taken,
    load_csv_data as load_csv_transactions,
    load_data as load_transactions_data,
    save_synced_data,
)
from src.metrics import (
    compute_summary_metrics,
    format_currency,
    safe_top_index,
    to_csv_download,
)
from src.risk_logic import (
    answer_historical_question,
    answer_case_question,
    answer_risk_question,
    build_case_timeline,
    get_open_high_risk_cases,
    sort_cases_by_priority,
)
from src.ui_components import (
    render_ai_answer,
    render_ai_output,
    render_case_detail_box,
    render_demo_banner,
    render_hero_summary,
    render_insight_footer,
    render_merchant_context_bar,
    render_page_header,
    render_today_action_panel,
    render_timeline_html,
    render_verification_templates,
)

st.set_page_config(page_title="Aignita Risk Intelligence", layout="wide")

st.markdown("""
<style>
/* === Aignita Risk Intelligence Design System === */

.stApp {
    background: #f8fafc;
}

.block-container {
    padding-top: 1.25rem;
    padding-bottom: 2rem;
    max-width: 96rem;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #dbeafe;
    border-radius: 8px;
    padding: 0.75rem 1rem !important;
}
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] p {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b !important;
}
[data-testid="stMetricValue"] > div {
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    color: #1e293b !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #eaf3ff;
    border-radius: 8px;
    padding: 3px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    font-size: 0.84rem;
    font-weight: 500;
    color: #64748b;
    padding: 0.35rem 0.9rem;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #1d4ed8 !important;
    box-shadow: 0 1px 3px rgba(37,99,235,0.14);
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f7fbff;
    border-right: 1px solid #bfdbfe;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label {
    color: #1e3a8a !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #475569 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {
    border-color: #bfdbfe !important;
    background: #ffffff !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div:hover,
[data-testid="stSidebar"] [data-baseweb="input"] > div:hover {
    border-color: #60a5fa !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: #dbeafe !important;
    color: #1d4ed8 !important;
    border: 1px solid #bfdbfe !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] span {
    color: #1d4ed8 !important;
}
[data-testid="stSidebar"] [role="option"][aria-selected="true"] {
    background: #dbeafe !important;
    color: #1d4ed8 !important;
}
[data-testid="stSidebar"] [data-baseweb="calendar"] button[aria-selected="true"],
[data-testid="stSidebar"] [data-baseweb="calendar"] button:hover {
    background: #2563eb !important;
    color: #ffffff !important;
}

/* Buttons */
.stButton > button {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    color: #475569;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 0.3rem 0.7rem;
    transition: border-color 0.1s ease, background 0.1s ease;
}
.stButton > button:hover {
    background: #eff6ff !important;
    border-color: #60a5fa !important;
    color: #1d4ed8 !important;
}

/* Download buttons */
[data-testid="stDownloadButton"] > button {
    background: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #2563eb !important;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 500;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
}

/* Page header */
.riq-page-header {
    padding: 0.75rem 0 0.75rem 0;
    margin-bottom: 0.75rem;
    border-bottom: 1px solid #bfdbfe;
}

/* Hero summary card */
.riq-hero {
    background: #ffffff;
    border: 1px solid #dbeafe;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
}

/* Case detail card */
.riq-detail {
    background: #ffffff;
    border: 1px solid #dbeafe;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-top: 0.75rem;
    margin-bottom: 1rem;
}

/* Accent note strip */
.riq-note {
    background: #f8f9fb;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #2563eb;
    border-radius: 0 6px 6px 0;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.75rem;
    font-size: 0.86rem;
    color: #475569;
}

/* Demo banner */
.riq-demo-banner {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 6px;
    padding: 0.45rem 0.85rem;
    margin-bottom: 1rem;
    font-size: 0.82rem;
    color: #1d4ed8;
}

/* Merchant context and action brief */
.riq-context-bar {
    display: grid;
    grid-template-columns: 1.2fr 0.9fr 1.4fr 1fr 1fr;
    gap: 0.9rem;
    background: #ffffff;
    border: 1px solid #dbeafe;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
}
.riq-action-panel {
    background: #ffffff;
    border: 1px solid #dbeafe;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
}
.riq-action-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(18rem, 1fr));
    gap: 1rem;
}
.riq-action-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-top: 3px solid #2563eb;
    border-radius: 7px;
    padding: 1rem 1.1rem;
    min-height: 7.25rem;
}
@media (max-width: 900px) {
    .riq-context-bar,
    .riq-action-grid {
        grid-template-columns: 1fr;
    }
}

/* AI answer area */
.riq-answer {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-top: 0.5rem;
    color: #1e293b;
    font-size: 0.88rem;
    line-height: 1.65;
    white-space: pre-wrap;
}

/* Generated output area */
.riq-output {
    background: #fefce8;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-top: 0.5rem;
    color: #1e293b;
    font-size: 0.88rem;
    line-height: 1.65;
    white-space: pre-wrap;
}

/* Risk & status badges */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    line-height: 1.7;
}
.badge-critical { background:#fef2f2; color:#dc2626; border:1px solid #fecaca; }
.badge-high     { background:#fff7ed; color:#ea580c; border:1px solid #fed7aa; }
.badge-moderate { background:#fffbeb; color:#d97706; border:1px solid #fde68a; }
.badge-low      { background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0; }
.badge-fraud    { background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; }
.badge-return   { background:#fffbeb; color:#d97706; border:1px solid #fde68a; }
.badge-open     { background:#eff6ff; color:#2563eb; border:1px solid #bfdbfe; }
.badge-closed   { background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0; }
.badge-neutral  { background:#f1f5f9; color:#475569; border:1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ── Chart styling ───────────────────────────────────────────────────────────────

CHART_COLORS = ["#2563eb", "#0ea5e9", "#16a34a", "#dc2626", "#d97706", "#7c3aed", "#64748b"]
PRIORITY_COLORS = {"High": "#dc2626", "Moderate": "#d97706", "Low": "#16a34a"}
RISK_TYPE_COLORS = {"Fraud Risk": "#2563eb", "Return Risk": "#d97706"}
DEMO_MERCHANT_NAME = "Meridian Commerce"
DEMO_PLATFORM = "Shopify"
DEMO_LAST_SYNC = "Demo Snapshot: January 2026"
WORK_MODE_OPTIONS = ["Live Risk Queue", "Historical Intelligence"]

def style_chart(
    fig,
    height: int = 320,
    showlegend: bool | None = None,
    x_tickangle: int = 0,
):
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif",
            size=12,
            color="#334155",
        ),
        title_font=dict(size=13, color="#1e293b"),
        title_pad=dict(l=0, t=4),
        margin=dict(l=58, r=18, t=48, b=72),
        height=height,
        xaxis=dict(
            gridcolor="#f1f5f9",
            linecolor="#e2e8f0",
            tickfont=dict(size=11, color="#64748b"),
            title_font=dict(size=11, color="#64748b"),
            zeroline=False,
            automargin=True,
            tickangle=x_tickangle,
            ticklabelposition="outside",
        ),
        yaxis=dict(
            gridcolor="#f1f5f9",
            linecolor="#e2e8f0",
            tickfont=dict(size=11, color="#64748b"),
            title_font=dict(size=11, color="#64748b"),
            zeroline=False,
            automargin=True,
            rangemode="tozero",
        ),
        legend=dict(
            font=dict(size=11, color="#64748b"),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        colorway=CHART_COLORS,
        bargap=0.28,
    )
    fig.update_traces(marker_line_width=0)
    return fig


def build_today_actions(
    data: pd.DataFrame,
    metrics: dict,
    top_category: str,
    top_state: str,
) -> list[dict]:
    open_high_risk = get_open_high_risk_cases(data)
    open_revenue = float(open_high_risk["order_value"].sum()) if not open_high_risk.empty else 0.0

    return [
        {
            "label": "First Priority",
            "title": "Review high-risk open orders before fulfillment",
            "body": (
                "Start with the highest-priority open cases so teams can hold, restrict, "
                "or release orders before avoidable loss accumulates."
            ),
            "metric": f'{metrics["high_cases"]:,} high-risk',
            "accent": "#dc2626",
        },
        {
            "label": "Revenue Control",
            "title": "Protect open revenue needing review",
            "body": (
                "Use the review queue to resolve the open high-risk orders currently "
                "driving the largest outstanding exposure."
            ),
            "metric": format_currency(open_revenue),
            "accent": "#2563eb",
        },
    ]


# ── Bootstrap ───────────────────────────────────────────────────────────────────

load_dotenv()
GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

@st.cache_data
def load_data(
    source: str = "csv",
    shop_domain: str | None = None,
    access_token: str | None = None,
    limit: int = 100,
    cache_version: str | None = None,
):
    return load_transactions_data(
        source=source,
        shop_domain=shop_domain,
        access_token=access_token,
        limit=limit,
    )


def build_data_cache_version(source: str) -> str:
    paths = []
    if source == "csv":
        paths = ["data/transactions.csv", LIVE_TRANSACTIONS_PATH]
    elif source == "saved_shopify":
        paths = [SHOPIFY_CACHE_PATH]

    versions = []
    for path in paths:
        if os.path.exists(path):
            versions.append(f"{path}:{os.path.getmtime(path)}")
        else:
            versions.append(f"{path}:missing")
    return "|".join(versions) or "api"


def ensure_latest_action_taken(data: pd.DataFrame) -> pd.DataFrame:
    """Keep cached or older loaded datasets compatible with the current UI."""
    data = data.copy()
    if "latest_action_taken" not in data.columns:
        data["latest_action_taken"] = data.apply(
            lambda row: derive_latest_action_taken(
                str(row.get("recommended_action", "")),
                str(row.get("case_status", "")),
                str(row.get("workflow_stage", "")),
            ),
            axis=1,
        )
    else:
        missing_action = data["latest_action_taken"].isna() | (
            data["latest_action_taken"].astype(str).str.strip() == ""
        )
        if missing_action.any():
            data.loc[missing_action, "latest_action_taken"] = data.loc[missing_action].apply(
                lambda row: derive_latest_action_taken(
                    str(row.get("recommended_action", "")),
                    str(row.get("case_status", "")),
                    str(row.get("workflow_stage", "")),
                ),
                axis=1,
            )
    return data


OPS_ACTIONS = {
    "Start Review": {
        "case_status": "Open",
        "workflow_stage": "Review Queue",
        "latest_action_taken": "Review started by operations",
        "outcome_type": "Pending",
    },
    "Send Verification": {
        "case_status": "Open",
        "workflow_stage": "Verification Sent",
        "latest_action_taken": "Customer verification sent",
        "outcome_type": "Pending",
    },
    "Await Customer Response": {
        "case_status": "Open",
        "workflow_stage": "Awaiting Customer Response",
        "latest_action_taken": "Awaiting customer response",
        "outcome_type": "Pending",
    },
    "Hold Fulfillment": {
        "case_status": "Open",
        "workflow_stage": "Review Queue",
        "latest_action_taken": "Fulfillment held by operations",
        "outcome_type": "Pending",
    },
    "Approve Order": {
        "case_status": "Closed",
        "workflow_stage": "Resolved",
        "latest_action_taken": "Order approved by operations",
        "outcome_type": "Approved",
    },
    "Restrict Return / Refund": {
        "case_status": "Closed",
        "workflow_stage": "Resolved",
        "latest_action_taken": "Return/refund restricted by operations",
        "outcome_type": "Return Restricted",
    },
    "Escalate to Senior Review": {
        "case_status": "Open",
        "workflow_stage": "Escalated to Senior Review",
        "latest_action_taken": "Escalated for senior review",
        "outcome_type": "Pending",
    },
    "Close Case": {
        "case_status": "Closed",
        "workflow_stage": "Resolved",
        "latest_action_taken": "Case closed by operations",
        "outcome_type": "Resolved",
    },
}

LEARNING_BACKLOG_PATH = "data/model_learning_backlog.json"
DEFAULT_LEARNING_BACKLOG = {
    "historical": [],
    "live": [
        {
            "created_at": "2026-02-01T16:00:00+00:00",
            "source": "Seeded Case Learning Note",
            "focus_area": "Returns",
            "condition_field": "product_category",
            "condition_value": "Apparel",
            "guidance": "Be stricter",
            "priority": "High",
            "owner": "Analytics",
            "model_signal": "product_category:Apparel:repeat_return_review",
            "rationale": "Evaluate stricter review routing for repeat Apparel returns with elevated historical loss.",
        },
        {
            "created_at": "2026-02-01T16:00:00+00:00",
            "source": "Seeded Case Learning Note",
            "focus_area": "Fraud",
            "condition_field": "recommended_action",
            "condition_value": "Hold Fulfillment",
            "guidance": "Require manual review",
            "priority": "High",
            "owner": "Fraud Ops",
            "model_signal": "ato:device_mismatch:manual_review",
            "rationale": "Escalate account-takeover signals when device mismatch and fulfillment holds occur together.",
        },
    ],
}
FINAL_DECISION_OPTIONS = {
    "Fraud": ["Pending", "Fraud Confirmed", "Fraud False Positive"],
    "Returns": ["Pending", "Return Approved", "Return Denied"],
    "None": ["Pending", "Order Approved"],
}
RESOLUTION_REASON_OPTIONS = {
    "Fraud": [
        "Transaction Fraud", "Account Takeover (ATO)", "Chargeback Abuse",
        "Item Not Received (INR)",
    ],
    "Returns": [
        "Doesn't Fit Me", "Not What I Expected", "Changed My Mind",
        "Product Quality Issues", "Fulfillment Issues",
    ],
    "None": ["Standard approval"],
}


def load_learning_backlog(key: str) -> list:
    if not os.path.exists(LEARNING_BACKLOG_PATH):
        return [item.copy() for item in DEFAULT_LEARNING_BACKLOG.get(key, [])]
    try:
        with open(LEARNING_BACKLOG_PATH) as f:
            return json.load(f).get(
                key,
                [item.copy() for item in DEFAULT_LEARNING_BACKLOG.get(key, [])],
            )
    except Exception:
        return [item.copy() for item in DEFAULT_LEARNING_BACKLOG.get(key, [])]


def save_learning_backlog(key: str, insights: list):
    existing: dict = {}
    if os.path.exists(LEARNING_BACKLOG_PATH):
        try:
            with open(LEARNING_BACKLOG_PATH) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing[key] = insights
    with open(LEARNING_BACKLOG_PATH, "w") as f:
        json.dump(existing, f, indent=2)


def reset_demo_state():
    if not os.path.exists(LIVE_DEMO_BASELINE_PATH):
        raise FileNotFoundError(
            f"{LIVE_DEMO_BASELINE_PATH} is missing. Run python3 generate_data.py first."
        )
    shutil.copyfile(LIVE_DEMO_BASELINE_PATH, LIVE_TRANSACTIONS_PATH)
    with open(LEARNING_BACKLOG_PATH, "w") as f:
        json.dump(DEFAULT_LEARNING_BACKLOG, f, indent=2)
    load_data.clear()
    for key in list(st.session_state.keys()):
        if key.startswith("filt_") or key in {
            "_working_data_key", "_active_data_source", "_active_work_mode", "working_df",
            "historical_model_insights", "live_model_insights", "ops_action_log",
            "selected_case", "last_ops_action_notice",
        }:
            st.session_state.pop(key, None)


def ensure_ops_action_tracking(data: pd.DataFrame) -> pd.DataFrame:
    """Add action audit columns used by the Review Queue action form."""
    data = data.copy()
    defaults = {
        "ops_action_taken": "",
        "ops_action_by": "",
        "ops_action_at": "",
        "ops_action_note": "",
        "ops_action_history": "",
        "ops_override_reason": "",
        "analyst_confidence": 0.0,
        "ops_resolution_reason": "",
    }
    for column, default in defaults.items():
        if column not in data.columns:
            data[column] = default
        else:
            data[column] = data[column].fillna(default)
    return data


def ensure_lifecycle_intelligence(data: pd.DataFrame) -> pd.DataFrame:
    """Add historical outcome and feedback-loop fields for older datasets."""
    data = data.copy()
    defaults = {
        "return_requested": 0,
        "return_completed": 0,
        "return_reason_code": "None",
        "return_sub_reason": "None",
        "return_reason_text": "",
        "customer_return_comment": "",
        "fraud_label": "Unknown",
        "fraud_reason_code": "None",
        "fraud_sub_reason": "None",
        "fraud_reason_text": "",
        "customer_dispute_comment": "",
        "ops_resolution": "Pending",
        "ops_resolution_note": "",
        "label_source": "System",
        "label_confidence": 0.0,
        "reviewed_by": "",
        "reviewed_at": "",
        "dataset_mode": "",
    }
    for column, default in defaults.items():
        if column not in data.columns:
            data[column] = default
        else:
            data[column] = data[column].fillna(default)

    closed_mask = data["case_status"].astype(str).eq("Closed")
    pending_resolution_mask = closed_mask & data["ops_resolution"].astype(str).eq("Pending")
    data.loc[pending_resolution_mask, "ops_resolution"] = (
        data.loc[pending_resolution_mask, "outcome_type"].astype(str)
    )
    confidence = pd.to_numeric(data["label_confidence"], errors="coerce").fillna(0.0)
    data["label_confidence"] = confidence
    data.loc[closed_mask & confidence.eq(0.0), "label_confidence"] = 0.85
    missing_mode = data["dataset_mode"].astype(str).str.strip().eq("")
    data.loc[missing_mode & data["order_date"].dt.year.eq(2026), "dataset_mode"] = "Live"
    data.loc[missing_mode & data["order_date"].dt.year.ne(2026), "dataset_mode"] = "Historical"
    return data


def apply_work_mode(data: pd.DataFrame, work_mode: str) -> pd.DataFrame:
    if "dataset_mode" in data.columns and data["dataset_mode"].astype(str).str.strip().any():
        target_mode = "Historical" if work_mode == "Historical Intelligence" else "Live"
        mode_data = data[data["dataset_mode"].astype(str).eq(target_mode)].copy()
        if not mode_data.empty:
            return mode_data

    if work_mode == "Historical Intelligence":
        return data[
            data["case_status"].astype(str).eq("Closed")
            | data["outcome_type"].astype(str).ne("Pending")
        ].copy()

    return data[
        data["case_status"].astype(str).isin(["Open", "Pending"])
        | data["outcome_type"].astype(str).eq("Pending")
    ].copy()


def render_mode_panel(work_mode: str, data: pd.DataFrame):
    mode_text = (
        "Actual outcomes, return/fraud labels, customer comments, and ops resolutions."
        if work_mode == "Historical Intelligence"
        else "Predicted risk, open workflow stages, recommended actions, and analyst feedback capture."
    )
    if data.empty:
        range_text = "No rows in the current view."
    else:
        range_text = (
            f"{len(data):,} rows from "
            f"{data['order_date'].min().date()} to {data['order_date'].max().date()}."
        )
    st.info(f"{work_mode}: {mode_text} {range_text}")


def build_historical_kpis(data: pd.DataFrame) -> dict:
    total = len(data)
    return_completed = int(data["return_completed"].astype(int).sum()) if total else 0
    confirmed_fraud = int(data["fraud_label"].astype(str).eq("Confirmed").sum()) if total else 0
    return_requested = int(data["return_requested"].astype(int).sum()) if total else 0
    false_positive = int(data["fraud_label"].astype(str).eq("False Positive").sum()) if total else 0
    reviewed = int(data["label_source"].astype(str).isin(["Ops Review", "System Auto Label"]).sum()) if total else 0
    return {
        "actual_returns": return_completed,
        "return_requests": return_requested,
        "confirmed_fraud": confirmed_fraud,
        "actual_return_rate": return_completed / total * 100 if total else 0.0,
        "confirmed_fraud_rate": confirmed_fraud / total * 100 if total else 0.0,
        "false_positive_count": false_positive,
        "reviewed_count": reviewed,
    }


def calculate_historical_total_loss(data: pd.DataFrame) -> float:
    view = data.copy()
    view["actual_outcome"] = view.apply(simplify_actual_outcome, axis=1)
    return float(
        view.loc[
            view["actual_outcome"].isin(["Return Approved", "Fraud Approved"]),
            "order_value",
        ].sum()
    )


def render_historical_insight_footer(data: pd.DataFrame):
    view = data.copy()
    view["actual_outcome"] = view.apply(simplify_actual_outcome, axis=1)
    total_orders = len(view)
    actual_returns = int(view["actual_outcome"].eq("Return Approved").sum())
    confirmed_fraud = int(view["actual_outcome"].eq("Fraud Denied").sum())
    total_loss = calculate_historical_total_loss(view)
    avoided_loss = float(
        view.loc[
            view["actual_outcome"].isin(["Return Denied", "Fraud Denied"]),
            "order_value",
        ].sum()
    )
    top_return_loss_category = safe_top_index(
        view.loc[view["actual_outcome"].eq("Return Approved")]
        .groupby("product_category")["order_value"]
        .sum()
        .sort_values(ascending=False),
        default="None",
    )
    top_fraud_loss_state = safe_top_index(
        view.loc[view["actual_outcome"].eq("Fraud Approved")]
        .groupby("state")["order_value"]
        .sum()
        .sort_values(ascending=False),
        default="None",
    )

    st.markdown("---")
    st.subheader("Historical Outcome Insights")
    st.info(f"""
- **{actual_returns:,}** approved returns occurred across **{total_orders:,}** completed historical orders.
- **{confirmed_fraud:,}** fraud cases were denied or blocked after review.
- **{format_currency(total_loss)}** in historical loss came from approved returns and fraud-approved orders.
- **{format_currency(avoided_loss)}** in exposure was avoided through denied returns and denied fraud.
- **{top_return_loss_category}** drove the highest approved-return loss.
- **{top_fraud_loss_state}** drove the highest fraud-approved loss by state.
""")


def render_historical_hero_summary(data: pd.DataFrame):
    view = data.copy()
    view["actual_outcome"] = view.apply(simplify_actual_outcome, axis=1)

    total_orders = len(view)
    return_count = int(view["actual_outcome"].isin(["Return Approved", "Return Denied"]).sum())
    fraud_count = int(view["actual_outcome"].isin(["Fraud Approved", "Fraud Denied"]).sum())
    denied_impact = float(
        view.loc[
            view["actual_outcome"].isin(["Return Denied", "Fraud Denied"]),
            "order_value",
        ].sum()
    )
    approved_impact = float(
        view.loc[
            view["actual_outcome"].isin(["Return Approved", "Fraud Approved"]),
            "order_value",
        ].sum()
    )
    top_return_reason = safe_top_index(
        view.loc[view["return_reason_code"].astype(str).ne("None"), "return_reason_code"].value_counts(),
        default="None",
    )
    top_fraud_reason = safe_top_index(
        view.loc[view["fraud_reason_code"].astype(str).ne("None"), "fraud_reason_code"].value_counts(),
        default="None",
    )

    st.markdown(f"""
<div class="riq-hero">
  <div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;color:#94a3b8;margin-bottom:0.4rem;">Historical Outcome Summary</div>
  <div style="font-size:0.9rem;color:#334155;line-height:1.75;">
    Across <b>{total_orders:,}</b> completed historical orders, there were
    <b>{return_count:,}</b> return outcomes and <b>{fraud_count:,}</b> fraud review outcomes.
    Denied returns and blocked fraud represented <b>{format_currency(denied_impact)}</b> in avoided exposure,
    while approved returns and released fraud reviews represented <b>{format_currency(approved_impact)}</b>.
    The most common return reason was <b>{top_return_reason}</b>, and the most common fraud reason was
    <b>{top_fraud_reason}</b>.
  </div>
</div>
""", unsafe_allow_html=True)


def render_reason_chart(container, data: pd.DataFrame, column: str, title: str, color: str):
    reason_counts = (
        data[data[column].astype(str).ne("None")][column]
        .value_counts()
        .head(8)
        .reset_index()
    )
    reason_counts.columns = [column, "count"]
    if reason_counts.empty:
        container.info(f"No {title.lower()} available for the current view.")
        return
    container.plotly_chart(
        style_chart(px.bar(
            reason_counts,
            x=column,
            y="count",
            title=title,
            labels={column: "Reason", "count": "Cases"},
            color_discrete_sequence=[color],
        ), x_tickangle=-20),
        width="stretch",
    )


def simplify_actual_outcome(row: pd.Series) -> str:
    outcome_type = str(row.get("outcome_type", ""))
    risk_type = str(row.get("risk_type", ""))
    fraud_label = str(row.get("fraud_label", ""))

    if risk_type == "Returns" or outcome_type.startswith("Return"):
        if outcome_type == "Return Approved":
            return "Return Approved"
        return "Return Denied"

    if risk_type == "Fraud" or fraud_label in {"Confirmed", "False Positive"}:
        if outcome_type == "Verified & Released" or fraud_label == "False Positive":
            return "Fraud Approved"
        return "Fraud Denied"

    return "Other"


def build_reason_outcome_breakdown(
    data: pd.DataFrame,
    outcomes: list[str],
    reason_column: str,
    value_mode: str,
) -> pd.DataFrame:
    view = data[
        data["actual_outcome"].isin(outcomes)
        & data[reason_column].astype(str).ne("None")
        & data[reason_column].astype(str).str.strip().ne("")
    ].copy()

    if view.empty:
        return pd.DataFrame(columns=[reason_column, "actual_outcome", "value"])

    if value_mode == "dollars":
        breakdown = (
            view.groupby([reason_column, "actual_outcome"], as_index=False)["order_value"]
            .sum()
            .rename(columns={"order_value": "value"})
        )
    else:
        breakdown = (
            view.groupby([reason_column, "actual_outcome"], as_index=False)
            .size()
            .rename(columns={"size": "value"})
        )

    approved_outcome = outcomes[0]
    approved_order = (
        breakdown[breakdown["actual_outcome"].eq(approved_outcome)]
        .groupby(reason_column)["value"]
        .sum()
        .sort_values(ascending=False)
    )
    total_order = breakdown.groupby(reason_column)["value"].sum().sort_values(ascending=False)
    reason_order = approved_order.index.tolist()
    reason_order.extend([reason for reason in total_order.index.tolist() if reason not in reason_order])
    reason_order = reason_order[:8]

    filtered = breakdown[breakdown[reason_column].isin(reason_order)].copy()
    filtered[reason_column] = pd.Categorical(filtered[reason_column], categories=reason_order, ordered=True)
    return filtered.sort_values([reason_column, "actual_outcome"])


def render_reason_outcome_chart(
    container,
    data: pd.DataFrame,
    outcomes: list[str],
    reason_column: str,
    title: str,
    value_mode: str,
):
    breakdown = build_reason_outcome_breakdown(
        data=data,
        outcomes=outcomes,
        reason_column=reason_column,
        value_mode=value_mode,
    )
    if breakdown.empty:
        container.info(f"No data available for {title.lower()}.")
        return

    y_label = "Dollar Impact" if value_mode == "dollars" else "Orders"
    fig = px.bar(
        breakdown,
        x=reason_column,
        y="value",
        color="actual_outcome",
        barmode="group",
        title=title,
        labels={
            reason_column: "Reason",
            "value": y_label,
            "actual_outcome": "Outcome",
        },
        color_discrete_map={
            "Return Approved": "#16a34a",
            "Return Denied": "#d97706",
            "Fraud Approved": "#0ea5e9",
            "Fraud Denied": "#dc2626",
        },
    )
    container.plotly_chart(style_chart(fig, showlegend=True, x_tickangle=-25), width="stretch")


def build_sub_reason_drilldown(
    data: pd.DataFrame,
    driver_column: str,
    selected_driver: str,
    sub_reason_column: str,
    approved_outcome: str,
    denied_outcome: str,
) -> pd.DataFrame:
    view = data[
        data[driver_column].astype(str).eq(str(selected_driver))
        & data[sub_reason_column].astype(str).ne("None")
        & data[sub_reason_column].astype(str).str.strip().ne("")
    ].copy()
    if view.empty:
        return pd.DataFrame(
            columns=[
                "sub_reason", "approved_loss", "denied_or_prevented_loss",
                "approved_cases", "denied_cases", "total_impact",
            ]
        )

    grouped = (
        view.groupby(sub_reason_column)
        .agg(
            approved_loss=("order_value", lambda s: s[view.loc[s.index, "actual_outcome"].eq(approved_outcome)].sum()),
            denied_or_prevented_loss=("order_value", lambda s: s[view.loc[s.index, "actual_outcome"].eq(denied_outcome)].sum()),
            approved_cases=("actual_outcome", lambda s: int(s.eq(approved_outcome).sum())),
            denied_cases=("actual_outcome", lambda s: int(s.eq(denied_outcome).sum())),
        )
        .reset_index()
        .rename(columns={sub_reason_column: "sub_reason"})
    )
    grouped["total_impact"] = grouped["approved_loss"] + grouped["denied_or_prevented_loss"]
    return grouped.sort_values("approved_loss", ascending=False)


def render_driver_drilldown(
    data: pd.DataFrame,
    driver_column: str,
    sub_reason_column: str,
    approved_outcome: str,
    denied_outcome: str,
    title: str,
    selector_key: str,
):
    driver_options = (
        data.loc[
            data[driver_column].astype(str).ne("None")
            & data[driver_column].astype(str).str.strip().ne(""),
            driver_column,
        ]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )
    if not driver_options:
        st.info(f"No driver detail available for {title.lower()}.")
        return

    selected_driver = st.selectbox(title, driver_options, key=selector_key)
    drilldown = build_sub_reason_drilldown(
        data=data,
        driver_column=driver_column,
        selected_driver=selected_driver,
        sub_reason_column=sub_reason_column,
        approved_outcome=approved_outcome,
        denied_outcome=denied_outcome,
    )
    if drilldown.empty:
        st.info("No sub-reason detail is available for this driver.")
        return

    chart_data = drilldown.melt(
        id_vars=["sub_reason", "approved_cases", "denied_cases"],
        value_vars=["approved_loss", "denied_or_prevented_loss"],
        var_name="impact_type",
        value_name="dollar_impact",
    )
    chart_data["impact_type"] = chart_data["impact_type"].map({
        "approved_loss": "Approved Loss",
        "denied_or_prevented_loss": "Denied / Prevented Exposure",
    })

    fig = px.bar(
        chart_data,
        x="sub_reason",
        y="dollar_impact",
        color="impact_type",
        barmode="group",
        title=f"{selected_driver} Level 2 Drivers",
        labels={
            "sub_reason": "Level 2 Driver",
            "dollar_impact": "Dollar Impact",
            "impact_type": "Impact",
            "approved_cases": "Approved Units",
            "denied_cases": "Denied Units",
        },
        hover_data={
            "approved_cases": True,
            "denied_cases": True,
            "impact_type": False,
        },
        color_discrete_map={
            "Approved Loss": "#16a34a" if approved_outcome.startswith("Return") else "#0ea5e9",
            "Denied / Prevented Exposure": "#d97706" if approved_outcome.startswith("Return") else "#dc2626",
        },
    )
    st.plotly_chart(style_chart(fig, showlegend=True, x_tickangle=-20), width="stretch")
    with st.expander("View details"):
        st.dataframe(
            drilldown.style.format({
                "approved_loss": "${:,.2f}",
                "denied_or_prevented_loss": "${:,.2f}",
                "total_impact": "${:,.2f}",
            }),
            width="stretch",
            hide_index=True,
        )


def render_historical_intelligence_view(data: pd.DataFrame):
    st.markdown(
        "Actual outcomes, reason codes, customer comments, and operations resolutions from completed cases."
    )

    historical_df = data[
        data["case_status"].astype(str).eq("Closed")
        | data["outcome_type"].astype(str).ne("Pending")
    ].copy()

    if historical_df.empty:
        st.info("No completed historical outcomes are available in the current view.")
        return

    historical_df["actual_outcome"] = historical_df.apply(simplify_actual_outcome, axis=1)

    dollar_impact = (
        historical_df[historical_df["actual_outcome"].ne("Other")]
        .groupby("actual_outcome", as_index=False)["order_value"]
        .sum()
    )
    dollar_impact = (
        dollar_impact.set_index("actual_outcome")
        .reindex(["Return Approved", "Return Denied", "Fraud Approved", "Fraud Denied"], fill_value=0)
        .reset_index()
    )
    impact_lookup = dict(zip(dollar_impact["actual_outcome"], dollar_impact["order_value"]))
    di1, di2, di3, di4 = st.columns(4)
    di1.metric("Return Approved $", format_currency(float(impact_lookup.get("Return Approved", 0))))
    di2.metric("Return Denied $", format_currency(float(impact_lookup.get("Return Denied", 0))))
    di3.metric("Fraud Approved $", format_currency(float(impact_lookup.get("Fraud Approved", 0))))
    di4.metric("Fraud Denied $", format_currency(float(impact_lookup.get("Fraud Denied", 0))))

    render_reason_outcome_chart(
        st,
        historical_df,
        ["Return Approved", "Return Denied"],
        "return_reason_code",
        "Return Approved vs Denied by Reason (Dollars)",
        "dollars",
    )
    render_reason_outcome_chart(
        st,
        historical_df,
        ["Fraud Approved", "Fraud Denied"],
        "fraud_reason_code",
        "Fraud Approved vs Denied by Reason (Dollars)",
        "dollars",
    )

    st.markdown("### Next-Level Driver Charts")
    st.caption("Select a top-level driver to chart the Level 2 sub-reasons contributing to approved loss and denied/prevented exposure.")
    render_driver_drilldown(
        historical_df,
        driver_column="return_reason_code",
        sub_reason_column="return_sub_reason",
        approved_outcome="Return Approved",
        denied_outcome="Return Denied",
        title="Return Driver",
        selector_key="historical_return_driver_drilldown",
    )
    render_driver_drilldown(
        historical_df,
        driver_column="fraud_reason_code",
        sub_reason_column="fraud_sub_reason",
        approved_outcome="Fraud Approved",
        denied_outcome="Fraud Denied",
        title="Fraud Driver",
        selector_key="historical_fraud_driver_drilldown",
    )

    with st.expander("Show raw filtered data"):
        historical_view = historical_df[
            [
                "order_id", "customer_id", "actual_outcome", "outcome_type",
                "return_reason_code", "return_sub_reason", "fraud_label", "fraud_reason_code", "fraud_sub_reason",
                "customer_return_comment", "customer_dispute_comment",
                "ops_resolution", "ops_resolution_note", "reviewed_by", "reviewed_at",
            ]
        ].head(100)
        st.dataframe(historical_view, width="stretch", hide_index=True)


def render_live_case_feedback_summary(data: pd.DataFrame):
    st.markdown(
        "Feedback from cases operations has reviewed. Each saved case insight becomes a learning signal for future model improvement."
    )

    learning_df = data.copy()
    learning_df["ops_override_clean"] = learning_df["ops_override_reason"].astype(str).str.strip()

    pending_labels = learning_df[
        learning_df["outcome_type"].astype(str).eq("Pending")
        | learning_df["label_source"].astype(str).eq("Model Prediction")
    ]
    case_insights = learning_df[learning_df["ops_override_clean"].ne("")]
    final_label_candidates = learning_df[
        learning_df["case_status"].astype(str).eq("Closed")
        & learning_df["outcome_type"].astype(str).ne("Pending")
    ]

    live_insights = load_learning_backlog("live")

    fb1, fb2, fb3, fb4 = st.columns(4)
    fb1.metric("Reviewed Labels", f"{len(final_label_candidates):,}")
    fb2.metric("Case Insights", f"{len(case_insights):,}")
    fb3.metric("Pending Labels", f"{len(pending_labels):,}")
    fb4.metric("Example Notes", f"{len(live_insights):,}")

    st.markdown("#### Case Insights Captured")
    st.caption(
        "Operations adds these insights while reviewing individual cases. "
        "Analytics can later evaluate them for model retraining or policy changes."
    )
    if case_insights.empty:
        st.info("No case insights have been captured in the current live view yet. Add one from the selected-case form.")
    else:
        feedback_view = case_insights[
            [
                "order_id", "risk_type_display", "priority_tier", "recommended_action",
                "ops_action_taken", "ops_resolution", "ops_resolution_reason",
                "ops_override_reason", "ops_action_by", "ops_action_note",
                "fraud_score", "return_risk_score", "risk_reason_summary",
            ]
        ].head(50)
        st.dataframe(
            feedback_view.style.format({
                "fraud_score": "{:.3f}",
                "return_risk_score": "{:.3f}",
            }),
            width="stretch",
            hide_index=True,
        )


def apply_ops_action(
    data: pd.DataFrame,
    order_id: str,
    action: str,
    actor: str,
    note: str,
    final_decision: str = "Pending",
    resolution_reason: str = "",
    override_reason: str = "",
    analyst_confidence: float = 0.0,
) -> pd.DataFrame:
    updated = ensure_ops_action_tracking(data)
    if action not in OPS_ACTIONS:
        raise ValueError(f"Unsupported ops action: {action}")

    action_mask = updated["order_id"].astype(str) == str(order_id)
    if not action_mask.any():
        raise ValueError(f"Order ID not found: {order_id}")

    action_at = datetime.now(timezone.utc).isoformat()
    history_event = {
        "ts": action_at,
        "action": action,
        "actor": actor.strip() or "Operations Team",
        "note": note.strip(),
        "case_status": OPS_ACTIONS[action]["case_status"],
        "workflow_stage": OPS_ACTIONS[action]["workflow_stage"],
        "latest_action_taken": OPS_ACTIONS[action]["latest_action_taken"],
    }

    existing_history = str(
        updated.loc[action_mask, "ops_action_history"].iloc[0] or ""
    ).strip()
    try:
        action_history = json.loads(existing_history) if existing_history else []
        if not isinstance(action_history, list):
            action_history = []
    except json.JSONDecodeError:
        action_history = []
    action_history.append(history_event)

    for column, value in OPS_ACTIONS[action].items():
        updated.loc[action_mask, column] = value

    updated.loc[action_mask, "ops_action_taken"] = action
    updated.loc[action_mask, "ops_action_by"] = actor.strip() or "Operations Team"
    updated.loc[action_mask, "ops_action_at"] = action_at
    updated.loc[action_mask, "ops_action_note"] = note.strip()
    updated.loc[action_mask, "ops_action_history"] = json.dumps(action_history)
    updated.loc[action_mask, "ops_override_reason"] = override_reason.strip()
    updated.loc[action_mask, "analyst_confidence"] = float(analyst_confidence)

    if final_decision != "Pending":
        resolution_updates = {
            "Order Approved": {
                "outcome_type": "Approved",
                "ops_resolution": "Order Approved",
            },
            "Return Approved": {
                "outcome_type": "Return Approved",
                "ops_resolution": "Return Approved",
                "return_requested": 1,
                "return_completed": 1,
            },
            "Return Denied": {
                "outcome_type": "Return Reviewed & Denied",
                "ops_resolution": "Return Denied",
                "return_requested": 1,
                "return_completed": 0,
            },
            "Fraud Confirmed": {
                "outcome_type": "Fraud Confirmed & Blocked",
                "ops_resolution": "Fraud Confirmed",
                "fraud_label": "Confirmed",
            },
            "Fraud False Positive": {
                "outcome_type": "Verified & Released",
                "ops_resolution": "Fraud False Positive",
                "fraud_label": "False Positive",
            },
        }
        if final_decision not in resolution_updates:
            raise ValueError(f"Unsupported final decision: {final_decision}")
        for column, value in resolution_updates[final_decision].items():
            updated.loc[action_mask, column] = value
        updated.loc[action_mask, "case_status"] = "Closed"
        updated.loc[action_mask, "workflow_stage"] = "Resolved"
        updated.loc[action_mask, "label_source"] = "Ops Review"
        updated.loc[action_mask, "label_confidence"] = float(analyst_confidence)
        updated.loc[action_mask, "reviewed_by"] = actor.strip() or "Operations Team"
        updated.loc[action_mask, "reviewed_at"] = action_at
        updated.loc[action_mask, "ops_resolution_reason"] = resolution_reason
        updated.loc[action_mask, "ops_resolution_note"] = (
            note.strip() or f"Operations recorded {final_decision.lower()} after review."
        )
        risk_type = str(updated.loc[action_mask, "risk_type"].iloc[0])
        if resolution_reason and risk_type == "Fraud":
            updated.loc[action_mask, "fraud_reason_code"] = resolution_reason
        elif resolution_reason and risk_type == "Returns":
            updated.loc[action_mask, "return_reason_code"] = resolution_reason
    return updated


def add_ops_action_session_event(case_row: pd.Series, action: str, actor: str, note: str):
    event = {
        "ts": str(case_row.get("ops_action_at", datetime.now(timezone.utc).isoformat())),
        "order_id": str(case_row.get("order_id", "")),
        "customer_id": str(case_row.get("customer_id", "")),
        "action": action,
        "actor": actor.strip() or "Operations Team",
        "note": note.strip(),
        "case_status": str(case_row.get("case_status", "")),
        "workflow_stage": str(case_row.get("workflow_stage", "")),
        "latest_action_taken": str(case_row.get("latest_action_taken", "")),
    }
    st.session_state.setdefault("ops_action_log", []).insert(0, event)


def get_action_persistence_path(source: str) -> str:
    if source == "saved_shopify":
        return SHOPIFY_CACHE_PATH
    if source == "shopify":
        return SHOPIFY_CACHE_PATH
    return "data/transactions.csv"


def persist_action_updates(data: pd.DataFrame, source: str) -> str:
    if source == "csv" and "dataset_mode" in data.columns:
        historical = data[data["dataset_mode"].astype(str).eq("Historical")].copy()
        live = data[data["dataset_mode"].astype(str).eq("Live")].copy()
        saved_paths = []
        if not historical.empty:
            saved_paths.append(save_synced_data(historical, path="data/transactions.csv"))
        if not live.empty:
            saved_paths.append(save_synced_data(live, path=LIVE_TRANSACTIONS_PATH))
        load_data.clear()
        return ", ".join(saved_paths)

    path = get_action_persistence_path(source)
    saved_path = save_synced_data(data, path=path)
    load_data.clear()
    return saved_path


def get_config_value(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


def get_user_id() -> str:
    return "anonymous"


def log_event(event_name: str, details: dict | None = None):
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": get_session_id(),
        "user_id": get_user_id(),
        "event": event_name,
        "details": details or {}
    }

    st.session_state.setdefault("event_log", []).append(event)
    print(json.dumps(event))


def log_once(key: str, event_name: str, details: dict | None = None):
    if not st.session_state.get(key):
        log_event(event_name, details)
        st.session_state[key] = True


def log_value_change(state_key: str, event_name: str, current_value):
    if state_key not in st.session_state:
        st.session_state[state_key] = current_value
        log_event(f"{event_name}_initialized", current_value if isinstance(
            current_value, dict) else {"value": current_value})
    elif st.session_state[state_key] != current_value:
        previous_value = st.session_state[state_key]
        log_event(event_name, {"before": previous_value,
                  "after": current_value})
        st.session_state[state_key] = current_value


def build_dashboard_context(data: pd.DataFrame) -> str:
    if data.empty:
        return "No dashboard data available."

    total_orders = len(data)
    flagged_orders = int((data["risk_level"] == "High").sum())
    flagged_pct = flagged_orders / total_orders * 100 if total_orders else 0
    revenue_at_risk = float(
        data.loc[data["risk_level"] == "High", "order_value"].sum())
    open_cases = int((data["case_status"] == "Open").sum())
    high_priority_cases = int((data["priority_tier"] == "High").sum())
    fraud_cases = int((data["risk_type_display"] == "Fraud Risk").sum())
    return_cases = int((data["risk_type_display"] == "Return Risk").sum())
    protected_revenue = float(
        data.loc[
            (data["risk_level"] == "High") & (data["case_status"] == "Closed"),
            "order_value"
        ].sum()
    )

    top_category = (
        data.groupby("product_category")["return_risk_score"]
        .mean()
        .sort_values(ascending=False)
        .head(3)
    )

    top_states = (
        data.groupby("state")["fraud_score"]
        .mean()
        .sort_values(ascending=False)
        .head(3)
    )

    top_actions = data["recommended_action"].value_counts().head(5)

    return f"""
Aignita Risk Intelligence Dashboard Summary

Total orders: {total_orders}
Flagged orders: {flagged_orders}
Flagged percent: {flagged_pct:.1f}%
Revenue at risk: ${revenue_at_risk:,.0f}
Protected revenue: ${protected_revenue:,.0f}

Open cases: {open_cases}
High-priority cases: {high_priority_cases}

Fraud cases: {fraud_cases}
Return abuse cases: {return_cases}

Top return abuse categories:
{top_category.to_string()}

Top fraud states:
{top_states.to_string()}

Most common actions:
{top_actions.to_string()}
"""


def build_historical_dashboard_context(data: pd.DataFrame) -> str:
    if data.empty:
        return "No historical outcome data available."

    view = data.copy()
    view["actual_outcome"] = view.apply(simplify_actual_outcome, axis=1)
    total_orders = len(view)
    actual_returns = int(view["actual_outcome"].eq("Return Approved").sum())
    return_denied = int(view["actual_outcome"].eq("Return Denied").sum())
    fraud_approved = int(view["actual_outcome"].eq("Fraud Approved").sum())
    fraud_denied = int(view["actual_outcome"].eq("Fraud Denied").sum())
    total_loss = calculate_historical_total_loss(view)
    avoided_loss = float(
        view.loc[
            view["actual_outcome"].isin(["Return Denied", "Fraud Denied"]),
            "order_value",
        ].sum()
    )

    return_loss_by_category = (
        view.loc[view["actual_outcome"].eq("Return Approved")]
        .groupby("product_category")["order_value"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    return_loss_by_product = (
        view.loc[view["actual_outcome"].eq("Return Approved")]
        .groupby("product_name")["order_value"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    fraud_loss_by_state = (
        view.loc[view["actual_outcome"].eq("Fraud Approved")]
        .groupby("state")["order_value"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    fraud_loss_by_payment = (
        view.loc[view["actual_outcome"].eq("Fraud Approved")]
        .groupby("payment_method")["order_value"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    return_reasons = (
        view.loc[view["actual_outcome"].eq("Return Approved"), "return_reason_code"]
        .value_counts()
        .head(5)
    )
    fraud_reasons = (
        view.loc[view["actual_outcome"].eq("Fraud Approved"), "fraud_reason_code"]
        .value_counts()
        .head(5)
    )

    return f"""
Aignita Risk Intelligence Historical Outcome Summary

Completed historical orders: {total_orders}
Actual approved returns: {actual_returns}
Denied returns: {return_denied}
Fraud-approved/released cases: {fraud_approved}
Fraud denied/blocked cases: {fraud_denied}

Total historical loss from approved returns and fraud-approved orders: ${total_loss:,.0f}
Avoided exposure from denied returns and denied fraud: ${avoided_loss:,.0f}

Approved return loss by product category:
{return_loss_by_category.to_string()}

Approved return loss by product:
{return_loss_by_product.to_string()}

Fraud-approved loss by state:
{fraud_loss_by_state.to_string()}

Fraud-approved loss by payment method:
{fraud_loss_by_payment.to_string()}

Top approved return reasons:
{return_reasons.to_string()}

Top fraud-approved reasons:
{fraud_reasons.to_string()}
"""


def build_case_context(case_row: pd.Series | None) -> str:
    if case_row is None:
        return "No selected case context."

    device_risk = case_row.get("device_risk", "Unknown")
    velocity = case_row.get("velocity_flag", 0)
    lifetime = case_row.get("customer_lifetime_days", 0)
    addr_mismatch = "Yes" if int(case_row.get("address_mismatch", 0)) else "No"
    cb_reason = case_row.get("chargeback_reason", "None")
    cb_won = case_row.get("chargebacks_won", 0)
    action_reason = case_row.get("action_reason", "")
    return_reason = case_row.get("return_reason_code", "None")
    return_comment = case_row.get("customer_return_comment", "")
    fraud_label = case_row.get("fraud_label", "Unknown")
    fraud_reason = case_row.get("fraud_reason_code", "None")
    dispute_comment = case_row.get("customer_dispute_comment", "")
    ops_resolution = case_row.get("ops_resolution", "Pending")
    label_source = case_row.get("label_source", "System")

    return f"""
Selected Order Case Details

Order ID: {case_row["order_id"]}
Customer ID: {case_row["customer_id"]}
Customer Lifetime: {lifetime} days

Risk Type: {case_row["risk_type_display"]}
Priority Tier: {case_row["priority_tier"]}
Case Status: {case_row["case_status"]}
Workflow Stage: {case_row["workflow_stage"]}
Latest Action Taken: {case_row.get("latest_action_taken", "No action yet")}

Order Value: ${case_row["order_value"]}

Fraud Score: {case_row["fraud_score"]}
Return Risk Score: {case_row["return_risk_score"]}
Unified Risk Score: {case_row["unified_risk_score"]}

Behavioral Signals:
- Device Risk: {device_risk}
- Order Velocity (last 24h): {velocity} orders from this customer
- Billing/Shipping Address Mismatch: {addr_mismatch}
- Past Returns: {case_row["past_returns"]}
- Chargeback History: {case_row["chargeback_history"]}
- Chargeback Reason: {cb_reason}
- Chargebacks Won in Appeals: {cb_won}

Historical / Feedback Signals:
- Return Reason: {return_reason}
- Customer Return Comment: {return_comment}
- Fraud Label: {fraud_label}
- Fraud Reason: {fraud_reason}
- Customer Dispute Comment: {dispute_comment}
- Ops Resolution: {ops_resolution}
- Label Source: {label_source}

Risk Explanation:
{case_row["risk_reason_summary"]}

Action Decision Reason:
{action_reason}

Recommended Action:
{case_row["recommended_action"]}

AI Agent Suggested Step:
{case_row["ai_agent_next_step"]}
"""


def ask_groq(prompt: str, system_prompt: str, max_tokens: int = 600) -> str:
    if groq_client is None:
        return "Groq API key not configured. Add GROQ_API_KEY to your .env file or Streamlit secrets."

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Groq request failed: {e}"


def ask_risk_copilot(
    question: str,
    dashboard_data: pd.DataFrame,
    work_mode: str = "Live Risk Queue",
) -> str:
    is_historical = work_mode == "Historical Intelligence"
    dashboard_context = (
        build_historical_dashboard_context(dashboard_data)
        if is_historical
        else build_dashboard_context(dashboard_data)
    )

    if is_historical:
        system_prompt = """
You are Aignita Risk Intelligence Copilot, an AI risk analyst for e-commerce merchants.

You help teams understand historical return and fraud outcomes, actual loss, avoided exposure, reason patterns, and where historical losses occurred.

Rules:
- Only use the provided context.
- Be concise, practical, and business-focused.
- Speak in actual outcomes and dollars, not model scores or live queue language.
- Do not recommend reviewing open cases unless the context says there are open cases.
- Do not invent facts not present in the context.
- Format responses using markdown: use **bold** for key numbers and metrics, and bullet points (- ) for lists.
"""
    else:
        system_prompt = """
You are Aignita Risk Intelligence Copilot, an AI risk analyst for e-commerce merchants.

You help risk and operations teams understand portfolio-level fraud risk, return abuse, and what actions to take across the current filtered queue.

Rules:
- Only use the provided context.
- Be concise, practical, and business-focused.
- Prioritize open critical cases first.
- Answer at the portfolio level. Do not analyze an individual order unless the dashboard context itself names that order.
- Do not invent facts not present in the context.
- Format responses using markdown: use **bold** for key numbers and metrics, and bullet points (- ) for lists.
"""

    user_prompt = f"""
Dashboard Context:
{dashboard_context}

Merchant Question:
{question}
"""
    return ask_groq(user_prompt, system_prompt, max_tokens=500)


def ask_case_copilot(question: str, selected_case: pd.Series | None) -> str:
    case_context = build_case_context(selected_case)

    system_prompt = """
You are Aignita Risk Intelligence Case Analyst Assistant.

You help fraud and risk analysts investigate a selected case.

Rules:
- Only use the supplied case context.
- Be concise and operational.
- Focus on what the analyst should understand and do next.
- Do not invent evidence not present in the context.
- Format responses using markdown: use **bold** for key numbers and signals, and bullet points (- ) for lists.
"""

    user_prompt = f"""
Selected Case Context:
{case_context}

Analyst Question:
{question}
"""
    return ask_groq(user_prompt, system_prompt, max_tokens=400)


def generate_ai_output(
    artifact_type: str,
    dashboard_data: pd.DataFrame,
    selected_case: pd.Series | None = None,
    work_mode: str = "Live Risk Queue",
) -> str:
    is_historical = work_mode == "Historical Intelligence"
    dashboard_context = (
        build_historical_dashboard_context(dashboard_data)
        if is_historical
        else build_dashboard_context(dashboard_data)
    )
    case_context = build_case_context(selected_case)

    if is_historical:
        system_prompt = """
You are Aignita Risk Intelligence Copilot.

Generate practical historical outcome analysis for risk, finance, operations, and analytics teams.
Ground everything in the supplied context.
Use actual outcomes and dollars, not live-risk queue language or model scores.
Do not claim actions were executed.
Format responses using markdown: use **bold** for key numbers and metrics, and bullet points (- ) for lists.
"""
    else:
        system_prompt = """
You are Aignita Risk Intelligence Copilot.

Generate practical outputs for risk and operations teams.
Ground everything in the supplied context.
Be concrete, concise, and ready to use.
Do not claim actions were executed.
Format responses using markdown: use **bold** for key numbers and metrics, and bullet points (- ) for lists.
"""

    if is_historical:
        artifact_prompts = {
            "merchant_summary": """
Create a short merchant-facing historical outcome summary.
Include:
- actual return and fraud outcome situation
- total historical loss and avoided exposure
- biggest product/category, state, payment, or reason drivers
- what the merchant should learn from the historical period
Use plain business language.
""",
            "policy_recommendation": """
Write a short policy recommendation memo based on historical outcomes.
Focus on reducing approved return loss and fraud-approved loss.
Include:
- 3 recommended policy or operational changes
- expected benefit
- possible customer or operational tradeoff
""",
            "merchant_action_brief": """
Create a merchant-facing historical outcome brief.
Include:
- what actually happened in the historical period
- top return-loss and fraud-loss drivers
- total loss and avoided exposure
- one operational next step and one analytics/model-learning next step
Avoid live queue language and do not discuss open case prioritization.
""",
            "customer_verification_draft": """
Historical mode does not generate customer verification drafts. Explain that this artifact is only available for selected live cases.
""",
            "ops_action_plan": """
Historical mode does not create live case action plans. Create a short historical operations improvement plan instead.
Include:
- what policies should be reviewed
- what data labels or reason codes should be improved
- what teams should monitor next period
""",
        }
    else:
        artifact_prompts = {
        "merchant_summary": """
Create a short merchant-facing risk summary for the current dashboard view.
Include:
- current risk situation
- biggest fraud/return risk drivers
- revenue at risk vs protected revenue
- what should be prioritized next
Use plain business language.
""",
        "policy_recommendation": """
Write a short policy recommendation memo for the merchant based on the dashboard context.
Focus on return risk and fraud controls.
Include:
- 3 recommended policy changes
- expected benefit
- possible tradeoff
""",
        "customer_verification_draft": """
Draft a customer verification message for the selected case.
Tone: polite, professional, non-accusatory.
Ask for only the minimum needed verification.
If no selected case exists, say that a selected case is required.
""",
        "ops_action_plan": """
Create a short ops action plan for the selected case.
Include:
- what the analyst should review first
- what evidence matters most
- whether the case should be escalated, verified, monitored, or closed
Keep it practical and concise.
""",
        "merchant_action_brief": """
Create a merchant-facing action brief for the current dashboard view.
Include:
- current risk situation in plain business language
- top 3 actions the merchant should take today
- which order groups, product categories, locations, or actions are driving risk
- revenue at risk and protected revenue
- one operational next step and one policy next step
Avoid technical model language and do not claim actions were executed.
""",
        }

    prompt = f"""
Dashboard Context:
{dashboard_context}

Case Context:
{case_context}

Task:
{artifact_prompts[artifact_type]}
"""
    return ask_groq(prompt, system_prompt, max_tokens=700)


# ── Data load ────────────────────────────────────────────────────────────────────

st.sidebar.markdown("### Demo Data")
with st.sidebar.expander("Advanced data options", expanded=False):
    data_source_label = st.selectbox(
        "Choose where Risk Intelligence loads merchant data from",
        ["Demo CSV", "Shopify API", "Saved Shopify CSV"],
        index=0,
    )

data_source = {
    "Demo CSV": "csv",
    "Shopify API": "shopify",
    "Saved Shopify CSV": "saved_shopify",
}[data_source_label]
shop_domain = get_config_value("SHOPIFY_SHOP_DOMAIN")
shop_token = get_config_value("SHOPIFY_ACCESS_TOKEN")
shopify_limit = 100

if data_source == "shopify":
    with st.sidebar.expander("Shopify sync settings", expanded=True):
        st.caption("Uses Shopify Admin API credentials from Streamlit secrets or environment variables.")
        shopify_limit = st.number_input(
            "Shopify orders to sync",
            min_value=10,
            max_value=250,
            value=100,
            step=10,
        )
        if st.button("Sync latest Shopify orders"):
            load_data.clear()
            st.session_state.pop("_active_data_source", None)
            st.rerun()
elif data_source == "saved_shopify":
    st.sidebar.caption(f"Loads the last saved Shopify sync from {SHOPIFY_CACHE_PATH}.")

try:
    data_cache_version = build_data_cache_version(data_source)
    df = load_data(
        source=data_source,
        shop_domain=shop_domain,
        access_token=shop_token,
        limit=int(shopify_limit),
        cache_version=data_cache_version,
    )
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    if data_source == "shopify":
        st.info(
            "To use Shopify API mode, add SHOPIFY_SHOP_DOMAIN and "
            "SHOPIFY_ACCESS_TOKEN in Streamlit secrets. Switch back to Demo CSV "
            "to use the local synthetic dataset."
        )
    elif data_source == "saved_shopify":
        st.info(
            "No saved Shopify sync was found yet. Select Shopify API, sync orders, "
            "then click Save Shopify Sync to CSV."
        )
    st.stop()

render_page_header()

log_once("session_started_logged", "session_start", {"app": "Aignita Risk Intelligence"})

if df is None:
    if data_source == "saved_shopify":
        st.error(f"{SHOPIFY_CACHE_PATH} not found.")
        st.info("Select Shopify API and save a sync before using Saved Shopify CSV.")
    else:
        st.error("data/transactions.csv not found.")
        st.code("python generate_data.py")
    st.stop()

if data_source == "csv":
    live_df = load_csv_transactions(LIVE_TRANSACTIONS_PATH)
    if live_df is not None:
        df["dataset_mode"] = df.get("dataset_mode", "Historical")
        live_df["dataset_mode"] = live_df.get("dataset_mode", "Live")
        df = pd.concat([df, live_df], ignore_index=True)

df = ensure_lifecycle_intelligence(ensure_ops_action_tracking(ensure_latest_action_taken(df)))

working_data_key = f"{data_source_label}:{shopify_limit}:split-datasets-v2:{build_data_cache_version(data_source)}"
if st.session_state.get("_working_data_key") != working_data_key:
    st.session_state["working_df"] = df.copy()
    st.session_state["_working_data_key"] = working_data_key
else:
    df = ensure_lifecycle_intelligence(ensure_ops_action_tracking(st.session_state["working_df"]))

if data_source == "shopify":
    st.sidebar.markdown("---")
    if st.sidebar.button("Save Shopify Sync to CSV"):
        saved_path = save_synced_data(st.session_state["working_df"])
        load_data.clear()
        st.sidebar.success(f"Saved {len(df):,} rows to {saved_path}")
        log_event("shopify_sync_saved", {
            "rows": len(df),
            "path": saved_path,
        })
elif data_source in {"csv", "saved_shopify"}:
    st.sidebar.markdown("---")
    if st.sidebar.button("Save Action Updates to CSV"):
        saved_path = persist_action_updates(st.session_state["working_df"], data_source)
        st.sidebar.success(f"Saved action updates to {saved_path}")
        log_event("ops_action_updates_saved", {
            "rows": len(st.session_state["working_df"]),
            "path": saved_path,
        })

if data_source == "csv":
    if st.sidebar.button("Reset Demo State"):
        try:
            reset_demo_state()
            st.sidebar.success("Demo snapshot restored.")
            st.rerun()
        except FileNotFoundError as e:
            st.sidebar.error(str(e))

if st.session_state.get("_active_data_source") != data_source_label:
    for _key in list(st.session_state.keys()):
        if _key.startswith("filt_"):
            del st.session_state[_key]
    st.session_state.pop("filt_init", None)
    st.session_state["_active_data_source"] = data_source_label
    st.rerun()

render_demo_banner()

st.sidebar.markdown("### Work Mode")
work_mode = st.sidebar.radio(
    "Analyze",
    WORK_MODE_OPTIONS,
    index=0,
    horizontal=False,
)
log_value_change("last_work_mode", "work_mode_changed", {"work_mode": work_mode})
if st.session_state.get("_active_work_mode") != work_mode:
    for _key in list(st.session_state.keys()):
        if _key.startswith("filt_"):
            del st.session_state[_key]
    st.session_state.pop("filt_init", None)
    st.session_state["_active_work_mode"] = work_mode

mode_df = apply_work_mode(df, work_mode)
if mode_df.empty:
    st.warning(f"No rows are available for {work_mode}. Showing the full dataset instead.")
    mode_df = df.copy()

# ── Filter state initialization ──────────────────────────────────────────────────
# Build defaults once (all options selected) so mode changes can reset cleanly.
_filter_defaults: dict = {
    "filt_category":    sorted(mode_df["product_category"].unique()),
    "filt_payment":     sorted(mode_df["payment_method"].unique()),
    "filt_state":       sorted(mode_df["state"].unique()),
    "filt_priority":    sorted(mode_df["priority_tier"].unique()),
    "filt_action":      sorted(mode_df["recommended_action"].unique()),
    "filt_workflow":    sorted(mode_df["workflow_stage"].unique()),
    "filt_case_status": sorted(mode_df["case_status"].unique()),
}
if "filt_init" not in st.session_state:
    for _k, _v in _filter_defaults.items():
        st.session_state[_k] = _v
    st.session_state["filt_init"] = True

for _k, _valid_options in _filter_defaults.items():
    _current_values = st.session_state.get(_k, _valid_options)
    _current_values = _current_values if isinstance(_current_values, list) else list(_current_values)
    _valid_set = set(_valid_options)
    _clean_values = [value for value in _current_values if value in _valid_set]
    st.session_state[_k] = _clean_values or _valid_options

st.sidebar.markdown("---")
st.sidebar.header("Filters")
st.sidebar.caption("Filters apply across all tabs and AI outputs.")

date_min = mode_df["order_date"].min().date()
date_max = mode_df["order_date"].max().date()

date_range = st.sidebar.date_input(
    "Order Date Range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max
)

category_filter = st.sidebar.multiselect(
    "Product Category",
    options=sorted(mode_df["product_category"].unique()),
    key="filt_category",
)

payment_filter = st.sidebar.multiselect(
    "Payment Method",
    options=sorted(mode_df["payment_method"].unique()),
    key="filt_payment",
)

state_filter = st.sidebar.multiselect(
    "State",
    options=sorted(mode_df["state"].unique()),
    key="filt_state",
)

priority_filter = st.sidebar.multiselect(
    "Priority Tier",
    options=sorted(mode_df["priority_tier"].unique()),
    key="filt_priority",
)

action_filter = st.sidebar.multiselect(
    "Recommended Action",
    options=sorted(mode_df["recommended_action"].unique()),
    key="filt_action",
)

workflow_filter = st.sidebar.multiselect(
    "Workflow Stage",
    options=sorted(mode_df["workflow_stage"].unique()),
    key="filt_workflow",
)

case_status_filter = st.sidebar.multiselect(
    "Case Status",
    options=sorted(mode_df["case_status"].unique()),
    key="filt_case_status",
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = date_min, date_max

current_filters = {
    "start_date": str(start_date),
    "end_date": str(end_date),
    "category_filter": category_filter,
    "payment_filter": payment_filter,
    "state_filter": state_filter,
    "priority_filter": priority_filter,
    "action_filter": action_filter,
    "workflow_filter": workflow_filter,
    "case_status_filter": case_status_filter,
}
log_value_change("last_filters", "filters_changed", current_filters)

filtered_df = mode_df[
    (mode_df["state"].isin(state_filter)) &
    (mode_df["product_category"].isin(category_filter)) &
    (mode_df["payment_method"].isin(payment_filter)) &
    (mode_df["recommended_action"].isin(action_filter)) &
    (mode_df["priority_tier"].isin(priority_filter)) &
    (mode_df["case_status"].isin(case_status_filter)) &
    (mode_df["workflow_stage"].isin(workflow_filter)) &
    (mode_df["order_date"].dt.date >= start_date) &
    (mode_df["order_date"].dt.date <= end_date)
].copy()

if filtered_df.empty:
    st.warning("No data found for the current filter selection.")
    st.stop()

metrics = compute_summary_metrics(filtered_df)

top_category = safe_top_index(
    filtered_df.groupby("product_category")["return_risk_score"]
    .mean()
    .sort_values(ascending=False)
)

top_state = safe_top_index(
    filtered_df.groupby("state")["fraud_score"]
    .mean()
    .sort_values(ascending=False)
)

review_window = f"{start_date} to {end_date}"
render_merchant_context_bar(
    DEMO_MERCHANT_NAME,
    "Shopify API" if data_source == "shopify" else DEMO_PLATFORM,
    review_window,
    len(filtered_df),
    "Live API sync" if data_source == "shopify" else DEMO_LAST_SYNC,
)

render_mode_panel(work_mode, filtered_df)
if work_mode == "Historical Intelligence":
    render_historical_hero_summary(filtered_df)
else:
    render_hero_summary(metrics, top_category, top_state)

c1, c2, c3, c4, c5, c6 = st.columns(6)
if work_mode == "Historical Intelligence":
    historical_kpis = build_historical_kpis(filtered_df)
    c1.metric("Total Historical Orders", f'{metrics["total_orders"]:,}')
    c2.metric("Actual Returns", f'{historical_kpis["actual_returns"]:,}')
    c3.metric("Actual Return Rate", f'{historical_kpis["actual_return_rate"]:.1f}%')
    c4.metric("Confirmed Fraud", f'{historical_kpis["confirmed_fraud"]:,}')
    c5.metric("Confirmed Fraud Rate", f'{historical_kpis["confirmed_fraud_rate"]:.1f}%')
    c6.metric("Total Loss", format_currency(calculate_historical_total_loss(filtered_df)))
else:
    c1.metric("New / Open Orders", f'{metrics["total_orders"]:,}')
    c2.metric("Predicted Flagged", f'{metrics["flagged_orders"]:,}')
    c3.metric("Predicted Flagged %", f'{metrics["flagged_pct"]:.1f}%')
    c4.metric("Predicted Fraud Cases", f'{metrics["fraud_cases"]:,}')
    c5.metric("Predicted Return Cases", f'{metrics["return_cases"]:,}')
    c6.metric("Revenue at Risk", format_currency(metrics["revenue_at_risk"]))

if work_mode == "Live Risk Queue":
    today_actions = build_today_actions(filtered_df, metrics, top_category, top_state)
    render_today_action_panel(today_actions)

st.markdown("---")

if work_mode == "Historical Intelligence":
    tab_ov, tab_cp, tab_fr, tab_rt, tab_model = st.tabs(
        ["Overview", "Risk Copilot", "Fraud Outcomes", "Return Outcomes", "Model Learning"]
    )
    tab_rq = None
else:
    tab_ov, tab_cp, tab_rq, tab_fr, tab_rt = st.tabs(
        ["Overview", "Risk Copilot", "Review Queue", "Fraud Risk", "Return Risk"]
    )
    tab_model = None

# ── Tab 1: Overview ──────────────────────────────────────────────────────────────

with tab_ov:
    log_once("tab_overview_seen", "tab_view", {"tab": "Overview"})
    st.subheader("Overview")

    if work_mode == "Historical Intelligence":
        render_historical_intelligence_view(filtered_df)
    else:
        col1, col2 = st.columns(2)

        trend_df = (
            filtered_df.assign(order_day=filtered_df["order_date"].dt.date)
            .groupby("order_day", as_index=False)
            .agg(
                total_orders=("order_id", "count"),
                flagged_orders=("risk_level", lambda s: (s == "High").sum())
            )
        )
        trend_df["flagged_rate"] = (
            pd.to_numeric(trend_df["flagged_orders"], errors="coerce")
            / pd.to_numeric(trend_df["total_orders"], errors="coerce")
            * 100
        ).round(2)

        if trend_df.empty:
            col1.info("No trend data available for the selected filters.")
        else:
            col1.plotly_chart(
                style_chart(px.line(
                    trend_df, x="order_day", y="flagged_rate",
                    title="Flagged Order Rate Over Time (%)",
                    labels={"order_day": "Order Date", "flagged_rate": "Flagged Rate (%)"},
                    color_discrete_sequence=["#2563eb"],
                )),
                width="stretch"
            )

        loss_exposure = (
            filtered_df[filtered_df["risk_type_display"].isin(["Fraud Risk", "Return Risk"])]
            .groupby("risk_type_display", as_index=False)["order_value"]
            .sum()
            .sort_values("order_value", ascending=False)
        )

        if loss_exposure.empty:
            col2.info("No loss exposure data available for the selected filters.")
        else:
            col2.plotly_chart(
                style_chart(px.bar(
                    loss_exposure, x="risk_type_display", y="order_value",
                    title="Loss Exposure by Risk Type",
                    labels={"risk_type_display": "Risk Type", "order_value": "Loss Exposure ($)"},
                    color="risk_type_display",
                    color_discrete_map=RISK_TYPE_COLORS,
                ), showlegend=False),
                width="stretch"
            )

        col3, col4 = st.columns(2)

        flagged_priority = (
            filtered_df[filtered_df["risk_level"] == "High"]["priority_tier"]
            .value_counts()
            .reindex(["High", "Moderate"], fill_value=0)
            .reset_index()
        )
        flagged_priority.columns = ["priority_tier", "count"]

        col3.plotly_chart(
            style_chart(px.bar(
                flagged_priority, x="priority_tier", y="count",
                title="Flagged Cases by Risk Exposure Priority",
                labels={"priority_tier": "Priority Tier", "count": "Number of Cases"},
                color="priority_tier",
                color_discrete_map=PRIORITY_COLORS,
                category_orders={"priority_tier": ["High", "Moderate"]},
            ), showlegend=False),
            width="stretch"
        )

        action_dist = filtered_df["recommended_action"].value_counts().reset_index()
        action_dist.columns = ["recommended_action", "count"]

        if action_dist.empty:
            col4.info("No recommended actions available for the selected filters.")
        else:
            col4.plotly_chart(
                style_chart(px.bar(
                    action_dist, x="recommended_action", y="count",
                    title="Recommended Actions",
                    labels={"recommended_action": "Recommended Action", "count": "Number of Cases"},
                ), x_tickangle=-20),
                width="stretch"
            )

        workflow_dist = (
            filtered_df[filtered_df["risk_level"] == "High"]["workflow_stage"]
            .value_counts()
            .reset_index()
        )
        workflow_dist.columns = ["workflow_stage", "count"]

        if workflow_dist.empty:
            st.info("No workflow stage data available for the selected filters.")
        else:
            st.plotly_chart(
                style_chart(px.bar(
                    workflow_dist, x="workflow_stage", y="count",
                    title="Workflow Stage Split",
                    labels={"workflow_stage": "Workflow Stage", "count": "Number of Cases"},
                ), x_tickangle=-15),
                width="stretch"
            )

    if work_mode == "Historical Intelligence":
        render_historical_insight_footer(filtered_df)
    else:
        render_insight_footer(metrics, top_category, top_state)

    with st.expander("Show raw filtered data"):
        st.dataframe(filtered_df.head(200), width="stretch")

# ── Tab 3: Fraud Risk ─────────────────────────────────────────────────────────────

with tab_fr:
    log_once("tab_fraudiq_seen", "tab_view", {"tab": "Fraud Risk"})
    st.subheader("Fraud Outcomes" if work_mode == "Historical Intelligence" else "Fraud Risk")
    fraud_df = filtered_df[filtered_df["risk_type_display"] == "Fraud Risk"].copy()

    if fraud_df.empty:
        st.warning(
            "No fraud-tagged transactions in the current filter selection.")
    elif work_mode == "Historical Intelligence":
        fraud_df["actual_outcome"] = fraud_df.apply(simplify_actual_outcome, axis=1)
        fraud_loss = float(fraud_df.loc[fraud_df["actual_outcome"].eq("Fraud Approved"), "order_value"].sum())
        fraud_prevented = float(fraud_df.loc[fraud_df["actual_outcome"].eq("Fraud Denied"), "order_value"].sum())
        confirmed_fraud = int(fraud_df["actual_outcome"].eq("Fraud Denied").sum())

        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Fraud Reviewed", f"{len(fraud_df):,}")
        fc2.metric("Fraud Approved Loss", format_currency(fraud_loss))
        fc3.metric("Fraud Denied Prevented", format_currency(fraud_prevented))
        fc4.metric("Fraud Denied Cases", f"{confirmed_fraud:,}")

        fcol1, fcol2 = st.columns(2)
        fraud_loss_by_state = (
            fraud_df[fraud_df["actual_outcome"].eq("Fraud Approved")]
            .groupby("state", as_index=False)["order_value"]
            .sum()
            .sort_values("order_value", ascending=False)
            .head(10)
        )
        fraud_loss_by_payment = (
            fraud_df[fraud_df["actual_outcome"].eq("Fraud Approved")]
            .groupby("payment_method", as_index=False)["order_value"]
            .sum()
            .sort_values("order_value", ascending=False)
            .head(10)
        )

        fcol1.plotly_chart(
            style_chart(px.bar(
                fraud_loss_by_state,
                x="state",
                y="order_value",
                title="Fraud Approved Loss by State",
                labels={"state": "State", "order_value": "Approved Loss"},
                color_discrete_sequence=["#0ea5e9"],
            ), x_tickangle=-25),
            width="stretch",
        )
        fcol2.plotly_chart(
            style_chart(px.bar(
                fraud_loss_by_payment,
                x="payment_method",
                y="order_value",
                title="Fraud Approved Loss by Payment Method",
                labels={"payment_method": "Payment Method", "order_value": "Approved Loss"},
                color_discrete_sequence=["#0ea5e9"],
            ), x_tickangle=-25),
            width="stretch",
        )

        fraud_view = fraud_df[
            [
                "order_id", "customer_id", "order_date", "actual_outcome",
                "fraud_reason_code", "fraud_reason_text", "customer_dispute_comment",
                "order_value", "state", "payment_method", "ops_resolution_note",
            ]
        ].sort_values(["actual_outcome", "order_value"], ascending=[True, False]).head(50)

        st.dataframe(
            fraud_view.style.format({"order_value": "${:,.2f}"}),
            width="stretch",
            hide_index=True,
        )
    else:
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Fraud Cases", f"{len(fraud_df):,}")
        fc2.metric("Fraud Risk Revenue", format_currency(
            fraud_df["order_value"].sum()))
        fc3.metric(
            "Top Fraud State",
            safe_top_index(
                fraud_df.groupby("state")["fraud_score"].mean(
                ).sort_values(ascending=False)
            )
        )
        fc4.metric(
            "Top Fraud Payment Method",
            safe_top_index(
                fraud_df.groupby("payment_method")[
                    "fraud_score"].mean().sort_values(ascending=False)
            )
        )

        fcol1, fcol2 = st.columns(2)

        fraud_by_state = fraud_df.groupby("state", as_index=False)[
            "fraud_score"].mean().sort_values("fraud_score", ascending=False)
        fraud_by_payment = fraud_df.groupby("payment_method", as_index=False)[
            "fraud_score"].mean().sort_values("fraud_score", ascending=False)

        fcol1.plotly_chart(
            style_chart(px.bar(
                fraud_by_state, x="state", y="fraud_score",
                title="Average Fraud Score by State",
                labels={"state": "State", "fraud_score": "Average Fraud Score"},
                color_discrete_sequence=["#2563eb"],
            )),
            width="stretch"
        )
        fcol2.plotly_chart(
            style_chart(px.bar(
                fraud_by_payment, x="payment_method", y="fraud_score",
                title="Average Fraud Score by Payment Method",
                labels={"payment_method": "Payment Method", "fraud_score": "Average Fraud Score"},
                color_discrete_sequence=["#2563eb"],
            ), x_tickangle=-15),
            width="stretch"
        )

        fraud_view = fraud_df[
            [
                "order_id", "customer_id", "order_date", "state", "payment_method",
                "order_value", "fraud_score", "priority_tier", "case_status",
                "workflow_stage", "latest_action_taken", "risk_reason_summary", "recommended_action",
                "ai_agent_next_step"
            ]
        ].sort_values(["fraud_score", "order_value"], ascending=[False, False]).head(50)

        st.dataframe(
            fraud_view.style.format({
                "order_value": "${:,.2f}",
                "fraud_score": "{:.3f}",
            }),
            width="stretch"
        )

# ── Tab 4: Return Risk ───────────────────────────────────────────────────────────

with tab_rt:
    log_once("tab_returniq_seen", "tab_view", {"tab": "Return Risk"})
    st.subheader("Return Outcomes" if work_mode == "Historical Intelligence" else "Return Risk")
    returns_df = filtered_df[filtered_df["risk_type_display"]
                             == "Return Risk"].copy()

    if returns_df.empty:
        st.warning(
            "No return risk transactions in the current filter selection.")
    elif work_mode == "Historical Intelligence":
        returns_df["actual_outcome"] = returns_df.apply(simplify_actual_outcome, axis=1)
        return_loss = float(returns_df.loc[returns_df["actual_outcome"].eq("Return Approved"), "order_value"].sum())
        return_prevented = float(returns_df.loc[returns_df["actual_outcome"].eq("Return Denied"), "order_value"].sum())
        denied_returns = int(returns_df["actual_outcome"].eq("Return Denied").sum())

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Return Reviewed", f"{len(returns_df):,}")
        rc2.metric("Return Approved Loss", format_currency(return_loss))
        rc3.metric("Return Denied Prevented", format_currency(return_prevented))
        rc4.metric("Return Denied Cases", f"{denied_returns:,}")

        rcol1, rcol2 = st.columns(2)
        return_loss_by_category = (
            returns_df[returns_df["actual_outcome"].eq("Return Approved")]
            .groupby("product_category", as_index=False)["order_value"]
            .sum()
            .sort_values("order_value", ascending=False)
            .head(10)
        )
        return_loss_by_product = (
            returns_df[returns_df["actual_outcome"].eq("Return Approved")]
            .groupby("product_name", as_index=False)["order_value"]
            .sum()
            .sort_values("order_value", ascending=False)
            .head(10)
        )

        rcol1.plotly_chart(
            style_chart(px.bar(
                return_loss_by_category,
                x="product_category",
                y="order_value",
                title="Return Approved Loss by Category",
                labels={"product_category": "Product Category", "order_value": "Approved Loss"},
                color_discrete_sequence=["#16a34a"],
            ), x_tickangle=-25),
            width="stretch",
        )
        rcol2.plotly_chart(
            style_chart(px.bar(
                return_loss_by_product,
                x="product_name",
                y="order_value",
                title="Return Approved Loss by Product",
                labels={"product_name": "Product", "order_value": "Approved Loss"},
                color_discrete_sequence=["#16a34a"],
            ), x_tickangle=-25),
            width="stretch",
        )

        returns_view = returns_df[
            [
                "order_id", "customer_id", "order_date", "actual_outcome",
                "return_reason_code", "return_reason_text", "customer_return_comment",
                "product_category", "product_name", "order_value", "ops_resolution_note",
            ]
        ].sort_values(["actual_outcome", "order_value"], ascending=[True, False]).head(50)

        st.dataframe(
            returns_view.style.format({"order_value": "${:,.2f}"}),
            width="stretch",
            hide_index=True,
        )
    else:
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Return Risk Cases", f"{len(returns_df):,}")
        rc2.metric("Return Risk Revenue", format_currency(
            returns_df["order_value"].sum()))
        rc3.metric(
            "Top Return Category",
            safe_top_index(
                returns_df.groupby("product_category")[
                    "return_risk_score"].mean().sort_values(ascending=False)
            )
        )
        rc4.metric(
            "Top Return Product",
            safe_top_index(
                returns_df.groupby("product_name")[
                    "return_risk_score"].mean().sort_values(ascending=False)
            )
        )

        rcol1, rcol2 = st.columns(2)

        returns_by_category = returns_df.groupby("product_category", as_index=False)[
            "return_risk_score"].mean().sort_values("return_risk_score", ascending=False)
        returns_by_product = (
            returns_df.groupby("product_name", as_index=False)[
                "return_risk_score"]
            .mean()
            .sort_values("return_risk_score", ascending=False)
            .head(10)
        )

        rcol1.plotly_chart(
            style_chart(px.bar(
                returns_by_category, x="product_category", y="return_risk_score",
                title="Return Risk Score by Category",
                labels={"product_category": "Product Category",
                        "return_risk_score": "Average Return Risk Score"},
                color_discrete_sequence=["#d97706"],
            ), x_tickangle=-15),
            width="stretch"
        )
        rcol2.plotly_chart(
            style_chart(px.bar(
                returns_by_product, x="product_name", y="return_risk_score",
                title="Top Products by Return Risk Score",
                labels={"product_name": "Product Name",
                        "return_risk_score": "Average Return Risk Score"},
                color_discrete_sequence=["#d97706"],
            ), x_tickangle=-30),
            width="stretch"
        )

        returns_view = returns_df[
            [
                "order_id", "customer_id", "order_date", "product_category", "product_name",
                "past_returns", "order_value", "return_risk_score", "priority_tier",
                "case_status", "workflow_stage", "latest_action_taken", "risk_reason_summary",
                "recommended_action", "ai_agent_next_step"
            ]
        ].sort_values(["return_risk_score", "order_value"], ascending=[False, False]).head(50)

        st.dataframe(
            returns_view.style.format({
                "order_value": "${:,.2f}",
                "return_risk_score": "{:.3f}",
            }),
            width="stretch"
        )

# ── Tab 2: Review Queue ───────────────────────────────────────────────────────────

if tab_rq is not None:
  with tab_rq:
    log_once("tab_ops_queue_seen", "tab_view", {"tab": "Review Queue"})
    st.subheader("Review Queue")

    if st.session_state.get("last_ops_action_notice"):
        st.success(st.session_state.pop("last_ops_action_notice"))

    flagged_df = sort_cases_by_priority(get_open_high_risk_cases(filtered_df))
    closed_flagged_cases = filtered_df[
        (filtered_df["risk_level"] == "High") & (
            filtered_df["case_status"] == "Closed")
    ].copy()
    all_flagged_cases = sort_cases_by_priority(
        filtered_df[filtered_df["risk_level"] == "High"].copy()
    )
    protected_revenue = float(closed_flagged_cases["order_value"].sum())
    closure_rate = (
        round(len(closed_flagged_cases) /
              (len(flagged_df) + len(closed_flagged_cases)) * 100, 1)
        if (len(flagged_df) + len(closed_flagged_cases)) > 0 else 0.0
    )

    oq1, oq2, oq3, oq4 = st.columns(4)
    oq1.metric("Open High-Risk Cases", f"{len(flagged_df):,}")
    oq2.metric(
        "Open Revenue",
        format_currency(
            float(flagged_df["order_value"].sum()) if not flagged_df.empty else 0.0)
    )
    oq3.metric("Protected Revenue", format_currency(protected_revenue))
    oq4.metric("Closure Rate", f"{closure_rate:.1f}%")

    if st.session_state.get("ops_action_log"):
        with st.expander("Recent operations actions", expanded=False):
            ops_action_log_df = pd.DataFrame(st.session_state["ops_action_log"])
            st.dataframe(
                ops_action_log_df[
                    [
                        "ts", "order_id", "customer_id", "action", "actor",
                        "case_status", "workflow_stage", "latest_action_taken", "note",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

    with st.expander("Downloads", expanded=False):
        qc1, qc2 = st.columns(2)
        download_open_cases = qc1.download_button(
            label="Download Open Cases CSV",
            data=to_csv_download(flagged_df.drop(
                columns=["priority_sort", "risk_type_display"], errors="ignore")),
            file_name="aignita_open_cases.csv",
            mime="text/csv",
            key="download_open_cases"
        )
        if download_open_cases:
            log_event("download_clicked", {"file": "aignita_open_cases.csv"})

        download_filtered_dataset = qc2.download_button(
            label="Download All Flagged Cases CSV",
            data=to_csv_download(all_flagged_cases.drop(
                columns=["priority_sort", "risk_type_display"], errors="ignore")),
            file_name="aignita_flagged_cases.csv",
            mime="text/csv",
            key="download_filtered_dataset"
        )
        if download_filtered_dataset:
            log_event("download_clicked", {"file": "aignita_flagged_cases.csv"})

    if flagged_df.empty:
        st.warning("No open cases in the current filter selection.")
    else:
        with st.expander("Show open cases", expanded=False):
            open_case_view = flagged_df[
                [
                    "order_id", "customer_id", "order_value", "unified_risk_score",
                    "risk_type_display", "priority_tier", "workflow_stage",
                    "recommended_action", "latest_action_taken", "risk_reason_summary",
                ]
            ].rename(columns={"risk_type_display": "risk_type"}).head(50)

            st.dataframe(
                open_case_view.style.format({
                    "order_value": "${:,.2f}",
                    "unified_risk_score": "{:.3f}",
                }),
                width="stretch"
            )

        st.markdown("### Selected Case")

        case_options = flagged_df["order_id"].tolist()
        selected_order_id = st.selectbox(
            "Select an Order ID to inspect",
            options=case_options
        )
        log_value_change(
            "last_selected_order_id",
            "case_selected",
            {"order_id": selected_order_id}
        )

        selected_case = flagged_df[flagged_df["order_id"]
                                   == selected_order_id].iloc[0]
        st.session_state["selected_case"] = selected_case

        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("Order Value", format_currency(
            float(selected_case["order_value"])))
        dc2.metric("Unified Risk Score",
                   f'{float(selected_case["unified_risk_score"]):.3f}')
        dc3.metric("Priority", str(selected_case["priority_tier"]))
        dc4.metric("Case Status", str(selected_case["case_status"]))

        render_case_detail_box(selected_case)

        with st.expander("Case signals and timeline", expanded=False):
            st.markdown("#### Risk Signals")
            signal_col1, signal_col2, signal_col3, signal_col4, signal_col5 = st.columns(5)
            signal_col1.metric(
                "Fraud Score", f'{float(selected_case["fraud_score"]):.3f}')
            signal_col2.metric("Return Risk Score",
                               f'{float(selected_case["return_risk_score"]):.3f}')
            signal_col3.metric("Unified Risk Score",
                               f'{float(selected_case["unified_risk_score"]):.3f}')
            signal_col4.metric(
                "Past Returns", f'{int(selected_case["past_returns"])}')
            signal_col5.metric("Chargeback History",
                               f'{int(selected_case["chargeback_history"])}')

            st.markdown("#### Behavioral Signals")
            bsig1, bsig2, bsig3, bsig4, bsig5 = st.columns(5)
            bsig1.metric("Device Risk", str(selected_case.get("device_risk", "Unknown")))
            bsig2.metric("Order Velocity (24h)", f'{int(selected_case.get("velocity_flag", 0))} orders')
            bsig3.metric("Customer Lifetime", f'{int(selected_case.get("customer_lifetime_days", 0))} days')
            bsig4.metric("Address Mismatch", "Yes" if int(selected_case.get("address_mismatch", 0)) else "No")
            bsig5.metric("Chargeback Reason", str(selected_case.get("chargeback_reason", "None")))

            st.markdown("#### Lifecycle Intelligence")
            lint1, lint2, lint3, lint4, lint5 = st.columns(5)
            lint1.metric("Return Reason", str(selected_case.get("return_reason_code", "None")))
            lint2.metric("Fraud Label", str(selected_case.get("fraud_label", "Unknown")))
            lint3.metric("Label Source", str(selected_case.get("label_source", "System")))
            lint4.metric("Label Confidence", f'{float(selected_case.get("label_confidence", 0.0)):.2f}')
            lint5.metric("Ops Resolution", str(selected_case.get("ops_resolution", "Pending")))
            st.caption(str(selected_case.get("ops_resolution_note", "")))
            comments = [
                str(selected_case.get("customer_return_comment", "")).strip(),
                str(selected_case.get("customer_dispute_comment", "")).strip(),
                str(selected_case.get("ops_action_note", "")).strip(),
                str(selected_case.get("ops_override_reason", "")).strip(),
            ]
            comments = [comment for comment in comments if comment]
            if comments:
                st.markdown("#### Customer / Ops Comments")
                for comment in comments[:3]:
                    st.write(f"- {comment}")

            timeline_steps = build_case_timeline(selected_case)
            components.html(render_timeline_html(timeline_steps),
                            height=190, scrolling=False)

        with st.expander("Case copilot and templates", expanded=False):
            st.markdown("#### Ask About This Case")
            case_sample_cols = st.columns(3)
            if case_sample_cols[0].button("Why was this case flagged?"):
                st.session_state["case_question"] = "Why was this case flagged?"
            if case_sample_cols[1].button("What should I do next?"):
                st.session_state["case_question"] = "What should I do next on this case?"
            if case_sample_cols[2].button("Is verification needed?"):
                st.session_state["case_question"] = "Is customer verification needed for this case?"

            case_question = st.text_input(
                "Ask about the selected case",
                value=st.session_state.get("case_question", "Why was this case flagged?")
            )

            case_mode = st.radio(
                "Case Response Mode",
                ["AI Analyst", "Quick Case Insights"],
                index=1,
                horizontal=True,
                key="case_mode_radio"
            )

            if not GROQ_API_KEY:
                st.caption(
                    "AI Analyst mode requires a configured GROQ_API_KEY. Quick Case Insights works without it.")

            if case_question:
                if st.session_state.get("last_case_question_logged") != {
                    "order_id": selected_order_id,
                    "question": case_question,
                    "mode": case_mode,
                }:
                    log_event("case_question_submitted", {
                        "order_id": selected_order_id,
                        "question": case_question,
                        "mode": case_mode,
                    })
                    st.session_state["last_case_question_logged"] = {
                        "order_id": selected_order_id,
                        "question": case_question,
                        "mode": case_mode,
                    }

                with st.spinner("Analyzing selected case..."):
                    if case_mode == "AI Analyst":
                        case_answer = ask_case_copilot(
                            case_question, selected_case)
                    else:
                        case_answer = answer_case_question(
                            case_question, selected_case)

                st.markdown("#### Case Answer")
                render_ai_answer(case_answer)

            st.markdown("#### Templates")
            render_verification_templates(
                order_id=str(selected_case["order_id"]),
                customer_id=str(selected_case["customer_id"]),
            )

            ops_col1, ops_col2 = st.columns(2)

            with ops_col1:
                if st.button("Draft Customer Verification Message"):
                    log_event("case_artifact_generated", {
                        "artifact_type": "customer_verification_draft",
                        "order_id": selected_order_id
                    })
                    with st.spinner("Generating verification draft..."):
                        st.session_state["case_output"] = generate_ai_output(
                            "customer_verification_draft",
                            filtered_df,
                            selected_case,
                            work_mode,
                        )

            with ops_col2:
                if st.button("Create Ops Action Plan"):
                    log_event("case_artifact_generated", {
                        "artifact_type": "ops_action_plan",
                        "order_id": selected_order_id
                    })
                    with st.spinner("Generating ops plan..."):
                        st.session_state["case_output"] = generate_ai_output(
                            "ops_action_plan",
                            filtered_df,
                            selected_case,
                            work_mode,
                        )

            if "case_output" in st.session_state and st.session_state["case_output"]:
                st.markdown("#### Generated Case Output")
                render_ai_output(st.session_state["case_output"])

        st.markdown("### Operations Action")
        action_status = str(selected_case.get("ops_action_taken", "")).strip()
        action_actor = str(selected_case.get("ops_action_by", "")).strip()
        action_at = str(selected_case.get("ops_action_at", "")).strip()
        if action_status:
            st.caption(
                f"Last ops action: {action_status}"
                f"{f' by {action_actor}' if action_actor else ''}"
                f"{f' at {action_at}' if action_at else ''}"
            )

        with st.form(key=f"ops_action_form_{selected_order_id}"):
            ops_form_col1, ops_form_col2 = st.columns(2)
            with ops_form_col1:
                ops_action = st.selectbox(
                    "Action taken by operations",
                    options=list(OPS_ACTIONS.keys()),
                    key=f"ops_action_select_{selected_order_id}",
                )
            with ops_form_col2:
                ops_actor = st.text_input(
                    "Action owner",
                    value=str(selected_case.get("action_owner", "Operations Team")),
                    key=f"ops_actor_{selected_order_id}",
                )
            risk_type = str(selected_case.get("risk_type", "None"))
            resolution_col1, resolution_col2, resolution_col3 = st.columns(3)
            final_decision = resolution_col1.selectbox(
                "Final learning label",
                options=FINAL_DECISION_OPTIONS.get(risk_type, FINAL_DECISION_OPTIONS["None"]),
                key=f"ops_final_decision_{selected_order_id}",
            )
            resolution_reason = resolution_col2.selectbox(
                "Resolution reason",
                options=RESOLUTION_REASON_OPTIONS.get(risk_type, RESOLUTION_REASON_OPTIONS["None"]),
                key=f"ops_resolution_reason_{selected_order_id}",
            )
            analyst_confidence = resolution_col3.slider(
                "Analyst confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.85,
                step=0.05,
                key=f"ops_confidence_{selected_order_id}",
            )
            override_reason = st.text_area(
                "Case insight for model feedback",
                value=str(selected_case.get("ops_override_reason", "")),
                placeholder=(
                    "Optional: what did ops learn while reviewing this case? "
                    "Example: trusted repeat customer passed verification, or carrier evidence contradicted the claim."
                ),
                key=f"ops_override_reason_{selected_order_id}",
                height=80,
            )
            ops_note = st.text_area(
                "Action note",
                value=str(selected_case.get("ops_action_note", "")),
                key=f"ops_note_{selected_order_id}",
                height=90,
            )
            submit_ops_action = st.form_submit_button("Save Case Feedback")

        if submit_ops_action:
            updated_df = apply_ops_action(
                st.session_state["working_df"],
                selected_order_id,
                ops_action,
                ops_actor,
                ops_note,
                final_decision,
                resolution_reason,
                override_reason,
                analyst_confidence,
            )
            st.session_state["working_df"] = updated_df
            updated_case = updated_df[
                updated_df["order_id"].astype(str) == str(selected_order_id)
            ].iloc[0]
            add_ops_action_session_event(
                updated_case,
                ops_action,
                ops_actor,
                ops_note,
            )
            log_event("ops_action_recorded", {
                "order_id": selected_order_id,
                "action": ops_action,
                "actor": ops_actor.strip() or "Operations Team",
                "final_decision": final_decision,
            })
            saved_path = persist_action_updates(updated_df, data_source)
            st.session_state["selected_case"] = updated_case
            st.session_state["last_ops_action_notice"] = (
                f"Saved case feedback for order {selected_order_id} and updated {saved_path}."
            )
            st.rerun()

        st.markdown("---")
        with st.expander("Model feedback from reviewed cases", expanded=False):
            render_live_case_feedback_summary(filtered_df)

# ── Model Learning ───────────────────────────────────────────────────────────────

if tab_model is not None:
  with tab_model:
    log_once("tab_model_learning_seen", "tab_view", {"tab": "Model Learning"})
    st.subheader("Model Learning")
    st.markdown(
        "Analytics leaders can turn historical outcome patterns into model guidance, policy proposals, and future training signals."
    )

    learning_df = filtered_df.copy()
    learning_df["actual_outcome"] = learning_df.apply(simplify_actual_outcome, axis=1)
    approved_return_df = learning_df[learning_df["actual_outcome"].eq("Return Approved")]
    approved_fraud_df = learning_df[learning_df["actual_outcome"].eq("Fraud Approved")]
    denied_df = learning_df[learning_df["actual_outcome"].isin(["Return Denied", "Fraud Denied"])]

    total_loss = calculate_historical_total_loss(learning_df)
    avoided_loss = float(denied_df["order_value"].sum())
    if "historical_model_insights" not in st.session_state:
        st.session_state["historical_model_insights"] = load_learning_backlog("historical")
    injected_insights = st.session_state["historical_model_insights"]

    ml1, ml2, ml3, ml4 = st.columns(4)
    ml1.metric("Loss To Learn From", format_currency(total_loss))
    ml2.metric("Avoided Exposure", format_currency(avoided_loss))
    ml3.metric("Outcome Labels", f"{len(learning_df):,}")
    ml4.metric("Injected Insights", f"{len(injected_insights):,}")

    category_opportunities = (
        approved_return_df.groupby("product_category", as_index=False)
        .agg(
            approved_loss=("order_value", "sum"),
            approved_returns=("order_id", "count"),
            avg_return_score=("return_risk_score", "mean"),
        )
        .sort_values("approved_loss", ascending=False)
        .head(8)
    )
    if not category_opportunities.empty:
        category_opportunities["signal"] = "product_category = " + category_opportunities["product_category"].astype(str)
        category_opportunities["suggested_guidance"] = "Be stricter"
        category_opportunities["priority"] = "High"
        category_opportunities["model_signal"] = (
            "product_category:" + category_opportunities["product_category"].astype(str) + ":stricter_return_review"
        )
    payment_opportunities = (
        approved_fraud_df.groupby("payment_method", as_index=False)
        .agg(
            approved_loss=("order_value", "sum"),
            approved_fraud=("order_id", "count"),
            avg_fraud_score=("fraud_score", "mean"),
        )
        .sort_values("approved_loss", ascending=False)
        .head(8)
    )
    if not payment_opportunities.empty:
        payment_opportunities["signal"] = "payment_method = " + payment_opportunities["payment_method"].astype(str)
        payment_opportunities["suggested_guidance"] = "Require manual review"
        payment_opportunities["priority"] = "High"
        payment_opportunities["model_signal"] = (
            "payment_method:" + payment_opportunities["payment_method"].astype(str) + ":manual_fraud_review"
        )

    st.markdown("### Suggested Learning Opportunities")
    return_learning = category_opportunities.rename(columns={
        "product_category": "where",
        "approved_returns": "cases",
        "avg_return_score": "avg_historical_score",
    })
    fraud_learning = payment_opportunities.rename(columns={
        "payment_method": "where",
        "approved_fraud": "cases",
        "avg_fraud_score": "avg_historical_score",
    })
    return_learning["focus_area"] = "Returns"
    fraud_learning["focus_area"] = "Fraud"
    suggested_learning = pd.concat([return_learning, fraud_learning], ignore_index=True)
    if suggested_learning.empty:
        st.info("No suggested learning opportunities are available for the current filters.")
    else:
        suggested_learning = suggested_learning.sort_values("approved_loss", ascending=False)
        st.dataframe(
            suggested_learning[
                [
                    "focus_area", "signal", "approved_loss", "cases",
                    "avg_historical_score", "suggested_guidance", "priority", "model_signal",
                ]
            ].style.format({
                "approved_loss": "${:,.2f}",
                "avg_historical_score": "{:.3f}",
            }),
            width="stretch",
            hide_index=True,
        )

    st.markdown("### Inject Analytics Guidance")
    with st.form("historical_model_insight_form"):
        form_col1, form_col2, form_col3 = st.columns(3)
        focus_area = form_col1.selectbox(
            "Focus area",
            ["Returns", "Fraud", "Customer Policy", "Review Workflow"],
        )
        condition_field = form_col2.selectbox(
            "Where this applies",
            ["product_category", "product_name", "state", "payment_method", "return_reason_code", "fraud_reason_code"],
        )
        field_options = sorted(
            [str(value) for value in learning_df[condition_field].dropna().unique().tolist()]
        )
        condition_value = form_col3.selectbox("Value", field_options)

        form_col4, form_col5, form_col6 = st.columns(3)
        guidance_type = form_col4.selectbox(
            "Guidance",
            ["Be stricter", "Be more lenient", "Monitor closely", "Require manual review", "Adjust threshold"],
        )
        priority = form_col5.selectbox("Priority", ["High", "Medium", "Low"])
        owner = form_col6.selectbox("Owner", ["Analytics", "Risk Ops", "Returns Ops", "Fraud Ops", "CX"])

        rationale = st.text_area(
            "Analytics rationale",
            value=(
                f"Historical outcomes show elevated loss for {condition_field} = {condition_value}. "
                "Use this as a candidate policy/model signal for future scoring."
            ),
            height=90,
        )
        proposed_signal = st.text_input(
            "Model-ready signal",
            value=f"{condition_field}:{condition_value}:{guidance_type.lower().replace(' ', '_')}",
        )
        submit_insight = st.form_submit_button("Add Insight To Learning Backlog")

    if submit_insight:
        insight = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "focus_area": focus_area,
            "condition_field": condition_field,
            "condition_value": condition_value,
            "guidance": guidance_type,
            "priority": priority,
            "owner": owner,
            "model_signal": proposed_signal,
            "rationale": rationale,
        }
        st.session_state["historical_model_insights"].insert(0, insight)
        save_learning_backlog("historical", st.session_state["historical_model_insights"])
        log_event("historical_model_insight_added", insight)
        st.success("Insight added to the model learning backlog.")

    backlog = pd.DataFrame(st.session_state.get("historical_model_insights", []))
    st.markdown("### Model Learning Backlog")
    if backlog.empty:
        starter_backlog = suggested_learning.head(5).copy() if "suggested_learning" in locals() else pd.DataFrame()
        if starter_backlog.empty:
            st.info("No learning backlog yet. Add an analytics insight above.")
        else:
            st.dataframe(
                starter_backlog[
                    [
                        "focus_area", "signal", "approved_loss", "cases",
                        "suggested_guidance", "priority", "model_signal",
                    ]
                ].style.format({
                    "approved_loss": "${:,.2f}",
                }),
                width="stretch",
                hide_index=True,
            )
    else:
        st.dataframe(backlog, width="stretch", hide_index=True)
        st.download_button(
            "Download Learning Backlog CSV",
            data=to_csv_download(backlog),
            file_name="aignita_model_learning_backlog.csv",
            mime="text/csv",
        )


# ── Tab 6: Risk Copilot ───────────────────────────────────────────────────────────

with tab_cp:
    log_once("tab_ai_copilot_seen", "tab_view", {"tab": "Risk Copilot"})
    st.subheader("Risk Copilot")
    if work_mode == "Historical Intelligence":
        st.markdown("Use Copilot to ask about **actual historical outcomes**, **approved return loss**, **fraud-approved loss**, avoided exposure, and policy lessons from the completed period.")
    else:
        st.markdown("Use Copilot to **ask broader portfolio questions** and **generate merchant-ready outputs** for risk, policy, and operations workflows. For order-specific investigation, use **Review Queue**.")

    st.markdown("### Ask Copilot")
    if work_mode == "Historical Intelligence":
        sample_cols = st.columns(3)
        if sample_cols[0].button("Summarize outcomes"):
            st.session_state["copilot_question"] = "Summarize the historical outcomes."
        if sample_cols[1].button("Return loss drivers"):
            st.session_state["copilot_question"] = "Which categories and products drove approved return loss?"
        if sample_cols[2].button("Fraud loss drivers"):
            st.session_state["copilot_question"] = "Which states and payment methods drove fraud-approved loss?"

        sample_cols2 = st.columns(3)
        if sample_cols2[0].button("Total loss"):
            st.session_state["copilot_question"] = "How much total historical loss occurred?"
        if sample_cols2[1].button("Avoided exposure"):
            st.session_state["copilot_question"] = "How much exposure was avoided through denied returns and denied fraud?"
        if sample_cols2[2].button("Policy lessons"):
            st.session_state["copilot_question"] = "What policy lessons should we take from the historical outcomes?"
    else:
        sample_cols = st.columns(3)
        if sample_cols[0].button("Summarize current risk"):
            st.session_state["copilot_question"] = "Summarize the current risk situation."
        if sample_cols[1].button("What should ops review first?"):
            st.session_state["copilot_question"] = "What should the ops team review first?"
        if sample_cols[2].button("Top fraud locations"):
            st.session_state["copilot_question"] = "Which states have the highest fraud risk?"

        sample_cols2 = st.columns(3)
        if sample_cols2[0].button("Return abuse drivers"):
            st.session_state["copilot_question"] = "Which categories are driving return abuse?"
        if sample_cols2[1].button("Recommended actions"):
            st.session_state["copilot_question"] = "What actions should the ops team prioritize?"
        if sample_cols2[2].button("Revenue at risk"):
            st.session_state["copilot_question"] = "How much revenue is at risk right now?"

    question = st.text_input(
        "Ask Aignita Copilot",
        value=st.session_state.get("copilot_question", "")
    )

    copilot_mode = st.radio(
        "Response Mode",
        ["Quick Insights", "AI Copilot"],
        horizontal=True
    )

    if not GROQ_API_KEY:
        st.caption(
            "Risk Copilot mode requires a configured GROQ_API_KEY. Quick Insights works without it.")

    if question:
        if st.session_state.get("last_copilot_question_logged") != {
            "question": question,
            "mode": copilot_mode,
        }:
            log_event("copilot_question_submitted", {
                "question": question,
                "mode": copilot_mode,
            })
            st.session_state["last_copilot_question_logged"] = {
                "question": question,
                "mode": copilot_mode,
            }

        with st.spinner("Copilot thinking..."):
            if copilot_mode == "Quick Insights":
                answer = (
                    answer_historical_question(question, filtered_df)
                    if work_mode == "Historical Intelligence"
                    else answer_risk_question(question, filtered_df)
                )
            else:
                answer = ask_risk_copilot(question, filtered_df, work_mode)

        st.markdown("### Copilot Answer")
        render_ai_answer(answer)

    with st.expander("Generate briefs and recommendations", expanded=False):
        ag1, ag2, ag3 = st.columns(3)
        with ag1:
            if st.button("Generate Merchant Action Brief"):
                log_event("strategic_artifact_generated", {
                    "artifact_type": "merchant_action_brief"
                })
                with st.spinner("Generating merchant action brief..."):
                    st.session_state["strategic_output"] = generate_ai_output(
                        "merchant_action_brief",
                        filtered_df,
                        st.session_state.get("selected_case", None),
                        work_mode,
                    )

        with ag2:
            if st.button("Generate Merchant Summary"):
                log_event("strategic_artifact_generated", {
                    "artifact_type": "merchant_summary"
                })
                with st.spinner("Generating merchant summary..."):
                    st.session_state["strategic_output"] = generate_ai_output(
                        "merchant_summary",
                        filtered_df,
                        st.session_state.get("selected_case", None),
                        work_mode,
                    )

        with ag3:
            if st.button("Generate Policy Recommendation"):
                log_event("strategic_artifact_generated", {
                    "artifact_type": "policy_recommendation"
                })
                with st.spinner("Generating policy recommendation..."):
                    st.session_state["strategic_output"] = generate_ai_output(
                        "policy_recommendation",
                        filtered_df,
                        st.session_state.get("selected_case", None),
                        work_mode,
                    )

    if "strategic_output" in st.session_state and st.session_state["strategic_output"]:
        st.markdown("### Generated Strategic Output")
        render_ai_output(st.session_state["strategic_output"])
