import os
import json
import shutil
import uuid
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
from src.model_learning import (
    INTERVENTIONS,
    METRIC_LABELS,
    build_backlog_candidates,
    compute_learning_kpis,
    find_return_patterns,
    measure_interventions,
    prepare_learning_frame,
    summarize_model_misses,
)
from src.return_profiles import (
    ACTION_TIER_ORDER,
    NETWORK_BANDS,
    SIMPLE_NETWORK_BANDS,
    SIMPLE_STORE_BANDS,
    NO_NETWORK_DATA,
    STORE_BANDS,
    assign_return_profiles,
    detailed_segments_in_simple,
    has_network_data,
    summarize_return_profiles,
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

st.set_page_config(page_title="Aignita Return Risk Intelligence", layout="wide")

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
.badge-return   { background:#fffbeb; color:#d97706; border:1px solid #fde68a; }
.badge-open     { background:#eff6ff; color:#2563eb; border:1px solid #bfdbfe; }
.badge-closed   { background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0; }
.badge-neutral  { background:#f1f5f9; color:#475569; border:1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ── Chart styling ───────────────────────────────────────────────────────────────

CHART_COLORS = ["#d97706", "#16a34a", "#2563eb", "#dc2626", "#0ea5e9", "#7c3aed", "#64748b"]
PRIORITY_COLORS = {"High": "#dc2626", "Moderate": "#d97706", "Low": "#16a34a"}
DEMO_MERCHANT_NAME = "Meridian Commerce"
DEMO_PLATFORM = "Shopify"
DEMO_LAST_SYNC = "January 2026 Snapshot"
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
    "Flag for Inspection": {
        "case_status": "Open",
        "workflow_stage": "Awaiting Item Inspection",
        "latest_action_taken": "Flagged for item inspection",
        "outcome_type": "Pending",
    },
    "Tag as High Returner": {
        "case_status": "Open",
        "workflow_stage": "High Returner Tagged",
        "latest_action_taken": "High returner tagged by operations",
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
    "Add to Watchlist": {
        "case_status": "Open",
        "workflow_stage": "Watchlist Monitoring",
        "latest_action_taken": "Customer added to return watchlist",
        "outcome_type": "Pending",
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
    ],
}
FINAL_DECISION_OPTIONS = {
    "Returns": ["Pending", "Return Approved", "Return Denied"],
    "None": ["Pending", "Order Approved"],
}
RESOLUTION_REASON_OPTIONS = {
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
        "Actual outcomes, return labels, customer comments, and ops resolutions."
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
    approved_mask = data["return_completed"].astype(int).eq(1)
    requested_mask = data["return_requested"].astype(int).eq(1)
    approved_returns = int(approved_mask.sum())
    return_requests = int(requested_mask.sum())
    return {
        "return_requests": return_requests,
        "return_request_rate": return_requests / total * 100 if total else 0.0,
        "approved_returns": approved_returns,
        "approved_return_rate": approved_returns / total * 100 if total else 0.0,
        "approved_return_value": float(data.loc[approved_mask, "order_value"].sum()),
    }


def calculate_historical_total_loss(data: pd.DataFrame) -> float:
    view = data.copy()
    view["actual_outcome"] = view.apply(simplify_actual_outcome, axis=1)
    return float(
        view.loc[view["actual_outcome"].eq("Return Approved"), "order_value"].sum()
    )


def render_historical_insight_footer(data: pd.DataFrame):
    view = data.copy()
    view["actual_outcome"] = view.apply(simplify_actual_outcome, axis=1)
    total_orders = len(view)
    actual_returns = int(view["actual_outcome"].eq("Return Approved").sum())
    denied_returns = int(view["actual_outcome"].eq("Return Denied").sum())
    total_loss = calculate_historical_total_loss(view)
    avoided_loss = float(
        view.loc[view["actual_outcome"].eq("Return Denied"), "order_value"].sum()
    )
    top_return_loss_category = safe_top_index(
        view.loc[view["actual_outcome"].eq("Return Approved")]
        .groupby("product_category")["order_value"]
        .sum()
        .sort_values(ascending=False),
        default="None",
    )
    top_return_loss_state = safe_top_index(
        view.loc[view["actual_outcome"].eq("Return Approved")]
        .groupby("state")["order_value"]
        .sum()
        .sort_values(ascending=False),
        default="None",
    )

    st.markdown("---")
    st.subheader("Historical Return Outcome Insights")
    st.info(f"""
- **{actual_returns:,}** approved returns occurred across **{total_orders:,}** completed historical orders.
- **{denied_returns:,}** return requests were denied or restricted after review.
- **{format_currency(total_loss)}** in historical loss came from approved returns.
- **{format_currency(avoided_loss)}** in exposure was avoided through denied returns.
- **{top_return_loss_category}** drove the highest approved-return loss by category.
- **{top_return_loss_state}** drove the highest approved-return loss by state.
""")


def render_historical_hero_summary(data: pd.DataFrame):
    view = data.copy()
    view["actual_outcome"] = view.apply(simplify_actual_outcome, axis=1)

    total_orders = len(view)
    return_count = int(view["actual_outcome"].isin(["Return Approved", "Return Denied"]).sum())
    denied_impact = float(
        view.loc[view["actual_outcome"].eq("Return Denied"), "order_value"].sum()
    )
    approved_impact = float(
        view.loc[view["actual_outcome"].eq("Return Approved"), "order_value"].sum()
    )
    top_return_reason = safe_top_index(
        view.loc[view["return_reason_code"].astype(str).ne("None"), "return_reason_code"].value_counts(),
        default="None",
    )

    st.markdown(f"""
<div class="riq-hero">
  <div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;color:#94a3b8;margin-bottom:0.4rem;">Historical Return Outcome Summary</div>
  <div style="font-size:0.9rem;color:#334155;line-height:1.75;">
    Across <b>{total_orders:,}</b> completed historical orders, there were
    <b>{return_count:,}</b> return outcomes reviewed.
    Denied returns represented <b>{format_currency(denied_impact)}</b> in avoided exposure,
    while approved returns represented <b>{format_currency(approved_impact)}</b> in actual loss.
    The most common return reason was <b>{top_return_reason}</b>.
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

    if risk_type == "Returns" or outcome_type.startswith("Return"):
        return "Return Approved" if outcome_type == "Return Approved" else "Return Denied"

    return "Other"


RETURN_OUTCOMES = ["Return Approved", "Return Denied"]
REQUESTED_APPROVED_COLORS = {
    "Requested Returns": "#d97706",
    "Approved Returns": "#16a34a",
}


def summarize_requested_vs_approved(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Per group: dollars and counts for all return requests and for approved returns."""
    view = data[
        data["actual_outcome"].isin(RETURN_OUTCOMES)
        & data[group_column].astype(str).ne("None")
        & data[group_column].astype(str).str.strip().ne("")
    ].copy()
    columns = [group_column, "requested_value", "approved_value", "requested_cases", "approved_cases"]
    if view.empty:
        return pd.DataFrame(columns=columns)

    view["approved_value"] = view["order_value"].where(view["actual_outcome"].eq("Return Approved"), 0.0)
    view["is_approved"] = view["actual_outcome"].eq("Return Approved").astype(int)
    summary = (
        view.groupby(group_column, as_index=False)
        .agg(
            requested_value=("order_value", "sum"),
            approved_value=("approved_value", "sum"),
            requested_cases=("order_id", "count"),
            approved_cases=("is_approved", "sum"),
        )
        .sort_values("requested_value", ascending=False)
    )
    return summary[columns]


def melt_requested_vs_approved(summary: pd.DataFrame, group_column: str) -> pd.DataFrame:
    chart_data = summary.melt(
        id_vars=[group_column, "requested_cases", "approved_cases"],
        value_vars=["requested_value", "approved_value"],
        var_name="measure",
        value_name="dollar_value",
    )
    chart_data["measure"] = chart_data["measure"].map({
        "requested_value": "Requested Returns",
        "approved_value": "Approved Returns",
    })
    return chart_data


def render_reason_outcome_chart(container, data: pd.DataFrame, reason_column: str, title: str):
    summary = summarize_requested_vs_approved(data, reason_column).head(8)
    if summary.empty:
        container.info(f"No data available for {title.lower()}.")
        return

    fig = px.bar(
        melt_requested_vs_approved(summary, reason_column),
        x=reason_column,
        y="dollar_value",
        color="measure",
        barmode="group",
        title=title,
        labels={
            reason_column: "Reason",
            "dollar_value": "Dollar Value",
            "measure": "",
            "requested_cases": "Requested Units",
            "approved_cases": "Approved Units",
        },
        hover_data={"requested_cases": True, "approved_cases": True, "measure": False},
        category_orders={"measure": list(REQUESTED_APPROVED_COLORS)},
        color_discrete_map=REQUESTED_APPROVED_COLORS,
    )
    container.plotly_chart(style_chart(fig, showlegend=True, x_tickangle=-25), width="stretch")


def render_driver_drilldown(
    data: pd.DataFrame,
    driver_column: str,
    sub_reason_column: str,
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
    driver_view = data[data[driver_column].astype(str).eq(str(selected_driver))]
    drilldown = summarize_requested_vs_approved(driver_view, sub_reason_column)
    if drilldown.empty:
        st.info("No sub-reason detail is available for this driver.")
        return

    fig = px.bar(
        melt_requested_vs_approved(drilldown, sub_reason_column),
        x=sub_reason_column,
        y="dollar_value",
        color="measure",
        barmode="group",
        title=f"{selected_driver} Level 2 Drivers",
        labels={
            sub_reason_column: "Level 2 Driver",
            "dollar_value": "Dollar Value",
            "measure": "",
            "requested_cases": "Requested Units",
            "approved_cases": "Approved Units",
        },
        hover_data={"requested_cases": True, "approved_cases": True, "measure": False},
        category_orders={"measure": list(REQUESTED_APPROVED_COLORS)},
        color_discrete_map=REQUESTED_APPROVED_COLORS,
    )
    st.plotly_chart(style_chart(fig, showlegend=True, x_tickangle=-20), width="stretch")
    with st.expander("View details"):
        details = drilldown.rename(columns={sub_reason_column: "sub_reason"}).assign(
            approval_rate=lambda d: d["approved_cases"] / d["requested_cases"]
        )
        st.dataframe(
            details.style.format({
                "requested_value": "${:,.2f}",
                "approved_value": "${:,.2f}",
                "approval_rate": "{:.0%}",
            }),
            width="stretch",
            hide_index=True,
        )


# Interventions from the demo case study (defined in src/model_learning.py). Drawn
# only when the date falls inside the filtered data range.
CASE_STUDY_MILESTONES = [(item["date"], item["label"]) for item in INTERVENTIONS]


def add_milestone_markers(fig, dates: pd.Series):
    if dates.empty:
        return fig
    start, end = dates.min(), dates.max() + pd.offsets.MonthEnd(0)
    for index, (date_text, label) in enumerate(CASE_STUDY_MILESTONES):
        marker = pd.Timestamp(date_text)
        if not (start <= marker <= end):
            continue
        fig.add_shape(
            type="line", x0=marker, x1=marker, y0=0, y1=1, yref="paper",
            line=dict(color="#16a34a", width=1.5, dash="dash"),
        )
        fig.add_annotation(
            x=marker, y=1 - 0.09 * index, yref="paper", text=label, showarrow=False,
            xanchor="left", xshift=4, font=dict(size=10, color="#15803d"),
            bgcolor="rgba(255,255,255,0.85)",
        )
    return fig


def render_monthly_return_trend(historical_df: pd.DataFrame):
    view = historical_df.assign(
        month=historical_df["order_date"].dt.to_period("M").dt.to_timestamp()
    )
    monthly = view.groupby("month").agg(
        orders=("order_id", "count"),
        requested=("actual_outcome", lambda s: s.isin(["Return Approved", "Return Denied"]).sum()),
        approved=("actual_outcome", lambda s: s.eq("Return Approved").sum()),
    ).reset_index()
    if len(monthly) < 2:
        return

    monthly["Return request rate"] = (monthly["requested"] / monthly["orders"] * 100).round(1)
    monthly["Approved return rate"] = (monthly["approved"] / monthly["orders"] * 100).round(1)
    rate_long = monthly.melt(
        id_vars="month",
        value_vars=["Return request rate", "Approved return rate"],
        var_name="metric",
        value_name="rate",
    )

    st.markdown("### Monthly Return Trend")
    trend_col1, trend_col2 = st.columns(2)

    rate_fig = px.line(
        rate_long, x="month", y="rate", color="metric", markers=True,
        title="Return Rate by Month (%)",
        labels={"month": "Order Month", "rate": "% of Orders", "metric": ""},
        color_discrete_sequence=["#d97706", "#dc2626"],
    )
    trend_col1.plotly_chart(
        style_chart(add_milestone_markers(rate_fig, monthly["month"]), height=400, showlegend=True),
        width="stretch",
    )

    reason_monthly = (
        view[view["actual_outcome"].isin(["Return Approved", "Return Denied"])]
        .groupby(["month", "return_reason_code"], as_index=False)["order_id"]
        .count()
        .rename(columns={"order_id": "returns", "return_reason_code": "reason"})
    )
    reason_fig = px.bar(
        reason_monthly, x="month", y="returns", color="reason",
        title="Return Requests by Reason",
        labels={"month": "Order Month", "returns": "Return Requests", "reason": ""},
    )
    reason_fig = style_chart(
        add_milestone_markers(reason_fig, monthly["month"]), height=400, showlegend=True
    )
    # Five reasons don't fit in the top legend strip, so place them below the axis.
    reason_fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(b=110),
    )
    trend_col2.plotly_chart(reason_fig, width="stretch")


ACTION_TIER_COLORS = {
    "Reward": "#15803d",
    "Allow": "#16a34a",
    "Fix Product": "#2563eb",
    "Monitor": "#0891b2",
    "Soft Review": "#d97706",
    "Restrict": "#dc2626",
    "Block / Manual": "#7f1d1d",
}
RESTRICTIVE_TIERS = ["Restrict", "Block / Manual"]


STORE_BAND_LABELS = {
    "Low": "<b>Store: Low</b><br>0–20% returned",
    "Frequent": "<b>Store: Frequent</b><br>20–40% returned",
    "Serial": "<b>Store: Serial</b><br>40%+ returned",
    "New / Thin File": "<b>Store: New</b><br>under 3 orders",
}
NETWORK_BAND_LABELS = {
    "Clean": "<b>Network: Clean</b><br>normal returns elsewhere",
    "Elevated": "<b>Network: Elevated</b><br>25%+ or 3+ stores",
    "Flagged": "<b>Network: Flagged</b><br>abuse flags / linked accounts",
}

SIMPLE_STORE_BAND_LABELS = {
    "Low": "<b>Store: Low</b><br>0–20% returned<br>incl. new accounts",
    "High": "<b>Store: High</b><br>20%+ returned",
}
SIMPLE_NETWORK_BAND_LABELS = {
    "Clean": "<b>Network: Clean</b><br>no abuse flags elsewhere",
    "Risky": "<b>Network: Risky</b><br>abuse flags / linked accounts",
}


def build_segment_grid_figure(
    summary: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    network_available: bool,
    simple: bool = False,
):
    store_bands = SIMPLE_STORE_BANDS if simple else STORE_BANDS
    store_labels = SIMPLE_STORE_BAND_LABELS if simple else STORE_BAND_LABELS
    network_labels = SIMPLE_NETWORK_BAND_LABELS if simple else NETWORK_BAND_LABELS
    all_network_bands = SIMPLE_NETWORK_BANDS if simple else NETWORK_BANDS
    network_cols = all_network_bands if network_available else ["Clean"]
    grid = summary.set_index(["store_band", "network_band"])
    z, text, hover = [], [], []
    for store_band in store_bands:
        z_row, t_row, h_row = [], [], []
        for network_band in network_cols:
            row = grid.loc[(store_band, network_band)]
            z_row.append(row[metric_col] if row["orders"] else None)
            t_row.append(
                f"<b>{row['segment_id']}. {row['segment_name']}</b><br>"
                f"<span style='font-size:20px'><b>{row[metric_col]:.1f}%</b></span><br>"
                f"{row['orders']:,} orders · {row['order_share_pct']:.1f}%<br>"
                f"<i>{row['action_tier']}</i>"
            )
            h_row.append(
                f"<b>{row['segment_id']}. {row['segment_name']}</b><br>"
                f"{metric_label}: {row[metric_col]:.1f}%<br>"
                f"Customers: {row['customers']:,} · Orders: {row['orders']:,}<br>"
                f"Action: {row['action_tier']} – {row['return_policy']}"
            )
        z.append(z_row)
        text.append(t_row)
        hover.append(h_row)

    # Cap the colour scale so one extreme segment doesn't wash out the differences between
    # the rest; cells above the cap show the top colour and their true value as text.
    values = [v for r in z for v in r if v is not None]
    median = float(pd.Series(values).median()) if values else 0.0
    zmax = min(max(values, default=1.0), max(40.0, 2.5 * median)) or 1.0
    zmin = 0.7 * min(values, default=0.0)

    x_labels = [network_labels[b] for b in network_cols] if network_available else [f"<b>{NO_NETWORK_DATA}</b>"]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=[store_labels[b] for b in store_bands],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=16 if simple else 14),
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
            colorscale=[[0, "#ecfdf5"], [0.35, "#fef3c7"], [0.7, "#fdba74"], [1, "#ef4444"]],
            zmin=zmin,
            zmax=zmax,
            xgap=6,
            ygap=6,
            colorbar=dict(title=dict(text=f"{metric_label} %", side="right"), thickness=14, len=0.9),
        )
    )
    fig.update_layout(
        height=600 if simple else 760,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif", size=13, color="#1e293b"),
        margin=dict(l=10, r=10, t=70, b=10),
        hoverlabel=dict(bgcolor="#ffffff", font_size=13),
    )
    fig.update_xaxes(side="top", showgrid=False, zeroline=False, tickfont=dict(size=13, color="#334155"))
    fig.update_yaxes(autorange="reversed", showgrid=False, zeroline=False, tickfont=dict(size=13, color="#334155"))
    return fig


def render_segment_card_html(
    row: pd.Series,
    metric_label: str,
    metric_col: str,
    network_available: bool,
    includes: list[str] | None = None,
) -> str:
    color = ACTION_TIER_COLORS.get(row["action_tier"], "#334155")
    network = row["network_band"] if network_available else NO_NETWORK_DATA
    return f"""
<div style="border:1px solid #e2e8f0;border-top:5px solid {color};border-radius:10px;padding:14px 16px;margin-bottom:14px;background:#ffffff;min-height:250px;">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
    <div style="font-weight:700;font-size:1.02rem;color:#0f172a;">{row['segment_id']}. {row['segment_name']}</div>
    <span style="background:{color};color:#fff;font-size:0.72rem;padding:2px 9px;border-radius:10px;white-space:nowrap;">{row['action_tier']}</span>
  </div>
  <div style="color:#64748b;font-size:0.82rem;margin:4px 0 10px 0;">Store: <b>{row['store_band']}</b> · Network: <b>{network}</b></div>
  <div style="display:flex;gap:14px;margin-bottom:10px;">
    <div><div style="font-size:1.15rem;font-weight:700;color:#0f172a;">{row['customers']:,}</div><div style="font-size:0.72rem;color:#64748b;">customers</div></div>
    <div><div style="font-size:1.15rem;font-weight:700;color:#0f172a;">{row['orders']:,}</div><div style="font-size:0.72rem;color:#64748b;">orders ({row['order_share_pct']:.1f}%)</div></div>
    <div><div style="font-size:1.15rem;font-weight:700;color:{color};">{row[metric_col]:.1f}%</div><div style="font-size:0.72rem;color:#64748b;">{metric_label.lower()}</div></div>
  </div>
  <div style="font-size:0.86rem;color:#0f172a;margin-bottom:6px;"><b>Action:</b> {row['recommended_action']}</div>
  <div style="font-size:0.8rem;color:#475569;"><b>Policy:</b> {row['return_policy']}<br><b>Owner:</b> {row['owner']}</div>
  {f'<div style="font-size:0.78rem;color:#64748b;margin-top:8px;"><b>Includes:</b> {", ".join(includes)}</div>' if includes else ''}
</div>
"""


def render_customer_return_profiles(data: pd.DataFrame, work_mode: str):
    is_historical = work_mode == "Historical Intelligence"
    st.markdown(
        "Customers are segmented on two axes: **return history in this store** "
        "(past returns ÷ past orders) and **return history at other stores** "
        "(cross-merchant network signals). Each segment carries a recommended return policy."
    )
    if data.empty:
        st.info("No orders in the current filter selection.")
        return

    network_available = has_network_data(data)
    if not network_available:
        st.warning(
            "External network signals (ext_* columns) are not available for this data source. "
            "Segments use store history only and the Clean-network treatment."
        )

    view = st.radio(
        "Segmentation view",
        ["Simple (4 segments)", "Detailed (12 segments)"],
        horizontal=True,
        key="return_profile_view",
        help="Simple rolls the 12 detailed segments up into a 2×2. Each detailed segment maps to exactly one simple box.",
    )
    simple = view.startswith("Simple")
    level = "simple" if simple else "detailed"
    name_col = "simple_segment_name" if simple else "segment_name"
    tier_col = "simple_action_tier" if simple else "segment_action_tier"

    profiled = assign_return_profiles(data)
    summary = summarize_return_profiles(profiled, level)
    active = summary[summary["orders"] > 0]

    restrictive = profiled[tier_col].isin(RESTRICTIVE_TIERS)
    network_flagged = profiled["network_band"].eq("Flagged")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Active Segments", f"{len(active)} / {len(summary)}")
    k2.metric(
        "Orders in Restrict / Block Segments",
        f"{int(restrictive.sum()):,}",
        f"{restrictive.mean() * 100:.1f}% of orders",
        delta_color="off",
    )
    k3.metric(
        "Network-Flagged Customers",
        f"{profiled.loc[network_flagged, 'customer_id'].nunique():,}" if network_available else "—",
    )
    if is_historical:
        restricted_loss = float(
            profiled.loc[restrictive & pd.to_numeric(profiled["return_completed"], errors="coerce").eq(1), "order_value"].sum()
        )
        k4.metric("Approved Return Value (Restrict / Block)", format_currency(restricted_loss))
    else:
        open_restricted = int((restrictive & profiled["case_status"].astype(str).eq("Open")).sum())
        k4.metric("Open Cases in Restrict / Block Segments", f"{open_restricted:,}")

    # ── Segment grid (full width) ───────────────────────────────────────────────
    metric_col = "return_request_rate_pct" if is_historical else "high_risk_pct"
    metric_label = "Return request rate" if is_historical else "High-risk rate"

    st.markdown("### Segment Grid")
    grid_caption = (
        f"Rows = return history in this store · Columns = return history at other stores · "
        f"Cell colour = {metric_label.lower()}. Hover a cell for full detail."
    )
    if simple:
        grid_caption += (
            " Network 'Risky' = flagged for abuse or linked accounts elsewhere. "
            "Exception: brand-new accounts flagged elsewhere go to **Restrict**, not Watch."
        )
    st.caption(grid_caption)
    st.plotly_chart(
        build_segment_grid_figure(summary, metric_col, metric_label, network_available, simple=simple),
        width="stretch",
        config={"displayModeBar": False},
    )

    # ── Segment size & impact charts ────────────────────────────────────────────
    chart_left, chart_right = st.columns(2)
    bar_metric = "approved_return_value" if is_historical else "orders"
    bar_title = "Approved Return Value by Segment" if is_historical else "Live Orders by Segment"
    bar_df = active.sort_values(bar_metric, ascending=True)
    bar = px.bar(
        bar_df,
        x=bar_metric,
        y="segment_name",
        color="action_tier",
        orientation="h",
        title=bar_title,
        color_discrete_map=ACTION_TIER_COLORS,
        category_orders={"action_tier": ACTION_TIER_ORDER},
        labels={bar_metric: "Approved Return Value ($)" if is_historical else "Orders", "segment_name": "", "action_tier": ""},
    )
    bar = style_chart(bar, height=360 if simple else 480, showlegend=True)
    bar.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
        margin=dict(l=10, r=18, t=48, b=90),
    )
    bar.update_yaxes(categoryorder="array", categoryarray=bar_df["segment_name"].tolist())
    chart_left.plotly_chart(bar, width="stretch")

    rate_df = active.sort_values(metric_col, ascending=True)
    rate = px.bar(
        rate_df,
        x=metric_col,
        y="segment_name",
        color="action_tier",
        orientation="h",
        title=f"{metric_label} by Segment (%)",
        text=rate_df[metric_col].map(lambda v: f"{v:.1f}%"),
        color_discrete_map=ACTION_TIER_COLORS,
        category_orders={"action_tier": ACTION_TIER_ORDER},
        labels={metric_col: f"{metric_label} (%)", "segment_name": "", "action_tier": ""},
    )
    rate = style_chart(rate, height=360 if simple else 480, showlegend=False)
    rate.update_traces(textposition="outside", cliponaxis=False)
    rate.update_layout(margin=dict(l=10, r=40, t=48, b=90))
    rate.update_yaxes(categoryorder="array", categoryarray=rate_df["segment_name"].tolist())
    chart_right.plotly_chart(rate, width="stretch")

    # ── Playbook: recommended action per segment ────────────────────────────────
    st.markdown("### Segment Playbook – Recommended Actions")
    playbook_rows = summary if network_available else summary[summary["network_band"].eq("Clean")]
    cards = [
        render_segment_card_html(
            row, metric_label, metric_col, network_available,
            includes=detailed_segments_in_simple(row["segment_id"]) if simple else None,
        )
        for _, row in playbook_rows.iterrows()
    ]
    per_row = 2 if simple else 3
    for start in range(0, len(cards), per_row):
        cols = st.columns(per_row)
        for col, card in zip(cols, cards[start:start + per_row]):
            col.markdown(card, unsafe_allow_html=True)

    with st.expander("Segment metrics table"):
        table_cols = [
            "segment_id", "segment_name", "store_band", "network_band", "action_tier", "customers", "orders",
            "order_share_pct", metric_col, "approved_return_value", "avg_order_value", "avg_return_risk_score",
            "avg_store_return_rate_pct", "avg_network_return_rate_pct", "discount_pct", "expedited_pct",
            "avg_tenure_days", "top_category", "top_product", "top_return_reason", "return_policy", "owner",
        ]
        table = playbook_rows[table_cols].rename(columns={
            "segment_id": "#", "segment_name": "Segment", "store_band": "Store History",
            "network_band": "Network History", "action_tier": "Action Tier", "customers": "Customers",
            "orders": "Orders", "order_share_pct": "% of Orders", metric_col: f"{metric_label} %",
            "approved_return_value": "Approved Return Value", "avg_order_value": "Avg Order Value",
            "avg_return_risk_score": "Avg Return Risk", "avg_store_return_rate_pct": "Store Return Rate %",
            "avg_network_return_rate_pct": "Network Return Rate %", "discount_pct": "Discount %",
            "expedited_pct": "Expedited %", "avg_tenure_days": "Avg Tenure (days)", "top_category": "Top Category",
            "top_product": "Top Product", "top_return_reason": "Top Return Reason",
            "return_policy": "Return Policy", "owner": "Owner",
        })
        if not network_available:
            table["Network History"] = NO_NETWORK_DATA
        pct = st.column_config.NumberColumn(format="%.1f%%")
        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config={
                "Approved Return Value": st.column_config.NumberColumn(format="$%.0f"),
                "Avg Order Value": st.column_config.NumberColumn(format="$%.0f"),
                "Avg Return Risk": st.column_config.NumberColumn(format="%.3f"),
                "% of Orders": pct, f"{metric_label} %": pct, "Store Return Rate %": pct,
                "Network Return Rate %": pct, "Discount %": pct, "Expedited %": pct,
            },
        )

    # ── Segment profile drill-down ──────────────────────────────────────────────
    st.markdown("### Segment Profile")
    if active.empty:
        return
    options = active.sort_values("segment_id")["segment_name"].tolist()
    default_name = active.sort_values(metric_col, ascending=False)["segment_name"].iloc[0]
    selected_name = st.selectbox(
        "Select a segment", options, index=options.index(default_name), key=f"return_profile_segment_{level}"
    )
    seg = summary[summary["segment_name"].eq(selected_name)].iloc[0]
    seg_rows = profiled[profiled[name_col].eq(selected_name)]
    includes_html = (
        f'<div style="color:#64748b;margin-top:6px;"><b>Includes detailed segments:</b> '
        f'{", ".join(detailed_segments_in_simple(seg["segment_id"]))}</div>'
        if simple else ""
    )
    tier_color = ACTION_TIER_COLORS.get(seg["action_tier"], "#334155")

    st.markdown(
        f"""
<div style="border-left:6px solid {tier_color};background:#f8fafc;padding:16px 20px;border-radius:8px;margin-bottom:14px;">
  <div style="font-size:1.15rem;font-weight:700;color:#0f172a;">{seg['segment_id']}. {seg['segment_name']}
    <span style="background:{tier_color};color:#fff;font-size:0.78rem;padding:2px 10px;border-radius:10px;margin-left:8px;vertical-align:middle;">{seg['action_tier']}</span>
  </div>
  <div style="color:#475569;margin:6px 0 10px 0;">Store history: <b>{seg['store_band']}</b> · Network history: <b>{seg['network_band'] if network_available else NO_NETWORK_DATA}</b> — {seg['profile']}</div>
  <div style="color:#0f172a;"><b>Recommended action:</b> {seg['recommended_action']}</div>
  <div style="color:#475569;margin-top:6px;"><b>Return policy:</b> {seg['return_policy']} · <b>Owner:</b> {seg['owner']}</div>
  {includes_html}
</div>
""",
        unsafe_allow_html=True,
    )

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Customers", f"{seg['customers']:,}")
    p2.metric("Orders", f"{seg['orders']:,}", f"{seg['order_share_pct']:.1f}% of orders", delta_color="off")
    p3.metric("Avg Order Value", format_currency(seg["avg_order_value"]))
    if is_historical:
        p4.metric("Return Request Rate", f"{seg['return_request_rate_pct']:.1f}%")
    else:
        p4.metric("High-Risk Rate", f"{seg['high_risk_pct']:.1f}%")

    q1, q2, q3, q4 = st.columns(4)
    if is_historical:
        q1.metric("Approved Return Value", format_currency(seg["approved_return_value"]))
    else:
        q1.metric("Open Cases", f"{int(seg_rows['case_status'].astype(str).eq('Open').sum()):,}")
    q2.metric("Store Return Rate", f"{seg['avg_store_return_rate_pct']:.1f}%")
    q3.metric(
        "Network Return Rate",
        f"{seg['avg_network_return_rate_pct']:.1f}%" if network_available and seg["avg_network_return_rate_pct"] is not None else "—",
    )
    q4.metric("Avg Tenure", f"{seg['avg_tenure_days']:,} days")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Discount Used", f"{seg['discount_pct']:.0f}%")
    r2.metric("Expedited Shipping", f"{seg['expedited_pct']:.0f}%")
    r3.metric("Top Category", seg["top_category"])
    r4.metric("Top Return Reason", seg["top_return_reason"])

    d1, d2 = st.columns(2)
    products = (
        seg_rows.groupby(["product_category", "product_name"], as_index=False)
        .agg(orders=("order_id", "count"), return_requests=("return_requested", "sum"), revenue=("order_value", "sum"))
        .sort_values("orders", ascending=False)
        .head(8)
    )
    products["return_rate_%"] = (products["return_requests"] / products["orders"] * 100).round(1)
    d1.markdown("**Top products in this segment**")
    d1.dataframe(
        products.rename(columns={
            "product_category": "Category", "product_name": "Product", "orders": "Orders",
            "return_requests": "Return Requests", "revenue": "Revenue", "return_rate_%": "Return Rate %",
        }),
        width="stretch",
        hide_index=True,
        column_config={"Revenue": st.column_config.NumberColumn(format="$%.0f")},
    )

    customers = (
        seg_rows.groupby("customer_id", as_index=False)
        .agg(
            orders=("order_id", "count"),
            store_return_rate=("store_return_rate", "max"),
            network_return_rate=("ext_return_rate", "max") if network_available else ("store_return_rate", "size"),
            revenue=("order_value", "sum"),
            latest_order=("order_date", "max"),
        )
        .sort_values(["store_return_rate", "revenue"], ascending=False)
        .head(10)
    )
    if not network_available:
        customers = customers.drop(columns=["network_return_rate"])
    d2.markdown("**Customers to review in this segment**")
    d2.dataframe(
        customers.rename(columns={
            "customer_id": "Customer", "orders": "Orders", "store_return_rate": "Store Return Rate",
            "network_return_rate": "Network Return Rate", "revenue": "Revenue", "latest_order": "Latest Order",
        }),
        width="stretch",
        hide_index=True,
        column_config={
            "Store Return Rate": st.column_config.NumberColumn(format="%.2f"),
            "Network Return Rate": st.column_config.NumberColumn(format="%.2f"),
            "Revenue": st.column_config.NumberColumn(format="$%.0f"),
            "Latest Order": st.column_config.DateColumn(format="YYYY-MM-DD"),
        },
    )

    export_cols = [
        c for c in [
            "order_id", "customer_id", "order_date", "product_category", "product_name", "order_value",
            "past_orders", "past_returns", "store_return_rate", "ext_return_rate", "ext_merchant_count",
            "ext_abuse_flags", "ext_linked_accounts", "store_band", "network_band", "simple_segment_name", "segment_name",
            "segment_action_tier", "segment_recommended_action", "risk_level", "case_status",
        ] if c in profiled.columns
    ]
    st.download_button(
        f"Download {selected_name} orders (CSV)",
        data=to_csv_download(seg_rows[export_cols]),
        file_name=f"return_profile_{level}_{seg['segment_id']}_{selected_name.lower().replace(' ', '_').replace('&', 'and')}.csv",
        mime="text/csv",
        key="return_profile_download",
    )


def render_approved_return_value_by_product(data: pd.DataFrame):
    """Approved return value by category and product (formerly the Return Outcomes tab)."""
    view = data.copy()
    view["actual_outcome"] = view.apply(simplify_actual_outcome, axis=1)
    approved = view[view["actual_outcome"].eq("Return Approved")]
    st.markdown("### Approved Return Value by Category and Product")
    if approved.empty:
        st.info("No approved returns in the current filter selection.")
        return

    by_category = (
        approved.groupby("product_category", as_index=False)["order_value"].sum()
        .sort_values("order_value", ascending=False)
    )
    by_product = (
        approved.groupby("product_name", as_index=False)["order_value"].sum()
        .sort_values("order_value", ascending=False)
        .head(10)
    )
    col1, col2 = st.columns(2)
    col1.plotly_chart(
        style_chart(px.bar(
            by_category, x="product_category", y="order_value",
            title="Approved Return Value by Category",
            labels={"product_category": "Product Category", "order_value": "Approved Return Value ($)"},
            color_discrete_sequence=["#dc2626"],
        ), x_tickangle=-25),
        width="stretch",
    )
    col2.plotly_chart(
        style_chart(px.bar(
            by_product, x="product_name", y="order_value",
            title="Top Products by Approved Return Value",
            labels={"product_name": "Product", "order_value": "Approved Return Value ($)"},
            color_discrete_sequence=["#dc2626"],
        ), x_tickangle=-25),
        width="stretch",
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

    render_monthly_return_trend(historical_df)

    st.markdown("### Level 1 Driver Charts")
    render_reason_outcome_chart(
        st,
        historical_df,
        "return_reason_code",
        "Requested vs Approved Returns by Reason (Dollars)",
    )

    st.markdown("### Next-Level Driver Charts")
    st.caption("Select a top-level driver to chart the Level 2 sub-reasons behind requested and approved returns.")
    render_driver_drilldown(
        historical_df,
        driver_column="return_reason_code",
        sub_reason_column="return_sub_reason",
        title="Return Driver",
        selector_key="historical_return_driver_drilldown",
    )

    with st.expander("Show raw filtered data"):
        historical_view = historical_df[
            [
                "order_id", "customer_id", "actual_outcome", "outcome_type",
                "return_reason_code", "return_sub_reason",
                "customer_return_comment",
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
                "return_risk_score", "risk_reason_summary",
            ]
        ].head(50)
        st.dataframe(
            feedback_view.style.format({
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
        if resolution_reason and risk_type == "Returns":
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
        data.groupby("state")["return_risk_score"]
        .mean()
        .sort_values(ascending=False)
        .head(3)
    )

    top_actions = data["recommended_action"].value_counts().head(5)

    return f"""
Aignita Return Risk Intelligence Dashboard Summary

Total orders: {total_orders}
Flagged for return risk: {flagged_orders}
Flagged percent: {flagged_pct:.1f}%
Revenue at risk: ${revenue_at_risk:,.0f}
Protected revenue: ${protected_revenue:,.0f}

Open cases: {open_cases}
High-priority cases: {high_priority_cases}

Return risk cases: {return_cases}

Top return risk categories:
{top_category.to_string()}

Top return risk states:
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
    total_loss = calculate_historical_total_loss(view)
    avoided_loss = float(
        view.loc[view["actual_outcome"].eq("Return Denied"), "order_value"].sum()
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
    return_loss_by_state = (
        view.loc[view["actual_outcome"].eq("Return Approved")]
        .groupby("state")["order_value"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    return_reasons = (
        view.loc[view["actual_outcome"].eq("Return Approved"), "return_reason_code"]
        .value_counts()
        .head(5)
    )

    return f"""
Aignita Return Risk Intelligence Historical Outcome Summary

Completed historical orders: {total_orders}
Actual approved returns: {actual_returns}
Denied/restricted returns: {return_denied}

Total historical return loss from approved returns: ${total_loss:,.0f}
Avoided exposure from denied returns: ${avoided_loss:,.0f}

Approved return loss by product category:
{return_loss_by_category.to_string()}

Approved return loss by product:
{return_loss_by_product.to_string()}

Approved return loss by state:
{return_loss_by_state.to_string()}

Top approved return reasons:
{return_reasons.to_string()}
"""


def build_case_context(case_row: pd.Series | None) -> str:
    if case_row is None:
        return "No selected case context."

    lifetime = case_row.get("customer_lifetime_days", 0)
    action_reason = case_row.get("action_reason", "")
    return_reason = case_row.get("return_reason_code", "None")
    return_comment = case_row.get("customer_return_comment", "")
    ops_resolution = case_row.get("ops_resolution", "Pending")
    label_source = case_row.get("label_source", "System")
    past_orders = int(case_row.get("past_orders", 1))
    past_returns = int(case_row["past_returns"])
    return_rate = past_returns / max(past_orders, 1)

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

Return Risk Score: {case_row["return_risk_score"]}

Customer & Return Signals:
- Past Orders: {past_orders}
- Past Returns: {past_returns} ({return_rate:.0%} return rate)
- Discount Used: {"Yes" if int(case_row.get("discount_used", 0)) else "No"}
- Expedited Shipping: {"Yes" if int(case_row.get("expedited_shipping", 0)) else "No"}
- Product Category: {case_row["product_category"]}
- Product Name: {case_row["product_name"]}

Historical / Feedback Signals:
- Return Reason: {return_reason}
- Customer Return Comment: {return_comment}
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

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    # gpt-oss models spend completion tokens on reasoning; keep it light so the answer isn't cut off.
    extra_body = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else None

    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_completion_tokens=max_tokens,
            extra_body=extra_body,
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
You are Aignita Return Risk Intelligence Copilot, an AI return risk analyst for e-commerce merchants.

You help teams understand historical return outcomes, actual loss from approved returns, avoided exposure from denied returns, reason patterns, and where historical return losses occurred.

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
You are Aignita Return Risk Intelligence Copilot, an AI return risk analyst for e-commerce merchants.

You help risk and operations teams understand portfolio-level return risk, repeat returner patterns, and what actions to take across the current filtered queue.

Rules:
- Only use the provided context.
- Be concise, practical, and business-focused.
- Prioritize open high-priority return risk cases first.
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
You are Aignita Return Risk Intelligence Case Analyst Assistant.

You help returns and risk analysts investigate a selected return risk case.

Rules:
- Only use the supplied case context.
- Be concise and operational.
- Focus on what the analyst should understand and do next regarding the return risk.
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
You are Aignita Return Risk Intelligence Copilot.

Generate practical historical return outcome analysis for risk, finance, operations, and analytics teams.
Ground everything in the supplied context.
Use actual return outcomes and dollars, not live-risk queue language or model scores.
Do not claim actions were executed.
Format responses using markdown: use **bold** for key numbers and metrics, and bullet points (- ) for lists.
"""
    else:
        system_prompt = """
You are Aignita Return Risk Intelligence Copilot.

Generate practical outputs for returns and operations teams.
Ground everything in the supplied context.
Be concrete, concise, and ready to use.
Do not claim actions were executed.
Format responses using markdown: use **bold** for key numbers and metrics, and bullet points (- ) for lists.
"""

    if is_historical:
        artifact_prompts = {
            "merchant_summary": """
Create a short merchant-facing historical return outcome summary.
Include:
- actual return outcome situation (approved vs denied)
- total historical return loss and avoided exposure
- biggest product/category, state, or return reason drivers
- what the merchant should learn from the historical period
Use plain business language.
""",
            "policy_recommendation": """
Write a short policy recommendation memo based on historical return outcomes.
Focus on reducing approved return loss.
Include:
- 3 recommended return policy or operational changes
- expected benefit
- possible customer or operational tradeoff
""",
            "merchant_action_brief": """
Create a merchant-facing historical return outcome brief.
Include:
- what actually happened in the historical period
- top return-loss drivers (category, product, state, reason)
- total loss and avoided exposure
- one operational next step and one analytics/model-learning next step
Avoid live queue language and do not discuss open case prioritization.
""",
            "customer_verification_draft": """
Historical mode does not generate customer verification drafts. Explain that this artifact is only available for selected live cases.
""",
            "ops_action_plan": """
Historical mode does not create live case action plans. Create a short historical returns operations improvement plan instead.
Include:
- what return policies should be reviewed
- what return reason codes or data labels should be improved
- what teams should monitor next period
""",
        }
    else:
        artifact_prompts = {
        "merchant_summary": """
Create a short merchant-facing return risk summary for the current dashboard view.
Include:
- current return risk situation
- biggest return risk drivers (categories, products, states)
- revenue at risk vs protected revenue
- what should be prioritized next
Use plain business language.
""",
        "policy_recommendation": """
Write a short policy recommendation memo for the merchant based on the dashboard context.
Focus on return risk controls and repeat returner management.
Include:
- 3 recommended return policy changes
- expected benefit
- possible customer or operational tradeoff
""",
        "customer_verification_draft": """
Draft a customer return verification message for the selected case.
Tone: polite, professional, non-accusatory.
Ask for only the minimum needed verification (e.g. item condition, photo, return reason).
If no selected case exists, say that a selected case is required.
""",
        "ops_action_plan": """
Create a short ops action plan for the selected return risk case.
Include:
- what the analyst should review first (return history, product category, reason)
- what evidence matters most
- whether the case should be escalated, inspected, restricted, or approved
Keep it practical and concise.
""",
        "merchant_action_brief": """
Create a merchant-facing action brief for the current return risk dashboard view.
Include:
- current return risk situation in plain business language
- top 3 actions the merchant should take today
- which order groups, product categories, or states are driving return risk
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

st.sidebar.markdown("### Data Source")
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
    filtered_df.groupby("state")["return_risk_score"]
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

if work_mode == "Historical Intelligence":
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    historical_kpis = build_historical_kpis(filtered_df)
    c1.metric("Total Historical Orders", f'{metrics["total_orders"]:,}')
    c2.metric("Return Requests", f'{historical_kpis["return_requests"]:,}')
    c3.metric("Return Request Rate", f'{historical_kpis["return_request_rate"]:.1f}%')
    c4.metric("Approved Returns", f'{historical_kpis["approved_returns"]:,}')
    c5.metric("Approved Return Rate", f'{historical_kpis["approved_return_rate"]:.1f}%')
    c6.metric("Approved Return Value", format_currency(historical_kpis["approved_return_value"]))
else:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("New / Open Orders", f'{metrics["total_orders"]:,}')
    c2.metric("Flagged for Return Risk", f'{metrics["flagged_orders"]:,}')
    c3.metric("Return Risk Rate", f'{metrics["flagged_pct"]:.1f}%')
    c4.metric("High Priority Cases", f'{metrics["high_cases"]:,}')
    c5.metric("Revenue at Risk", format_currency(metrics["revenue_at_risk"]))

if work_mode == "Live Risk Queue":
    today_actions = build_today_actions(filtered_df, metrics, top_category, top_state)
    render_today_action_panel(today_actions)

st.markdown("---")

if work_mode == "Historical Intelligence":
    tab_ov, tab_profiles, tab_model, tab_cp = st.tabs(
        ["Overview", "Customer Return Profiles", "Model Learning", "Return Risk Copilot"]
    )
    tab_rq = None
else:
    tab_rq, tab_ov, tab_profiles, tab_cp = st.tabs(
        ["Review Queue", "Overview", "Customer Return Profiles", "Return Risk Copilot"]
    )
    tab_model = None

# ── Tab: Overview ──────────────────────────────────────────────────────────────

with tab_ov:
    log_once("tab_overview_seen", "tab_view", {"tab": "Overview"})
    st.subheader("Overview")

    if work_mode == "Historical Intelligence":
        render_historical_intelligence_view(filtered_df)
        render_approved_return_value_by_product(filtered_df)
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

        return_by_category = (
            filtered_df[filtered_df["risk_type_display"] == "Return Risk"]
            .groupby("product_category", as_index=False)["order_value"]
            .sum()
            .sort_values("order_value", ascending=False)
        )

        if return_by_category.empty:
            col2.info("No return risk data available for the selected filters.")
        else:
            col2.plotly_chart(
                style_chart(px.bar(
                    return_by_category, x="product_category", y="order_value",
                    title="Return Risk Revenue by Category",
                    labels={"product_category": "Category", "order_value": "Revenue at Risk ($)"},
                    color_discrete_sequence=["#d97706"],
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

        col5, col6 = st.columns(2)
        if workflow_dist.empty:
            col5.info("No workflow stage data available for the selected filters.")
        else:
            col5.plotly_chart(
                style_chart(px.bar(
                    workflow_dist, x="workflow_stage", y="count",
                    title="Workflow Stage Split",
                    labels={"workflow_stage": "Workflow Stage", "count": "Number of Cases"},
                ), x_tickangle=-15),
                width="stretch"
            )

        returns_by_product = (
            filtered_df[filtered_df["risk_type_display"] == "Return Risk"]
            .groupby("product_name", as_index=False)["return_risk_score"]
            .mean()
            .sort_values("return_risk_score", ascending=False)
            .head(10)
        )
        if returns_by_product.empty:
            col6.info("No return risk data available for the selected filters.")
        else:
            col6.plotly_chart(
                style_chart(px.bar(
                    returns_by_product, x="product_name", y="return_risk_score",
                    title="Top Products by Return Risk Score",
                    labels={"product_name": "Product", "return_risk_score": "Average Return Risk Score"},
                    color_discrete_sequence=["#d97706"],
                ), x_tickangle=-30),
                width="stretch"
            )

    if work_mode == "Historical Intelligence":
        render_historical_insight_footer(filtered_df)
    else:
        render_insight_footer(metrics, top_category, top_state)

    with st.expander("Show raw filtered data"):
        st.dataframe(filtered_df.head(200), width="stretch")

# ── Tab: Review Queue ───────────────────────────────────────────────────────────

# ── Tab: Customer Return Profiles ────────────────────────────────────────────────

with tab_profiles:
    log_once("tab_return_profiles_seen", "tab_view", {"tab": "Customer Return Profiles"})
    st.subheader("Customer Return Profiles")
    render_customer_return_profiles(filtered_df, work_mode)

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
                    "order_id", "customer_id", "order_value", "return_risk_score",
                    "risk_type_display", "priority_tier", "workflow_stage",
                    "recommended_action", "latest_action_taken", "risk_reason_summary",
                ]
            ].rename(columns={"risk_type_display": "risk_type"}).head(50)

            st.dataframe(
                open_case_view.style.format({
                    "order_value": "${:,.2f}",
                    "return_risk_score": "{:.3f}",
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
        dc2.metric("Return Risk Score",
                   f'{float(selected_case["return_risk_score"]):.3f}')
        dc3.metric("Priority", str(selected_case["priority_tier"]))
        dc4.metric("Case Status", str(selected_case["case_status"]))

        render_case_detail_box(selected_case)

        with st.expander("Case signals and timeline", expanded=False):
            st.markdown("#### Return Risk Signals")
            signal_col1, signal_col2, signal_col3, signal_col4 = st.columns(4)
            signal_col1.metric("Return Risk Score",
                               f'{float(selected_case["return_risk_score"]):.3f}')
            signal_col2.metric(
                "Past Returns", f'{int(selected_case["past_returns"])}')
            signal_col3.metric(
                "Past Orders", f'{int(selected_case.get("past_orders", 1))}')
            signal_col4.metric("Customer Lifetime",
                               f'{int(selected_case.get("customer_lifetime_days", 0))} days')

            st.markdown("#### Purchase Signals")
            bsig1, bsig2, bsig3, bsig4, bsig5 = st.columns(5)
            bsig1.metric("Product Category", str(selected_case.get("product_category", "Unknown")))
            bsig2.metric("Product Name", str(selected_case.get("product_name", "Unknown")))
            bsig3.metric("Order Value", format_currency(float(selected_case["order_value"])))
            bsig4.metric("Discount Used", "Yes" if int(selected_case.get("discount_used", 0)) else "No")
            bsig5.metric("Expedited Shipping", "Yes" if int(selected_case.get("expedited_shipping", 0)) else "No")

            st.markdown("#### Return Intelligence")
            lint1, lint2, lint3, lint4, lint5 = st.columns(5)
            lint1.metric("Return Reason", str(selected_case.get("return_reason_code", "None")))
            lint2.metric("Return Sub Reason", str(selected_case.get("return_sub_reason", "None")))
            lint3.metric("Label Source", str(selected_case.get("label_source", "System")))
            lint4.metric("Label Confidence", f'{float(selected_case.get("label_confidence", 0.0)):.2f}')
            lint5.metric("Ops Resolution", str(selected_case.get("ops_resolution", "Pending")))
            st.caption(str(selected_case.get("ops_resolution_note", "")))
            comments = [
                str(selected_case.get("customer_return_comment", "")).strip(),
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
                st.session_state["case_question"] = "Why was this case flagged for return risk?"
            if case_sample_cols[1].button("What should I do next?"):
                st.session_state["case_question"] = "What should I do next on this case?"
            if case_sample_cols[2].button("Should we restrict this return?"):
                st.session_state["case_question"] = "Should we restrict or approve this return?"

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
        "What the outcomes teach us: where the model missed, which return patterns need which fix, "
        "and whether the actions already taken worked."
    )

    learning_df = prepare_learning_frame(filtered_df)
    if "historical_model_insights" not in st.session_state:
        st.session_state["historical_model_insights"] = load_learning_backlog("historical")

    learning_kpis = compute_learning_kpis(learning_df)
    patterns = find_return_patterns(learning_df)
    misses = summarize_model_misses(learning_df)
    interventions = measure_interventions(learning_df)
    resolved_patterns = int(patterns["status"].eq("Resolved").sum()) if not patterns.empty else 0

    ml1, ml2, ml3, ml4 = st.columns(4)
    ml1.metric("Return Requests", f'{learning_kpis["return_requests"]:,}')
    ml2.metric(
        "Missed by Model",
        format_currency(learning_kpis["missed_value"]),
        delta=f'{learning_kpis["missed_returns"]:,} returns not flagged',
        delta_color="off",
    )
    ml3.metric(
        "Patterns Found",
        f"{len(patterns):,}",
        delta=f"{resolved_patterns} resolved",
        delta_color="off",
    )
    ml4.metric("Backlog Items", f'{len(st.session_state["historical_model_insights"]):,}')

    # 1. Intervention impact
    st.markdown("### Did the Actions Work?")
    if not interventions:
        st.info("No interventions fall inside the current filter range.")
    else:
        st.caption("Each action compared over the 90 days before and after it, for the segment it targeted.")
        for item in interventions:
            st.markdown(
                f"**{item['label']}** · {pd.Timestamp(item['date']):%b %d, %Y} · "
                f"`{item['field']} = {item['value']}`"
            )
            ic1, ic2, ic3 = st.columns(3)
            ic1.metric(
                f"{item['metric_label']} (after)",
                f"{item['after_value']:.0%}",
                delta=f"{(item['after_value'] - item['before_value']) * 100:+.0f} pts vs {item['before_value']:.0%} before",
                delta_color="inverse",
            )
            ic2.metric(
                "Approved return value / month (after)",
                format_currency(item["after_loss_per_month"]),
                delta=f"{format_currency(item['after_loss_per_month'] - item['before_loss_per_month'])} vs before",
                delta_color="inverse",
            )
            ic3.metric("Orders compared", f"{item['before_orders']:,} → {item['after_orders']:,}")

    # 2. Emerging patterns
    st.markdown("### Return Patterns and Recommended Fix")
    st.caption(
        "Segments returning at least 1.3x the baseline rate. The pattern type decides the fix: a product issue "
        "goes to Merchandising, abuse goes to stricter policy. Status compares the last 3 months with the 3 before."
    )
    if patterns.empty:
        st.info("No segment is returning meaningfully above baseline in the current filters.")
    else:
        pattern_view = patterns.assign(
            status_change=patterns.apply(
                lambda r: (
                    f"{METRIC_LABELS[r['status_metric']]}: "
                    + ("–" if pd.isna(r["prior_value"]) else f"{r['prior_value']:.0%}")
                    + " → "
                    + ("–" if pd.isna(r["recent_value"]) else f"{r['recent_value']:.0%}")
                ),
                axis=1,
            )
        )
        st.dataframe(
            pattern_view[
                [
                    "pattern_type", "dimension", "value", "status", "status_change", "return_rate", "lift",
                    "dominant_reason", "approval_rate", "repeat_returner_share", "approved_value",
                    "guidance", "owner",
                ]
            ].style.format({
                "return_rate": "{:.0%}",
                "lift": "{:.1f}x",
                "approval_rate": "{:.0%}",
                "repeat_returner_share": "{:.0%}",
                "approved_value": "${:,.0f}",
            }),
            width="stretch",
            hide_index=True,
        )

    # 3. Model misses
    st.markdown("### Where the Model Missed")
    st.caption(
        "Missed = returns the model did not flag. Flagged-but-approved = returns the model flagged that ops "
        "approved anyway, which adds friction for genuine customers."
    )
    if misses.empty:
        st.info("No return requests in the current filters.")
    else:
        st.dataframe(
            misses.style.format({
                "missed_value": "${:,.0f}",
                "miss_rate": "{:.0%}",
                "flagged_approval_rate": "{:.0%}",
            }),
            width="stretch",
            hide_index=True,
        )

    # 4. Backlog
    st.markdown("### Add to Learning Backlog")
    candidates = build_backlog_candidates(patterns, misses)
    existing_signals = {item.get("model_signal") for item in st.session_state["historical_model_insights"]}
    open_candidates = [c for c in candidates if c["model_signal"] not in existing_signals]
    if not open_candidates:
        st.caption("Every suggestion above is already in the backlog.")
    else:
        selected_labels = st.multiselect(
            "Suggested items",
            [c["label"] for c in open_candidates],
            help="Each item carries its rationale, owner, priority, and model-ready signal.",
        )
        if st.button("Add Selected to Backlog", disabled=not selected_labels):
            created_at = datetime.now(timezone.utc).isoformat()
            for candidate in open_candidates:
                if candidate["label"] in selected_labels:
                    insight = {"created_at": created_at, **{k: v for k, v in candidate.items() if k != "label"}}
                    st.session_state["historical_model_insights"].insert(0, insight)
                    log_event("historical_model_insight_added", insight)
            save_learning_backlog("historical", st.session_state["historical_model_insights"])
            st.session_state["learning_backlog_notice"] = (
                f"Added {len(selected_labels)} item(s) to the learning backlog."
            )
            st.rerun()

    with st.expander("Add custom guidance"):
        form_col1, form_col2, form_col3 = st.columns(3)
        focus_area = form_col1.selectbox(
            "Focus area", ["Returns", "Customer Policy", "Review Workflow"], key="custom_focus_area"
        )
        condition_field = form_col2.selectbox(
            "Where this applies",
            ["product_name", "product_category", "state", "payment_method", "return_reason_code", "customer_segment"],
            key="custom_condition_field",
        )
        field_options = sorted(str(value) for value in learning_df[condition_field].dropna().unique())
        condition_value = form_col3.selectbox("Value", field_options, key="custom_condition_value")

        form_col4, form_col5, form_col6 = st.columns(3)
        guidance_type = form_col4.selectbox(
            "Guidance",
            ["Be stricter", "Be more lenient", "Fix product / sizing info", "Monitor closely",
             "Require manual review", "Adjust threshold"],
            key="custom_guidance",
        )
        priority = form_col5.selectbox("Priority", ["High", "Medium", "Low"], key="custom_priority")
        owner = form_col6.selectbox(
            "Owner", ["Analytics", "Risk Ops", "Returns Ops", "Merchandising", "CX"], key="custom_owner"
        )
        rationale = st.text_area(
            "Analytics rationale",
            value=f"Historical outcomes show elevated returns for {condition_field} = {condition_value}.",
            height=80,
        )
        proposed_signal = st.text_input(
            "Model-ready signal",
            value=f"{condition_field}:{condition_value}:{guidance_type.lower().replace(' / ', '_').replace(' ', '_')}",
        )
        if st.button("Add Custom Guidance to Backlog"):
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
                "source": "Analyst",
            }
            st.session_state["historical_model_insights"].insert(0, insight)
            save_learning_backlog("historical", st.session_state["historical_model_insights"])
            log_event("historical_model_insight_added", insight)
            st.session_state["learning_backlog_notice"] = "Custom guidance added to the learning backlog."
            st.rerun()

    st.markdown("### Model Learning Backlog")
    notice = st.session_state.pop("learning_backlog_notice", None)
    if notice:
        st.success(notice)
    backlog = pd.DataFrame(st.session_state.get("historical_model_insights", []))
    if backlog.empty:
        st.info("The backlog is empty. Add suggested items or custom guidance above.")
    else:
        backlog_columns = [
            c for c in ["priority", "guidance", "condition_field", "condition_value", "owner",
                        "model_signal", "rationale", "source", "created_at"]
            if c in backlog.columns
        ]
        st.dataframe(backlog[backlog_columns], width="stretch", hide_index=True)
        st.download_button(
            "Download Learning Backlog CSV",
            data=to_csv_download(backlog),
            file_name="aignita_model_learning_backlog.csv",
            mime="text/csv",
        )


# ── Tab: Return Risk Copilot ───────────────────────────────────────────────────────────

with tab_cp:
    log_once("tab_ai_copilot_seen", "tab_view", {"tab": "Risk Copilot"})
    st.subheader("Return Risk Copilot")
    if work_mode == "Historical Intelligence":
        st.markdown("Use Copilot to ask about **actual historical outcomes**, **approved return loss**, avoided exposure, and policy lessons from the completed period.")
    else:
        st.markdown("Use Copilot to **ask broader portfolio questions** and **generate merchant-ready outputs** for risk, policy, and operations workflows. For order-specific investigation, use **Review Queue**.")

    st.markdown("### Ask Copilot")
    if work_mode == "Historical Intelligence":
        sample_cols = st.columns(3)
        if sample_cols[0].button("Summarize outcomes"):
            st.session_state["copilot_question"] = "Summarize the historical outcomes."
        if sample_cols[1].button("Return loss drivers"):
            st.session_state["copilot_question"] = "Which categories and products drove approved return loss?"
        if sample_cols[2].button("Top return states"):
            st.session_state["copilot_question"] = "Which states drove the highest approved return loss?"

        sample_cols2 = st.columns(3)
        if sample_cols2[0].button("Total loss"):
            st.session_state["copilot_question"] = "How much total historical loss occurred?"
        if sample_cols2[1].button("Avoided exposure"):
            st.session_state["copilot_question"] = "How much exposure was avoided through denied returns?"
        if sample_cols2[2].button("Policy lessons"):
            st.session_state["copilot_question"] = "What policy lessons should we take from the historical outcomes?"
    else:
        sample_cols = st.columns(3)
        if sample_cols[0].button("Summarize current risk"):
            st.session_state["copilot_question"] = "Summarize the current risk situation."
        if sample_cols[1].button("What should ops review first?"):
            st.session_state["copilot_question"] = "What should the ops team review first?"
        if sample_cols[2].button("Top return locations"):
            st.session_state["copilot_question"] = "Which states have the highest return risk?"

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
