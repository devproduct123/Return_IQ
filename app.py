import os
import json
import uuid
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from groq import Groq
from dotenv import load_dotenv

from src.data_loader import load_data as load_transactions_data
from src.metrics import (
    compute_summary_metrics,
    format_currency,
    safe_top_index,
    to_csv_download,
)
from src.risk_logic import (
    answer_case_question,
    answer_risk_question,
    build_case_timeline,
    get_open_high_risk_cases,
    select_fallback_case,
    sort_cases_by_priority,
)
from src.ui_components import (
    render_ai_answer,
    render_ai_output,
    render_case_detail_box,
    render_demo_banner,
    render_hero_summary,
    render_insight_footer,
    render_outcome_summary,
    render_page_header,
    render_timeline_html,
)

st.set_page_config(page_title="Aignita RiskIQ", layout="wide")

st.markdown("""
<style>
/* === Aignita RiskIQ Design System === */

.block-container {
    padding-top: 1.25rem;
    padding-bottom: 2rem;
    max-width: 96rem;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
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
    background: #f1f5f9;
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
    color: #1e293b !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
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
    background: #f8f9fb !important;
    border-color: #94a3b8 !important;
    color: #1e293b !important;
}

/* Download buttons */
[data-testid="stDownloadButton"] > button {
    background: #1e293b !important;
    color: #ffffff !important;
    border: 1px solid #1e293b !important;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 500;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #334155 !important;
    border-color: #334155 !important;
}

/* Page header */
.riq-page-header {
    padding: 0.75rem 0 0.75rem 0;
    margin-bottom: 0.75rem;
    border-bottom: 1px solid #e2e8f0;
}

/* Hero summary card */
.riq-hero {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
}

/* Case detail card */
.riq-detail {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-top: 0.75rem;
    margin-bottom: 1rem;
}

/* Accent note strip */
.riq-note {
    background: #f8f9fb;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #4f46e5;
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
.badge-fraud    { background:#eef2ff; color:#4f46e5; border:1px solid #c7d2fe; }
.badge-return   { background:#fffbeb; color:#d97706; border:1px solid #fde68a; }
.badge-open     { background:#eff6ff; color:#2563eb; border:1px solid #bfdbfe; }
.badge-closed   { background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0; }
.badge-neutral  { background:#f1f5f9; color:#475569; border:1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ── Chart styling ───────────────────────────────────────────────────────────────

CHART_COLORS = ["#4f46e5", "#d97706", "#16a34a", "#dc2626", "#0284c7", "#7c3aed", "#64748b"]
PRIORITY_COLORS = {"Critical": "#dc2626", "High": "#ea580c", "Moderate": "#d97706", "Low": "#16a34a"}
RISK_TYPE_COLORS = {"Fraud": "#4f46e5", "Return Abuse": "#d97706"}


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


# ── Bootstrap ───────────────────────────────────────────────────────────────────

load_dotenv()
GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

@st.cache_data
def load_data():
    return load_transactions_data()


def get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


def get_user_id() -> str:
    viewer_email = st.session_state.get("viewer_email", "").strip().lower()
    return viewer_email if viewer_email else "anonymous"


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
    critical_cases = int((data["priority_tier"] == "Critical").sum())
    fraud_cases = int((data["risk_type_display"] == "Fraud").sum())
    return_cases = int((data["risk_type_display"] == "Return Abuse").sum())
    protected_revenue = float(
        data.loc[
            (data["risk_level"] == "High") & (data["case_status"] == "Closed"),
            "order_value"
        ].sum()
    )

    top_category = (
        data.groupby("product_category")["return_abuse_score"]
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
Aignita RiskIQ Dashboard Summary

Total orders: {total_orders}
Flagged orders: {flagged_orders}
Flagged percent: {flagged_pct:.1f}%
Revenue at risk: ${revenue_at_risk:,.0f}
Protected revenue: ${protected_revenue:,.0f}

Open cases: {open_cases}
Critical cases: {critical_cases}

Fraud cases: {fraud_cases}
Return abuse cases: {return_cases}

Top return abuse categories:
{top_category.to_string()}

Top fraud states:
{top_states.to_string()}

Most common actions:
{top_actions.to_string()}
"""


def build_case_context(case_row: pd.Series | None) -> str:
    if case_row is None:
        return "No selected case context."

    return f"""
Selected Order Case Details

Order ID: {case_row["order_id"]}
Customer ID: {case_row["customer_id"]}

Risk Type: {case_row["risk_type_display"]}
Priority Tier: {case_row["priority_tier"]}
Case Status: {case_row["case_status"]}
Workflow Stage: {case_row["workflow_stage"]}

Order Value: ${case_row["order_value"]}

Fraud Score: {case_row["fraud_score"]}
Return Abuse Score: {case_row["return_abuse_score"]}
Unified Risk Score: {case_row["unified_risk_score"]}

Past Returns: {case_row["past_returns"]}
Chargeback History: {case_row["chargeback_history"]}

Risk Explanation:
{case_row["risk_reason_summary"]}

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


def ask_risk_copilot(question: str, dashboard_data: pd.DataFrame, selected_case: pd.Series | None = None) -> str:
    dashboard_context = build_dashboard_context(dashboard_data)
    case_context = build_case_context(selected_case)

    system_prompt = """
You are Aignita RiskIQ Copilot, an AI risk analyst for e-commerce merchants.

You help risk and operations teams understand fraud risk, return abuse, and what actions to take.

Rules:
- Only use the provided context.
- Be concise, practical, and business-focused.
- Prioritize open critical cases first.
- Do not invent facts not present in the context.
"""

    user_prompt = f"""
Dashboard Context:
{dashboard_context}

Case Context:
{case_context}

Merchant Question:
{question}
"""
    return ask_groq(user_prompt, system_prompt, max_tokens=500)


def ask_case_copilot(question: str, selected_case: pd.Series | None) -> str:
    case_context = build_case_context(selected_case)

    system_prompt = """
You are Aignita RiskIQ Case Analyst Assistant.

You help fraud and risk analysts investigate a selected case.

Rules:
- Only use the supplied case context.
- Be concise and operational.
- Focus on what the analyst should understand and do next.
- Do not invent evidence not present in the context.
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
    selected_case: pd.Series | None = None
) -> str:
    dashboard_context = build_dashboard_context(dashboard_data)
    case_context = build_case_context(selected_case)

    system_prompt = """
You are Aignita RiskIQ Copilot.

Generate practical outputs for risk and operations teams.
Ground everything in the supplied context.
Be concrete, concise, and ready to use.
Do not claim actions were executed.
"""

    artifact_prompts = {
        "merchant_summary": """
Create a short merchant-facing risk summary for the current dashboard view.
Include:
- current risk situation
- biggest fraud/return-abuse drivers
- revenue at risk vs protected revenue
- what should be prioritized next
Use plain business language.
""",
        "policy_recommendation": """
Write a short policy recommendation memo for the merchant based on the dashboard context.
Focus on return-abuse and fraud controls.
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

try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

render_page_header()

log_once("session_started_logged", "session_start", {"app": "Aignita RiskIQ"})

viewer_email = st.sidebar.text_input(
    "Viewer Email (for demo tracking)",
    value=st.session_state.get("viewer_email", "")
).strip().lower()
if viewer_email:
    st.session_state["viewer_email"] = viewer_email

if df is None:
    st.error("data/transactions.csv not found.")
    st.code("python generate_data.py")
    st.stop()

render_demo_banner()

st.sidebar.header("Filters")
st.sidebar.caption("Filters apply across all tabs and AI outputs.")

date_min = df["order_date"].min().date()
date_max = df["order_date"].max().date()

date_range = st.sidebar.date_input(
    "Order Date Range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max
)

category_filter = st.sidebar.multiselect(
    "Product Category",
    options=sorted(df["product_category"].unique()),
    default=sorted(df["product_category"].unique())
)

payment_filter = st.sidebar.multiselect(
    "Payment Method",
    options=sorted(df["payment_method"].unique()),
    default=sorted(df["payment_method"].unique())
)

state_filter = st.sidebar.multiselect(
    "State",
    options=sorted(df["state"].unique()),
    default=sorted(df["state"].unique())
)

priority_filter = st.sidebar.multiselect(
    "Priority Tier",
    options=sorted(df["priority_tier"].unique()),
    default=sorted(df["priority_tier"].unique())
)

action_filter = st.sidebar.multiselect(
    "Recommended Action",
    options=sorted(df["recommended_action"].unique()),
    default=sorted(df["recommended_action"].unique())
)

workflow_filter = st.sidebar.multiselect(
    "Workflow Stage",
    options=sorted(df["workflow_stage"].unique()),
    default=sorted(df["workflow_stage"].unique())
)

case_status_filter = st.sidebar.multiselect(
    "Case Status",
    options=sorted(df["case_status"].unique()),
    default=sorted(df["case_status"].unique())
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

filtered_df = df[
    (df["state"].isin(state_filter)) &
    (df["product_category"].isin(category_filter)) &
    (df["payment_method"].isin(payment_filter)) &
    (df["recommended_action"].isin(action_filter)) &
    (df["priority_tier"].isin(priority_filter)) &
    (df["case_status"].isin(case_status_filter)) &
    (df["workflow_stage"].isin(workflow_filter)) &
    (df["order_date"].dt.date >= start_date) &
    (df["order_date"].dt.date <= end_date)
].copy()

if filtered_df.empty:
    st.warning("No data found for the current filter selection.")
    st.stop()

metrics = compute_summary_metrics(filtered_df)

top_category = safe_top_index(
    filtered_df.groupby("product_category")["return_abuse_score"]
    .mean()
    .sort_values(ascending=False)
)

top_state = safe_top_index(
    filtered_df.groupby("state")["fraud_score"]
    .mean()
    .sort_values(ascending=False)
)

render_hero_summary(metrics, top_category, top_state)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Orders", f'{metrics["total_orders"]:,}')
c2.metric("Flagged Orders", f'{metrics["flagged_orders"]:,}')
c3.metric("Flagged %", f'{metrics["flagged_pct"]:.1f}%')
c4.metric("Fraud Cases", f'{metrics["fraud_cases"]:,}')
c5.metric("Return Abuse Cases", f'{metrics["return_cases"]:,}')
c6.metric("Revenue at Risk", format_currency(metrics["revenue_at_risk"]))

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "FraudIQ", "ReturnIQ",
        "Action Center", "Ops Queue", "AI Risk Copilot"]
)

# ── Tab 1: Overview ──────────────────────────────────────────────────────────────

with tab1:
    log_once("tab_overview_seen", "tab_view", {"tab": "Overview"})
    st.subheader("Executive Overview")

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
        trend_df["flagged_orders"] / trend_df["total_orders"] * 100
    ).round(2)

    if trend_df.empty:
        col1.info("No trend data available for the selected filters.")
    else:
        col1.plotly_chart(
            style_chart(px.line(
                trend_df, x="order_day", y="flagged_rate",
                title="Flagged Order Rate Over Time (%)",
                labels={"order_day": "Order Date", "flagged_rate": "Flagged Rate (%)"},
                color_discrete_sequence=["#4f46e5"],
            )),
            use_container_width=True
        )

    loss_exposure = (
        filtered_df[filtered_df["risk_type_display"].isin(
            ["Fraud", "Return Abuse"])]
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
            use_container_width=True
        )

    col3, col4 = st.columns(2)

    flagged_priority = (
        filtered_df[filtered_df["risk_level"] == "High"]["priority_tier"]
        .value_counts()
        .reindex(["Critical", "High", "Moderate"], fill_value=0)
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
            category_orders={"priority_tier": ["Critical", "High", "Moderate"]},
        ), showlegend=False),
        use_container_width=True
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
            use_container_width=True
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
            use_container_width=True
        )

# ── Tab 2: FraudIQ ────────────────────────────────────────────────────────────────

with tab2:
    log_once("tab_fraudiq_seen", "tab_view", {"tab": "FraudIQ"})
    st.subheader("FraudIQ")
    fraud_df = filtered_df[filtered_df["risk_type_display"] == "Fraud"].copy()

    if fraud_df.empty:
        st.warning(
            "No fraud-tagged transactions in the current filter selection.")
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
                color_discrete_sequence=["#4f46e5"],
            )),
            use_container_width=True
        )
        fcol2.plotly_chart(
            style_chart(px.bar(
                fraud_by_payment, x="payment_method", y="fraud_score",
                title="Average Fraud Score by Payment Method",
                labels={"payment_method": "Payment Method", "fraud_score": "Average Fraud Score"},
                color_discrete_sequence=["#4f46e5"],
            ), x_tickangle=-15),
            use_container_width=True
        )

        fraud_view = fraud_df[
            [
                "order_id", "customer_id", "order_date", "state", "payment_method",
                "order_value", "fraud_score", "priority_tier", "case_status",
                "workflow_stage", "risk_reason_summary", "recommended_action",
                "ai_agent_next_step"
            ]
        ].sort_values(["fraud_score", "order_value"], ascending=[False, False]).head(50)

        st.dataframe(
            fraud_view.style.format({
                "order_value": "${:,.2f}",
                "fraud_score": "{:.3f}",
            }),
            use_container_width=True
        )

# ── Tab 3: ReturnIQ ───────────────────────────────────────────────────────────────

with tab3:
    log_once("tab_returniq_seen", "tab_view", {"tab": "ReturnIQ"})
    st.subheader("ReturnIQ")
    returns_df = filtered_df[filtered_df["risk_type_display"]
                             == "Return Abuse"].copy()

    if returns_df.empty:
        st.warning(
            "No return-abuse transactions in the current filter selection.")
    else:
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Return Abuse Cases", f"{len(returns_df):,}")
        rc2.metric("Return Risk Revenue", format_currency(
            returns_df["order_value"].sum()))
        rc3.metric(
            "Top Return Category",
            safe_top_index(
                returns_df.groupby("product_category")[
                    "return_abuse_score"].mean().sort_values(ascending=False)
            )
        )
        rc4.metric(
            "Top Return Product",
            safe_top_index(
                returns_df.groupby("product_name")[
                    "return_abuse_score"].mean().sort_values(ascending=False)
            )
        )

        rcol1, rcol2 = st.columns(2)

        returns_by_category = returns_df.groupby("product_category", as_index=False)[
            "return_abuse_score"].mean().sort_values("return_abuse_score", ascending=False)
        returns_by_product = (
            returns_df.groupby("product_name", as_index=False)[
                "return_abuse_score"]
            .mean()
            .sort_values("return_abuse_score", ascending=False)
            .head(10)
        )

        rcol1.plotly_chart(
            style_chart(px.bar(
                returns_by_category, x="product_category", y="return_abuse_score",
                title="Return Abuse Score by Category",
                labels={"product_category": "Product Category",
                        "return_abuse_score": "Average Return Abuse Score"},
                color_discrete_sequence=["#d97706"],
            ), x_tickangle=-15),
            use_container_width=True
        )
        rcol2.plotly_chart(
            style_chart(px.bar(
                returns_by_product, x="product_name", y="return_abuse_score",
                title="Top Products by Return Abuse Score",
                labels={"product_name": "Product Name",
                        "return_abuse_score": "Average Return Abuse Score"},
                color_discrete_sequence=["#d97706"],
            ), x_tickangle=-30),
            use_container_width=True
        )

        returns_view = returns_df[
            [
                "order_id", "customer_id", "order_date", "product_category", "product_name",
                "past_returns", "order_value", "return_abuse_score", "priority_tier",
                "case_status", "workflow_stage", "risk_reason_summary",
                "recommended_action", "ai_agent_next_step"
            ]
        ].sort_values(["return_abuse_score", "order_value"], ascending=[False, False]).head(50)

        st.dataframe(
            returns_view.style.format({
                "order_value": "${:,.2f}",
                "return_abuse_score": "{:.3f}",
            }),
            use_container_width=True
        )

# ── Tab 4: Action Center ──────────────────────────────────────────────────────────

with tab4:
    log_once("tab_action_center_seen", "tab_view", {"tab": "Action Center"})
    st.subheader("Action Center")

    open_flagged_cases = filtered_df[
        (filtered_df["risk_level"] == "High") & (
            filtered_df["case_status"] == "Open")
    ].copy()

    closed_flagged_cases = filtered_df[
        (filtered_df["risk_level"] == "High") & (
            filtered_df["case_status"] == "Closed")
    ].copy()

    potential_loss_prevented = float(closed_flagged_cases["order_value"].sum())
    closure_rate = (
        round(len(closed_flagged_cases) /
              (len(open_flagged_cases) + len(closed_flagged_cases)) * 100, 1)
        if (len(open_flagged_cases) + len(closed_flagged_cases)) > 0 else 0.0
    )

    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Open Flagged Cases", f"{len(open_flagged_cases):,}")
    kc2.metric("Closed Flagged Cases", f"{len(closed_flagged_cases):,}")
    kc3.metric("Protected Revenue", format_currency(potential_loss_prevented))
    kc4.metric("Closure Rate", f"{closure_rate:.1f}%")

    render_outcome_summary(
        len(closed_flagged_cases),
        potential_loss_prevented,
        len(open_flagged_cases),
    )

    ac1, ac2 = st.columns(2)

    action_summary = (
        filtered_df.groupby(
            ["recommended_action", "action_owner",
                "automation_level", "priority_tier", "case_status"],
            as_index=False
        )
        .agg(
            orders=("order_id", "count"),
            revenue=("order_value", "sum"),
            avg_risk_score=("unified_risk_score", "mean")
        )
        .sort_values(["orders", "revenue"], ascending=False)
    )

    if action_summary.empty:
        st.warning("No action summary available for the selected filters.")
    else:
        ac1.plotly_chart(
            style_chart(px.bar(
                action_summary,
                x="recommended_action",
                y="orders",
                color="automation_level",
                title="Action Volume by Recommendation",
                labels={"recommended_action": "Recommended Action",
                        "orders": "Number of Cases", "automation_level": "Automation Level"},
            ), showlegend=True, x_tickangle=-20),
            use_container_width=True
        )

        ac2.plotly_chart(
            style_chart(px.bar(
                action_summary,
                x="action_owner",
                y="orders",
                color="priority_tier",
                title="Action Ownership by Priority",
                labels={"action_owner": "Action Owner",
                        "orders": "Number of Cases", "priority_tier": "Priority Tier"},
                color_discrete_map=PRIORITY_COLORS,
            ), showlegend=True, x_tickangle=-15),
            use_container_width=True
        )

        st.markdown("### AI Workflow Recommendations")
        st.dataframe(
            action_summary.style.format({
                "revenue": "${:,.2f}",
                "avg_risk_score": "{:.3f}",
            }),
            use_container_width=True
        )

    st.markdown("### Sample Action Queue")
    action_queue = filtered_df[
        [
            "order_id", "customer_id", "risk_type_display", "priority_tier", "case_status",
            "workflow_stage", "recommended_action", "ai_agent_next_step", "action_owner",
            "automation_level", "risk_reason_summary", "order_value", "unified_risk_score"
        ]
    ].rename(columns={"risk_type_display": "risk_type"}).sort_values(
        ["unified_risk_score", "order_value"], ascending=[False, False]
    ).head(50)

    st.dataframe(
        action_queue.style.format({
            "order_value": "${:,.2f}",
            "unified_risk_score": "{:.3f}",
        }),
        use_container_width=True
    )

    st.markdown("### Resolved Flagged Cases")
    if closed_flagged_cases.empty:
        st.info("No resolved flagged cases in the current filter selection.")
    else:
        resolved_view = closed_flagged_cases[
            [
                "order_id", "customer_id", "order_date", "risk_type_display", "priority_tier",
                "recommended_action", "workflow_stage", "order_value",
                "unified_risk_score", "risk_reason_summary", "action_owner"
            ]
        ].rename(columns={"risk_type_display": "risk_type"}).sort_values(
            ["unified_risk_score", "order_value"], ascending=[False, False]
        ).head(50)

        st.dataframe(
            resolved_view.style.format({
                "order_value": "${:,.2f}",
                "unified_risk_score": "{:.3f}",
            }),
            use_container_width=True
        )

# ── Tab 5: Ops Queue ──────────────────────────────────────────────────────────────

with tab5:
    log_once("tab_ops_queue_seen", "tab_view", {"tab": "Ops Queue"})
    st.subheader("Ops Queue")

    flagged_df = sort_cases_by_priority(get_open_high_risk_cases(filtered_df))

    oq1, oq2 = st.columns(2)
    oq1.metric("Open High-Risk Cases", f"{len(flagged_df):,}")
    oq2.metric(
        "Open Revenue at Risk",
        format_currency(
            float(flagged_df["order_value"].sum()) if not flagged_df.empty else 0.0)
    )

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
        label="Download Filtered Dataset CSV",
        data=to_csv_download(filtered_df.drop(
            columns=["risk_type_display"], errors="ignore")),
        file_name="aignita_filtered_dataset.csv",
        mime="text/csv",
        key="download_filtered_dataset"
    )
    if download_filtered_dataset:
        log_event("download_clicked", {"file": "aignita_filtered_dataset.csv"})

    if flagged_df.empty:
        st.warning("No open cases in the current filter selection.")
    else:
        open_case_view = flagged_df[
            [
                "order_id", "customer_id", "order_date", "product_category", "product_name",
                "country", "state", "payment_method", "order_value",
                "fraud_score", "return_abuse_score", "unified_risk_score",
                "risk_type_display", "priority_tier", "case_status", "workflow_stage",
                "risk_reason_summary", "recommended_action", "ai_agent_next_step",
                "action_owner", "automation_level"
            ]
        ].rename(columns={"risk_type_display": "risk_type"}).head(100)

        st.dataframe(
            open_case_view.style.format({
                "order_value": "${:,.2f}",
                "fraud_score": "{:.3f}",
                "return_abuse_score": "{:.3f}",
                "unified_risk_score": "{:.3f}",
            }),
            use_container_width=True
        )

        st.markdown("---")
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

        st.markdown("#### Key Risk Signals")
        signal_col1, signal_col2, signal_col3, signal_col4, signal_col5 = st.columns(5)
        signal_col1.metric(
            "Fraud Score", f'{float(selected_case["fraud_score"]):.3f}')
        signal_col2.metric("Return Abuse Score",
                           f'{float(selected_case["return_abuse_score"]):.3f}')
        signal_col3.metric("Unified Risk Score",
                           f'{float(selected_case["unified_risk_score"]):.3f}')
        signal_col4.metric(
            "Past Returns", f'{int(selected_case["past_returns"])}')
        signal_col5.metric("Chargeback History",
                           f'{int(selected_case["chargeback_history"])}')

        timeline_steps = build_case_timeline(selected_case)
        components.html(render_timeline_html(timeline_steps),
                        height=190, scrolling=False)

        st.markdown("---")
        st.markdown("### Ask About This Case")
        st.caption("Let analysts ask focused questions about the selected case.")

        case_sample_cols = st.columns(3)
        if case_sample_cols[0].button("Why was this case flagged?"):
            st.session_state["case_question"] = "Why was this case flagged?"
        if case_sample_cols[1].button("What should I do next?"):
            st.session_state["case_question"] = "What should I do next on this case?"
        if case_sample_cols[2].button("Is verification needed?"):
            st.session_state["case_question"] = "Is customer verification needed for this case?"

        case_sample_cols2 = st.columns(3)
        if case_sample_cols2[0].button("Summarize this case"):
            st.session_state["case_question"] = "Summarize this case."
        if case_sample_cols2[1].button("Explain fraud signals"):
            st.session_state["case_question"] = "Explain the fraud signals in this case."
        if case_sample_cols2[2].button("Explain return-abuse signals"):
            st.session_state["case_question"] = "Explain the return-abuse signals in this case."

        case_question = st.text_input(
            "Ask about the selected case",
            value=st.session_state.get("case_question", "")
        )

        case_mode = st.radio(
            "Case Response Mode",
            ["AI Analyst", "Quick Case Insights"],
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

        st.markdown("---")
        st.markdown("### Case Action Assistant")
        st.caption("Create analyst-ready outputs based on the selected case.")

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
                        selected_case
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
                        selected_case
                    )

        if "case_output" in st.session_state and st.session_state["case_output"]:
            st.markdown("#### Generated Case Output")
            render_ai_output(st.session_state["case_output"])

# ── Tab 6: AI Risk Copilot ────────────────────────────────────────────────────────

with tab6:
    log_once("tab_ai_copilot_seen", "tab_view", {"tab": "AI Risk Copilot"})
    st.subheader("AI Risk Copilot")
    st.markdown("Use Copilot to **ask broader portfolio questions** and **generate strategic outputs** for merchant and policy workflows.")

    st.markdown("### Ask Copilot")
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
        ["AI Copilot", "Quick Insights"],
        horizontal=True
    )

    if not GROQ_API_KEY:
        st.caption(
            "AI Copilot mode requires a configured GROQ_API_KEY. Quick Insights works without it.")

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

        selected_case_for_ai = st.session_state.get("selected_case", None)
        used_fallback_case = False

        if selected_case_for_ai is None:
            selected_case_for_ai, used_fallback_case = select_fallback_case(
                filtered_df)

        if used_fallback_case:
            st.markdown("""
<div class="riq-demo-banner">
  No case was manually selected in Ops Queue. Copilot is using the highest-priority open case from the current filters.
</div>
""", unsafe_allow_html=True)

        with st.spinner("Copilot thinking..."):
            if copilot_mode == "AI Copilot":
                answer = ask_risk_copilot(
                    question, filtered_df, selected_case_for_ai)
            else:
                answer = answer_risk_question(question, filtered_df)

        st.markdown("### Copilot Answer")
        render_ai_answer(answer)

    st.markdown("---")
    st.markdown("### Generate Strategic Outputs")
    st.caption(
        "Create merchant-facing and policy-oriented outputs from the current dashboard view.")

    ag1, ag2 = st.columns(2)
    with ag1:
        if st.button("Generate Merchant Summary"):
            log_event("strategic_artifact_generated", {
                "artifact_type": "merchant_summary"
            })
            with st.spinner("Generating merchant summary..."):
                st.session_state["strategic_output"] = generate_ai_output(
                    "merchant_summary",
                    filtered_df,
                    st.session_state.get("selected_case", None)
                )

    with ag2:
        if st.button("Generate Policy Recommendation"):
            log_event("strategic_artifact_generated", {
                "artifact_type": "policy_recommendation"
            })
            with st.spinner("Generating policy recommendation..."):
                st.session_state["strategic_output"] = generate_ai_output(
                    "policy_recommendation",
                    filtered_df,
                    st.session_state.get("selected_case", None)
                )

    if "strategic_output" in st.session_state and st.session_state["strategic_output"]:
        st.markdown("### Generated Strategic Output")
        render_ai_output(st.session_state["strategic_output"])

# ── Footer ────────────────────────────────────────────────────────────────────────

render_insight_footer(metrics, top_category, top_state)

with st.expander("Show raw filtered data"):
    st.dataframe(filtered_df.head(200), use_container_width=True)

with st.expander("Show event log (demo tracking)"):
    event_log_df = pd.DataFrame(st.session_state.get("event_log", []))
    if event_log_df.empty:
        st.caption("No tracked events yet in this session.")
    else:
        st.dataframe(event_log_df, use_container_width=True)
