import os

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from groq import Groq
from dotenv import load_dotenv

st.set_page_config(page_title="Aignita RiskIQ", layout="wide")

st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 96rem;
    }
    .hero-box {
        padding: 1rem 1.2rem;
        border: 1px solid rgba(120,120,120,0.2);
        border-radius: 14px;
        margin-bottom: 1rem;
        background: rgba(250,250,250,0.03);
    }
    .detail-box {
        padding: 1rem 1.2rem;
        border: 1px solid rgba(120,120,120,0.18);
        border-radius: 12px;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        background: rgba(250,250,250,0.02);
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()
GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

REQUIRED_COLUMNS = [
    "order_id", "customer_id", "order_date", "product_category", "product_name",
    "payment_method", "country", "state", "order_value", "past_orders",
    "past_returns", "chargeback_history", "discount_used", "expedited_shipping",
    "fraud_score", "return_abuse_score", "unified_risk_score", "risk_level",
    "risk_type", "recommended_action", "ai_agent_next_step", "action_owner",
    "automation_level", "risk_reason_summary", "priority_tier", "case_status",
    "workflow_stage"
]


@st.cache_data
def load_data():
    path = "data/transactions.csv"
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"])
    return df


def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def format_currency(x: float) -> str:
    if x >= 1_000_000:
        return f"${x / 1_000_000:.1f}M"
    if x >= 1_000:
        return f"${x / 1_000:.1f}K"
    return f"${x:,.0f}"


def safe_top_index(series: pd.Series, default: str = "N/A") -> str:
    return series.index[0] if len(series) > 0 else default


def compute_summary_metrics(data: pd.DataFrame):
    total_orders = len(data)
    flagged_orders = int((data["risk_level"] == "High").sum())
    flagged_pct = round((flagged_orders / total_orders)
                        * 100, 1) if total_orders else 0.0
    fraud_cases = int((data["risk_type"] == "Fraud").sum())
    return_cases = int((data["risk_type"] == "Returns").sum())
    revenue_at_risk = float(
        data.loc[data["risk_level"] == "High", "order_value"].sum())
    critical_cases = int((data["priority_tier"] == "Critical").sum())
    open_cases = int((data["case_status"] == "Open").sum())
    closed_cases = int((data["case_status"] == "Closed").sum())

    return {
        "total_orders": total_orders,
        "flagged_orders": flagged_orders,
        "flagged_pct": flagged_pct,
        "fraud_cases": fraud_cases,
        "return_cases": return_cases,
        "revenue_at_risk": revenue_at_risk,
        "critical_cases": critical_cases,
        "open_cases": open_cases,
        "closed_cases": closed_cases,
    }


def build_case_timeline(case_row: pd.Series):
    current_stage = str(case_row["workflow_stage"])
    recommended_action = str(case_row["recommended_action"])
    case_status = str(case_row["case_status"])

    steps = [("Case Created", "done")]

    stage_map = {
        "Verification Sent": "current",
        "Awaiting Customer Response": "current",
        "Review Queue": "current",
        "Policy Applied": "current",
        "Awaiting Item Inspection": "current",
        "Escalated": "current",
        "Senior Review": "current",
        "Watchlist Monitoring": "current",
        "Action Applied": "done",
        "Auto Approved": "done",
    }

    steps.append((current_stage, stage_map.get(current_stage, "current")))

    if recommended_action == "Allow":
        action_label = "Auto Approval Completed"
    else:
        action_label = f"Action: {recommended_action}"

    steps.append((action_label, "done" if case_status ==
                 "Closed" else "current"))
    final_label = "Resolved" if case_status == "Closed" else "Pending Resolution"
    steps.append((final_label, "done" if case_status ==
                 "Closed" else "pending"))

    return steps


def render_timeline_html(steps):
    status_styles = {
        "done": "background:#dcfce7;color:#166534;border:1px solid #bbf7d0;",
        "current": "background:#dbeafe;color:#1d4ed8;border:1px solid #bfdbfe;",
        "pending": "background:#f3f4f6;color:#4b5563;border:1px solid #e5e7eb;",
    }

    parts = []
    for i, (label, status) in enumerate(steps):
        style = status_styles.get(status, status_styles["pending"])
        parts.append(
            f"""
            <div style="display:flex;align-items:center;">
                <div style="padding:8px 12px;border-radius:999px;{style}font-size:0.9rem;font-weight:600;white-space:nowrap;">
                    {label}
                </div>
            </div>
            """
        )
        if i < len(steps) - 1:
            parts.append(
                """<div style="font-size:1.2rem;color:#9ca3af;padding:0 6px;">→</div>""")

    return f"""
    <html>
    <body style="margin:0;padding:0;font-family:Arial,sans-serif;background:transparent;">
        <div style="padding:12px 14px;border:1px solid rgba(120,120,120,0.15);border-radius:12px;background:rgba(250,250,250,0.02);">
            <div style="font-size:1rem;font-weight:700;margin-bottom:12px;">Case Timeline</div>
            <div style="display:flex;align-items:center;flex-wrap:wrap;gap:8px;">
                {''.join(parts)}
            </div>
        </div>
    </body>
    </html>
    """


def answer_risk_question(question: str, data: pd.DataFrame) -> str:
    q = question.lower().strip()

    if data.empty:
        return "No data is available for the current filters."

    total_orders = len(data)
    flagged_orders = int((data["risk_level"] == "High").sum())
    revenue_at_risk = float(
        data.loc[data["risk_level"] == "High", "order_value"].sum())
    open_cases = int((data["case_status"] == "Open").sum())
    critical_open = data[
        (data["priority_tier"] == "Critical") & (data["case_status"] == "Open")
    ]

    top_category = safe_top_index(
        data.groupby("product_category")["return_abuse_score"]
        .mean()
        .sort_values(ascending=False)
    )

    top_state = safe_top_index(
        data.groupby("state")["fraud_score"]
        .mean()
        .sort_values(ascending=False)
    )

    top_action = safe_top_index(data["recommended_action"].value_counts())

    if "summary" in q or "summarize" in q or "overall" in q:
        return (
            f"In the current filtered view, {flagged_orders:,} out of {total_orders:,} orders are flagged high risk. "
            f"Revenue at risk is {format_currency(revenue_at_risk)}. "
            f"There are {open_cases:,} open cases. "
            f"Return abuse is highest in {top_category}, fraud risk is strongest in {top_state}, "
            f"and the most common recommended action is {top_action}."
        )

    if "return" in q and ("category" in q or "product" in q or "driver" in q):
        top_return_products = (
            data.groupby("product_name")["return_abuse_score"]
            .mean()
            .sort_values(ascending=False)
            .head(3)
            .index.tolist()
        )
        return (
            f"The highest return-abuse category is {top_category}. "
            f"Top products by return-abuse score are: {', '.join(top_return_products)}."
        )

    if "fraud" in q and ("state" in q or "where" in q or "location" in q):
        top_fraud_states = (
            data.groupby("state")["fraud_score"]
            .mean()
            .sort_values(ascending=False)
            .head(3)
            .index.tolist()
        )
        return f"The highest fraud-risk states in the current view are: {', '.join(top_fraud_states)}."

    if "revenue" in q and "risk" in q:
        return f"Revenue at risk in the current filtered view is {format_currency(revenue_at_risk)}."

    if "priority" in q or "review first" in q or "what should i do first" in q:
        if critical_open.empty:
            return "There are no open critical cases in the current filtered view. The next priority is High-priority open cases."
        return (
            f"There are {len(critical_open):,} open critical cases that should be reviewed first. "
            f"Focus on the highest-value orders with the highest unified risk scores."
        )

    if "action" in q or "recommended" in q:
        action_counts = data["recommended_action"].value_counts().head(3)
        action_text = ", ".join(
            [f"{k} ({v})" for k, v in action_counts.items()])
        return f"The top recommended actions in the current view are: {action_text}."

    if "open case" in q or "open cases" in q:
        return f"There are currently {open_cases:,} open cases in the filtered view."

    return (
        "I can help summarize current risk, identify top fraud states, top return-abuse categories or products, "
        "revenue at risk, open cases, and what the ops team should prioritize first."
    )


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
    fraud_cases = int((data["risk_type"] == "Fraud").sum())
    return_cases = int((data["risk_type"] == "Returns").sum())

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

Risk Type: {case_row["risk_type"]}
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


def generate_agent_artifact(
    artifact_type: str,
    dashboard_data: pd.DataFrame,
    selected_case: pd.Series | None = None
) -> str:
    dashboard_context = build_dashboard_context(dashboard_data)
    case_context = build_case_context(selected_case)

    system_prompt = """
You are Aignita RiskIQ Agent Studio.

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
- biggest fraud/returns drivers
- what should be prioritized next
Use plain business language.
""",
        "customer_verification_draft": """
Draft a customer verification message for the selected case.
Tone: polite, professional, non-accusatory.
Ask for only the minimum needed verification.
If no selected case exists, say that a selected case is required.
""",
        "policy_recommendation": """
Write a short policy recommendation memo for the merchant based on the dashboard context.
Focus on return-abuse and fraud controls.
Include:
- 3 recommended policy changes
- expected benefit
- possible tradeoff
""",
        "ops_action_plan": """
Create a short ops action plan for today's risk team.
Include:
- what to review first
- what to defer
- what can be automated
Use bullet points.
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


try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

st.title("Aignita RiskIQ")
st.caption("AI-Powered Risk Intelligence and Action Orchestration for E-commerce")

if df is None:
    st.error("data/transactions.csv not found.")
    st.code("python generate_data.py")
    st.stop()

st.sidebar.header("Filters")

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


# risk_filter = st.sidebar.multiselect(
#     "Risk Level",
#     options=sorted(df["risk_level"].unique()),
#     default=sorted(df["risk_level"].unique())
# )

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = date_min, date_max

filtered_df = df[
    (df["state"].isin(state_filter)) &
    (df["product_category"].isin(category_filter)) &
    # (df["risk_level"].isin(risk_filter)) &
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

st.markdown(f"""
<div class="hero-box">
    <div style="font-size:1.15rem; font-weight:700;">Merchant Risk + Action Summary</div>
    <div style="margin-top:0.45rem;">
        <b>{metrics["flagged_pct"]:.1f}%</b> of filtered orders are flagged high risk, with
        <b>{format_currency(metrics["revenue_at_risk"])}</b> in revenue at risk.
        There are <b>{metrics["critical_cases"]:,}</b> critical-priority cases.
        Return pressure is highest in <b>{top_category}</b>, and fraud pressure is strongest in
        <b>{top_state}</b>.
    </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Orders", f'{metrics["total_orders"]:,}')
c2.metric("Flagged Orders", f'{metrics["flagged_orders"]:,}')
c3.metric("Flagged %", f'{metrics["flagged_pct"]:.1f}%')
c4.metric("Fraud Cases", f'{metrics["fraud_cases"]:,}')
c5.metric("Return Abuse Cases", f'{metrics["return_cases"]:,}')
c6.metric("Revenue at Risk", format_currency(metrics["revenue_at_risk"]))

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "FraudIQ", "ReturnIQ", "Action Center", "Ops Queue", "AI Copilot"]
)

with tab1:
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
            px.line(trend_df, x="order_day", y="flagged_rate",
                    title="Flagged Order Rate Over Time (%)"),
            use_container_width=True
        )

    loss_exposure = (
        filtered_df[filtered_df["risk_type"].isin(["Fraud", "Returns"])]
        .groupby("risk_type", as_index=False)["order_value"]
        .sum()
        .sort_values("order_value", ascending=False)
    )

    if loss_exposure.empty:
        col2.info("No loss exposure data available for the selected filters.")
    else:
        col2.plotly_chart(
            px.bar(loss_exposure, x="risk_type", y="order_value",
                   title="Loss Exposure by Risk Type"),
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
        px.bar(flagged_priority, x="priority_tier", y="count",
               title="Flagged Cases by Risk Exposure Priority"),
        use_container_width=True
    )

    action_dist = filtered_df["recommended_action"].value_counts(
    ).reset_index()
    action_dist.columns = ["recommended_action", "count"]

    if action_dist.empty:
        col4.info("No recommended actions available for the selected filters.")
    else:
        col4.plotly_chart(
            px.bar(action_dist, x="recommended_action",
                   y="count", title="Recommended Actions"),
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
            px.bar(workflow_dist, x="workflow_stage",
                   y="count", title="Workflow Stage Split"),
            use_container_width=True
        )

with tab2:
    st.subheader("FraudIQ")
    fraud_df = filtered_df[filtered_df["risk_type"] == "Fraud"].copy()

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
            px.bar(fraud_by_state, x="state", y="fraud_score",
                   title="Average Fraud Score by State"),
            use_container_width=True
        )
        fcol2.plotly_chart(
            px.bar(fraud_by_payment, x="payment_method", y="fraud_score",
                   title="Average Fraud Score by Payment Method"),
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

with tab3:
    st.subheader("ReturnIQ")
    returns_df = filtered_df[filtered_df["risk_type"] == "Returns"].copy()

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
            px.bar(returns_by_category, x="product_category",
                   y="return_abuse_score", title="Return Abuse Score by Category"),
            use_container_width=True
        )
        rcol2.plotly_chart(
            px.bar(returns_by_product, x="product_name", y="return_abuse_score",
                   title="Top Products by Return Abuse Score"),
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

with tab4:
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
    kc3.metric("Potential Loss Prevented",
               format_currency(potential_loss_prevented))
    kc4.metric("Closure Rate", f"{closure_rate:.1f}%")

    st.markdown("---")

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
            px.bar(
                action_summary,
                x="recommended_action",
                y="orders",
                color="automation_level",
                title="Action Volume by Recommendation"
            ),
            use_container_width=True
        )

        ac2.plotly_chart(
            px.bar(
                action_summary,
                x="action_owner",
                y="orders",
                color="priority_tier",
                title="Action Ownership by Priority"
            ),
            use_container_width=True
        )

        st.markdown("### AI Agent Workflow Recommendations")
        st.dataframe(
            action_summary.style.format({
                "revenue": "${:,.2f}",
                "avg_risk_score": "{:.3f}",
            }),
            use_container_width=True
        )

    st.markdown("### Sample Agent-Ready Action Queue")
    action_queue = filtered_df[
        [
            "order_id", "customer_id", "risk_type", "priority_tier", "case_status",
            "workflow_stage", "recommended_action", "ai_agent_next_step", "action_owner",
            "automation_level", "risk_reason_summary", "order_value", "unified_risk_score"
        ]
    ].sort_values(["unified_risk_score", "order_value"], ascending=[False, False]).head(50)

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
                "order_id", "customer_id", "order_date", "risk_type", "priority_tier",
                "recommended_action", "workflow_stage", "order_value",
                "unified_risk_score", "risk_reason_summary", "action_owner"
            ]
        ].sort_values(["unified_risk_score", "order_value"], ascending=[False, False]).head(50)

        st.dataframe(
            resolved_view.style.format({
                "order_value": "${:,.2f}",
                "unified_risk_score": "{:.3f}",
            }),
            use_container_width=True
        )

with tab5:
    st.subheader("Ops Queue")

    flagged_df = filtered_df[
        (filtered_df["risk_level"] == "High") & (
            filtered_df["case_status"] == "Open")
    ].copy()

    priority_order = {"Critical": 0, "High": 1, "Moderate": 2, "Low": 3}

    if not flagged_df.empty:
        flagged_df["priority_sort"] = flagged_df["priority_tier"].map(
            priority_order)
        flagged_df = flagged_df.sort_values(
            ["priority_sort", "unified_risk_score", "order_value"],
            ascending=[True, False, False]
        )

    oq1, oq2 = st.columns(2)
    oq1.metric("Open High-Risk Cases", f"{len(flagged_df):,}")
    oq2.metric(
        "Open Revenue at Risk",
        format_currency(
            float(flagged_df["order_value"].sum()) if not flagged_df.empty else 0.0)
    )

    qc1, qc2 = st.columns(2)
    qc1.download_button(
        label="Download Open Cases CSV",
        data=to_csv_download(flagged_df.drop(
            columns=["priority_sort"], errors="ignore")),
        file_name="aignita_open_cases.csv",
        mime="text/csv"
    )
    qc2.download_button(
        label="Download Filtered Dataset CSV",
        data=to_csv_download(filtered_df),
        file_name="aignita_filtered_dataset.csv",
        mime="text/csv"
    )

    if flagged_df.empty:
        st.warning("No open cases in the current filter selection.")
    else:
        open_case_view = flagged_df[
            [
                "order_id", "customer_id", "order_date", "product_category", "product_name",
                "country", "state", "payment_method", "order_value",
                "fraud_score", "return_abuse_score", "unified_risk_score",
                "risk_type", "priority_tier", "case_status", "workflow_stage",
                "risk_reason_summary", "recommended_action", "ai_agent_next_step",
                "action_owner", "automation_level"
            ]
        ].head(100)

        st.dataframe(
            open_case_view.style.format({
                "order_value": "${:,.2f}",
                "fraud_score": "{:.3f}",
                "return_abuse_score": "{:.3f}",
                "unified_risk_score": "{:.3f}",
            }),
            use_container_width=True
        )

        st.markdown("### Selected Case")

        case_options = flagged_df["order_id"].tolist()
        selected_order_id = st.selectbox(
            "Select an Order ID to inspect",
            options=case_options
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

        st.markdown(f"""
        <div class="detail-box">
            <div style="font-size:1.05rem; font-weight:700; margin-bottom:0.5rem;">
                Case Summary: {selected_case["order_id"]}
            </div>
            <div><b>Customer ID:</b> {selected_case["customer_id"]}</div>
            <div><b>Order Date:</b> {pd.to_datetime(selected_case["order_date"]).date()}</div>
            <div><b>Risk Type:</b> {selected_case["risk_type"]}</div>
            <div><b>Product:</b> {selected_case["product_name"]} ({selected_case["product_category"]})</div>
            <div><b>Location:</b> {selected_case["state"]}, {selected_case["country"]}</div>
            <div><b>Payment Method:</b> {selected_case["payment_method"]}</div>
            <div><b>Workflow Stage:</b> {selected_case["workflow_stage"]}</div>
            <div><b>Recommended Action:</b> {selected_case["recommended_action"]}</div>
            <div><b>Action Owner:</b> {selected_case["action_owner"]}</div>
            <div><b>Automation Level:</b> {selected_case["automation_level"]}</div>
            <div style="margin-top:0.7rem;"><b>Why this was flagged:</b> {selected_case["risk_reason_summary"]}</div>
            <div style="margin-top:0.5rem;"><b>AI Agent Next Step:</b> {selected_case["ai_agent_next_step"]}</div>
        </div>
        """, unsafe_allow_html=True)

        timeline_steps = build_case_timeline(selected_case)
        components.html(render_timeline_html(timeline_steps),
                        height=190, scrolling=False)

        st.markdown("#### Risk Signals")
        signal_col1, signal_col2, signal_col3, signal_col4 = st.columns(4)
        signal_col1.metric(
            "Fraud Score", f'{float(selected_case["fraud_score"]):.3f}')
        signal_col2.metric("Return Abuse Score",
                           f'{float(selected_case["return_abuse_score"]):.3f}')
        signal_col3.metric(
            "Past Returns", f'{int(selected_case["past_returns"])}')
        signal_col4.metric("Chargeback History",
                           f'{int(selected_case["chargeback_history"])}')

with tab6:
    st.subheader("AI Risk Copilot")
    st.markdown(
        "Ask questions about the **dashboard risk trends or the selected case**.")

    sample_cols = st.columns(3)
    if sample_cols[0].button("Summarize current risk"):
        st.session_state["copilot_question"] = "Summarize the current risk situation."
    if sample_cols[1].button("What should ops review first?"):
        st.session_state["copilot_question"] = "What should the ops team review first?"
    if sample_cols[2].button("Explain the selected case"):
        st.session_state["copilot_question"] = "Explain why this order was flagged."

    sample_cols2 = st.columns(3)
    if sample_cols2[0].button("Top fraud locations"):
        st.session_state["copilot_question"] = "Which states have the highest fraud risk?"
    if sample_cols2[1].button("Return abuse drivers"):
        st.session_state["copilot_question"] = "Which categories are driving return abuse?"
    if sample_cols2[2].button("Recommended actions"):
        st.session_state["copilot_question"] = "What actions should the ops team prioritize?"

    question = st.text_input(
        "Ask Aignita Copilot",
        value=st.session_state.get("copilot_question", "")
    )

    copilot_mode = st.radio(
        "Copilot Mode",
        ["LLM", "Rule-based"],
        horizontal=True
    )

    if question:
        selected_case_for_ai = st.session_state.get("selected_case", None)

        if selected_case_for_ai is None:
            fallback_cases = filtered_df[
                (filtered_df["risk_level"] == "High") & (
                    filtered_df["case_status"] == "Open")
            ].copy()
            if not fallback_cases.empty:
                fallback_cases["priority_sort"] = fallback_cases["priority_tier"].map(
                    {"Critical": 0, "High": 1, "Moderate": 2, "Low": 3}
                )
                fallback_cases = fallback_cases.sort_values(
                    ["priority_sort", "unified_risk_score", "order_value"],
                    ascending=[True, False, False]
                )
                selected_case_for_ai = fallback_cases.iloc[0]

        with st.spinner("Copilot thinking..."):
            if copilot_mode == "LLM":
                answer = ask_risk_copilot(
                    question, filtered_df, selected_case_for_ai)
            else:
                answer = answer_risk_question(question, filtered_df)

        st.markdown("### Copilot Answer")
        st.info(answer)

    st.markdown("### Agent Studio")

    ag1, ag2 = st.columns(2)
    with ag1:
        if st.button("Generate Merchant Summary"):
            with st.spinner("Generating..."):
                st.session_state["agent_output"] = generate_agent_artifact(
                    "merchant_summary",
                    filtered_df,
                    st.session_state.get("selected_case", None)
                )
        if st.button("Draft Customer Verification Message"):
            with st.spinner("Generating..."):
                st.session_state["agent_output"] = generate_agent_artifact(
                    "customer_verification_draft",
                    filtered_df,
                    st.session_state.get("selected_case", None)
                )

    with ag2:
        if st.button("Generate Policy Recommendation"):
            with st.spinner("Generating..."):
                st.session_state["agent_output"] = generate_agent_artifact(
                    "policy_recommendation",
                    filtered_df,
                    st.session_state.get("selected_case", None)
                )
        if st.button("Create Ops Action Plan"):
            with st.spinner("Generating..."):
                st.session_state["agent_output"] = generate_agent_artifact(
                    "ops_action_plan",
                    filtered_df,
                    st.session_state.get("selected_case", None)
                )

    if "agent_output" in st.session_state and st.session_state["agent_output"]:
        st.markdown("### Agent Output")
        st.write(st.session_state["agent_output"])

st.markdown("---")
st.subheader("AI Risk Insights")

insight_text = f"""
- **{metrics["flagged_pct"]:.1f}%** of filtered orders are flagged high risk.
- **{metrics["critical_cases"]:,}** cases are currently marked **Critical** and should be handled first.
- **{metrics["open_cases"]:,}** cases are currently open, while **{metrics["closed_cases"]:,}** have already been resolved.
- **{top_category}** is the strongest return-abuse category.
- **{top_state}** has the highest average fraud score.
"""

st.info(insight_text)

with st.expander("Show raw filtered data"):
    st.dataframe(filtered_df.head(200), use_container_width=True)
