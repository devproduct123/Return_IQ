import pandas as pd
import streamlit as st

from src.metrics import format_currency


def render_hero_summary(metrics: dict, top_category: str, top_state: str):
    st.markdown(f"""
<div class="hero-box">
    <div style="font-size:1.2rem; font-weight:700;">Risk Intelligence Summary</div>
    <div style="margin-top:0.45rem;">
        <b>{metrics["flagged_pct"]:.1f}%</b> of filtered orders are flagged high risk, with
        <b>{format_currency(metrics["revenue_at_risk"])}</b> currently at risk and
        <b>{format_currency(metrics["protected_revenue"])}</b> already protected across closed flagged cases.
        There are <b>{metrics["critical_cases"]:,}</b> critical-priority cases.
        Return-abuse pressure is highest in <b>{top_category}</b>, and fraud pressure is strongest in
        <b>{top_state}</b>.
    </div>
</div>
""", unsafe_allow_html=True)


def render_section_note(html_body: str):
    st.markdown(f"""
<div class="section-note">
    {html_body}
</div>
""", unsafe_allow_html=True)


def render_outcome_summary(
    closed_count: int,
    protected_revenue: float,
    open_count: int,
):
    st.markdown(f"""
    <div class="section-note">
        <b>Outcome Summary:</b> The current filtered view shows
        <b>{closed_count:,}</b> resolved flagged cases protecting
        <b>{format_currency(protected_revenue)}</b> in revenue, while
        <b>{open_count:,}</b> flagged cases remain active for review.
    </div>
    """, unsafe_allow_html=True)


def render_case_detail_box(selected_case: pd.Series):
    st.markdown(f"""
        <div class="detail-box">
            <div style="font-size:1.05rem; font-weight:700; margin-bottom:0.5rem;">
                Case Summary: {selected_case["order_id"]}
            </div>
            <div><b>Customer ID:</b> {selected_case["customer_id"]}</div>
            <div><b>Order Date:</b> {pd.to_datetime(selected_case["order_date"]).date()}</div>
            <div><b>Risk Type:</b> {selected_case["risk_type_display"]}</div>
            <div><b>Product:</b> {selected_case["product_name"]} ({selected_case["product_category"]})</div>
            <div><b>Location:</b> {selected_case["state"]}, {selected_case["country"]}</div>
            <div><b>Payment Method:</b> {selected_case["payment_method"]}</div>
            <div><b>Workflow Stage:</b> {selected_case["workflow_stage"]}</div>
            <div><b>Recommended Action:</b> {selected_case["recommended_action"]}</div>
            <div><b>Action Owner:</b> {selected_case["action_owner"]}</div>
            <div><b>Handling Mode:</b> {selected_case["automation_level"]}</div>
            <div style="margin-top:0.7rem;"><b>Why this was flagged:</b> {selected_case["risk_reason_summary"]}</div>
            <div style="margin-top:0.5rem;"><b>AI Suggested Next Step:</b> {selected_case["ai_agent_next_step"]}</div>
        </div>
        """, unsafe_allow_html=True)


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


def render_insight_footer(metrics: dict, top_category: str, top_state: str):
    st.markdown("---")
    st.subheader("AI Risk Insights")

    insight_text = f"""
- **{metrics["flagged_pct"]:.1f}%** of filtered orders are flagged high risk.
- **{metrics["critical_cases"]:,}** cases are currently marked **Critical** and should be handled first.
- **{metrics["open_cases"]:,}** cases are currently open, while **{metrics["closed_cases"]:,}** have already been resolved.
- **{format_currency(metrics["protected_revenue"])}** in revenue has already been protected across closed flagged cases.
- **{top_category}** is the strongest return-abuse category.
- **{top_state}** has the highest average fraud score.
"""

    st.info(insight_text)

    st.markdown("""
<div class="section-note">
    <b>Pilot with Aignita:</b> Interested in using RiskIQ for your store or risk operations workflow? Reach out to discuss a pilot or custom implementation.
</div>
""", unsafe_allow_html=True)
