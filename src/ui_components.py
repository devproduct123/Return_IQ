import html as _html

import pandas as pd
import streamlit as st

from src.metrics import format_currency


# ── Badge helpers ───────────────────────────────────────────────────────────────

_BADGE_CLASS = {
    "High":         "badge-high",
    "Moderate":     "badge-moderate",
    "Low":          "badge-low",
    "Fraud Risk":   "badge-fraud",
    "Return Risk": "badge-return",
    "Open":         "badge-open",
    "Closed":       "badge-closed",
}


def _badge(text: str) -> str:
    css = _BADGE_CLASS.get(text, "badge-neutral")
    return f'<span class="badge {css}">{_html.escape(text)}</span>'


# ── Page-level components ───────────────────────────────────────────────────────

def render_page_header():
    st.markdown("""
<div class="riq-page-header">
  <div style="display:flex;align-items:baseline;gap:0.5rem;">
    <span style="font-size:1.3rem;font-weight:700;color:#1d4ed8;letter-spacing:-0.02em;">Aignita RiskIQ</span>
    <span class="badge badge-neutral" style="font-size:0.62rem;padding:1px 6px;">DEMO</span>
  </div>
  <div style="font-size:0.8rem;color:#64748b;margin-top:2px;">
    AI-powered fraud, return risk, and risk operations intelligence for e-commerce merchants
  </div>
</div>
""", unsafe_allow_html=True)


def render_demo_banner():
    st.markdown("""
<div class="riq-demo-banner">
  <b>Want to explore RiskIQ for your store?</b> Contact <b>dg@aignita.com</b> to discuss a pilot.
</div>
""", unsafe_allow_html=True)


# ── Summary components ──────────────────────────────────────────────────────────

def render_merchant_context_bar(
    merchant_name: str,
    platform: str,
    date_range: str,
    orders_analyzed: int,
    last_sync: str,
):
    items = [
        ("Merchant", merchant_name),
        ("Platform", platform),
        ("Review Window", date_range),
        ("Orders Analyzed", f"{orders_analyzed:,}"),
        ("Last Sync", last_sync),
    ]

    cells = "".join(
        f"""<div>
  <div style="font-size:0.66rem;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;color:#94a3b8;">{_html.escape(label)}</div>
  <div style="font-size:0.86rem;font-weight:600;color:#334155;margin-top:2px;">{_html.escape(value)}</div>
</div>"""
        for label, value in items
    )

    st.markdown(
        f'<div class="riq-context-bar">{cells}</div>',
        unsafe_allow_html=True,
    )


def render_today_action_panel(actions: list[dict]):
    cards = []
    for action in actions:
        label = _html.escape(str(action["label"]))
        title = _html.escape(str(action["title"]))
        body = _html.escape(str(action["body"]))
        metric = _html.escape(str(action["metric"]))
        accent = _html.escape(str(action.get("accent", "#2563eb")))
        cards.append(f"""<div class="riq-action-card" style="border-top-color:{accent};">
  <div style="display:flex;justify-content:space-between;gap:0.75rem;align-items:flex-start;">
    <div style="font-size:0.66rem;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;color:#94a3b8;">{label}</div>
    <div class="badge badge-neutral">{metric}</div>
  </div>
  <div style="font-size:0.95rem;font-weight:700;color:#1e293b;margin-top:0.45rem;">{title}</div>
  <div style="font-size:0.84rem;color:#475569;line-height:1.55;margin-top:0.25rem;">{body}</div>
</div>""")

    st.markdown(f"""
<div class="riq-action-panel">
  <div style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-end;margin-bottom:0.75rem;">
    <div>
      <div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;
        letter-spacing:0.07em;color:#94a3b8;">Merchant Priority Brief</div>
      <div style="font-size:1rem;font-weight:700;color:#1e293b;">What should I do today?</div>
    </div>
    <div style="font-size:0.78rem;color:#64748b;">Generated from the current filters</div>
  </div>
  <div class="riq-action-grid">
    {''.join(cards)}
  </div>
</div>
""", unsafe_allow_html=True)

def render_hero_summary(metrics: dict, top_category: str, top_state: str):
    st.markdown(f"""
<div class="riq-hero">
  <div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;color:#94a3b8;margin-bottom:0.4rem;">Risk Intelligence Summary</div>
  <div style="font-size:0.9rem;color:#334155;line-height:1.75;">
    <b>{metrics["flagged_pct"]:.1f}%</b> of filtered orders are flagged high risk, with
    <b>{format_currency(metrics["revenue_at_risk"])}</b> currently at risk and
    <b>{format_currency(metrics["protected_revenue"])}</b> already protected across closed flagged cases.
    There are <b>{metrics["high_cases"]:,}</b> high-priority cases.
    Return risk pressure is highest in <b>{_html.escape(str(top_category))}</b>, and fraud pressure is strongest in
    <b>{_html.escape(str(top_state))}</b>.
  </div>
</div>
""", unsafe_allow_html=True)


def render_section_note(html_body: str):
    st.markdown(f'<div class="riq-note">{html_body}</div>', unsafe_allow_html=True)


def render_outcome_summary(
    closed_count: int,
    protected_revenue: float,
    open_count: int,
    breakdown: dict | None = None,
):
    st.markdown(f"""
<div class="riq-note" style="border-left-color:#16a34a;">
  <b>Outcome Summary:</b> {closed_count:,} resolved flagged cases protecting
  <b style="color:#16a34a;">{format_currency(protected_revenue)}</b> in revenue &nbsp;·&nbsp;
  <b style="color:#dc2626;">{open_count:,}</b> flagged cases remain active for review.
</div>
""", unsafe_allow_html=True)

    if not breakdown:
        return

    _OUTCOME_ACCENTS = {
        "Chargeback Prevented": "#16a34a",
        "Return Stopped": "#d97706",
        "Fraud Confirmed & Blocked": "#dc2626",
        "Verified & Released": "#2563eb",
        "Return Reviewed & Denied": "#7c3aed",
        "Return Approved": "#64748b",
    }

    cells = ""
    for outcome, stats in sorted(breakdown.items(), key=lambda x: -x[1]["revenue"]):
        accent = _OUTCOME_ACCENTS.get(outcome, "#64748b")
        label = _html.escape(outcome)
        count = stats["count"]
        rev = format_currency(stats["revenue"])
        cells += f"""<div style="background:#ffffff;border:1px solid #e2e8f0;
            border-left:3px solid {accent};border-radius:0 6px 6px 0;
            padding:0.55rem 0.85rem;">
          <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
            letter-spacing:0.05em;color:{accent};">{label}</div>
          <div style="font-size:0.92rem;font-weight:700;color:#1e293b;margin-top:2px;">{rev}</div>
          <div style="font-size:0.75rem;color:#64748b;">{count:,} cases</div>
        </div>"""

    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));'
        f'gap:0.6rem;margin-top:0.5rem;">{cells}</div>',
        unsafe_allow_html=True,
    )


# ── Case components ─────────────────────────────────────────────────────────────

def render_case_detail_box(selected_case: pd.Series):
    priority    = str(selected_case["priority_tier"])
    risk_type   = str(selected_case["risk_type_display"])
    case_status = str(selected_case["case_status"])

    order_id    = _html.escape(str(selected_case["order_id"]))
    customer_id = _html.escape(str(selected_case["customer_id"]))
    order_date  = str(pd.to_datetime(selected_case["order_date"]).date())
    product     = _html.escape(f"{selected_case['product_name']} ({selected_case['product_category']})")
    location    = _html.escape(f"{selected_case['state']}, {selected_case['country']}")
    payment     = _html.escape(str(selected_case["payment_method"]))
    workflow    = _html.escape(str(selected_case["workflow_stage"]))
    action      = _html.escape(str(selected_case["recommended_action"]))
    owner       = _html.escape(str(selected_case["action_owner"]))
    handling    = _html.escape(str(selected_case["automation_level"]))
    latest_action = _html.escape(str(selected_case.get("latest_action_taken", "No action yet")))
    reason      = _html.escape(str(selected_case["risk_reason_summary"]))
    next_step   = _html.escape(str(selected_case["ai_agent_next_step"]))
    decision    = _html.escape(str(selected_case["recommended_action"]))

    def _field(label: str, value: str) -> str:
        return (
            f'<div>'
            f'<div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.05em;color:#94a3b8;">{label}</div>'
            f'<div style="font-size:0.84rem;color:#334155;margin-top:1px;">{value}</div>'
            f'</div>'
        )

    st.markdown(f"""
<div class="riq-detail">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.8rem;">
    <div>
      <span style="font-size:0.68rem;font-weight:600;text-transform:uppercase;
        letter-spacing:0.06em;color:#94a3b8;">Order</span>
      <span style="font-size:1rem;font-weight:700;color:#1e293b;margin-left:0.4rem;">{order_id}</span>
    </div>
    <div style="display:flex;gap:0.35rem;">
      {_badge(priority)}{_badge(risk_type)}{_badge(case_status)}
    </div>
  </div>
  <div style="background:#f8f9fb;border:1px solid #e2e8f0;border-left:3px solid #2563eb;
    border-radius:0 6px 6px 0;padding:0.6rem 0.8rem;margin-bottom:0.8rem;">
    <div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;
      letter-spacing:0.06em;color:#94a3b8;margin-bottom:0.2rem;">Decision Recommendation</div>
    <div style="font-size:0.9rem;font-weight:700;color:#1e293b;">{decision}</div>
    <div style="font-size:0.82rem;color:#475569;margin-top:0.15rem;">
      Use this as the merchant-facing action before approving, holding, verifying, or closing the order.
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.55rem 1.5rem;margin-bottom:0.8rem;">
    {_field("Customer", customer_id)}
    {_field("Order Date", order_date)}
    {_field("Product", product)}
    {_field("Location", location)}
    {_field("Payment", payment)}
    {_field("Workflow Stage", workflow)}
    {_field("Latest Action", latest_action)}
    {_field("Recommended Action", action)}
    {_field("Action Owner", owner)}
    {_field("Handling Mode", handling)}
  </div>
  <div style="border-top:1px solid #e2e8f0;padding-top:0.65rem;">
    <div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;
      letter-spacing:0.06em;color:#94a3b8;margin-bottom:0.25rem;">Why Flagged</div>
    <div style="font-size:0.84rem;color:#334155;">{reason}</div>
  </div>
  <div style="margin-top:0.55rem;">
    <div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;
      letter-spacing:0.06em;color:#94a3b8;margin-bottom:0.25rem;">AI Suggested Next Step</div>
    <div style="font-size:0.84rem;color:#334155;background:#f8f9fb;
      border:1px solid #e2e8f0;border-radius:5px;padding:0.45rem 0.7rem;">{next_step}</div>
  </div>
</div>
""", unsafe_allow_html=True)


def render_timeline_html(steps):
    status_styles = {
        "done":    ("background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0;", "✓ "),
        "current": ("background:#eff6ff;color:#2563eb;border:1px solid #bfdbfe;", ""),
        "pending": ("background:#f8f9fb;color:#94a3b8;border:1px solid #e2e8f0;", ""),
    }

    parts = []
    for i, (label, status) in enumerate(steps):
        style, prefix = status_styles.get(status, status_styles["pending"])
        parts.append(
            f'<div style="display:flex;align-items:center;">'
            f'<div style="padding:5px 11px;border-radius:999px;{style}'
            f'font-size:0.8rem;font-weight:600;white-space:nowrap;">'
            f'{prefix}{_html.escape(label)}'
            f'</div></div>'
        )
        if i < len(steps) - 1:
            parts.append('<div style="font-size:1rem;color:#cbd5e1;padding:0 4px;">→</div>')

    return f"""
<html>
<body style="margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;background:#ffffff;">
  <div style="padding:10px 14px;border:1px solid #e2e8f0;border-radius:8px;background:#ffffff;">
    <div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;
      letter-spacing:0.07em;color:#94a3b8;margin-bottom:10px;">Case Timeline</div>
    <div style="display:flex;align-items:center;flex-wrap:wrap;gap:6px;">
      {''.join(parts)}
    </div>
  </div>
</body>
</html>"""


# ── Verification templates ──────────────────────────────────────────────────────

_VERIFICATION_TEMPLATES = [
    {
        "label": "Hold — Identity Verification",
        "scenario": "High-fraud score, new customer or flagged geography",
        "body": (
            "Hi {customer_id},\n\n"
            "We're doing a quick security check on order {order_id} before we ship.\n\n"
            "Could you confirm: (1) the billing address on file, and (2) the last 4 digits "
            "of your payment card?\n\n"
            "Reply within 24 hours and we'll get your order on its way. If we don't hear back, "
            "the order will be held for your protection.\n\n"
            "Thanks,\nRisk & Trust Team"
        ),
    },
    {
        "label": "Return — Inspection Required",
        "scenario": "Repeat returner or high-return-risk product category",
        "body": (
            "Hi {customer_id},\n\n"
            "We received your return request for order {order_id}. Based on your return history, "
            "we need to inspect the item before processing the refund.\n\n"
            "Please ship the item back in its original packaging. Once we receive and verify it, "
            "we'll process your refund or exchange within 3–5 business days.\n\n"
            "Thanks,\nReturns Team"
        ),
    },
    {
        "label": "Soft Review — Friendly Check",
        "scenario": "Moderate signals — order needs brief review before fulfilling",
        "body": (
            "Hi {customer_id},\n\n"
            "We noticed something unusual on order {order_id} and wanted to check in. "
            "We may need a moment to verify a few details before we ship.\n\n"
            "No action is needed from you right now — we'll follow up within 4 hours "
            "if anything is required.\n\n"
            "Thanks,\nCustomer Trust Team"
        ),
    },
]


def render_verification_templates(order_id: str = "{order_id}", customer_id: str = "{customer_id}"):
    with st.expander("Customer Message Templates", expanded=False):
        st.caption("Pre-written messages for common risk scenarios. Copy and customize before sending.")
        for tmpl in _VERIFICATION_TEMPLATES:
            st.markdown(f"**{_html.escape(tmpl['label'])}**  \n"
                        f"<span style='font-size:0.78rem;color:#64748b;'>{_html.escape(tmpl['scenario'])}</span>",
                        unsafe_allow_html=True)
            body = tmpl["body"].format(order_id=order_id, customer_id=customer_id)
            st.code(body, language=None)


# ── AI output components ────────────────────────────────────────────────────────

def render_ai_answer(text: str):
    escaped = _html.escape(str(text))
    st.markdown(f'<div class="riq-answer">{escaped}</div>', unsafe_allow_html=True)


def render_ai_output(text: str):
    escaped = _html.escape(str(text))
    st.markdown(f'<div class="riq-output">{escaped}</div>', unsafe_allow_html=True)


# ── Footer ──────────────────────────────────────────────────────────────────────

def render_insight_footer(metrics: dict, top_category: str, top_state: str):
    st.markdown("---")
    st.subheader("AI Risk Insights")

    insight_text = f"""
- **{metrics["flagged_pct"]:.1f}%** of filtered orders are flagged high risk.
- **{metrics["high_cases"]:,}** cases are currently marked **High** priority and should be handled first.
- **{metrics["open_cases"]:,}** cases are currently open, while **{metrics["closed_cases"]:,}** have already been resolved.
- **{format_currency(metrics["protected_revenue"])}** in revenue has already been protected across closed flagged cases.
- **{_html.escape(str(top_category))}** is the strongest return risk category.
- **{_html.escape(str(top_state))}** has the highest average fraud score.
"""

    st.info(insight_text)
