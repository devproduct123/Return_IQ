import pandas as pd

from src import ui_components


class _FakeStreamlit:
    def __init__(self):
        self.markdowns = []
        self.infos = []
        self.subheaders = []

    def markdown(self, body, unsafe_allow_html=False):
        self.markdowns.append({
            "body": body,
            "unsafe_allow_html": unsafe_allow_html,
        })

    def info(self, body):
        self.infos.append(body)

    def subheader(self, body):
        self.subheaders.append(body)


def test_render_timeline_html_contains_steps_and_status_styles():
    html = ui_components.render_timeline_html([
        ("Case Created", "done"),
        ("Review Queue", "current"),
        ("Pending Resolution", "pending"),
    ])

    assert "Case Timeline" in html
    assert "Case Created" in html
    assert "Review Queue" in html
    assert "Pending Resolution" in html
    assert "background:#dcfce7" in html
    assert "background:#dbeafe" in html
    assert "background:#f3f4f6" in html


def test_render_hero_summary_preserves_text(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(ui_components, "st", fake_st)

    ui_components.render_hero_summary(
        {
            "flagged_pct": 12.3,
            "revenue_at_risk": 1200.0,
            "protected_revenue": 500.0,
            "critical_cases": 4,
        },
        "Apparel",
        "CA",
    )

    rendered = fake_st.markdowns[0]
    assert rendered["unsafe_allow_html"] is True
    assert "Risk Intelligence Summary" in rendered["body"]
    assert "$1.2K" in rendered["body"]
    assert "$500" in rendered["body"]
    assert "Return-abuse pressure is highest in <b>Apparel</b>" in rendered["body"]
    assert "fraud pressure is strongest in" in rendered["body"]


def test_render_case_detail_box_preserves_case_fields(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(ui_components, "st", fake_st)

    selected_case = pd.Series({
        "order_id": "O00001",
        "customer_id": "C1000",
        "order_date": "2025-01-01",
        "risk_type_display": "Fraud",
        "product_name": "Hoodie",
        "product_category": "Apparel",
        "state": "CA",
        "country": "US",
        "payment_method": "Stripe",
        "workflow_stage": "Review Queue",
        "recommended_action": "Hold Fulfillment",
        "action_owner": "Fraud Ops",
        "automation_level": "Semi-Auto",
        "risk_reason_summary": "high-value order",
        "ai_agent_next_step": "Pause shipment and verify customer",
    })

    ui_components.render_case_detail_box(selected_case)

    rendered = fake_st.markdowns[0]
    assert rendered["unsafe_allow_html"] is True
    assert "Case Summary: O00001" in rendered["body"]
    assert "<b>Customer ID:</b> C1000" in rendered["body"]
    assert "<b>Order Date:</b> 2025-01-01" in rendered["body"]
    assert "<b>AI Suggested Next Step:</b> Pause shipment and verify customer" in rendered["body"]


def test_render_insight_footer_preserves_info_and_pilot_note(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(ui_components, "st", fake_st)

    ui_components.render_insight_footer(
        {
            "flagged_pct": 10.0,
            "critical_cases": 2,
            "open_cases": 3,
            "closed_cases": 4,
            "protected_revenue": 1500.0,
        },
        "Footwear",
        "NY",
    )

    assert fake_st.subheaders == ["AI Risk Insights"]
    assert "**10.0%** of filtered orders are flagged high risk" in fake_st.infos[0]
    assert "**$1.5K** in revenue has already been protected" in fake_st.infos[0]
    assert "Pilot with Aignita" in fake_st.markdowns[-1]["body"]
