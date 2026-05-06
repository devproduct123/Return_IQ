import pandas as pd

from src.risk_logic import (
    answer_case_question,
    answer_risk_question,
    build_case_timeline,
    get_open_high_risk_cases,
    select_fallback_case,
    sort_cases_by_priority,
)


def _case_row(**overrides):
    row = {
        "order_id": "O00001",
        "customer_id": "C1000",
        "risk_type_display": "Fraud",
        "priority_tier": "Critical",
        "case_status": "Open",
        "workflow_stage": "Review Queue",
        "recommended_action": "Hold Fulfillment",
        "ai_agent_next_step": "Pause shipment and verify customer",
        "risk_reason_summary": "high-value order",
        "fraud_score": 0.8,
        "return_abuse_score": 0.2,
        "unified_risk_score": 0.8,
        "order_value": 300.0,
        "past_returns": 1,
        "chargeback_history": 0,
        "payment_method": "Stripe",
        "product_name": "Hoodie",
        "product_category": "Apparel",
        "state": "CA",
        "country": "US",
        "risk_level": "High",
    }
    row.update(overrides)
    return pd.Series(row)


def _risk_data():
    return pd.DataFrame([
        {
            "order_id": "O00001",
            "risk_level": "High",
            "risk_type_display": "Fraud",
            "order_value": 300.0,
            "case_status": "Open",
            "priority_tier": "Critical",
            "product_category": "Apparel",
            "product_name": "Hoodie",
            "return_abuse_score": 0.2,
            "state": "CA",
            "fraud_score": 0.8,
            "recommended_action": "Hold Fulfillment",
            "unified_risk_score": 0.8,
        },
        {
            "order_id": "O00002",
            "risk_level": "High",
            "risk_type_display": "Return Abuse",
            "order_value": 200.0,
            "case_status": "Closed",
            "priority_tier": "High",
            "product_category": "Footwear",
            "product_name": "Boots",
            "return_abuse_score": 0.7,
            "state": "NY",
            "fraud_score": 0.1,
            "recommended_action": "Review Return Behavior",
            "unified_risk_score": 0.7,
        },
        {
            "order_id": "O00003",
            "risk_level": "Low",
            "risk_type_display": "None",
            "order_value": 50.0,
            "case_status": "Closed",
            "priority_tier": "Low",
            "product_category": "Apparel",
            "product_name": "Hoodie",
            "return_abuse_score": 0.1,
            "state": "CA",
            "fraud_score": 0.05,
            "recommended_action": "Allow",
            "unified_risk_score": 0.1,
        },
    ])


def test_build_case_timeline_for_open_case():
    assert build_case_timeline(_case_row()) == [
        ("Case Created", "done"),
        ("Review Queue", "current"),
        ("Action: Hold Fulfillment", "current"),
        ("Pending Resolution", "pending"),
    ]


def test_build_case_timeline_for_closed_allowed_case():
    assert build_case_timeline(_case_row(
        case_status="Closed",
        workflow_stage="Resolved",
        recommended_action="Allow",
    )) == [
        ("Case Created", "done"),
        ("Resolved", "current"),
        ("Auto Approval Completed", "done"),
        ("Resolved", "done"),
    ]


def test_answer_risk_question_summary_matches_existing_output():
    assert answer_risk_question("Summarize current risk", _risk_data()) == (
        "In the current filtered view, 2 out of 3 orders are flagged high risk. "
        "Revenue at risk is $500. "
        "Protected revenue on closed flagged cases is $200. "
        "There are 1 open cases. "
        "Return abuse is highest in Footwear, fraud risk is strongest in CA, "
        "and the most common recommended action is Hold Fulfillment."
    )


def test_answer_risk_question_handles_empty_data():
    assert answer_risk_question("summary", pd.DataFrame()) == (
        "No data is available for the current filters."
    )


def test_answer_case_question_why_and_none_case():
    assert answer_case_question("why was this flagged?", _case_row()) == (
        "This case was flagged as Fraud. "
        "Fraud score is 0.800, return-abuse score is 0.200, "
        "and unified risk score is 0.800. "
        "Primary reason: high-value order"
    )
    assert answer_case_question("summary", None) == (
        "No case is selected. Please select an order in Ops Queue first."
    )


def test_open_high_risk_cases_and_priority_sorting():
    cases = pd.DataFrame([
        _case_row(order_id="LOW", priority_tier="Low", unified_risk_score=0.9,
                  order_value=900.0),
        _case_row(order_id="HIGH_LOW_VALUE", priority_tier="High",
                  unified_risk_score=0.6, order_value=100.0),
        _case_row(order_id="HIGH_HIGH_VALUE", priority_tier="High",
                  unified_risk_score=0.6, order_value=200.0),
        _case_row(order_id="CRITICAL", priority_tier="Critical",
                  unified_risk_score=0.1, order_value=50.0),
        _case_row(order_id="CLOSED", priority_tier="Critical",
                  case_status="Closed", unified_risk_score=1.0,
                  order_value=1000.0),
    ])

    open_cases = get_open_high_risk_cases(cases)
    sorted_cases = sort_cases_by_priority(open_cases)

    assert sorted_cases["order_id"].tolist() == [
        "CRITICAL",
        "HIGH_HIGH_VALUE",
        "HIGH_LOW_VALUE",
        "LOW",
    ]
    assert "priority_sort" in sorted_cases.columns


def test_select_fallback_case_uses_highest_priority_open_case():
    selected_case, used_fallback = select_fallback_case(pd.DataFrame([
        _case_row(order_id="LOW", priority_tier="Low", unified_risk_score=0.9,
                  order_value=900.0),
        _case_row(order_id="CRITICAL", priority_tier="Critical",
                  unified_risk_score=0.1, order_value=50.0),
    ]))

    assert used_fallback is True
    assert selected_case["order_id"] == "CRITICAL"


def test_select_fallback_case_handles_no_open_cases():
    selected_case, used_fallback = select_fallback_case(pd.DataFrame([
        _case_row(order_id="CLOSED", case_status="Closed"),
    ]))

    assert selected_case is None
    assert used_fallback is False
