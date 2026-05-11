import pandas as pd

from src.metrics import (
    compute_summary_metrics,
    format_currency,
    safe_top_index,
    to_csv_download,
)


def test_format_currency_matches_existing_behavior():
    assert format_currency(999) == "$999"
    assert format_currency(1_200) == "$1.2K"
    assert format_currency(1_250_000) == "$1.2M"


def test_safe_top_index_returns_first_index_or_default():
    series = pd.Series([10, 5], index=["A", "B"])

    assert safe_top_index(series) == "A"
    assert safe_top_index(pd.Series(dtype=float)) == "N/A"
    assert safe_top_index(pd.Series(dtype=float), default="None") == "None"


def test_to_csv_download_returns_utf8_csv_bytes():
    df = pd.DataFrame([{"order_id": "O00001", "order_value": 100}])

    assert to_csv_download(df) == b"order_id,order_value\nO00001,100\n"


def test_compute_summary_metrics():
    df = pd.DataFrame([
        {
            "risk_level": "High",
            "risk_type_display": "Fraud Risk",
            "order_value": 100.0,
            "priority_tier": "High",
            "case_status": "Open",
        },
        {
            "risk_level": "High",
            "risk_type_display": "Return Risk",
            "order_value": 200.0,
            "priority_tier": "High",
            "case_status": "Closed",
        },
        {
            "risk_level": "Low",
            "risk_type_display": "None",
            "order_value": 50.0,
            "priority_tier": "Low",
            "case_status": "Closed",
        },
    ])

    assert compute_summary_metrics(df) == {
        "total_orders": 3,
        "flagged_orders": 2,
        "flagged_pct": 66.7,
        "fraud_cases": 1,
        "return_cases": 1,
        "revenue_at_risk": 300.0,
        "high_cases": 2,
        "open_cases": 1,
        "closed_cases": 2,
        "protected_revenue": 200.0,
    }


def test_compute_summary_metrics_handles_empty_data():
    df = pd.DataFrame(columns=[
        "risk_level",
        "risk_type_display",
        "order_value",
        "priority_tier",
        "case_status",
    ])

    assert compute_summary_metrics(df)["flagged_pct"] == 0.0
