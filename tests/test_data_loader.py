import pandas as pd
import pytest

from src.data_loader import REQUIRED_COLUMNS, load_data


def _valid_row(**overrides):
    row = {
        "order_id": "O00001",
        "customer_id": "C1000",
        "order_date": "2025-01-01",
        "product_category": "Apparel",
        "product_name": "Hoodie",
        "payment_method": "Stripe",
        "country": "US",
        "state": "CA",
        "order_value": 100.0,
        "past_orders": 5,
        "past_returns": 1,
        "chargeback_history": 0,
        "discount_used": 0,
        "expedited_shipping": 0,
        "fraud_score": 0.1,
        "return_risk_score": 0.2,
        "unified_risk_score": 0.2,
        "risk_level": "High",
        "risk_type": "Returns",
        "recommended_action": "Review Return Behavior",
        "ai_agent_next_step": "Generate return risk summary",
        "action_owner": "Returns Ops",
        "automation_level": "Assisted",
        "risk_reason_summary": "elevated return risk score",
        "priority_tier": "High",
        "case_status": "Open",
        "workflow_stage": "Review Queue",
    }
    row.update(overrides)
    return row


def test_load_data_validates_and_adds_display_type(tmp_path):
    path = tmp_path / "transactions.csv"
    pd.DataFrame([_valid_row()]).to_csv(path, index=False)

    df = load_data(str(path))

    assert list(df.columns).count("risk_type_display") == 1
    assert df.loc[0, "risk_type_display"] == "Return Risk"
    assert pd.api.types.is_datetime64_any_dtype(df["order_date"])


def test_load_data_maps_missing_risk_type_to_none_display(tmp_path):
    path = tmp_path / "transactions.csv"
    pd.DataFrame([_valid_row(risk_type=None)]).to_csv(path, index=False)

    df = load_data(str(path))

    assert df.loc[0, "risk_type_display"] == "None"




def test_load_data_drops_invalid_dates(tmp_path):
    path = tmp_path / "transactions.csv"
    pd.DataFrame([
        _valid_row(order_id="O00001", order_date="2025-01-01"),
        _valid_row(order_id="O00002", order_date="not-a-date"),
    ]).to_csv(path, index=False)

    df = load_data(str(path))

    assert df["order_id"].tolist() == ["O00001"]


def test_load_data_raises_for_missing_required_columns(tmp_path):
    path = tmp_path / "transactions.csv"
    columns = [c for c in REQUIRED_COLUMNS if c != "order_id"]
    pd.DataFrame([{c: "x" for c in columns}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_data(str(path))


def test_load_data_returns_none_for_missing_file(tmp_path):
    assert load_data(str(tmp_path / "missing.csv")) is None
