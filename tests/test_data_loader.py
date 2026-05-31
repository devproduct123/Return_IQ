import pandas as pd
import pytest

from src.data_loader import (
    REQUIRED_COLUMNS,
    derive_latest_action_taken,
    load_connector_data,
    load_data,
    normalize_shopify_orders,
    save_synced_data,
)


class _FakeConnector:
    def __init__(self, orders):
        self.orders = orders

    def fetch_orders(self, limit=100):
        return self.orders[:limit]


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
        "latest_action_taken": "Soft review queued",
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


def test_load_data_derives_latest_action_when_missing(tmp_path):
    path = tmp_path / "transactions.csv"
    row = _valid_row()
    row.pop("latest_action_taken")
    pd.DataFrame([row]).to_csv(path, index=False)

    df = load_data(str(path))

    assert df.loc[0, "latest_action_taken"] == "Soft review queued"


def test_derive_latest_action_taken_for_common_workflow_states():
    assert derive_latest_action_taken("Allow", "Closed", "Resolved") == "Order approved"
    assert derive_latest_action_taken(
        "Hold Fulfillment", "Open", "Verification Sent"
    ) == "Customer verification sent"
    assert derive_latest_action_taken(
        "Restrict Return / Refund", "Pending", "Awaiting Item Inspection"
    ) == "Awaiting item inspection"




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


def test_normalize_shopify_orders_maps_to_required_schema():
    df = normalize_shopify_orders([
        {
            "id": 123,
            "name": "#1001",
            "created_at": "2026-01-01T10:00:00Z",
            "total_price": "129.50",
            "customer": {"id": 456},
            "line_items": [{"title": "Premium Hoodie", "product_type": "Apparel"}],
            "shipping_address": {"country_code": "US", "province_code": "CA"},
            "payment_gateway_names": ["shopify_payments"],
            "refunds": [],
            "cancelled_at": None,
        }
    ])

    for column in REQUIRED_COLUMNS:
        assert column in df.columns
    assert df.loc[0, "order_id"] == "#1001"
    assert df.loc[0, "customer_id"] == "456"
    assert df.loc[0, "product_category"] == "Apparel"
    assert df.loc[0, "payment_method"] == "shopify_payments"
    assert df.loc[0, "risk_type"] == "None"
    assert df.loc[0, "recommended_action"] == "Allow"
    assert df.loc[0, "latest_action_taken"] == "Order approved"
    assert df.loc[0, "return_reason_code"] == "None"
    assert df.loc[0, "fraud_label"] == "Unknown"


def test_load_connector_data_finalizes_shopify_payload():
    connector = _FakeConnector([
        {
            "id": 123,
            "created_at": "2026-01-01T10:00:00Z",
            "total_price": "129.50",
            "customer": {"id": 456},
            "line_items": [{"title": "Premium Hoodie", "product_type": "Apparel"}],
            "shipping_address": {"country_code": "US", "province_code": "CA"},
            "payment_gateway_names": ["shopify_payments"],
            "refunds": [{"id": 1}],
        }
    ])

    df = load_connector_data(connector)

    assert df.loc[0, "risk_type_display"] == "Return Risk"
    assert pd.api.types.is_datetime64_any_dtype(df["order_date"])


def test_load_data_requires_shopify_credentials():
    with pytest.raises(ValueError, match="SHOPIFY_SHOP_DOMAIN"):
        load_data(source="shopify")


def test_save_synced_data_drops_display_column_and_writes_csv(tmp_path):
    path = tmp_path / "shopify_transactions.csv"
    df = pd.DataFrame([_valid_row()])
    df["risk_type_display"] = "Return Risk"

    saved_path = save_synced_data(df, str(path))
    saved_df = pd.read_csv(saved_path)

    assert saved_path == str(path)
    assert "risk_type_display" not in saved_df.columns
    assert saved_df.loc[0, "order_id"] == "O00001"
