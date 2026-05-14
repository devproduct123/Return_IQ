import os
from typing import Any

import pandas as pd

from src.integrations.base import CommerceConnector
from src.integrations.shopify import ShopifyConnector


REQUIRED_COLUMNS = [
    "order_id", "customer_id", "order_date", "product_category", "product_name",
    "payment_method", "country", "state", "order_value", "past_orders",
    "past_returns", "chargeback_history", "discount_used", "expedited_shipping",
    "fraud_score", "return_risk_score", "unified_risk_score", "risk_level",
    "risk_type", "recommended_action", "ai_agent_next_step", "action_owner",
    "automation_level", "risk_reason_summary", "priority_tier", "case_status",
    "workflow_stage"
]


def _finalize_loaded_data(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"]).copy()

    df["risk_type_display"] = (
        df["risk_type"]
        .fillna("None")
        .replace({
            "Fraud": "Fraud Risk",
            "Returns": "Return Risk",
        })
    )

    return df


def load_csv_data(path: str = "data/transactions.csv"):
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    return _finalize_loaded_data(df)


def _first_line_item(order: dict[str, Any]) -> dict[str, Any]:
    line_items = order.get("line_items") or []
    return line_items[0] if line_items else {}


def _first_gateway(order: dict[str, Any]) -> str:
    gateways = order.get("payment_gateway_names") or []
    return str(gateways[0]) if gateways else "Unknown"


def _address_value(order: dict[str, Any], key: str, default: str = "Unknown") -> str:
    shipping = order.get("shipping_address") or {}
    billing = order.get("billing_address") or {}
    return str(shipping.get(key) or billing.get(key) or default)


def normalize_shopify_orders(raw_orders: list[dict[str, Any]]) -> pd.DataFrame:
    """Map Shopify order payloads into the RiskIQ dashboard schema."""

    rows = []
    for order in raw_orders:
        line_item = _first_line_item(order)
        customer = order.get("customer") or {}
        order_value = float(order.get("total_price") or 0.0)
        refunded = bool(order.get("refunds"))
        cancelled = bool(order.get("cancelled_at"))
        expedited = "express" in str(order.get("shipping_lines", "")).lower()
        fraud_score = 0.35 if cancelled else 0.08
        return_risk_score = 0.32 if refunded else 0.10
        unified_risk_score = max(fraud_score, return_risk_score)

        if cancelled:
            risk_level = "High"
            risk_type = "Fraud"
            recommended_action = "Soft Review"
            ai_next_step = "Review cancellation, payment, and fulfillment signals before approving future orders."
            action_owner = "Fraud Ops"
            automation_level = "Assisted"
            risk_reason = "Shopify order has cancellation signal"
            priority_tier = "Moderate"
            case_status = "Open"
            workflow_stage = "Review Queue"
        elif refunded:
            risk_level = "High"
            risk_type = "Returns"
            recommended_action = "Soft Review"
            ai_next_step = "Review refund and return history before approving repeat returns."
            action_owner = "Returns Ops"
            automation_level = "Assisted"
            risk_reason = "Shopify order has refund signal"
            priority_tier = "Moderate"
            case_status = "Open"
            workflow_stage = "Review Queue"
        else:
            risk_level = "Low"
            risk_type = "None"
            recommended_action = "Allow"
            ai_next_step = "No action needed"
            action_owner = "System"
            automation_level = "Auto"
            risk_reason = "standard low-risk profile from Shopify sync"
            priority_tier = "Low"
            case_status = "Closed"
            workflow_stage = "Resolved"

        rows.append({
            "order_id": str(order.get("name") or order.get("id")),
            "customer_id": str(customer.get("id") or "Unknown"),
            "order_date": order.get("created_at"),
            "product_category": str(line_item.get("product_type") or "Unknown"),
            "product_name": str(line_item.get("title") or "Unknown"),
            "payment_method": _first_gateway(order),
            "country": _address_value(order, "country_code"),
            "state": _address_value(order, "province_code"),
            "order_value": order_value,
            "past_orders": 1,
            "past_returns": 1 if refunded else 0,
            "chargeback_history": 0,
            "discount_used": 1 if order.get("discount_codes") else 0,
            "expedited_shipping": int(expedited),
            "fraud_score": round(fraud_score, 3),
            "return_risk_score": round(return_risk_score, 3),
            "unified_risk_score": round(unified_risk_score, 3),
            "risk_level": risk_level,
            "risk_type": risk_type,
            "recommended_action": recommended_action,
            "ai_agent_next_step": ai_next_step,
            "action_owner": action_owner,
            "automation_level": automation_level,
            "risk_reason_summary": risk_reason,
            "priority_tier": priority_tier,
            "case_status": case_status,
            "workflow_stage": workflow_stage,
        })

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def load_connector_data(connector: CommerceConnector, limit: int = 100) -> pd.DataFrame:
    raw_orders = connector.fetch_orders(limit=limit)
    df = normalize_shopify_orders(raw_orders)
    return _finalize_loaded_data(df)


def load_shopify_data(
    shop_domain: str,
    access_token: str,
    limit: int = 100,
    api_version: str = "2024-10",
) -> pd.DataFrame:
    connector = ShopifyConnector(
        shop_domain=shop_domain,
        access_token=access_token,
        api_version=api_version,
    )
    return load_connector_data(connector, limit=limit)


def load_data(
    path: str = "data/transactions.csv",
    source: str = "csv",
    shop_domain: str | None = None,
    access_token: str | None = None,
    limit: int = 100,
):
    if source == "csv":
        return load_csv_data(path)

    if source == "shopify":
        if not shop_domain or not access_token:
            raise ValueError(
                "Shopify source requires SHOPIFY_SHOP_DOMAIN and SHOPIFY_ACCESS_TOKEN."
            )
        return load_shopify_data(
            shop_domain=shop_domain,
            access_token=access_token,
            limit=limit,
        )

    raise ValueError(f"Unsupported data source: {source}")
