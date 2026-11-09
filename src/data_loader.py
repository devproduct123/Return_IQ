import os
from typing import Any

import pandas as pd

from src.integrations.base import CommerceConnector
from src.integrations.shopify import ShopifyConnector


REQUIRED_COLUMNS = [
    "order_id", "customer_id", "order_date", "product_category", "product_name",
    "payment_method", "country", "state", "order_value", "past_orders",
    "past_returns", "discount_used", "expedited_shipping",
    "return_risk_score", "unified_risk_score", "risk_level",
    "risk_type", "recommended_action", "ai_agent_next_step", "action_owner",
    "automation_level", "risk_reason_summary", "priority_tier", "case_status",
    "workflow_stage", "latest_action_taken"
]
SHOPIFY_CACHE_PATH = "data/shopify_transactions.csv"
LIVE_TRANSACTIONS_PATH = "data/live_transactions.csv"
LIVE_DEMO_BASELINE_PATH = "data/live_transactions_demo_baseline.csv"


def derive_latest_action_taken(
    recommended_action: str,
    case_status: str,
    workflow_stage: str,
) -> str:
    if case_status == "Closed" or workflow_stage == "Resolved":
        if recommended_action == "Allow":
            return "Order approved"
        return "Case resolved"

    if workflow_stage == "Case Created":
        return "Case created"
    if workflow_stage == "Review Queue":
        return "Soft review queued"
    if workflow_stage == "Awaiting Item Inspection":
        return "Awaiting item inspection"
    if workflow_stage == "High Returner Tagged":
        return "High returner tagged"
    if workflow_stage == "Escalated to Senior Review":
        return "Escalated for senior review"
    if workflow_stage == "Watchlist Monitoring":
        return "Case under watchlist monitoring"

    if recommended_action == "Restrict Return / Refund":
        return "Return/refund restricted"
    if recommended_action == "Soft Review":
        return "Soft review started"

    return "No action yet"


def _finalize_loaded_data(df: pd.DataFrame) -> pd.DataFrame:
    if "latest_action_taken" not in df.columns:
        df["latest_action_taken"] = df.apply(
            lambda row: derive_latest_action_taken(
                str(row.get("recommended_action", "")),
                str(row.get("case_status", "")),
                str(row.get("workflow_stage", "")),
            ),
            axis=1,
        )

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

    # Fill optional enriched columns with safe defaults when absent (e.g. Shopify source)
    _optional_defaults = {
        "customer_lifetime_days": 0,
        "action_reason": "",
        "outcome_type": "Pending",
        "return_requested": 0,
        "return_completed": 0,
        "return_reason_code": "None",
        "return_sub_reason": "None",
        "return_reason_text": "",
        "customer_return_comment": "",
        "ops_resolution": "Pending",
        "ops_resolution_note": "",
        "label_source": "System",
        "label_confidence": 0.0,
        "reviewed_by": "",
        "reviewed_at": "",
        "dataset_mode": "",
        "ops_action_taken": "",
        "ops_action_by": "",
        "ops_action_at": "",
        "ops_action_note": "",
        "ops_action_history": "",
        "ops_override_reason": "",
        "analyst_confidence": 0.0,
        "ops_resolution_reason": "",
    }
    for col, default in _optional_defaults.items():
        if col not in df.columns:
            df[col] = default

    return df


def load_csv_data(path: str = "data/transactions.csv"):
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    return _finalize_loaded_data(df)


def save_synced_data(df: pd.DataFrame, path: str = SHOPIFY_CACHE_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    export_df = df.drop(columns=["risk_type_display"], errors="ignore")
    export_df.to_csv(path, index=False)
    return path


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
    """Map Shopify order payloads into the Return Risk Intelligence dashboard schema."""

    rows = []
    for order in raw_orders:
        line_item = _first_line_item(order)
        customer = order.get("customer") or {}
        order_value = float(order.get("total_price") or 0.0)
        refunded = bool(order.get("refunds"))
        expedited = "express" in str(order.get("shipping_lines", "")).lower()
        return_risk_score = 0.42 if refunded else 0.10
        unified_risk_score = return_risk_score

        if refunded:
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
            "discount_used": 1 if order.get("discount_codes") else 0,
            "expedited_shipping": int(expedited),
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
            "latest_action_taken": derive_latest_action_taken(
                recommended_action,
                case_status,
                workflow_stage,
            ),
            "return_requested": int(refunded),
            "return_completed": int(refunded),
            "return_reason_code": "Shopify Refund" if refunded else "None",
            "return_sub_reason": "Refund Created" if refunded else "None",
            "return_reason_text": "Refund present in Shopify order payload" if refunded else "",
            "customer_return_comment": "",
            "ops_resolution": "Pending" if refunded else "Auto Approved",
            "ops_resolution_note": "",
            "label_source": "Shopify Sync",
            "label_confidence": 0.55 if refunded else 0.25,
            "reviewed_by": "",
            "reviewed_at": "",
            "dataset_mode": "Live",
        })

    return pd.DataFrame(rows)


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

    if source == "saved_shopify":
        return load_csv_data(SHOPIFY_CACHE_PATH)

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
