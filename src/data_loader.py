import os

import pandas as pd


REQUIRED_COLUMNS = [
    "order_id", "customer_id", "order_date", "product_category", "product_name",
    "payment_method", "country", "state", "order_value", "past_orders",
    "past_returns", "chargeback_history", "discount_used", "expedited_shipping",
    "fraud_score", "return_abuse_score", "unified_risk_score", "risk_level",
    "risk_type", "recommended_action", "ai_agent_next_step", "action_owner",
    "automation_level", "risk_reason_summary", "priority_tier", "case_status",
    "workflow_stage"
]


def load_data(path: str = "data/transactions.csv"):
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"])

    df["risk_type_display"] = df["risk_type"].replace({
        "Returns": "Return Abuse"
    })

    return df
