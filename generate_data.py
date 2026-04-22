import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

np.random.seed(42)
random.seed(42)

n = 10000

product_categories = ["Apparel", "Electronics", "Beauty", "Home", "Footwear"]
products_by_category = {
    "Apparel": ["Premium Jacket", "Cotton T-Shirt", "Denim Jeans", "Hoodie", "Sneakers"],
    "Electronics": ["Wireless Earbuds", "Bluetooth Speaker", "Smart Watch", "Tablet Case", "USB Hub"],
    "Beauty": ["Face Serum", "Lip Kit", "Skin Cream", "Hair Dryer", "Perfume"],
    "Home": ["Lamp", "Throw Blanket", "Storage Box", "Coffee Mug", "Desk Organizer"],
    "Footwear": ["Running Shoes", "Formal Shoes", "Sandals", "Boots", "Slippers"],
}

payment_methods = ["Stripe", "PayPal", "Shop Pay", "Apple Pay"]
countries = ["US", "CA", "UK", "AU"]
states = ["CA", "NY", "TX", "FL", "WA", "NJ", "IL", "GA"]

start_date = datetime(2025, 1, 1)
data = []

STAGE_MAP = {
    "Hold Fulfillment": [
        "Case Created",
        "Review Queue",
        "Verification Sent",
        "Awaiting Customer Response",
        "Action Applied",
        "Resolved",
    ],
    "Verify Customer": [
        "Case Created",
        "Review Queue",
        "Verification Sent",
        "Awaiting Customer Response",
        "Action Applied",
        "Resolved",
    ],
    "Restrict Return / Refund": [
        "Case Created",
        "Review Queue",
        "Policy Applied",
        "Awaiting Item Inspection",
        "Action Applied",
        "Resolved",
    ],
    "Review Return Behavior": [
        "Case Created",
        "Review Queue",
        "Watchlist Monitoring",
        "Policy Applied",
        "Action Applied",
        "Resolved",
    ],
    "Escalate for Investigation": [
        "Case Created",
        "Escalated",
        "Senior Review",
        "Action Applied",
        "Resolved",
    ],
    "Allow": [
        "Auto Approved",
        "Resolved",
    ],
}

PENDING_STAGES = {"Awaiting Customer Response", "Awaiting Item Inspection"}


def assign_actions(
    risk_level: str,
    risk_type: str,
    fraud_score: float,
    return_abuse_score: float,
    order_value: float,
    past_returns: int,
    chargeback_history: int,
):
    if risk_level == "Low":
        return {
            "recommended_action": "Allow",
            "ai_agent_next_step": "No action needed",
            "action_owner": "System",
            "automation_level": "Auto",
        }

    both_high = fraud_score >= 0.72 and return_abuse_score >= 0.72
    high_value = order_value >= 250
    repeat_returner = past_returns >= 3
    chargeback_risk = chargeback_history > 0

    if both_high:
        return {
            "recommended_action": "Escalate for Investigation",
            "ai_agent_next_step": "Generate case summary, cluster linked signals, and route to senior risk analyst",
            "action_owner": "Risk Ops",
            "automation_level": "Semi-Auto",
        }

    if risk_type == "Fraud":
        if fraud_score >= 0.78 or (high_value and chargeback_risk):
            return {
                "recommended_action": "Hold Fulfillment",
                "ai_agent_next_step": "Pause shipment, send identity verification request, and request payment re-authentication",
                "action_owner": "Fraud Ops",
                "automation_level": "Semi-Auto",
            }
        return {
            "recommended_action": "Verify Customer",
            "ai_agent_next_step": "Send address and identity verification workflow and draft fraud-review notes",
            "action_owner": "Fraud Ops",
            "automation_level": "Semi-Auto",
        }

    if risk_type == "Returns":
        if return_abuse_score >= 0.76 or repeat_returner:
            return {
                "recommended_action": "Restrict Return / Refund",
                "ai_agent_next_step": "Disable instant refund, require item inspection, and recommend store credit or exchange",
                "action_owner": "Returns Ops",
                "automation_level": "Semi-Auto",
            }
        return {
            "recommended_action": "Review Return Behavior",
            "ai_agent_next_step": "Generate return-abuse summary and flag account for watchlist monitoring",
            "action_owner": "Returns Ops",
            "automation_level": "Assisted",
        }

    return {
        "recommended_action": "Allow",
        "ai_agent_next_step": "No action needed",
        "action_owner": "System",
        "automation_level": "Auto",
    }


def build_risk_reason_summary(
    risk_type: str,
    category: str,
    state: str,
    order_value: float,
    past_returns: int,
    past_orders: int,
    chargeback_history: int,
    expedited_shipping: int,
    discount_used: int,
    fraud_score: float,
    return_abuse_score: float,
):
    reasons = []

    if risk_type == "Fraud":
        if chargeback_history > 0:
            reasons.append(f"{chargeback_history} prior chargeback signal(s)")
        if state in ["FL", "NJ"]:
            reasons.append(f"elevated-risk geography: {state}")
        if order_value >= 250:
            reasons.append("high-value order")
        if expedited_shipping == 1:
            reasons.append("expedited shipping pattern")
        if fraud_score >= 0.70:
            reasons.append("very high fraud score")

    elif risk_type == "Returns":
        return_rate = past_returns / max(past_orders, 1)
        if category == "Apparel":
            reasons.append("apparel return-abuse pattern")
        if category == "Footwear":
            reasons.append("footwear return pressure")
        if past_returns >= 3:
            reasons.append(
                f"repeat returner with {past_returns} historical returns")
        if return_rate >= 0.35:
            reasons.append(f"high historical return rate ({return_rate:.0%})")
        if discount_used == 1:
            reasons.append("discount-driven purchase behavior")
        if return_abuse_score >= 0.65:
            reasons.append("elevated return-abuse score")

    if not reasons:
        reasons.append("standard low-risk profile")

    return " | ".join(reasons[:3])


def assign_priority_tier(
    risk_level: str,
    risk_type: str,
    fraud_score: float,
    return_abuse_score: float,
    order_value: float,
):
    if risk_level == "Low":
        return "Low"

    if risk_type == "Fraud":
        if fraud_score >= 0.72 or (order_value >= 300 and fraud_score >= 0.60):
            return "Critical"
        if fraud_score >= 0.55:
            return "High"
        return "Moderate"

    if risk_type == "Returns":
        if return_abuse_score >= 0.68 or (order_value >= 250 and return_abuse_score >= 0.58):
            return "Critical"
        if return_abuse_score >= 0.50:
            return "High"
        return "Moderate"

    return "Low"


def derive_case_status(workflow_stage: str) -> str:
    if workflow_stage == "Resolved":
        return "Closed"
    if workflow_stage in PENDING_STAGES:
        return "Pending"
    return "Open"


def assign_case_status_and_stage(risk_level: str, recommended_action: str):
    if risk_level == "Low" or recommended_action == "Allow":
        return "Closed", "Resolved"

    stages = STAGE_MAP.get(
        recommended_action,
        ["Case Created", "Review Queue", "Action Applied", "Resolved"]
    )

    # Heavier weight on earlier / in-progress stages so the dataset feels realistic
    default_weights = {
        4: np.array([0.45, 0.30, 0.18, 0.07]),
        5: np.array([0.34, 0.26, 0.20, 0.14, 0.06]),
        6: np.array([0.28, 0.22, 0.18, 0.14, 0.10, 0.08]),
    }

    weights = default_weights.get(len(stages))
    if weights is None:
        weights = np.ones(len(stages)) / len(stages)
    else:
        weights = weights / weights.sum()

    idx = np.random.choice(len(stages), p=weights)
    workflow_stage = stages[idx]
    case_status = derive_case_status(workflow_stage)

    return case_status, workflow_stage


for i in range(n):
    order_id = f"O{i+1:05d}"
    customer_id = f"C{np.random.randint(1000, 3000)}"

    category = np.random.choice(product_categories, p=[
                                0.38, 0.18, 0.14, 0.15, 0.15])
    product_name = np.random.choice(products_by_category[category])

    payment_method = np.random.choice(payment_methods)
    country = np.random.choice(countries, p=[0.72, 0.10, 0.09, 0.09])
    state = np.random.choice(states)
    order_value = round(np.random.uniform(20, 400), 2)
    order_date = start_date + timedelta(days=np.random.randint(0, 365))

    past_orders = np.random.randint(1, 15)
    past_returns = np.random.binomial(past_orders, 0.22)
    chargeback_history = np.random.binomial(2, 0.05)
    discount_used = np.random.choice([0, 1], p=[0.65, 0.35])
    expedited_shipping = np.random.choice([0, 1], p=[0.78, 0.22])

    suspicious_fraud = (
        (state in ["FL", "NJ"])
        and (order_value > 280 or expedited_shipping == 1 or chargeback_history > 0)
    )

    fraud_score = (
        0.04
        + 0.18 * (chargeback_history > 0)
        + 0.08 * (state in ["FL", "NJ"])
        + 0.08 * (order_value > 300)
        + 0.05 * expedited_shipping
        + 0.12 * suspicious_fraud
        + np.random.normal(0, 0.04)
    )

    suspicious_return = (
        (category == "Apparel")
        and (past_returns >= 2 or (past_returns / max(past_orders, 1)) > 0.35)
    )

    return_abuse_score = (
        0.08
        + 0.22 * (past_returns / max(past_orders, 1))
        + 0.14 * (category == "Apparel")
        + 0.05 * (category == "Footwear")
        + 0.05 * discount_used
        + 0.16 * suspicious_return
        + np.random.normal(0, 0.05)
    )

    fraud_score = float(np.clip(fraud_score, 0, 1))
    return_abuse_score = float(np.clip(return_abuse_score, 0, 1))

    # Returns intentionally more common than fraud
    if fraud_score >= 0.42:
        risk_level = "High"
        risk_type = "Fraud"
    elif return_abuse_score >= 0.34:
        risk_level = "High"
        risk_type = "Returns"
    else:
        risk_level = "Low"
        risk_type = "None"

    unified_risk_score = round(max(fraud_score, return_abuse_score), 3)

    action_info = assign_actions(
        risk_level=risk_level,
        risk_type=risk_type,
        fraud_score=fraud_score,
        return_abuse_score=return_abuse_score,
        order_value=order_value,
        past_returns=past_returns,
        chargeback_history=chargeback_history,
    )

    risk_reason_summary = build_risk_reason_summary(
        risk_type=risk_type,
        category=category,
        state=state,
        order_value=order_value,
        past_returns=past_returns,
        past_orders=past_orders,
        chargeback_history=chargeback_history,
        expedited_shipping=expedited_shipping,
        discount_used=discount_used,
        fraud_score=fraud_score,
        return_abuse_score=return_abuse_score,
    )

    priority_tier = assign_priority_tier(
        risk_level=risk_level,
        risk_type=risk_type,
        fraud_score=fraud_score,
        return_abuse_score=return_abuse_score,
        order_value=order_value,
    )

    case_status, workflow_stage = assign_case_status_and_stage(
        risk_level=risk_level,
        recommended_action=action_info["recommended_action"],
    )

    data.append(
        {
            "order_id": order_id,
            "customer_id": customer_id,
            "order_date": order_date.date(),
            "product_category": category,
            "product_name": product_name,
            "payment_method": payment_method,
            "country": country,
            "state": state,
            "order_value": order_value,
            "past_orders": past_orders,
            "past_returns": past_returns,
            "chargeback_history": chargeback_history,
            "discount_used": discount_used,
            "expedited_shipping": expedited_shipping,
            "fraud_score": round(fraud_score, 3),
            "return_abuse_score": round(return_abuse_score, 3),
            "unified_risk_score": unified_risk_score,
            "risk_level": risk_level,
            "risk_type": risk_type,
            "recommended_action": action_info["recommended_action"],
            "ai_agent_next_step": action_info["ai_agent_next_step"],
            "action_owner": action_info["action_owner"],
            "automation_level": action_info["automation_level"],
            "risk_reason_summary": risk_reason_summary,
            "priority_tier": priority_tier,
            "case_status": case_status,
            "workflow_stage": workflow_stage,
        }
    )

df = pd.DataFrame(data)

os.makedirs("data", exist_ok=True)
df.to_csv("data/transactions.csv", index=False)

print("\nRisk Level Distribution:")
print(df["risk_level"].value_counts())

print("\nRisk Type Distribution:")
print(df["risk_type"].value_counts())

print("\nRisk Type %:")
print((df["risk_type"].value_counts(normalize=True) * 100).round(2))

print("\nPriority Tier Distribution:")
print(df["priority_tier"].value_counts())

print("\nCase Status Distribution:")
print(df["case_status"].value_counts())

print("\nWorkflow Stage Distribution:")
print(df["workflow_stage"].value_counts())

print("\nDataset created: data/transactions.csv")
print(df.head())
