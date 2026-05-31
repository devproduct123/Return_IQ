import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

np.random.seed(42)
random.seed(42)

n = 10000
TARGET_HISTORICAL_ACTUAL_RETURNS = 1599

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

CHARGEBACK_REASONS = [
    "Transaction Fraud",
    "Account Takeover (ATO)",
    "Chargeback Abuse",
    "Item Not Received (INR)",
]

RETURN_REASON_DETAILS = {
    "Doesn't Fit Me": (
        "Customer reported fit or sizing issues",
        [
            "Customer said the item did not fit as expected.",
            "Customer ordered multiple sizes and returned the extras.",
            "Fit guidance did not match the received item.",
        ],
    ),
    "Not What I Expected": (
        "Customer expectation did not match the item received",
        [
            "Customer said product details were unclear.",
            "Customer said images did not match the item.",
            "Customer said the item was different than expected.",
        ],
    ),
    "Changed My Mind": (
        "Customer no longer needed the item or changed their mind",
        [
            "Customer said the item was no longer needed.",
            "Customer returned after comparing alternatives.",
            "Customer requested refund without product defect.",
        ],
    ),
    "Product Quality Issues": (
        "Customer reported product quality or defect issues",
        [
            "Customer said product quality missed expectations.",
            "Item defect reported during return intake.",
            "Customer reported quality issue after opening the package.",
        ],
    ),
    "Fulfillment Issues": (
        "Shipping, delivery, or damaged-in-transit issue",
        [
            "Customer said delivery arrived after event date.",
            "Customer no longer needed the item due to shipping delay.",
            "Package arrived with visible damage.",
        ],
    ),
}

RETURN_SUB_REASONS = {
    "Doesn't Fit Me": [
        "Runs Too Small",
        "Runs Too Large",
        "Style/Fit Mismatch",
    ],
    "Not What I Expected": [
        "Color Mismatch",
        "Material Mismatch",
        "Feature Mismatch",
        "Quality Expectation Gap",
    ],
    "Changed My Mind": [
        "No Longer Needed",
        "Accidental Purchase",
        "Price Regret",
    ],
    "Product Quality Issues": [
        "Damaged On Arrival",
        "Missing Components",
        "Performance Failure",
    ],
    "Fulfillment Issues": [
        "Late Delivery",
        "Wrong Item",
        "Partial Shipment",
        "Return to Origin (RTO)",
    ],
}

FRAUD_REASON_DETAILS = {
    "Transaction Fraud": (
        "Transaction shows payment or authorization fraud indicators",
        [
            "Customer stated they did not recognize the transaction.",
            "Payment behavior matched known fraud patterns.",
            "High-risk payment signal appeared during review.",
        ],
    ),
    "Account Takeover (ATO)": (
        "Account behavior suggests unauthorized account access",
        [
            "Account behavior changed from historical pattern.",
            "Verification response did not match account details.",
            "Login and order behavior suggested account takeover.",
        ],
    ),
    "Chargeback Abuse": (
        "Dispute pattern suggests post-purchase chargeback abuse",
        [
            "Customer disputed after receiving support response.",
            "Prior similar disputes were found in history.",
            "Dispute reason changed during review.",
        ],
    ),
    "Item Not Received (INR)": (
        "Customer disputed delivery receipt",
        [
            "Customer claimed package was not received.",
            "Delivery confirmation conflicts with customer report.",
            "Customer requested refund after delivery scan.",
        ],
    ),
}

FRAUD_SUB_REASONS = {
    "Transaction Fraud": [
        "Unauthorized payment",
        "High-risk payment signal",
        "Velocity anomaly",
        "Address mismatch",
        "Stolen card pattern",
    ],
    "Account Takeover (ATO)": [
        "Suspicious login",
        "Account detail changed",
        "Device mismatch",
        "Verification mismatch",
        "Unusual order behavior",
    ],
    "Chargeback Abuse": [
        "Friendly fraud",
        "Repeat chargeback user",
        "Post-delivery dispute",
        "Policy exploitation",
        "Dispute reason changed",
    ],
    "Item Not Received (INR)": [
        "Delivery confirmation dispute",
        "Lost package claim",
        "Porch theft claim",
        "Carrier scan conflict",
        "Refund after delivery scan",
    ],
}

start_date = datetime(2025, 1, 1)
data = []

STAGE_MAP = {
    "Hold Fulfillment": [
        "Case Created",
        "Review Queue",
        "Verification Sent",
        "Awaiting Customer Response",
        "Escalated to Senior Review",
        "Resolved",
    ],
    "Restrict Return / Refund": [
        "Case Created",
        "Review Queue",
        "High Returner Tagged",
        "Awaiting Item Inspection",
        "Watchlist Monitoring",
        "Resolved",
    ],
    "Soft Review": [
        "Case Created",
        "Review Queue",
        "Resolved",
    ],
    "Allow": [
        "Auto Approved",
        "Resolved",
    ],
}

PENDING_STAGES = {"Awaiting Customer Response", "Awaiting Item Inspection"}


def assign_priority_tier(
    risk_level: str,
    risk_type: str,
    fraud_score: float,
    return_risk_score: float,
    order_value: float,
):
    if risk_level == "Low":
        return "Low"

    if risk_type == "Fraud":
        if fraud_score >= 0.57 or (order_value >= 300 and fraud_score >= 0.46):
            return "High"
        return "Moderate"

    if risk_type == "Returns":
        if return_risk_score >= 0.53 or (order_value >= 250 and return_risk_score >= 0.44):
            return "High"
        return "Moderate"

    return "Low"


def assign_actions(risk_level: str, risk_type: str, priority_tier: str):
    if risk_level == "Low" or priority_tier == "Low":
        return {
            "recommended_action": "Allow",
            "ai_agent_next_step": "No action needed",
            "action_owner": "System",
            "automation_level": "Auto",
        }

    if priority_tier == "High":
        if risk_type == "Fraud":
            return {
                "recommended_action": "Hold Fulfillment",
                "ai_agent_next_step": "Pause shipment, send identity verification request, and request payment re-authentication",
                "action_owner": "Fraud Ops",
                "automation_level": "Semi-Auto",
            }
        else:  # Returns
            return {
                "recommended_action": "Restrict Return / Refund",
                "ai_agent_next_step": "Disable instant refund, require item inspection, and recommend store credit or exchange",
                "action_owner": "Returns Ops",
                "automation_level": "Semi-Auto",
            }

    # Moderate risk — soft review for both types
    if risk_type == "Fraud":
        return {
            "recommended_action": "Soft Review",
            "ai_agent_next_step": "Review account signals, payment history, and shipping pattern before approving or blocking the order",
            "action_owner": "Fraud Ops",
            "automation_level": "Assisted",
        }
    else:  # Returns
        return {
            "recommended_action": "Soft Review",
            "ai_agent_next_step": "Review return history, product category, and discount usage before approving or restricting the return",
            "action_owner": "Returns Ops",
            "automation_level": "Assisted",
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
    return_risk_score: float,
    device_risk: str = "Known Device",
    velocity_flag: int = 0,
    address_mismatch: int = 0,
):
    reasons = []

    if risk_type == "Fraud":
        if chargeback_history > 0:
            reasons.append(f"{chargeback_history} prior chargeback signal(s)")
        if device_risk == "High-Risk Device":
            reasons.append("high-risk device fingerprint")
        elif device_risk == "New Device" and fraud_score >= 0.55:
            reasons.append("new device on high-value order")
        if address_mismatch:
            reasons.append("billing/shipping address mismatch")
        if velocity_flag >= 3:
            reasons.append(f"velocity flag ({velocity_flag} orders/24h)")
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
            reasons.append("apparel return risk pattern")
        if category == "Footwear":
            reasons.append("footwear return pressure")
        if past_returns >= 3:
            reasons.append(f"repeat returner with {past_returns} historical returns")
        if return_rate >= 0.35:
            reasons.append(f"high historical return rate ({return_rate:.0%})")
        if discount_used == 1:
            reasons.append("discount-driven purchase behavior")
        if return_risk_score >= 0.65:
            reasons.append("elevated return risk score")

    if not reasons:
        reasons.append("standard low-risk profile")

    return " | ".join(reasons[:3])


def build_action_reason(
    recommended_action: str,
    risk_type: str,
    order_value: float,
    chargeback_history: int,
    state: str,
    category: str,
    past_returns: int,
    past_orders: int,
    fraud_score: float,
    return_risk_score: float,
    device_risk: str,
    velocity_flag: int,
    address_mismatch: int,
) -> str:
    if recommended_action == "Hold Fulfillment":
        drivers = []
        if chargeback_history > 0:
            drivers.append(f"{chargeback_history} prior chargeback(s)")
        if device_risk == "High-Risk Device":
            drivers.append("high-risk device signal")
        elif device_risk == "New Device" and fraud_score >= 0.55:
            drivers.append("new device on high-value order")
        if address_mismatch:
            drivers.append("billing/shipping address mismatch")
        if velocity_flag >= 3:
            drivers.append(f"velocity flag ({velocity_flag} orders/24h)")
        if state in ["FL", "NJ"]:
            drivers.append(f"flagged geography ({state})")
        if order_value >= 300:
            drivers.append(f"high-value order (${order_value:.0f})")
        if not drivers:
            drivers.append(f"elevated fraud score ({fraud_score:.2f})")
        return "Fulfillment held — " + ", ".join(drivers[:3])

    elif recommended_action == "Restrict Return / Refund":
        drivers = []
        return_rate = past_returns / max(past_orders, 1)
        if past_returns >= 3:
            drivers.append(f"repeat returner ({past_returns} prior returns)")
        if return_rate >= 0.35:
            drivers.append(f"high return rate ({return_rate:.0%})")
        if category in ["Apparel", "Footwear"]:
            drivers.append(f"high-return category ({category})")
        if not drivers:
            drivers.append(f"elevated return score ({return_risk_score:.2f})")
        return "Return restricted — " + ", ".join(drivers[:3])

    elif recommended_action == "Soft Review":
        if risk_type == "Fraud":
            return f"Moderate fraud signals — analyst review before fulfilling ${order_value:.0f} order"
        else:
            return f"Moderate return risk in {category} — review history before approving"

    else:  # Allow
        return "No significant risk signals — order auto-approved"


def derive_outcome_type(
    case_status: str,
    recommended_action: str,
    risk_type: str,
    rng_val: float,
) -> str:
    if case_status != "Closed":
        return "Pending"

    if recommended_action == "Allow" or risk_type == "None":
        return "Auto Approved"

    if risk_type == "Fraud":
        if recommended_action == "Hold Fulfillment":
            return "Chargeback Prevented" if rng_val < 0.72 else "Fraud Confirmed & Blocked"
        else:  # Soft Review
            return "Verified & Released" if rng_val < 0.58 else "Fraud Confirmed & Blocked"

    if risk_type == "Returns":
        if recommended_action == "Restrict Return / Refund":
            return "Return Stopped" if rng_val < 0.82 else "Return Approved"
        else:  # Soft Review
            return "Return Reviewed & Denied" if rng_val < 0.48 else "Return Approved"

    return "Resolved"


def derive_resolution_note(outcome_type: str, risk_type: str) -> str:
    if outcome_type == "Pending":
        return "Case is still in progress; final analyst label is not available yet."
    if outcome_type == "Auto Approved":
        return "Order had no material risk signals and was auto approved."
    if risk_type == "Fraud":
        return "Fraud operations reviewed payment, identity, and dispute signals before resolution."
    if risk_type == "Returns":
        return "Returns operations reviewed history, reason code, and inspection requirements before resolution."
    return "Operations completed the case review."


def assign_return_intelligence(
    risk_type: str,
    outcome_type: str,
    category: str,
    past_returns: int,
) -> dict:
    if risk_type != "Returns" and outcome_type not in {"Return Approved", "Return Stopped", "Return Reviewed & Denied", "Return Restricted"}:
        return {
            "return_requested": 0,
            "return_completed": 0,
            "return_reason_code": "None",
            "return_sub_reason": "None",
            "return_reason_text": "",
            "customer_return_comment": "",
        }

    if outcome_type == "Pending":
        requested = 1
        completed = 0
    else:
        requested = 1
        completed = int(outcome_type == "Return Approved")

    if past_returns >= 3:
        reason_code = "Changed My Mind"
    elif category in ["Apparel", "Footwear"]:
        reason_code = np.random.choice(["Doesn't Fit Me", "Not What I Expected", "Changed My Mind", "Product Quality Issues"], p=[0.48, 0.24, 0.18, 0.10])
    else:
        reason_code = np.random.choice(["Fulfillment Issues", "Changed My Mind", "Not What I Expected", "Product Quality Issues"], p=[0.25, 0.35, 0.20, 0.20])

    reason_text, comments = RETURN_REASON_DETAILS[reason_code]
    sub_reason = str(np.random.choice(RETURN_SUB_REASONS[reason_code]))
    return {
        "return_requested": requested,
        "return_completed": completed,
        "return_reason_code": reason_code,
        "return_sub_reason": sub_reason,
        "return_reason_text": reason_text,
        "customer_return_comment": str(np.random.choice(comments)),
    }


def assign_fraud_intelligence(
    risk_type: str,
    outcome_type: str,
    chargeback_reason: str,
    address_mismatch: int,
) -> dict:
    if risk_type != "Fraud":
        return {
            "fraud_label": "Cleared" if outcome_type == "Auto Approved" else "Unknown",
            "fraud_reason_code": "None",
            "fraud_sub_reason": "None",
            "fraud_reason_text": "",
            "customer_dispute_comment": "",
        }

    if outcome_type == "Pending":
        fraud_label = "Suspected"
    elif outcome_type == "Verified & Released":
        fraud_label = "False Positive"
    else:
        fraud_label = "Confirmed"

    if chargeback_reason != "None":
        reason_code = chargeback_reason
    elif address_mismatch:
        reason_code = "Account Takeover (ATO)"
    else:
        reason_code = np.random.choice(
            ["Transaction Fraud", "Account Takeover (ATO)", "Chargeback Abuse", "Item Not Received (INR)"],
            p=[0.38, 0.27, 0.22, 0.13],
        )

    reason_text, comments = FRAUD_REASON_DETAILS[reason_code]
    sub_reason = str(np.random.choice(FRAUD_SUB_REASONS[reason_code]))
    return {
        "fraud_label": fraud_label,
        "fraud_reason_code": reason_code,
        "fraud_sub_reason": sub_reason,
        "fraud_reason_text": reason_text,
        "customer_dispute_comment": str(np.random.choice(comments)) if fraud_label != "Suspected" else "",
    }


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
        ["Case Created", "Review Queue", "Resolved"]
    )

    default_weights = {
        3: np.array([0.35, 0.28, 0.37]),
        6: np.array([0.30, 0.24, 0.18, 0.14, 0.08, 0.06]),
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
    if workflow_stage == "Verification Sent":
        return "Customer verification sent"
    if workflow_stage == "Awaiting Customer Response":
        return "Awaiting customer response"
    if workflow_stage == "Awaiting Item Inspection":
        return "Awaiting item inspection"
    if workflow_stage == "High Returner Tagged":
        return "High returner tagged"
    if workflow_stage == "Escalated to Senior Review":
        return "Escalated for senior review"
    if workflow_stage == "Watchlist Monitoring":
        return "Case under watchlist monitoring"

    if recommended_action == "Hold Fulfillment":
        return "Fulfillment held"
    if recommended_action == "Restrict Return / Refund":
        return "Return/refund restricted"
    if recommended_action == "Soft Review":
        return "Soft review started"

    return "No action yet"


for i in range(n):
    order_id = f"O{i+1:05d}"
    customer_id = f"C{np.random.randint(1000, 3000)}"

    category = np.random.choice(product_categories, p=[0.38, 0.18, 0.14, 0.15, 0.15])
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

    # ── New behavioral signals ───────────────────────────────────────────────────
    device_risk = np.random.choice(
        ["Known Device", "New Device", "High-Risk Device"],
        p=[0.70, 0.22, 0.08]
    )
    velocity_flag = int(np.random.choice([0, 1, 2, 3, 4, 5], p=[0.60, 0.18, 0.10, 0.06, 0.04, 0.02]))
    customer_lifetime_days = int(np.random.randint(0, 2001))
    address_mismatch = int(np.random.choice([0, 1], p=[0.88, 0.12]))

    chargeback_reason = (
        np.random.choice(CHARGEBACK_REASONS)
        if chargeback_history > 0 else "None"
    )
    chargebacks_won = int(np.random.binomial(chargeback_history, 0.35))

    # ── Fraud score ──────────────────────────────────────────────────────────────
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
        + 0.09 * (device_risk == "High-Risk Device")
        + 0.04 * (device_risk == "New Device")
        + 0.05 * min(velocity_flag / 5.0, 1.0)
        + 0.06 * address_mismatch
        + np.random.normal(0, 0.04)
    )

    # ── Return abuse score ───────────────────────────────────────────────────────
    suspicious_return = (
        (category == "Apparel")
        and (past_returns >= 2 or (past_returns / max(past_orders, 1)) > 0.35)
    )

    return_risk_score = (
        0.08
        + 0.22 * (past_returns / max(past_orders, 1))
        + 0.14 * (category == "Apparel")
        + 0.05 * (category == "Footwear")
        + 0.05 * discount_used
        + 0.16 * suspicious_return
        + np.random.normal(0, 0.05)
    )

    fraud_score = float(np.clip(fraud_score, 0, 1))
    return_risk_score = float(np.clip(return_risk_score, 0, 1))

    # ── Risk classification ──────────────────────────────────────────────────────
    if fraud_score >= 0.42:
        risk_level = "High"
        risk_type = "Fraud"
    elif return_risk_score >= 0.34:
        risk_level = "High"
        risk_type = "Returns"
    else:
        risk_level = "Low"
        risk_type = "None"

    unified_risk_score = round(max(fraud_score, return_risk_score), 3)

    # ── Priority and actions ─────────────────────────────────────────────────────
    priority_tier = assign_priority_tier(
        risk_level=risk_level,
        risk_type=risk_type,
        fraud_score=fraud_score,
        return_risk_score=return_risk_score,
        order_value=order_value,
    )

    action_info = assign_actions(
        risk_level=risk_level,
        risk_type=risk_type,
        priority_tier=priority_tier,
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
        return_risk_score=return_risk_score,
        device_risk=device_risk,
        velocity_flag=velocity_flag,
        address_mismatch=address_mismatch,
    )

    action_reason = build_action_reason(
        recommended_action=action_info["recommended_action"],
        risk_type=risk_type,
        order_value=order_value,
        chargeback_history=chargeback_history,
        state=state,
        category=category,
        past_returns=past_returns,
        past_orders=past_orders,
        fraud_score=fraud_score,
        return_risk_score=return_risk_score,
        device_risk=device_risk,
        velocity_flag=velocity_flag,
        address_mismatch=address_mismatch,
    )

    case_status, workflow_stage = assign_case_status_and_stage(
        risk_level=risk_level,
        recommended_action=action_info["recommended_action"],
    )
    latest_action_taken = derive_latest_action_taken(
        action_info["recommended_action"],
        case_status,
        workflow_stage,
    )

    outcome_rng = float(np.random.random())
    outcome_type = derive_outcome_type(
        case_status=case_status,
        recommended_action=action_info["recommended_action"],
        risk_type=risk_type,
        rng_val=outcome_rng,
    )
    return_intel = assign_return_intelligence(
        risk_type=risk_type,
        outcome_type=outcome_type,
        category=category,
        past_returns=past_returns,
    )
    fraud_intel = assign_fraud_intelligence(
        risk_type=risk_type,
        outcome_type=outcome_type,
        chargeback_reason=chargeback_reason,
        address_mismatch=address_mismatch,
    )
    ops_resolution = "Pending" if outcome_type == "Pending" else outcome_type
    ops_resolution_note = derive_resolution_note(outcome_type, risk_type)
    label_source = (
        "Ops Review" if case_status == "Closed" and risk_type != "None"
        else "System Auto Label" if case_status == "Closed"
        else "Model Prediction"
    )
    label_confidence = (
        round(float(np.random.uniform(0.82, 0.98)), 2) if case_status == "Closed"
        else round(float(np.random.uniform(0.45, 0.72)), 2)
    )
    reviewed_by = (
        np.random.choice(["Fraud Ops", "Returns Ops", "Senior Review", "System"])
        if case_status == "Closed" else ""
    )
    reviewed_at = (
        (order_date + timedelta(days=int(np.random.randint(1, 14)))).date()
        if case_status == "Closed" else ""
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
            "device_risk": device_risk,
            "velocity_flag": velocity_flag,
            "customer_lifetime_days": customer_lifetime_days,
            "address_mismatch": address_mismatch,
            "chargeback_reason": chargeback_reason,
            "chargebacks_won": chargebacks_won,
            "fraud_score": round(fraud_score, 3),
            "return_risk_score": round(return_risk_score, 3),
            "unified_risk_score": unified_risk_score,
            "risk_level": risk_level,
            "risk_type": risk_type,
            "recommended_action": action_info["recommended_action"],
            "ai_agent_next_step": action_info["ai_agent_next_step"],
            "action_owner": action_info["action_owner"],
            "automation_level": action_info["automation_level"],
            "action_reason": action_reason,
            "risk_reason_summary": risk_reason_summary,
            "priority_tier": priority_tier,
            "case_status": case_status,
            "workflow_stage": workflow_stage,
            "latest_action_taken": latest_action_taken,
            "outcome_type": outcome_type,
            **return_intel,
            **fraud_intel,
            "ops_resolution": ops_resolution,
            "ops_resolution_note": ops_resolution_note,
            "label_source": label_source,
            "label_confidence": label_confidence,
            "reviewed_by": reviewed_by,
            "reviewed_at": reviewed_at,
        }
    )

df = pd.DataFrame(data)


def finalize_historical_dataset(source_df: pd.DataFrame) -> pd.DataFrame:
    historical = source_df.copy()
    for idx, row in historical.iterrows():
        risk_type = str(row["risk_type"])
        recommended_action = str(row["recommended_action"])
        outcome_type = str(row["outcome_type"])

        if outcome_type == "Pending":
            outcome_type = derive_outcome_type(
                case_status="Closed",
                recommended_action=recommended_action,
                risk_type=risk_type,
                rng_val=float(np.random.random()),
            )

        return_intel = assign_return_intelligence(
            risk_type=risk_type,
            outcome_type=outcome_type,
            category=str(row["product_category"]),
            past_returns=int(row["past_returns"]),
        )
        fraud_intel = assign_fraud_intelligence(
            risk_type=risk_type,
            outcome_type=outcome_type,
            chargeback_reason=str(row["chargeback_reason"]),
            address_mismatch=int(row["address_mismatch"]),
        )

        historical.loc[idx, "case_status"] = "Closed"
        historical.loc[idx, "workflow_stage"] = "Resolved"
        historical.loc[idx, "latest_action_taken"] = derive_latest_action_taken(
            recommended_action,
            "Closed",
            "Resolved",
        )
        historical.loc[idx, "outcome_type"] = outcome_type
        for column, value in {**return_intel, **fraud_intel}.items():
            historical.loc[idx, column] = value
        historical.loc[idx, "ops_resolution"] = outcome_type
        historical.loc[idx, "ops_resolution_note"] = derive_resolution_note(outcome_type, risk_type)
        historical.loc[idx, "label_source"] = "Ops Review" if risk_type != "None" else "System Auto Label"
        historical.loc[idx, "label_confidence"] = round(float(np.random.uniform(0.82, 0.98)), 2)
        historical.loc[idx, "reviewed_by"] = (
            np.random.choice(["Fraud Ops", "Returns Ops", "Senior Review"])
            if risk_type != "None" else "System"
        )
        historical.loc[idx, "reviewed_at"] = (
            pd.to_datetime(row["order_date"]) + timedelta(days=int(np.random.randint(1, 14)))
        ).date()

    current_returns = int(historical["return_completed"].astype(int).sum())
    if current_returns < TARGET_HISTORICAL_ACTUAL_RETURNS:
        needed = TARGET_HISTORICAL_ACTUAL_RETURNS - current_returns
        approval_candidates = (
            historical[
                (historical["risk_type"].astype(str).eq("Returns"))
                & (historical["return_completed"].astype(int).eq(0))
            ]
            .sort_values(["return_risk_score", "order_value"], ascending=[True, True])
            .head(needed)
            .index
        )
        historical.loc[approval_candidates, "outcome_type"] = "Return Approved"
        historical.loc[approval_candidates, "return_requested"] = 1
        historical.loc[approval_candidates, "return_completed"] = 1
        historical.loc[approval_candidates, "ops_resolution"] = "Return Approved"
        historical.loc[approval_candidates, "ops_resolution_note"] = (
            "Returns operations approved the completed return after review."
        )
    elif current_returns > TARGET_HISTORICAL_ACTUAL_RETURNS:
        excess = current_returns - TARGET_HISTORICAL_ACTUAL_RETURNS
        denial_candidates = (
            historical[
                (historical["risk_type"].astype(str).eq("Returns"))
                & (historical["return_completed"].astype(int).eq(1))
            ]
            .sort_values(["return_risk_score", "order_value"], ascending=[False, False])
            .head(excess)
            .index
        )
        historical.loc[denial_candidates, "outcome_type"] = "Return Reviewed & Denied"
        historical.loc[denial_candidates, "return_requested"] = 1
        historical.loc[denial_candidates, "return_completed"] = 0
        historical.loc[denial_candidates, "ops_resolution"] = "Return Reviewed & Denied"
        historical.loc[denial_candidates, "ops_resolution_note"] = (
            "Returns operations denied the return after review."
        )

    historical["dataset_mode"] = "Historical"
    return historical


def build_live_dataset(source_df: pd.DataFrame, live_n: int = 2000) -> pd.DataFrame:
    live = source_df.sample(n=live_n, random_state=2026).copy().reset_index(drop=True)
    live["order_id"] = [f"L{i + 1:05d}" for i in range(len(live))]
    live["order_date"] = [
        (datetime(2026, 1, 1) + timedelta(days=int(day))).date()
        for day in np.random.randint(0, 31, size=len(live))
    ]

    for idx, row in live.iterrows():
        risk_level = str(row["risk_level"])
        recommended_action = str(row["recommended_action"])
        risk_type = str(row["risk_type"])

        if risk_level == "Low" or recommended_action == "Allow":
            case_status = "Open"
            workflow_stage = "Prediction Only"
            latest_action = "Predicted allow; awaiting outcome"
        else:
            stages = [stage for stage in STAGE_MAP.get(recommended_action, ["Case Created", "Review Queue"]) if stage != "Resolved"]
            workflow_stage = str(np.random.choice(stages))
            case_status = derive_case_status(workflow_stage)
            latest_action = derive_latest_action_taken(recommended_action, case_status, workflow_stage)

        live.loc[idx, "case_status"] = case_status
        live.loc[idx, "workflow_stage"] = workflow_stage
        live.loc[idx, "latest_action_taken"] = latest_action
        live.loc[idx, "outcome_type"] = "Pending"
        live.loc[idx, "return_requested"] = 0
        live.loc[idx, "return_completed"] = 0
        live.loc[idx, "return_reason_code"] = "None"
        live.loc[idx, "return_sub_reason"] = "None"
        live.loc[idx, "return_reason_text"] = ""
        live.loc[idx, "customer_return_comment"] = ""
        live.loc[idx, "fraud_label"] = "Suspected" if risk_type == "Fraud" else "Unknown"
        live.loc[idx, "fraud_reason_code"] = "None"
        live.loc[idx, "fraud_sub_reason"] = "None"
        live.loc[idx, "fraud_reason_text"] = ""
        live.loc[idx, "customer_dispute_comment"] = ""
        live.loc[idx, "ops_resolution"] = "Pending"
        live.loc[idx, "ops_resolution_note"] = "New January 2026 transaction; actual return/fraud outcome is not known yet."
        live.loc[idx, "label_source"] = "Model Prediction"
        live.loc[idx, "label_confidence"] = round(float(np.random.uniform(0.45, 0.72)), 2)
        live.loc[idx, "reviewed_by"] = ""
        live.loc[idx, "reviewed_at"] = ""

    live["dataset_mode"] = "Live"
    return live


historical_df = finalize_historical_dataset(df)
live_df = build_live_dataset(df)

os.makedirs("data", exist_ok=True)
historical_df.to_csv("data/transactions.csv", index=False)
live_df.to_csv("data/live_transactions.csv", index=False)
df = historical_df

print("\nRisk Level Distribution:")
print(df["risk_level"].value_counts())

print("\nRisk Type Distribution:")
print(df["risk_type"].value_counts())

print("\nPriority Tier Distribution:")
print(df["priority_tier"].value_counts())

print("\nRecommended Action Distribution:")
print(df["recommended_action"].value_counts())

print("\nCase Status Distribution:")
print(df["case_status"].value_counts())

print("\nWorkflow Stage Distribution:")
print(df["workflow_stage"].value_counts())

print("\nOutcome Type Distribution:")
print(df["outcome_type"].value_counts())

print("\nReturn Reason Distribution:")
print(df["return_reason_code"].value_counts())

print("\nFraud Label Distribution:")
print(df["fraud_label"].value_counts())

print("\nDevice Risk Distribution:")
print(df["device_risk"].value_counts())

print("\nHistorical Date Range:")
print(f'{historical_df["order_date"].min()} to {historical_df["order_date"].max()}')

print("\nLive Date Range:")
print(f'{live_df["order_date"].min()} to {live_df["order_date"].max()}')

print("\nLive Case Status Distribution:")
print(live_df["case_status"].value_counts())

print("\nDatasets created: data/transactions.csv and data/live_transactions.csv")
print(df.head())
