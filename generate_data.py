"""Synthetic demo data for Aignita RiskIQ, scripted as a return-risk case study.

Storyline (FY2025 historical + January 2026 live queue):

  Jan–May 2025   Baseline. Return rate sits around 14%.
  Jun 2025       "Premium Jacket" launches. It runs small, so fit returns
                 ("Doesn't Fit Me → Runs Too Small") climb immediately.
  Jul–Sep 2025   A group of ~40 linked accounts in TX/FL starts exploiting the
                 summer promo: buy high-value Electronics/Footwear on discount,
                 return as "Changed My Mind". Ops approves most of them.
  Sep 15 2025    ACTION 1 – serial-returner policy: linked accounts are tagged and
                 their returns restricted (inspection + store credit only).
  Oct 1 2025     ACTION 2 – Premium Jacket size chart corrected and a
                 "size up" fit note added. Fit returns fall back to normal.
  Oct–Dec 2025   Recovery: return rate back near baseline; the cluster goes quiet.
  Jan 2026 live  An emerging pattern from Jan 19: brand-new accounts buying Boots
                 with the winter promo and expedited shipping (NY/NJ). The live
                 queue's flagged rate climbs, and the cases are open for action.
"""

import os
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

np.random.seed(42)

# ── Story calendar ───────────────────────────────────────────────────────────────
HIST_START = datetime(2025, 1, 1)
HIST_END = datetime(2025, 12, 31)
JACKET_LAUNCH = datetime(2025, 6, 1)
JACKET_MODEL_SIGNAL = datetime(2025, 8, 15)   # model starts weighting jacket fit returns
JACKET_FIX = datetime(2025, 10, 1)            # ACTION 2
RING_START = datetime(2025, 7, 1)
RING_POLICY = datetime(2025, 9, 15)           # ACTION 1
LIVE_START = datetime(2026, 1, 1)
LIVE_DAYS = 31
LIVE_TREND_START = datetime(2026, 1, 19)

HISTORICAL_N = 10000
LIVE_N = 2000

FLAG_THRESHOLD = 0.34

RING_CUSTOMERS = [f"C{9001 + i}" for i in range(40)]
RING_STATES = (["TX", "FL"], [0.6, 0.4])
RING_PRODUCTS = [
    ("Electronics", "Smart Watch"),
    ("Electronics", "Bluetooth Speaker"),
    ("Footwear", "Running Shoes"),
    ("Footwear", "Boots"),
]
# Orders per month from the linked-account cluster: ramps up, then collapses after the policy.
RING_MONTHLY_VOLUME = {7: 60, 8: 90, 9: 120, 10: 25, 11: 12, 12: 8}

# Premium Jacket orders per month: launch hype Jun–Sep, steady afterwards.
JACKET_MONTHLY_VOLUME = {6: 130, 7: 130, 8: 130, 9: 130, 10: 70, 11: 70, 12: 70}

product_categories = ["Apparel", "Electronics", "Beauty", "Home", "Footwear"]
category_weights = [0.38, 0.18, 0.14, 0.15, 0.15]
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

# Baseline probability that an order in each category comes back.
CATEGORY_RETURN_PROPENSITY = {
    "Apparel": 0.19,
    "Footwear": 0.16,
    "Electronics": 0.10,
    "Beauty": 0.07,
    "Home": 0.06,
}
# Category weight inside the risk model.
CATEGORY_SCORE_WEIGHT = {
    "Apparel": 0.10,
    "Footwear": 0.07,
    "Electronics": 0.04,
    "Beauty": 0.02,
    "Home": 0.00,
}

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
    "Doesn't Fit Me": ["Runs Too Small", "Runs Too Large", "Style/Fit Mismatch"],
    "Not What I Expected": ["Color Mismatch", "Material Mismatch", "Feature Mismatch", "Quality Expectation Gap"],
    "Changed My Mind": ["No Longer Needed", "Accidental Purchase", "Price Regret"],
    "Product Quality Issues": ["Damaged On Arrival", "Missing Components", "Performance Failure"],
    "Fulfillment Issues": ["Late Delivery", "Wrong Item", "Partial Shipment", "Return to Origin (RTO)"],
}

JACKET_FIT_COMMENTS = [
    "Ordered my usual size and the jacket is too tight across the shoulders.",
    "Sleeves are noticeably short — runs at least one size small.",
    "Size chart says M fits me, but it feels like an S.",
    "Love the jacket, but it runs small. Returning to reorder a size up.",
    "Too snug to zip over a sweater; sizing seems off.",
]
RING_COMMENTS = {
    "Changed My Mind": [
        "No longer needed.",
        "Found it cheaper elsewhere after the promo.",
        "Bought as a gift, recipient already had one.",
        "Changed my mind, item unopened.",
    ],
    "Not What I Expected": [
        "Features were not what I expected.",
        "Doesn't work with my setup.",
        "Not as described.",
    ],
}

STAGE_MAP = {
    "Restrict Return / Refund": [
        "Case Created",
        "Review Queue",
        "High Returner Tagged",
        "Awaiting Item Inspection",
        "Watchlist Monitoring",
        "Resolved",
    ],
    "Soft Review": ["Case Created", "Review Queue", "Resolved"],
    "Allow": ["Auto Approved", "Resolved"],
}

PENDING_STAGES = {"Awaiting Item Inspection"}


# ── Shared decision helpers ──────────────────────────────────────────────────────

def assign_priority_tier(risk_level: str, risk_type: str, return_risk_score: float, order_value: float):
    if risk_level == "Low":
        return "Low"
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
        return {
            "recommended_action": "Restrict Return / Refund",
            "ai_agent_next_step": "Disable instant refund, require item inspection, and recommend store credit or exchange",
            "action_owner": "Returns Ops",
            "automation_level": "Semi-Auto",
        }
    return {
        "recommended_action": "Soft Review",
        "ai_agent_next_step": "Review return history, product category, and discount usage before approving or restricting the return",
        "action_owner": "Returns Ops",
        "automation_level": "Assisted",
    }


def build_risk_reason_summary(
    risk_type: str,
    category: str,
    past_returns: int,
    past_orders: int,
    discount_used: int,
    return_risk_score: float,
    customer_lifetime_days: int,
    scenario_reason: str = "",
):
    reasons = []
    if scenario_reason:
        reasons.append(scenario_reason)

    if risk_type == "Returns":
        return_rate = past_returns / max(past_orders, 1)
        if past_returns >= 3:
            reasons.append(f"repeat returner with {past_returns} historical returns")
        if return_rate >= 0.35:
            reasons.append(f"high historical return rate ({return_rate:.0%})")
        if category == "Apparel":
            reasons.append("apparel return risk pattern")
        if category == "Footwear":
            reasons.append("footwear return pressure")
        if discount_used == 1:
            reasons.append("discount-driven purchase behavior")
        if customer_lifetime_days < 30:
            reasons.append("new customer account")
        if return_risk_score >= 0.65:
            reasons.append("elevated return risk score")

    if not reasons:
        reasons.append("standard low-risk profile")
    return " | ".join(reasons[:3])


def build_action_reason(
    recommended_action: str,
    category: str,
    past_returns: int,
    past_orders: int,
    return_risk_score: float,
    scenario: str,
    order_date: datetime,
) -> str:
    if recommended_action == "Restrict Return / Refund":
        if scenario == "ring" and order_date >= RING_POLICY:
            return "Return restricted — linked serial-returner account under Sep 15 policy"
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

    if recommended_action == "Soft Review":
        return f"Moderate return risk in {category} — review history before approving"

    return "No significant risk signals — order auto-approved"


def derive_case_status(workflow_stage: str) -> str:
    if workflow_stage == "Resolved":
        return "Closed"
    if workflow_stage in PENDING_STAGES:
        return "Pending"
    return "Open"


def derive_latest_action_taken(recommended_action: str, case_status: str, workflow_stage: str) -> str:
    if case_status == "Closed" or workflow_stage == "Resolved":
        if recommended_action == "Allow":
            return "Order approved"
        return "Case resolved"

    stage_actions = {
        "Case Created": "Case created",
        "Review Queue": "Soft review queued",
        "Awaiting Item Inspection": "Awaiting item inspection",
        "High Returner Tagged": "High returner tagged",
        "Escalated to Senior Review": "Escalated for senior review",
        "Watchlist Monitoring": "Case under watchlist monitoring",
    }
    if workflow_stage in stage_actions:
        return stage_actions[workflow_stage]
    if recommended_action == "Restrict Return / Refund":
        return "Return/refund restricted"
    if recommended_action == "Soft Review":
        return "Soft review started"
    return "No action yet"


# ── Order generation ─────────────────────────────────────────────────────────────

def random_date_in_month(year: int, month: int, start_day: int = 1, end_day: int | None = None) -> datetime:
    first = datetime(year, month, 1)
    next_month = datetime(year + (month == 12), month % 12 + 1, 1)
    last_day = (next_month - timedelta(days=1)).day
    end_day = min(end_day or last_day, last_day)
    return datetime(year, month, int(np.random.randint(start_day, end_day + 1)))


def baseline_order_dates(count: int, start: datetime, days: int, holiday_lift: bool) -> list[datetime]:
    day_list = [start + timedelta(days=d) for d in range(days)]
    weights = np.array([1.2 if holiday_lift and d.month in (11, 12) else 1.0 for d in day_list])
    weights = weights / weights.sum()
    picks = np.random.choice(len(day_list), size=count, p=weights)
    return [day_list[i] for i in picks]


def new_order(order_date: datetime, scenario: str, live: bool = False) -> dict:
    """Draw the order-level attributes for one order in a given scenario."""
    customer_id = f"C{np.random.randint(1000, 3000)}"
    payment_method = str(np.random.choice(payment_methods))
    country = str(np.random.choice(countries, p=[0.72, 0.10, 0.09, 0.09]))
    state = str(np.random.choice(states))
    past_orders = int(np.random.randint(1, 15))
    past_returns = int(np.random.binomial(past_orders, 0.18))
    discount_used = int(np.random.choice([0, 1], p=[0.65, 0.35]))
    expedited_shipping = int(np.random.choice([0, 1], p=[0.78, 0.22]))
    customer_lifetime_days = int(np.random.randint(30, 2001)) if np.random.random() > 0.08 else int(np.random.randint(0, 30))

    if scenario == "jacket":
        category, product_name = "Apparel", "Premium Jacket"
        order_value = round(float(np.random.uniform(220, 340)), 2)
    elif scenario == "ring":
        category, product_name = RING_PRODUCTS[np.random.choice(len(RING_PRODUCTS), p=[0.30, 0.25, 0.25, 0.20])]
        order_value = round(float(np.random.uniform(150, 400)), 2)
        customer_id = str(np.random.choice(RING_CUSTOMERS))
        state = str(np.random.choice(RING_STATES[0], p=RING_STATES[1]))
        country = "US"
        payment_method = str(np.random.choice(["PayPal", "Stripe"], p=[0.7, 0.3]))
        discount_used = 1
        account_created = datetime(2025, 6, 10) + timedelta(days=int(np.random.randint(0, 21)))
        customer_lifetime_days = max((order_date - account_created).days, 0)
        # The cluster's return history builds up month by month.
        months_active = max((order_date - RING_START).days / 30.0, 0)
        past_orders = int(np.random.randint(1, 4) + 2 * months_active)
        past_returns = int(np.random.binomial(past_orders, min(0.30 + 0.12 * months_active, 0.80)))
    elif scenario == "boots_promo":
        category, product_name = "Footwear", "Boots"
        order_value = round(float(np.random.uniform(160, 320)), 2)
        state = str(np.random.choice(["NY", "NJ", "CA", "WA"], p=[0.45, 0.25, 0.15, 0.15]))
        country = "US"
        discount_used = 1
        expedited_shipping = int(np.random.random() < 0.8)
        customer_lifetime_days = int(np.random.randint(0, 21))
        past_orders = int(np.random.randint(1, 3))
        past_returns = 0
    else:
        category = str(np.random.choice(product_categories, p=category_weights))
        products = products_by_category[category]
        # Premium Jacket orders are generated by the jacket scenario in the historical year.
        if category == "Apparel" and not live:
            products = [p for p in products if p != "Premium Jacket"]
        product_name = str(np.random.choice(products))
        order_value = round(float(np.random.uniform(20, 400)), 2)

    return {
        "customer_id": customer_id,
        "order_date": order_date,
        "product_category": category,
        "product_name": product_name,
        "payment_method": payment_method,
        "country": country,
        "state": state,
        "order_value": order_value,
        "past_orders": past_orders,
        "past_returns": min(past_returns, past_orders),
        "discount_used": discount_used,
        "expedited_shipping": expedited_shipping,
        "customer_lifetime_days": customer_lifetime_days,
        "scenario": scenario,
    }


def return_propensity(order: dict) -> float:
    scenario = order["scenario"]
    date = order["order_date"]
    if scenario == "jacket":
        return 0.50 if date < JACKET_FIX else 0.13
    if scenario == "ring":
        return 0.78
    if scenario == "boots_promo":
        return 0.70
    if order["product_name"] == "Premium Jacket":  # live orders, after the size-chart fix
        return 0.13
    history_rate = order["past_returns"] / max(order["past_orders"], 1)
    p = CATEGORY_RETURN_PROPENSITY[order["product_category"]] * (0.6 + 1.6 * history_rate)
    p += 0.03 * order["discount_used"]
    return float(np.clip(p, 0.02, 0.6))


def score_order(order: dict, will_return: bool) -> float:
    """Return risk model. The model learns the case-study signals as the year goes on."""
    date = order["order_date"]
    scenario = order["scenario"]
    history_rate = order["past_returns"] / max(order["past_orders"], 1)
    new_account = order["customer_lifetime_days"] < 30

    score = (
        0.05
        + 0.30 * history_rate
        + CATEGORY_SCORE_WEIGHT[order["product_category"]]
        + 0.05 * order["discount_used"]
        + 0.03 * order["expedited_shipping"]
        + 0.05 * new_account
        + 0.08 * (order["past_returns"] >= 3)
        + 0.12 * will_return  # behavioural signal the model partially observes
    )
    if scenario == "jacket" and JACKET_MODEL_SIGNAL <= date < JACKET_FIX:
        score += 0.08
    if scenario == "ring" and date >= RING_POLICY:
        score += 0.20  # linked-account signal added with the policy
    if new_account and order["discount_used"] and order["expedited_shipping"] and order["product_category"] == "Footwear":
        score += 0.20  # new-account promo velocity
    score += np.random.normal(0, 0.045)
    return float(np.clip(score, 0.01, 0.99))


def scenario_reason(order: dict) -> str:
    scenario = order["scenario"]
    date = order["order_date"]
    if scenario == "jacket" and date < JACKET_FIX:
        return "Premium Jacket fit-return trend (runs small)"
    if scenario == "ring":
        if date >= RING_POLICY:
            return "linked serial-returner cluster (summer promo)"
        return "promo-driven high-value return pattern"
    if scenario == "boots_promo":
        return "new account + promo + expedited Boots order (emerging pattern)"
    return ""


def pick_return_reason(order: dict) -> tuple[str, str, str]:
    scenario = order["scenario"]
    category = order["product_category"]

    if scenario == "jacket" and order["order_date"] < JACKET_FIX:
        reason = str(np.random.choice(
            ["Doesn't Fit Me", "Product Quality Issues", "Not What I Expected"], p=[0.80, 0.10, 0.10]
        ))
        if reason == "Doesn't Fit Me":
            sub = str(np.random.choice(["Runs Too Small", "Style/Fit Mismatch"], p=[0.85, 0.15]))
            return reason, sub, str(np.random.choice(JACKET_FIT_COMMENTS))
    elif scenario == "ring":
        reason = str(np.random.choice(["Changed My Mind", "Not What I Expected"], p=[0.6, 0.4]))
        sub = (
            str(np.random.choice(["No Longer Needed", "Price Regret"], p=[0.6, 0.4]))
            if reason == "Changed My Mind" else "Feature Mismatch"
        )
        return reason, sub, str(np.random.choice(RING_COMMENTS[reason]))
    elif scenario == "boots_promo":
        reason = str(np.random.choice(["Changed My Mind", "Doesn't Fit Me"], p=[0.65, 0.35]))
    elif order["past_returns"] >= 3:
        reason = str(np.random.choice(["Changed My Mind", "Not What I Expected"], p=[0.7, 0.3]))
    elif category in ["Apparel", "Footwear"]:
        reason = str(np.random.choice(
            ["Doesn't Fit Me", "Not What I Expected", "Changed My Mind", "Product Quality Issues"],
            p=[0.40, 0.24, 0.22, 0.14],
        ))
    else:
        reason = str(np.random.choice(
            ["Fulfillment Issues", "Changed My Mind", "Not What I Expected", "Product Quality Issues"],
            p=[0.25, 0.35, 0.20, 0.20],
        ))

    sub = str(np.random.choice(RETURN_SUB_REASONS[reason]))
    comment = str(np.random.choice(RETURN_REASON_DETAILS[reason][1]))
    return reason, sub, comment


def approval_probability(order: dict, recommended_action: str) -> float:
    scenario = order["scenario"]
    date = order["order_date"]
    if scenario == "jacket" and date < JACKET_FIX:
        # Genuine fit problems: inspection passes, so most are approved.
        return {"Allow": 0.97, "Soft Review": 0.88, "Restrict Return / Refund": 0.70}[recommended_action]
    if scenario == "ring":
        if date >= RING_POLICY:
            return 0.06
        # Before the policy, ops had no rule to deny these.
        return {"Allow": 0.97, "Soft Review": 0.82, "Restrict Return / Refund": 0.55}[recommended_action]
    return {"Allow": 0.95, "Soft Review": 0.55, "Restrict Return / Refund": 0.18}[recommended_action]


def resolution_note(order: dict, outcome_type: str, recommended_action: str) -> str:
    scenario = order["scenario"]
    date = order["order_date"]
    if outcome_type == "Auto Approved":
        return "Order had no material risk signals and was auto approved."
    if scenario == "jacket" and date < JACKET_FIX and outcome_type == "Return Approved":
        return "Fit issue verified at inspection — return approved; sizing feedback logged to merchandising."
    if scenario == "ring" and date >= RING_POLICY and outcome_type != "Return Approved":
        return "Return stopped under Sep 15 serial-returner policy: linked account, promo-driven high-value return."
    if scenario == "ring" and outcome_type == "Return Approved":
        return "Return approved after review — no policy in place to restrict repeat promo returns."
    if recommended_action == "Allow" and outcome_type == "Return Approved":
        return "Return auto-approved — low predicted risk at return intake."
    if outcome_type == "Return Approved":
        return "Returns operations approved the return after review."
    return "Returns operations reviewed history, reason code, and inspection requirements before denying the return."


def decide(order: dict, risk_score: float, force_policy: bool) -> dict:
    risk_level = "High" if risk_score >= FLAG_THRESHOLD else "Low"
    risk_type = "Returns" if risk_level == "High" else "None"
    priority_tier = assign_priority_tier(risk_level, risk_type, risk_score, order["order_value"])
    if force_policy:
        risk_level, risk_type, priority_tier = "High", "Returns", "High"
    actions = assign_actions(risk_level, risk_type, priority_tier)
    return {
        "risk_level": risk_level,
        "risk_type": risk_type,
        "priority_tier": priority_tier,
        **actions,
    }


def build_row(order: dict, live: bool) -> dict:
    will_return = bool(np.random.random() < return_propensity(order))
    risk_score = score_order(order, will_return)

    if not live and not will_return:
        # Historical: return cases open when a return is requested, so orders
        # that never came back stay below the review threshold.
        risk_score = min(risk_score, FLAG_THRESHOLD - 0.01 - abs(np.random.normal(0, 0.03)))
        risk_score = max(risk_score, 0.01)

    force_policy = (
        order["scenario"] == "ring" and order["order_date"] >= RING_POLICY and (will_return or live)
    )
    decision = decide(order, risk_score, force_policy)
    reason = scenario_reason(order) if decision["risk_type"] == "Returns" else ""

    row = {
        "customer_id": order["customer_id"],
        "order_date": order["order_date"].date(),
        "product_category": order["product_category"],
        "product_name": order["product_name"],
        "payment_method": order["payment_method"],
        "country": order["country"],
        "state": order["state"],
        "order_value": order["order_value"],
        "past_orders": order["past_orders"],
        "past_returns": order["past_returns"],
        "discount_used": order["discount_used"],
        "expedited_shipping": order["expedited_shipping"],
        "customer_lifetime_days": order["customer_lifetime_days"],
        "return_risk_score": round(risk_score, 3),
        "unified_risk_score": round(risk_score, 3),
        "risk_level": decision["risk_level"],
        "risk_type": decision["risk_type"],
        "recommended_action": decision["recommended_action"],
        "ai_agent_next_step": decision["ai_agent_next_step"],
        "action_owner": decision["action_owner"],
        "automation_level": decision["automation_level"],
        "action_reason": build_action_reason(
            decision["recommended_action"], order["product_category"], order["past_returns"],
            order["past_orders"], risk_score, order["scenario"], order["order_date"],
        ),
        "risk_reason_summary": build_risk_reason_summary(
            decision["risk_type"], order["product_category"], order["past_returns"], order["past_orders"],
            order["discount_used"], risk_score, order["customer_lifetime_days"], reason,
        ),
        "priority_tier": decision["priority_tier"],
    }

    if live:
        row.update(live_case_fields(decision["risk_level"], decision["recommended_action"], order))
    else:
        row.update(historical_outcome_fields(order, decision, will_return))
    return row


def historical_outcome_fields(order: dict, decision: dict, will_return: bool) -> dict:
    action = decision["recommended_action"]
    if will_return:
        approved = np.random.random() < approval_probability(order, action)
        if approved:
            outcome_type = "Return Approved"
        elif action == "Restrict Return / Refund":
            outcome_type = "Return Stopped"
        else:
            outcome_type = "Return Reviewed & Denied"
        reason, sub_reason, comment = pick_return_reason(order)
        return_fields = {
            "return_requested": 1,
            "return_completed": int(approved),
            "return_reason_code": reason,
            "return_sub_reason": sub_reason,
            "return_reason_text": RETURN_REASON_DETAILS[reason][0],
            "customer_return_comment": comment,
        }
    else:
        outcome_type = "Auto Approved"
        return_fields = {
            "return_requested": 0,
            "return_completed": 0,
            "return_reason_code": "None",
            "return_sub_reason": "None",
            "return_reason_text": "",
            "customer_return_comment": "",
        }

    is_ops_case = decision["risk_type"] == "Returns"
    if is_ops_case:
        reviewed_by = "Senior Review" if order["scenario"] == "ring" and order["order_date"] >= RING_POLICY else str(
            np.random.choice(["Returns Ops", "Senior Review"], p=[0.8, 0.2])
        )
    else:
        reviewed_by = "System"

    return {
        "case_status": "Closed",
        "workflow_stage": "Resolved",
        "latest_action_taken": derive_latest_action_taken(action, "Closed", "Resolved"),
        "outcome_type": outcome_type,
        **return_fields,
        "ops_resolution": outcome_type,
        "ops_resolution_note": resolution_note(order, outcome_type, action),
        "label_source": "Ops Review" if is_ops_case else "System Auto Label",
        "label_confidence": round(float(np.random.uniform(0.82, 0.98)), 2),
        "reviewed_by": reviewed_by,
        "reviewed_at": (order["order_date"] + timedelta(days=int(np.random.randint(3, 21)))).date(),
    }


def live_case_fields(risk_level: str, recommended_action: str, order: dict) -> dict:
    if risk_level == "Low" or recommended_action == "Allow":
        case_status = "Open"
        workflow_stage = "Prediction Only"
        latest_action = "Predicted allow; awaiting outcome"
    else:
        if order["scenario"] == "ring":
            workflow_stage = "High Returner Tagged"
        elif order["scenario"] == "boots_promo":
            # Emerging pattern: most of these cases are still waiting for triage.
            workflow_stage = str(np.random.choice(["Case Created", "Review Queue"], p=[0.7, 0.3]))
        else:
            stages = [s for s in STAGE_MAP.get(recommended_action, ["Case Created", "Review Queue"]) if s != "Resolved"]
            workflow_stage = str(np.random.choice(stages))
        case_status = derive_case_status(workflow_stage)
        latest_action = derive_latest_action_taken(recommended_action, case_status, workflow_stage)

    return {
        "case_status": case_status,
        "workflow_stage": workflow_stage,
        "latest_action_taken": latest_action,
        "outcome_type": "Pending",
        "return_requested": 0,
        "return_completed": 0,
        "return_reason_code": "None",
        "return_sub_reason": "None",
        "return_reason_text": "",
        "customer_return_comment": "",
        "ops_resolution": "Pending",
        "ops_resolution_note": "New January 2026 transaction; actual return outcome is not known yet.",
        "label_source": "Model Prediction",
        "label_confidence": round(float(np.random.uniform(0.45, 0.72)), 2),
        "reviewed_by": "",
        "reviewed_at": "",
        "ops_action_taken": "",
        "ops_action_by": "",
        "ops_action_at": "",
        "ops_action_note": "",
        "ops_action_history": "",
        "ops_override_reason": "",
        "analyst_confidence": 0.0,
        "ops_resolution_reason": "",
    }


def finalize(rows: list[dict], prefix: str, mode: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows).sort_values("order_date", kind="stable").reset_index(drop=True)
    frame.insert(0, "order_id", [f"{prefix}{i + 1:05d}" for i in range(len(frame))])
    frame["dataset_mode"] = mode
    return frame


def build_historical_dataset() -> pd.DataFrame:
    orders = []
    for month, volume in JACKET_MONTHLY_VOLUME.items():
        orders += [new_order(random_date_in_month(2025, month), "jacket") for _ in range(volume)]
    for month, volume in RING_MONTHLY_VOLUME.items():
        orders += [new_order(random_date_in_month(2025, month), "ring") for _ in range(volume)]

    baseline_n = HISTORICAL_N - len(orders)
    days = (HIST_END - HIST_START).days + 1
    orders += [new_order(d, "baseline") for d in baseline_order_dates(baseline_n, HIST_START, days, True)]

    return finalize([build_row(o, live=False) for o in orders], "O", "Historical")


def build_live_dataset() -> pd.DataFrame:
    orders = []
    trend_days = (LIVE_START + timedelta(days=LIVE_DAYS) - LIVE_TREND_START).days
    # Emerging pattern ramps from ~5 to ~30 orders a day.
    for day_index in range(trend_days):
        volume = int(round(5 + (25 * day_index / max(trend_days - 1, 1))))
        day = LIVE_TREND_START + timedelta(days=day_index)
        orders += [new_order(day, "boots_promo", live=True) for _ in range(volume)]
    # A handful of returning cluster accounts, already tagged by the policy.
    orders += [
        new_order(LIVE_START + timedelta(days=int(np.random.randint(0, LIVE_DAYS))), "ring", live=True)
        for _ in range(6)
    ]
    baseline_n = LIVE_N - len(orders)
    orders += [new_order(d, "baseline", live=True) for d in baseline_order_dates(baseline_n, LIVE_START, LIVE_DAYS, False)]

    return finalize([build_row(o, live=True) for o in orders], "L", "Live")


def seed_live_demo_feedback(live: pd.DataFrame) -> pd.DataFrame:
    seeded = live.copy()
    return_indexes = seeded[
        seeded["risk_type"].astype(str).eq("Returns")
        & seeded["risk_level"].astype(str).eq("High")
        & seeded["order_date"].astype(str).lt(LIVE_TREND_START.date().isoformat())
    ].head(8).index.tolist()
    allow_indexes = seeded[seeded["recommended_action"].astype(str).eq("Allow")].head(2).index.tolist()

    return_specs = [
        ("Return Approved", "Return Approved", "Doesn't Fit Me", "Returns Ops", "", 0.94),
        ("Return Approved", "Return Approved", "Product Quality Issues", "Returns Ops", "", 0.92),
        ("Return Approved", "Return Approved", "Not What I Expected", "Returns Ops", "", 0.90),
        ("Return Denied", "Return Reviewed & Denied", "Changed My Mind", "Returns Ops", "Repeat return behavior supported a stricter refund decision.", 0.91),
        ("Return Denied", "Return Stopped", "Not What I Expected", "Senior Review", "Item history and inspection result did not support the requested refund.", 0.89),
        ("Return Denied", "Return Stopped", "Fulfillment Issues", "Returns Ops", "Carrier scan and fulfillment record contradicted the claim.", 0.87),
        ("Return Denied", "Return Reviewed & Denied", "Changed My Mind", "Returns Ops", "Customer had three prior returns in the same category.", 0.88),
        ("Return Approved", "Return Approved", "Fulfillment Issues", "Returns Ops", "Delivery damage confirmed by carrier.", 0.93),
    ]
    allow_specs = [
        ("Order Approved", "Approved", "Standard approval", "Risk Ops", "Manual review confirmed a trusted customer profile.", 0.95),
        ("Order Approved", "Approved", "Standard approval", "Risk Ops", "", 0.93),
    ]

    specs = [
        *zip(return_indexes, return_specs[:len(return_indexes)]),
        *zip(allow_indexes, allow_specs[:len(allow_indexes)]),
    ]

    for idx, (resolution, outcome_type, reason, reviewer, override_reason, confidence) in specs:
        reviewed_at = "2026-02-01T16:00:00+00:00"
        action = "Approve Order" if resolution in {"Order Approved", "Return Approved"} else "Close Case"
        note = f"Seeded demo review: {resolution}."
        seeded.loc[idx, "case_status"] = "Closed"
        seeded.loc[idx, "workflow_stage"] = "Resolved"
        seeded.loc[idx, "latest_action_taken"] = note
        seeded.loc[idx, "outcome_type"] = outcome_type
        seeded.loc[idx, "ops_resolution"] = resolution
        seeded.loc[idx, "ops_resolution_reason"] = reason
        seeded.loc[idx, "ops_resolution_note"] = note
        seeded.loc[idx, "label_source"] = "Ops Review"
        seeded.loc[idx, "label_confidence"] = confidence
        seeded.loc[idx, "analyst_confidence"] = confidence
        seeded.loc[idx, "reviewed_by"] = reviewer
        seeded.loc[idx, "reviewed_at"] = reviewed_at
        seeded.loc[idx, "ops_action_taken"] = action
        seeded.loc[idx, "ops_action_by"] = reviewer
        seeded.loc[idx, "ops_action_at"] = reviewed_at
        seeded.loc[idx, "ops_action_note"] = note
        seeded.loc[idx, "ops_override_reason"] = override_reason
        seeded.loc[idx, "ops_action_history"] = json.dumps([{
            "ts": reviewed_at,
            "action": action,
            "actor": reviewer,
            "note": note,
            "case_status": "Closed",
            "workflow_stage": "Resolved",
            "latest_action_taken": note,
        }])

        if str(seeded.loc[idx, "risk_type"]) == "Returns":
            seeded.loc[idx, "return_requested"] = 1
            seeded.loc[idx, "return_completed"] = int(resolution == "Return Approved")
            seeded.loc[idx, "return_reason_code"] = reason
            seeded.loc[idx, "return_sub_reason"] = str(np.random.choice(RETURN_SUB_REASONS.get(reason, ["Other"])))

    return seeded


def print_story_check(historical: pd.DataFrame, live: pd.DataFrame):
    view = historical.copy()
    view["month"] = pd.to_datetime(view["order_date"]).dt.to_period("M")
    monthly = view.groupby("month").agg(
        orders=("order_id", "count"),
        return_requests=("return_requested", "sum"),
        approved=("return_completed", "sum"),
        denied=("outcome_type", lambda s: s.isin(["Return Stopped", "Return Reviewed & Denied"]).sum()),
        approved_loss=("order_value", lambda s: s[view.loc[s.index, "return_completed"].eq(1)].sum()),
    )
    monthly["return_rate_%"] = (monthly["return_requests"] / monthly["orders"] * 100).round(1)
    monthly["approved_rate_%"] = (monthly["approved"] / monthly["orders"] * 100).round(1)
    monthly["approved_loss"] = monthly["approved_loss"].round(0)

    print("\nHistorical monthly story check:")
    print(monthly.to_string())
    print("\nHistorical risk level:", historical["risk_level"].value_counts().to_dict())
    print("Historical outcomes:", historical["outcome_type"].value_counts().to_dict())

    live_view = live.copy()
    live_view["day"] = pd.to_datetime(live_view["order_date"]).dt.day
    daily = live_view.groupby("day")["risk_level"].apply(lambda s: round((s == "High").mean() * 100, 1))
    print("\nLive daily flagged rate (%):")
    print(daily.to_string())
    print("\nLive case status:", live["case_status"].value_counts().to_dict())


if __name__ == "__main__":
    historical_df = build_historical_dataset()
    live_df = seed_live_demo_feedback(build_live_dataset())

    os.makedirs("data", exist_ok=True)
    historical_df.to_csv("data/transactions.csv", index=False)
    live_df.to_csv("data/live_transactions.csv", index=False)
    live_df.to_csv("data/live_transactions_demo_baseline.csv", index=False)

    print_story_check(historical_df, live_df)
    print("\nDatasets created: data/transactions.csv, data/live_transactions.csv, data/live_transactions_demo_baseline.csv")
