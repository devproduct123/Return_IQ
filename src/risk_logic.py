import pandas as pd

from src.metrics import format_currency, safe_top_index


PRIORITY_ORDER = {"High": 0, "Moderate": 1, "Low": 2}


def build_case_timeline(case_row: pd.Series):
    current_stage = str(case_row["workflow_stage"])
    recommended_action = str(case_row["recommended_action"])
    case_status = str(case_row["case_status"])

    steps = [("Case Created", "done")]

    stage_map = {
        "Verification Sent": "current",
        "Awaiting Customer Response": "current",
        "Review Queue": "current",
        "High Returner Tagged": "current",
        "Awaiting Item Inspection": "current",
        "Escalated": "current",
        "Senior Review": "current",
        "Watchlist Monitoring": "current",
        "Auto Approved": "done",
    }

    steps.append((current_stage, stage_map.get(current_stage, "current")))

    if recommended_action == "Allow":
        action_label = "Auto Approval Completed"
    else:
        action_label = f"Action: {recommended_action}"

    steps.append((action_label, "done" if case_status ==
                 "Closed" else "current"))
    final_label = "Resolved" if case_status == "Closed" else "Pending Resolution"
    steps.append((final_label, "done" if case_status ==
                 "Closed" else "pending"))

    return steps


def get_open_high_risk_cases(data: pd.DataFrame) -> pd.DataFrame:
    return data[
        (data["risk_level"] == "High") & (
            data["case_status"] == "Open")
    ].copy()


def sort_cases_by_priority(cases: pd.DataFrame) -> pd.DataFrame:
    sorted_cases = cases.copy()
    if sorted_cases.empty:
        return sorted_cases

    sorted_cases["priority_sort"] = sorted_cases["priority_tier"].map(
        PRIORITY_ORDER)
    return sorted_cases.sort_values(
        ["priority_sort", "unified_risk_score", "order_value"],
        ascending=[True, False, False]
    )


def select_fallback_case(data: pd.DataFrame):
    fallback_cases = sort_cases_by_priority(get_open_high_risk_cases(data))
    if fallback_cases.empty:
        return None, False
    return fallback_cases.iloc[0], True


def answer_risk_question(question: str, data: pd.DataFrame) -> str:
    q = question.lower().strip()

    if data.empty:
        return "No data is available for the current filters."

    total_orders = len(data)
    flagged_orders = int((data["risk_level"] == "High").sum())
    revenue_at_risk = float(
        data.loc[data["risk_level"] == "High", "order_value"].sum())
    open_cases = int((data["case_status"] == "Open").sum())
    protected_revenue = float(
        data.loc[
            (data["risk_level"] == "High") & (data["case_status"] == "Closed"),
            "order_value"
        ].sum()
    )
    critical_open = data[
        (data["priority_tier"] == "High") & (data["case_status"] == "Open")
    ]

    top_category = safe_top_index(
        data.groupby("product_category")["return_risk_score"]
        .mean()
        .sort_values(ascending=False)
    )

    top_state = safe_top_index(
        data.groupby("state")["fraud_score"]
        .mean()
        .sort_values(ascending=False)
    )

    high_cases = int((data["priority_tier"] == "High").sum())

    top_action = safe_top_index(data["recommended_action"].value_counts())

    if "summary" in q or "summarize" in q or "overall" in q:
        return (
            f"In the current filtered view, {flagged_orders:,} out of {total_orders:,} orders are flagged high risk. "
            f"Revenue at risk is {format_currency(revenue_at_risk)}. "
            f"Protected revenue on closed flagged cases is {format_currency(protected_revenue)}. "
            f"There are {open_cases:,} open cases and {high_cases:,} high-priority cases. "
            f"Return risk is highest in {top_category}, fraud risk is strongest in {top_state}, "
            f"and the most common recommended action is {top_action}."
        )

    if "return" in q and ("category" in q or "product" in q or "driver" in q):
        top_return_products = (
            data.groupby("product_name")["return_risk_score"]
            .mean()
            .sort_values(ascending=False)
            .head(3)
            .index.tolist()
        )
        return (
            f"The highest return risk category is {top_category}. "
            f"Top products by return risk score are: {', '.join(top_return_products)}."
        )

    if "fraud" in q and ("state" in q or "where" in q or "location" in q):
        top_fraud_states = (
            data.groupby("state")["fraud_score"]
            .mean()
            .sort_values(ascending=False)
            .head(3)
            .index.tolist()
        )
        return f"The highest fraud-risk states in the current view are: {', '.join(top_fraud_states)}."

    if "revenue" in q and "risk" in q:
        return (
            f"Revenue at risk in the current filtered view is {format_currency(revenue_at_risk)}. "
            f"Protected revenue from closed flagged cases is {format_currency(protected_revenue)}."
        )

    if "priority" in q or "review first" in q or "what should i do first" in q:
        if critical_open.empty:
            return "There are no open high-priority cases in the current filtered view."
        return (
            f"There are {len(critical_open):,} open high-priority cases that should be reviewed first. "
            f"Focus on the highest-value orders with the highest unified risk scores."
        )

    if "action" in q or "recommended" in q:
        action_counts = data["recommended_action"].value_counts().head(3)
        action_text = ", ".join(
            [f"{k} ({v})" for k, v in action_counts.items()])
        return f"The top recommended actions in the current view are: {action_text}."

    if "open case" in q or "open cases" in q:
        return f"There are currently {open_cases:,} open cases in the filtered view."

    return (
        "I can help summarize current risk, identify top fraud states, top return risk categories or products, "
        "revenue at risk, protected revenue, open cases, and what the ops team should prioritize first."
    )


def answer_case_question(question: str, case_row: pd.Series | None) -> str:
    if case_row is None:
        return "No case is selected. Please select an order in Ops Queue first."

    q = question.lower().strip()

    order_id = case_row["order_id"]
    risk_type = case_row["risk_type_display"]
    priority = case_row["priority_tier"]
    case_status = case_row["case_status"]
    workflow_stage = case_row["workflow_stage"]
    recommended_action = case_row["recommended_action"]
    next_step = case_row["ai_agent_next_step"]
    risk_reason = case_row["risk_reason_summary"]
    fraud_score = float(case_row["fraud_score"])
    return_score = float(case_row["return_risk_score"])
    unified_score = float(case_row["unified_risk_score"])
    order_value = float(case_row["order_value"])
    past_returns = int(case_row["past_returns"])
    chargeback_history = int(case_row["chargeback_history"])
    payment_method = case_row["payment_method"]
    product_name = case_row["product_name"]
    product_category = case_row["product_category"]
    state = case_row["state"]
    country = case_row["country"]

    if "summary" in q or "summarize" in q or "case" in q:
        return (
            f"Order {order_id} is a {priority}-priority {risk_type} case with status {case_status}. "
            f"The order value is {format_currency(order_value)}, unified risk score is {unified_score:.3f}, "
            f"and the current workflow stage is {workflow_stage}. "
            f"Recommended action is {recommended_action}. Main reason flagged: {risk_reason}"
        )

    if "why" in q or "flagged" in q or "reason" in q:
        return (
            f"This case was flagged as {risk_type}. "
            f"Fraud score is {fraud_score:.3f}, return risk score is {return_score:.3f}, "
            f"and unified risk score is {unified_score:.3f}. "
            f"Primary reason: {risk_reason}"
        )

    if "next step" in q or "what should i do" in q or "action" in q or "review first" in q:
        return (
            f"The recommended action for order {order_id} is {recommended_action}. "
            f"The suggested next step is: {next_step}. "
            f"Current stage is {workflow_stage}, so the analyst should align the next review step with that stage."
        )

    if "fraud" in q:
        return (
            f"Fraud score for order {order_id} is {fraud_score:.3f}. "
            f"Payment method is {payment_method}, and the order location is {state}, {country}."
        )

    if "return" in q:
        return (
            f"Return risk score for order {order_id} is {return_score:.3f}. "
            f"The product is {product_name} in category {product_category}, and the customer has {past_returns} past returns."
        )

    if "chargeback" in q:
        return f"The customer has {chargeback_history} prior chargebacks associated with this case."

    if "customer" in q or "verification" in q:
        return (
            f"For this case, a reasonable analyst path is to use the recommended action '{recommended_action}' "
            f"and the AI next step '{next_step}' while requesting only the minimum required customer verification."
        )

    return (
        f"I can help explain why order {order_id} was flagged, summarize the case, recommend the next step, "
        f"or answer questions about fraud signals, return risk signals, verification, and escalation."
    )
