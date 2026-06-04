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


def _simplify_actual_outcome(row: pd.Series) -> str:
    outcome_type = str(row.get("outcome_type", ""))
    risk_type = str(row.get("risk_type", ""))
    fraud_label = str(row.get("fraud_label", ""))

    if risk_type == "Returns" or outcome_type.startswith("Return"):
        return "Return Approved" if outcome_type == "Return Approved" else "Return Denied"

    if risk_type == "Fraud" or fraud_label in {"Confirmed", "False Positive"}:
        if outcome_type == "Verified & Released" or fraud_label == "False Positive":
            return "Fraud Approved"
        return "Fraud Denied"

    return "Other"


def answer_historical_question(question: str, data: pd.DataFrame) -> str:
    q = question.lower().strip()

    if data.empty:
        return "No historical outcome data is available for the current filters."

    view = data.copy()
    view["actual_outcome"] = view.apply(_simplify_actual_outcome, axis=1)
    total_orders = len(view)
    actual_returns = int(view["actual_outcome"].eq("Return Approved").sum())
    denied_returns = int(view["actual_outcome"].eq("Return Denied").sum())
    fraud_approved = int(view["actual_outcome"].eq("Fraud Approved").sum())
    fraud_denied = int(view["actual_outcome"].eq("Fraud Denied").sum())
    total_loss = float(
        view.loc[
            view["actual_outcome"].isin(["Return Approved", "Fraud Approved"]),
            "order_value",
        ].sum()
    )
    avoided_loss = float(
        view.loc[
            view["actual_outcome"].isin(["Return Denied", "Fraud Denied"]),
            "order_value",
        ].sum()
    )

    top_return_category = safe_top_index(
        view.loc[view["actual_outcome"].eq("Return Approved")]
        .groupby("product_category")["order_value"]
        .sum()
        .sort_values(ascending=False),
        default="None",
    )
    top_return_product = safe_top_index(
        view.loc[view["actual_outcome"].eq("Return Approved")]
        .groupby("product_name")["order_value"]
        .sum()
        .sort_values(ascending=False),
        default="None",
    )
    top_fraud_state = safe_top_index(
        view.loc[view["actual_outcome"].eq("Fraud Approved")]
        .groupby("state")["order_value"]
        .sum()
        .sort_values(ascending=False),
        default="None",
    )
    top_fraud_payment = safe_top_index(
        view.loc[view["actual_outcome"].eq("Fraud Approved")]
        .groupby("payment_method")["order_value"]
        .sum()
        .sort_values(ascending=False),
        default="None",
    )

    if "summary" in q or "summarize" in q or "overall" in q:
        return (
            f"In the historical view, {total_orders:,} completed orders produced {actual_returns:,} approved returns, "
            f"{denied_returns:,} denied returns, {fraud_approved:,} fraud-approved/released cases, and "
            f"{fraud_denied:,} fraud-denied/blocked cases. Total loss was {format_currency(total_loss)}, "
            f"while avoided exposure was {format_currency(avoided_loss)}."
        )

    if "return" in q and ("category" in q or "product" in q or "driver" in q):
        return (
            f"Approved return loss is highest in category {top_return_category}. "
            f"The top product by approved return loss is {top_return_product}."
        )

    if "fraud" in q and ("state" in q or "where" in q or "location" in q):
        return f"Fraud-approved loss is highest in {top_fraud_state}."

    if "payment" in q:
        return f"The payment method with the highest fraud-approved loss is {top_fraud_payment}."

    if "loss" in q or "dollar" in q or "revenue" in q:
        return (
            f"Total historical loss from approved returns and fraud-approved orders is {format_currency(total_loss)}. "
            f"Avoided exposure from denied returns and denied fraud is {format_currency(avoided_loss)}."
        )

    return (
        "I can summarize actual historical outcomes, approved return loss, fraud-approved loss, avoided exposure, "
        "top return-loss categories/products, top fraud-loss states/payment methods, and reason-code patterns."
    )


def answer_case_question(question: str, case_row: pd.Series | None) -> str:
    if case_row is None:
        return "No case is selected. Please select an order in Review Queue first."

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
    return_reason = case_row.get("return_reason_code", "None")
    return_comment = str(case_row.get("customer_return_comment", "")).strip()
    fraud_label = case_row.get("fraud_label", "Unknown")
    fraud_reason = case_row.get("fraud_reason_code", "None")
    dispute_comment = str(case_row.get("customer_dispute_comment", "")).strip()

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
        comment_text = f" Customer dispute comment: {dispute_comment}" if dispute_comment else ""
        return (
            f"Fraud score for order {order_id} is {fraud_score:.3f}. "
            f"Payment method is {payment_method}, and the order location is {state}, {country}. "
            f"Current fraud label is {fraud_label}; reason code is {fraud_reason}.{comment_text}"
        )

    if "return" in q:
        comment_text = f" Customer return comment: {return_comment}" if return_comment else ""
        return (
            f"Return risk score for order {order_id} is {return_score:.3f}. "
            f"The product is {product_name} in category {product_category}, and the customer has {past_returns} past returns. "
            f"Return reason code is {return_reason}.{comment_text}"
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
