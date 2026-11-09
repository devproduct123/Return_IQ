import pandas as pd

from src.metrics import format_currency, safe_top_index


PRIORITY_ORDER = {"High": 0, "Moderate": 1, "Low": 2}


def build_case_timeline(case_row: pd.Series):
    current_stage = str(case_row["workflow_stage"])
    recommended_action = str(case_row["recommended_action"])
    case_status = str(case_row["case_status"])

    steps = [("Case Created", "done")]

    stage_map = {
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
        data.groupby("state")["return_risk_score"]
        .mean()
        .sort_values(ascending=False)
    )

    high_cases = int((data["priority_tier"] == "High").sum())
    top_action = safe_top_index(data["recommended_action"].value_counts())

    if "summary" in q or "summarize" in q or "overall" in q:
        return (
            f"In the current filtered view, {flagged_orders:,} out of {total_orders:,} orders are flagged for return risk. "
            f"Revenue at risk is {format_currency(revenue_at_risk)}. "
            f"Protected revenue on closed flagged cases is {format_currency(protected_revenue)}. "
            f"There are {open_cases:,} open cases and {high_cases:,} high-priority cases. "
            f"Return risk pressure is highest in {top_category}, with {top_state} as the top-risk state, "
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

    if "state" in q or "where" in q or "location" in q:
        top_return_states = (
            data.groupby("state")["return_risk_score"]
            .mean()
            .sort_values(ascending=False)
            .head(3)
            .index.tolist()
        )
        return f"The highest return-risk states in the current view are: {', '.join(top_return_states)}."

    if "revenue" in q and "risk" in q:
        return (
            f"Revenue at risk in the current filtered view is {format_currency(revenue_at_risk)}. "
            f"Protected revenue from closed flagged cases is {format_currency(protected_revenue)}."
        )

    if "priority" in q or "review first" in q or "what should i do first" in q:
        if critical_open.empty:
            return "There are no open high-priority cases in the current filtered view."
        return (
            f"There are {len(critical_open):,} open high-priority return risk cases that should be reviewed first. "
            f"Focus on the highest-value orders with the highest return risk scores."
        )

    if "action" in q or "recommended" in q:
        action_counts = data["recommended_action"].value_counts().head(3)
        action_text = ", ".join(
            [f"{k} ({v})" for k, v in action_counts.items()])
        return f"The top recommended actions in the current view are: {action_text}."

    if "open case" in q or "open cases" in q:
        return f"There are currently {open_cases:,} open cases in the filtered view."

    return (
        "I can help summarize current return risk, identify top return-risk categories, products, or states, "
        "revenue at risk, protected revenue, open cases, and what the ops team should prioritize first."
    )


def _simplify_actual_outcome(row: pd.Series) -> str:
    outcome_type = str(row.get("outcome_type", ""))
    risk_type = str(row.get("risk_type", ""))

    if risk_type == "Returns" or outcome_type.startswith("Return"):
        return "Return Approved" if outcome_type == "Return Approved" else "Return Denied"

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
    total_loss = float(
        view.loc[view["actual_outcome"].eq("Return Approved"), "order_value"].sum()
    )
    avoided_loss = float(
        view.loc[view["actual_outcome"].eq("Return Denied"), "order_value"].sum()
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
    top_return_state = safe_top_index(
        view.loc[view["actual_outcome"].eq("Return Approved")]
        .groupby("state")["order_value"]
        .sum()
        .sort_values(ascending=False),
        default="None",
    )

    if "summary" in q or "summarize" in q or "overall" in q:
        return (
            f"In the historical view, {total_orders:,} completed orders produced {actual_returns:,} approved returns "
            f"and {denied_returns:,} denied returns. Total return loss was {format_currency(total_loss)}, "
            f"while avoided exposure was {format_currency(avoided_loss)}."
        )

    if "category" in q or "product" in q or "driver" in q:
        return (
            f"Approved return loss is highest in category {top_return_category}. "
            f"The top product by approved return loss is {top_return_product}."
        )

    if "state" in q or "where" in q or "location" in q:
        return f"Approved return loss is highest in {top_return_state}."

    if "loss" in q or "dollar" in q or "revenue" in q:
        return (
            f"Total historical loss from approved returns is {format_currency(total_loss)}. "
            f"Avoided exposure from denied returns is {format_currency(avoided_loss)}."
        )

    return (
        "I can summarize actual historical return outcomes, approved return loss, avoided exposure, "
        "top return-loss categories, products, states, and reason-code patterns."
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
    return_score = float(case_row["return_risk_score"])
    order_value = float(case_row["order_value"])
    past_returns = int(case_row["past_returns"])
    past_orders = int(case_row.get("past_orders", 1))
    product_name = case_row["product_name"]
    product_category = case_row["product_category"]
    return_reason = case_row.get("return_reason_code", "None")
    return_comment = str(case_row.get("customer_return_comment", "")).strip()

    if "summary" in q or "summarize" in q or "case" in q:
        return (
            f"Order {order_id} is a {priority}-priority {risk_type} case with status {case_status}. "
            f"The order value is {format_currency(order_value)}, return risk score is {return_score:.3f}, "
            f"and the current workflow stage is {workflow_stage}. "
            f"Recommended action is {recommended_action}. Main reason flagged: {risk_reason}"
        )

    if "why" in q or "flagged" in q or "reason" in q:
        return (
            f"This case was flagged as {risk_type}. "
            f"Return risk score is {return_score:.3f}. "
            f"Primary reason: {risk_reason}"
        )

    if "next step" in q or "what should i do" in q or "action" in q or "review first" in q:
        return (
            f"The recommended action for order {order_id} is {recommended_action}. "
            f"The suggested next step is: {next_step}. "
            f"Current stage is {workflow_stage}, so the analyst should align the next review step with that stage."
        )

    if "return" in q:
        return_rate = past_returns / max(past_orders, 1)
        comment_text = f" Customer return comment: {return_comment}" if return_comment else ""
        return (
            f"Return risk score for order {order_id} is {return_score:.3f}. "
            f"The product is {product_name} in category {product_category}. "
            f"The customer has {past_returns} past returns out of {past_orders} orders ({return_rate:.0%} return rate). "
            f"Return reason code is {return_reason}.{comment_text}"
        )

    if "customer" in q or "verification" in q or "inspect" in q:
        return (
            f"For this case, a reasonable analyst path is to use the recommended action '{recommended_action}' "
            f"and the AI next step '{next_step}' while requesting only the minimum required customer verification."
        )

    return (
        f"I can help explain why order {order_id} was flagged, summarize the case, recommend the next step, "
        f"or answer questions about return risk signals, return history, verification, and escalation."
    )
