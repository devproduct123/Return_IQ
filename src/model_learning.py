"""Model Learning analytics.

Turns historical return outcomes into three kinds of learning:
  1. Where the model missed — returns it never flagged, and flagged returns ops approved anyway.
  2. Emerging return patterns — segments returning well above baseline, classified by the
     fix they need (product fix vs. stricter policy vs. monitoring), with a current status.
  3. Intervention impact — before/after for actions already taken.
"""

import numpy as np
import pandas as pd

REPEAT_RETURNER_MIN_RETURNS = 4
REPEAT_RETURNER_SEGMENT = f"Repeat returners ({REPEAT_RETURNER_MIN_RETURNS}+ returns)"
PRODUCT_ISSUE_REASONS = {"Doesn't Fit Me", "Product Quality Issues", "Not What I Expected"}

# Actions taken in the demo case study (see generate_data.py). Each one names the
# segment it targeted and the metric it was meant to move.
INTERVENTIONS = [
    {
        "date": "2025-09-15",
        "label": "Serial-returner policy",
        "field": "customer_segment",
        "value": REPEAT_RETURNER_SEGMENT,
        "metric": "approved_rate",
    },
    {
        "date": "2025-10-01",
        "label": "Jacket size chart fixed",
        "field": "product_name",
        "value": "Premium Jacket",
        "metric": "request_rate",
    },
]

METRIC_LABELS = {
    "request_rate": "Return request rate",
    "approved_rate": "Approved return rate",
}

PATTERN_PLAYBOOK = {
    "Product issue": {
        "guidance": "Fix product / sizing info",
        "owner": "Merchandising",
        "status_metric": "request_rate",
        "signal_suffix": "product_fix",
        "why": "Returns are mostly fit or quality complaints and are approved after inspection, "
               "so restricting them would punish genuine customers.",
    },
    "Return abuse": {
        "guidance": "Be stricter",
        "owner": "Returns Ops",
        "status_metric": "approved_rate",
        "signal_suffix": "stricter_return_review",
        "why": "Returns are concentrated in repeat returners or no-defect reasons, "
               "which points to policy abuse rather than a product problem.",
    },
    "Elevated returns": {
        "guidance": "Monitor closely",
        "owner": "Analytics",
        "status_metric": "request_rate",
        "signal_suffix": "monitor",
        "why": "Return rate is above baseline without a single clear driver yet.",
    },
}

PATTERN_COLUMNS = [
    "dimension", "value", "pattern_type", "orders", "return_rate", "lift",
    "dominant_reason", "approval_rate", "repeat_returner_share", "approved_value",
    "status", "status_metric", "prior_value", "recent_value", "guidance", "owner",
]
MISS_COLUMNS = [
    "requests", "missed", "missed_value", "miss_rate",
    "flagged", "flagged_approved", "flagged_approval_rate", "suggestion",
]


def prepare_learning_frame(df: pd.DataFrame) -> pd.DataFrame:
    view = df.copy()
    view["requested"] = pd.to_numeric(view["return_requested"], errors="coerce").fillna(0).astype(int)
    view["approved"] = pd.to_numeric(view["return_completed"], errors="coerce").fillna(0).astype(int)
    view["flagged"] = view["risk_level"].astype(str).eq("High")
    view["order_date"] = pd.to_datetime(view["order_date"], errors="coerce")
    returns_per_customer = view.groupby("customer_id")["requested"].transform("sum")
    view["customer_segment"] = np.where(
        returns_per_customer >= REPEAT_RETURNER_MIN_RETURNS, REPEAT_RETURNER_SEGMENT, "Other customers"
    )
    return view


def metric_value(frame: pd.DataFrame, metric: str) -> float | None:
    if frame.empty:
        return None
    column = "requested" if metric == "request_rate" else "approved"
    return float(frame[column].mean())


def compute_learning_kpis(view: pd.DataFrame) -> dict:
    missed = view[(view["requested"] == 1) & ~view["flagged"]]
    return {
        "return_requests": int(view["requested"].sum()),
        "missed_returns": len(missed),
        "missed_value": float(missed["order_value"].sum()),
        "flagged_but_approved": int((view["flagged"] & (view["approved"] == 1)).sum()),
    }


def summarize_model_misses(
    view: pd.DataFrame,
    group_column: str = "product_name",
    top_n: int = 8,
    min_cases: int = 10,
) -> pd.DataFrame:
    returns = view[view["requested"] == 1].copy()
    if returns.empty:
        return pd.DataFrame(columns=[group_column, *MISS_COLUMNS])

    returns["is_missed"] = (~returns["flagged"]).astype(int)
    returns["missed_value"] = returns["order_value"] * returns["is_missed"]
    returns["is_flagged"] = returns["flagged"].astype(int)
    returns["flagged_approved"] = (returns["flagged"] & (returns["approved"] == 1)).astype(int)

    summary = (
        returns.groupby(group_column, as_index=False)
        .agg(
            requests=("requested", "sum"),
            missed=("is_missed", "sum"),
            missed_value=("missed_value", "sum"),
            flagged=("is_flagged", "sum"),
            flagged_approved=("flagged_approved", "sum"),
        )
    )
    summary["miss_rate"] = summary["missed"] / summary["requests"]
    summary["flagged_approval_rate"] = np.where(
        summary["flagged"] > 0, summary["flagged_approved"] / summary["flagged"].clip(lower=1), 0.0
    )

    def suggest(row) -> str:
        if row["missed"] >= min_cases and row["miss_rate"] >= 0.5:
            return "Add signal — model under-scores these returns"
        if row["flagged"] >= min_cases and row["flagged_approval_rate"] >= 0.7:
            return "Be more lenient — flagged returns are mostly approved"
        return "No change"

    summary["suggestion"] = summary.apply(suggest, axis=1)
    return summary.sort_values("missed_value", ascending=False).head(top_n)[[group_column, *MISS_COLUMNS]]


def classify_pattern(returns: pd.DataFrame) -> dict:
    if returns.empty:
        return {
            "pattern_type": "Elevated returns", "dominant_reason": "None",
            "approval_rate": 0.0, "repeat_returner_share": 0.0,
        }

    reasons = returns["return_reason_code"].astype(str).value_counts(normalize=True)
    dominant_reason = str(reasons.index[0])
    product_reason_share = float(reasons[reasons.index.isin(PRODUCT_ISSUE_REASONS)].sum())
    approval_rate = float(returns["approved"].mean())
    repeat_share = float(returns["customer_segment"].eq(REPEAT_RETURNER_SEGMENT).mean())

    if repeat_share >= 0.3 or (dominant_reason == "Changed My Mind" and reasons.iloc[0] >= 0.4):
        pattern_type = "Return abuse"
    elif product_reason_share >= 0.6 and approval_rate >= 0.6:
        pattern_type = "Product issue"
    else:
        pattern_type = "Elevated returns"

    return {
        "pattern_type": pattern_type,
        "dominant_reason": dominant_reason,
        "approval_rate": approval_rate,
        "repeat_returner_share": repeat_share,
    }


def pattern_status(view: pd.DataFrame, group: pd.DataFrame, metric: str, window_months: int = 3) -> dict:
    """Compare the latest window against the window before it, and against the overall rate."""
    months = view["order_date"].dt.to_period("M")
    group_months = months.loc[group.index]
    last = months.max()
    recent_periods = {last - i for i in range(window_months)}
    prior_periods = {last - window_months - i for i in range(window_months)}

    recent = metric_value(group[group_months.isin(recent_periods)], metric)
    prior = metric_value(group[group_months.isin(prior_periods)], metric)
    overall = metric_value(view, metric) or 0.0
    tolerance = overall * 1.15

    if recent is None:
        status = "Insufficient data"
    elif recent <= tolerance:
        status = "Resolved"
    elif prior is not None and recent <= prior * 0.75:
        status = "Improving"
    else:
        status = "Active"

    return {"status": status, "prior_value": prior, "recent_value": recent}


def find_return_patterns(
    view: pd.DataFrame,
    dimensions: tuple[str, ...] = ("product_name", "state", "payment_method", "customer_segment"),
    min_orders: int = 100,
    min_lift: float = 1.3,
) -> pd.DataFrame:
    if view.empty or view["requested"].sum() == 0:
        return pd.DataFrame(columns=PATTERN_COLUMNS)

    baseline_rate = float(view["requested"].mean())
    rows = []
    for dimension in dimensions:
        for value, group in view.groupby(dimension):
            if dimension == "customer_segment" and value != REPEAT_RETURNER_SEGMENT:
                continue
            if len(group) < min_orders:
                continue
            return_rate = float(group["requested"].mean())
            lift = return_rate / baseline_rate
            if lift < min_lift:
                continue

            returns = group[group["requested"] == 1]
            classification = classify_pattern(returns)
            playbook = PATTERN_PLAYBOOK[classification["pattern_type"]]
            status = pattern_status(view, group, playbook["status_metric"])
            rows.append({
                "dimension": dimension,
                "value": str(value),
                "orders": len(group),
                "return_rate": return_rate,
                "lift": lift,
                "approved_value": float(group.loc[group["approved"] == 1, "order_value"].sum()),
                **classification,
                **status,
                "status_metric": playbook["status_metric"],
                "guidance": playbook["guidance"],
                "owner": playbook["owner"],
            })

    if not rows:
        return pd.DataFrame(columns=PATTERN_COLUMNS)
    return pd.DataFrame(rows).sort_values("approved_value", ascending=False)[PATTERN_COLUMNS]


def measure_interventions(
    view: pd.DataFrame,
    interventions: list[dict] = INTERVENTIONS,
    window_days: int = 90,
) -> list[dict]:
    results = []
    for item in interventions:
        if item["field"] not in view.columns:
            continue
        date = pd.Timestamp(item["date"])
        segment = view[view[item["field"]].astype(str).eq(item["value"])]
        before = segment[(segment["order_date"] >= date - pd.Timedelta(days=window_days)) & (segment["order_date"] < date)]
        after = segment[(segment["order_date"] >= date) & (segment["order_date"] < date + pd.Timedelta(days=window_days))]
        if before.empty or after.empty:
            continue

        def approved_value_per_month(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
            start = max(start, view["order_date"].min())
            end = min(end, view["order_date"].max() + pd.Timedelta(days=1))
            months = max((end - start).days / 30.0, 1 / 30.0)
            return float(frame.loc[frame["approved"] == 1, "order_value"].sum()) / months

        results.append({
            **item,
            "metric_label": METRIC_LABELS[item["metric"]],
            "before_value": metric_value(before, item["metric"]),
            "after_value": metric_value(after, item["metric"]),
            "before_orders": len(before),
            "after_orders": len(after),
            "before_loss_per_month": approved_value_per_month(before, date - pd.Timedelta(days=window_days), date),
            "after_loss_per_month": approved_value_per_month(after, date, date + pd.Timedelta(days=window_days)),
        })
    return results


def build_backlog_candidates(patterns: pd.DataFrame, misses: pd.DataFrame, miss_column: str = "product_name") -> list[dict]:
    """Turn patterns and model misses into backlog-ready items, keyed by a readable label."""
    candidates = []
    for _, row in patterns.iterrows():
        playbook = PATTERN_PLAYBOOK[row["pattern_type"]]
        priority = "Low" if row["status"] == "Resolved" else "High" if row["lift"] >= 1.8 else "Medium"
        candidates.append({
            "label": f"{row['pattern_type']}: {row['dimension']} = {row['value']} → {row['guidance']}",
            "focus_area": "Returns",
            "condition_field": row["dimension"],
            "condition_value": row["value"],
            "guidance": row["guidance"],
            "priority": priority,
            "owner": row["owner"],
            "model_signal": f"{row['dimension']}:{row['value']}:{playbook['signal_suffix']}",
            "rationale": (
                f"Return rate {row['return_rate']:.0%} ({row['lift']:.1f}x baseline), mostly "
                f"'{row['dominant_reason']}', {row['approval_rate']:.0%} approved. {playbook['why']} "
                f"Status: {row['status']}."
            ),
            "source": "Suggested pattern",
        })

    for _, row in misses[misses["suggestion"].ne("No change")].iterrows():
        lenient = row["suggestion"].startswith("Be more lenient")
        candidates.append({
            "label": f"Model miss: {miss_column} = {row[miss_column]} → {row['suggestion'].split(' — ')[0]}",
            "focus_area": "Returns",
            "condition_field": miss_column,
            "condition_value": str(row[miss_column]),
            "guidance": "Be more lenient" if lenient else "Add model signal",
            "priority": "Medium",
            "owner": "Analytics",
            "model_signal": f"{miss_column}:{row[miss_column]}:{'reduce_friction' if lenient else 'add_signal'}",
            "rationale": (
                f"{row['flagged_approved']} of {row['flagged']} flagged returns were approved by ops."
                if lenient else
                f"{row['missed']} of {row['requests']} returns ({row['miss_rate']:.0%}) were not flagged, "
                f"${row['missed_value']:,.0f} in approved exposure."
            ),
            "source": "Model miss analysis",
        })
    return candidates
