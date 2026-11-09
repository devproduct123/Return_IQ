import pandas as pd


def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def format_currency(x: float) -> str:
    if x >= 1_000_000:
        return f"${x / 1_000_000:.1f}M"
    if x >= 1_000:
        return f"${x / 1_000:.1f}K"
    return f"${x:,.0f}"


def safe_top_index(series: pd.Series, default: str = "N/A") -> str:
    return series.index[0] if len(series) > 0 else default


def compute_summary_metrics(data: pd.DataFrame):
    total_orders = len(data)
    flagged_orders = int((data["risk_level"] == "High").sum())
    flagged_pct = round((flagged_orders / total_orders)
                        * 100, 1) if total_orders else 0.0
    return_cases = int((data["risk_type_display"] == "Return Risk").sum())
    revenue_at_risk = float(
        data.loc[data["risk_level"] == "High", "order_value"].sum())
    high_cases = int((data["priority_tier"] == "High").sum())
    open_cases = int((data["case_status"] == "Open").sum())
    closed_cases = int((data["case_status"] == "Closed").sum())
    protected_revenue = float(
        data.loc[
            (data["risk_level"] == "High") & (data["case_status"] == "Closed"),
            "order_value"
        ].sum()
    )

    return {
        "total_orders": total_orders,
        "flagged_orders": flagged_orders,
        "flagged_pct": flagged_pct,
        "return_cases": return_cases,
        "revenue_at_risk": revenue_at_risk,
        "high_cases": high_cases,
        "open_cases": open_cases,
        "closed_cases": closed_cases,
        "protected_revenue": protected_revenue,
    }


def compute_outcome_breakdown(data: pd.DataFrame) -> dict[str, dict]:
    """Break protected revenue into outcome categories for closed flagged cases."""
    if "outcome_type" not in data.columns:
        return {}

    closed_flagged = data[
        (data["risk_level"] == "High")
        & (data["case_status"] == "Closed")
        & (~data["outcome_type"].isin(["Pending", "Auto Approved", ""]))
    ]

    breakdown: dict[str, dict] = {}
    for outcome in closed_flagged["outcome_type"].unique():
        subset = closed_flagged[closed_flagged["outcome_type"] == outcome]
        breakdown[outcome] = {
            "count": len(subset),
            "revenue": float(subset["order_value"].sum()),
        }

    return breakdown
