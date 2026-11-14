"""Customer Return Profiles: segment orders by own-store return history x external network history.

Two axes:
  * Store history  – the customer's return rate at this store (past_returns / past_orders).
    Customers with fewer than THIN_FILE_MIN_ORDERS prior orders are "New / Thin File".
  * Network history – cross-merchant return behaviour (ext_* columns, e.g. from a shared
    returns network). When those columns are absent the axis reports "No Network Data"
    and segments fall back to the Clean-network treatment.
"""

import pandas as pd

THIN_FILE_MIN_ORDERS = 3
STORE_LOW_MAX = 0.20
STORE_FREQUENT_MAX = 0.40

STORE_BANDS = ["Low", "Frequent", "Serial", "New / Thin File"]
NETWORK_BANDS = ["Clean", "Elevated", "Flagged"]
NO_NETWORK_DATA = "No Network Data"

EXTERNAL_COLUMNS = ["ext_return_rate", "ext_merchant_count", "ext_abuse_flags", "ext_linked_accounts"]

ACTION_TIER_ORDER = ["Reward", "Allow", "Fix Product", "Monitor", "Soft Review", "Restrict", "Block / Manual"]

# (store_band, network_band) -> segment definition
SEGMENTS = {
    ("Low", "Clean"): {
        "id": 1, "name": "Loyal Keepers",
        "profile": "Rarely return here or anywhere else. Core, high-trust customers.",
        "action_tier": "Reward",
        "recommended_action": "Frictionless returns – instant refund, free return label, loyalty perks.",
        "return_policy": "Standard window, instant refund on scan",
        "owner": "CX / Loyalty",
    },
    ("Low", "Elevated"): {
        "id": 2, "name": "Hidden Risk",
        "profile": "Look clean in this store but return heavily at other merchants.",
        "action_tier": "Monitor",
        "recommended_action": "Watchlist – require a return reason code and refund after carrier scan, not on request.",
        "return_policy": "Standard window, refund on carrier scan",
        "owner": "Returns Ops",
    },
    ("Low", "Flagged"): {
        "id": 3, "name": "Incoming Abuser",
        "profile": "Clean history here, but abuse flags or linked accounts across the network.",
        "action_tier": "Soft Review",
        "recommended_action": "Inspect before refund; block refunds to a different payment method; check linked accounts.",
        "return_policy": "Refund after inspection",
        "owner": "Returns Ops",
    },
    ("Frequent", "Clean"): {
        "id": 4, "name": "Fit Seekers",
        "profile": "Return often here but not elsewhere – usually a sizing or expectation gap, not abuse.",
        "action_tier": "Fix Product",
        "recommended_action": "Push exchanges over refunds, show size guidance and fit notes; route reason trends to merchandising.",
        "return_policy": "Exchange-first, refund allowed",
        "owner": "Merchandising / CX",
    },
    ("Frequent", "Elevated"): {
        "id": 5, "name": "Habitual Returners",
        "profile": "Regular returners in this store and across the network.",
        "action_tier": "Soft Review",
        "recommended_action": "Offer store credit before cash refund, shorten the return window, charge return shipping.",
        "return_policy": "Shortened window, store credit preferred",
        "owner": "Returns Ops",
    },
    ("Frequent", "Flagged"): {
        "id": 6, "name": "Opportunists",
        "profile": "Frequent returns plus network abuse signals; often promo- or discount-driven.",
        "action_tier": "Restrict",
        "recommended_action": "No cash refund on discounted items, mandatory inspection, exclude from future promos.",
        "return_policy": "Inspection required, store credit only on promo items",
        "owner": "Returns Ops",
    },
    ("Serial", "Clean"): {
        "id": 7, "name": "Store-Specific Problem",
        "profile": "Over-return only with this store – points at product data, sizing, or listing quality.",
        "action_tier": "Fix Product",
        "recommended_action": "Investigate top returned products and reasons; fix listings or size charts before restricting customers.",
        "return_policy": "Exchange-first, review repeat products",
        "owner": "Merchandising",
    },
    ("Serial", "Elevated"): {
        "id": 8, "name": "Serial Returners",
        "profile": "High return rate here and above-normal returns elsewhere.",
        "action_tier": "Restrict",
        "recommended_action": "Restricted returns – inspection plus store credit only; cap returns per period.",
        "return_policy": "Store credit only, inspection required",
        "owner": "Returns Ops",
    },
    ("Serial", "Flagged"): {
        "id": 9, "name": "Confirmed Abusers",
        "profile": "Serial returners here with network abuse flags or linked accounts.",
        "action_tier": "Block / Manual",
        "recommended_action": "Block self-serve returns – manual senior approval only; review linked accounts and payment instruments.",
        "return_policy": "Manual approval only",
        "owner": "Senior Review / Trust & Safety",
    },
    ("New / Thin File", "Clean"): {
        "id": 10, "name": "New & Clean",
        "profile": "Too few orders here to judge; nothing concerning across the network.",
        "action_tier": "Allow",
        "recommended_action": "Default policy; re-score after the third order.",
        "return_policy": "Standard window",
        "owner": "System",
    },
    ("New / Thin File", "Elevated"): {
        "id": 11, "name": "New & Elevated",
        "profile": "New here, with above-normal returns at other merchants.",
        "action_tier": "Soft Review",
        "recommended_action": "Soft review – refund after carrier scan; watch for discount + expedited shipping combos.",
        "return_policy": "Refund on carrier scan",
        "owner": "Returns Ops",
    },
    ("New / Thin File", "Flagged"): {
        "id": 12, "name": "New & Flagged",
        "profile": "Brand-new account with network abuse flags – classic buy-to-return or promo abuse pattern.",
        "action_tier": "Restrict",
        "recommended_action": "Hold refund for inspection, store credit only on promo orders, verify identity on high-value orders.",
        "return_policy": "Inspection required, store credit on promo orders",
        "owner": "Returns Ops / Trust & Safety",
    },
}


# ── Simple view: 2x2 rollup ──────────────────────────────────────────────────────
# Store: Low (<=20% returned, incl. new accounts) vs High (>20%).
# Network: Clean (Clean or Elevated) vs Risky (Flagged).
# Override: new accounts flagged across the network (#12 New & Flagged) roll up to Restrict,
# otherwise they would sit in the "Watch" box despite being the riskiest new-customer pattern.
SIMPLE_STORE_BANDS = ["Low", "High"]
SIMPLE_NETWORK_BANDS = ["Clean", "Risky"]

SIMPLE_SEGMENTS = {
    ("Low", "Clean"): {
        "id": 1, "name": "Trusted",
        "profile": "Low return rate here and no abuse signals elsewhere – includes new accounts with a clean network record.",
        "action_tier": "Allow",
        "recommended_action": "Frictionless returns and loyalty perks; re-score new accounts after their third order.",
        "return_policy": "Standard window, instant refund on scan",
        "owner": "System / CX",
    },
    ("Low", "Risky"): {
        "id": 2, "name": "Watch",
        "profile": "Clean history in this store, but abuse flags or linked accounts at other merchants.",
        "action_tier": "Soft Review",
        "recommended_action": "Refund after inspection, require a reason code, and check for linked accounts.",
        "return_policy": "Refund after inspection",
        "owner": "Returns Ops",
    },
    ("High", "Clean"): {
        "id": 3, "name": "Fix Experience",
        "profile": "Return often in this store without network abuse signals – usually sizing, listing, or expectation gaps.",
        "action_tier": "Fix Product",
        "recommended_action": "Push exchanges over refunds and fix size guidance / listings on top returned products; store credit for repeat serial returners.",
        "return_policy": "Exchange-first, store credit for repeat returners",
        "owner": "Merchandising / Returns Ops",
    },
    ("High", "Risky"): {
        "id": 4, "name": "Restrict",
        "profile": "Frequent returns plus network abuse signals, or brand-new accounts flagged across the network.",
        "action_tier": "Restrict",
        "recommended_action": "Inspection and store credit only, exclude from promos; manual senior approval for confirmed abusers.",
        "return_policy": "Inspection required, store credit only",
        "owner": "Returns Ops / Trust & Safety",
    },
}

# Detailed segment id -> simple (store, network) box. Every detailed segment maps to exactly one box.
DETAILED_TO_SIMPLE = {
    1: ("Low", "Clean"),    # Loyal Keepers
    2: ("Low", "Clean"),    # Hidden Risk
    3: ("Low", "Risky"),    # Incoming Abuser
    4: ("High", "Clean"),   # Fit Seekers
    5: ("High", "Clean"),   # Habitual Returners
    6: ("High", "Risky"),   # Opportunists
    7: ("High", "Clean"),   # Store-Specific Problem
    8: ("High", "Clean"),   # Serial Returners
    9: ("High", "Risky"),   # Confirmed Abusers
    10: ("Low", "Clean"),   # New & Clean
    11: ("Low", "Clean"),   # New & Elevated
    12: ("High", "Risky"),  # New & Flagged (override)
}


def detailed_segments_in_simple(simple_id: int) -> list[str]:
    """Names of the detailed segments that roll up into a simple box."""
    by_id = {seg["id"]: seg["name"] for seg in SEGMENTS.values()}
    key = next(k for k, v in SIMPLE_SEGMENTS.items() if v["id"] == simple_id)
    return [f"{i}. {by_id[i]}" for i, box in DETAILED_TO_SIMPLE.items() if box == key]


def has_network_data(data: pd.DataFrame) -> bool:
    return all(col in data.columns for col in EXTERNAL_COLUMNS) and data["ext_return_rate"].notna().any()


def assign_store_band(past_orders, past_returns) -> pd.Series:
    orders = pd.to_numeric(pd.Series(past_orders), errors="coerce").fillna(0)
    returns = pd.to_numeric(pd.Series(past_returns), errors="coerce").fillna(0)
    rate = returns / orders.clip(lower=1)
    band = pd.Series("Low", index=orders.index, dtype=object)
    band[rate > STORE_LOW_MAX] = "Frequent"
    band[rate > STORE_FREQUENT_MAX] = "Serial"
    band[orders < THIN_FILE_MIN_ORDERS] = "New / Thin File"
    return band


def assign_network_band(data: pd.DataFrame) -> pd.Series:
    if not has_network_data(data):
        return pd.Series(NO_NETWORK_DATA, index=data.index, dtype=object)
    rate = pd.to_numeric(data["ext_return_rate"], errors="coerce").fillna(0)
    merchants = pd.to_numeric(data["ext_merchant_count"], errors="coerce").fillna(0)
    flags = pd.to_numeric(data["ext_abuse_flags"], errors="coerce").fillna(0)
    linked = pd.to_numeric(data["ext_linked_accounts"], errors="coerce").fillna(0)

    band = pd.Series("Clean", index=data.index, dtype=object)
    band[(rate >= 0.25) | (merchants >= 3)] = "Elevated"
    band[(flags >= 1) | (linked >= 2) | ((rate >= 0.45) & (merchants >= 5))] = "Flagged"
    return band


def assign_return_profiles(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of data with store_band, network_band, segment_id/name and action columns."""
    out = data.copy()
    if out.empty:
        for col in ["store_return_rate", "store_band", "network_band", "segment_id", "segment_name",
                    "segment_action_tier", "segment_recommended_action", "simple_store_band",
                    "simple_network_band", "simple_segment_id", "simple_segment_name", "simple_action_tier"]:
            out[col] = pd.Series(dtype=object)
        return out

    orders = pd.to_numeric(out["past_orders"], errors="coerce").fillna(0)
    returns = pd.to_numeric(out["past_returns"], errors="coerce").fillna(0)
    out["store_return_rate"] = (returns / orders.clip(lower=1)).round(3)
    out["store_band"] = assign_store_band(orders, returns).values
    out["network_band"] = assign_network_band(out).values

    # Without network data, fall back to the Clean-network treatment for each store band.
    lookup_network = out["network_band"].replace(NO_NETWORK_DATA, "Clean")
    keys = list(zip(out["store_band"], lookup_network))
    out["segment_id"] = [SEGMENTS[k]["id"] for k in keys]
    out["segment_name"] = [SEGMENTS[k]["name"] for k in keys]
    out["segment_action_tier"] = [SEGMENTS[k]["action_tier"] for k in keys]
    out["segment_recommended_action"] = [SEGMENTS[k]["recommended_action"] for k in keys]

    simple_keys = [DETAILED_TO_SIMPLE[i] for i in out["segment_id"]]
    out["simple_store_band"] = [k[0] for k in simple_keys]
    out["simple_network_band"] = [k[1] for k in simple_keys]
    out["simple_segment_id"] = [SIMPLE_SEGMENTS[k]["id"] for k in simple_keys]
    out["simple_segment_name"] = [SIMPLE_SEGMENTS[k]["name"] for k in simple_keys]
    out["simple_action_tier"] = [SIMPLE_SEGMENTS[k]["action_tier"] for k in simple_keys]
    return out


def _top_value(series: pd.Series, exclude=("None", "", "nan")) -> str:
    values = series.astype(str)
    values = values[~values.isin(exclude)]
    if values.empty:
        return "—"
    return str(values.value_counts().index[0])


def summarize_return_profiles(profiled: pd.DataFrame, level: str = "detailed") -> pd.DataFrame:
    """One row per segment, empty segments included with zero orders.

    level="detailed" -> the 12-segment grid; level="simple" -> the 4-box 2x2 rollup.
    """
    definitions = SIMPLE_SEGMENTS if level == "simple" else SEGMENTS
    id_col = "simple_segment_id" if level == "simple" else "segment_id"
    total_orders = max(len(profiled), 1)
    rows = []
    for (store_band, network_band), seg in definitions.items():
        part = profiled[profiled[id_col].eq(seg["id"])] if not profiled.empty else profiled
        n = len(part)
        requested = pd.to_numeric(part.get("return_requested", pd.Series(dtype=float)), errors="coerce").fillna(0)
        completed = pd.to_numeric(part.get("return_completed", pd.Series(dtype=float)), errors="coerce").fillna(0)
        value = pd.to_numeric(part.get("order_value", pd.Series(dtype=float)), errors="coerce").fillna(0)
        returned = part[requested.eq(1)] if n else part
        rows.append({
            "segment_id": seg["id"],
            "segment_name": seg["name"],
            "store_band": store_band,
            "network_band": network_band,
            "customers": int(part["customer_id"].nunique()) if n else 0,
            "orders": n,
            "order_share_pct": round(n / total_orders * 100, 1),
            "revenue": round(float(value.sum()), 2),
            "avg_order_value": round(float(value.mean()), 2) if n else 0.0,
            "return_request_rate_pct": round(float(requested.mean()) * 100, 1) if n else 0.0,
            "approved_return_rate_pct": round(float(completed.mean()) * 100, 1) if n else 0.0,
            "approved_return_value": round(float(value[completed.eq(1)].sum()), 2) if n else 0.0,
            "avg_return_risk_score": round(float(pd.to_numeric(part["return_risk_score"], errors="coerce").mean()), 3) if n else 0.0,
            "high_risk_pct": round(float(part["risk_level"].astype(str).eq("High").mean()) * 100, 1) if n else 0.0,
            "avg_store_return_rate_pct": round(float(part["store_return_rate"].mean()) * 100, 1) if n else 0.0,
            "avg_network_return_rate_pct": (
                round(float(pd.to_numeric(part["ext_return_rate"], errors="coerce").mean()) * 100, 1)
                if n and "ext_return_rate" in part.columns else None
            ),
            "discount_pct": round(float(pd.to_numeric(part["discount_used"], errors="coerce").mean()) * 100, 1) if n else 0.0,
            "expedited_pct": round(float(pd.to_numeric(part["expedited_shipping"], errors="coerce").mean()) * 100, 1) if n else 0.0,
            "avg_tenure_days": int(pd.to_numeric(part["customer_lifetime_days"], errors="coerce").mean()) if n else 0,
            "top_category": _top_value(part["product_category"]) if n else "—",
            "top_product": _top_value(part["product_name"]) if n else "—",
            "top_return_reason": _top_value(returned["return_reason_code"]) if len(returned) and "return_reason_code" in returned.columns else "—",
            "action_tier": seg["action_tier"],
            "recommended_action": seg["recommended_action"],
            "return_policy": seg["return_policy"],
            "owner": seg["owner"],
            "profile": seg["profile"],
        })
    return pd.DataFrame(rows)
