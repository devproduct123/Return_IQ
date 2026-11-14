import pandas as pd

from src.return_profiles import (
    NO_NETWORK_DATA,
    SEGMENTS,
    assign_network_band,
    assign_return_profiles,
    assign_store_band,
    summarize_return_profiles,
)


def _orders(rows):
    base = {
        "order_id": "O1",
        "customer_id": "C1",
        "order_value": 100.0,
        "return_requested": 0,
        "return_completed": 0,
        "return_reason_code": "None",
        "return_risk_score": 0.2,
        "risk_level": "Low",
        "discount_used": 0,
        "expedited_shipping": 0,
        "customer_lifetime_days": 400,
        "product_category": "Apparel",
        "product_name": "Hoodie",
        "case_status": "Closed",
        "ext_return_rate": 0.05,
        "ext_merchant_count": 1,
        "ext_abuse_flags": 0,
        "ext_linked_accounts": 0,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_store_band_thresholds_and_thin_file():
    bands = assign_store_band([10, 10, 10, 10, 2], [0, 2, 4, 5, 2]).tolist()
    assert bands == ["Low", "Low", "Frequent", "Serial", "New / Thin File"]


def test_network_band_rules():
    df = _orders([
        {"ext_return_rate": 0.10, "ext_merchant_count": 1},
        {"ext_return_rate": 0.30, "ext_merchant_count": 1},
        {"ext_return_rate": 0.10, "ext_merchant_count": 4},
        {"ext_return_rate": 0.10, "ext_abuse_flags": 1},
        {"ext_return_rate": 0.10, "ext_linked_accounts": 2},
        {"ext_return_rate": 0.50, "ext_merchant_count": 6},
    ])
    assert assign_network_band(df).tolist() == [
        "Clean", "Elevated", "Elevated", "Flagged", "Flagged", "Flagged",
    ]


def test_network_band_without_external_columns():
    df = _orders([{}]).drop(columns=["ext_return_rate", "ext_merchant_count", "ext_abuse_flags", "ext_linked_accounts"])
    assert assign_network_band(df).tolist() == [NO_NETWORK_DATA]
    profiled = assign_return_profiles(df.assign(past_orders=10, past_returns=0))
    assert profiled["segment_name"].iloc[0] == "Loyal Keepers"


def test_assign_return_profiles_maps_grid_to_segments():
    df = _orders([
        {"past_orders": 10, "past_returns": 0},
        {"past_orders": 10, "past_returns": 6, "ext_abuse_flags": 2},
        {"past_orders": 1, "past_returns": 0, "ext_return_rate": 0.6, "ext_merchant_count": 8},
        {"past_orders": 10, "past_returns": 3},
    ])
    profiled = assign_return_profiles(df)
    assert profiled["segment_name"].tolist() == [
        "Loyal Keepers", "Confirmed Abusers", "New & Flagged", "Fit Seekers",
    ]
    assert profiled["segment_action_tier"].tolist() == ["Reward", "Block / Manual", "Restrict", "Fix Product"]


def test_summarize_return_profiles_covers_all_segments():
    df = _orders([
        {"order_id": "O1", "customer_id": "C1", "past_orders": 10, "past_returns": 6,
         "ext_abuse_flags": 1, "return_requested": 1, "return_completed": 1, "order_value": 200.0,
         "return_reason_code": "Changed My Mind"},
        {"order_id": "O2", "customer_id": "C1", "past_orders": 10, "past_returns": 6,
         "ext_abuse_flags": 1, "order_value": 100.0},
    ])
    summary = summarize_return_profiles(assign_return_profiles(df))
    assert len(summary) == len(SEGMENTS)
    abusers = summary[summary["segment_name"].eq("Confirmed Abusers")].iloc[0]
    assert abusers["orders"] == 2
    assert abusers["customers"] == 1
    assert abusers["return_request_rate_pct"] == 50.0
    assert abusers["approved_return_value"] == 200.0
    assert abusers["top_return_reason"] == "Changed My Mind"
    assert summary["orders"].sum() == 2


def test_empty_frame_is_handled():
    profiled = assign_return_profiles(_orders([{"past_orders": 1, "past_returns": 0}]).iloc[0:0])
    summary = summarize_return_profiles(profiled)
    assert summary["orders"].sum() == 0


def test_every_detailed_segment_rolls_up_to_one_simple_box():
    from src.return_profiles import DETAILED_TO_SIMPLE, SIMPLE_SEGMENTS
    assert sorted(DETAILED_TO_SIMPLE) == sorted(seg["id"] for seg in SEGMENTS.values())
    assert set(DETAILED_TO_SIMPLE.values()) == set(SIMPLE_SEGMENTS)


def test_simple_rollup_and_new_flagged_override():
    df = _orders([
        {"past_orders": 10, "past_returns": 0},                                   # Loyal Keepers -> Trusted
        {"past_orders": 10, "past_returns": 0, "ext_abuse_flags": 1},             # Incoming Abuser -> Watch
        {"past_orders": 10, "past_returns": 3},                                   # Fit Seekers -> Fix Experience
        {"past_orders": 10, "past_returns": 6, "ext_abuse_flags": 2},             # Confirmed Abusers -> Restrict
        {"past_orders": 1, "past_returns": 0, "ext_abuse_flags": 1},              # New & Flagged -> Restrict (override)
        {"past_orders": 1, "past_returns": 0, "ext_return_rate": 0.3},            # New & Elevated -> Trusted
    ])
    profiled = assign_return_profiles(df)
    assert profiled["simple_segment_name"].tolist() == [
        "Trusted", "Watch", "Fix Experience", "Restrict", "Restrict", "Trusted",
    ]


def test_simple_summary_has_four_boxes_and_matches_detailed_totals():
    df = _orders([
        {"order_id": "O1", "past_orders": 10, "past_returns": 0},
        {"order_id": "O2", "past_orders": 10, "past_returns": 6, "ext_abuse_flags": 1},
        {"order_id": "O3", "past_orders": 1, "past_returns": 0, "ext_abuse_flags": 1},
    ])
    profiled = assign_return_profiles(df)
    simple = summarize_return_profiles(profiled, "simple")
    detailed = summarize_return_profiles(profiled)
    assert simple["segment_name"].tolist() == ["Trusted", "Watch", "Fix Experience", "Restrict"]
    assert simple["orders"].sum() == detailed["orders"].sum() == 3
    assert simple.set_index("segment_name").loc["Restrict", "orders"] == 2


def test_detailed_segments_in_simple():
    from src.return_profiles import detailed_segments_in_simple
    assert detailed_segments_in_simple(4) == ["6. Opportunists", "9. Confirmed Abusers", "12. New & Flagged"]
