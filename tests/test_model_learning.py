import pandas as pd

from src.model_learning import (
    REPEAT_RETURNER_SEGMENT,
    build_backlog_candidates,
    classify_pattern,
    compute_learning_kpis,
    find_return_patterns,
    measure_interventions,
    prepare_learning_frame,
    summarize_model_misses,
)


def _orders(count, *, date="2025-03-01", product="Mug", state="CA", customer=None,
            requested=0, approved=0, flagged=False, reason="None", value=100.0):
    return [
        {
            "order_id": f"{product}-{state}-{date}-{i}",
            "customer_id": customer or f"C{i}-{product}-{date}",
            "order_date": date,
            "product_name": product,
            "state": state,
            "payment_method": "Stripe",
            "order_value": value,
            "return_requested": requested,
            "return_completed": approved,
            "risk_level": "High" if flagged else "Low",
            "return_reason_code": reason,
        }
        for i in range(count)
    ]


def _frame(*groups):
    return prepare_learning_frame(pd.DataFrame([row for group in groups for row in group]))


def test_prepare_tags_repeat_returners():
    view = _frame(
        _orders(4, customer="C1", requested=1, reason="Changed My Mind"),
        _orders(3, customer="C2", requested=1, reason="Changed My Mind"),
    )

    segments = view.groupby("customer_id")["customer_segment"].first()
    assert segments["C1"] == REPEAT_RETURNER_SEGMENT
    assert segments["C2"] == "Other customers"


def test_kpis_count_unflagged_returns_as_missed():
    view = _frame(
        _orders(2, requested=1, approved=1, flagged=False, value=50.0),
        _orders(3, product="Lamp", requested=1, approved=1, flagged=True),
        _orders(5, product="Pen"),
    )

    kpis = compute_learning_kpis(view)
    assert kpis == {
        "return_requests": 5,
        "missed_returns": 2,
        "missed_value": 100.0,
        "flagged_but_approved": 3,
    }


def test_model_misses_suggest_signal_or_leniency():
    view = _frame(
        _orders(12, product="Earbuds", requested=1, approved=1, flagged=False),
        _orders(12, product="Jacket", requested=1, approved=1, flagged=True),
        _orders(12, product="Lamp", requested=1, approved=0, flagged=True),
    )

    misses = summarize_model_misses(view).set_index("product_name")
    assert misses.loc["Earbuds", "suggestion"].startswith("Add signal")
    assert misses.loc["Jacket", "suggestion"].startswith("Be more lenient")
    assert misses.loc["Lamp", "suggestion"] == "No change"


def test_classify_separates_product_issue_from_abuse():
    product_issue = _frame(_orders(10, requested=1, approved=1, reason="Doesn't Fit Me"))
    abuse = _frame(_orders(10, requested=1, approved=1, reason="Changed My Mind"))

    assert classify_pattern(product_issue)["pattern_type"] == "Product issue"
    assert classify_pattern(abuse)["pattern_type"] == "Return abuse"


def test_find_patterns_flags_high_lift_segment_and_marks_it_resolved():
    view = _frame(
        # Baseline products, steady low return rate through the year.
        *[_orders(90, date=f"2025-{m:02d}-01", product="Mug") for m in range(1, 7)],
        *[_orders(10, date=f"2025-{m:02d}-01", product="Mug", requested=1, approved=1, reason="Changed My Mind")
          for m in range(1, 7)],
        # Jacket returns heavily for three months, then normalises after a fix.
        *[_orders(50, date=f"2025-{m:02d}-01", product="Jacket", requested=1, approved=1, reason="Doesn't Fit Me")
          for m in (1, 2, 3)],
        *[_orders(50, date=f"2025-{m:02d}-01", product="Jacket") for m in (1, 2, 3)],
        *[_orders(48, date=f"2025-{m:02d}-01", product="Jacket") for m in (4, 5, 6)],
        *[_orders(2, date=f"2025-{m:02d}-01", product="Jacket", requested=1, approved=1, reason="Doesn't Fit Me")
          for m in (4, 5, 6)],
    )

    patterns = find_return_patterns(view, dimensions=("product_name",))
    assert patterns["value"].tolist() == ["Jacket"]
    jacket = patterns.iloc[0]
    assert jacket["pattern_type"] == "Product issue"
    assert jacket["status"] == "Resolved"
    assert jacket["prior_value"] == 0.5
    assert jacket["recent_value"] == 0.04


def test_measure_interventions_compares_before_and_after():
    view = _frame(
        _orders(10, date="2025-09-01", product="Jacket", requested=1, approved=1, reason="Doesn't Fit Me"),
        _orders(10, date="2025-09-01", product="Jacket"),
        _orders(2, date="2025-10-10", product="Jacket", requested=1, approved=1, reason="Doesn't Fit Me"),
        _orders(18, date="2025-10-10", product="Jacket"),
    )
    interventions = [{
        "date": "2025-10-01", "label": "Size chart fixed", "field": "product_name",
        "value": "Jacket", "metric": "request_rate",
    }]

    result = measure_interventions(view, interventions)
    assert len(result) == 1
    assert result[0]["before_value"] == 0.5
    assert result[0]["after_value"] == 0.1
    assert (result[0]["before_orders"], result[0]["after_orders"]) == (20, 20)


def test_measure_interventions_skips_when_window_missing():
    view = _frame(_orders(10, date="2025-03-01", product="Jacket"))
    interventions = [{
        "date": "2025-10-01", "label": "Fix", "field": "product_name",
        "value": "Jacket", "metric": "request_rate",
    }]

    assert measure_interventions(view, interventions) == []


def test_backlog_candidates_carry_signal_and_rationale():
    patterns = pd.DataFrame([{
        "dimension": "product_name", "value": "Jacket", "pattern_type": "Product issue",
        "return_rate": 0.4, "lift": 2.4, "dominant_reason": "Doesn't Fit Me",
        "approval_rate": 0.87, "status": "Resolved", "guidance": "Fix product / sizing info",
        "owner": "Merchandising",
    }])
    misses = pd.DataFrame([{
        "product_name": "Earbuds", "requests": 34, "missed": 26, "missed_value": 7210.0,
        "miss_rate": 0.76, "flagged": 8, "flagged_approved": 6, "flagged_approval_rate": 0.75,
        "suggestion": "Add signal — model under-scores these returns",
    }])

    candidates = build_backlog_candidates(patterns, misses)
    assert [c["model_signal"] for c in candidates] == [
        "product_name:Jacket:product_fix",
        "product_name:Earbuds:add_signal",
    ]
    assert candidates[0]["priority"] == "Low"  # resolved patterns are low priority
    assert candidates[0]["owner"] == "Merchandising"
    assert "26 of 34 returns" in candidates[1]["rationale"]
