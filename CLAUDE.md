# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
python3 -m pip install -r requirements.txt

# Run the app
python3 -m streamlit run app.py

# Generate synthetic demo data (creates data/transactions.csv)
python3 generate_data.py

# Run all tests
python3 -m pytest

# Run a single test file
python3 -m pytest tests/test_risk_logic.py
```

The app is served at `http://localhost:8501`. No linting tools are configured.

## Environment

Copy `.env` and set `GROQ_API_KEY` for LLM features. Without it, the app falls back to deterministic "Quick Insights" answers from `src/risk_logic.py`.

## Architecture

**Entry point:** `app.py` — Streamlit orchestration, historical/live work modes, sidebar filters, session state, chart/table rendering, structured operations feedback, model-learning views, and LLM calls.

**Supporting modules in `src/`:**

| Module | Purpose |
|--------|---------|
| `data_loader.py` | Loads `data/transactions.csv`, validates the 25 required columns, parses dates, adds `risk_type_display` |
| `metrics.py` | Pure functions: `format_currency`, `compute_summary_metrics`, `safe_top_index`, `to_csv_download` |
| `risk_logic.py` | Deterministic logic: case priority sorting, timeline building, fallback case selection, Quick Insights rule-based answers |
| `ui_components.py` | Rendering helpers: page header, hero summary, case detail box, timeline HTML, AI answer blocks, badges |

**Data flow:**
1. `data_loader.load_data()` reads CSV into a DataFrame
2. Sidebar filters applied in `app.py`
3. `metrics.compute_summary_metrics()` computes headline KPIs
4. When AI Copilot is queried: build context from filtered DataFrame + selected case, call Groq API, or fall back to `risk_logic.answer_*_question()`
5. If no case is selected, `risk_logic.select_fallback_case()` picks the highest-priority open case

**Key data schema fields:** `order_id`, `fraud_score`, `return_risk_score`, `unified_risk_score`, `risk_level` (High/Moderate/Low), `risk_type` (Fraud/Returns/None), `priority_tier`, `workflow_stage`, `case_status` (Open/Closed), `recommended_action`.

**AI Copilot:** Two modes — Dashboard Copilot (portfolio-level) and Case Copilot (order-specific). Both construct a context string and call Groq with a system prompt that restricts the LLM to supplied context only. Quick Insights is the rule-based fallback when `GROQ_API_KEY` is absent.

**Tests:** Unit tests in `tests/` cover all four `src/` modules. No end-to-end Streamlit tests exist.
