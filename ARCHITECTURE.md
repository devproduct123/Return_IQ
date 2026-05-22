# Aignita RiskIQ Architecture

This document describes the current architecture of the Aignita RiskIQ Streamlit app after the initial modularization work. The goal is to make the project easier to extend without overstating its current production readiness.

## High-Level Architecture

RiskIQ is currently a single Streamlit application with extracted helper modules for data loading, metrics, risk logic, and reusable UI rendering.

```text
data/transactions.csv
        |
        v
src/data_loader.py
        |
        v
app.py  ----> src/metrics.py
  |       \--> src/risk_logic.py
  |       \--> src/ui_components.py
  |
  +---- optional Groq API call for AI Copilot
  |
  v
Streamlit dashboard, Ops Queue, case review, Copilot output
```

The app uses pandas for data manipulation, Plotly for charts, Streamlit for UI, and Groq for optional LLM responses.

## Runtime Flow

1. `app.py` configures Streamlit and loads environment variables.
2. `app.py` initializes the Groq client if `GROQ_API_KEY` is available.
3. `app.py` calls the cached Streamlit wrapper around `src.data_loader.load_data`.
4. Sidebar filters are applied to the loaded dataframe.
5. `src.metrics.compute_summary_metrics` calculates headline metrics.
6. `app.py` renders tabs, charts, tables, downloads, and selected-case flows.
7. `src.risk_logic` provides priority sorting, timeline steps, and Quick Insights.
8. `src.ui_components` renders stable reusable UI blocks.
9. AI Copilot calls are assembled and executed in `app.py`.

## Module Responsibilities

### `app.py`

`app.py` remains the Streamlit entrypoint and runtime orchestrator.

It owns:

- Streamlit page configuration.
- CSS injection.
- Environment and secret loading.
- Groq client initialization.
- Streamlit cache wrapper for data loading.
- Sidebar filters.
- Tab structure and chart/table layout.
- `st.session_state` behavior.
- Demo event tracking.
- LLM prompt orchestration and Groq API calls.

This file is intentionally still larger than ideal because behavior-preserving refactoring is being done in phases.

### `src/data_loader.py`

Owns dataset loading and validation.

Responsibilities:

- Define `REQUIRED_COLUMNS`.
- Return `None` if `data/transactions.csv` does not exist.
- Read CSV data with pandas.
- Validate required columns.
- Parse `order_date`.
- Drop rows with invalid dates.
- Add `risk_type_display`, mapping `Returns` to `Return Abuse`.

### `src/metrics.py`

Owns reusable metric and formatting helpers.

Responsibilities:

- `format_currency`
- `safe_top_index`
- `to_csv_download`
- `compute_summary_metrics`

The functions are intentionally small and deterministic so they are easy to unit test.

### `src/risk_logic.py`

Owns deterministic risk and case behavior that does not require Streamlit.

Responsibilities:

- `build_case_timeline`
- `get_open_high_risk_cases`
- `sort_cases_by_priority`
- `answer_risk_question`
- `answer_case_question`

This module powers Ops Queue ordering, selected-case timeline steps, and Quick Insights answers.

### `src/ui_components.py`

Owns reusable Streamlit/HTML rendering helpers.

Responsibilities:

- Hero summary rendering.
- Outcome summary note rendering.
- Selected-case detail box rendering.
- Timeline HTML rendering.
- AI Risk Insights footer rendering.

The module contains rendering code only. It should not own business logic, filters, session state, or event tracking.

### `generate_data.py`

Owns synthetic data generation.

Responsibilities:

- Generate simulated orders.
- Assign synthetic fraud and return-abuse scores.
- Assign risk levels, recommended actions, workflow stages, and case statuses.
- Write `data/transactions.csv`.

The generated data is useful for demos and tests, but it is not a validated production risk model.

### `tests/`

Current tests cover extracted logic:

- Data loading and schema handling.
- Metric calculations and formatting.
- Risk/case quick insights and priority sorting.
- UI helper rendered content.

## AI Copilot Flow

The app supports LLM-assisted workflows through Groq when `GROQ_API_KEY` is configured.

### Dashboard Copilot

1. User asks a portfolio-level question in the AI Risk Copilot tab.
2. `app.py` builds a dashboard context from the current filtered dataframe.
3. `app.py` sends the dashboard prompt and system instructions to Groq.
4. The response is displayed in Streamlit.

### Case Copilot

1. User selects an order in Ops Queue.
2. User asks a case-level question.
3. `app.py` builds selected-case context.
4. In AI Analyst mode, Groq receives the case context and question.
5. In Quick Case Insights mode, `src.risk_logic.answer_case_question` returns a deterministic response.

### Generated Artifacts

The app can generate:

- Merchant summary.
- Policy recommendation.
- Customer verification draft.
- Ops action plan.

These are generated from supplied dashboard and/or selected-case context. The app instructs the LLM not to invent facts and not to claim that actions were executed.

## Key Design Decisions

- Use phased refactoring to preserve current product behavior.
- Keep Streamlit stateful behavior in `app.py` until the module boundaries are more mature.
- Extract deterministic logic before extracting LLM orchestration.
- Keep Quick Insights available as a no-key fallback path.
- Keep UI helpers thin and rendering-focused.
- Use unit tests to protect behavior during refactoring.

## Trade-Offs

### Streamlit-Oriented Entrypoint

Streamlit makes the demo fast to build and easy to run, but the entrypoint still mixes orchestration, layout, and LLM flow. Further extraction will be needed before this resembles a layered production service.

### Simulated Data

The synthetic dataset enables realistic product demos without exposing real merchant data. The trade-off is that metrics, scores, and workflows are illustrative.

### External LLM Provider

Groq integration enables useful AI workflows quickly. The trade-off is that production deployments need privacy review, redaction, provider configuration, cost controls, and observability.

### In-Memory Session State

Streamlit session state is simple and appropriate for this demo. It is not a replacement for durable user, case, event, or audit storage.

## Security And Privacy Notes

Current risks and gaps:

- No authentication or authorization.
- No tenant isolation.
- Viewer email is collected for demo tracking.
- Demo events are stored in session state and printed to logs.
- Selected-case and dashboard context may be sent to Groq.
- No formal redaction layer before LLM calls.
- Raw filtered data can be displayed in the app.

Before production use, the app needs explicit data classification, access controls, audit policies, retention rules, and LLM data-minimization controls.

## Current Limitations

- No production database.
- No case persistence across sessions.
- No real merchant integration.
- No model evaluation workflow.
- No fraud/return policy configuration UI.
- No user or team management.
- No durable event pipeline.
- No deployment-specific secrets management beyond environment variables and Streamlit secrets.
- Limited automated UI coverage.

## Future Roadmap

### Phase 4: LLM Hardening

- Move Groq client code to `src/llm_client.py`.
- Move prompts and context builders to `src/prompts.py`.
- Add prompt/context tests.
- Add safer error handling.
- Add configurable model/provider settings.

### Phase 5: Event Tracking

- Move event helpers to `src/event_tracking.py`.
- Add structured event schemas.
- Add optional persistent event sinks.
- Avoid logging unnecessary personal data.

### Phase 6: Data And Modeling Foundation

- Add stronger schema validation.
- Add score version and threshold version fields.
- Separate observed fields from derived risk fields.
- Add model evaluation fixtures.

### Phase 7: Product And Operations

- Add persistent case storage.
- Add action state transitions.
- Add user roles and permissions.
- Add customer-specific policies.
- Add review queues, SLAs, and audit history.

### Phase 8: Production Readiness

- Add authentication and tenant isolation.
- Add deployment configuration.
- Add observability and monitoring.
- Add privacy controls for LLM context.
- Add end-to-end tests and CI.
