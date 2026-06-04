# Aignita Risk Intelligence

Aignita Risk Intelligence is a Streamlit demo application for AI-assisted e-commerce risk operations. It combines fraud risk, return-abuse risk, operational case workflow, dashboard metrics, CSV exports, and an LLM-powered copilot experience for merchant and analyst workflows.

The current app is a product foundation and demo, not a fully production-ready risk platform. It uses simulated transaction data and local Streamlit session state to demonstrate the core experience.

## Product Purpose

Risk Intelligence is designed to help e-commerce merchants and risk teams:

- Monitor fraud and return-abuse exposure across orders.
- Identify high-priority cases for operational review.
- Understand why orders were flagged.
- Generate analyst-facing and merchant-facing risk summaries.
- Explore risk drivers through a dashboard and quick insight prompts.

The app focuses on a closed-loop workflow: historical outcome analysis, live risk prioritization, selected-case review, structured operations feedback, model-learning backlog capture, and an AI Copilot.

## Current Features

- Historical Intelligence mode with actual return and fraud outcomes.
- Return and fraud driver charts with Level 2 taxonomy drilldowns.
- Live Risk Queue mode with predicted exposure and prioritized cases.
- Review Queue with selected-case details, timeline, structured actions, final labels, override reasons, and analyst confidence.
- Historical Model Learning view for analytics guidance and future training signals.
- Live Review Queue feedback capture for case-level model improvement.
- Quick Insights mode for deterministic, rule-based answers.
- AI Copilot mode using Groq when `GROQ_API_KEY` is configured.
- Demo reset button and seeded live feedback examples for repeatable presentations.
- Optional Shopify API and saved-sync integration modes under advanced data options.
- CSV downloads for open cases and filtered datasets.
- Demo event tracking stored in Streamlit session state.

## Project Structure

```text
.
├── app.py
├── generate_data.py
├── requirements.txt
├── data/
│   ├── transactions.csv
│   ├── live_transactions.csv
│   └── live_transactions_demo_baseline.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── metrics.py
│   ├── risk_logic.py
│   └── ui_components.py
└── tests/
    ├── test_data_loader.py
    ├── test_metrics.py
    ├── test_risk_logic.py
    └── test_ui_components.py
```

## Module Responsibilities

- `app.py`: Streamlit entrypoint, page layout, tab wiring, filters, session state, event tracking, and LLM orchestration.
- `src/data_loader.py`: required dataset schema, CSV loading, date parsing, and derived display fields.
- `src/metrics.py`: reusable metric calculations and formatting helpers.
- `src/risk_logic.py`: case timeline logic, priority sorting, fallback selected-case logic, and deterministic quick insight answers.
- `src/ui_components.py`: reusable Streamlit/HTML rendering helpers for stable UI blocks.
- `generate_data.py`: synthetic dataset generation for demo data.
- `tests/`: unit tests for extracted data, metrics, risk logic, and UI rendering helpers.

## Setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the app:

```bash
python3 -m streamlit run app.py
```

Run tests:

```bash
python3 -m pytest
```

Optionally configure Groq:

```bash
export GROQ_API_KEY="your-key"
```

Without `GROQ_API_KEY`, the app still works with Quick Insights, but AI Copilot responses that require Groq will show a configuration message.

## Data

The app reads:

- `data/transactions.csv`: completed historical outcomes from `2025-01-01` to `2025-12-31`.
- `data/live_transactions.csv`: January 2026 live-risk snapshot with mostly pending outcomes and a small set of seeded operations feedback examples.
- `data/live_transactions_demo_baseline.csv`: resettable copy of the live demo snapshot.

If the files are missing or need to be refreshed, regenerate demo data:

```bash
python3 generate_data.py
```

The dataset is simulated and should not be treated as a trained model output, production fraud data, or evidence of real-world risk performance.

Use **Reset Demo State** in the sidebar to restore the seeded live snapshot and clear local model-learning backlog entries after a presentation.

## Suggested Demo Flow

1. Open **Historical Intelligence** and show realized loss.
2. Drill into return and fraud Level 1 and Level 2 drivers.
3. Add one historical analytics insight to the Model Learning backlog.
4. Switch to **Live Risk Queue** and open a high-priority case.
5. Record an operations action, final learning label, reason, case insight, and analyst confidence.
6. Open **Model feedback from reviewed cases** in the Review Queue to show the reviewed case entering the feedback queue.
7. Finish in **Risk Copilot** with deterministic Quick Insights or the optional AI Copilot.

## AI Copilot Flow

Risk Intelligence has two answer modes:

- Quick Insights: deterministic pandas-based responses from `src/risk_logic.py`.
- AI Copilot / AI Analyst: prompt-and-context responses through Groq.

For Risk Copilot flows, `app.py` builds dashboard context from the current filtered queue, sends it to the configured Groq model, and displays the response. Case-level AI Analyst flows stay inside Review Queue and use only the selected order context. The prompts instruct the model to use only supplied context, stay concise, avoid invented facts, and avoid claiming that actions were executed.

## Design Decisions

- Keep `app.py` as the Streamlit entrypoint so page behavior remains easy to follow.
- Extract pure data, metric, and risk helpers first to make the app more testable.
- Keep Streamlit `session_state`, event tracking, and LLM calls in `app.py` for now to avoid changing runtime behavior during refactoring.
- Preserve current labels, tabs, charts, and user-facing outputs during modularization.
- Use deterministic Quick Insights as a reliable fallback when LLM access is unavailable.

## Trade-Offs

- The app is still Streamlit-first, so UI and orchestration remain coupled in `app.py`.
- The LLM client and prompt construction have not yet been extracted into dedicated modules.
- Event tracking is session-local and printed to logs, not persisted in a production analytics system.
- The scoring and recommendations are based on synthetic data rules, not a validated production risk model.
- Tests cover extracted logic, but there is not yet an end-to-end Streamlit smoke test.

## Current Limitations

- Simulated data only.
- No authentication, authorization, or tenant isolation.
- No persistent case management database.
- No real-time ingestion from commerce platforms or payment processors.
- No model versioning, threshold governance, or monitoring.
- No production privacy controls around LLM context redaction.
- No formal audit trail beyond demo session events.

## Roadmap

Near-term:

- Extract LLM client and prompt templates into dedicated modules.
- Add tests for prompt/context construction.
- Add Streamlit smoke tests for key pages.
- Separate runtime and development dependencies.

Medium-term:

- Add persistent storage for cases, actions, and events.
- Introduce user authentication and role-based access.
- Add stronger data validation and schema contracts.
- Add configurable model/provider settings.
- Add redaction and data minimization before LLM calls.

Longer-term:

- Integrate with real commerce, payment, return, and support systems.
- Add model versioning, threshold management, and evaluation workflows.
- Add production observability, audit logging, and governance controls.
- Support multi-tenant deployments and customer-specific policies.
