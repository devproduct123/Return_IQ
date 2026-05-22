# app.py Code Walkthrough

This document explains the current `app.py` file section by section. The app is a Streamlit dashboard for Aignita RiskIQ, an AI-assisted e-commerce fraud, return-abuse, and risk-operations demo.

At a high level, `app.py` does five jobs:

- Loads dependencies, helper functions, data, and optional Groq configuration.
- Defines the Streamlit page styling and shared Plotly chart styling.
- Builds dashboard context for Risk Copilot and selected-case context for Review Queue AI prompts.
- Applies sidebar filters and computes metrics for the filtered dataset.
- Renders the six main product tabs: Overview, Fraud Risk, Return Abuse, Recommended Actions, Review Queue, and Risk Copilot.

## Imports

The file starts by importing standard Python utilities:

- `os` reads environment variables.
- `json` serializes demo-tracking events before printing them.
- `uuid` creates anonymous session IDs.
- `datetime` and `timezone` timestamp demo events in UTC.

It then imports third-party app libraries:

- `pandas` powers dataframe filtering, grouping, aggregation, and table formatting.
- `plotly.express` creates the charts.
- `streamlit` renders the web app.
- `streamlit.components.v1` embeds custom HTML, currently used for the case timeline.
- `Groq` is the optional LLM client.
- `load_dotenv` loads local environment variables from `.env`.

Finally, it imports local project helpers:

- `src.data_loader.load_data` loads `data/transactions.csv`.
- `src.metrics` provides metric calculation, currency formatting, safe top-value selection, and CSV export formatting.
- `src.risk_logic` provides deterministic case sorting, timeline generation, and Quick Insights answers.
- `src.ui_components` renders reusable visual pieces such as the page header, hero summary, case detail box, AI answer blocks, and timeline HTML.

## Streamlit Page Setup

```python
st.set_page_config(page_title="Aignita RiskIQ", layout="wide")
```

This configures the browser tab title and tells Streamlit to use a wide dashboard layout. The wide layout is important because the app uses multi-column metric rows, charts, and large dataframes.

## CSS Design System

The large `st.markdown("""<style>...</style>""", unsafe_allow_html=True)` block injects custom CSS into the Streamlit app.

This is the app's visual design layer. It styles:

- The main content width and page spacing.
- Metric cards.
- Tab controls.
- Sidebar background and border.
- Buttons and download buttons.
- Page header.
- Summary cards.
- Case detail cards.
- Demo banner.
- AI answer and generated-output blocks.
- Risk/status badges.

The CSS uses Streamlit's generated `data-testid` selectors for some native components like metrics and the sidebar. This is common in Streamlit apps, but it also means the styling depends on Streamlit's DOM structure staying mostly stable.

The CSS classes beginning with `riq-` are used by `src/ui_components.py`. For example:

- `.riq-page-header` styles `render_page_header`.
- `.riq-hero` styles `render_hero_summary`.
- `.riq-detail` styles `render_case_detail_box`.
- `.riq-answer` styles AI/case answers.
- `.riq-output` styles generated artifacts.

The `.badge-*` classes provide visual tags for priority, risk type, and case status.

## Chart Styling

The chart styling section defines shared Plotly colors and a `style_chart` helper.

```python
CHART_COLORS = [...]
PRIORITY_COLORS = {...}
RISK_TYPE_COLORS = {...}
```

These constants keep chart colors consistent:

- Priority colors map `Critical`, `High`, `Moderate`, and `Low` to red/orange/green tones.
- Risk type colors map `Fraud` and `Return Abuse` to distinct colors.
- `CHART_COLORS` is the default color sequence for other charts.

`style_chart(fig, height=320, showlegend=None, x_tickangle=0)` applies the shared chart system to every Plotly figure. It standardizes:

- Chart height.
- Background color.
- Font family and color.
- Title size and spacing.
- Left, right, top, and bottom margins.
- X-axis and Y-axis gridlines.
- Axis font sizes.
- Legend placement.
- Bar gap.
- Marker border removal.

The optional arguments let each chart adjust only the parts that differ:

- `showlegend=False` hides redundant legends when bar color already matches the x-axis category.
- `showlegend=True` keeps legends for grouped charts where color represents another field.
- `x_tickangle` rotates long category labels so chart axes stay aligned and readable.

## Bootstrap And Environment Setup

```python
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
```

This loads the optional Groq API key from either:

- A local `.env` file.
- Streamlit secrets.

If the key exists, the app creates a Groq client. If not, the app still works, but AI Copilot mode returns a configuration message and Quick Insights remains available.

## Cached Data Loading

```python
@st.cache_data
def load_data():
    return load_transactions_data()
```

This wraps the project data loader in Streamlit's cache. Streamlit reruns the script often, so caching prevents the CSV from being reloaded on every UI interaction.

The actual dataset loading and validation logic lives in `src/data_loader.py`.

## Session And Demo Event Tracking

The next helper functions manage lightweight demo analytics:

- `get_session_id()` creates or returns a UUID stored in `st.session_state`.
- `get_user_id()` returns the viewer email if entered, otherwise `"anonymous"`.
- `log_event(event_name, details=None)` creates an event record, stores it in session state, and prints it as JSON.
- `log_once(key, event_name, details=None)` logs an event only once per browser session.
- `log_value_change(state_key, event_name, current_value)` logs when filters, selected case, or similar values change.

These functions are not production analytics. They are useful for demos because they show how a user interacted with the app during a single Streamlit session.

## Dashboard Context Builder

```python
def build_dashboard_context(data: pd.DataFrame) -> str:
```

This function converts the current filtered dataframe into a plain-text summary for the AI Copilot.

It calculates:

- Total orders.
- Number and percentage of high-risk flagged orders.
- Revenue at risk.
- Protected revenue from closed high-risk cases.
- Open cases.
- Critical cases.
- Fraud cases.
- Return-abuse cases.
- Top return-abuse categories.
- Top fraud states.
- Most common recommended actions.

The output is a text block, not JSON. It is intentionally readable by an LLM and easy to inspect during debugging.

If the dataframe is empty, it returns `"No dashboard data available."`

## Selected Case Context Builder

```python
def build_case_context(case_row: pd.Series | None) -> str:
```

This function converts one selected order case into a plain-text case summary for AI prompts.

It includes:

- Order ID and customer ID.
- Risk type.
- Priority tier.
- Case status.
- Workflow stage.
- Order value.
- Fraud score.
- Return-abuse score.
- Unified risk score.
- Past returns.
- Chargeback history.
- Risk explanation.
- Recommended action.
- AI suggested next step.

If no case is selected, it returns `"No selected case context."`

## Groq Request Helper

```python
def ask_groq(prompt: str, system_prompt: str, max_tokens: int = 600) -> str:
```

This is the low-level LLM call wrapper.

It first checks whether `groq_client` exists. If no API key is configured, it returns a user-facing configuration message instead of crashing.

If Groq is configured, it sends:

- A system prompt.
- A user prompt.
- A low temperature of `0.2` for more stable, practical answers.
- A max token limit controlled by the caller.

Any exception is caught and returned as `"Groq request failed: ..."` so the UI can display the failure cleanly.

## Dashboard Copilot Function

```python
def ask_risk_copilot(question, dashboard_data, selected_case=None) -> str:
```

This function powers the broader Risk Copilot tab.

It builds:

- Dashboard context from the current filtered dataset.
- Case context from the selected case, if one is available.

Then it creates a system prompt that frames the model as an e-commerce risk analyst. The rules tell the model to:

- Use only supplied context.
- Stay concise and practical.
- Prioritize open critical cases.
- Avoid invented facts.

The final prompt combines dashboard context, case context, and the merchant question.

## Case Copilot Function

```python
def ask_case_copilot(question, selected_case) -> str:
```

This function powers AI Analyst mode inside the Review Queue selected-case workflow.

It builds selected-case context and asks the LLM to answer as a case analyst assistant. The prompt is narrower than the dashboard Copilot prompt. It focuses on:

- Understanding the selected case.
- Explaining what matters operationally.
- Suggesting what the analyst should do next.
- Avoiding invented evidence.

## AI Artifact Generator

```python
def generate_ai_output(artifact_type, dashboard_data, selected_case=None) -> str:
```

This function creates longer generated outputs used by two areas:

- Case Action Assistant in Review Queue.
- Strategic Outputs in Risk Copilot.

It supports four artifact types:

- `merchant_summary`
- `policy_recommendation`
- `customer_verification_draft`
- `ops_action_plan`

Each artifact type maps to a task-specific prompt. The shared system prompt tells the model to produce practical, grounded, concise outputs and not to claim that actions were executed.

## Data Load And Initial Page Render

The app loads the dataset:

```python
try:
    df = load_data()
except Exception as e:
    st.error(...)
    st.stop()
```

If loading raises an exception, the app shows an error and stops.

Then it renders the page header:

```python
render_page_header()
```

It logs the session start once, renders a sidebar text input for viewer email, and checks whether the dataset exists.

If `df` is `None`, the app shows:

- An error that `data/transactions.csv` was not found.
- The command to regenerate data: `python generate_data.py`.

Then it renders the demo banner.

## Sidebar Filters

The sidebar contains all global filters. These filters affect every tab and AI output.

The app creates filters for:

- Order date range.
- Product category.
- Payment method.
- State.
- Priority tier.
- Recommended action.
- Workflow stage.
- Case status.

Each multiselect defaults to all available values, so the initial dashboard shows the full dataset.

The date widget may return either a two-value tuple or something else while the user is interacting with it. The code handles that by falling back to the full date range if the selected date range is incomplete.

## Filter Change Logging

After creating the filter widgets, the app stores the current filter state in a dictionary:

```python
current_filters = {...}
log_value_change("last_filters", "filters_changed", current_filters)
```

This records filter changes in the session event log.

## Filtered Dataset Creation

The app creates `filtered_df` by applying all sidebar filters to `df`.

The filter conditions include:

- State is in `state_filter`.
- Product category is in `category_filter`.
- Payment method is in `payment_filter`.
- Recommended action is in `action_filter`.
- Priority tier is in `priority_filter`.
- Case status is in `case_status_filter`.
- Workflow stage is in `workflow_filter`.
- Order date is between the selected start and end dates.

The `.copy()` call creates an independent dataframe for downstream operations.

If no rows match the filters, the app shows a warning and stops.

## Top-Level Metrics And Summary

The app computes summary metrics:

```python
metrics = compute_summary_metrics(filtered_df)
```

It also calculates:

- `top_category`: product category with the highest average return-abuse score.
- `top_state`: state with the highest average fraud score.

Then it renders:

- The hero summary with `render_hero_summary`.
- Six top metric cards:
  - Total Orders
  - Flagged Orders
  - Flagged %
  - Fraud Cases
  - Return Abuse Cases
  - Revenue at Risk

These metrics provide the dashboard's high-level operational readout before the user enters a specific tab.

## Tab Creation

The app creates six tabs:

```python
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([...])
```

The tabs are:

- Overview
- Fraud Risk
- Return Abuse
- Recommended Actions
- Review Queue
- Risk Copilot

Each tab logs a `tab_view` event the first time it is seen.

## Tab 1: Overview

The Overview tab is the executive dashboard.

It contains four main chart sections.

### Flagged Order Rate Over Time

The app groups filtered orders by day and calculates:

- Total orders per day.
- High-risk flagged orders per day.
- Flagged rate percentage.

It renders a line chart showing how the flagged-order rate changes over time.

### Loss Exposure By Risk Type

The app filters to fraud and return-abuse rows, groups by risk type, and sums order value.

It renders a bar chart showing loss exposure by:

- Fraud
- Return Abuse

The chart uses `RISK_TYPE_COLORS` and hides the legend because the x-axis already identifies each risk type.

### Flagged Cases By Risk Exposure Priority

The app counts high-risk cases by priority tier and explicitly orders the categories:

- Critical
- High
- Moderate

It renders a priority-colored bar chart and hides the legend because color duplicates the x-axis meaning.

### Recommended Actions

The app counts recommended actions across the filtered dataset.

It renders a bar chart with slightly rotated x-axis labels because recommended-action names are longer than priority labels.

### Workflow Stage Split

The app counts workflow stages for high-risk orders and renders another bar chart.

This helps show where high-risk work is sitting operationally.

## Tab 2: Fraud Risk

Fraud Risk focuses only on rows where `risk_type_display == "Fraud"`.

If no fraud rows exist under the current filters, the tab shows a warning.

Otherwise it renders four metrics:

- Fraud Cases
- Fraud Risk Revenue
- Top Fraud State
- Top Fraud Payment Method

Then it renders two charts:

- Average Fraud Score by State
- Average Fraud Score by Payment Method

Finally, it displays the top 50 fraud rows sorted by fraud score and order value. The table includes identifiers, scores, priority, status, workflow stage, risk reason, recommended action, and AI next step.

## Tab 3: Return Abuse

Return Abuse focuses only on rows where `risk_type_display == "Return Abuse"`.

If no return-abuse rows exist under the current filters, the tab shows a warning.

Otherwise it renders four metrics:

- Return Abuse Cases
- Return Risk Revenue
- Top Return Category
- Top Return Product

Then it renders two charts:

- Return Abuse Score by Category
- Top Products by Return Abuse Score

The product chart is limited to the top 10 products so it stays readable.

Finally, it displays the top 50 return-abuse rows sorted by return-abuse score and order value.

## Tab 4: Recommended Actions

Recommended Actions focuses on operational work and outcomes.

It first creates:

- `open_flagged_cases`: high-risk cases with `case_status == "Open"`.
- `closed_flagged_cases`: high-risk cases with `case_status == "Closed"`.

It calculates:

- Protected revenue from closed flagged cases.
- Closure rate across open and closed flagged cases.

Then it renders four metrics:

- Open Flagged Cases
- Closed Flagged Cases
- Protected Revenue
- Closure Rate

The `render_outcome_summary` helper adds a compact written summary.

### Action Summary

The tab groups the filtered dataframe by:

- Recommended action.
- Action owner.
- Automation level.
- Priority tier.
- Case status.

For each group, it calculates:

- Number of orders.
- Total revenue.
- Average unified risk score.

Then it renders:

- Action Volume by Recommendation, colored by automation level.
- Action Ownership by Priority, colored by priority tier.

These charts keep legends because color adds information beyond the x-axis.

### AI Workflow Recommendations Table

The grouped `action_summary` dataframe is shown as a table with formatted revenue and average risk score.

### Sample Action Queue

This table shows the top 50 filtered rows sorted by unified risk score and order value. It is a broader action list, not limited only to open high-risk cases.

### Resolved Flagged Cases

If there are closed flagged cases, the app shows the top 50 resolved cases sorted by unified risk score and order value.

If there are none, it shows an informational message.

## Tab 5: Review Queue

Review Queue is the main analyst workbench for open high-risk cases.

It starts with:

```python
flagged_df = sort_cases_by_priority(get_open_high_risk_cases(filtered_df))
```

This means the queue includes only open high-risk cases from the current filters, sorted by priority.

The tab renders two metrics:

- Open High-Risk Cases
- Open Revenue at Risk

### CSV Downloads

The tab provides two downloads:

- Open Cases CSV
- Filtered Dataset CSV

When either download button is clicked, the app logs a `download_clicked` event.

### Open Case Table

If there are open cases, the app shows up to 100 cases in a dataframe. It includes case identifiers, product and location fields, payment method, scores, risk type, priority, status, workflow stage, explanation, recommended action, owner, and automation level.

Numeric fields are formatted for currency and decimal scores.

### Selected Case

The user selects an order ID with a dropdown.

The app logs selected-case changes and stores the selected case in:

```python
st.session_state["selected_case"]
```

This session-state value powers the case-level analyst tools in Review Queue.

The selected case area renders:

- Four metric cards: order value, unified risk score, priority, case status.
- A detailed case summary via `render_case_detail_box`.
- Five key risk signal metrics: fraud score, return-abuse score, unified risk score, past returns, chargeback history.
- A timeline generated by `build_case_timeline` and rendered with `components.html`.

### Ask About This Case

This section lets analysts ask questions about the selected case.

There are six prompt buttons that prefill `st.session_state["case_question"]`:

- Why was this case flagged?
- What should I do next?
- Is verification needed?
- Summarize this case.
- Explain fraud signals.
- Explain return-abuse signals.

The user can also type a custom question.

The response mode can be:

- `AI Analyst`: calls Groq through `ask_case_copilot`.
- `Quick Case Insights`: calls deterministic logic in `answer_case_question`.

If no Groq API key is configured, the app explains that Quick Case Insights still works.

When a question is submitted, the app logs a `case_question_submitted` event. The response is displayed with `render_ai_answer`.

### Case Action Assistant

This section generates selected-case artifacts:

- Draft Customer Verification Message.
- Create Ops Action Plan.

Each button logs a `case_artifact_generated` event, calls `generate_ai_output`, stores the result in `st.session_state["case_output"]`, and displays it with `render_ai_output`.

## Tab 6: Risk Copilot

The Risk Copilot tab handles broader portfolio-level questions and strategic outputs.

It starts with sample question buttons that set `st.session_state["copilot_question"]`:

- Summarize current risk.
- What should ops review first?
- Top fraud locations.
- Return abuse drivers.
- Recommended actions.
- Revenue at risk.

The user can also enter a custom question.

The response mode can be:

- `AI Copilot`: calls Groq through `ask_risk_copilot`.
- `Quick Insights`: calls deterministic logic in `answer_risk_question`.

If no Groq API key is configured, the app explains that Quick Insights still works.

### Portfolio Scope

Risk Copilot answers from the filtered dashboard context only. Selected-order investigation stays in Review Queue through the case copilot and templates.

### Copilot Answer

When the user asks a question:

- The app logs `copilot_question_submitted`.
- It either calls `ask_risk_copilot` or `answer_risk_question`.
- The answer is displayed with `render_ai_answer`.

### Strategic Outputs

The tab can generate:

- Merchant Summary.
- Policy Recommendation.

Each button logs a `strategic_artifact_generated` event, calls `generate_ai_output`, stores the output in `st.session_state["strategic_output"]`, and displays it with `render_ai_output`.

## Footer And Expanders

After the six tabs, the app renders:

```python
render_insight_footer(metrics, top_category, top_state)
```

This displays a concise AI Risk Insights section and a pilot note.

Then there are two expanders:

- `Show raw filtered data`: displays the first 200 rows from `filtered_df`.
- `Show event log (demo tracking)`: displays events stored in `st.session_state["event_log"]`.

The event log is session-local. It is useful for demo observability but is not persisted after the session ends.

## Data Flow Summary

The main runtime flow is:

1. Configure Streamlit and CSS.
2. Configure chart styles.
3. Load environment variables and optional Groq client.
4. Load transaction data through cached `load_data`.
5. Render the page header and demo banner.
6. Build sidebar filters.
7. Apply filters to create `filtered_df`.
8. Compute metrics from `filtered_df`.
9. Render top-level summary and metric cards.
10. Render each tab from the same filtered dataset.
11. Store selected case and generated outputs in `st.session_state`.
12. Use deterministic Quick Insights or optional Groq calls for question-answer flows.
13. Render footer, raw filtered data, and demo event log.

## Important State Values

The app uses `st.session_state` for several cross-rerun values:

- `session_id`: unique browser-session ID.
- `viewer_email`: optional user-entered email.
- `event_log`: list of demo events.
- `last_filters`: previous filter set used for change logging.
- `last_selected_order_id`: previous selected case used for change logging.
- `selected_case`: selected Review Queue case used by case-level analyst tools.
- `case_question`: prefilled or typed case question.
- `last_case_question_logged`: prevents duplicate case-question logs.
- `case_output`: generated selected-case artifact.
- `copilot_question`: prefilled or typed portfolio question.
- `last_copilot_question_logged`: prevents duplicate Copilot-question logs.
- `strategic_output`: generated merchant or policy artifact.

## Where Business Logic Lives

`app.py` still owns a lot of orchestration and rendering, but important business logic is already extracted:

- Dataset loading and validation: `src/data_loader.py`.
- Metric calculations and formatting: `src/metrics.py`.
- Case sorting, timeline logic, and deterministic answers: `src/risk_logic.py`.
- Reusable HTML/CSS UI blocks: `src/ui_components.py`.

This separation keeps the app easier to test and makes future refactoring safer.

## Production Readiness Notes

This app is still a demo foundation, not a production risk platform.

Important gaps include:

- No authentication or authorization.
- No tenant isolation.
- No persistent case database.
- No durable event/audit log.
- No real commerce/payment integration.
- Simulated transaction data.
- Optional LLM calls may send filtered dashboard context or selected-case context to Groq, depending on whether the user is in Risk Copilot or Review Queue.

Before production use, the app would need privacy review, redaction, access controls, persistent storage, observability, and model/provider governance.
