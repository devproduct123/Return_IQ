# Risk Intelligence Integrations

Risk Intelligence now has a small integrations layer for loading merchant data from external APIs while keeping the existing demo CSV flow.

## Current Data Sources

The Streamlit sidebar has an **Advanced data options** expander with a data-source selector:

- `Demo CSV`: loads `data/transactions.csv`.
- `Shopify API`: fetches recent Shopify orders and normalizes them into the Risk Intelligence schema.
- `Saved Shopify CSV`: loads the last saved Shopify sync from `data/shopify_transactions.csv`.

`Demo CSV` remains the default so the live merchant integration choices stay out of the way during a presentation.

When using `Shopify API`, the sidebar includes **Save Shopify Sync to CSV**. This writes
the normalized dataframe to:

```text
data/shopify_transactions.csv
```

That file is intentionally git-ignored because it can contain merchant/order data.

## Shopify API Setup

To use Shopify API mode, add these values to Streamlit secrets or local environment variables:

```toml
SHOPIFY_SHOP_DOMAIN = "your-store.myshopify.com"
SHOPIFY_ACCESS_TOKEN = "shpat_..."
```

The connector uses Shopify Admin API orders:

```text
GET /admin/api/2024-10/orders.json
```

The app currently fetches recent orders, normalizes them, and applies a basic demo risk classification:

- Cancelled orders become `Fraud Risk`.
- Refunded orders become `Return Risk`.
- Other orders become low-risk `None`.

This is intentionally conservative demo logic. Production usage should replace this with real merchant risk rules, chargeback data, return events, and policy configuration.

## Code Structure

```text
src/integrations/
  base.py       # Common connector interface
  shopify.py    # Shopify Admin API connector

src/data_loader.py
  load_csv_data()
  normalize_shopify_orders()
  load_connector_data()
  load_shopify_data()
  save_synced_data()
  load_data()
```

Every source is normalized to the same dashboard schema, so `app.py`, metrics, filters, tables, and Copilot flows can keep using one dataframe shape.

## Future Integrations

Good next connectors:

- Stripe for payments, disputes, and chargebacks.
- Gorgias/Zendesk for support conversations.
- ReturnLogic/Loop/Narvar for returns.
- Recharge for subscription signals.

Production connectors should add OAuth, tenant isolation, background sync jobs, retry/backoff, rate-limit handling, encrypted secrets, audit logging, and disconnect/delete flows.
