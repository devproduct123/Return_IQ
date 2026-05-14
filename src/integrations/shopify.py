import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.integrations.base import CommerceConnector


class ShopifyConnector(CommerceConnector):
    """Minimal Shopify Admin API connector for recent order syncs."""

    def __init__(self, shop_domain: str, access_token: str, api_version: str = "2024-10"):
        self.shop_domain = shop_domain.replace("https://", "").replace("http://", "").strip("/")
        self.access_token = access_token
        self.api_version = api_version

    def fetch_orders(self, limit: int = 100) -> list[dict[str, Any]]:
        params = urlencode({
            "limit": max(1, min(int(limit), 250)),
            "status": "any",
            "fields": (
                "id,name,created_at,total_price,currency,customer,line_items,"
                "shipping_address,billing_address,payment_gateway_names,refunds,"
                "fulfillment_status,cancelled_at"
            ),
        })
        url = (
            f"https://{self.shop_domain}/admin/api/{self.api_version}/orders.json"
            f"?{params}"
        )
        request = Request(
            url,
            headers={
                "X-Shopify-Access-Token": self.access_token,
                "Accept": "application/json",
            },
        )

        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))

        return payload.get("orders", [])

