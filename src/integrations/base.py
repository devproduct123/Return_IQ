from abc import ABC, abstractmethod
from typing import Any


class CommerceConnector(ABC):
    """Common interface for commerce platform connectors."""

    @abstractmethod
    def fetch_orders(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return raw order records from the external system."""
        raise NotImplementedError

