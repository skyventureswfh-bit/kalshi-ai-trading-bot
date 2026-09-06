"""
Kalshi REST client for market data, portfolio data, and order execution.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlencode

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from src.config.settings import settings
from src.utils.kalshi_normalization import format_count_fp, format_price_dollars
from src.utils.kalshi_auth import resolve_private_key_path
from src.utils.logging_setup import TradingLoggerMixin


class KalshiAPIError(Exception):
    """Custom exception for Kalshi API errors."""


class KalshiClient(TradingLoggerMixin):
    """Async Kalshi client using the docs-native RSA-PSS auth flow."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        private_key_path: Optional[str] = None,
        max_retries: int = 5,
        backoff_factor: float = 0.5,
        base_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or settings.api.kalshi_api_key
        self.base_url = (base_url or settings.api.kalshi_base_url).rstrip("/")
        self.private_key_path = resolve_private_key_path(private_key_path)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.private_key = None
        self._private_key_loaded = False
        self._series_cache: Dict[str, Dict[str, Any]] = {}

        self.client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )

        self.logger.info(
            "Kalshi client initialized",
            base_url=self.base_url,
            api_key_length=len(self.api_key) if self.api_key else 0,
        )

    def _load_private_key(self) -> None:
        """Load the PEM-encoded RSA private key."""
        try:
            key_path = Path(self.private_key_path)
            if not key_path.exists():
                raise KalshiAPIError(f"Private key file not found: {self.private_key_path}")

            with open(key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(f.read(), password=None)
            self._private_key_loaded = True
            self.logger.info("Private key loaded successfully", key_path=str(key_path))
        except Exception as exc:
            self.logger.error("Failed to load private key", error=str(exc))
            raise KalshiAPIError(f"Failed to load private key: {exc}") from exc

    def _ensure_auth_ready(self) -> None:
        """Validate auth configuration before hitting private Kalshi endpoints."""
        if not self.api_key:
            raise KalshiAPIError("KALSHI_API_KEY is not configured")

        if not self._private_key_loaded or self.private_key is None:
            self._load_private_key()

    def _sign_request(self, timestamp: str, method: str, path: str) -> str:
        """Sign ``timestamp + method + path`` with RSA-PSS."""
        message = (timestamp + method.upper() + path).encode("utf-8")
        try:
            signature = self.private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH,
                ),
                hashes.SHA256(),
            )
            return base64.b64encode(signature).decode("utf-8")
        except Exception as exc:
            self.logger.error("Failed to sign request", error=str(exc))
            raise KalshiAPIError(f"Failed to sign request: {exc}") from exc

    async def _make_authenticated_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        require_auth: bool = True,
    ) -> Dict[str, Any]:
        """Make a Kalshi REST request with retries and optional auth."""
        query_string = urlencode(
            {
                key: value
                for key, value in (params or {}).items()
                if value is not None and value != ""
            },
            doseq=True,
        )
        url = f"{self.base_url}{endpoint}"
        if query_string:
            url = f"{url}?{query_string}"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if require_auth:
            self._ensure_auth_ready()
            timestamp = str(int(time.time() * 1000))
            signature = self._sign_request(timestamp, method, endpoint)
            headers.update(
                {
                    "KALSHI-ACCESS-KEY": self.api_key,
                    "KALSHI-ACCESS-TIMESTAMP": timestamp,
                    "KALSHI-ACCESS-SIGNATURE": signature,
                }
            )

        body = None
        if json_data is not None:
            body = json.dumps(json_data, separators=(",", ":"))

        last_exception: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                await asyncio.sleep(0.2)
                response = await self.client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    content=body,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                last_exception = exc
                status_code = exc.response.status_code
                if status_code == 429 or status_code >= 500:
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    self.logger.warning(
                        "Kalshi API request failed and will be retried",
                        endpoint=endpoint,
                        status_code=status_code,
                        attempt=attempt + 1,
                        backoff_s=sleep_time,
                    )
                    await asyncio.sleep(sleep_time)
                    continue

                error_msg = f"HTTP {status_code}: {exc.response.text}"
                self.logger.error("Kalshi API request failed", endpoint=endpoint, error=error_msg)
                raise KalshiAPIError(error_msg) from exc
            except Exception as exc:
                last_exception = exc
                sleep_time = self.backoff_factor * (2 ** attempt)
                self.logger.warning(
                    "Kalshi API request failed with transport error",
                    endpoint=endpoint,
                    attempt=attempt + 1,
                    backoff_s=sleep_time,
                    error=str(exc),
                )
                await asyncio.sleep(sleep_time)

        raise KalshiAPIError(
            f"API request failed after {self.max_retries} retries: {last_exception}"
        )

    async def get_balance(self) -> Dict[str, Any]:
        """Get account balance."""
        return await self._make_authenticated_request("GET", "/trade-api/v2/portfolio/balance")

    async def get_positions(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """Get portfolio positions."""
        params = {"ticker": ticker} if ticker else None
        return await self._make_authenticated_request(
            "GET", "/trade-api/v2/portfolio/positions", params=params
        )

    async def get_fills(
        self,
        ticker: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get recent fill history."""
        params: Dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if cursor:
            params["cursor"] = cursor
        if order_id:
            params["order_id"] = order_id
        return await self._make_authenticated_request(
            "GET", "/trade-api/v2/portfolio/fills", params=params
        )

    async def get_orders(
        self,
        ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get current order history."""
        params: Dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        return await self._make_authenticated_request(
            "GET", "/trade-api/v2/portfolio/orders", params=params
        )

    async def get_events(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        status: Optional[str] = None,
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        with_nested_markets: bool = False,
        with_milestones: bool = False,
    ) -> Dict[str, Any]:
        """Get events, optionally including nested markets."""
        params: Dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if with_nested_markets:
            params["with_nested_markets"] = "true"
        if with_milestones:
            params["with_milestones"] = "true"

        return await self._make_authenticated_request(
            "GET", "/trade-api/v2/events", params=params, require_auth=False
        )

    async def get_markets(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        status: Optional[str] = None,
        tickers: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Get market data."""
        params: Dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        if tickers:
            params["tickers"] = ",".join(tickers)

        return await self._make_authenticated_request(
            "GET", "/trade-api/v2/markets", params=params, require_auth=False
        )

    async def get_market(self, ticker: str) -> Dict[str, Any]:
        """Get a specific market."""
        return await self._make_authenticated_request(
            "GET", f"/trade-api/v2/markets/{ticker}", require_auth=False
        )

    async def get_series(
        self,
        series_ticker: str,
        *,
        include_volume: bool = False,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        """Get metadata for a specific series, caching repeat lookups."""
        normalized_series_ticker = str(series_ticker or "").strip()
        if not normalized_series_ticker:
            return {}

        if not refresh and normalized_series_ticker in self._series_cache:
            return self._series_cache[normalized_series_ticker]

        params = {"include_volume": "true"} if include_volume else None
        response = await self._make_authenticated_request(
            "GET",
            f"/trade-api/v2/series/{normalized_series_ticker}",
            params=params,
            require_auth=False,
        )
        if isinstance(response, dict):
            self._series_cache[normalized_series_ticker] = response
        return response

    async def get_orderbook(self, ticker: str, depth: int = 100) -> Dict[str, Any]:
        """Get market orderbook."""
        return await self._make_authenticated_request(
            "GET",
            f"/trade-api/v2/markets/{ticker}/orderbook",
            params={"depth": depth},
            require_auth=False,
        )

    async def get_market_trades(
        self,
        ticker: str,
        limit: int = 100,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get recent public trades for a specific market."""
        params: Dict[str, Any] = {"ticker": ticker, "limit": limit}
        if min_ts is not None:
            params["min_ts"] = int(min_ts)
        if max_ts is not None:
            params["max_ts"] = int(max_ts)
        if cursor:
            params["cursor"] = cursor

        return await self._make_authenticated_request(
            "GET",
            "/trade-api/v2/markets/trades",
            params=params,
            require_auth=False,
        )

    async def get_historical_cutoff(self) -> Dict[str, Any]:
        """Get the Kalshi live/historical data cutoff."""
        return await self._make_authenticated_request(
            "GET", "/trade-api/v2/historical/cutoff", require_auth=False
        )

    async def get_historical_market(self, ticker: str) -> Dict[str, Any]:
        """Get historical market data once the market has crossed the cutoff."""
        return await self._make_authenticated_request(
            "GET", f"/trade-api/v2/historical/markets/{ticker}", require_auth=False
        )

    async def get_historical_orders(
        self,
        ticker: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get historical orders once they are no longer in the live window."""
        params: Dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if cursor:
            params["cursor"] = cursor
        if status:
            params["status"] = status
        return await self._make_authenticated_request(
            "GET", "/trade-api/v2/historical/orders", params=params
        )

    async def get_historical_fills(
        self,
        ticker: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get historical fills once they are no longer in the live window."""
        params: Dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if cursor:
            params["cursor"] = cursor
        if order_id:
            params["order_id"] = order_id
        return await self._make_authenticated_request(
            "GET", "/trade-api/v2/historical/fills", params=params
        )

    async def place_order(
        self,
        ticker: str,
        client_order_id: str,
        side: str,
        action: str,
        count: float,
        type_: str = "limit",
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        yes_price_dollars: Optional[float | str] = None,
        no_price_dollars: Optional[float | str] = None,
        expiration_ts: Optional[int] = None,
        time_in_force: Optional[str] = None,
        post_only: Optional[bool] = None,
        reduce_only: Optional[bool] = None,
        buy_max_cost: Optional[int] = None,
        sell_position_floor: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Place a docs-compatible limit order with a final live-buy safety gate."""
        normalized_count = format_count_fp(count)
        order_data: Dict[str, Any] = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "side": side,
            "action": action,
            "type": type_,
        }
        if normalized_count.endswith(".00"):
            order_data["count"] = int(float(normalized_count))
        else:
            order_data["count_fp"] = normalized_count

        if yes_price_dollars is not None:
            order_data["yes_price_dollars"] = format_price_dollars(yes_price_dollars)
        elif yes_price is not None:
            order_data["yes_price"] = int(yes_price)

        if no_price_dollars is not None:
            order_data["no_price_dollars"] = format_price_dollars(no_price_dollars)
        elif no_price is not None:
            order_data["no_price"] = int(no_price)

        if expiration_ts is not None:
            order_data["expiration_ts"] = int(expiration_ts)
        if time_in_force:
            order_data["time_in_force"] = time_in_force
        if post_only is not None:
            order_data["post_only"] = bool(post_only)
        if reduce_only is not None:
            order_data["reduce_only"] = bool(reduce_only)
        if buy_max_cost is not None:
            order_data["buy_max_cost"] = int(buy_max_cost)
        if sell_position_floor is not None:
            order_data["sell_position_floor"] = int(sell_position_floor)

        # Final money gate: every live BUY must still satisfy the protected
        # reserve, per-trade cap, and daily realized-loss cap immediately before
        # the exchange POST. SELL orders are intentionally exempt so safety
        # controls can never prevent reducing risk or exiting a position.
        if str(action or "").lower() == "buy":
            try:
                if str(side or "").lower() == "no":
                    if no_price_dollars is not None:
                        unit_price = float(no_price_dollars)
                    elif no_price is not None:
                        unit_price = float(no_price) / 100.0
                    else:
                        unit_price = 0.0
                else:
                    if yes_price_dollars is not None:
                        unit_price = float(yes_price_dollars)
                    elif yes_price is not None:
                        unit_price = float(yes_price) / 100.0
                    else:
                        unit_price = 0.0

                proposed_trade_value = max(0.0, float(count) * unit_price)
                if buy_max_cost is not None and buy_max_cost > 0:
                    proposed_trade_value = max(
                        proposed_trade_value,
                        float(buy_max_cost) / 100.0,
                    )

                from src.utils.cash_reserves import CashReservesManager
                from src.utils.database import DatabaseManager

                safety_db = DatabaseManager()
                reserve_manager = CashReservesManager(safety_db, self)
                reserve_check = await reserve_manager.check_cash_reserves(
                    proposed_trade_value=proposed_trade_value
                )
                if not reserve_check.can_trade:
                    self.logger.warning(
                        "Live buy blocked by final capital safety gate",
                        ticker=ticker,
                        side=side,
                        proposed_trade_value=proposed_trade_value,
                        reason=reserve_check.reason,
                    )
                    raise KalshiAPIError(
                        f"Live buy blocked by capital safety gate: {reserve_check.reason}"
                    )
            except KalshiAPIError:
                raise
            except Exception as exc:
                # This gate is fail-closed. If the safety state cannot be read,
                # do not send a new-money order to the exchange.
                self.logger.error(
                    "Final capital safety gate failed; blocking live buy",
                    ticker=ticker,
                    side=side,
                    error=str(exc),
                )
                raise KalshiAPIError(
                    f"Live buy blocked because capital safety could not be verified: {exc}"
                ) from exc

        return await self._make_authenticated_request(
            "POST", "/trade-api/v2/portfolio/orders", json_data=order_data
        )

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order."""
        return await self._make_authenticated_request(
            "DELETE", f"/trade-api/v2/portfolio/orders/{order_id}"
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.aclose()
        self.logger.info("Kalshi client closed")

    async def __aenter__(self) -> "KalshiClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()