"""
Python-side crypto data adapter for the live-trade agent loop.

Primary sources:
- CoinGecko spot price + 24h context.
- CoinGecko 1-day market chart.
- Binance public futures funding rate.

Fallback sources:
- Coinbase Exchange public ticker for spot price.
- Coinbase Exchange public candles for 5-minute bars.

Funding is useful enrichment, but a geographically blocked futures endpoint
must not invalidate otherwise healthy spot + chart pricing. The adapter still
fails closed when primary pricing itself is unavailable.
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

import httpx

from src.utils.logging_setup import TradingLoggerMixin

SOURCE_NAME = "coingecko+coinbase+binance.futures"
CATEGORY = "crypto"
DEFAULT_TIMEOUT_SECONDS = 3.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF = 0.25
DEFAULT_CACHE_TTL = 20.0

ASSET_REGISTRY: Dict[str, Dict[str, str]] = {
    "BTC": {
        "coingecko_id": "bitcoin",
        "binance_symbol": "BTCUSDT",
        "coinbase_product": "BTC-USD",
        "display_name": "Bitcoin",
    },
    "ETH": {
        "coingecko_id": "ethereum",
        "binance_symbol": "ETHUSDT",
        "coinbase_product": "ETH-USD",
        "display_name": "Ethereum",
    },
    "SOL": {
        "coingecko_id": "solana",
        "binance_symbol": "SOLUSDT",
        "coinbase_product": "SOL-USD",
        "display_name": "Solana",
    },
    "XRP": {
        "coingecko_id": "ripple",
        "binance_symbol": "XRPUSDT",
        "coinbase_product": "XRP-USD",
        "display_name": "XRP",
    },
    "DOGE": {
        "coingecko_id": "dogecoin",
        "binance_symbol": "DOGEUSDT",
        "coinbase_product": "DOGE-USD",
        "display_name": "Dogecoin",
    },
}


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _iso_utc(now: Optional[datetime] = None) -> str:
    moment = now or datetime.now(timezone.utc)
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


class CryptoAdapter(TradingLoggerMixin):
    """Resilient crypto enrichment for Kalshi live-trade research."""

    COINGECKO_BASE = "https://api.coingecko.com/api/v3"
    COINBASE_BASE = "https://api.exchange.coinbase.com"
    BINANCE_FUTURES_BASE = "https://fapi.binance.com"

    def __init__(
        self,
        *,
        http_client: Optional[httpx.AsyncClient] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL,
    ) -> None:
        self._owns_client = http_client is None
        self.http_client = http_client or httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={"User-Agent": "kalshi-ai-trading-bot/2.0 (crypto-adapter)"},
        )
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))
        self.cache_ttl_seconds = cache_ttl_seconds
        self._spot_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._chart_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._funding_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    async def aclose(self) -> None:
        if self._owns_client:
            await self.http_client.aclose()

    async def fetch_context(self, market: Mapping[str, Any]) -> Dict[str, Any]:
        start = time.monotonic()
        payload: Dict[str, Any] = {
            "category": CATEGORY,
            "timestamp_utc": _iso_utc(),
            "signals": {},
            "freshness_seconds": 0,
            "source": SOURCE_NAME,
            "error": None,
        }

        symbol = self._detect_asset(market)
        if symbol is None:
            payload["error"] = "unknown_crypto_asset"
            payload["freshness_seconds"] = int(time.monotonic() - start)
            return payload

        registry = ASSET_REGISTRY[symbol]
        spot_task = self._spot_with_fallback(
            registry["coingecko_id"], registry["coinbase_product"]
        )
        chart_task = self._bars_with_fallback(
            registry["coingecko_id"], registry["coinbase_product"]
        )
        funding_task = self._funding(registry["binance_symbol"])

        spot_result, chart_result, funding_result = await asyncio.gather(
            spot_task, chart_task, funding_task, return_exceptions=True
        )

        signals: Dict[str, Any] = {
            "asset": symbol,
            "display_name": registry["display_name"],
        }
        pricing_errors: List[str] = []
        warnings: List[str] = []

        if isinstance(spot_result, Exception):
            self.logger.warning(
                "crypto spot fetch failed", asset=symbol, error=str(spot_result)
            )
            pricing_errors.append(f"spot:{spot_result.__class__.__name__}")
        else:
            signals["spot"] = spot_result

        if isinstance(chart_result, Exception):
            self.logger.warning(
                "crypto chart fetch failed", asset=symbol, error=str(chart_result)
            )
            pricing_errors.append(f"bars:{chart_result.__class__.__name__}")
        else:
            signals["bars_1m"] = chart_result.get("bars_1m", [])
            signals["bars_5m"] = chart_result.get("bars_5m", [])

        if isinstance(funding_result, Exception):
            self.logger.warning(
                "crypto funding fetch failed", asset=symbol, error=str(funding_result)
            )
            warnings.append(f"funding:{funding_result.__class__.__name__}")
            signals["funding"] = {
                "symbol": registry["binance_symbol"],
                "available": False,
            }
        else:
            funding_result = dict(funding_result)
            funding_result["available"] = True
            signals["funding"] = funding_result

        if warnings:
            signals["warnings"] = warnings

        payload["signals"] = signals
        # Spot and bars are the pricing-critical inputs. Funding is optional
        # enrichment and must not turn healthy price context into a hard error.
        if pricing_errors:
            payload["error"] = ";".join(pricing_errors)
        payload["freshness_seconds"] = int(time.monotonic() - start)
        return payload

    @staticmethod
    def _detect_asset(market: Mapping[str, Any]) -> Optional[str]:
        blob_parts: List[str] = []
        for key in ("ticker", "event_ticker", "series_ticker"):
            value = market.get(key)
            if value:
                blob_parts.append(str(value).upper())
        for key in ("title", "sub_title", "yes_sub_title"):
            value = market.get(key)
            if value:
                blob_parts.append(str(value))
        blob = " ".join(blob_parts)
        if not blob:
            return None
        upper = blob.upper()

        for symbol in ASSET_REGISTRY:
            if re.search(rf"\bKX{symbol}", upper):
                return symbol

        lower = blob.lower()
        keyword_map = {
            "BTC": ("bitcoin", "btc"),
            "ETH": ("ethereum", "ether", "eth"),
            "SOL": ("solana", "sol"),
            "XRP": ("ripple", "xrp"),
            "DOGE": ("dogecoin", "doge"),
        }
        for symbol, needles in keyword_map.items():
            for needle in needles:
                if re.search(rf"\b{re.escape(needle)}\b", lower):
                    return symbol
        return None

    async def _spot_with_fallback(
        self, coingecko_id: str, coinbase_product: str
    ) -> Dict[str, Any]:
        try:
            return await self._spot(coingecko_id)
        except Exception as primary_error:
            self.logger.warning(
                "coingecko spot unavailable; trying coinbase fallback",
                asset=coingecko_id,
                error=str(primary_error),
            )
            try:
                return await self._coinbase_spot(coinbase_product)
            except Exception as fallback_error:
                raise RuntimeError(
                    f"spot providers failed: coingecko={primary_error}; "
                    f"coinbase={fallback_error}"
                ) from fallback_error

    async def _bars_with_fallback(
        self, coingecko_id: str, coinbase_product: str
    ) -> Dict[str, Any]:
        try:
            return await self._bars(coingecko_id)
        except Exception as primary_error:
            self.logger.warning(
                "coingecko chart unavailable; trying coinbase fallback",
                asset=coingecko_id,
                error=str(primary_error),
            )
            try:
                return await self._coinbase_bars(coinbase_product)
            except Exception as fallback_error:
                raise RuntimeError(
                    f"bar providers failed: coingecko={primary_error}; "
                    f"coinbase={fallback_error}"
                ) from fallback_error

    async def _spot(self, coingecko_id: str) -> Dict[str, Any]:
        cached = self._spot_cache.get(coingecko_id)
        if cached and (time.monotonic() - cached[0]) < self.cache_ttl_seconds:
            return cached[1]

        url = (
            f"{self.COINGECKO_BASE}/simple/price"
            f"?ids={coingecko_id}&vs_currencies=usd"
            "&include_24hr_change=true&include_24hr_vol=true&include_market_cap=true"
            "&include_last_updated_at=true"
        )
        data = await self._request_json(url)
        block = data.get(coingecko_id, {}) if isinstance(data, dict) else {}
        price = _safe_float(block.get("usd"))
        if price is None:
            raise ValueError("coingecko_spot_missing_price")
        snapshot = {
            "price_usd": price,
            "change_24h_pct": _safe_float(block.get("usd_24h_change")),
            "volume_24h_usd": _safe_float(block.get("usd_24h_vol")),
            "market_cap_usd": _safe_float(block.get("usd_market_cap")),
            "last_updated_at": block.get("last_updated_at"),
            "provider": "coingecko",
        }
        self._spot_cache[coingecko_id] = (time.monotonic(), snapshot)
        return snapshot

    async def _coinbase_spot(self, product: str) -> Dict[str, Any]:
        cache_key = f"coinbase:{product}"
        cached = self._spot_cache.get(cache_key)
        if cached and (time.monotonic() - cached[0]) < self.cache_ttl_seconds:
            return cached[1]

        data = await self._request_json(f"{self.COINBASE_BASE}/products/{product}/ticker")
        if not isinstance(data, dict):
            raise ValueError("unexpected_coinbase_ticker_payload")
        price = _safe_float(data.get("price"))
        if price is None:
            raise ValueError("coinbase_spot_missing_price")
        snapshot = {
            "price_usd": price,
            "change_24h_pct": None,
            "volume_24h_usd": None,
            "market_cap_usd": None,
            "last_updated_at": data.get("time"),
            "provider": "coinbase",
        }
        self._spot_cache[cache_key] = (time.monotonic(), snapshot)
        return snapshot

    async def _bars(self, coingecko_id: str) -> Dict[str, Any]:
        cached = self._chart_cache.get(coingecko_id)
        if cached and (time.monotonic() - cached[0]) < self.cache_ttl_seconds:
            return cached[1]

        url = (
            f"{self.COINGECKO_BASE}/coins/{coingecko_id}/market_chart"
            "?vs_currency=usd&days=1"
        )
        data = await self._request_json(url)
        points = data.get("prices", []) if isinstance(data, dict) else []
        bars = []
        for point in points:
            if not isinstance(point, list) or len(point) < 2:
                continue
            ts_ms = _safe_float(point[0])
            price = _safe_float(point[1])
            if ts_ms is None or price is None:
                continue
            bars.append(
                {
                    "timestamp_utc": datetime.fromtimestamp(
                        ts_ms / 1000.0, tz=timezone.utc
                    ).isoformat(timespec="seconds"),
                    "price_usd": price,
                    "provider": "coingecko",
                }
            )
        if not bars:
            raise ValueError("coingecko_chart_missing_prices")
        result = {"bars_5m": bars[-60:], "bars_1m": bars[-12:]}
        self._chart_cache[coingecko_id] = (time.monotonic(), result)
        return result

    async def _coinbase_bars(self, product: str) -> Dict[str, Any]:
        cache_key = f"coinbase:{product}"
        cached = self._chart_cache.get(cache_key)
        if cached and (time.monotonic() - cached[0]) < self.cache_ttl_seconds:
            return cached[1]

        url = f"{self.COINBASE_BASE}/products/{product}/candles?granularity=300"
        data = await self._request_json(url)
        if not isinstance(data, list):
            raise ValueError("unexpected_coinbase_candles_payload")

        bars: List[Dict[str, Any]] = []
        for row in data:
            # Coinbase candle format: [time, low, high, open, close, volume].
            if not isinstance(row, list) or len(row) < 5:
                continue
            ts = _safe_float(row[0])
            close = _safe_float(row[4])
            if ts is None or close is None:
                continue
            bars.append(
                {
                    "timestamp_utc": datetime.fromtimestamp(
                        ts, tz=timezone.utc
                    ).isoformat(timespec="seconds"),
                    "price_usd": close,
                    "provider": "coinbase",
                }
            )
        if not bars:
            raise ValueError("coinbase_candles_missing_prices")
        bars.sort(key=lambda item: item["timestamp_utc"])
        result = {"bars_5m": bars[-60:], "bars_1m": bars[-12:]}
        self._chart_cache[cache_key] = (time.monotonic(), result)
        return result

    async def _funding(self, binance_symbol: str) -> Dict[str, Any]:
        cached = self._funding_cache.get(binance_symbol)
        if cached and (time.monotonic() - cached[0]) < self.cache_ttl_seconds:
            return cached[1]

        url = (
            f"{self.BINANCE_FUTURES_BASE}/fapi/v1/premiumIndex"
            f"?symbol={binance_symbol}"
        )
        data = await self._request_json(url)
        if not isinstance(data, dict):
            raise ValueError("unexpected_funding_payload")

        next_funding_ts = data.get("nextFundingTime")
        next_iso = None
        if next_funding_ts is not None:
            parsed = _safe_float(next_funding_ts)
            if parsed is not None:
                next_iso = datetime.fromtimestamp(
                    parsed / 1000.0, tz=timezone.utc
                ).isoformat(timespec="seconds")

        snapshot = {
            "symbol": binance_symbol,
            "mark_price_usd": _safe_float(data.get("markPrice")),
            "index_price_usd": _safe_float(data.get("indexPrice")),
            "last_funding_rate": _safe_float(data.get("lastFundingRate")),
            "next_funding_at_utc": next_iso,
            "provider": "binance",
        }
        self._funding_cache[binance_symbol] = (time.monotonic(), snapshot)
        return snapshot

    async def _request_json(self, url: str) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.http_client.get(url, timeout=self.timeout_seconds)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(self.retry_backoff * (2**attempt))
        assert last_error is not None
        raise last_error


async def fetch_context(
    market: Mapping[str, Any],
    *,
    http_client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    adapter = CryptoAdapter(http_client=http_client)
    try:
        return await adapter.fetch_context(market)
    finally:
        await adapter.aclose()
