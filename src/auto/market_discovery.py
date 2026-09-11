"""Discover the current open BTC 15-minute market from Kalshi's public REST API."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict
from urllib.parse import urlencode
from urllib.request import urlopen


PUBLIC_API = "https://external-api.kalshi.com/trade-api/v2"


class MarketDiscoveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiscoveredMarket:
    ticker: str
    target: float
    expiration_epoch: float


def discover_open_btc_market(
    *, series_ticker: str = "KXBTC15M", now_epoch: float | None = None,
    fetch_json: Callable[[str], Dict[str, Any]] | None = None,
) -> DiscoveredMarket:
    now = time.time() if now_epoch is None else now_epoch
    query = urlencode({"series_ticker": series_ticker, "status": "open", "limit": 20})
    url = f"{PUBLIC_API}/markets?{query}"
    payload = (fetch_json or _fetch_json)(url)
    candidates = []
    for market in payload.get("markets", []):
        expiration = market.get("expected_expiration_time") or market.get("expiration_time")
        target = market.get("floor_strike")
        if not expiration or target is None:
            continue
        expiration_epoch = datetime.fromisoformat(expiration.replace("Z", "+00:00")).timestamp()
        if expiration_epoch > now:
            candidates.append(DiscoveredMarket(str(market["ticker"]), float(target), expiration_epoch))
    if not candidates:
        raise MarketDiscoveryError(f"no usable open market found for {series_ticker}")
    return min(candidates, key=lambda item: item.expiration_epoch)


def _fetch_json(url: str) -> Dict[str, Any]:
    with urlopen(url, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))

