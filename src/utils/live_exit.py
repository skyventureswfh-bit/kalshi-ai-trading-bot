"""Confirmed live exit helper.

This module provides a single risk-reducing exit path for live positions. A
position should not be marked closed in local state until Kalshi reports that
the sell actually filled.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict

from src.clients.kalshi_client import KalshiClient
from src.utils.database import Position
from src.utils.kalshi_normalization import (
    get_best_bid_price,
    get_order_average_fill_price,
    get_order_fill_count,
)
from src.utils.logging_setup import get_trading_logger


@dataclass
class LiveExitResult:
    filled: bool
    filled_quantity: float
    fill_price: float
    order_id: str | None
    status: str
    reason: str
    raw_order: Dict[str, Any]


async def execute_confirmed_live_exit(
    *,
    position: Position,
    market_info: Dict[str, Any],
    kalshi_client: KalshiClient,
) -> LiveExitResult:
    """Submit a reduce-only FOK sell and only report success on a confirmed fill.

    A risk exit should never create a new position, so this uses reduce_only.
    Fill-or-kill avoids leaving a tracker-generated emergency exit resting while
    local state incorrectly assumes the exposure is gone.
    """
    logger = get_trading_logger("confirmed_live_exit")
    side = str(position.side or "").lower()
    if side not in {"yes", "no"}:
        return LiveExitResult(False, 0.0, 0.0, None, "invalid", "invalid position side", {})

    best_bid = float(get_best_bid_price(market_info, position.side) or 0.0)
    if best_bid <= 0 or best_bid >= 1:
        return LiveExitResult(
            False,
            0.0,
            0.0,
            None,
            "no_bid",
            "no executable bid available for live exit",
            {},
        )

    quantity = float(position.quantity or 0.0)
    if quantity <= 0:
        return LiveExitResult(False, 0.0, 0.0, None, "invalid", "position quantity is zero", {})

    client_order_id = str(uuid.uuid4())
    order_kwargs: Dict[str, Any] = {
        "ticker": position.market_id,
        "client_order_id": client_order_id,
        "side": side,
        "action": "sell",
        "count": quantity,
        "type_": "limit",
        "time_in_force": "fill_or_kill",
        "reduce_only": True,
    }
    if side == "yes":
        order_kwargs["yes_price_dollars"] = best_bid
    else:
        order_kwargs["no_price_dollars"] = best_bid

    response = await kalshi_client.place_order(**order_kwargs)
    order_info = response.get("order", {}) if isinstance(response, dict) else {}
    status = str(order_info.get("status", "")).lower()
    filled_quantity = float(get_order_fill_count(order_info) or 0.0)
    filled = status in {"filled", "executed", "completed"} or filled_quantity >= quantity - 1e-9

    if not filled:
        logger.warning(
            "Live exit not confirmed filled",
            ticker=position.market_id,
            side=position.side,
            status=status or "unknown",
            filled_quantity=filled_quantity,
            requested_quantity=quantity,
        )
        return LiveExitResult(
            False,
            filled_quantity,
            0.0,
            order_info.get("order_id") or client_order_id,
            status or "unknown",
            "exchange did not confirm full exit fill",
            order_info,
        )

    fill_price = float(
        get_order_average_fill_price(order_info, side=position.side) or best_bid
    )
    return LiveExitResult(
        True,
        quantity if filled_quantity <= 0 else filled_quantity,
        fill_price,
        order_info.get("order_id") or client_order_id,
        status or "filled",
        "confirmed live exit fill",
        order_info,
    )
