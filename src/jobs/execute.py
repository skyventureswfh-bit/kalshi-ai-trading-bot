"""
Trade execution helpers for live and paper positions.
"""

from __future__ import annotations

from datetime import datetime
import math
import uuid
from typing import Any, Dict, Optional

from src.clients.kalshi_client import KalshiAPIError, KalshiClient
from src.config.settings import settings
from src.utils.database import (
    DatabaseManager,
    Position,
    ShadowOrder,
    SimulatedOrder,
    TradeLog,
)
from src.utils.execution_safety import evaluate_pre_execution_safety
from src.utils.kalshi_normalization import (
    build_limit_order_price_fields,
    dollars_to_cents,
    find_fill_price_for_order,
    get_best_ask_price,
    get_best_ask_size,
    get_best_bid_price,
    get_fill_count,
    get_market_result,
    get_market_series_ticker,
    get_market_status,
    get_market_tick_size,
    get_mid_price,
    get_order_average_fill_price,
    get_order_fill_count,
    is_tradeable_market,
)
from src.utils.logging_setup import get_trading_logger
from src.utils.trade_pricing import (
    FeeMetadata,
    calculate_entry_cost,
    calculate_position_pnl,
    estimate_kalshi_fee,
    extract_fee_metadata,
    fee_divergence,
    sum_fill_fees,
)


def _validate_executable_price(*, ticker: str, side: str, price: float) -> float:
    """Validate a live-paper executable price and return it unchanged."""
    price_cents = dollars_to_cents(price)
    if price <= 0 or price >= 1 or price_cents <= 0 or price_cents >= 100:
        raise ValueError(
            f"Skipping {ticker}: {side.lower()} ask price {price:.4f} "
            f"({price_cents}c rounded) is outside the valid range"
        )
    return price


def _entry_price_exceeds_limit(*, executable_price: float, limit_price: float) -> bool:
    """Return True when the executable price is above the caller-approved cap."""
    return dollars_to_cents(executable_price) > dollars_to_cents(limit_price)


def _floor_to_valid_tick(price: float, tick_size: float) -> float:
    """Round a sell limit down to the nearest valid tick inside Kalshi bounds."""
    effective_tick = tick_size if tick_size > 0 else 0.01
    bounded_price = max(effective_tick, min(float(price or 0.0), 1.0 - effective_tick))
    floored = math.floor((bounded_price + 1e-9) / effective_tick) * effective_tick
    return round(max(effective_tick, min(1.0 - effective_tick, floored)), 4)


def _align_sell_limit_price(*, market_info: Dict[str, Any], price: float) -> float:
    """Normalize a sell limit to the market's current tick size."""
    tick_size = get_market_tick_size(market_info, price)
    return _floor_to_valid_tick(price, tick_size)


def _resolve_displayed_entry_liquidity(*, market_info: Dict[str, Any], side: str) -> float:
    """Return the visible top-of-book size for an entry when the API provides it."""
    return get_best_ask_size(market_info, side)


def _validate_entry_liquidity(
    *,
    ticker: str,
    side: str,
    quantity: float,
    available_quantity: float,
) -> None:
    """Reject paper/live entry attempts that exceed visible best-ask depth."""
    if available_quantity <= 0:
        return
    if float(quantity or 0.0) <= available_quantity + 1e-9:
        return

    raise ValueError(
        f"Skipping {ticker}: requested {float(quantity or 0.0):.2f} {side.upper()} contracts "
        f"but only {available_quantity:.2f} are visible at the current best ask"
    )


def _get_simulated_order_id(*, prefix: str, position: Position) -> str:
    """Build a deterministic-ish local order identifier for paper fills."""
    return f"{prefix}_{position.market_id}_{position.side}_{uuid.uuid4().hex[:12]}"


def _resolve_shadow_mode(shadow_mode: Optional[bool]) -> bool:
    """Resolve an explicit shadow-mode override against runtime settings."""
    if shadow_mode is None:
        return bool(getattr(settings.trading, "shadow_mode_enabled", False))
    return bool(shadow_mode)


async def _resolve_market_fee_metadata(
    *,
    kalshi_client: KalshiClient,
    ticker: str,
    market_info: Dict[str, Any],
) -> FeeMetadata:
    """Resolve fee metadata from the market payload and, when needed, its series."""
    metadata = extract_fee_metadata(market_info)
    if metadata.fee_type:
        return metadata

    series_ticker = get_market_series_ticker(market_info)
    if not series_ticker:
        return metadata

    try:
        series_response = await kalshi_client.get_series(series_ticker)
        series_info = series_response.get("series", {}) if isinstance(series_response, dict) else {}
        return extract_fee_metadata(market_info, series_info)
    except Exception as exc:
        logger = get_trading_logger("trade_execution")
        logger.debug(
            f"Could not resolve series fee metadata for {ticker} ({series_ticker}); "
            "falling back to market-level fee assumptions",
            error=str(exc),
        )
        return metadata


def _extract_order_fee_dollars(order_info: Dict[str, Any], *, maker: bool) -> Optional[float]:
    """Return order-level fee metadata when the API includes it."""
    fee_field = "maker_fees_dollars" if maker else "taker_fees_dollars"
    try:
        fee_value = order_info.get(fee_field)
        if fee_value in (None, ""):
            return None
        return max(0.0, float(fee_value))
    except (TypeError, ValueError):
        return None


def _extract_reported_order_fee(order_info: Dict[str, Any]) -> tuple[Optional[float], Optional[bool]]:
    """
    Return the reported order fee plus its maker/taker hint when present.

    Kalshi sometimes includes either `maker_fees_dollars` or
    `taker_fees_dollars` on the order response. Preserve which fee lane was
    populated so exit-fee reconciliation can compare against the right estimate.
    """
    maker_fee = _extract_order_fee_dollars(order_info, maker=True)
    if maker_fee is not None:
        return maker_fee, True

    taker_fee = _extract_order_fee_dollars(order_info, maker=False)
    if taker_fee is not None:
        return taker_fee, False

    return None, None


async def resolve_live_order_fee(
    *,
    kalshi_client: KalshiClient,
    ticker: str,
    client_order_id: str,
    order_response: Dict,
    maker: bool,
) -> Optional[float]:
    """Return Kalshi-reported live fees, preferring order-level metadata over fills."""
    order = order_response.get("order", {}) if isinstance(order_response, dict) else {}
    order_fee = _extract_order_fee_dollars(order, maker=maker)
    if order_fee is not None:
        return order_fee
    return await _reconcile_fill_fee(
        kalshi_client=kalshi_client,
        ticker=ticker,
        client_order_id=client_order_id,
        order_response=order_response,
    )


async def _live_fee_divergence_already_recorded(
    *,
    db_manager: DatabaseManager,
    market_id: str,
    leg: str,
    position_id: Optional[int],
    order_id: Optional[str],
    estimated_fee: float,
    actual_fee: float,
    quantity: Optional[float],
    price: Optional[float],
) -> bool:
    """Best-effort duplicate check so immediate and delayed exit paths do not log twice."""
    get_entries = getattr(db_manager, "get_fee_divergence_entries", None)
    if not callable(get_entries):
        return False

    try:
        entries = await get_entries(market_id=market_id, leg=leg, limit=50)
    except Exception:
        return False

    for entry in entries or []:
        existing_order_id = entry.get("order_id")
        existing_position_id = entry.get("position_id")
        if order_id and existing_order_id and str(existing_order_id) == str(order_id):
            if position_id is None or existing_position_id in (None, position_id):
                return True

        if position_id is not None and existing_position_id != position_id:
            continue

        try:
            existing_estimated = float(entry.get("estimated_fee", 0.0) or 0.0)
            existing_actual = float(entry.get("actual_fee", 0.0) or 0.0)
            existing_quantity = (
                None
                if entry.get("quantity") in (None, "")
                else float(entry.get("quantity", 0.0) or 0.0)
            )
            existing_price = (
                None
                if entry.get("price") in (None, "")
                else float(entry.get("price", 0.0) or 0.0)
            )
        except (TypeError, ValueError):
            continue

        quantity_matches = quantity is None or existing_quantity is None or abs(existing_quantity - quantity) <= 1e-9
        price_matches = price is None or existing_price is None or abs(existing_price - price) <= 1e-9
        if (
            abs(existing_estimated - estimated_fee) <= 1e-9
            and abs(existing_actual - actual_fee) <= 1e-9
            and quantity_matches
            and price_matches
        ):
            return True

    return False


async def record_live_fee_divergence_if_needed(
    *,
    db_manager: DatabaseManager,
    market_id: str,
    side: str,
    leg: str,
    estimated_fee: float,
    actual_fee: Optional[float],
    position_id: Optional[int] = None,
    order_id: Optional[str] = None,
    quantity: Optional[float] = None,
    price: Optional[float] = None,
) -> None:
    """Persist a live fee drift row only when Kalshi reported a real divergence."""
    if actual_fee is None:
        return

    try:
        estimated = float(estimated_fee or 0.0)
        actual = float(actual_fee or 0.0)
    except (TypeError, ValueError):
        return

    if abs(fee_divergence(actual, estimated)) <= 1e-9:
        return

    record_divergence = getattr(db_manager, "record_fee_divergence", None)
    if not callable(record_divergence):
        return

    if await _live_fee_divergence_already_recorded(
        db_manager=db_manager,
        market_id=market_id,
        leg=leg,
        position_id=position_id,
        order_id=order_id,
        estimated_fee=estimated,
        actual_fee=actual,
        quantity=quantity,
        price=price,
    ):
        return

    await record_divergence(
        market_id=market_id,
        side=side,
        leg=leg,
        estimated_fee=estimated,
        actual_fee=actual,
        position_id=position_id,
        order_id=order_id,
        quantity=quantity,
        price=price,
    )


async def _record_filled_paper_entry_order(
    *,
    position: Position,
    db_manager: DatabaseManager,
    fill_price: float,
    filled_at: datetime,
    order_id: str,
) -> None:
    """Persist a filled paper buy order so paper mode has the same execution trail as live."""
    await db_manager.add_simulated_order(
        SimulatedOrder(
            strategy=position.strategy or "directional_trading",
            market_id=position.market_id,
            side=position.side,
            action="buy",
            price=fill_price,
            quantity=position.quantity,
            status="filled",
            live=False,
            order_id=order_id,
            placed_at=filled_at,
            filled_at=filled_at,
            filled_price=fill_price,
            position_id=position.id,
        )
    )


async def _record_shadow_entry_order(
    *,
    position: Position,
    db_manager: DatabaseManager,
    top_of_book_price: float,
    quantity: float,
    market_info: Dict[str, Any],
    placed_at: datetime,
    orderbook: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a shadow comparison entry order for a position."""
    shadow_fill_price: Optional[float] = top_of_book_price
    shadow_status = "filled"
    effective_orderbook = orderbook or {}

    if market_info and effective_orderbook:
        tick_size = get_market_tick_size(market_info, top_of_book_price)
        depth_summary = _simulate_fok_depth_walk(
            orderbook=effective_orderbook,
            side=position.side,
            quantity=quantity,
            max_slippage_ticks=1,
            tick_size=tick_size,
            best_ask=top_of_book_price,
        )
        if depth_summary["can_fill"]:
            shadow_fill_price = float(depth_summary["average_price"])
        else:
            shadow_status = "rejected"
            shadow_fill_price = None

    await db_manager.add_shadow_order(
        ShadowOrder(
            strategy=position.strategy or "directional_trading",
            market_id=position.market_id,
            side=position.side,
            action="buy",
            price=top_of_book_price,
            quantity=quantity,
            status=shadow_status,
            live=True,
            order_id=_get_simulated_order_id(prefix="shadow_buy", position=position),
            placed_at=placed_at,
            filled_at=placed_at if shadow_status == "filled" else None,
            filled_price=shadow_fill_price,
            target_price=top_of_book_price,
            position_id=position.id,
        )
    )


async def _record_shadow_sell_limit_order(
    *,
    position: Position,
    limit_price: float,
    db_manager: DatabaseManager,
    market_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a shadow comparison sell-limit order for a position."""
    strategy = position.strategy or "directional_trading"
    now = datetime.now()

    if position.id is not None:
        resting_orders_same_position = [
            order
            for order in await db_manager.get_shadow_orders(
                strategy=strategy,
                market_id=position.market_id,
                side=position.side,
                action="sell",
                status="resting",
            )
            if order.position_id == position.id
        ]

        for order in resting_orders_same_position:
            if (
                abs(float(order.price) - limit_price) < 1e-9
                and abs(float(order.quantity) - position.quantity) < 1e-9
            ):
                return

        for order in resting_orders_same_position:
            await db_manager.update_shadow_order(int(order.id), status="cancelled")

    effective_market_info = market_info or {}
    best_bid = (
        get_best_bid_price(effective_market_info, position.side)
        if effective_market_info
        else 0.0
    )
    shadow_order_id = _get_simulated_order_id(prefix="shadow_sell", position=position)

    if best_bid > 0 and best_bid + 1e-9 >= limit_price:
        await db_manager.add_shadow_order(
            ShadowOrder(
                strategy=strategy,
                market_id=position.market_id,
                side=position.side,
                action="sell",
                price=limit_price,
                quantity=position.quantity,
                status="filled",
                live=True,
                order_id=shadow_order_id,
                placed_at=now,
                filled_at=now,
                filled_price=best_bid,
                target_price=limit_price,
                position_id=position.id,
            )
        )
        return

    await db_manager.add_shadow_order(
        ShadowOrder(
            strategy=strategy,
            market_id=position.market_id,
            side=position.side,
            action="sell",
            price=limit_price,
            quantity=position.quantity,
            status="resting",
            live=True,
            order_id=shadow_order_id,
            placed_at=now,
            target_price=limit_price,
            position_id=position.id,
        )
    )


async def _get_current_executable_entry_quote(
    *,
    kalshi_client: KalshiClient,
    ticker: str,
    side: str,
) -> tuple[float, float, Dict[str, Any]]:
    """Fetch the current buy price, visible best-ask size, and raw market snapshot."""
    market_data = await kalshi_client.get_market(ticker)
    market = market_data.get("market", {})

    if not market:
        raise ValueError(f"Skipping {ticker}: no market data returned")

    if not is_tradeable_market(market):
        raise ValueError(f"Skipping {ticker}: collection/aggregate ticker")

    ask_dollars = get_best_ask_price(market, side)
    ask_size = _resolve_displayed_entry_liquidity(market_info=market, side=side)
    return _validate_executable_price(ticker=ticker, side=side, price=ask_dollars), ask_size, market


async def _record_pre_execution_kalshi_health(
    *,
    db_manager: DatabaseManager,
    ticker: str,
    market_info: Dict[str, Any],
) -> None:
    """Persist a fresh Kalshi source-health row after a successful entry quote fetch."""
    recorder = getattr(db_manager, "record_source_snapshot", None)
    if not callable(recorder):
        return
    try:
        await recorder(
            category="kalshi",
            source="kalshi.public-api",
            status="healthy",
            freshness_seconds=0,
            payload={
                "ticker": ticker,
                "phase": "pre_execution_entry_quote",
                "status": get_market_status(market_info),
            },
        )
    except Exception:
        return


def _normalize_book_levels(
    orderbook: Dict[str, Any], side: str, *, ascending: bool
) -> list[tuple[float, float]]:
    """
    Return sorted `(price, size)` levels for the resting side that an entry order
    would lift. For buying YES (side="YES") we want the ask side of the YES book;
    the Kalshi orderbook payload stores the ask side as the requested side's list
    (the resting sell orders), so the caller picks `ascending=True` to walk from
    cheapest ask to deepest.
    """
    raw_levels = orderbook.get(f"{side.lower()}_dollars") or orderbook.get(side.lower(), [])
    levels: list[tuple[float, float]] = []

    for level in raw_levels or []:
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            continue
        try:
            price = float(level[0])
            size = float(level[1])
        except (TypeError, ValueError):
            continue
        if price > 1.0:
            price = price / 100.0
        if price <= 0 or size <= 0:
            continue
        levels.append((price, size))

    levels.sort(key=lambda item: item[0], reverse=not ascending)
    return levels


def _simulate_fok_depth_walk(
    *,
    orderbook: Dict[str, Any],
    side: str,
    quantity: float,
    max_slippage_ticks: int = 1,
    tick_size: float = 0.01,
    best_ask: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Simulate a FOK-style buy that walks the visible ask side of the book.

    Returns a dict with:
      - filled_quantity: number of contracts that can be filled within slippage
      - average_price: size-weighted average price across filled levels (dollars)
      - worst_price: deepest level touched
      - levels: list of (price, qty) that were consumed
      - can_fill: True when `filled_quantity >= quantity` within slippage cap

    Notes:
      - Kalshi's order book on the entry side is a stack of resting sell orders;
        we walk from best (lowest) ask upwards.
      - `max_slippage_ticks` guards against "fat book" situations where the FOK
        would silently walk multiple ticks. A true FOK that can't fill within the
        cap is rejected (can_fill=False), matching live behavior when the order
        would need to sweep beyond the submitted limit.
    """
    levels = _normalize_book_levels(orderbook, side, ascending=True)
    if not levels:
        return {
            "filled_quantity": 0.0,
            "average_price": 0.0,
            "worst_price": 0.0,
            "levels": [],
            "can_fill": False,
        }

    target_quantity = max(0.0, float(quantity or 0.0))
    if target_quantity <= 0:
        return {
            "filled_quantity": 0.0,
            "average_price": float(levels[0][0]),
            "worst_price": float(levels[0][0]),
            "levels": [],
            "can_fill": True,
        }

    effective_tick = tick_size if tick_size > 0 else 0.01
    reference_ask = float(best_ask) if best_ask and best_ask > 0 else float(levels[0][0])
    price_cap = reference_ask + (max_slippage_ticks * effective_tick) + 1e-9

    filled_qty = 0.0
    cost = 0.0
    consumed: list[tuple[float, float]] = []
    worst_price = reference_ask

    for price, size in levels:
        if price > price_cap:
            break
        take = min(size, target_quantity - filled_qty)
        if take <= 0:
            break
        filled_qty += take
        cost += take * price
        worst_price = max(worst_price, price)
        consumed.append((price, take))
        if filled_qty + 1e-9 >= target_quantity:
            break

    average_price = (cost / filled_qty) if filled_qty > 0 else reference_ask
    can_fill = filled_qty + 1e-9 >= target_quantity
    return {
        "filled_quantity": filled_qty,
        "average_price": average_price,
        "worst_price": worst_price,
        "levels": consumed,
        "can_fill": can_fill,
    }


async def _fetch_entry_orderbook(
    *,
    kalshi_client: KalshiClient,
    ticker: str,
) -> Dict[str, Any]:
    """Fetch the live orderbook payload used for depth-aware FOK simulation."""
    try:
        response = await kalshi_client.get_orderbook(ticker, depth=10)
    except Exception:
        return {}
    if not isinstance(response, dict):
        return {}
    return response.get("orderbook_fp") or response.get("orderbook") or response or {}


async def _get_current_executable_entry_price(
    *,
    kalshi_client: KalshiClient,
    ticker: str,
    side: str,
) -> float:
    """Fetch the best currently executable buy price for the requested side."""
    ask_dollars, _ask_size, _market = await _get_current_executable_entry_quote(
        kalshi_client=kalshi_client,
        ticker=ticker,
        side=side,
    )
    return ask_dollars


async def _reconcile_fill_price(
    *,
    kalshi_client: KalshiClient,
    ticker: str,
    side: str,
    client_order_id: str,
    order_response: Dict,
    fallback_price: float,
) -> float:
    """Resolve a live fill price from the order payload or recent fills."""
    logger = get_trading_logger("trade_execution")
    order = order_response.get("order", {}) if isinstance(order_response, dict) else {}

    order_fill_price = get_order_average_fill_price(order, side=side)
    if order_fill_price and order_fill_price > 0:
        return order_fill_price

    try:
        fills_response = await kalshi_client.get_fills(ticker=ticker, limit=20)
        fills = fills_response.get("fills", []) if isinstance(fills_response, dict) else []
        fill_price = find_fill_price_for_order(
            fills,
            side=side,
            order_id=order.get("order_id"),
            client_order_id=client_order_id,
            ticker=ticker,
        )
        if fill_price and fill_price > 0:
            return fill_price
    except Exception as exc:
        logger.warning(
            f"Could not reconcile fills for {ticker}; falling back to limit price",
            error=str(exc),
        )

    logger.warning(
        f"Could not resolve exact fill price for {ticker}; using submitted limit price {fallback_price:.4f}"
    )
    return fallback_price


async def _reconcile_fill_quantity(
    *,
    kalshi_client: KalshiClient,
    ticker: str,
    client_order_id: str,
    order_response: Dict,
    fallback_quantity: float,
) -> float:
    """Resolve the executed quantity from order payload or recent fills."""
    logger = get_trading_logger("trade_execution")
    order = order_response.get("order", {}) if isinstance(order_response, dict) else {}

    order_fill_count = get_order_fill_count(order)
    if order_fill_count > 0:
        return order_fill_count

    try:
        fills_response = await kalshi_client.get_fills(ticker=ticker, limit=20)
        fills = fills_response.get("fills", []) if isinstance(fills_response, dict) else []
        order_id = order.get("order_id")
        matched_quantity = sum(
            get_fill_count(fill)
            for fill in fills
            if (order_id and fill.get("order_id") == order_id)
            or (not order_id and client_order_id and fill.get("client_order_id") == client_order_id)
        )
        if matched_quantity > 0:
            return matched_quantity
    except Exception as exc:
        logger.warning(
            f"Could not reconcile fill quantity for {ticker}; falling back to requested size",
            error=str(exc),
        )

    return fallback_quantity


async def _reconcile_fill_fee(
    *,
    kalshi_client: KalshiClient,
    ticker: str,
    client_order_id: str,
    order_response: Dict,
) -> Optional[float]:
    """
    Return the total Kalshi-reported fee for this order, when available.

    Pulls per-fill `fee_cost` rows from `/fills` and sums them so the caller
    can compare to the estimated fee schedule. Returns None if Kalshi did
    not include any fee metadata on the fills we can match.
    """
    logger = get_trading_logger("trade_execution")
    order = order_response.get("order", {}) if isinstance(order_response, dict) else {}
    order_id = order.get("order_id")

    try:
        fills_response = await kalshi_client.get_fills(ticker=ticker, limit=20)
    except Exception as exc:
        logger.warning(
            f"Could not fetch fills for fee reconciliation on {ticker}",
            error=str(exc),
        )
        return None

    fills = fills_response.get("fills", []) if isinstance(fills_response, dict) else []
    matched = [
        fill
        for fill in fills
        if (order_id and fill.get("order_id") == order_id)
        or (client_order_id and fill.get("client_order_id") == client_order_id)
    ]
    if not matched:
        return None
    return sum_fill_fees(matched)


async def _record_live_exit_fee_divergence_if_filled(
    *,
    position: Position,
    limit_price: float,
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    client_order_id: str,
    order_response: Dict[str, Any],
    fee_metadata: FeeMetadata,
    allow_partial_fill: bool = False,
) -> None:
    """
    Persist exit-leg fee drift when a live sell limit already executed.

    Resting sell limits reconcile later when positions are closed. By default
    this helper only records fully filled exits; IOC callers can opt in to
    partial terminal fills via `allow_partial_fill=True`.
    """
    order_info = order_response.get("order", {}) if isinstance(order_response, dict) else {}
    order_status = str(order_info.get("status", "")).lower()
    filled_quantity = get_order_fill_count(order_info)
    if filled_quantity <= 0 and order_status not in {"filled", "executed"}:
        return
    if filled_quantity > 0 and order_status not in {"filled", "executed"} and not allow_partial_fill:
        return

    if filled_quantity <= 0:
        filled_quantity = float(position.quantity or 0.0)
    if filled_quantity <= 0:
        return

    actual_fee, maker_hint = _extract_reported_order_fee(order_info)
    if actual_fee is None:
        actual_fee = await resolve_live_order_fee(
            kalshi_client=kalshi_client,
            ticker=position.market_id,
            client_order_id=client_order_id,
            order_response=order_response,
            maker=False,
        )
    if actual_fee is None:
        return

    fill_price = get_order_average_fill_price(order_info, side=position.side) or limit_price
    estimated_fee = estimate_kalshi_fee(
        fill_price,
        filled_quantity,
        maker=bool(maker_hint) if maker_hint is not None else False,
        fee_type=fee_metadata.fee_type,
        fee_multiplier=fee_metadata.fee_multiplier,
        fee_waiver_expiration_time=fee_metadata.fee_waiver_expiration_time,
        trade_ts=datetime.now(),
    )
    await record_live_fee_divergence_if_needed(
        db_manager=db_manager,
        market_id=position.market_id,
        side=position.side,
        leg="exit",
        estimated_fee=estimated_fee,
        actual_fee=actual_fee,
        position_id=position.id,
        order_id=order_info.get("order_id") or client_order_id,
        quantity=filled_quantity,
        price=fill_price,
    )


async def record_simulated_position_exit(
    *,
    position: Position,
    exit_price: float,
    db_manager: DatabaseManager,
    rationale_suffix: str,
    entry_maker: bool = False,
    exit_maker: bool = False,
    charge_entry_fee: bool = True,
    charge_exit_fee: bool = True,
    exit_fee_type: Optional[str] = None,
    exit_fee_multiplier: Optional[float] = None,
    exit_fee_waiver_expiration_time: Any = None,
    exit_trade_ts: Optional[datetime] = None,
) -> Dict[str, float | bool]:
    """Persist a paper exit using the shared fee-aware PnL model."""
    stored_entry_fee = (
        position.entry_fee
        if position.entry_order_id or position.contracts_cost > 0 or position.entry_fee > 0
        else None
    )
    stored_contracts_cost = (
        position.contracts_cost
        if position.entry_order_id or position.contracts_cost > 0
        else None
    )
    entry_cost = calculate_entry_cost(
        price=position.entry_price,
        quantity=position.quantity,
        maker=entry_maker,
        fee_override=stored_entry_fee,
    )
    pnl_details = calculate_position_pnl(
        entry_price=position.entry_price,
        exit_price=exit_price,
        quantity=position.quantity,
        entry_maker=entry_maker,
        exit_maker=exit_maker,
        charge_entry_fee=charge_entry_fee,
        charge_exit_fee=charge_exit_fee,
        entry_fee_override=stored_entry_fee,
        entry_trade_ts=position.timestamp,
        exit_fee_type=exit_fee_type,
        exit_fee_multiplier=exit_fee_multiplier,
        exit_fee_waiver_expiration_time=exit_fee_waiver_expiration_time,
        exit_trade_ts=exit_trade_ts or datetime.now(),
    )

    trade_log = TradeLog(
        market_id=position.market_id,
        side=position.side,
        entry_price=position.entry_price,
        exit_price=exit_price,
        quantity=position.quantity,
        pnl=pnl_details["net_pnl"],
        entry_timestamp=position.timestamp,
        exit_timestamp=datetime.now(),
        rationale=f"{position.rationale} | {rationale_suffix}",
        live=False,
        strategy=position.strategy,
        entry_fee=pnl_details["entry_fee"],
        exit_fee=pnl_details["exit_fee"],
        fees_paid=pnl_details["fees_paid"],
        contracts_cost=(
            stored_contracts_cost
            if stored_contracts_cost is not None
            else entry_cost["contracts_cost"]
        ),
    )
    await db_manager.add_trade_log(trade_log)
    await db_manager.update_position_status(position.id, "closed")

    return {
        "success": True,
        "gross_pnl": pnl_details["gross_pnl"],
        "net_pnl": pnl_details["net_pnl"],
        "fees_paid": pnl_details["fees_paid"],
        "entry_fee": pnl_details["entry_fee"],
        "exit_fee": pnl_details["exit_fee"],
        "is_win": pnl_details["net_pnl"] > 0,
        "is_loss": pnl_details["net_pnl"] <= 0,
    }


def _strategy_entry_is_maker(strategy: Optional[str]) -> bool:
    """Infer the entry fee model for a paper position from its strategy."""
    return str(strategy or "").strip().lower() == "market_making"


async def submit_simulated_sell_limit_order(
    *,
    position: Position,
    limit_price: float,
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
) -> Dict[str, Any]:
    """
    Persist or immediately fill a paper sell order using the current live book.

    Paper orders now mirror live intent more closely:
    - if the limit crosses the current best bid, simulate an immediate taker fill
    - otherwise persist a resting local order for later reconciliation

    W2 Gap 2: resting orders are keyed by `position_id` (not `(market_id, side)`)
    so two positions on the same side cannot race for the same fill during
    reconciliation.
    """
    logger = get_trading_logger("simulated_sell_limit_order")
    strategy = position.strategy or "directional_trading"
    now = datetime.now()

    if position.id is None:
        logger.warning(
            "Cannot submit simulated sell order without a persisted position_id for %s %s",
            position.market_id,
            position.side,
        )
        return {
            "success": False,
            "filled": False,
            "orders_placed": 0,
            "positions_closed": 0,
            "net_pnl": 0.0,
            "fees_paid": 0.0,
            "filled_price": None,
            "order_id": None,
        }

    # Key resting orders by position_id. Any legacy market-keyed rows that
    # belong to other positions are left alone by this query.
    resting_orders_same_position = [
        order
        for order in await db_manager.get_simulated_orders(
            strategy=strategy,
            market_id=position.market_id,
            side=position.side,
            action="sell",
            status="resting",
        )
        if order.position_id == position.id
    ]

    for order in resting_orders_same_position:
        if abs(float(order.price) - limit_price) < 1e-9 and abs(float(order.quantity) - position.quantity) < 1e-9:
            logger.info(
                "Paper sell limit already resting for position %s (%s %s) at $%.4f; reusing existing order.",
                position.id,
                position.market_id,
                position.side,
                limit_price,
            )
            return {
                "success": True,
                "filled": False,
                "orders_placed": 0,
                "positions_closed": 0,
                "net_pnl": 0.0,
                "fees_paid": 0.0,
                "filled_price": None,
                "order_id": order.order_id,
            }

    for order in resting_orders_same_position:
        await db_manager.update_simulated_order(int(order.id), status="cancelled")

    market_info: Dict[str, Any] = {}
    fee_metadata = FeeMetadata()
    try:
        market_response = await kalshi_client.get_market(position.market_id)
        market_info = market_response.get("market", {}) if isinstance(market_response, dict) else {}
        if market_info:
            fee_metadata = await _resolve_market_fee_metadata(
                kalshi_client=kalshi_client,
                ticker=position.market_id,
                market_info=market_info,
            )
    except Exception as exc:
        logger.warning(
            "Could not fetch live market data while submitting a paper sell order for %s; storing as resting.",
            position.market_id,
            error=str(exc),
        )

    best_bid = get_best_bid_price(market_info, position.side) if market_info else 0.0
    order_id = f"sim_sell_{position.market_id}_{position.side}_{int(now.timestamp())}"

    if best_bid > 0 and best_bid + 1e-9 >= limit_price:
        simulated_order = SimulatedOrder(
            strategy=strategy,
            market_id=position.market_id,
            side=position.side,
            action="sell",
            price=limit_price,
            quantity=position.quantity,
            status="filled",
            live=False,
            order_id=order_id,
            placed_at=now,
            filled_at=now,
            filled_price=best_bid,
            target_price=limit_price,
            position_id=position.id,
        )
        await db_manager.add_simulated_order(simulated_order)
        exit_result = await record_simulated_position_exit(
            position=position,
            exit_price=best_bid,
            db_manager=db_manager,
            rationale_suffix=f"PAPER LIMIT SELL FILLED @ ${best_bid:.4f}",
            entry_maker=_strategy_entry_is_maker(strategy),
            exit_maker=False,
            charge_entry_fee=True,
            charge_exit_fee=True,
            exit_fee_type=fee_metadata.fee_type,
            exit_fee_multiplier=fee_metadata.fee_multiplier,
            exit_fee_waiver_expiration_time=fee_metadata.fee_waiver_expiration_time,
            exit_trade_ts=now,
        )
        logger.info(
            "Paper sell limit executed immediately for %s at $%.4f (limit $%.4f).",
            position.market_id,
            best_bid,
            limit_price,
        )
        return {
            "success": True,
            "filled": True,
            "orders_placed": 1,
            "positions_closed": 1,
            "net_pnl": float(exit_result["net_pnl"]),
            "fees_paid": float(exit_result["fees_paid"]),
            "filled_price": best_bid,
            "order_id": order_id,
        }

    simulated_order = SimulatedOrder(
        strategy=strategy,
        market_id=position.market_id,
        side=position.side,
        action="sell",
        price=limit_price,
        quantity=position.quantity,
        status="resting",
        live=False,
        order_id=order_id,
        placed_at=now,
        target_price=limit_price,
        position_id=position.id,
    )
    await db_manager.add_simulated_order(simulated_order)
    logger.info(
        "Stored resting paper sell limit for %s %s x%s at $%.4f.",
        position.market_id,
        position.side,
        position.quantity,
        limit_price,
    )
    return {
        "success": True,
        "filled": False,
        "orders_placed": 1,
        "positions_closed": 0,
        "net_pnl": 0.0,
        "fees_paid": 0.0,
        "filled_price": None,
        "order_id": order_id,
    }


async def reconcile_simulated_exit_orders(
    *,
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    strategy: Optional[str] = None,
    market_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Fill resting paper sell orders when the live book or settlement supports it."""
    logger = get_trading_logger("simulated_order_reconciliation")
    results = {
        "orders_filled": 0,
        "positions_closed": 0,
        "orders_cancelled": 0,
        "net_pnl": 0.0,
        "fees_paid": 0.0,
    }
    resting_orders = await db_manager.get_simulated_orders(
        strategy=strategy,
        market_id=market_id,
        action="sell",
        status="resting",
    )

    for order in resting_orders:
        try:
            # W2 Gap 2: look up the position by position_id first so two orders
            # that happen to share (market_id, side) cannot race for the same
            # reconciliation. Fall back to the legacy (market_id, side) path
            # only when an order predates the position_id column.
            position: Optional[Position] = None
            if order.position_id is not None:
                position = await db_manager.get_position_by_id(int(order.position_id))
                if position and position.status != "open":
                    await db_manager.update_simulated_order(int(order.id), status="cancelled")
                    results["orders_cancelled"] += 1
                    continue
            else:
                position = await db_manager.get_position_by_market_and_side(
                    order.market_id, order.side
                )

            if not position:
                await db_manager.update_simulated_order(int(order.id), status="cancelled")
                results["orders_cancelled"] += 1
                continue

            market_response = await kalshi_client.get_market(order.market_id)
            market_info = market_response.get("market", {}) if isinstance(market_response, dict) else {}
            if not market_info:
                continue
            fee_metadata = await _resolve_market_fee_metadata(
                kalshi_client=kalshi_client,
                ticker=order.market_id,
                market_info=market_info,
            )

            market_status = get_market_status(market_info)
            exit_price: Optional[float] = None
            rationale_suffix = ""
            exit_maker = True
            charge_exit_fee = True

            if market_status in {"closed", "settled", "finalized"}:
                market_result = get_market_result(market_info)
                if market_result:
                    exit_price = 1.0 if str(market_result).upper() == position.side.upper() else 0.0
                else:
                    exit_price = get_best_bid_price(market_info, position.side) or float(order.price)
                rationale_suffix = f"PAPER MARKET RESOLUTION @ ${exit_price:.4f}"
                exit_maker = False
                charge_exit_fee = False
            else:
                best_bid = get_best_bid_price(market_info, position.side)
                if best_bid <= 0 or best_bid + 1e-9 < float(order.price):
                    continue
                exit_price = float(order.price)
                rationale_suffix = f"PAPER LIMIT SELL FILLED @ ${exit_price:.4f}"

            exit_result = await record_simulated_position_exit(
                position=position,
                exit_price=exit_price,
                db_manager=db_manager,
                rationale_suffix=rationale_suffix,
                entry_maker=_strategy_entry_is_maker(position.strategy),
                exit_maker=exit_maker,
                charge_entry_fee=True,
                charge_exit_fee=charge_exit_fee,
                exit_fee_type=fee_metadata.fee_type,
                exit_fee_multiplier=fee_metadata.fee_multiplier,
                exit_fee_waiver_expiration_time=fee_metadata.fee_waiver_expiration_time,
            )
            await db_manager.update_simulated_order(
                int(order.id),
                status="filled",
                filled_price=exit_price,
                filled_at=datetime.now(),
                position_id=position.id,
            )
            results["orders_filled"] += 1
            results["positions_closed"] += 1
            results["net_pnl"] += float(exit_result["net_pnl"])
            results["fees_paid"] += float(exit_result["fees_paid"])
            logger.info(
                "Filled resting paper exit for %s %s at $%.4f.",
                position.market_id,
                position.side,
                exit_price,
            )
        except Exception as exc:
            logger.warning(
                "Could not reconcile paper exit order for %s %s; leaving it resting.",
                order.market_id,
                order.side,
                error=str(exc),
            )

    return results


async def execute_position(
    position: Position,
    live_mode: bool,
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    *,
    shadow_mode: Optional[bool] = None,
    paper_market_info: Optional[Dict[str, Any]] = None,
    client_order_id: Optional[str] = None,
) -> bool:
    """
    Execute a single trade position.

    Args:
        client_order_id: Optional caller-supplied client order ID for the
            live entry order. Defaults to None, which preserves existing
            behavior exactly — a random UUID is generated internally, as
            before. Callers that need idempotent retries (e.g. an
            unattended supervisor) may pass a deterministic value here;
            Manual/interactive call sites should never set this.

    Returns:
        True when the position was successfully activated, otherwise False.
    """
    logger = get_trading_logger("trade_execution")
    logger.info(f"Executing position for market: {position.market_id}")
    logger.info(f"Live mode: {live_mode}")
    shadow_mode = _resolve_shadow_mode(shadow_mode)

    if not live_mode:
        try:
            top_of_book_price, displayed_ask_size, market_info = await _get_current_executable_entry_quote(
                kalshi_client=kalshi_client,
                ticker=position.market_id,
                side=position.side,
            )
            await _record_pre_execution_kalshi_health(
                db_manager=db_manager,
                ticker=position.market_id,
                market_info=market_info,
            )
        except Exception as exc:
            if paper_market_info:
                top_of_book_price = position.entry_price
                market_info = dict(paper_market_info)
                displayed_ask_size = _resolve_displayed_entry_liquidity(
                    market_info=market_info,
                    side=position.side,
                )
            else:
                top_of_book_price = position.entry_price
                displayed_ask_size = 0.0
                market_info = {}
            logger.warning(
                f"Could not fetch live market data for paper entry on {position.market_id}; "
                f"falling back to requested entry price {top_of_book_price:.4f}",
                error=str(exc),
            )

        safety = await evaluate_pre_execution_safety(
            kalshi_client=kalshi_client,
            db_manager=db_manager,
            position=position,
            live_mode=False,
            market_info=market_info,
        )
        if not safety.allowed:
            logger.warning(
                "Skipping paper entry for %s: execution safety rejected trade (%s)",
                position.market_id,
                safety.reason,
            )
            return False

        if _entry_price_exceeds_limit(
            executable_price=top_of_book_price,
            limit_price=position.entry_price,
        ):
            logger.warning(
                f"Skipping paper entry for {position.market_id}: executable ask "
                f"${top_of_book_price:.4f} exceeds approved limit "
                f"${position.entry_price:.4f}"
            )
            return False

        # Depth-aware FOK simulation: walk the visible book so paper fills match
        # what a real FOK limit would actually achieve. If the order can't be fully
        # filled within one tick of the submitted limit, reject the entry just like
        # Kalshi would.
        paper_entry_price = top_of_book_price
        depth_summary: Optional[Dict[str, Any]] = None
        orderbook: Optional[Dict[str, Any]] = None
        if market_info:
            orderbook = await _fetch_entry_orderbook(
                kalshi_client=kalshi_client,
                ticker=position.market_id,
            )
            if orderbook:
                tick_size = get_market_tick_size(market_info, top_of_book_price)
                depth_summary = _simulate_fok_depth_walk(
                    orderbook=orderbook,
                    side=position.side,
                    quantity=position.quantity,
                    max_slippage_ticks=1,
                    tick_size=tick_size,
                    best_ask=top_of_book_price,
                )
                if not depth_summary["can_fill"]:
                    logger.warning(
                        f"Skipping paper entry for {position.market_id}: "
                        f"visible book can only fill {depth_summary['filled_quantity']:.2f} of "
                        f"{position.quantity:.2f} contracts within one tick of "
                        f"${top_of_book_price:.4f}"
                    )
                    return False
                if _entry_price_exceeds_limit(
                    executable_price=float(depth_summary["average_price"]),
                    limit_price=position.entry_price,
                ):
                    logger.warning(
                        f"Skipping paper entry for {position.market_id}: depth-walk "
                        f"average ${depth_summary['average_price']:.4f} exceeds "
                        f"approved limit ${position.entry_price:.4f}"
                    )
                    return False
                paper_entry_price = depth_summary["average_price"]
            else:
                try:
                    _validate_entry_liquidity(
                        ticker=position.market_id,
                        side=position.side,
                        quantity=position.quantity,
                        available_quantity=displayed_ask_size,
                    )
                except ValueError as exc:
                    logger.warning(str(exc))
                    return False

        filled_at = datetime.now()
        fee_metadata = await _resolve_market_fee_metadata(
            kalshi_client=kalshi_client,
            ticker=position.market_id,
            market_info=market_info,
        ) if market_info else FeeMetadata()
        entry_cost = calculate_entry_cost(
            paper_entry_price,
            position.quantity,
            maker=False,
            fee_type=fee_metadata.fee_type,
            fee_multiplier=fee_metadata.fee_multiplier,
            fee_waiver_expiration_time=fee_metadata.fee_waiver_expiration_time,
            trade_ts=filled_at,
        )
        order_id = _get_simulated_order_id(prefix="sim_buy", position=position)
        position.entry_price = paper_entry_price
        position.entry_fee = entry_cost["fee"]
        position.contracts_cost = entry_cost["contracts_cost"]
        position.entry_order_id = order_id
        await _record_filled_paper_entry_order(
            position=position,
            db_manager=db_manager,
            fill_price=paper_entry_price,
            filled_at=filled_at,
            order_id=order_id,
        )
        await db_manager.update_position_execution_details(
            position.id,
            entry_price=paper_entry_price,
            quantity=position.quantity,
            live=False,
            stop_loss_price=position.stop_loss_price,
            take_profit_price=position.take_profit_price,
            max_hold_hours=position.max_hold_hours,
            entry_fee=position.entry_fee,
            contracts_cost=position.contracts_cost,
            entry_order_id=position.entry_order_id,
        )
        if shadow_mode:
            try:
                await _record_shadow_entry_order(
                    position=position,
                    db_manager=db_manager,
                    top_of_book_price=top_of_book_price,
                    quantity=position.quantity,
                    market_info=market_info,
                    placed_at=filled_at,
                    orderbook=orderbook,
                )
            except Exception as exc:
                logger.warning(
                    "Could not persist shadow entry order for %s; paper entry remains unchanged.",
                    position.market_id,
                    error=str(exc),
                )
        if depth_summary and depth_summary.get("levels"):
            walk_trail = ", ".join(
                f"{qty:.2f}@${price:.4f}" for price, qty in depth_summary["levels"]
            )
            logger.info(
                f"PAPER TRADE EXECUTED for {position.market_id} "
                f"avg ${paper_entry_price:.4f} (top ${top_of_book_price:.4f}) "
                f"via FOK walk: {walk_trail}"
            )
        else:
            logger.info(
                f"PAPER TRADE EXECUTED for {position.market_id} at ${paper_entry_price:.4f} "
                f"using live market data"
            )
        if displayed_ask_size > 0:
            logger.info(
                f"Visible top-of-book liquidity at entry: {displayed_ask_size:.2f} contracts"
            )
        logger.info(
            f"Estimated deployed capital: ${entry_cost['contracts_cost']:.2f} "
            f"+ fees ${entry_cost['fee']:.2f} = ${entry_cost['total_cost']:.2f}"
        )
        return True

    logger.warning(f"PLACING LIVE ORDER - real money will be used for {position.market_id}")

    try:
        side_lower = position.side.lower()
        try:
            ask_dollars, displayed_ask_size, market_info = await _get_current_executable_entry_quote(
                kalshi_client=kalshi_client,
                ticker=position.market_id,
                side=position.side,
            )
            await _record_pre_execution_kalshi_health(
                db_manager=db_manager,
                ticker=position.market_id,
                market_info=market_info,
            )
        except Exception as exc:
            safety = await evaluate_pre_execution_safety(
                kalshi_client=kalshi_client,
                db_manager=db_manager,
                position=position,
                live_mode=True,
            )
            if not safety.allowed:
                logger.warning(
                    "Skipping live entry for %s: execution safety rejected trade (%s)",
                    position.market_id,
                    safety.reason,
                )
                return False
            logger.warning(
                f"Skipping live entry for {position.market_id}: market data unavailable",
                error=str(exc),
            )
            return False
        safety = await evaluate_pre_execution_safety(
            kalshi_client=kalshi_client,
            db_manager=db_manager,
            position=position,
            live_mode=True,
            market_info=market_info,
        )
        if not safety.allowed:
            logger.warning(
                "Skipping live entry for %s: execution safety rejected trade (%s)",
                position.market_id,
                safety.reason,
            )
            return False
        if _entry_price_exceeds_limit(
            executable_price=ask_dollars,
            limit_price=position.entry_price,
        ):
            logger.warning(
                f"Skipping live entry for {position.market_id}: executable ask "
                f"${ask_dollars:.4f} exceeds approved limit "
                f"${position.entry_price:.4f}"
            )
            return False
        _validate_entry_liquidity(
            ticker=position.market_id,
            side=position.side,
            quantity=position.quantity,
            available_quantity=displayed_ask_size,
        )
        shadow_orderbook: Optional[Dict[str, Any]] = None
        shadow_placed_at = datetime.now()
        if shadow_mode:
            shadow_orderbook = await _fetch_entry_orderbook(
                kalshi_client=kalshi_client,
                ticker=position.market_id,
            )

        client_order_id = client_order_id or str(uuid.uuid4())
        order_params = {
            "ticker": position.market_id,
            "client_order_id": client_order_id,
            "side": side_lower,
            "action": "buy",
            "count": position.quantity,
            "type_": "limit",
            "time_in_force": "fill_or_kill",
            **build_limit_order_price_fields(position.side, ask_dollars),
        }

        logger.info(f"Placing order with params: {order_params}")
        order_response = await kalshi_client.place_order(**order_params)
        order_info = order_response.get("order", {}) if isinstance(order_response, dict) else {}
        order_status = str(order_info.get("status", "")).lower()
        if get_order_fill_count(order_info) <= 0 and order_status not in {"filled", "executed", "completed"}:
            logger.warning(
                f"Kalshi did not fill order {order_info.get('order_id', client_order_id)} for {position.market_id}; status={order_status or 'unknown'}"
            )
            return False

        fill_price = await _reconcile_fill_price(
            kalshi_client=kalshi_client,
            ticker=position.market_id,
            side=position.side,
            client_order_id=client_order_id,
            order_response=order_response,
            fallback_price=ask_dollars,
        )
        fill_quantity = await _reconcile_fill_quantity(
            kalshi_client=kalshi_client,
            ticker=position.market_id,
            client_order_id=client_order_id,
            order_response=order_response,
            fallback_quantity=position.quantity,
        )

        position.entry_price = fill_price
        position.quantity = fill_quantity
        position.live = True
        fee_metadata = await _resolve_market_fee_metadata(
            kalshi_client=kalshi_client,
            ticker=position.market_id,
            market_info=market_info,
        )
        estimated_entry_fee = estimate_kalshi_fee(
            fill_price,
            fill_quantity,
            maker=False,
            fee_type=fee_metadata.fee_type,
            fee_multiplier=fee_metadata.fee_multiplier,
            fee_waiver_expiration_time=fee_metadata.fee_waiver_expiration_time,
            trade_ts=datetime.now(),
        )
        # W2 Gap 3: prefer the fee values Kalshi actually reported (order-level
        # `taker_fees_dollars` / per-fill `fee_cost`) over the public-formula
        # estimate. When live and estimated diverge, log the delta so the
        # paper-vs-live dashboard can surface fee drift.
        actual_entry_fee = await resolve_live_order_fee(
            kalshi_client=kalshi_client,
            ticker=position.market_id,
            client_order_id=client_order_id,
            order_response=order_response,
            maker=False,
        )

        if actual_entry_fee is not None:
            position.entry_fee = actual_entry_fee
            await record_live_fee_divergence_if_needed(
                db_manager=db_manager,
                market_id=position.market_id,
                side=position.side,
                leg="entry",
                estimated_fee=estimated_entry_fee,
                actual_fee=actual_entry_fee,
                position_id=position.id,
                order_id=order_info.get("order_id") or client_order_id,
                quantity=fill_quantity,
                price=fill_price,
            )
        else:
            position.entry_fee = estimated_entry_fee
        position.contracts_cost = fill_price * fill_quantity
        position.entry_order_id = order_info.get("order_id") or client_order_id
        await db_manager.update_position_execution_details(
            position.id,
            entry_price=fill_price,
            quantity=fill_quantity,
            live=True,
            stop_loss_price=position.stop_loss_price,
            take_profit_price=position.take_profit_price,
            max_hold_hours=position.max_hold_hours,
            entry_fee=position.entry_fee,
            contracts_cost=position.contracts_cost,
            entry_order_id=position.entry_order_id,
        )
        logger.info(
            f"LIVE ORDER PLACED for {position.market_id}. Order ID: {order_response.get('order', {}).get('order_id')}"
        )
        logger.info(
            f"Real money used: ${position.contracts_cost:.2f} + fees ${position.entry_fee:.2f}"
        )
        if shadow_mode:
            try:
                await _record_shadow_entry_order(
                    position=position,
                    db_manager=db_manager,
                    top_of_book_price=ask_dollars,
                    quantity=fill_quantity,
                    market_info=market_info,
                    placed_at=shadow_placed_at,
                    orderbook=shadow_orderbook,
                )
            except Exception as exc:
                logger.warning(
                    "Could not persist shadow entry order for %s; live entry remains unchanged.",
                    position.market_id,
                    error=str(exc),
                )
        return True

    except ValueError as exc:
        logger.warning(str(exc))
        return False
    except KalshiAPIError as exc:
        logger.error(f"FAILED to place LIVE order for {position.market_id}: {exc}")
        return False


async def place_sell_limit_order(
    position: Position,
    limit_price: float,
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    *,
    live_mode: Optional[bool] = None,
    reduce_only: bool = False,
    shadow_mode: Optional[bool] = None,
) -> bool:
    """
    Place a limit order to close an existing position.

    In paper mode, simulate the exit order locally without hitting Kalshi.
    In live mode, use a resting GTC limit by default. Kalshi currently rejects
    `reduce_only=True` on non-IoC orders, so callers should only enable
    `reduce_only` for immediate-or-cancel / fill-or-kill exit flows.
    """
    logger = get_trading_logger("sell_limit_order")
    if live_mode is None:
        live_mode = getattr(settings.trading, "live_trading_enabled", False)
    shadow_mode = _resolve_shadow_mode(shadow_mode)

    try:
        side = position.side.lower()
        client_order_id = str(uuid.uuid4())
        order_params = {
            "ticker": position.market_id,
            "client_order_id": client_order_id,
            "side": side,
            "action": "sell",
            "count": position.quantity,
            "type_": "limit",
            "time_in_force": "good_till_canceled",
            **build_limit_order_price_fields(position.side, limit_price),
        }
        if reduce_only:
            order_params["reduce_only"] = True

        if not live_mode:
            paper_result = await submit_simulated_sell_limit_order(
                position=position,
                limit_price=limit_price,
                db_manager=db_manager,
                kalshi_client=kalshi_client,
            )
            if shadow_mode:
                shadow_market_info: Dict[str, Any] = {}
                try:
                    shadow_market_response = await kalshi_client.get_market(position.market_id)
                    shadow_market_info = (
                        shadow_market_response.get("market", {})
                        if isinstance(shadow_market_response, dict)
                        else {}
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not fetch market snapshot for shadow paper exit on %s; storing best-effort shadow telemetry.",
                        position.market_id,
                        error=str(exc),
                    )
                try:
                    await _record_shadow_sell_limit_order(
                        position=position,
                        limit_price=limit_price,
                        db_manager=db_manager,
                        market_info=shadow_market_info,
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not persist shadow sell order for %s; paper sell order remains unchanged.",
                        position.market_id,
                        error=str(exc),
                    )
            logger.info(
                f"SIMULATED SELL LIMIT order: {position.quantity} {side.upper()} "
                f"at ${limit_price:.4f} for {position.market_id}"
            )
            return bool(paper_result.get("success"))

        logger.info(
            f"Placing SELL LIMIT order: {position.quantity} {side.upper()} at ${limit_price:.4f} for {position.market_id}"
        )
        shadow_market_info: Dict[str, Any] = {}
        if shadow_mode:
            try:
                shadow_market_response = await kalshi_client.get_market(position.market_id)
                shadow_market_info = (
                    shadow_market_response.get("market", {})
                    if isinstance(shadow_market_response, dict)
                    else {}
                )
            except Exception as exc:
                logger.warning(
                    "Could not fetch market snapshot for shadow exit on %s; placing live order without it.",
                    position.market_id,
                    error=str(exc),
                )
        response = await kalshi_client.place_order(**order_params)

        if response and "order" in response:
            order_id = response["order"].get("order_id", client_order_id)
            fee_market_info = shadow_market_info
            if not fee_market_info:
                try:
                    fee_market_response = await kalshi_client.get_market(position.market_id)
                    fee_market_info = (
                        fee_market_response.get("market", {})
                        if isinstance(fee_market_response, dict)
                        else {}
                    )
                except Exception as exc:
                    logger.debug(
                        "Could not refresh market metadata for exit-fee reconciliation on %s.",
                        position.market_id,
                        error=str(exc),
                    )
                    fee_market_info = {}
            fee_metadata = await _resolve_market_fee_metadata(
                kalshi_client=kalshi_client,
                ticker=position.market_id,
                market_info=fee_market_info,
            )
            await _record_live_exit_fee_divergence_if_filled(
                position=position,
                limit_price=limit_price,
                db_manager=db_manager,
                kalshi_client=kalshi_client,
                client_order_id=client_order_id,
                order_response=response,
                fee_metadata=fee_metadata,
            )
            logger.info(f"SELL LIMIT ORDER placed successfully. Order ID: {order_id}")
            logger.info(f"Market: {position.market_id}")
            logger.info(f"Side: {side.upper()} (selling {position.quantity} shares)")
            logger.info(f"Limit Price: ${limit_price:.4f}")
            logger.info(f"Expected Proceeds: ${limit_price * position.quantity:.2f}")
            if shadow_mode:
                try:
                    await _record_shadow_sell_limit_order(
                        position=position,
                        limit_price=limit_price,
                        db_manager=db_manager,
                        market_info=shadow_market_info,
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not persist shadow sell order for %s; live sell order remains unchanged.",
                        position.market_id,
                        error=str(exc),
                    )
            return True

        logger.error(f"Failed to place sell limit order: {response}")
        return False
    except Exception as exc:
        logger.error(f"Error placing sell limit order for {position.market_id}: {exc}")
        return False


async def place_profit_taking_orders(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    profit_threshold: float = 0.25,
    *,
    live_mode: Optional[bool] = None,
    shadow_mode: Optional[bool] = None,
) -> Dict[str, int]:
    """Place sell limits for positions that have reached profit targets."""
    logger = get_trading_logger("profit_taking")
    if live_mode is None:
        live_mode = getattr(settings.trading, "live_trading_enabled", False)
    shadow_mode = _resolve_shadow_mode(shadow_mode)

    results = {"orders_placed": 0, "positions_processed": 0, "positions_closed": 0}

    try:
        positions = (
            await db_manager.get_open_live_positions()
            if live_mode
            else await db_manager.get_open_non_live_positions()
        )
        if not positions:
            logger.info("No open positions to process for profit taking")
            return results

        logger.info(f"Checking {len(positions)} positions for profit-taking opportunities")
        for position in positions:
            try:
                if position.strategy in {"quick_flip_scalping", "market_making"}:
                    continue
                results["positions_processed"] += 1
                market_response = await kalshi_client.get_market(position.market_id)
                market_data = market_response.get("market", {})
                if not market_data:
                    logger.warning(f"Could not get market data for {position.market_id}")
                    continue
                if get_market_status(market_data) in {"closed", "settled", "finalized"}:
                    continue

                current_price = get_mid_price(market_data, position.side)
                if current_price <= 0:
                    continue

                profit_pct = (current_price - position.entry_price) / position.entry_price
                unrealized_pnl = (current_price - position.entry_price) * position.quantity
                logger.debug(
                    f"Position {position.market_id}: Entry=${position.entry_price:.3f}, "
                    f"Current=${current_price:.3f}, Profit={profit_pct:.1%}, PnL=${unrealized_pnl:.2f}"
                )

                if profit_pct >= profit_threshold:
                    # Near-certain winners are worth more held to settlement
                    # (fee-free $1 payout) than sold early minus fee + spread.
                    hold_threshold = float(
                        getattr(settings.trading, "hold_winners_to_settlement_price", 0.95) or 0.95
                    )
                    if (
                        bool(getattr(settings.trading, "hold_winners_to_settlement", True))
                        and current_price >= hold_threshold
                    ):
                        logger.info(
                            f"HOLDING TO SETTLEMENT: {position.market_id} at ${current_price:.2f} "
                            f"(>= ${hold_threshold:.2f}) — skipping early profit-taking to avoid exit fees"
                        )
                        continue
                    logger.info(
                        f"PROFIT TARGET HIT: {position.market_id} - {profit_pct:.1%} profit (${unrealized_pnl:.2f})"
                    )

                    sell_price = _align_sell_limit_price(
                        market_info=market_data,
                        price=current_price * 0.98,
                    )
                    if live_mode:
                        success = await place_sell_limit_order(
                            position=position,
                            limit_price=sell_price,
                            db_manager=db_manager,
                            kalshi_client=kalshi_client,
                            live_mode=True,
                            shadow_mode=shadow_mode,
                        )
                        if success:
                            results["orders_placed"] += 1
                            logger.info(f"Profit-taking order placed for {position.market_id}")
                        else:
                            logger.error(f"Failed to place profit-taking order for {position.market_id}")
                    else:
                        paper_result = await submit_simulated_sell_limit_order(
                            position=position,
                            limit_price=sell_price,
                            db_manager=db_manager,
                            kalshi_client=kalshi_client,
                        )
                        if shadow_mode and paper_result.get("success"):
                            try:
                                await _record_shadow_sell_limit_order(
                                    position=position,
                                    limit_price=sell_price,
                                    db_manager=db_manager,
                                    market_info=market_data,
                                )
                            except Exception as exc:
                                logger.warning(
                                    "Could not persist shadow profit-taking order for %s.",
                                    position.market_id,
                                    error=str(exc),
                                )
                        if paper_result.get("success"):
                            results["orders_placed"] += int(paper_result.get("orders_placed", 0))
                            results["positions_closed"] += int(paper_result.get("positions_closed", 0))
                            if paper_result.get("filled"):
                                logger.info(
                                    f"Paper profit-taking exit executed for {position.market_id}: "
                                    f"net=${float(paper_result['net_pnl']):.2f}"
                                )
                            else:
                                logger.info(
                                    f"Paper profit-taking order resting for {position.market_id} "
                                    f"at ${sell_price:.4f}"
                                )
            except Exception as exc:
                logger.error(f"Error processing position {position.market_id} for profit taking: {exc}")

        logger.info(
            f"Profit-taking summary: {results['orders_placed']} orders placed, "
            f"{results['positions_closed']} paper positions closed from {results['positions_processed']} positions"
        )
        return results
    except Exception as exc:
        logger.error(f"Error in profit-taking order placement: {exc}")
        return results


async def place_stop_loss_orders(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    stop_loss_threshold: float = -0.10,
    *,
    live_mode: Optional[bool] = None,
    shadow_mode: Optional[bool] = None,
) -> Dict[str, int]:
    """Place sell limits for positions that need stop-loss protection."""
    logger = get_trading_logger("stop_loss_orders")
    if live_mode is None:
        live_mode = getattr(settings.trading, "live_trading_enabled", False)
    shadow_mode = _resolve_shadow_mode(shadow_mode)

    results = {"orders_placed": 0, "positions_processed": 0, "positions_closed": 0}

    try:
        positions = (
            await db_manager.get_open_live_positions()
            if live_mode
            else await db_manager.get_open_non_live_positions()
        )
        if not positions:
            logger.info("No open positions to process for stop-loss orders")
            return results

        logger.info(f"Checking {len(positions)} positions for stop-loss protection")
        for position in positions:
            try:
                if position.strategy in {"quick_flip_scalping", "market_making"}:
                    continue
                results["positions_processed"] += 1
                market_response = await kalshi_client.get_market(position.market_id)
                market_data = market_response.get("market", {})
                if not market_data:
                    logger.warning(f"Could not get market data for {position.market_id}")
                    continue
                if get_market_status(market_data) in {"closed", "settled", "finalized"}:
                    continue

                current_price = get_mid_price(market_data, position.side)
                if current_price <= 0:
                    continue

                loss_pct = (current_price - position.entry_price) / position.entry_price
                unrealized_pnl = (current_price - position.entry_price) * position.quantity
                if loss_pct <= stop_loss_threshold:
                    stop_price = _align_sell_limit_price(
                        market_info=market_data,
                        price=max(0.01, position.entry_price * (1 + stop_loss_threshold * 1.1)),
                    )
                    logger.info(
                        f"STOP LOSS TRIGGERED: {position.market_id} - {loss_pct:.1%} loss (${unrealized_pnl:.2f})"
                    )

                    if live_mode:
                        success = await place_sell_limit_order(
                            position=position,
                            limit_price=stop_price,
                            db_manager=db_manager,
                            kalshi_client=kalshi_client,
                            live_mode=True,
                            shadow_mode=shadow_mode,
                        )
                        if success:
                            results["orders_placed"] += 1
                            logger.info(f"Stop-loss order placed for {position.market_id}")
                        else:
                            logger.error(f"Failed to place stop-loss order for {position.market_id}")
                    else:
                        paper_result = await submit_simulated_sell_limit_order(
                            position=position,
                            limit_price=stop_price,
                            db_manager=db_manager,
                            kalshi_client=kalshi_client,
                        )
                        if shadow_mode and paper_result.get("success"):
                            try:
                                await _record_shadow_sell_limit_order(
                                    position=position,
                                    limit_price=stop_price,
                                    db_manager=db_manager,
                                    market_info=market_data,
                                )
                            except Exception as exc:
                                logger.warning(
                                    "Could not persist shadow stop-loss order for %s.",
                                    position.market_id,
                                    error=str(exc),
                                )
                        if paper_result.get("success"):
                            results["orders_placed"] += int(paper_result.get("orders_placed", 0))
                            results["positions_closed"] += int(paper_result.get("positions_closed", 0))
                            if paper_result.get("filled"):
                                logger.info(
                                    f"Paper stop-loss exit executed for {position.market_id}: "
                                    f"net=${float(paper_result['net_pnl']):.2f}"
                                )
                            else:
                                logger.info(
                                    f"Paper stop-loss order resting for {position.market_id} "
                                    f"at ${stop_price:.4f}"
                                )
            except Exception as exc:
                logger.error(f"Error processing position {position.market_id} for stop loss: {exc}")

        logger.info(
            f"Stop-loss summary: {results['orders_placed']} orders placed, "
            f"{results['positions_closed']} paper positions closed from {results['positions_processed']} positions"
        )
        return results
    except Exception as exc:
        logger.error(f"Error in stop-loss order placement: {exc}")
        return results
