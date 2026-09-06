"""
Market making strategy with persisted paper-order simulation.

Paper mode keeps local resting orders across cycles, reconciles them against
live market quotes, and converts filled maker entries/exits into the same
position and trade-log records used elsewhere in the bot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.clients.kalshi_client import KalshiClient
from src.config.settings import settings
from src.jobs.execute import record_simulated_position_exit
from src.utils.database import DatabaseManager, Market, Position, SimulatedOrder
from src.utils.kalshi_normalization import (
    build_limit_order_price_fields,
    get_best_ask_price,
    get_best_bid_price,
    get_market_result,
    get_market_series_ticker,
    get_market_status,
    get_mid_price,
)
from src.utils.logging_setup import get_trading_logger
from src.utils.trade_pricing import FeeMetadata, calculate_entry_cost, extract_fee_metadata


@dataclass
class LimitOrder:
    """Represents a market-making limit order."""

    market_id: str
    side: str
    action: str
    price: float
    quantity: int
    order_type: str = "limit"
    status: str = "pending"
    order_id: Optional[str] = None
    placed_at: Optional[datetime] = None
    expected_profit: float = 0.0
    target_price: Optional[float] = None
    position_id: Optional[int] = None
    id: Optional[int] = None


@dataclass
class MarketMakingOpportunity:
    """Represents a market making opportunity with calculated spreads."""

    market_id: str
    market_title: str
    current_yes_price: float
    current_no_price: float
    ai_predicted_prob: float
    ai_confidence: float
    optimal_yes_bid: float
    optimal_yes_ask: float
    optimal_no_bid: float
    optimal_no_ask: float
    yes_spread_profit: float
    no_spread_profit: float
    total_expected_profit: float
    inventory_risk: float
    volatility_estimate: float
    optimal_yes_size: int
    optimal_no_size: int


class AdvancedMarketMaker:
    """
    Advanced market making strategy that provides liquidity while capturing edge.

    Live mode places real GTC maker orders. Paper mode persists those maker
    orders locally and simulates fills when live quotes cross the resting price.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        kalshi_client: KalshiClient,
        xai_client: Any,
    ):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.xai_client = xai_client
        self.logger = get_trading_logger("market_maker")

        self.min_spread = getattr(settings.trading, "min_spread_for_making", 0.03)
        self.max_spread = getattr(settings.trading, "max_bid_ask_spread", 0.10)
        self.target_inventory = 0.0
        self.inventory_penalty = getattr(settings.trading, "max_inventory_risk", 0.01)
        self.volatility_multiplier = 2.0

        self.active_orders: Dict[str, List[LimitOrder]] = {}
        self.filled_orders: List[LimitOrder] = []
        self.total_pnl = 0.0

        self.markets_traded = 0
        self.total_volume = 0.0
        self.win_rate = 0.0

    @staticmethod
    def _limit_order_from_simulated(order: SimulatedOrder) -> LimitOrder:
        """Convert a persisted simulated order into the runtime order model."""
        return LimitOrder(
            market_id=order.market_id,
            side=order.side,
            action=order.action,
            price=order.price,
            quantity=int(order.quantity),
            status=order.status,
            order_id=order.order_id,
            placed_at=order.placed_at,
            expected_profit=float(order.expected_profit or 0.0),
            target_price=order.target_price,
            position_id=order.position_id,
            id=order.id,
        )

    async def _load_persisted_paper_orders(self) -> List[LimitOrder]:
        """Load current resting paper market-making orders into memory."""
        persisted_orders = await self.db_manager.get_simulated_orders(
            strategy="market_making",
            status="resting",
        )
        orders = [self._limit_order_from_simulated(order) for order in persisted_orders]
        self.active_orders = {}
        for order in orders:
            self.active_orders.setdefault(order.market_id, []).append(order)
        return orders

    async def _find_resting_paper_order(
        self,
        *,
        market_id: str,
        side: str,
        action: str,
    ) -> Optional[LimitOrder]:
        """Return an existing resting paper order for the same market/side/action."""
        orders = await self.db_manager.get_simulated_orders(
            strategy="market_making",
            market_id=market_id,
            side=side,
            action=action,
            status="resting",
        )
        if not orders:
            return None
        return self._limit_order_from_simulated(orders[0])

    async def _persist_paper_order(self, order: LimitOrder) -> LimitOrder:
        """Write a paper order to SQLite so it survives later cycles."""
        simulated_order = SimulatedOrder(
            strategy="market_making",
            market_id=order.market_id,
            side=order.side,
            action=order.action,
            price=order.price,
            quantity=float(order.quantity),
            status=order.status,
            live=False,
            order_id=order.order_id,
            placed_at=order.placed_at or datetime.now(),
            expected_profit=order.expected_profit,
            target_price=order.target_price,
            position_id=order.position_id,
        )
        order.id = await self.db_manager.add_simulated_order(simulated_order)
        return order

    async def _resolve_fee_metadata(self, market_info: Dict[str, object]) -> FeeMetadata:
        """Resolve fee metadata from the market payload and its parent series."""
        metadata = extract_fee_metadata(market_info)
        if metadata.fee_type:
            return metadata

        series_ticker = get_market_series_ticker(market_info)
        if not series_ticker:
            return metadata

        try:
            series_response = await self.kalshi_client.get_series(series_ticker)
            series_info = series_response.get("series", {}) if isinstance(series_response, dict) else {}
            return extract_fee_metadata(market_info, series_info)
        except Exception as exc:
            self.logger.debug(
                f"Could not resolve series fee metadata for {series_ticker}: {exc}"
            )
            return metadata

    async def _fill_paper_entry_order(
        self,
        order: LimitOrder,
        *,
        fee_metadata: FeeMetadata,
    ) -> Dict[str, float]:
        """Convert a filled maker entry order into a paper position plus paired exit order."""
        filled_at = datetime.now()
        entry_cost = calculate_entry_cost(
            price=order.price,
            quantity=order.quantity,
            maker=True,
            fee_type=fee_metadata.fee_type,
            fee_multiplier=fee_metadata.fee_multiplier,
            fee_waiver_expiration_time=fee_metadata.fee_waiver_expiration_time,
            trade_ts=filled_at,
        )
        position = Position(
            market_id=order.market_id,
            side=order.side,
            entry_price=order.price,
            quantity=order.quantity,
            timestamp=filled_at,
            entry_fee=entry_cost["fee"],
            contracts_cost=entry_cost["contracts_cost"],
            entry_order_id=order.order_id,
            rationale=(
                f"PAPER MARKET MAKING MAKER ENTRY @ ${order.price:.4f}"
                f" target=${float(order.target_price or order.price):.4f}"
            ),
            live=False,
            strategy="market_making",
        )
        position_id = await self.db_manager.add_position(position)
        if position_id is None:
            self.logger.warning(
                "Paper market-making entry filled for %s %s but an open position already exists; "
                "marking the order filled without creating another position.",
                order.market_id,
                order.side,
            )
            existing_position = await self.db_manager.get_position_by_market_and_side(
                order.market_id,
                order.side,
            )
            position_id = existing_position.id if existing_position else None

        order.position_id = position_id
        order.status = "filled"
        await self.db_manager.update_simulated_order(
            int(order.id),
            status="filled",
            filled_price=order.price,
            filled_at=filled_at,
            position_id=position_id,
        )
        self.filled_orders.append(order)
        self.total_volume += float(order.quantity)

        if order.target_price and position_id is not None:
            exit_order = LimitOrder(
                market_id=order.market_id,
                side=order.side,
                action="sell",
                price=float(order.target_price),
                quantity=order.quantity,
                status="resting",
                order_id=f"sim_exit_{order.market_id}_{order.side}_{int(filled_at.timestamp())}",
                placed_at=filled_at,
                expected_profit=order.expected_profit,
                position_id=position_id,
            )
            await self._persist_paper_order(exit_order)
            self.active_orders.setdefault(exit_order.market_id, []).append(exit_order)

        self.logger.info(
            "Paper market-making entry filled: %s %s x%s @ $%.4f",
            order.market_id,
            order.side,
            order.quantity,
            order.price,
        )
        return {"entries_filled": 1.0, "exits_filled": 0.0, "realized_pnl": 0.0}

    async def _fill_paper_exit_order(
        self,
        order: LimitOrder,
        *,
        fee_metadata: FeeMetadata,
    ) -> Dict[str, float]:
        """Close the linked paper position when the paired maker exit fills."""
        position = await self.db_manager.get_position_by_market_and_side(order.market_id, order.side)
        if not position:
            await self.db_manager.update_simulated_order(int(order.id), status="cancelled")
            self.logger.warning(
                "Paper market-making exit order %s had no matching open position; cancelling local order.",
                order.order_id or order.id,
            )
            return {"entries_filled": 0.0, "exits_filled": 0.0, "realized_pnl": 0.0}

        exit_result = await record_simulated_position_exit(
            position=position,
            exit_price=order.price,
            db_manager=self.db_manager,
            rationale_suffix=f"PAPER MARKET MAKING EXIT @ ${order.price:.4f}",
            entry_maker=True,
            exit_maker=True,
            charge_entry_fee=True,
            charge_exit_fee=True,
            exit_fee_type=fee_metadata.fee_type,
            exit_fee_multiplier=fee_metadata.fee_multiplier,
            exit_fee_waiver_expiration_time=fee_metadata.fee_waiver_expiration_time,
        )
        await self.db_manager.update_simulated_order(
            int(order.id),
            status="filled",
            filled_price=order.price,
            filled_at=datetime.now(),
            position_id=position.id,
        )
        order.status = "filled"
        self.filled_orders.append(order)
        self.total_pnl += float(exit_result["net_pnl"])
        self.markets_traded += 1

        self.logger.info(
            "Paper market-making exit filled: %s %s x%s @ $%.4f net=$%.2f",
            order.market_id,
            order.side,
            order.quantity,
            order.price,
            float(exit_result["net_pnl"]),
        )
        return {
            "entries_filled": 0.0,
            "exits_filled": 1.0,
            "realized_pnl": float(exit_result["net_pnl"]),
        }

    @staticmethod
    def _market_settlement_exit_price(
        market_info: Dict[str, object],
        side: str,
    ) -> float:
        """Return settlement exit price for a paper order in a closed market."""
        market_result = get_market_result(market_info)
        if market_result:
            return 1.0 if str(market_result).upper() == str(side).upper() else 0.0

        fallback_price = get_best_bid_price(market_info, side)
        if fallback_price <= 0:
            return 0.0
        return fallback_price

    async def reconcile_persisted_paper_orders(self) -> Dict[str, float]:
        """Simulate fills for resting paper market-making orders using live quotes."""
        results = {"entries_filled": 0.0, "exits_filled": 0.0, "realized_pnl": 0.0}
        if getattr(settings.trading, "live_trading_enabled", False):
            return results

        resting_orders = await self._load_persisted_paper_orders()
        for order in resting_orders:
            try:
                market_data = await self.kalshi_client.get_market(order.market_id)
                market_info = market_data.get("market", {}) if isinstance(market_data, dict) else {}
                if not market_info:
                    continue
                fee_metadata = await self._resolve_fee_metadata(market_info)

                market_status = get_market_status(market_info)
                if market_status in {"closed", "settled", "finalized"}:
                    if order.action == "buy":
                        await self.db_manager.update_simulated_order(
                            int(order.id),
                            status="cancelled",
                        )
                        continue

                    position = await self.db_manager.get_position_by_market_and_side(
                        order.market_id,
                        order.side,
                    )
                    if (
                        not position
                        or (order.position_id is not None and position.id != order.position_id)
                    ):
                        await self.db_manager.update_simulated_order(
                            int(order.id),
                            status="cancelled",
                        )
                        continue

                    exit_price = self._market_settlement_exit_price(
                        market_info,
                        order.side,
                    )
                    exit_result = await record_simulated_position_exit(
                        position=position,
                        exit_price=exit_price,
                        db_manager=self.db_manager,
                        rationale_suffix=f"PAPER MARKET MAKING SETTLEMENT @ ${exit_price:.4f}",
                        entry_maker=True,
                        exit_maker=False,
                        charge_entry_fee=True,
                        charge_exit_fee=False,
                        exit_fee_type=fee_metadata.fee_type,
                        exit_fee_multiplier=fee_metadata.fee_multiplier,
                        exit_fee_waiver_expiration_time=fee_metadata.fee_waiver_expiration_time,
                    )
                    await self.db_manager.update_simulated_order(
                        int(order.id),
                        status="filled",
                        filled_price=exit_price,
                        filled_at=datetime.now(),
                        position_id=position.id,
                    )
                    order.status = "filled"
                    self.filled_orders.append(order)
                    self.total_pnl += float(exit_result["net_pnl"])
                    self.markets_traded += 1
                    self.logger.info(
                        "Paper market-making position closed by settlement: %s %s x%s @ $%.4f net=$%.2f",
                        order.market_id,
                        order.side,
                        order.quantity,
                        exit_price,
                        float(exit_result["net_pnl"]),
                    )
                    for key in results:
                        results[key] += float(
                            {"entries_filled": 0.0, "exits_filled": 1.0, "realized_pnl": exit_result["net_pnl"]}[
                                key
                            ]
                        )
                    continue

                if order.action == "buy":
                    best_ask = get_best_ask_price(market_info, order.side)
                    if best_ask > 0 and best_ask <= order.price + 1e-9:
                        fill_result = await self._fill_paper_entry_order(
                            order,
                            fee_metadata=fee_metadata,
                        )
                        for key in results:
                            results[key] += float(fill_result.get(key, 0.0))
                else:
                    best_bid = get_best_bid_price(market_info, order.side)
                    if best_bid > 0 and best_bid + 1e-9 >= order.price:
                        fill_result = await self._fill_paper_exit_order(
                            order,
                            fee_metadata=fee_metadata,
                        )
                        for key in results:
                            results[key] += float(fill_result.get(key, 0.0))
            except Exception as exc:
                self.logger.error(
                    f"Error reconciling paper market-making order {order.order_id or order.id}: {exc}"
                )

        await self._load_persisted_paper_orders()
        return results

    async def analyze_market_making_opportunities(
        self,
        markets: List[Market],
    ) -> List[MarketMakingOpportunity]:
        """Analyze a bounded, high-volume candidate pool for market-making opportunities."""
        opportunities = []

        max_execution_markets = max(
            1,
            int(getattr(settings.trading, "max_concurrent_markets", 10)),
        )
        analysis_limit = max_execution_markets * 3
        ranked_markets = sorted(
            markets,
            key=lambda market: int(getattr(market, "volume", 0) or 0),
            reverse=True,
        )
        markets_to_analyze = ranked_markets[:analysis_limit]

        if len(markets) > len(markets_to_analyze):
            self.logger.info(
                "Market-making analysis capped at %s of %s eligible markets "
                "(top volume candidates; execution limit=%s)",
                len(markets_to_analyze),
                len(markets),
                max_execution_markets,
            )

        for market in markets_to_analyze:
            try:
                market_data = await self.kalshi_client.get_market(market.market_id)
                market_info = market_data.get("market", {}) if isinstance(market_data, dict) else {}
                if not market_info:
                    continue

                current_yes_price = get_mid_price(market_info, "YES")
                current_no_price = get_mid_price(market_info, "NO")
                if current_yes_price < 0.02 or current_yes_price > 0.98:
                    continue

                analysis = await self._get_ai_analysis(market)
                if not analysis:
                    continue

                ai_prob = analysis.get("probability", 0.5)
                ai_confidence = analysis.get("confidence", 0.5)

                from src.utils.edge_filter import EdgeFilter

                yes_edge_result = EdgeFilter.calculate_edge(ai_prob, current_yes_price, ai_confidence)
                no_edge_result = EdgeFilter.calculate_edge(1 - ai_prob, current_no_price, ai_confidence)

                if yes_edge_result.passes_filter or no_edge_result.passes_filter:
                    opportunity = await self._calculate_market_making_opportunity(
                        market,
                        current_yes_price,
                        current_no_price,
                        ai_prob,
                        ai_confidence,
                    )
                    if opportunity and opportunity.total_expected_profit > 0:
                        opportunities.append(opportunity)
                        self.logger.info(
                            f"Market making approved: {market.market_id} - "
                            f"YES edge: {yes_edge_result.edge_percentage:.1%}, "
                            f"NO edge: {no_edge_result.edge_percentage:.1%}"
                        )
                else:
                    self.logger.info(
                        f"Market making filtered: {market.market_id} - insufficient edge on both sides"
                    )
            except Exception as e:
                self.logger.error(f"Error analyzing market {market.market_id}: {e}")
                continue

        opportunities.sort(key=lambda item: item.total_expected_profit, reverse=True)
        return opportunities

    async def _calculate_market_making_opportunity(
        self,
        market: Market,
        yes_price: float,
        no_price: float,
        ai_prob: float,
        ai_confidence: float,
    ) -> Optional[MarketMakingOpportunity]:
        """Calculate optimal market-making prices and expected profits."""
        try:
            yes_edge = ai_prob - yes_price
            no_edge = (1 - ai_prob) - no_price
            volatility = self._estimate_volatility(yes_price, market)

            base_spread = max(
                self.min_spread,
                min(self.max_spread, volatility * self.volatility_multiplier),
            )
            edge_adjustment = abs(yes_edge) * ai_confidence
            adjusted_spread = base_spread * (1 + edge_adjustment)

            if yes_edge > 0:
                optimal_yes_bid = yes_price + (adjusted_spread / 2)
                optimal_yes_ask = yes_price + adjusted_spread
                optimal_no_bid = no_price - adjusted_spread
                optimal_no_ask = no_price - (adjusted_spread / 2)
            else:
                optimal_yes_bid = yes_price - adjusted_spread
                optimal_yes_ask = yes_price - (adjusted_spread / 2)
                optimal_no_bid = no_price + (adjusted_spread / 2)
                optimal_no_ask = no_price + adjusted_spread

            optimal_yes_bid = max(0.01, min(0.99, optimal_yes_bid))
            optimal_yes_ask = max(0.01, min(0.99, optimal_yes_ask))
            optimal_no_bid = max(0.01, min(0.99, optimal_no_bid))
            optimal_no_ask = max(0.01, min(0.99, optimal_no_ask))

            yes_spread_profit = (optimal_yes_ask - optimal_yes_bid) * ai_confidence
            no_spread_profit = (optimal_no_ask - optimal_no_bid) * ai_confidence
            total_expected_profit = yes_spread_profit + no_spread_profit

            yes_size, no_size = self._calculate_optimal_sizes(
                yes_edge,
                no_edge,
                volatility,
                ai_confidence,
            )

            return MarketMakingOpportunity(
                market_id=market.market_id,
                market_title=market.title,
                current_yes_price=yes_price,
                current_no_price=no_price,
                ai_predicted_prob=ai_prob,
                ai_confidence=ai_confidence,
                optimal_yes_bid=optimal_yes_bid,
                optimal_yes_ask=optimal_yes_ask,
                optimal_no_bid=optimal_no_bid,
                optimal_no_ask=optimal_no_ask,
                yes_spread_profit=yes_spread_profit,
                no_spread_profit=no_spread_profit,
                total_expected_profit=total_expected_profit,
                inventory_risk=volatility,
                volatility_estimate=volatility,
                optimal_yes_size=yes_size,
                optimal_no_size=no_size,
            )
        except Exception as e:
            self.logger.error(f"Error calculating opportunity for {market.market_id}: {e}")
            return None

    def _estimate_volatility(self, price: float, market: Market) -> float:
        """Estimate market volatility based on price level and time to expiry."""
        try:
            if getattr(market, "expiration_ts", None):
                expiry_time = datetime.fromtimestamp(market.expiration_ts)
                time_to_expiry = (expiry_time - datetime.now()).total_seconds() / 86400
                time_to_expiry = max(0.1, time_to_expiry)
            else:
                time_to_expiry = 7.0

            intrinsic_vol = np.sqrt(price * (1 - price) / time_to_expiry)
            return max(0.01, min(0.20, intrinsic_vol))
        except Exception as e:
            self.logger.error(f"Error estimating volatility: {e}")
            return 0.05

    def _calculate_optimal_sizes(
        self,
        yes_edge: float,
        no_edge: float,
        volatility: float,
        confidence: float,
    ) -> Tuple[int, int]:
        """Calculate optimal position sizes using Kelly-style sizing."""
        del volatility
        try:
            available_capital = getattr(settings.trading, "max_position_size", 1000)

            if yes_edge > 0:
                win_prob = 0.5 + (yes_edge * confidence)
                kelly_yes = max(0, min(0.25, (win_prob - 0.5) / 0.5))
                yes_size = int(available_capital * kelly_yes)
            else:
                yes_size = int(available_capital * 0.05)

            if no_edge > 0:
                win_prob = 0.5 + (no_edge * confidence)
                kelly_no = max(0, min(0.25, (win_prob - 0.5) / 0.5))
                no_size = int(available_capital * kelly_no)
            else:
                no_size = int(available_capital * 0.05)

            return max(10, yes_size), max(10, no_size)
        except Exception as e:
            self.logger.error(f"Error calculating sizes: {e}")
            return 50, 50

    async def execute_market_making_strategy(
        self,
        opportunities: List[MarketMakingOpportunity],
    ) -> Dict:
        """Execute the market-making strategy on top opportunities."""
        results = {
            "orders_placed": 0,
            "total_exposure": 0.0,
            "expected_profit": 0.0,
            "markets_count": 0,
            "paper_entries_filled": 0,
            "paper_exits_filled": 0,
            "realized_pnl": 0.0,
        }

        if not getattr(settings.trading, "live_trading_enabled", False):
            reconciliation = await self.reconcile_persisted_paper_orders()
            results["paper_entries_filled"] += int(reconciliation.get("entries_filled", 0))
            results["paper_exits_filled"] += int(reconciliation.get("exits_filled", 0))
            results["realized_pnl"] += float(reconciliation.get("realized_pnl", 0.0))

        max_markets = getattr(settings.trading, "max_concurrent_markets", 10)
        top_opportunities = opportunities[:max_markets]

        for opportunity in top_opportunities:
            try:
                placed_orders = await self._place_market_making_orders(opportunity)
                if placed_orders > 0:
                    results["orders_placed"] += placed_orders
                    results["total_exposure"] += (
                        opportunity.optimal_yes_size + opportunity.optimal_no_size
                    )
                    results["expected_profit"] += opportunity.total_expected_profit
                    results["markets_count"] += 1

                self.logger.info(
                    f"Market making orders placed for {opportunity.market_title}: "
                    f"expected profit ${opportunity.total_expected_profit:.2f}"
                )
            except Exception as e:
                self.logger.error(f"Error executing market making for {opportunity.market_id}: {e}")
                continue

        if not getattr(settings.trading, "live_trading_enabled", False):
            reconciliation = await self.reconcile_persisted_paper_orders()
            results["paper_entries_filled"] += int(reconciliation.get("entries_filled", 0))
            results["paper_exits_filled"] += int(reconciliation.get("exits_filled", 0))
            results["realized_pnl"] += float(reconciliation.get("realized_pnl", 0.0))

        return results

    async def _place_market_making_orders(self, opportunity: MarketMakingOpportunity) -> int:
        """Place the buy-side market-making orders for one opportunity."""
        orders = [
            LimitOrder(
                market_id=opportunity.market_id,
                side="YES",
                action="buy",
                price=opportunity.optimal_yes_bid,
                quantity=opportunity.optimal_yes_size,
                expected_profit=opportunity.yes_spread_profit,
                target_price=opportunity.optimal_yes_ask,
            ),
            LimitOrder(
                market_id=opportunity.market_id,
                side="NO",
                action="buy",
                price=opportunity.optimal_no_bid,
                quantity=opportunity.optimal_no_size,
                expected_profit=opportunity.no_spread_profit,
                target_price=opportunity.optimal_no_ask,
            ),
        ]

        placed_orders = 0
        for order in orders:
            if await self._place_limit_order(order):
                placed_orders += 1

        resting_orders = [order for order in orders if order.status == "resting"]
        if resting_orders:
            self.active_orders.setdefault(opportunity.market_id, []).extend(resting_orders)
        return placed_orders

    async def _place_limit_order(self, order: LimitOrder) -> bool:
        """Place a limit order with the exchange or persist it locally in paper mode."""
        try:
            live_mode = getattr(settings.trading, "live_trading_enabled", False)

            if live_mode:
                import uuid

                client_order_id = str(uuid.uuid4())
                order_params = {
                    "ticker": order.market_id,
                    "client_order_id": client_order_id,
                    "side": order.side.lower(),
                    "action": order.action.lower(),
                    "count": order.quantity,
                    "type_": "limit",
                    "time_in_force": "good_till_canceled",
                    **build_limit_order_price_fields(order.side, order.price),
                }
                response = await self.kalshi_client.place_order(**order_params)

                if response and "order" in response:
                    order.status = "resting"
                    order.placed_at = datetime.now()
                    order.order_id = response["order"].get("order_id", client_order_id)
                    self.logger.info(
                        f"LIVE limit order placed: {order.action.upper()} {order.side} {order.quantity} "
                        f"at ${order.price:.4f} for market {order.market_id} "
                        f"(Order ID: {order.order_id})"
                    )
                    return True

                self.logger.error(f"Failed to place live order: {response}")
                order.status = "failed"
                return False

            existing_order = await self._find_resting_paper_order(
                market_id=order.market_id,
                side=order.side,
                action=order.action,
            )
            if existing_order:
                self.logger.info(
                    "Paper market-making order already resting for %s %s %s; skipping duplicate placement.",
                    order.market_id,
                    order.action.upper(),
                    order.side,
                )
                order.status = "cancelled"
                return False

            order.status = "resting"
            order.placed_at = datetime.now()
            order.order_id = (
                f"sim_{order.action}_{order.market_id}_{order.side}_{int(datetime.now().timestamp())}"
            )
            await self._persist_paper_order(order)
            self.logger.info(
                f"SIMULATED limit order placed: {order.action.upper()} {order.side} {order.quantity} "
                f"at ${order.price:.4f} for market {order.market_id}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Error placing limit order: {e}")
            order.status = "failed"
            return False

    async def _get_ai_analysis(self, market: Market) -> Optional[Dict]:
        """Get AI analysis for market-making edge calculation."""
        try:
            prompt = f"""
            MARKET MAKING ANALYSIS REQUEST

            Market: {market.title}

            Provide a quick assessment for market making in JSON format:
            {{
                "probability": [0.0-1.0 probability estimate],
                "confidence": [0.0-1.0 confidence level],
                "volatility_factors": "brief description",
                "stability": [0.0-1.0 price stability estimate]
            }}

            Focus on probability estimate and confidence in that estimate.
            """

            response = await self.xai_client.get_completion(
                prompt,
                max_tokens=3000,
                temperature=0.1,
            )

            if response is None:
                self.logger.info(
                    f"AI analysis unavailable for {market.market_id}, using conservative defaults"
                )
                return {
                    "probability": 0.5,
                    "confidence": 0.2,
                    "volatility_factors": "API unavailable",
                    "stability": 0.3,
                }

            try:
                import json
                import re

                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    parsed_response = json.loads(json_match.group(0))
                    probability = parsed_response.get("probability")
                    confidence = parsed_response.get("confidence")
                    if (
                        isinstance(probability, (int, float))
                        and 0 <= probability <= 1
                        and isinstance(confidence, (int, float))
                        and 0 <= confidence <= 1
                    ):
                        return parsed_response
            except (json.JSONDecodeError, ValueError) as exc:
                self.logger.warning(f"Failed to parse AI response for {market.market_id}: {exc}")

            self.logger.warning(
                f"AI analysis failed for {market.market_id}, using conservative defaults"
            )
            return {
                "probability": 0.5,
                "confidence": 0.3,
                "volatility_factors": "AI analysis failed",
                "stability": 0.5,
            }
        except Exception as e:
            self.logger.error(f"Error getting AI analysis: {e}")
            return {
                "probability": 0.5,
                "confidence": 0.3,
                "volatility_factors": "Error in analysis",
                "stability": 0.5,
            }

    async def monitor_and_update_orders(self):
        """Refresh the in-memory view of active paper orders and check competitiveness."""
        if not getattr(settings.trading, "live_trading_enabled", False):
            await self.reconcile_persisted_paper_orders()

        for market_id, orders in list(self.active_orders.items()):
            try:
                for order in list(orders):
                    if order.status != "resting":
                        continue
                    should_update = await self._should_update_order(order)
                    if should_update:
                        await self._update_order(order)
            except Exception as e:
                self.logger.error(f"Error monitoring orders for {market_id}: {e}")

    async def _should_update_order(self, order: LimitOrder) -> bool:
        """Determine if an order should be updated based on market conditions."""
        try:
            market_data = await self.kalshi_client.get_market(order.market_id)
            market_info = market_data.get("market", {}) if isinstance(market_data, dict) else {}
            if not market_info:
                return False

            if order.action == "buy":
                current_reference = get_mid_price(market_info, order.side)
            else:
                current_reference = get_best_bid_price(market_info, order.side)

            price_diff = abs(current_reference - order.price)
            return price_diff > 0.05
        except Exception as e:
            self.logger.error(f"Error checking order update: {e}")
            return False

    async def _update_order(self, order: LimitOrder):
        """Mark an order cancelled when it is no longer competitive."""
        try:
            order.status = "cancelled"
            if order.id is not None and not getattr(settings.trading, "live_trading_enabled", False):
                await self.db_manager.update_simulated_order(int(order.id), status="cancelled")
            self.logger.info(f"Updated order {order.order_id}")
        except Exception as e:
            self.logger.error(f"Error updating order: {e}")

    def get_performance_summary(self) -> Dict:
        """Get performance summary of the market-making strategy."""
        try:
            active_count = sum(len(orders) for orders in self.active_orders.values())
            filled_count = len(self.filled_orders)
            return {
                "total_pnl": self.total_pnl,
                "active_orders": active_count,
                "filled_orders": filled_count,
                "markets_traded": self.markets_traded,
                "win_rate": self.win_rate,
                "total_volume": self.total_volume,
            }
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}


async def run_market_making_strategy(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    xai_client: Any,
) -> Dict:
    """Main entry point for the market-making strategy."""
    logger = get_trading_logger("market_making_main")

    try:
        market_maker = AdvancedMarketMaker(db_manager, kalshi_client, xai_client)
        markets = await db_manager.get_eligible_markets(
            volume_min=30000,
            max_days_to_expiry=365,
        )

        if not markets:
            logger.warning("No eligible markets found for market making")
            return {"error": "No markets available"}

        logger.info(f"Analyzing {len(markets)} markets for market making opportunities")
        opportunities = await market_maker.analyze_market_making_opportunities(markets)

        if not opportunities:
            logger.warning("No profitable market making opportunities found")
            return {"opportunities": 0}

        logger.info(f"Found {len(opportunities)} profitable market making opportunities")
        results = await market_maker.execute_market_making_strategy(opportunities)
        results["performance"] = market_maker.get_performance_summary()
        logger.info(f"Market making strategy completed: {results}")
        return results
    except Exception as e:
        logger.error(f"Error in market making strategy: {e}")
        return {"error": str(e)}
