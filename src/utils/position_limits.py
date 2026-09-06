"""Position count and sizing guardrails for the trading system.

This module is intentionally aligned with ``cash_reserves.py`` so the two
safety layers cannot disagree about protected cash.

Environment variables:
- MAX_OPEN_POSITIONS (default: 15)
- MAX_POSITION_SIZE_PCT (default: 5.0)
- EMERGENCY_POSITION_LIMIT (default: 20)
- MIN_CASH_RESERVE_PCT (default: 33.0)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import os

from src.clients.kalshi_client import KalshiClient
from src.utils.database import DatabaseManager, Position
from src.utils.kalshi_normalization import (
    get_balance_dollars,
    get_portfolio_value_dollars,
    get_position_exposure_dollars,
)
from src.utils.logging_setup import get_trading_logger


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class PositionLimitResult:
    can_trade: bool
    reason: str
    current_positions: int
    max_positions: int
    current_portfolio_usage: float
    max_position_size: float
    recommended_actions: List[str]


@dataclass
class PositionToClose:
    position_id: int
    market_id: str
    side: str
    current_pnl: float
    confidence: float
    age_hours: float
    priority_score: float


class PositionLimitsManager:
    """Centralized position-count, position-size, and deployment limits."""

    def __init__(self, db_manager: DatabaseManager, kalshi_client: KalshiClient):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.logger = get_trading_logger("position_limits")

        self.max_positions = max(1, _env_int("MAX_OPEN_POSITIONS", 15))
        self.max_position_size_pct = max(0.1, _env_float("MAX_POSITION_SIZE_PCT", 5.0))
        self.warning_threshold = max(1, self.max_positions - 3)
        self.emergency_position_limit = max(
            self.max_positions + 1,
            _env_int("EMERGENCY_POSITION_LIMIT", 20),
        )
        self.min_cash_reserve_pct = max(0.0, _env_float("MIN_CASH_RESERVE_PCT", 33.0))

    @property
    def max_portfolio_usage_pct(self) -> float:
        """Maximum deployable share after protecting the reserve."""
        return max(0.0, 100.0 - self.min_cash_reserve_pct)

    async def check_position_limits(
        self,
        proposed_position_size: float,
        portfolio_value: Optional[float] = None,
    ) -> PositionLimitResult:
        try:
            proposed_position_size = max(0.0, float(proposed_position_size))
            if portfolio_value is None:
                portfolio_value = await self._get_portfolio_value()
            if portfolio_value <= 0:
                raise ValueError("portfolio value must be positive")

            current_positions = await self._get_position_count()
            current_usage = await self._calculate_portfolio_usage(portfolio_value)
            proposed_position_pct = proposed_position_size / portfolio_value * 100.0
            max_position_size = portfolio_value * self.max_position_size_pct / 100.0
            projected_usage = current_usage + proposed_position_pct

            recommendations: List[str] = []
            can_trade = True
            reason = "Position limits satisfied"

            if current_positions >= self.emergency_position_limit:
                can_trade = False
                reason = (
                    f"Emergency position count {current_positions} at/above "
                    f"{self.emergency_position_limit}"
                )
                recommendations.append("HALT new positions and reduce exposure")
            elif current_positions >= self.max_positions:
                can_trade = False
                reason = f"Position count {current_positions} at/above limit {self.max_positions}"
                recommendations.append(
                    f"Close at least {current_positions - self.max_positions + 1} position(s) before adding another"
                )
            elif current_positions >= self.warning_threshold:
                recommendations.append(
                    f"Approaching position limit ({current_positions}/{self.max_positions})"
                )

            if proposed_position_size > max_position_size + 1e-9:
                can_trade = False
                reason = (
                    f"Position size ${proposed_position_size:.2f} exceeds "
                    f"${max_position_size:.2f} ({self.max_position_size_pct:.1f}%) cap"
                )
                recommendations.append(f"Reduce position to at most ${max_position_size:.2f}")

            if projected_usage > self.max_portfolio_usage_pct + 1e-9:
                can_trade = False
                reason = (
                    f"Projected portfolio usage {projected_usage:.1f}% exceeds "
                    f"{self.max_portfolio_usage_pct:.1f}% deployable limit"
                )
                recommendations.append(
                    f"Keep {self.min_cash_reserve_pct:.1f}% of portfolio protected as cash"
                )

            available_cash = await self._get_available_cash()
            cash_after_trade = available_cash - proposed_position_size
            minimum_cash = portfolio_value * self.min_cash_reserve_pct / 100.0
            if cash_after_trade < minimum_cash - 1e-9:
                can_trade = False
                reason = (
                    f"Trade would leave ${cash_after_trade:.2f} cash, below protected "
                    f"minimum ${minimum_cash:.2f}"
                )
                recommendations.append(
                    f"Maintain at least {self.min_cash_reserve_pct:.1f}% protected cash"
                )

            if can_trade and not recommendations:
                recommendations.append("Position sizing and protected reserve checks passed")

            return PositionLimitResult(
                can_trade=can_trade,
                reason=reason,
                current_positions=current_positions,
                max_positions=self.max_positions,
                current_portfolio_usage=current_usage,
                max_position_size=max_position_size,
                recommended_actions=recommendations,
            )
        except Exception as exc:
            self.logger.error(f"Error checking position limits: {exc}")
            return PositionLimitResult(
                can_trade=False,
                reason=f"Position safety check failed closed: {exc}",
                current_positions=0,
                max_positions=self.max_positions,
                current_portfolio_usage=0.0,
                max_position_size=0.0,
                recommended_actions=["HALT new positions until risk state can be verified"],
            )

    async def enforce_position_limits(self, force_closure: bool = False) -> Dict[str, Any]:
        """Reduce locally tracked open-position count when limits are exceeded.

        Note: this method preserves the repository's existing behavior and marks
        selected local positions closed. Exchange-side order/position exit logic
        remains the responsibility of the execution layer.
        """
        try:
            current_positions = await self._get_position_count()
            target = self.warning_threshold if force_closure else self.max_positions
            positions_to_close = max(0, current_positions - target)
            if positions_to_close == 0:
                return {
                    "action": "no_action_needed",
                    "current_positions": current_positions,
                    "message": "Position count within limits",
                }

            candidates = await self._get_positions_for_closure(positions_to_close)
            closed: List[str] = []
            for candidate in candidates:
                if await self._close_position(candidate):
                    closed.append(candidate.market_id)

            return {
                "action": "positions_closed" if closed else "no_positions_closed",
                "positions_closed": len(closed),
                "closed_markets": closed,
                "remaining_positions": current_positions - len(closed),
                "message": f"Closed {len(closed)} locally tracked position(s) to enforce limits",
            }
        except Exception as exc:
            self.logger.error(f"Error enforcing position limits: {exc}")
            return {"action": "error", "message": str(exc)}

    async def get_position_limits_status(self) -> Dict[str, Any]:
        try:
            current_positions = await self._get_position_count()
            portfolio_value = await self._get_portfolio_value()
            portfolio_usage = await self._calculate_portfolio_usage(portfolio_value)
            available_cash = await self._get_available_cash()

            if current_positions >= self.emergency_position_limit:
                status = "EMERGENCY"
            elif current_positions >= self.max_positions or portfolio_usage > self.max_portfolio_usage_pct:
                status = "OVER_LIMIT"
            elif current_positions >= self.warning_threshold:
                status = "WARNING"
            else:
                status = "HEALTHY"

            cash_reserve_pct = available_cash / portfolio_value * 100.0 if portfolio_value > 0 else 0.0
            return {
                "status": status,
                "current_positions": current_positions,
                "max_positions": self.max_positions,
                "position_utilization": f"{current_positions}/{self.max_positions}",
                "portfolio_usage_pct": portfolio_usage,
                "max_portfolio_usage_pct": self.max_portfolio_usage_pct,
                "available_cash": available_cash,
                "portfolio_value": portfolio_value,
                "max_position_size": portfolio_value * self.max_position_size_pct / 100.0,
                "cash_reserve_pct": cash_reserve_pct,
                "minimum_cash_reserve_pct": self.min_cash_reserve_pct,
                "recommendations": self._get_status_recommendations(current_positions, portfolio_usage),
            }
        except Exception as exc:
            self.logger.error(f"Error getting position limits status: {exc}")
            return {
                "status": "ERROR",
                "message": str(exc),
                "position_utilization": "unknown",
                "recommendations": ["HALT new positions until risk state can be verified"],
            }

    async def _get_position_count(self) -> int:
        positions = await self.db_manager.get_open_positions()
        return len(positions)

    async def _get_portfolio_value(self) -> float:
        balance = await self.kalshi_client.get_balance()
        available_cash = get_balance_dollars(balance)
        marked_value = get_portfolio_value_dollars(balance)
        if marked_value > 0:
            return available_cash + marked_value

        positions_response = await self.kalshi_client.get_positions()
        positions = positions_response.get("event_positions", []) if isinstance(positions_response, dict) else []
        position_value = sum(
            get_position_exposure_dollars(position)
            for position in positions
            if isinstance(position, dict)
        )
        return available_cash + position_value

    async def _get_available_cash(self) -> float:
        balance = await self.kalshi_client.get_balance()
        return get_balance_dollars(balance)

    async def _calculate_portfolio_usage(self, portfolio_value: float) -> float:
        if portfolio_value <= 0:
            return 0.0
        available_cash = await self._get_available_cash()
        used_capital = max(0.0, portfolio_value - available_cash)
        return used_capital / portfolio_value * 100.0

    async def _get_positions_for_closure(self, count: int) -> List[PositionToClose]:
        positions = await self.db_manager.get_open_positions()
        candidates: List[PositionToClose] = []
        for position in positions:
            priority = await self._calculate_closure_priority(position)
            age_hours = max(0.0, (datetime.now() - position.timestamp).total_seconds() / 3600.0)
            candidates.append(
                PositionToClose(
                    position_id=position.id,
                    market_id=position.market_id,
                    side=position.side,
                    current_pnl=0.0,
                    confidence=float(position.confidence or 0.5),
                    age_hours=age_hours,
                    priority_score=priority,
                )
            )
        candidates.sort(key=lambda item: item.priority_score, reverse=True)
        return candidates[: max(0, count)]

    async def _calculate_closure_priority(self, position: Position) -> float:
        priority = 0.0
        confidence = float(position.confidence or 0.5)
        if confidence < 0.6:
            priority += 3.0
        elif confidence < 0.7:
            priority += 1.0

        age_hours = max(0.0, (datetime.now() - position.timestamp).total_seconds() / 3600.0)
        if age_hours > 72:
            priority += 2.0
        elif age_hours > 24:
            priority += 1.0

        position_value = float(position.quantity) * float(position.entry_price)
        if position_value > 50:
            priority += 1.0
        if not position.stop_loss_price:
            priority += 2.0
        return priority

    async def _close_position(self, candidate: PositionToClose) -> bool:
        try:
            await self.db_manager.update_position_status(candidate.position_id, "closed")
            self.logger.info(
                f"Position {candidate.market_id} marked closed by position-limit enforcement"
            )
            return True
        except Exception as exc:
            self.logger.error(f"Error closing position {candidate.market_id}: {exc}")
            return False

    def _get_status_recommendations(self, positions: int, usage: float) -> List[str]:
        recommendations: List[str] = []
        if positions >= self.emergency_position_limit:
            recommendations.append("EMERGENCY: halt new positions and reduce exposure")
        elif positions >= self.max_positions:
            recommendations.append(
                f"Close at least {positions - self.max_positions + 1} position(s) before adding another"
            )
        elif positions >= self.warning_threshold:
            recommendations.append(f"Approaching position limit ({positions}/{self.max_positions})")

        if usage > self.max_portfolio_usage_pct:
            recommendations.append(
                f"Portfolio deployment exceeds {self.max_portfolio_usage_pct:.1f}%; restore protected reserve"
            )
        elif usage > self.max_portfolio_usage_pct - 5:
            recommendations.append("Portfolio deployment is near the protected-reserve boundary")

        if not recommendations:
            recommendations.append("Position limits healthy")
        return recommendations


async def check_can_add_position(
    position_size: float,
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
) -> Tuple[bool, str]:
    manager = PositionLimitsManager(db_manager, kalshi_client)
    result = await manager.check_position_limits(position_size)
    return result.can_trade, result.reason


async def enforce_limits_if_needed(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
) -> bool:
    manager = PositionLimitsManager(db_manager, kalshi_client)
    current_count = await manager._get_position_count()
    if current_count > manager.max_positions:
        result = await manager.enforce_position_limits()
        return result.get("action") == "positions_closed"
    return False


async def get_max_position_size(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
) -> float:
    manager = PositionLimitsManager(db_manager, kalshi_client)
    portfolio_value = await manager._get_portfolio_value()
    return portfolio_value * manager.max_position_size_pct / 100.0
