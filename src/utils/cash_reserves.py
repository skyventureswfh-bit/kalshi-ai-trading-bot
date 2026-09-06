"""
Cash reserve, daily-loss, drawdown, and position-limit safety brakes for the trading system.

Defaults are intentionally conservative and can be adjusted with environment
variables. The reserve is treated as untouchable capital for new trades.

Environment variables:
- MIN_CASH_RESERVE_PCT (default: 33.0)
- OPTIMAL_CASH_RESERVE_PCT (default: 40.0)
- EMERGENCY_CASH_RESERVE_PCT (default: 25.0)
- CRITICAL_CASH_RESERVE_PCT (default: 15.0)
- MAX_SINGLE_TRADE_IMPACT_PCT (default: 5.0)
- DAILY_LOSS_CAP_PCT (default: 3.0)
- DAILY_LOSS_CAP_DOLLARS (default: 0; disabled when 0)
- MAX_DRAWDOWN_PCT (default: 15.0; enforced by DrawdownGuard)
- MAX_OPEN_POSITIONS (default: 15; enforced by PositionLimitsManager)
- MAX_POSITION_SIZE_PCT (default: 5.0; enforced by PositionLimitsManager)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os

import aiosqlite

from src.clients.kalshi_client import KalshiClient
from src.config.settings import settings
from src.utils.account_snapshot import AccountSafetySnapshot, get_account_safety_snapshot
from src.utils.database import DatabaseManager
from src.utils.drawdown_guard import check_drawdown_guard
from src.utils.logging_setup import get_trading_logger
from src.utils.position_limits import PositionLimitsManager


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class CashReserveResult:
    can_trade: bool
    reason: str
    current_cash: float
    portfolio_value: float
    cash_reserve_pct: float
    required_reserve_pct: float
    emergency_status: bool
    recommended_actions: List[str]


@dataclass
class CashEmergencyAction:
    action_type: str  # close_positions | halt_trading | raise_alert | no_action
    urgency: str
    positions_to_close: int
    expected_cash_freed: float
    reason: str


class CashReservesManager:
    """Central safety manager for reserve, loss, drawdown, and position limits."""

    def __init__(self, db_manager: DatabaseManager, kalshi_client: KalshiClient):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.logger = get_trading_logger("cash_reserves")

        self.minimum_reserve_pct = _env_float("MIN_CASH_RESERVE_PCT", 33.0)
        self.optimal_reserve_pct = _env_float("OPTIMAL_CASH_RESERVE_PCT", 40.0)
        self.emergency_threshold_pct = _env_float("EMERGENCY_CASH_RESERVE_PCT", 25.0)
        self.critical_threshold_pct = _env_float("CRITICAL_CASH_RESERVE_PCT", 15.0)

        self.max_single_trade_impact = _env_float("MAX_SINGLE_TRADE_IMPACT_PCT", 5.0)
        self.buffer_for_opportunities = _env_float("CASH_OPPORTUNITY_BUFFER_PCT", 2.0)
        self.daily_loss_cap_pct = max(0.0, _env_float("DAILY_LOSS_CAP_PCT", 3.0))
        self.daily_loss_cap_dollars = max(0.0, _env_float("DAILY_LOSS_CAP_DOLLARS", 0.0))

    def _daily_loss_limit_dollars(self, portfolio_value: float) -> float:
        limits: List[float] = []
        if self.daily_loss_cap_pct > 0 and portfolio_value > 0:
            limits.append(portfolio_value * self.daily_loss_cap_pct / 100.0)
        if self.daily_loss_cap_dollars > 0:
            limits.append(self.daily_loss_cap_dollars)
        return min(limits) if limits else 0.0

    async def _get_daily_realized_pnl(self) -> float:
        """Return today's realized P&L from local trade logs, split by live/paper mode."""
        live_mode = bool(getattr(settings.trading, "live_trading_enabled", False))
        try:
            async with aiosqlite.connect(self.db_manager.db_path) as db:
                cursor = await db.execute(
                    """
                    SELECT COALESCE(SUM(pnl), 0.0)
                    FROM trade_logs
                    WHERE date(exit_timestamp) = date('now')
                      AND live = ?
                    """,
                    (1 if live_mode else 0,),
                )
                row = await cursor.fetchone()
                return float(row[0] or 0.0) if row else 0.0
        except Exception as exc:
            self.logger.error(f"Unable to read daily realized P&L: {exc}")
            raise

    async def _daily_loss_status(self, portfolio_value: float) -> Dict[str, Any]:
        pnl = await self._get_daily_realized_pnl()
        limit = self._daily_loss_limit_dollars(portfolio_value)
        loss = max(0.0, -pnl)
        hit = bool(limit > 0 and loss >= limit)
        return {
            "daily_realized_pnl": pnl,
            "daily_realized_loss": loss,
            "daily_loss_limit": limit,
            "daily_loss_cap_pct": self.daily_loss_cap_pct,
            "daily_loss_cap_dollars": self.daily_loss_cap_dollars,
            "daily_loss_halt": hit,
        }

    async def check_cash_reserves(
        self,
        proposed_trade_value: float = 0.0,
        portfolio_value: Optional[float] = None,
        snapshot: Optional[AccountSafetySnapshot] = None,
    ) -> CashReserveResult:
        try:
            snapshot = snapshot or await get_account_safety_snapshot(self.kalshi_client)

            # One safety decision must use one equity number. Older callers may
            # still pass ``portfolio_value`` explicitly, so accept it only when
            # it agrees with the shared account snapshot. Mixing a caller value
            # for reserve math with snapshot cash/position data can make the same
            # trade look safe in one brake and unsafe in another.
            if portfolio_value is not None:
                supplied_value = float(portfolio_value)
                tolerance = max(0.01, abs(snapshot.portfolio_value) * 0.001)
                if abs(supplied_value - snapshot.portfolio_value) > tolerance:
                    raise ValueError(
                        "portfolio_value conflicts with shared account snapshot "
                        f"({supplied_value:.2f} vs {snapshot.portfolio_value:.2f})"
                    )
            portfolio_value = snapshot.portfolio_value

            current_cash = snapshot.available_cash
            current_reserve_pct = (current_cash / portfolio_value) * 100 if portfolio_value > 0 else 0.0
            cash_after_trade = current_cash - proposed_trade_value
            reserve_after_trade = (cash_after_trade / portfolio_value) * 100 if portfolio_value > 0 else 0.0
            daily = await self._daily_loss_status(portfolio_value)
            drawdown = await check_drawdown_guard(
                self.db_manager,
                self.kalshi_client,
                snapshot=snapshot,
            )
            position_limits = await PositionLimitsManager(
                self.db_manager,
                self.kalshi_client,
            ).check_position_limits(
                proposed_position_size=proposed_trade_value,
                portfolio_value=portfolio_value,
                available_cash=current_cash,
                snapshot=snapshot,
            )

            recommendations: List[str] = []
            can_trade = True
            emergency_status = False
            reason = "Capital safety checks passed"

            if not drawdown.can_trade:
                can_trade = False
                emergency_status = True
                reason = drawdown.reason
                recommendations.append("HALT all new risk until drawdown safety is cleared")
            elif daily["daily_loss_halt"]:
                can_trade = False
                emergency_status = True
                reason = (
                    f"DAILY LOSS HALT: realized loss ${daily['daily_realized_loss']:.2f} "
                    f"reached ${daily['daily_loss_limit']:.2f} limit"
                )
                recommendations.append("HALT all new trading until the next UTC trading day")
            elif current_reserve_pct < self.critical_threshold_pct:
                can_trade = False
                emergency_status = True
                reason = (
                    f"CRITICAL: cash reserves {current_reserve_pct:.1f}% below "
                    f"{self.critical_threshold_pct:.1f}% threshold"
                )
                recommendations.extend([
                    "HALT all new trading until reserves are restored",
                    "Review open positions before freeing capital",
                ])
            elif current_reserve_pct < self.emergency_threshold_pct:
                can_trade = False
                emergency_status = True
                reason = (
                    f"EMERGENCY: cash reserves {current_reserve_pct:.1f}% below "
                    f"{self.emergency_threshold_pct:.1f}% threshold"
                )
                recommendations.append("Suspend new trading and rebuild protected cash")
            elif reserve_after_trade < self.minimum_reserve_pct:
                can_trade = False
                reason = (
                    f"Trade would reduce protected cash to {reserve_after_trade:.1f}%, "
                    f"below {self.minimum_reserve_pct:.1f}% minimum"
                )
                recommendations.append("Reduce trade size; protected reserve is untouchable")
            elif current_reserve_pct < self.minimum_reserve_pct:
                can_trade = False
                reason = (
                    f"Current reserves {current_reserve_pct:.1f}% below protected "
                    f"minimum {self.minimum_reserve_pct:.1f}%"
                )
                recommendations.append("No new positions until protected reserve is restored")
            elif not position_limits.can_trade:
                can_trade = False
                reason = f"POSITION LIMIT HALT: {position_limits.reason}"
                recommendations.extend(position_limits.recommended_actions)

            trade_impact_pct = (proposed_trade_value / portfolio_value) * 100 if portfolio_value > 0 else 0.0
            if trade_impact_pct > self.max_single_trade_impact + 0.01:
                can_trade = False
                reason = (
                    f"Trade impact {trade_impact_pct:.1f}% exceeds "
                    f"{self.max_single_trade_impact:.1f}% per-trade cap"
                )
                recommendations.append(
                    f"Reduce trade to at most ${portfolio_value * self.max_single_trade_impact / 100:.2f}"
                )

            if can_trade and reserve_after_trade < self.minimum_reserve_pct + self.buffer_for_opportunities:
                recommendations.append("Warning: trade would leave little cash above the protected reserve")
            elif can_trade:
                recommendations.append("Safety checks passed")

            return CashReserveResult(
                can_trade=can_trade,
                reason=reason,
                current_cash=current_cash,
                portfolio_value=portfolio_value,
                cash_reserve_pct=current_reserve_pct,
                required_reserve_pct=self.minimum_reserve_pct,
                emergency_status=emergency_status,
                recommended_actions=recommendations,
            )
        except Exception as exc:
            self.logger.error(f"Error checking cash reserves: {exc}")
            return CashReserveResult(
                can_trade=False,
                reason=f"Safety check failed closed: {exc}",
                current_cash=0.0,
                portfolio_value=0.0,
                cash_reserve_pct=0.0,
                required_reserve_pct=self.minimum_reserve_pct,
                emergency_status=True,
                recommended_actions=["HALT new trading until safety state can be verified"],
            )

    async def handle_cash_emergency(self) -> CashEmergencyAction:
        try:
            snapshot = await get_account_safety_snapshot(self.kalshi_client)
            portfolio_value = snapshot.portfolio_value
            current_cash = snapshot.available_cash
            current_reserve_pct = (current_cash / portfolio_value) * 100 if portfolio_value > 0 else 0.0
            daily = await self._daily_loss_status(portfolio_value)
            drawdown = await check_drawdown_guard(
                self.db_manager,
                self.kalshi_client,
                snapshot=snapshot,
            )

            if not drawdown.can_trade:
                return CashEmergencyAction(
                    action_type="halt_trading",
                    urgency="critical",
                    positions_to_close=0,
                    expected_cash_freed=0.0,
                    reason=drawdown.reason,
                )

            if daily["daily_loss_halt"]:
                return CashEmergencyAction(
                    action_type="halt_trading",
                    urgency="critical",
                    positions_to_close=0,
                    expected_cash_freed=0.0,
                    reason=(
                        f"Daily realized loss ${daily['daily_realized_loss']:.2f} reached "
                        f"${daily['daily_loss_limit']:.2f} limit"
                    ),
                )

            required_cash = portfolio_value * self.minimum_reserve_pct / 100.0
            cash_shortfall = max(0.0, required_cash - current_cash)
            if current_reserve_pct >= self.minimum_reserve_pct:
                return CashEmergencyAction("no_action", "info", 0, 0.0, "Cash reserves adequate")

            if current_reserve_pct < self.critical_threshold_pct:
                action_type, urgency = "halt_trading", "critical"
            elif current_reserve_pct < self.emergency_threshold_pct:
                action_type, urgency = "close_positions", "critical"
            else:
                action_type, urgency = "raise_alert", "warning"

            positions = await self.db_manager.get_open_positions()
            positions_to_close = 0
            expected_cash_freed = 0.0
            if positions and cash_shortfall > 0:
                avg_position_value = sum(
                    max(0.0, float(getattr(p, "entry_price", 0.0)) * float(getattr(p, "quantity", 0.0)))
                    for p in positions
                ) / max(1, len(positions))
                if avg_position_value > 0:
                    positions_to_close = min(len(positions), max(1, int((cash_shortfall / avg_position_value) + 0.999)))
                    expected_cash_freed = positions_to_close * avg_position_value

            return CashEmergencyAction(
                action_type=action_type,
                urgency=urgency,
                positions_to_close=positions_to_close,
                expected_cash_freed=expected_cash_freed,
                reason=f"Need ${cash_shortfall:.2f} to restore {self.minimum_reserve_pct:.1f}% protected reserve",
            )
        except Exception as exc:
            self.logger.error(f"Error handling cash emergency: {exc}")
            return CashEmergencyAction(
                action_type="halt_trading",
                urgency="critical",
                positions_to_close=0,
                expected_cash_freed=0.0,
                reason=f"Safety state unavailable: {exc}",
            )

    async def get_cash_status(self) -> Dict[str, Any]:
        try:
            snapshot = await get_account_safety_snapshot(self.kalshi_client)
            portfolio_value = snapshot.portfolio_value
            current_cash = snapshot.available_cash
            current_reserve_pct = (current_cash / portfolio_value) * 100 if portfolio_value > 0 else 0.0
            daily = await self._daily_loss_status(portfolio_value)
            drawdown = await check_drawdown_guard(
                self.db_manager,
                self.kalshi_client,
                snapshot=snapshot,
            )

            if not drawdown.can_trade:
                status = "DRAWDOWN_HALT"
            elif daily["daily_loss_halt"]:
                status = "DAILY_LOSS_HALT"
            elif current_reserve_pct >= self.optimal_reserve_pct:
                status = "EXCELLENT"
            elif current_reserve_pct >= self.minimum_reserve_pct:
                status = "GOOD"
            elif current_reserve_pct >= self.emergency_threshold_pct:
                status = "WARNING"
            elif current_reserve_pct >= self.critical_threshold_pct:
                status = "EMERGENCY"
            else:
                status = "CRITICAL"

            minimum_cash = portfolio_value * self.minimum_reserve_pct / 100.0
            optimal_cash = portfolio_value * self.optimal_reserve_pct / 100.0
            emergency = bool(
                (not drawdown.can_trade)
                or daily["daily_loss_halt"]
                or current_reserve_pct < self.emergency_threshold_pct
            )

            return {
                "status": status,
                "current_cash": current_cash,
                "portfolio_value": portfolio_value,
                "reserve_percentage": current_reserve_pct,
                "minimum_required": self.minimum_reserve_pct,
                "optimal_target": self.optimal_reserve_pct,
                "cash_shortfall": max(0.0, minimum_cash - current_cash),
                "cash_to_optimal": max(0.0, optimal_cash - current_cash),
                "trading_permitted": (
                    current_reserve_pct >= self.minimum_reserve_pct
                    and not daily["daily_loss_halt"]
                    and drawdown.can_trade
                ),
                "emergency_status": emergency,
                "max_trade_size": max(0.0, current_cash - minimum_cash),
                "drawdown_pct": drawdown.drawdown_pct,
                "drawdown_limit_pct": drawdown.limit_pct,
                "high_watermark": drawdown.high_watermark,
                "drawdown_reason": drawdown.reason,
                "recommendations": (
                    ["DRAWDOWN HALT: no new risk until drawdown safety is cleared"]
                    if not drawdown.can_trade
                    else self._get_cash_recommendations(current_reserve_pct, daily)
                ),
                **daily,
            }
        except Exception as exc:
            self.logger.error(f"Error getting cash status: {exc}")
            return {
                "status": "ERROR",
                "message": str(exc),
                "trading_permitted": False,
                "emergency_status": True,
                "reserve_percentage": 0.0,
                "recommendations": ["HALT new trading until safety state can be verified"],
            }

    async def _get_portfolio_value(self) -> float:
        return (await get_account_safety_snapshot(self.kalshi_client)).portfolio_value

    async def _get_available_cash(self) -> float:
        return (await get_account_safety_snapshot(self.kalshi_client)).available_cash

    def _get_cash_recommendations(self, reserve_pct: float, daily: Optional[Dict[str, Any]] = None) -> List[str]:
        daily = daily or {}
        if daily.get("daily_loss_halt"):
            return ["DAILY LOSS CAP HIT: no new trades until the next UTC trading day"]
        if reserve_pct < self.critical_threshold_pct:
            return ["CRITICAL: halt new trading", "Restore protected cash before resuming"]
        if reserve_pct < self.emergency_threshold_pct:
            return ["EMERGENCY: suspend new trading", "Restore protected cash"]
        if reserve_pct < self.minimum_reserve_pct:
            return [f"Protected reserve below {self.minimum_reserve_pct:.1f}%", "No new positions"]
        if reserve_pct < self.optimal_reserve_pct:
            return ["Reserve is protected; keep sizing conservative"]
        return ["Protected reserve healthy", "Trading permitted within position limits"]


async def check_can_trade_with_cash_reserves(
    trade_value: float,
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
) -> Tuple[bool, str]:
    manager = CashReservesManager(db_manager, kalshi_client)
    result = await manager.check_cash_reserves(trade_value)
    return result.can_trade, result.reason


async def get_max_trade_size_for_reserves(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
) -> float:
    manager = CashReservesManager(db_manager, kalshi_client)
    status = await manager.get_cash_status()
    return float(status.get("max_trade_size", 0.0))


async def is_cash_emergency(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
) -> bool:
    manager = CashReservesManager(db_manager, kalshi_client)
    status = await manager.get_cash_status()
    return bool(status.get("emergency_status", True))
