"""Persistent portfolio drawdown guard.

Tracks the live/paper portfolio high-water mark in the local SQLite database and
halts new risk when the portfolio falls more than the configured percentage
from that peak. This is intentionally separate from modeled/estimated drawdown
used by portfolio optimization.

Environment variable:
- MAX_DRAWDOWN_PCT (default: 15.0)
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import aiosqlite

from src.clients.kalshi_client import KalshiClient
from src.config.settings import settings
from src.utils.database import DatabaseManager
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


@dataclass
class DrawdownStatus:
    can_trade: bool
    current_value: float
    high_watermark: float
    drawdown_pct: float
    limit_pct: float
    reason: str


class DrawdownGuard:
    """Fail-closed high-water-mark drawdown brake for new risk."""

    def __init__(self, db_manager: DatabaseManager, kalshi_client: KalshiClient):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.limit_pct = max(0.0, _env_float("MAX_DRAWDOWN_PCT", 15.0))
        self.logger = get_trading_logger("drawdown_guard")

    async def check(self) -> DrawdownStatus:
        try:
            current_value = await self._get_portfolio_value()
            if current_value <= 0:
                raise ValueError("portfolio value is zero or negative")

            live_mode = 1 if bool(getattr(settings.trading, "live_trading_enabled", False)) else 0
            async with aiosqlite.connect(self.db_manager.db_path) as db:
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS drawdown_guard_state (
                        live INTEGER PRIMARY KEY,
                        high_watermark REAL NOT NULL,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cursor = await db.execute(
                    "SELECT high_watermark FROM drawdown_guard_state WHERE live = ?",
                    (live_mode,),
                )
                row = await cursor.fetchone()
                high_watermark = float(row[0]) if row and row[0] is not None else current_value

                if current_value > high_watermark:
                    high_watermark = current_value

                await db.execute(
                    """
                    INSERT INTO drawdown_guard_state (live, high_watermark, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(live) DO UPDATE SET
                        high_watermark = excluded.high_watermark,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (live_mode, high_watermark),
                )
                await db.commit()

            drawdown_pct = (
                max(0.0, (high_watermark - current_value) / high_watermark * 100.0)
                if high_watermark > 0
                else 0.0
            )
            halted = bool(self.limit_pct > 0 and drawdown_pct >= self.limit_pct)
            reason = (
                f"DRAWDOWN HALT: portfolio is {drawdown_pct:.1f}% below high-water mark "
                f"(${current_value:.2f} vs ${high_watermark:.2f}), limit {self.limit_pct:.1f}%"
                if halted
                else f"Drawdown {drawdown_pct:.1f}% within {self.limit_pct:.1f}% limit"
            )
            return DrawdownStatus(
                can_trade=not halted,
                current_value=current_value,
                high_watermark=high_watermark,
                drawdown_pct=drawdown_pct,
                limit_pct=self.limit_pct,
                reason=reason,
            )
        except Exception as exc:
            self.logger.error("Drawdown safety state unavailable", error=str(exc))
            return DrawdownStatus(
                can_trade=False,
                current_value=0.0,
                high_watermark=0.0,
                drawdown_pct=0.0,
                limit_pct=self.limit_pct,
                reason=f"Drawdown safety failed closed: {exc}",
            )

    async def _get_portfolio_value(self) -> float:
        balance_response = await self.kalshi_client.get_balance()
        available_cash = get_balance_dollars(balance_response)
        marked_portfolio_value = get_portfolio_value_dollars(balance_response)
        if marked_portfolio_value > 0:
            return available_cash + marked_portfolio_value

        positions_response = await self.kalshi_client.get_positions()
        positions = positions_response.get("event_positions", []) if isinstance(positions_response, dict) else []
        position_value = sum(
            get_position_exposure_dollars(position)
            for position in positions
            if isinstance(position, dict)
        )
        return available_cash + position_value


async def check_drawdown_guard(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
) -> DrawdownStatus:
    """Convenience wrapper used by execution/risk gates."""
    return await DrawdownGuard(db_manager, kalshi_client).check()
