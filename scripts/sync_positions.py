#!/usr/bin/env python3
"""Audit Kalshi positions against the local database without mutating state.

This script used to delete every local position and rebuild the table from the
current Kalshi snapshot. That was unsafe because it could erase cost basis,
trade history context, stop-loss metadata, and other local execution state.
It also synthesized zero-P&L closes and replaced true entry prices with current
market prices.

The tool is now intentionally READ-ONLY. It reports mismatches so operators can
use the normal reconciliation/execution paths to repair them safely.
"""

import asyncio
import os
import sys
from typing import Dict, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clients.kalshi_client import KalshiClient
from src.utils.database import DatabaseManager
from src.utils.kalshi_normalization import get_position_size, get_position_ticker


def _exchange_position_map(positions_response: Dict) -> Dict[str, Tuple[str, float]]:
    """Return ticker -> (side, quantity) for non-zero exchange positions."""
    result: Dict[str, Tuple[str, float]] = {}
    for item in positions_response.get("market_positions", []) or []:
        ticker = get_position_ticker(item)
        size = float(get_position_size(item) or 0.0)
        if not ticker or abs(size) <= 1e-9:
            continue
        result[ticker] = ("YES" if size > 0 else "NO", abs(size))
    return result


async def audit_positions() -> bool:
    """Compare exchange exposure with local open positions; never write to DB."""
    kalshi_client = KalshiClient()
    db_manager = DatabaseManager()

    try:
        await db_manager.initialize()
        exchange = _exchange_position_map(await kalshi_client.get_positions())
        local_positions = await db_manager.get_open_positions()

        local: Dict[str, Tuple[str, float]] = {}
        for position in local_positions:
            ticker = str(getattr(position, "market_id", "") or "")
            if not ticker:
                continue
            local[ticker] = (
                str(getattr(position, "side", "") or "").upper(),
                float(getattr(position, "quantity", 0.0) or 0.0),
            )

        print("Position reconciliation audit (READ ONLY)")
        print(f"Exchange open positions: {len(exchange)}")
        print(f"Local open positions:    {len(local)}")

        mismatches = 0
        for ticker in sorted(set(exchange) | set(local)):
            exchange_state = exchange.get(ticker)
            local_state = local.get(ticker)
            if exchange_state == local_state:
                continue
            mismatches += 1
            print(
                f"MISMATCH {ticker}: exchange={exchange_state or 'none'} "
                f"local={local_state or 'none'}"
            )

        if mismatches:
            print(
                f"\n{mismatches} mismatch(es) found. No database rows were changed. "
                "Use the strategy-specific reconciliation path or confirmed exchange-exit "
                "workflow to repair state safely."
            )
            return False

        print("\nAudit clean. No database rows were changed.")
        return True
    except Exception as exc:
        print(f"Position audit failed closed: {exc}")
        return False
    finally:
        await kalshi_client.close()
        await db_manager.close()


async def sync_positions_to_database() -> bool:
    """Backward-compatible name; now performs the same read-only audit."""
    return await audit_positions()


async def verify_sync() -> bool:
    """Backward-compatible verification entry point; remains read-only."""
    return await audit_positions()


async def main() -> None:
    print("Position Database Sync Tool - SAFE AUDIT MODE")
    print("This command no longer deletes, closes, or rewrites positions.")
    success = await audit_positions()
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
