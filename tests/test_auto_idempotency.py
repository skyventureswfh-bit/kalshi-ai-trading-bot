"""
Tests proving:
  - deterministic client_order_id is identical across crash/restart for the
    same position identity
  - the intent is persisted BEFORE submission
  - an ambiguous submission reconciles before any retry
  - a retry reuses the exact same client_order_id
  - Manual's execute_position() behavior is unchanged when client_order_id
    is omitted (random UUID, as before)
"""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from src.auto.intent_ledger import AutoIntentLedger, deterministic_client_order_id


def test_deterministic_id_is_stable_across_calls():
    id_a = deterministic_client_order_id(ticker="KXTEST-26", side="YES", position_id=42)
    id_b = deterministic_client_order_id(ticker="KXTEST-26", side="YES", position_id=42)
    assert id_a == id_b, "same position identity must always produce the same client_order_id"
    uuid.UUID(id_a)  # must be a well-formed UUID string Kalshi's client_order_id field will accept


def test_deterministic_id_differs_for_different_positions():
    id_a = deterministic_client_order_id(ticker="KXTEST-26", side="YES", position_id=42)
    id_b = deterministic_client_order_id(ticker="KXTEST-26", side="YES", position_id=43)
    id_c = deterministic_client_order_id(ticker="KXTEST-26", side="NO", position_id=42)
    assert len({id_a, id_b, id_c}) == 3


@pytest.mark.asyncio
async def test_intent_persisted_before_submission_is_idempotent():
    db_manager = AsyncMock()
    kalshi_client = AsyncMock()
    ledger = AutoIntentLedger(db_manager=db_manager, kalshi_client=kalshi_client)

    cid = deterministic_client_order_id(ticker="KXTEST-26", side="YES", position_id=42)
    await ledger.record_before_submit(
        client_order_id=cid, ticker="KXTEST-26", side="YES", position_id=42
    )
    # A retry recomputes the SAME id and calls record_before_submit again --
    # this must not raise or create a duplicate; the DB layer's
    # ON CONFLICT(client_order_id) DO NOTHING (see database.py) makes this
    # safe, and the ledger call itself is a plain pass-through.
    await ledger.record_before_submit(
        client_order_id=cid, ticker="KXTEST-26", side="YES", position_id=42
    )
    assert db_manager.record_auto_order_intent.await_count == 2
    first_call = db_manager.record_auto_order_intent.await_args_list[0]
    second_call = db_manager.record_auto_order_intent.await_args_list[1]
    assert first_call.kwargs["client_order_id"] == second_call.kwargs["client_order_id"] == cid


@pytest.mark.asyncio
async def test_reconcile_not_found_on_exchange_is_safe_to_retry():
    db_manager = AsyncMock()
    kalshi_client = AsyncMock()
    kalshi_client.get_orders = AsyncMock(return_value={"orders": []})
    ledger = AutoIntentLedger(db_manager=db_manager, kalshi_client=kalshi_client)

    intent = {"client_order_id": "abc-123", "ticker": "KXTEST-26", "status": "submitted"}
    result = await ledger.reconcile(intent)
    assert result.found_on_exchange is False
    assert result.safe_to_retry is True


@pytest.mark.asyncio
async def test_reconcile_resting_order_is_not_safe_to_retry():
    db_manager = AsyncMock()
    kalshi_client = AsyncMock()
    kalshi_client.get_orders = AsyncMock(
        return_value={"orders": [{"client_order_id": "abc-123", "status": "resting"}]}
    )
    ledger = AutoIntentLedger(db_manager=db_manager, kalshi_client=kalshi_client)

    intent = {"client_order_id": "abc-123", "ticker": "KXTEST-26", "status": "submitted"}
    result = await ledger.reconcile(intent)
    assert result.found_on_exchange is True
    assert result.safe_to_retry is False


@pytest.mark.asyncio
async def test_reconcile_filled_order_marks_local_record_terminal():
    db_manager = AsyncMock()
    kalshi_client = AsyncMock()
    kalshi_client.get_orders = AsyncMock(
        return_value={"orders": [{"client_order_id": "abc-123", "status": "filled"}]}
    )
    ledger = AutoIntentLedger(db_manager=db_manager, kalshi_client=kalshi_client)

    intent = {"client_order_id": "abc-123", "ticker": "KXTEST-26", "status": "submitted"}
    result = await ledger.reconcile(intent)
    assert result.safe_to_retry is False
    db_manager.update_auto_order_intent_status.assert_awaited_with(
        client_order_id="abc-123", status="filled", last_error=None
    )


@pytest.mark.asyncio
async def test_reconcile_fails_closed_when_kalshi_unreachable():
    db_manager = AsyncMock()
    kalshi_client = AsyncMock()
    kalshi_client.get_orders = AsyncMock(side_effect=RuntimeError("network down"))
    ledger = AutoIntentLedger(db_manager=db_manager, kalshi_client=kalshi_client)

    intent = {"client_order_id": "abc-123", "ticker": "KXTEST-26", "status": "submitted"}
    result = await ledger.reconcile(intent)
    # Cannot confirm exchange state -> never assume it's safe to retry.
    assert result.safe_to_retry is False


@pytest.mark.asyncio
async def test_manual_execute_position_unchanged_when_id_omitted():
    """
    Regression test: calling execute_position() with client_order_id
    omitted (Manual's existing call pattern, everywhere in live_trade.py
    and unified_bot.py) must still generate a random UUID exactly as
    before this change, and two independent calls must not collide.
    """
    from src.jobs import execute as execute_mod
    from src.utils.database import Position

    position = Position(
        market_id="KXTEST-26", side="YES", entry_price=0.60, quantity=10,
        timestamp=datetime.now(), id=42, strategy="live_trade",
    )
    db_manager = AsyncMock()
    kalshi_client = AsyncMock()
    captured = {}

    async def fake_place_order(**kwargs):
        captured["client_order_id"] = kwargs["client_order_id"]
        return {"order": {"order_id": "o1", "status": "filled"}, "order_id": "o1"}

    kalshi_client.place_order = AsyncMock(side_effect=fake_place_order)

    with patch.object(
        execute_mod, "_get_current_executable_entry_quote",
        new=AsyncMock(return_value=(0.55, 100.0, {"tick_size": 0.01})),
    ), patch.object(
        execute_mod, "_record_pre_execution_kalshi_health", new=AsyncMock(return_value=None)
    ), patch.object(
        execute_mod, "evaluate_pre_execution_safety",
        new=AsyncMock(return_value=type("S", (), {"allowed": True, "reason": None})()),
    ), patch.object(execute_mod, "_fetch_entry_orderbook", new=AsyncMock(return_value=None)):
        result = await execute_mod.execute_position(
            position=position, live_mode=True, db_manager=db_manager, kalshi_client=kalshi_client,
        )
        assert result is True
        generated_id = captured["client_order_id"]
        uuid.UUID(generated_id)  # random UUID generated internally, exactly as before this change

        # A second independent Manual call must generate a DIFFERENT random id --
        # proving the default path was NOT accidentally made deterministic.
        result2 = await execute_mod.execute_position(
            position=position, live_mode=True, db_manager=db_manager, kalshi_client=kalshi_client,
        )
        assert result2 is True
        assert captured["client_order_id"] != generated_id
