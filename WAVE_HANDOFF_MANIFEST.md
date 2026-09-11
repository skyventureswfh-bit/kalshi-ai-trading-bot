# Beast Wave Phase 6 — Paper-Forward Handoff

## What changed

The Wave core is now merged against the complete Beast repository and connected
to its authenticated, read-only BTC WebSocket observer. It still has no path to
place an order.

Base repository: `skyventureswfh-bit/kalshi-ai-trading-bot`, `main` tree
`5705e02e274a050c5cd987d87ec1139b3ece0a75`.

## Files

- `src/auto/wave_strategy.py` — deterministic momentum scoring, Jeff
  tie-breaker, fee-aware $10 sizing, fee-aware $3 stop, 89-cent profit exit,
  45-second time exit, one-trade lock, journal, and scorecard.
- `src/auto/wave_paper.py` — synchronized BRTI/order-book paper fills.
- `src/auto/observation_listener.py` — optional post-validation observation
  callback.
- `src/auto/capture_session.py` — `--wave-paper` and Wave journal support.
- `cli.py` — exposes the paper-forward flags through `observe-btc`.
- `tests/test_wave_strategy.py` — pure Wave rules.
- `tests/test_wave_paper.py` — feed synchronization and simulated fills.
- `tests/test_observation_callback.py` — validated callback forwarding.
- `WAVE_BASELINE.md` — operating contract and promotion requirements.

## Verification completed

- Python compilation passed for every changed Python file.
- 16 standard-library Wave/callback tests passed.
- `git diff --check` passed.
- The capture-session parser accepts `--wave-paper`.

The full repository's pytest safety suite could not run in the temporary build
machine because its development dependencies are not installed. Installing
them was blocked by the workspace network approval guard; that is an
environment limitation, not a skipped test result.

## Run safely

```bash
python cli.py observe-btc --wave-paper
```

This command consumes authenticated live market data and simulates entries and
exits. Do not connect the Wave engine to `place_order` yet.

## Still required before live money

1. Run the complete existing pytest suite in the normal Beast environment.
2. Capture a fixed paper-forward sample and review net results after estimated
   fees.
3. Persist the one-trade lock in the database and restore it after restarts.
4. Add deterministic, reconciled idempotency for sell orders.
5. Add live fill reconciliation and use actual reported fees/slippage.
6. Require a separate explicit live opt-in behind the existing kill switch and
   capital safety gate.
