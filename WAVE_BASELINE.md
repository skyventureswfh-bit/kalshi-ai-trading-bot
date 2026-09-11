# Beast Wave Baseline

This is the locked, paper-first baseline for BTC 15-minute markets.

## Operating contract

- Observe every market; trade only a qualified wave.
- Maximum one entry per market, including after a restart.
- Use executable order-book prices, not the displayed probability alone.
- Enter UP or DOWN only after timed underlying and contract-price momentum agree.
- Entry ask must be 58–62 cents (60-cent target).
- Maximum capital is $10 including estimated entry fees; contract count is rounded down.
- Offer an exit when the executable bid reaches 89 cents.
- Offer a stop when immediate liquidation after estimated entry and exit fees
  would lose $3 or more.
- Exit at the executable bid with 45 seconds remaining rather than drifting
  into settlement when neither price rail has fired.
- Formula supplies 90% of the score. The Jeff Factor supplies 10% and may
  reject a marginal/dirty wave; it cannot bypass price, spread, time, or risk
  limits.
- Parameters remain fixed during a trade. Review changes only after a measured
  paper/replay batch.

## Deployment status

`src/auto/wave_strategy.py` is a deterministic decision core with no exchange
side effects. `src/auto/wave_paper.py` connects that core to synchronized
read-only observations and simulates immediate ask/bid fills. It remains
intentionally **not live-wired**. Before live use, the repository must add an
adapter that:

1. supplies synchronized BRTI and Kalshi bid/ask ticks;
2. persists and restores the one-trade market lock;
3. submits entries and exits through the audited guarded client with stable
   idempotency keys;
4. calls `confirm_entry` or `confirm_exit` only after confirmed fills;
5. records estimated and actual fees plus fills in the append-only journal;
6. passes session replay and paper-forward acceptance tests.

## Measurement

Track at least: qualified setups, skipped reasons, entries, wins, stopped
losses, average gross/net win, average gross/net loss, fees, slippage, and net
profit. The first promotion review should use a fixed sample (for example 100
paper-qualified trades), not a good or bad afternoon.

## Paper-forward command

With the existing read-only observation credentials configured:

```bash
python cli.py observe-btc --wave-paper
```

This listens to the live BTC market, prints only simulated entry/exit events,
records every decision to a Wave JSONL journal, and reports a fee-aware
scorecard after the market closes. It has no order-placement path.
