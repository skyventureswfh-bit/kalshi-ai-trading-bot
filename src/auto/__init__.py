"""
Beast Auto V1 — unattended supervisor for the existing live-trade pipeline.

This package contains ONLY the autonomous-loop wrapper. It does not
reimplement or alter any decision, sizing, or risk-gate logic — those
continue to live in src/jobs/live_trade.py, src/jobs/execute.py,
src/clients/kalshi_client.py, and the (not included in this handoff)
src/utils/execution_safety.py, cash_reserves.py, drawdown_guard.py,
position_limits.py, and src/strategies/portfolio_enforcer.py.

Auto reuses LiveTradeDecisionLoop exactly as Manual does, via its existing
execute_position_fn dependency-injection seam, so every existing safety gate
applies to Auto's orders automatically and unconditionally.
"""
