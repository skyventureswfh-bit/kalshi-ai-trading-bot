#!/usr/bin/env python3
"""
Kalshi AI Trading Bot -- Unified CLI

Provides a single entry point for all bot operations:
    python cli.py run          Start the trading bot
    python cli.py dashboard    Launch the Node dashboard stack
    python cli.py status       Show portfolio balance, positions, and P&L
    python cli.py backtest     Run backtests (placeholder)
    python cli.py health       Verify API connections, database, and configuration
"""

import argparse
import asyncio
import inspect
import os
import sys
from pathlib import Path


def _ensure_utf8_console() -> None:
    """Prefer UTF-8 console streams so icon output doesn't fail on Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            encoding = getattr(stream, "encoding", "") or ""
            if encoding.lower() != "utf-8":
                reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue


_ensure_utf8_console()


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

async def _run_with_runtime_guard(
    operation,
    *,
    label: str,
    max_runtime_seconds: int | None = None,
    on_timeout=None,
):
    """Run an async operation with an optional hard runtime cap."""
    try:
        if max_runtime_seconds and max_runtime_seconds > 0:
            return await asyncio.wait_for(operation(), timeout=max_runtime_seconds)
        return await operation()
    except asyncio.TimeoutError:
        if callable(on_timeout):
            on_timeout()
        print(
            f"\nReached the {label} runtime limit "
            f"({max_runtime_seconds} seconds). Shutting down cleanly."
        )
        return None


async def _run_bot_entrypoint(
    bot,
    *,
    once: bool = False,
    smoke: bool = False,
    max_runtime_seconds: int | None = None,
):
    """Run a bot entrypoint with optional safety modes and timeout guard."""
    if smoke:
        operation = bot.run_smoke_test
    elif once:
        operation = bot.run_single_cycle
    else:
        operation = bot.run
    return await _run_with_runtime_guard(
        operation,
        label="bot",
        max_runtime_seconds=max_runtime_seconds,
        on_timeout=getattr(bot, "request_shutdown", None),
    )


def _format_live_trade_loop_summary(summary) -> str:
    """Render a compact one-line summary for the live-trade loop."""
    skipped_reason = getattr(summary, "skipped_reason", None)
    line = (
        "Live-trade cycle "
        f"{getattr(summary, 'run_id', 'unknown')}: "
        f"events={getattr(summary, 'events_scanned', 0)}, "
        f"shortlisted={getattr(summary, 'shortlisted_events', 0)}, "
        f"specialists={getattr(summary, 'specialist_candidates', 0)}, "
        f"executed={getattr(summary, 'executed_positions', 0)}"
    )
    if skipped_reason:
        line += f" | skip={skipped_reason}"
    return line


async def _run_live_trade_loop_entrypoint(
    *,
    once: bool = False,
    max_runtime_seconds: int | None = None,
    live_mode: bool = False,
    shadow_mode: bool = False,
):
    """Run the live-trade decision loop directly from the CLI."""
    from src.clients.kalshi_client import KalshiClient
    from src.config.settings import settings as runtime_settings
    from src.jobs.live_trade import LiveTradeDecisionLoop
    from src.utils.database import DatabaseManager

    db_manager = DatabaseManager()
    kalshi_client = KalshiClient()
    loop = LiveTradeDecisionLoop(
        db_manager=db_manager,
        kalshi_client=kalshi_client,
        manage_quick_flip_positions_each_cycle=True,
    )
    previous_live_mode = bool(getattr(runtime_settings.trading, "live_trading_enabled", False))
    previous_shadow_mode = bool(getattr(runtime_settings.trading, "shadow_mode_enabled", False))
    runtime_settings.trading.live_trading_enabled = bool(live_mode)
    runtime_settings.trading.shadow_mode_enabled = bool(shadow_mode) and not bool(live_mode)
    interval_seconds = max(
        int(getattr(runtime_settings.trading, "market_scan_interval", 30) or 30),
        5,
    )

    async def _run_once():
        summary = await loop.run_once()
        print(_format_live_trade_loop_summary(summary))
        return summary

    async def _run_forever():
        while True:
            await _run_once()
            print(
                "Sleeping "
                f"{interval_seconds} seconds before the next live-trade cycle."
            )
            await asyncio.sleep(interval_seconds)

    try:
        initialize = getattr(db_manager, "initialize", None)
        if callable(initialize):
            result = initialize()
            if inspect.isawaitable(result):
                await result
        return await _run_with_runtime_guard(
            _run_once if once else _run_forever,
            label="live-trade loop",
            max_runtime_seconds=max_runtime_seconds,
        )
    finally:
        runtime_settings.trading.live_trading_enabled = previous_live_mode
        runtime_settings.trading.shadow_mode_enabled = previous_shadow_mode
        await _close_async_resource(loop)
        await _close_async_resource(kalshi_client)
        await _close_async_resource(db_manager)


def _run_live_trade_loop_command(
    *,
    once: bool = False,
    max_runtime_seconds: int | None = None,
    live_mode: bool = False,
    shadow_mode: bool = False,
) -> None:
    """Synchronous wrapper for the dedicated live-trade CLI path."""
    try:
        asyncio.run(
            _run_live_trade_loop_entrypoint(
                once=once,
                max_runtime_seconds=max_runtime_seconds,
                live_mode=live_mode,
                shadow_mode=shadow_mode,
            )
        )
    except KeyboardInterrupt:
        print("\nLive-trade loop stopped by user.")


def _resolve_db_path() -> Path:
    """Return the primary trading database path for CLI status/health helpers."""
    return Path(__file__).parent / "trading_system.db"


def _resolve_run_mode(args: argparse.Namespace) -> str:
    """Normalize mutually-exclusive run-mode flags to a single mode string."""
    live = bool(getattr(args, "live", False))
    paper = bool(getattr(args, "paper", False))
    shadow = bool(getattr(args, "shadow", False))

    selected_modes = [
        mode_name
        for mode_name, enabled in (
            ("live", live),
            ("paper", paper),
            ("shadow", shadow),
        )
        if enabled
    ]
    if len(selected_modes) > 1:
        print("Error: choose only one of --live, --paper, or --shadow.")
        sys.exit(1)

    if live:
        return "live"
    if shadow:
        return "shadow"
    return "paper"


def _class_accepts_kwarg(cls, arg_name: str) -> bool:
    """Return True when a class constructor explicitly accepts `arg_name`."""
    try:
        parameters = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return False
    return arg_name in parameters


def _construct_runtime(cls, **kwargs):
    """Instantiate a runtime object using only supported keyword arguments."""
    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if _class_accepts_kwarg(cls, key)
    }
    return cls(**supported_kwargs)


def _format_currency_or_placeholder(
    value: float | None,
    *,
    width: int = 10,
    placeholder: str = "unavailable",
) -> str:
    """Format dollars for status output, or a placeholder if unavailable."""
    if value is None:
        return f"{placeholder:>{width}}"
    return f"${value:>{width},.2f}"


def _format_summary_payload(summary) -> str | None:
    """Normalize helper output into a one-line status summary string."""
    if summary is None:
        return None
    if isinstance(summary, str):
        text = summary.strip()
        return text or None
    if isinstance(summary, dict):
        for key in ("summary", "line", "text", "message"):
            value = summary.get(key)
            if value:
                text = str(value).strip()
                if text:
                    return text
        return None
    text = str(summary).strip()
    return text or None


async def _maybe_call_db_summary_helper(
    db_manager,
    helper_names: tuple[str, ...],
) -> tuple[str | None, str | None]:
    """Call the first available DB helper that returns a printable summary."""
    for helper_name in helper_names:
        helper = getattr(db_manager, helper_name, None)
        if not callable(helper):
            continue
        try:
            result = helper()
            if inspect.isawaitable(result):
                result = await result
        except Exception:
            continue
        summary = _format_summary_payload(result)
        if summary:
            return summary, helper_name
    return None, None


async def _load_status_db_manager(db_path: Path):
    """Best-effort DB manager loader for resilient status output."""
    if not db_path.exists():
        return None

    try:
        from src.utils.database import DatabaseManager
    except ImportError:
        return None

    db_manager = DatabaseManager(db_path=str(db_path))
    initialize = getattr(db_manager, "initialize", None)
    if callable(initialize):
        try:
            result = initialize()
            if inspect.isawaitable(result):
                await result
        except Exception:
            # Status is read-mostly; a partial DB failure should not prevent
            # placeholder rendering for the rest of the command.
            return db_manager
    return db_manager


async def _close_async_resource(resource) -> None:
    """Close an async resource if it exposes a close coroutine/method."""
    if resource is None:
        return
    close = getattr(resource, "close", None)
    if not callable(close):
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception:
        return


def _configured_local_portfolio_floor() -> float:
    """Return a conservative local bankroll floor when the API is unavailable."""
    try:
        from src.config.settings import settings as runtime_settings

        return max(float(getattr(runtime_settings.trading, "min_balance", 0.0) or 0.0), 0.0)
    except Exception:
        return 0.0


def _estimate_local_position_cost_basis(position) -> float:
    """Estimate locally deployed capital for one persisted open position."""
    if isinstance(position, dict):
        getter = position.get
    else:
        getter = lambda key, default=None: getattr(position, key, default)

    def _as_float(value) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    contracts_cost = _as_float(getter("contracts_cost", 0.0))
    if contracts_cost > 0:
        return contracts_cost

    quantity = _as_float(getter("quantity", 0.0))
    entry_price = _as_float(getter("entry_price", 0.0))
    entry_fee = max(_as_float(getter("entry_fee", 0.0)), 0.0)
    estimated_cost = (quantity * entry_price) + entry_fee
    return estimated_cost if estimated_cost > 0 else 0.0


async def _get_local_open_position_snapshot(db_manager) -> tuple[int | None, float | None]:
    """Return local open-position count plus a conservative portfolio estimate."""
    getter = getattr(db_manager, "get_open_positions", None)
    if not callable(getter):
        return None, None
    try:
        result = getter()
        if inspect.isawaitable(result):
            result = await result
        positions = list(result or [])
        estimated_portfolio_value = sum(
            _estimate_local_position_cost_basis(position)
            for position in positions
        )
        if not positions and estimated_portfolio_value <= 0:
            estimated_portfolio_value = _configured_local_portfolio_floor()
        return (
            len(positions),
            estimated_portfolio_value if estimated_portfolio_value > 0 else None,
        )
    except Exception:
        return None, None


async def _build_paper_live_summary_line(db_manager) -> str:
    """Render a paper-vs-live divergence line using helper hooks or a proxy."""
    helper_summary, _helper_name = await _maybe_call_db_summary_helper(
        db_manager,
        (
            "get_paper_live_divergence_summary",
            "get_paper_vs_live_divergence_summary",
            "get_status_paper_live_divergence_summary",
        ),
    )
    if helper_summary:
        return helper_summary

    get_fee_divergence_entries = getattr(db_manager, "get_fee_divergence_entries", None)
    if not callable(get_fee_divergence_entries):
        return "pending DB helper"

    try:
        entries = get_fee_divergence_entries(limit=25)
        if inspect.isawaitable(entries):
            entries = await entries
    except Exception:
        return "pending DB helper"

    divergences = []
    for entry in entries or []:
        try:
            divergences.append(abs(float((entry or {}).get("divergence", 0.0) or 0.0)))
        except (AttributeError, TypeError, ValueError):
            continue

    if not divergences:
        return "pending DB helper"

    avg_abs_divergence = sum(divergences) / len(divergences)
    worst_divergence = max(divergences)
    return (
        "helper pending | "
        f"fee-drift proxy avg abs ${avg_abs_divergence:.4f} "
        f"across {len(divergences)} recent fills "
        f"(worst ${worst_divergence:.4f})"
    )


async def _build_ai_spend_summary_line(db_manager) -> str:
    """Render an AI spend/provider line using helper hooks or local DB stats."""
    helper_summary, _helper_name = await _maybe_call_db_summary_helper(
        db_manager,
        (
            "get_ai_spend_provider_breakdown",
            "get_ai_spend_provider_summary",
            "get_status_ai_spend_provider_summary",
        ),
    )
    if helper_summary:
        return helper_summary

    today_cost = None
    get_daily_ai_cost = getattr(db_manager, "get_daily_ai_cost", None)
    if callable(get_daily_ai_cost):
        try:
            today_cost = get_daily_ai_cost()
            if inspect.isawaitable(today_cost):
                today_cost = await today_cost
            today_cost = float(today_cost or 0.0)
        except Exception:
            today_cost = None

    llm_stats = {}
    get_llm_stats_by_strategy = getattr(db_manager, "get_llm_stats_by_strategy", None)
    if callable(get_llm_stats_by_strategy):
        try:
            llm_stats = get_llm_stats_by_strategy()
            if inspect.isawaitable(llm_stats):
                llm_stats = await llm_stats
            llm_stats = llm_stats or {}
        except Exception:
            llm_stats = {}

    total_queries = 0
    total_cost = 0.0
    for stats in llm_stats.values():
        if not isinstance(stats, dict):
            continue
        try:
            total_queries += int(stats.get("query_count") or 0)
        except (TypeError, ValueError):
            pass
        try:
            total_cost += float(stats.get("total_cost") or 0.0)
        except (TypeError, ValueError):
            pass

    if llm_stats:
        today_display = (
            f"today ${today_cost:.2f}"
            if today_cost is not None
            else "today unavailable"
        )
        return (
            f"{today_display} | "
            f"7d logged ${total_cost:.2f} across {total_queries} queries / {len(llm_stats)} strategies | "
            "provider breakdown pending DB helper"
        )

    if today_cost is not None:
        return f"today ${today_cost:.2f} | provider breakdown pending DB helper"

    return "pending DB helper"


async def _print_status_local_analytics(db_manager) -> None:
    """Print helper-backed or placeholder local analytics for `status`."""
    print()
    print("  Local Analytics:")
    if db_manager is None:
        print("  Paper vs Live:    local DB unavailable")
        print("  AI Spend:         local DB unavailable")
        return

    paper_live_summary = await _build_paper_live_summary_line(db_manager)
    ai_spend_summary = await _build_ai_spend_summary_line(db_manager)
    print(f"  Paper vs Live:    {paper_live_summary}")
    print(f"  AI Spend:         {ai_spend_summary}")


def cmd_run(args: argparse.Namespace) -> None:
    """Start the trading bot (disciplined mode by default)."""
    from src.utils.logging_setup import setup_logging

    log_level = getattr(args, "log_level", "INFO")
    setup_logging(log_level=log_level)

    run_mode = _resolve_run_mode(args)
    live = run_mode == "live"
    shadow = run_mode == "shadow"
    beast = getattr(args, "beast", False)
    disciplined = getattr(args, "disciplined", False)
    safe_compounder = getattr(args, "safe_compounder", False)
    live_trade_only = getattr(args, "live_trade", False)
    once = getattr(args, "once", False)
    smoke = getattr(args, "smoke", False)
    max_runtime_seconds = getattr(args, "max_runtime_seconds", None)

    if smoke and live:
        print("Error: --smoke cannot be combined with --live.")
        sys.exit(1)

    if max_runtime_seconds is not None and max_runtime_seconds <= 0:
        print("Error: --max-runtime-seconds must be greater than 0.")
        sys.exit(1)

    live_mode = live
    if smoke:
        once = True

    if live_trade_only:
        if smoke:
            print("Error: --live-trade does not support --smoke.")
            sys.exit(1)
        if beast or safe_compounder:
            print("Error: --live-trade cannot be combined with other runtime modes.")
            sys.exit(1)
        print("LIVE-TRADE LOOP MODE")
        if live_mode:
            print("   Live execution enabled for generic intents.")
            print("   Quick-flip live intents still require ENABLE_LIVE_QUICK_FLIP.")
        elif shadow:
            print("   Shadow mode enabled: paper execution plus shadow-side telemetry.")
        else:
            print("   Paper execution with SQLite decision logging.")
        _run_live_trade_loop_command(
            once=once,
            max_runtime_seconds=max_runtime_seconds,
            live_mode=live_mode,
            shadow_mode=shadow,
        )
        return

    if live_mode:
        print("WARNING: LIVE TRADING MODE ENABLED")
        print("   This will use real money and place actual trades.")
    elif shadow:
        print("SHADOW MODE ENABLED")
        print("   No real orders will be placed.")

    # --safe-compounder mode: edge-based NO-side only
    if safe_compounder:
        _run_safe_compounder(
            live_mode=live_mode,
            shadow_mode=shadow,
            max_runtime_seconds=max_runtime_seconds,
        )
        return

    # --beast mode: original aggressive settings (NOT default)
    if beast:
        print("BEAST MODE: Aggressive settings enabled.")
        print("   WARNING: Aggressive settings with no guardrails. Use at your own risk.")
        if smoke:
            print("   Smoke safety mode enabled: startup + ingestion only, no OpenRouter calls.")
        elif once:
            print("   Single-pass safety mode enabled: one cycle, then exit.")
        from src.runtime.unified_bot import UnifiedTradingBot
        if shadow and not _class_accepts_kwarg(UnifiedTradingBot, "shadow_mode"):
            print("   Shadow runtime hook not available yet; falling back to paper execution semantics.")
        bot = _construct_runtime(
            UnifiedTradingBot,
            live_mode=live_mode,
            shadow_mode=shadow,
        )
        try:
            asyncio.run(
                _run_bot_entrypoint(
                    bot,
                    once=once,
                    smoke=smoke,
                    max_runtime_seconds=max_runtime_seconds,
                )
            )
        except KeyboardInterrupt:
            print("\nTrading bot stopped by user.")
        return

    from src.config import settings as cfg

    # DEFAULT: AI Ensemble mode (disciplined settings active)
    api_settings = getattr(cfg.settings, "api", None)
    resolve_provider = getattr(api_settings, "resolve_llm_provider", None)
    provider = resolve_provider() if callable(resolve_provider) else "openrouter"
    print("AI ENSEMBLE MODE (default)")
    if provider == "codex":
        print("   Codex CLI routing: GPT-5.4 + GPT-5.4 Mini via ChatGPT plan quota.")
    elif provider == "openai":
        print("   Direct OpenAI routing: GPT-5.4 / o3 / GPT-4.1 via OPENAI_API_KEY.")
    else:
        print("   5-model OpenRouter ensemble: Claude Sonnet 4.5, Gemini 3.1 Pro Preview, GPT-5.4, DeepSeek V3.2, Grok 4.1 Fast.")
    print("   Category scoring + portfolio guardrails active.")
    print("   Use --safe-compounder for conservative math-only mode.")
    print("   Use --beast to run without guardrails (not recommended).")
    if smoke:
        print("   Smoke safety mode enabled: startup + ingestion only, no OpenRouter calls.")
    elif once:
        print("   Single-pass safety mode enabled: one cycle, then exit.")

    from src.runtime.unified_bot import UnifiedTradingBot
    from src.strategies.category_scorer import CategoryScorer
    from src.strategies.portfolio_enforcer import PortfolioEnforcer

    # Apply disciplined settings overrides
    cfg.settings.trading.min_confidence_to_trade = 0.45  # LOOSENED from 0.65 (approved 2026-03-29)
    cfg.settings.trading.max_position_size_pct = 3.0
    cfg.settings.trading.kelly_fraction = 0.25
    cfg.settings.trading.max_drawdown = 0.15
    cfg.settings.trading.max_sector_exposure = 0.30

    if shadow and not _class_accepts_kwarg(UnifiedTradingBot, "shadow_mode"):
        print("   Shadow runtime hook not available yet; falling back to paper execution semantics.")
    bot = _construct_runtime(
        UnifiedTradingBot,
        live_mode=live_mode,
        shadow_mode=shadow,
    )
    try:
        asyncio.run(
            _run_bot_entrypoint(
                bot,
                once=once,
                smoke=smoke,
                max_runtime_seconds=max_runtime_seconds,
            )
        )
    except KeyboardInterrupt:
        print("\nTrading bot stopped by user.")


def _run_safe_compounder(
    live_mode: bool = False,
    shadow_mode: bool = False,
    max_runtime_seconds: int | None = None,
) -> None:
    """Run the Safe Compounder strategy."""
    from src.clients.kalshi_client import KalshiClient
    from src.strategies.safe_compounder import SafeCompounder

    print("SAFE COMPOUNDER MODE")
    print("   NO-side only | Edge-based | Near-certain outcomes")
    if shadow_mode:
        print("   SHADOW RUN - dry-run execution until the strategy adds explicit shadow hooks")
    elif not live_mode:
        print("   DRY RUN - no real orders will be placed")

    async def _run():
        client = KalshiClient()
        try:
            # SafeCompounder has no explicit shadow execution hooks yet, so shadow
            # mode must remain paper-style dry-run semantics.
            compounder = SafeCompounder(client=client, dry_run=not live_mode)
            stats = await compounder.run()
            return stats
        finally:
            await client.close()

    async def _guarded_run():
        return await _run_with_runtime_guard(
            _run,
            label="safe compounder",
            max_runtime_seconds=max_runtime_seconds,
        )

    try:
        asyncio.run(_guarded_run())
    except KeyboardInterrupt:
        print("\nSafe Compounder stopped by user.")


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch the Node dashboard stack (web + API + Python bridge)."""
    import subprocess
    import shutil

    repo_root = Path(__file__).parent
    node_executable = shutil.which("node") or shutil.which("node.exe")
    npm_executable = shutil.which("npm") or shutil.which("npm.cmd")
    root_package = repo_root / "package.json"
    node_dashboard_runner = repo_root / "scripts" / "run-dashboard-stack.mjs"

    if root_package.exists() and (node_executable or npm_executable):
        print("Launching Node dashboard stack...", flush=True)
        print(
            "This starts the Python analysis bridge, Fastify API, and Next.js web app.",
            flush=True,
        )
        try:
            if node_executable and node_dashboard_runner.exists():
                subprocess.run(
                    [node_executable, str(node_dashboard_runner), "dev"],
                    check=False,
                    cwd=repo_root,
                )
            else:
                subprocess.run(
                    [npm_executable, "run", "dashboard"],
                    check=False,
                    cwd=repo_root,
                )
        except KeyboardInterrupt:
            print("\nDashboard stopped by user.")
        return

    # Legacy fallback only when the Node dashboard workspace is unavailable.
    dashboard_script = repo_root / "scripts" / "launch_dashboard.py"
    beast_dashboard = repo_root / "beast_mode_dashboard.py"

    if dashboard_script.exists():
        print("Node dashboard workspace not available; falling back to legacy Streamlit launcher.")
        subprocess.run([sys.executable, str(dashboard_script)], check=False)
    elif beast_dashboard.exists():
        # Fall back to the legacy dashboard runtime when the canonical repo-root
        # dashboard entrypoint is still present.
        from src.utils.logging_setup import setup_logging
        from src.runtime.unified_bot import UnifiedTradingBot

        setup_logging(log_level="INFO")
        bot = UnifiedTradingBot(live_mode=False, dashboard_mode=True)
        try:
            asyncio.run(bot.run())
        except KeyboardInterrupt:
            print("\nDashboard stopped by user.")
    else:
        print("Error: No dashboard script found.")
        sys.exit(1)


async def _print_strategy_budget_status(portfolio_value: float) -> None:
    """W7: Print per-strategy daily-loss budget remaining line.

    Silently no-ops if the enforcer cannot initialize (e.g. fresh DB on a
    machine that never ran the bot). We don't want status to crash on
    missing state.
    """
    try:
        from src.strategies.portfolio_enforcer import (
            PortfolioEnforcer,
            STRATEGY_QUICK_FLIP,
            STRATEGY_LIVE_TRADE,
        )
    except ImportError:
        return

    db_path = str(_resolve_db_path())
    if not Path(db_path).exists():
        return

    try:
        enforcer = PortfolioEnforcer(
            db_path=db_path,
            portfolio_value=portfolio_value,
        )
        await enforcer.initialize()
        print()
        print("  Strategy Risk Budgets (daily):")
        print(f"  {'Strategy':<14} {'Status':>10} {'Loss':>10} {'Budget':>10} {'Remaining':>12} {'Trades/hr':>10}")
        print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")
        for strategy in (STRATEGY_QUICK_FLIP, STRATEGY_LIVE_TRADE):
            status = await enforcer.get_strategy_status(strategy)
            state = "HALTED" if status["halted"] else "active"
            loss = f"${status['daily_loss_dollars']:.2f}"
            budget = (
                f"${status['daily_loss_budget_dollars']:.2f}"
                if status["daily_loss_budget_dollars"] is not None
                else "n/a"
            )
            remaining = (
                f"${status['daily_loss_budget_remaining_dollars']:.2f}"
                if status["daily_loss_budget_remaining_dollars"] is not None
                else "n/a"
            )
            rate = (
                f"{status['trades_last_hour']}/{status['max_trades_per_hour']}"
                if status["max_trades_per_hour"] is not None
                else f"{status['trades_last_hour']}"
            )
            print(
                f"  {status['strategy']:<14} {state:>10} "
                f"{loss:>10} {budget:>10} {remaining:>12} {rate:>10}"
            )
            if status.get("drift_halt"):
                avg_delta = status.get("drift_halt_avg_abs_entry_delta")
                cost_drift = status.get("drift_halt_total_entry_cost_delta")
                avg_delta_str = f"${avg_delta:.4f}" if avg_delta is not None else "n/a"
                cost_drift_str = f"${cost_drift:.2f}" if cost_drift is not None else "n/a"
                print(
                    f"  drift halt: {status.get('drift_halt_reason')} "
                    f"(avg delta {avg_delta_str}, cost drift {cost_drift_str})"
                )
    except Exception as exc:
        print(f"  (strategy budgets unavailable: {exc})")


def cmd_status(args: argparse.Namespace) -> None:
    """Show current portfolio status: balance, positions, and P&L."""

    async def _status() -> None:
        from src.clients.kalshi_client import KalshiClient
        from src.utils.kalshi_normalization import (
            get_balance_dollars,
            get_portfolio_value_dollars,
            get_position_exposure_dollars,
        )

        db_path = _resolve_db_path()
        db_manager = await _load_status_db_manager(db_path)
        local_open_positions, local_portfolio_estimate = (
            await _get_local_open_position_snapshot(db_manager)
        )

        client = None
        api_error = None
        balance_usd = None
        portfolio_value_usd = None
        active_positions = []
        total_exposure = 0.0
        total_realized_pnl = 0.0
        total_fees = 0.0
        try:
            client = KalshiClient()
            balance_resp = await client.get_balance()
            balance_usd = get_balance_dollars(balance_resp)

            portfolio_value_usd = get_portfolio_value_dollars(balance_resp)

            positions_resp = await client.get_positions()
            event_positions = positions_resp.get("event_positions", [])
            active_positions = [
                p for p in event_positions
                if get_position_exposure_dollars(p) > 0
            ]
        except Exception as exc:
            api_error = exc
        finally:
            await _close_async_resource(client)

        print("=" * 56)
        print("  PORTFOLIO STATUS")
        print("=" * 56)
        print(
            "  Kalshi API:        "
            + (
                "connected"
                if api_error is None
                else f"unavailable ({api_error})"
            )
        )
        total_portfolio = (
            balance_usd + portfolio_value_usd
            if balance_usd is not None and portfolio_value_usd is not None
            else None
        )
        print(f"  Available Cash:     {_format_currency_or_placeholder(balance_usd)}")
        print(f"  Position Value:     {_format_currency_or_placeholder(portfolio_value_usd)}")
        print(f"  Total Portfolio:    {_format_currency_or_placeholder(total_portfolio)}")
        if api_error is None:
            print(f"  Active Positions:   {len(active_positions):>10}")
        else:
            local_positions_display = (
                f"{local_open_positions:>10}"
                if local_open_positions is not None
                else f"{'n/a':>10}"
            )
            print(f"  Active Positions:   {'API unavailable':>10}")
            print(f"  Local Open Positions:{local_positions_display}")
            if local_portfolio_estimate is not None:
                print(
                    "  Local Portfolio Est: "
                    f"{_format_currency_or_placeholder(local_portfolio_estimate)}"
                )

        if active_positions:
            print()
            print(f"  {'Event':<30} {'Exposure':>10} {'Cost':>10} {'P&L':>10} {'Fees':>8}")
            print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

            for pos in active_positions:
                ticker = pos.get("event_ticker", "???")
                exposure = get_position_exposure_dollars(pos)
                cost = float(pos.get("total_cost_dollars", "0"))
                pnl = float(pos.get("realized_pnl_dollars", "0"))
                fees = float(pos.get("fees_paid_dollars", "0"))
                total_exposure += exposure
                total_realized_pnl += pnl
                total_fees += fees
                print(
                    f"  {ticker:<30} ${exposure:>8.2f} ${cost:>8.2f} "
                    f"${pnl:>8.2f} ${fees:>6.2f}"
                )

            print()
            print(f"  Total Exposure:     ${total_exposure:>10,.2f}")
            print(f"  Total Realized P&L: ${total_realized_pnl:>10,.2f}")
            print(f"  Total Fees Paid:    ${total_fees:>10,.2f}")
        elif api_error is not None:
            print()
            print("  Open-position detail unavailable from Kalshi; showing local analytics below.")

        strategy_budget_portfolio_value = (
            total_portfolio
            if total_portfolio is not None
            else (local_portfolio_estimate or 0.0)
        )
        await _print_strategy_budget_status(strategy_budget_portfolio_value)
        await _print_status_local_analytics(db_manager)
        print("=" * 56)
        await _close_async_resource(db_manager)

    try:
        asyncio.run(_status())
    except Exception as exc:
        print(f"Error fetching status: {exc}")
        sys.exit(1)


def cmd_scores(args: argparse.Namespace) -> None:
    """Show current category scores from the scoring system."""

    async def _scores():
        from src.strategies.category_scorer import CategoryScorer
        scorer = CategoryScorer()
        await scorer.initialize()
        scores = await scorer.get_all_scores()
        print(scorer.format_scores_table(scores))
        print()
        print("  Key: Score < 30 = BLOCKED | Alloc = max portfolio % allowed")
        print()

    try:
        asyncio.run(_scores())
    except Exception as exc:
        print(f"Error fetching scores: {exc}")
        sys.exit(1)


def cmd_history(args: argparse.Namespace) -> None:
    """Show trade history with category breakdown."""
    limit = getattr(args, "limit", 50)

    async def _history():
        import aiosqlite

        db_path = Path(__file__).parent / "trading_system.db"
        if not db_path.exists():
            print("No trading database found.")
            return

        async with aiosqlite.connect(str(db_path)) as db:
            db.row_factory = aiosqlite.Row

            # Overall stats
            cursor = await db.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl
                FROM trade_logs
            """)
            overview = await cursor.fetchone()

            print("=" * 70)
            print("  TRADE HISTORY")
            print("=" * 70)
            if overview and overview["total"]:
                total = overview["total"]
                wins = overview["wins"] or 0
                pnl = overview["total_pnl"] or 0.0
                print(f"  Total Trades:  {total}")
                print(f"  Win Rate:      {wins/total*100:.1f}%")
                print(f"  Total P&L:     ${pnl:.2f}")
                print(f"  Avg per trade: ${(pnl/total):.2f}")
            print()

            # Category breakdown
            cursor = await db.execute("""
                SELECT
                    strategy as category,
                    COUNT(*) as trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(pnl) as total_pnl
                FROM trade_logs
                GROUP BY strategy
                ORDER BY total_pnl DESC
            """)
            cats = await cursor.fetchall()

            if cats:
                print(f"  {'Category':<22} {'Trades':>7} {'WR':>6} {'P&L':>10}")
                print(f"  {'-'*22} {'-'*7} {'-'*6} {'-'*10}")
                for row in cats:
                    cat = row["category"] or "unknown"
                    t = row["trades"]
                    w = row["wins"] or 0
                    p = row["total_pnl"] or 0.0
                    wr = f"{w/t*100:.0f}%" if t > 0 else "n/a"
                    print(f"  {cat:<22} {t:>7} {wr:>6} ${p:>9.2f}")
                print()

            # Recent trades
            cursor = await db.execute(f"""
                SELECT market_id, side, entry_price, exit_price, quantity, pnl,
                       entry_timestamp, strategy
                FROM trade_logs
                ORDER BY entry_timestamp DESC
                LIMIT {limit}
            """)
            trades = await cursor.fetchall()

            if trades:
                print(f"  Recent {limit} trades:")
                print(f"  {'Market':<28} {'Side':>4} {'Entry':>6} {'Exit':>6} {'Qty':>4} {'P&L':>8} {'Category'}")
                print(f"  {'-'*28} {'-'*4} {'-'*6} {'-'*6} {'-'*4} {'-'*8} {'-'*12}")
                for t in trades:
                    ts = (t["entry_timestamp"] or "")[:10]
                    cat = t["strategy"] or ""
                    print(
                        f"  {t['market_id'][:28]:<28} {t['side']:>4} "
                        f"{t['entry_price']:>6.2f} {t['exit_price']:>6.2f} "
                        f"{t['quantity']:>4} ${t['pnl']:>7.2f}  {cat}"
                    )

            cursor2 = await db.execute("""
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table' AND name = 'blocked_trades'
                LIMIT 1
            """)
            if await cursor2.fetchone():
                cursor3 = await db.execute("""
                    SELECT COUNT(*) FROM blocked_trades
                """)
                blocked_summary = await cursor3.fetchone()
                blocked_count = blocked_summary[0] if blocked_summary else 0
                if blocked_count:
                    print(
                        f"\n  [BLOCKED] {blocked_count} trades blocked by portfolio enforcer "
                        "(use 'python cli.py health' for details)"
                    )
                if False:
                    print(f"\n  [BLOCKED] {r2[0]} trades blocked by portfolio enforcer (use 'python cli.py health' for details)")

            print("=" * 70)
            return

            # Blocked trades summary
            cursor2 = await db.execute("""
                SELECT COUNT(*) FROM blocked_trades
            """)
            r2 = await cursor2.fetchone()
            if r2 and r2[0]:
                print(f"\n  [BLOCKED] {r2[0]} trades blocked by portfolio enforcer (use 'python cli.py health' for details)")

            print("=" * 70)

    try:
        asyncio.run(_history())
    except Exception as exc:
        print(f"Error fetching history: {exc}")
        sys.exit(1)


def cmd_backtest(args: argparse.Namespace) -> None:
    """Run backtests (placeholder)."""
    print("=" * 56)
    print("  BACKTESTING")
    print("=" * 56)
    print()
    print("  Backtesting engine coming soon.")
    print()
    print("  Planned features:")
    print("    - Historical market replay")
    print("    - Strategy parameter optimization")
    print("    - Walk-forward analysis")
    print("    - Monte Carlo simulation")
    print()
    print("=" * 56)


def cmd_health(args: argparse.Namespace) -> None:
    """Run health checks on configuration, API, and database."""

    checks_passed = 0
    checks_failed = 0

    def ok(label: str, detail: str = "") -> None:
        nonlocal checks_passed
        checks_passed += 1
        suffix = f" -- {detail}" if detail else ""
        print(f"  [PASS] {label}{suffix}")

    def fail(label: str, detail: str = "") -> None:
        nonlocal checks_failed
        checks_failed += 1
        suffix = f" -- {detail}" if detail else ""
        print(f"  [FAIL] {label}{suffix}")

    print("=" * 56)
    print("  HEALTH CHECK")
    print("=" * 56)
    print()

    # 1. .env file
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        ok(".env file exists")
    else:
        fail(".env file missing", "copy env.template to .env and fill in keys")

    # 2. Required environment variables
    from dotenv import load_dotenv
    load_dotenv()

    for var, placeholder in (
        ("KALSHI_API_KEY", "your_kalshi_api_key_here"),
        ("OPENROUTER_API_KEY", "your_openrouter_api_key_here"),
    ):
        val = os.getenv(var, "")
        if val and val not in ("", placeholder):
            ok(f"{var} is set")
        else:
            fail(f"{var} is missing or placeholder")

    from src.utils.kalshi_auth import resolve_private_key_path

    configured_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip() or "kalshi_private_key"
    private_key_path = resolve_private_key_path(configured_key_path)
    if Path(private_key_path).exists():
        ok("KALSHI private key file exists", private_key_path)
    else:
        fail("KALSHI private key file missing", private_key_path)

    # 3. Kalshi API connection
    async def _check_api() -> None:
        from src.clients.kalshi_client import KalshiClient
        from src.utils.kalshi_normalization import get_balance_dollars
        client = KalshiClient()
        try:
            balance_resp = await client.get_balance()
            balance_usd = get_balance_dollars(balance_resp)
            ok("Kalshi API connection", f"balance=${balance_usd:,.2f}")
        except Exception as exc:
            fail("Kalshi API connection", str(exc))
        finally:
            await client.close()

    try:
        asyncio.run(_check_api())
    except Exception as exc:
        fail("Kalshi API connection", str(exc))

    # 4. Database
    db_path = Path(__file__).parent / "trading_system.db"
    try:
        import aiosqlite

        async def _check_db() -> None:
            from src.utils.database import DatabaseManager
            db_manager = DatabaseManager()
            await db_manager.initialize()
            ok("Database initialization", str(db_path))

        asyncio.run(_check_db())
    except Exception as exc:
        fail("Database initialization", str(exc))

    # 5. Python version
    if sys.version_info >= (3, 12):
        ok("Python version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    else:
        fail("Python version", f"requires >=3.12, found {sys.version}")

    # Summary
    print()
    total = checks_passed + checks_failed
    print(f"  {checks_passed}/{total} checks passed")
    if checks_failed:
        print(f"  {checks_failed} issue(s) need attention")
    else:
        print("  All systems operational.")
    print("=" * 56)

    if checks_failed:
        sys.exit(1)


def cmd_mcp(args: argparse.Namespace) -> None:
    """Run the localhost operator API with MCP-style tool endpoints."""
    import uvicorn

    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8765)
    print("Starting localhost operator API")
    print(f"  URL: http://{host}:{port}")
    print("  Transport: HTTP JSON-RPC at POST /mcp/jsonrpc")
    print("  Tools: GET /mcp/tools and POST /mcp/call/{tool_name}")
    print("  Live order calls require OPERATOR_API_ALLOW_LIVE_ORDERS=true")
    uvicorn.run("src.operator_api:app", host=host, port=port, log_level="info")


def cmd_explain_market(args: argparse.Namespace) -> None:
    """Explain contract mechanics for a market ticker."""

    async def _explain() -> None:
        from src.clients.kalshi_client import KalshiClient
        from src.data.weather_adapter import interpret_temperature_market

        client = KalshiClient()
        try:
            response = await client.get_market(args.ticker)
            market = response.get("market", response) if isinstance(response, dict) else {}
            interpretation = interpret_temperature_market(market)
            print("=" * 70)
            print(f"  MARKET EXPLANATION: {args.ticker}")
            print("=" * 70)
            print(f"  Title: {market.get('title', 'n/a')}")
            print(f"  Status: {market.get('status', 'n/a')}")
            print()
            if interpretation.detected:
                print("  Weather contract interpreter")
                print(f"  Bucket:            {interpretation.bucket_label or 'unknown'}")
                print(f"  Threshold:         {interpretation.threshold if interpretation.threshold is not None else 'unknown'}")
                print(f"  Settlement source: {interpretation.settlement_source or 'unknown'}")
                print(f"  Confidence:        {interpretation.confidence:.0%}")
                print(f"  Tradeable:         {'yes' if interpretation.can_trade else 'blocked'}")
                if interpretation.block_reason:
                    print(f"  Block reason:      {interpretation.block_reason}")
                print(f"  Notes:             {interpretation.notes}")
            else:
                print("  No specialized contract interpreter matched this market yet.")
            print("=" * 70)
        finally:
            await client.close()

    asyncio.run(_explain())


def cmd_weather(args: argparse.Namespace) -> None:
    """Deterministic forecast probabilities + edge for a Kalshi weather event."""

    async def _run() -> None:
        from src.clients.kalshi_client import KalshiClient
        from src.config.settings import settings
        from src.data.weather_adapter import WeatherAdapter
        from src.utils.kalshi_normalization import get_market_prices
        from src.utils.probability_engine import fee_aware_ev, kelly_fraction

        client = KalshiClient()
        adapter = WeatherAdapter()
        try:
            if args.ticker:
                response = await client.get_market(args.ticker)
                market = response.get("market", response) if isinstance(response, dict) else {}
                markets = [market] if market else []
                event_ticker = str(market.get("event_ticker") or args.ticker)
            else:
                response = await client.get_markets(event_ticker=args.event, limit=100)
                markets = list(response.get("markets") or [])
                event_ticker = args.event

            if not markets:
                print(f"No markets found for {args.ticker or args.event}.")
                return

            context = await adapter.fetch_context(
                {"event_ticker": event_ticker, "markets": markets}
            )
            signals = context.get("signals") or {}
            station = signals.get("station") or {}
            period = signals.get("target_period") or {}
            forecast = signals.get("forecast") or {}
            probabilities = signals.get("market_probabilities") or {}

            print("=" * 88)
            print(f"  WEATHER MODEL: {event_ticker}")
            print("=" * 88)
            print(f"  Station:      {station.get('name') or 'UNRESOLVED'}"
                  + ("" if station.get("verified", True) else "  (geocoded — not the settlement station!)"))
            print(f"  Period:       {period.get('kind') or '?'} {period.get('start') or '?'}"
                  f"  (lead {signals.get('lead_days', '?')} days)")
            nws_daily = forecast.get("nws_daily") or {}
            om_daily = forecast.get("open_meteo_daily") or {}
            print(f"  NWS forecast: high {nws_daily.get('high_f', '—')}F / low {nws_daily.get('low_f', '—')}F"
                  f"   Open-Meteo: high {om_daily.get('high_f', '—')}F / low {om_daily.get('low_f', '—')}F")
            if forecast.get("running_max_f") is not None:
                print(f"  Running obs:  max {forecast.get('running_max_f'):.1f}F"
                      f" / min {forecast.get('running_min_f'):.1f}F"
                      f" (through {forecast.get('running_through_local')})")
            print(f"  Members:      {forecast.get('ensemble_member_count', 0)}"
                  f"  method={forecast.get('forecast_method') or signals.get('model_status')}")
            if context.get("error"):
                print(f"  Adapter error: {context['error']}")
            print("-" * 88)

            if not probabilities:
                print(f"  No bucket probabilities ({signals.get('model_status')}).")
                print("=" * 88)
                return

            min_edge = float(getattr(settings.trading, "live_trade_min_net_edge", 0.02) or 0.02)
            kelly_multiplier = float(getattr(settings.trading, "kelly_fraction", 0.25) or 0.25)
            kelly_cap = float(getattr(settings.trading, "max_single_position", 0.03) or 0.03)

            header = (
                f"  {'TICKER':<32}{'BUCKET':<26}{'ASK Y/N':<14}"
                f"{'P(YES)':<9}{'EDGE':<16}ACTION"
            )
            print(header)
            print("-" * 88)

            by_ticker = {str(m.get("ticker") or ""): m for m in markets}
            for ticker, entry in sorted(probabilities.items()):
                market = by_ticker.get(ticker, {})
                yes_bid, yes_ask, no_bid, no_ask = get_market_prices(market)
                p_yes = float(entry.get("model_yes_probability") or 0.5)

                yes_edge = no_edge = None
                if yes_ask and 0 < yes_ask < 1:
                    yes_edge = fee_aware_ev(
                        win_probability=p_yes, entry_price=yes_ask, side="YES"
                    ).net_edge
                if no_ask and 0 < no_ask < 1:
                    no_edge = fee_aware_ev(
                        win_probability=1.0 - p_yes, entry_price=no_ask, side="NO"
                    ).net_edge

                action = "PASS"
                size_note = ""
                if yes_edge is not None and yes_edge > min_edge and (no_edge or -1) <= yes_edge:
                    action = "BUY YES"
                    kelly = kelly_fraction(
                        win_probability=p_yes,
                        entry_price=yes_ask,
                        multiplier=kelly_multiplier,
                        cap=kelly_cap,
                    )
                    size_note = f" ({kelly * 100:.1f}% bankroll)"
                elif no_edge is not None and no_edge > min_edge:
                    action = "BUY NO"
                    kelly = kelly_fraction(
                        win_probability=1.0 - p_yes,
                        entry_price=no_ask,
                        multiplier=kelly_multiplier,
                        cap=kelly_cap,
                    )
                    size_note = f" ({kelly * 100:.1f}% bankroll)"

                edge_text = (
                    f"Y{(yes_edge if yes_edge is not None else 0) * 100:+.1f}c"
                    f"/N{(no_edge if no_edge is not None else 0) * 100:+.1f}c"
                )
                ask_text = (
                    f"{yes_ask:.2f}/{no_ask:.2f}"
                    if yes_ask and no_ask
                    else "—"
                )
                quality = float(entry.get("quality") or 0.0)
                print(
                    f"  {ticker:<32}{str(entry.get('bucket_label') or '')[:24]:<26}"
                    f"{ask_text:<14}{p_yes:<9.3f}{edge_text:<16}{action}{size_note}"
                )
                if not entry.get("can_trade", True):
                    print(f"  {'':<32}^ interpreter blocked: confidence {entry.get('interpretation_confidence')}")
                if quality < 0.5:
                    print(f"  {'':<32}^ low model quality ({quality:.2f}, {entry.get('method')})")
            print("-" * 88)
            print(
                f"  Edges are net of Kalshi taker fees; minimum tradable edge is "
                f"{min_edge * 100:.1f}c/contract. Kelly sizes use {kelly_multiplier:.2f}x "
                f"fractional Kelly capped at {kelly_cap * 100:.0f}%."
            )
            print("=" * 88)
        finally:
            await adapter.aclose()
            await client.close()

    asyncio.run(_run())


def cmd_weather_scan(args: argparse.Namespace) -> None:
    """Sweep all open weather events with the deterministic model."""

    async def _run() -> None:
        import json

        from src.config.settings import settings
        from src.jobs.weather_scan import run_weather_scan

        summary = await run_weather_scan(execute=bool(getattr(args, "trade", False)))

        if getattr(args, "json", False):
            payload = {
                "started_at": summary.started_at,
                "series_scanned": summary.series_scanned,
                "events_scanned": summary.events_scanned,
                "markets_evaluated": summary.markets_evaluated,
                "positions_opened": summary.positions_opened,
                "errors": summary.errors,
                "candidates": [c.to_payload() for c in summary.candidates],
            }
            print(json.dumps(payload, indent=2, default=str))
            return

        print("=" * 96)
        print(
            f"  WEATHER SCAN  ({summary.started_at})  series={len(summary.series_scanned)}"
            f"  events={summary.events_scanned}  markets={summary.markets_evaluated}"
        )
        print("=" * 96)
        if summary.errors:
            for error in summary.errors[:8]:
                print(f"  ! {error}")
            print("-" * 96)
        if not summary.candidates:
            print("  No fee-positive candidates above the configured thresholds.")
            print("=" * 96)
            return
        header = (
            f"  {'TICKER':<30}{'SIDE':<6}{'ASK':<7}{'P(WIN)':<9}"
            f"{'NET EDGE':<10}{'QUAL':<7}{'LEAD':<6}KELLY"
        )
        print(header)
        print("-" * 96)
        for candidate in summary.candidates[:25]:
            print(
                f"  {candidate.market_ticker:<30}{candidate.side:<6}"
                f"{candidate.entry_price * 100:>4.0f}c  "
                f"{candidate.side_win_probability:<9.3f}"
                f"{candidate.net_edge * 100:>+7.1f}c  "
                f"{candidate.quality:<7.2f}{candidate.lead_days:<6.1f}"
                f"{candidate.kelly_fraction * 100:.1f}%"
            )
        print("-" * 96)
        trading_active = bool(getattr(args, "trade", False)) or bool(
            getattr(settings.weather, "scan_trade_enabled", False)
        )
        print(
            f"  Positions opened this run: {summary.positions_opened}"
            + (
                ""
                if trading_active
                else "  (scan-only; use --trade or WEATHER_SCAN_TRADE_ENABLED to execute)"
            )
        )
        print("=" * 96)

    asyncio.run(_run())


def cmd_backfill_results(args: argparse.Namespace) -> None:
    """Backfill settlement results for snapshotted markets."""

    async def _run() -> None:
        from src.jobs.settlement_backfill import run_settlement_backfill

        summary = await run_settlement_backfill(
            max_tickers=getattr(args, "limit", None),
            force_fit=bool(getattr(args, "fit", False)),
        )
        print("=" * 72)
        print(f"  SETTLEMENT BACKFILL  ({summary.started_at})")
        print("=" * 72)
        print(f"  Tickers checked : {summary.tickers_checked}")
        print(f"  Settled         : {summary.settled}")
        print(f"  Voided          : {summary.voided}")
        print(f"  Still pending   : {summary.pending}")
        print(f"  Missing at API  : {summary.missing}")
        print(f"  Model refit     : {'yes' if summary.model_refit else 'no'}")
        print(f"  Active segments : {summary.models_active}")
        for error in summary.errors[:8]:
            print(f"  ! {error}")
        print("=" * 72)

    asyncio.run(_run())


def cmd_fit_market_prior(args: argparse.Namespace) -> None:
    """Refit the market-prior calibration from labelled snapshots."""

    async def _run() -> None:
        from src.jobs.settlement_backfill import refresh_market_prior_models
        from src.utils.database import DatabaseManager
        from src.utils.market_prior import invalidate_market_prior_cache

        db = DatabaseManager()
        await db.initialize()
        active = await refresh_market_prior_models(db)
        invalidate_market_prior_cache()
        rows = await db.get_market_prior_models()
        print("=" * 88)
        print("  MARKET-PRIOR CALIBRATION FIT")
        print("=" * 88)
        if not rows:
            print("  No labelled snapshot samples yet — run `python cli.py backfill-results` first.")
            print("=" * 88)
            return
        header = (
            f"  {'SEGMENT':<10}{'A':>8}{'B':>8}{'N_TRAIN':>10}{'N_HOLD':>9}"
            f"{'BRIER(MODEL)':>14}{'BRIER(MID)':>12}{'ACTIVE':>8}"
        )
        print(header)
        print("-" * 88)
        for row in rows:
            hb_model = row.get("holdout_brier_model")
            hb_ident = row.get("holdout_brier_identity")
            print(
                f"  {str(row['segment']):<10}{float(row['intercept']):>8.3f}"
                f"{float(row['slope']):>8.3f}{int(row['n_train']):>10}"
                f"{int(row['n_holdout']):>9}"
                f"{(f'{hb_model:.5f}' if hb_model is not None else 'n/a'):>14}"
                f"{(f'{hb_ident:.5f}' if hb_ident is not None else 'n/a'):>12}"
                f"{('yes' if row.get('active') else 'no'):>8}"
            )
        print("-" * 88)
        print(
            f"  Active segments: {active}. The EV gates use active segments only; "
            "raw mid is used otherwise."
        )
        print("=" * 88)

    asyncio.run(_run())


def cmd_scan_arb(args: argparse.Namespace) -> None:
    """Run an alert-only Kalshi vs Polymarket scan."""

    async def _scan() -> None:
        from src.clients.kalshi_client import KalshiClient
        from src.data.polymarket_adapter import PolymarketAdapter
        from src.utils.database import DatabaseManager

        client = KalshiClient()
        adapter = PolymarketAdapter()
        db = DatabaseManager()
        try:
            await db.initialize()
            response = await client.get_markets(limit=args.kalshi_limit, status="open")
            markets = response.get("markets", []) if isinstance(response, dict) else []
            try:
                candidates = await adapter.scan_kalshi_markets(
                    markets,
                    limit=args.polymarket_limit,
                    min_mapping_confidence=args.min_confidence,
                    min_edge=args.min_edge,
                    strict=getattr(args, "strict", False),
                )
            except Exception as exc:
                await db.record_source_snapshot(
                    category="cross_market",
                    source="polymarket.gamma",
                    status="unavailable",
                    freshness_seconds=1,
                    payload={"phase": "cli_scan_arb", "error": str(exc)},
                )
                raise
            polymarket_payload = getattr(adapter, "last_fetch_payload", None)
            polymarket_markets = (
                polymarket_payload.get("signals", {}).get("markets", [])
                if isinstance(polymarket_payload, dict)
                else []
            )
            await db.record_source_snapshot(
                category="cross_market",
                source="polymarket.gamma",
                status="healthy",
                freshness_seconds=(
                    int(polymarket_payload.get("freshness_seconds") or 0)
                    if isinstance(polymarket_payload, dict)
                    else max((c.freshness_seconds for c in candidates), default=0)
                ),
                payload={
                    "phase": "cli_scan_arb",
                    "candidate_count": len(candidates),
                    "polymarket_market_count": (
                        len(polymarket_markets)
                        if isinstance(polymarket_markets, list)
                        else 0
                    ),
                },
            )
            for candidate in candidates[: args.limit]:
                await db.record_arbitrage_candidate(candidate.to_dict())
            print("=" * 96)
            print("  CROSS-MARKET ARBITRAGE WATCHLIST (alert-only)")
            print("=" * 96)
            if not candidates:
                print("  No candidates passed the configured edge and mapping thresholds.")
                return
            print(
                f"  {'Kalshi':<22} {'Side':>4} {'K':>6} {'Poly':>6}"
                f" {'Gross':>6} {'Net':>6} {'Map':>6}  Notes / Question"
            )
            print(
                f"  {'-'*22} {'-'*4} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}  {'-'*32}"
            )
            for candidate in candidates[: args.limit]:
                detail = (
                    candidate.notes
                    if candidate.notes and candidate.notes != "ok"
                    else candidate.polymarket_question[:48]
                )
                print(
                    f"  {candidate.kalshi_ticker[:22]:<22} {candidate.side:>4} "
                    f"{candidate.kalshi_price:>6.2f} {candidate.polymarket_price:>6.2f} "
                    f"{candidate.estimated_edge:>6.2f} {candidate.net_edge:>6.2f} "
                    f"{candidate.mapping_confidence:>6.0%}  "
                    f"{detail}"
                )
            print()
            print("  These are not executable orders. Confirm mapping, fees, and fillability manually.")
        finally:
            await adapter.aclose()
            await client.close()

    asyncio.run(_scan())


def cmd_refresh_calibration(args: argparse.Namespace) -> None:
    """Rebuild settlement calibration rows from trade logs."""

    async def _refresh() -> None:
        from src.utils.database import DatabaseManager

        db = DatabaseManager()
        await db.initialize()
        refreshed = await db.refresh_settlement_calibration()
        print(f"Refreshed {refreshed} calibration rows from trade_logs.")

    asyncio.run(_refresh())


def cmd_safety_status(args: argparse.Namespace) -> None:
    """Show recent execution safety blocks and source-health snapshots."""

    async def _status() -> None:
        import aiosqlite

        from src.utils.calibration_metrics import (
            expected_calibration_error,
            probability_buckets,
        )
        from src.utils.database import DatabaseManager

        db = DatabaseManager()
        await db.initialize()
        refreshed = await db.refresh_settlement_calibration()
        async with aiosqlite.connect(db.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            print("=" * 72)
            print("  EXECUTION SAFETY STATUS")
            print("=" * 72)
            print(f"  Calibration rows refreshed from trade logs: {refreshed}")

            cursor = await conn.execute(
                "SELECT predicted_probability, outcome FROM settlement_calibration"
            )
            samples = [
                (float(row["predicted_probability"] or 0.0), int(row["outcome"] or 0))
                for row in await cursor.fetchall()
            ]
            if samples:
                ece = expected_calibration_error(samples)
                buckets = probability_buckets(samples)
                non_empty = [bucket for bucket in buckets if bucket.count > 0]
                print(
                    f"  Calibration samples: {len(samples)} | ECE: {ece:.3f} "
                    f"| populated buckets: {len(non_empty)}/{len(buckets)}"
                )
            cursor = await conn.execute(
                """
                SELECT rejected_at, ticker, side, reason, score
                FROM anomaly_rejections
                ORDER BY rejected_at DESC, id DESC
                LIMIT ?
                """,
                (args.limit,),
            )
            rows = await cursor.fetchall()
            if rows:
                print(f"  {'Time':<19} {'Ticker':<24} {'Side':>4} {'Reason':<28} {'Score':>5}")
                print(f"  {'-'*19} {'-'*24} {'-'*4} {'-'*28} {'-'*5}")
                for row in rows:
                    print(
                        f"  {str(row['rejected_at'])[:19]:<19} {row['ticker'][:24]:<24} "
                        f"{row['side']:>4} {row['reason'][:28]:<28} {float(row['score'] or 0):>5.2f}"
                    )
            else:
                print("  No execution-safety rejections recorded.")

            cursor = await conn.execute(
                """
                SELECT category, source, status, freshness_seconds, captured_at
                FROM source_snapshots
                ORDER BY captured_at DESC, id DESC
                LIMIT 8
                """
            )
            sources = await cursor.fetchall()
            if sources:
                print()
                print("  Recent source snapshots")
                for row in sources:
                    print(
                        f"  {row['captured_at'][:19]} {row['category']}/{row['source']} "
                        f"{row['status']} ({row['freshness_seconds']}s)"
                    )
            print("=" * 72)

    asyncio.run(_status())


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def cmd_auto(args: argparse.Namespace) -> None:
    """
    Beast Auto V1 — unattended supervisor around the existing live-trade
    decision/risk/execution pipeline. Manual (`run --live-trade`,
    `run --live`, etc.) is completely separate and unaffected by this
    command. Quick Flip is never touched by Auto.
    """
    from src.auto.kill_switch import AutoKillSwitch
    from src.config.settings import settings

    if args.auto_action == "kill":
        switch = AutoKillSwitch(settings.auto.kill_switch_path)
        switch.activate(reason=args.reason or "manual operator kill", activated_by="cli")
        print(f"Auto kill switch ACTIVATED ({settings.auto.kill_switch_path}).")
        return

    if args.auto_action == "resume":
        switch = AutoKillSwitch(settings.auto.kill_switch_path)
        switch.deactivate(deactivated_by="cli")
        print(f"Auto kill switch DEACTIVATED ({settings.auto.kill_switch_path}).")
        return

    if args.auto_action == "status":
        switch = AutoKillSwitch(settings.auto.kill_switch_path)
        state = switch.read_state()
        print("=" * 60)
        print("  BEAST AUTO STATUS")
        print("=" * 60)
        print(f"  Kill switch active: {state.active}")
        if state.reason:
            print(f"  Reason: {state.reason}")
        if state.activated_by:
            print(f"  Set by: {state.activated_by}")
        print(f"  Cadence: {settings.auto.cadence_seconds}s")
        print(f"  Max consecutive failures before auto-halt: {settings.auto.max_consecutive_failures}")
        print("=" * 60)
        return

    # args.auto_action == "start"
    if not args.live:
        print(
            "Refusing to start: Beast Auto V1 requires --live to submit real "
            "orders. There is no paper/shadow mode for Auto in V1 — use "
            "`python cli.py run --live-trade` for paper/shadow supervision."
        )
        return

    async def _run() -> None:
        from src.auto.runner import AutoRunner
        from src.clients.kalshi_client import KalshiClient
        from src.utils.database import DatabaseManager

        db_manager = DatabaseManager()
        await db_manager.initialize()
        async with KalshiClient() as kalshi_client:
            runner = AutoRunner(db_manager=db_manager, kalshi_client=kalshi_client)
            print("Beast Auto V1 starting (LIVE). Ctrl-C for graceful shutdown.")
            await runner.run_forever()

    asyncio.run(_run())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kalshi-bot",
        description="Kalshi AI Trading Bot -- Multi-model AI trading for prediction markets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python cli.py run                      Start AI Ensemble mode (default, paper)\n"
            "  python cli.py run --live               AI Ensemble with real capital\n"
            "  python cli.py run --live-trade         Loop-only live-trade runtime (paper by default)\n"
            "  python cli.py run --live-trade --shadow  Loop-only live-trade runtime with shadow telemetry\n"
            "  python cli.py run --live-trade --live  Loop-only live-trade runtime with live execution\n"
            "  python cli.py run --shadow             Shadow mode (dry-run execution, live-like analytics)\n"
            "  python cli.py run --safe-compounder    Safe Compounder: conservative, math-only\n"
            "  python cli.py run --safe-compounder --live  Safe Compounder live\n"
            "  python cli.py run --once               Run one ingest/trade cycle and exit\n"
            "  python cli.py run --smoke              Run a no-LLM smoke test and exit\n"
            "  python cli.py run --once --max-runtime-seconds 120  Hard-stop a smoke run\n"
            "  python cli.py run --beast              Beast mode (aggressive, not recommended)\n"
            "  python cli.py scores                   Show category scores\n"
            "  python cli.py history                  Show trade history + category breakdown\n"
            "  python cli.py status                   Check portfolio balance and positions\n"
            "  python cli.py health                   Verify all connections and config\n"
            "  python cli.py mcp                      Run localhost operator API tools\n"
            "  python cli.py explain-market TICKER    Explain contract mechanics\n"
            "  python cli.py scan-arb                 Alert-only Kalshi/Polymarket scan\n"
            "  python cli.py safety-status            Show blocked execution attempts\n"
            "  python cli.py refresh-calibration      Rebuild settlement calibration rows\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = subparsers.add_parser(
        "run",
        help="Start the trading bot (disciplined mode by default)",
        description=(
            "Launch the trading bot. Default is AI Ensemble mode: five frontier LLMs "
            "(Claude Sonnet 4.5, Gemini 3.1 Pro Preview, GPT-5.4, DeepSeek V3.2, Grok 4.1 Fast) debate "
            "every trade with category scoring and portfolio guardrails. "
            "Use --safe-compounder for conservative math-only mode. "
            "Use --beast for aggressive mode without guardrails (not recommended)."
        ),
    )
    live_group = p_run.add_mutually_exclusive_group()
    live_group.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading with real capital (default: paper trading)",
    )
    live_group.add_argument(
        "--paper",
        action="store_true",
        help="Run in paper-trading mode (no real orders)",
    )
    live_group.add_argument(
        "--shadow",
        action="store_true",
        help="Run in shadow mode: dry-run execution with live-vs-paper parity hooks",
    )
    strategy_group = p_run.add_mutually_exclusive_group()
    strategy_group.add_argument(
        "--disciplined",
        action="store_true",
        default=True,
        help="Disciplined mode: category scoring + portfolio enforcement (DEFAULT)",
    )
    strategy_group.add_argument(
        "--beast",
        action="store_true",
        help="Beast mode: aggressive settings, no guardrails (not recommended)",
    )
    strategy_group.add_argument(
        "--safe-compounder",
        action="store_true",
        dest="safe_compounder",
        help="Safe Compounder: NO-side only, edge-based, near-certain outcomes",
    )
    p_run.add_argument(
        "--live-trade",
        action="store_true",
        dest="live_trade",
        help="Run only the live-trade decision loop in paper, shadow, or live mode",
    )
    p_run.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging verbosity (default: INFO)",
    )
    p_run.add_argument(
        "--once",
        action="store_true",
        help="Run one bounded ingest/trading pass and exit cleanly",
    )
    p_run.add_argument(
        "--smoke",
        action="store_true",
        help="Run a no-LLM startup smoke test (startup + ingestion only)",
    )
    p_run.add_argument(
        "--max-runtime-seconds",
        type=int,
        help="Force a clean shutdown after this many seconds",
    )
    p_run.set_defaults(func=cmd_run)

    # --- scores ---
    p_scores = subparsers.add_parser(
        "scores",
        help="Show current category scores",
        description="Display all trading category scores, win rates, ROI, and allocation limits.",
    )
    p_scores.set_defaults(func=cmd_scores)

    # --- history ---
    p_history = subparsers.add_parser(
        "history",
        help="Show trade history with category breakdown",
        description="Display closed trade history grouped by category, win rate, and P&L.",
    )
    p_history.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of recent trades to show (default: 50)",
    )
    p_history.set_defaults(func=cmd_history)

    # --- dashboard ---
    p_dash = subparsers.add_parser(
        "dashboard",
        help="Launch the Node dashboard stack",
        description="Open the Next.js web UI, Fastify API, and FastAPI analysis bridge for the current dashboard experience.",
    )
    p_dash.set_defaults(func=cmd_dashboard)

    # --- status ---
    p_status = subparsers.add_parser(
        "status",
        help="Show portfolio balance, positions, and P&L",
        description="Connect to the Kalshi API and display current account balance, open positions, and estimated portfolio value.",
    )
    p_status.set_defaults(func=cmd_status)

    # --- backtest ---
    p_bt = subparsers.add_parser(
        "backtest",
        help="Run backtests (coming soon)",
        description="Backtest trading strategies against historical market data. This feature is under development.",
    )
    p_bt.set_defaults(func=cmd_backtest)

    # --- health ---
    p_health = subparsers.add_parser(
        "health",
        help="Verify API connections, database, and configuration",
        description="Run a series of diagnostic checks: .env presence, API key configuration, Kalshi API connectivity, database initialization, and Python version.",
    )
    p_health.set_defaults(func=cmd_health)

    # --- mcp/operator API ---
    p_mcp = subparsers.add_parser(
        "mcp",
        help="Run the localhost operator API with MCP-style tool endpoints",
        description="Expose local market, book, portfolio, fill, and paper/live-gated order tools over HTTP.",
    )
    p_mcp.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p_mcp.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    p_mcp.set_defaults(func=cmd_mcp)

    # --- explain-market ---
    p_explain = subparsers.add_parser(
        "explain-market",
        help="Explain market contract mechanics",
        description="Fetch a market and run specialized contract interpreters such as weather bucket parsing.",
    )
    p_explain.add_argument("ticker", help="Kalshi market ticker to explain")
    p_explain.set_defaults(func=cmd_explain_market)

    # --- scan-arb ---
    p_arb = subparsers.add_parser(
        "scan-arb",
        help="Run an alert-only Kalshi vs Polymarket opportunity scan",
        description="Rank likely cross-market pricing gaps without placing orders.",
    )
    p_arb.add_argument("--limit", type=int, default=20, help="Candidates to print and persist")
    p_arb.add_argument("--kalshi-limit", type=int, default=100, help="Open Kalshi markets to scan")
    p_arb.add_argument("--polymarket-limit", type=int, default=100, help="Polymarket markets to fetch")
    p_arb.add_argument("--min-edge", type=float, default=0.03, help="Minimum estimated edge")
    p_arb.add_argument("--min-confidence", type=float, default=0.28, help="Minimum mapping confidence")
    p_arb.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Drop candidates that fail any quality guard "
            "(stale Polymarket trade, wide Kalshi spread, thin top "
            "liquidity, low Polymarket volume) instead of annotating them."
        ),
    )
    p_arb.set_defaults(func=cmd_scan_arb)

    # --- weather ---
    p_weather = subparsers.add_parser(
        "weather",
        help="Forecast-model probabilities and edge for a Kalshi weather event",
        description=(
            "Fetch a weather event's bucket markets, run the deterministic "
            "ensemble-forecast probability model (Open-Meteo + NWS), and print "
            "model P(YES) vs market price with fee-adjusted edge and Kelly size."
        ),
    )
    weather_target = p_weather.add_mutually_exclusive_group(required=True)
    weather_target.add_argument(
        "--event", help="Kalshi event ticker, e.g. KXHIGHNY-26JUN12"
    )
    weather_target.add_argument(
        "--ticker", help="Single market ticker, e.g. KXHIGHNY-26JUN12-B70.5"
    )
    p_weather.set_defaults(func=cmd_weather)

    # --- weather-scan ---
    p_weather_scan = subparsers.add_parser(
        "weather-scan",
        help="Sweep all open weather events with the deterministic model",
        description=(
            "Enumerate every open Kalshi weather event (station registry x "
            "KXHIGH/KXLOW), run the deterministic forecast model on each "
            "bucket, rank fee-positive divergences, and optionally execute "
            "them through the EV gate and portfolio guardrails. No LLM calls."
        ),
    )
    p_weather_scan.add_argument(
        "--trade",
        action="store_true",
        help=(
            "Execute ranked candidates (paper by default; live requires "
            "LIVE_TRADING_ENABLED and WEATHER_SCAN_LIVE)"
        ),
    )
    p_weather_scan.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    p_weather_scan.set_defaults(func=cmd_weather_scan)

    # --- safety-status ---
    p_safety = subparsers.add_parser(
        "safety-status",
        help="Show recent execution safety rejections",
        description="Display anomaly guard rejections and recent source-health snapshots.",
    )
    p_safety.add_argument("--limit", type=int, default=20, help="Rows to show")
    p_safety.set_defaults(func=cmd_safety_status)

    # --- refresh-calibration ---
    p_calib = subparsers.add_parser(
        "refresh-calibration",
        help="Rebuild settlement calibration rows from trade logs",
        description="Refresh the settlement_calibration table using stored trade outcomes.",
    )
    p_calib.set_defaults(func=cmd_refresh_calibration)

    # --- backfill-results ---
    p_backfill = subparsers.add_parser(
        "backfill-results",
        help="Fetch final YES/NO results for expired snapshotted markets",
        description=(
            "Batched Kalshi lookups that label the market_snapshots archive "
            "with settlement outcomes — the training data for the "
            "market-prior calibration. Run repeatedly to work through the "
            "backlog; the unified runtime also runs this hourly."
        ),
    )
    p_backfill.add_argument(
        "--limit", type=int, default=None, help="Max tickers to check this run"
    )
    p_backfill.add_argument(
        "--fit", action="store_true", help="Force a market-prior refit afterwards"
    )
    p_backfill.set_defaults(func=cmd_backfill_results)

    # --- auto ---
    p_auto = subparsers.add_parser(
        "auto",
        help="Beast Auto V1 -- unattended live-trade supervisor (separate from Manual)",
        description=(
            "Beast Auto V1 wraps the existing live-trade decision/risk/"
            "execution pipeline in an unattended loop with a dedicated "
            "fail-closed kill switch and idempotent order submission. "
            "Manual (`run --live-trade`, `run --live`) is unaffected. "
            "Quick Flip is never touched by Auto."
        ),
    )
    p_auto.add_argument(
        "auto_action",
        choices=["start", "kill", "resume", "status"],
        help=(
            "start: run the Auto supervisor loop. "
            "kill: activate the fail-closed kill switch (blocks new buys). "
            "resume: deactivate the kill switch. "
            "status: show current kill-switch state and Auto config."
        ),
    )
    p_auto.add_argument(
        "--live",
        action="store_true",
        help="Required for `start` -- Auto V1 has no paper/shadow mode; submits real orders.",
    )
    p_auto.add_argument(
        "--reason",
        default=None,
        help="Optional reason recorded when using `kill`.",
    )
    p_auto.set_defaults(func=cmd_auto)

    # --- fit-market-prior ---
    p_fit_prior = subparsers.add_parser(
        "fit-market-prior",
        help="Refit the market-prior calibration from labelled snapshots",
        description=(
            "Fits per-horizon Platt scalers mapping market mid price to "
            "realized settlement probability (favorite-longshot correction). "
            "Segments activate only when they beat the raw mid on a "
            "ticker-level holdout."
        ),
    )
    p_fit_prior.set_defaults(func=cmd_fit_market_prior)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
