"""
Pytest bootstrap for local repo imports and opt-in live Kalshi tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live-kalshi",
        action="store_true",
        default=False,
        help="Run opt-in tests that talk to the real or demo Kalshi API.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live_kalshi: test requires Kalshi credentials and network access",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_live = config.getoption("--live-kalshi") or os.getenv("RUN_LIVE_KALSHI_TESTS", "").lower() == "true"
    if run_live:
        return

    skip_live = pytest.mark.skip(reason="requires --live-kalshi or RUN_LIVE_KALSHI_TESTS=true")
    for item in items:
        if "live_kalshi" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def _isolate_beast_auto_budget_env(request, monkeypatch):
    """Keep legacy AutoRunner tests independent from the new AI budget gate.

    Dedicated cost-circuit-breaker tests explicitly set and verify
    AUTO_AI_DAILY_COST_LIMIT. Older supervisor tests use MagicMock routers,
    whose numeric coercion can resemble nonzero spend, so disable only this
    extra gate for those tests and leave their intended behavior unchanged.
    """
    if request.node.fspath.basename == "test_auto_runner.py":
        monkeypatch.setenv("AUTO_AI_DAILY_COST_LIMIT", "0")
