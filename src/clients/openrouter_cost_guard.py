"""Runtime OpenRouter cost guards for live-trade requests.

The guard is deliberately narrow: it changes retry/cost behavior only. Trading
thresholds, Kelly sizing, spread/depth checks, kill switches, and execution
rules are untouched.
"""
from __future__ import annotations

from typing import Optional

from src.clients.openrouter_client import OpenRouterClient

_INSTALLED = False
_PROMPT_CREDIT_EXHAUSTED = False
_ORIGINAL_GET_COMPLETION = OpenRouterClient.get_completion
_ORIGINAL_IS_RETRYABLE_ERROR = OpenRouterClient._is_retryable_error
_ORIGINAL_EXTRACT_AFFORDABLE_MAX_TOKENS = OpenRouterClient._extract_affordable_max_tokens


def _is_prompt_credit_exhaustion_error(exc: Exception) -> bool:
    """True when OpenRouter says the available credit cannot fund the prompt."""
    text = str(exc).lower()
    return "402" in text and "prompt tokens limit exceeded" in text


def _trip_prompt_credit_exhaustion_latch(exc: Exception) -> bool:
    """Latch the process closed after the first unaffordable-input 402."""
    global _PROMPT_CREDIT_EXHAUSTED
    if _is_prompt_credit_exhaustion_error(exc):
        _PROMPT_CREDIT_EXHAUSTED = True
        return True
    return False


def prompt_credit_exhausted() -> bool:
    """Return whether this process has already hit an unaffordable prompt 402."""
    return _PROMPT_CREDIT_EXHAUSTED


def reset_prompt_credit_exhaustion_latch() -> None:
    """Test/support helper. A normal process restart also clears the latch."""
    global _PROMPT_CREDIT_EXHAUSTED
    _PROMPT_CREDIT_EXHAUSTED = False


def _guarded_is_retryable_error(exc: Exception) -> bool:
    if _trip_prompt_credit_exhaustion_latch(exc):
        return False
    return _ORIGINAL_IS_RETRYABLE_ERROR(exc)


def _guarded_extract_affordable_max_tokens(exc: Exception) -> Optional[int]:
    # The original retry path lowers output tokens and retries. That is useful
    # only when the prompt itself is still affordable. If the prompt is already
    # over the credit limit, returning None prevents a doomed retry loop.
    if _trip_prompt_credit_exhaustion_latch(exc):
        return None
    return _ORIGINAL_EXTRACT_AFFORDABLE_MAX_TOKENS(exc)


def _cap_live_trade_max_tokens(
    strategy: str,
    query_type: str,
    requested: Optional[int],
) -> Optional[int]:
    if strategy != "live_trade":
        return requested

    if query_type.endswith("_specialist"):
        cap = 512
    elif query_type.startswith("live_trade_final_"):
        cap = 768
    else:
        return requested

    if requested is None:
        return cap
    return min(int(requested), cap)


async def _guarded_get_completion(self, prompt=None, **kwargs):
    # Once OpenRouter has explicitly said the account cannot fund the input
    # prompt, do not keep hitting the API on every later Auto cycle. A fresh
    # process (after credits/config are deliberately changed) clears the latch.
    if _PROMPT_CREDIT_EXHAUSTED:
        return None

    strategy = str(kwargs.get("strategy", "unknown"))
    query_type = str(kwargs.get("query_type", "completion"))
    kwargs["max_tokens"] = _cap_live_trade_max_tokens(
        strategy,
        query_type,
        kwargs.get("max_tokens"),
    )
    return await _ORIGINAL_GET_COMPLETION(self, prompt, **kwargs)


def install_openrouter_cost_guard() -> None:
    """Install the guard once for every OpenRouterClient instance."""
    global _INSTALLED
    if _INSTALLED:
        return

    OpenRouterClient._is_retryable_error = staticmethod(_guarded_is_retryable_error)
    OpenRouterClient._extract_affordable_max_tokens = staticmethod(
        _guarded_extract_affordable_max_tokens
    )
    OpenRouterClient.get_completion = _guarded_get_completion
    _INSTALLED = True


install_openrouter_cost_guard()
