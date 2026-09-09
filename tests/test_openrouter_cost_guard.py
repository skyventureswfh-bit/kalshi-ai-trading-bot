import pytest

from src.clients.openrouter_cost_guard import (
    _cap_live_trade_max_tokens,
    _guarded_extract_affordable_max_tokens,
    _guarded_get_completion,
    _guarded_is_retryable_error,
    prompt_credit_exhausted,
    reset_prompt_credit_exhaustion_latch,
)


class CreditError(Exception):
    pass


def setup_function():
    reset_prompt_credit_exhaustion_latch()


def test_prompt_credit_exhaustion_is_not_retried():
    exc = CreditError(
        "Error code: 402 - Prompt tokens limit exceeded; can only afford 44"
    )
    assert _guarded_is_retryable_error(exc) is False
    assert _guarded_extract_affordable_max_tokens(exc) is None
    assert prompt_credit_exhausted() is True


def test_reducible_output_credit_error_can_still_retry():
    exc = CreditError(
        "Error code: 402 - This request requires more credits, or fewer max_tokens; can only afford 500"
    )
    assert _guarded_is_retryable_error(exc) is True
    affordable = _guarded_extract_affordable_max_tokens(exc)
    assert affordable is not None
    assert 0 < affordable < 500
    assert prompt_credit_exhausted() is False


@pytest.mark.asyncio
async def test_prompt_credit_latch_short_circuits_later_requests_without_network_call():
    exc = CreditError(
        "Error code: 402 - Prompt tokens limit exceeded; can only afford 44"
    )
    assert _guarded_is_retryable_error(exc) is False

    # Once latched, _guarded_get_completion returns before touching `self`,
    # proving later cycles do not keep hammering OpenRouter.
    result = await _guarded_get_completion(
        object(), prompt="should never be sent", strategy="live_trade"
    )
    assert result is None


def test_live_trade_token_caps_are_narrow():
    assert _cap_live_trade_max_tokens("live_trade", "live_trade_weather_specialist", None) == 512
    assert _cap_live_trade_max_tokens("live_trade", "live_trade_final_trader", None) == 768
    assert _cap_live_trade_max_tokens("live_trade", "live_trade_final_trader", 200) == 200
    assert _cap_live_trade_max_tokens("other", "live_trade_final_trader", None) is None
