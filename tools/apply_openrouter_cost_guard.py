"""Apply the OpenRouter low-credit circuit breaker and compact live-trade token caps.

This is intentionally a surgical source patcher so the change can be reviewed on a
branch before it is applied to the working tree. It does not alter trading gates,
position sizing, Kelly, spread/depth checks, or execution guardrails.
"""
from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"Expected source block not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    openrouter = Path("src/clients/openrouter_client.py")
    live_trade = Path("src/jobs/live_trade.py")

    replace_once(
        openrouter,
        '''    @staticmethod\n    def _is_retryable_error(exc: Exception) -> bool:\n        error_str = str(exc).lower()\n        if OpenRouterClient._extract_affordable_max_tokens(exc) is not None:\n            return True\n''',
        '''    @staticmethod\n    def _is_prompt_credit_exhaustion_error(exc: Exception) -> bool:\n        \"\"\"Return True when available credit cannot even fund the input prompt.\"\"\"\n        error_str = str(exc).lower()\n        return (\n            \"402\" in error_str\n            and (\n                \"prompt tokens limit exceeded\" in error_str\n                or \"more credits\" in error_str\n            )\n        )\n\n    @staticmethod\n    def _is_retryable_error(exc: Exception) -> bool:\n        error_str = str(exc).lower()\n        if OpenRouterClient._is_prompt_credit_exhaustion_error(exc):\n            return False\n        if OpenRouterClient._extract_affordable_max_tokens(exc) is not None:\n            return True\n''',
    )

    replace_once(
        openrouter,
        '''                is_retryable = self._is_retryable_error(exc)\n                affordable_max_tokens = self._extract_affordable_max_tokens(exc)\n                self.logger.warning(\n''',
        '''                prompt_credit_exhausted = self._is_prompt_credit_exhaustion_error(exc)\n                is_retryable = self._is_retryable_error(exc)\n                affordable_max_tokens = self._extract_affordable_max_tokens(exc)\n                self.logger.warning(\n''',
    )

    replace_once(
        openrouter,
        '''                if (\n                    affordable_max_tokens is not None\n                    and attempt < self.MAX_RETRIES_PER_REQUEST - 1\n                ):\n''',
        '''                if prompt_credit_exhausted:\n                    self.logger.error(\n                        \"OpenRouter credit cannot fund prompt; circuit breaker opened\",\n                        model=model,\n                        attempt=attempt + 1,\n                    )\n                    break\n\n                if (\n                    affordable_max_tokens is not None\n                    and attempt < self.MAX_RETRIES_PER_REQUEST - 1\n                ):\n''',
    )

    replace_once(
        live_trade,
        '''            response_format=_response_format("live_trade_specialist", SPECIALIST_SCHEMA),\n        )\n''',
        '''            response_format=_response_format("live_trade_specialist", SPECIALIST_SCHEMA),\n            max_tokens=512,\n        )\n''',
    )

    replace_once(
        live_trade,
        '''                    market_id=selected_candidate.get("market_ticker") or selected_candidate.get("event_ticker"),\n                    **request_options,\n                )\n''',
        '''                    market_id=selected_candidate.get("market_ticker") or selected_candidate.get("event_ticker"),\n                    max_tokens=min(int(request_options.pop("max_tokens", 768) or 768), 768),\n                    **request_options,\n                )\n''',
    )

    print("Applied OpenRouter low-credit circuit breaker and live-trade token caps.")


if __name__ == "__main__":
    main()
