"""
Unified model routing layer for the Kalshi AI Trading Bot.

Routes requests across Codex, direct OpenAI, and OpenRouter providers while
keeping aggregate cost tracking, daily budget enforcement, and transparent
fallback behavior in one place.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.clients.shared_types import (
    DailyUsageTracker,
    TradingDecision,
    load_daily_tracker_pickle,
)
from src.clients.openai_client import OpenAIClient
from src.clients.openrouter_client import OpenRouterClient, MODEL_PRICING
from src.clients.codex_client import CodexClient, is_codex_authenticated, resolve_codex_cli_path
from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin


# ---------------------------------------------------------------------------
# Capability-to-model mapping — all via OpenRouter (April 2026 models)
# ---------------------------------------------------------------------------

CAPABILITY_MAP: Dict[str, List[Tuple[str, str]]] = {
    "fast": [
        ("x-ai/grok-4.1-fast", "openrouter"),
        ("google/gemini-3.1-pro-preview", "openrouter"),
    ],
    "cheap": [
        ("deepseek/deepseek-v3.2", "openrouter"),
        ("google/gemini-3.1-pro-preview", "openrouter"),
    ],
    "reasoning": [
        ("anthropic/claude-sonnet-4.5", "openrouter"),
        ("openai/gpt-5.4", "openrouter"),
        ("google/gemini-3.1-pro-preview", "openrouter"),
    ],
    "balanced": [
        ("anthropic/claude-sonnet-4.5", "openrouter"),
        ("openai/gpt-5.4", "openrouter"),
        ("x-ai/grok-4.1-fast", "openrouter"),
    ],
}

# Full fleet: ordered by quality/priority for fallback chains.
FULL_FLEET: List[Tuple[str, str]] = [
    ("anthropic/claude-sonnet-4.5", "openrouter"),
    ("google/gemini-3.1-pro-preview", "openrouter"),
    ("openai/gpt-5.4", "openrouter"),
    ("deepseek/deepseek-v3.2", "openrouter"),
    ("x-ai/grok-4.1-fast", "openrouter"),
]

OPENAI_CAPABILITY_MAP: Dict[str, List[Tuple[str, str]]] = {
    "fast": [
        ("openai/gpt-4.1", "openai"),
        ("openai/o3", "openai"),
    ],
    "cheap": [
        ("openai/gpt-4.1", "openai"),
        ("openai/o3", "openai"),
    ],
    "reasoning": [
        ("openai/gpt-5.4", "openai"),
        ("openai/o3", "openai"),
        ("openai/gpt-4.1", "openai"),
    ],
    "balanced": [
        ("openai/gpt-5.4", "openai"),
        ("openai/gpt-4.1", "openai"),
        ("openai/o3", "openai"),
    ],
}

OPENAI_FULL_FLEET: List[Tuple[str, str]] = [
    ("openai/gpt-5.4", "openai"),
    ("openai/o3", "openai"),
    ("openai/gpt-4.1", "openai"),
]

# Codex CLI fleet — free plan-quota usage via a signed-in ChatGPT plan.
# Models exposed by the Codex CLI track the OpenAI reasoning lineup but are
# invoked through subprocess, not via OpenAI billing.
CODEX_CAPABILITY_MAP: Dict[str, List[Tuple[str, str]]] = {
    "fast": [
        ("codex/gpt-5.4-mini", "codex"),
        ("codex/gpt-5.4", "codex"),
    ],
    "cheap": [
        ("codex/gpt-5.4-mini", "codex"),
        ("codex/gpt-5.4", "codex"),
    ],
    "reasoning": [
        ("codex/gpt-5.4", "codex"),
        ("codex/gpt-5.4-mini", "codex"),
    ],
    "balanced": [
        ("codex/gpt-5.4", "codex"),
        ("codex/gpt-5.4-mini", "codex"),
    ],
}

CODEX_FULL_FLEET: List[Tuple[str, str]] = [
    ("codex/gpt-5.4", "codex"),
    ("codex/gpt-5.4-mini", "codex"),
]


# ---------------------------------------------------------------------------
# Per-model health tracking
# ---------------------------------------------------------------------------

@dataclass
class ModelHealth:
    """Tracks success/failure rates for a single model."""
    model: str
    provider: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    total_latency: float = 0.0  # cumulative seconds

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0  # Assume healthy until proven otherwise
        return self.successful_requests / self.total_requests

    @property
    def avg_latency(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency / self.successful_requests

    @property
    def is_healthy(self) -> bool:
        """
        A model is considered unhealthy if it has 5+ consecutive failures
        and the last failure was within the past 5 minutes.
        """
        if self.consecutive_failures < 5:
            return True
        if self.last_failure_time is None:
            return True
        cooldown = timedelta(minutes=5)
        return datetime.now() - self.last_failure_time > cooldown

    def record_success(self, latency: float) -> None:
        self.total_requests += 1
        self.successful_requests += 1
        self.consecutive_failures = 0
        self.last_success_time = datetime.now()
        self.total_latency += latency

    def record_failure(self) -> None:
        self.total_requests += 1
        self.failed_requests += 1
        self.consecutive_failures += 1
        self.last_failure_time = datetime.now()


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------

class ModelRouter(TradingLoggerMixin):
    """
    Unified routing layer that dispatches AI requests through the active provider.

    Usage::

        router = ModelRouter()
        # By capability
        text = await router.get_completion("prompt", capability="fast")
        # By explicit model
        text = await router.get_completion("prompt", model="openai/gpt-5.4")
        # Trading decision
        decision = await router.get_trading_decision(market, portfolio)
    """

    def __init__(
        self,
        openai_client: Optional[OpenAIClient] = None,
        openrouter_client: Optional[OpenRouterClient] = None,
        codex_client: Optional[CodexClient] = None,
        db_manager: Any = None,
        # xai_client param accepted for backward compat but ignored; routing
        # now goes through the configured provider stack.
        xai_client: Any = None,
    ):
        self.db_manager = db_manager
        self.default_provider = settings.api.resolve_llm_provider()

        self.openai_client: Optional[OpenAIClient] = openai_client
        self.openrouter_client: Optional[OpenRouterClient] = openrouter_client
        self.codex_client: Optional[CodexClient] = codex_client

        # Daily cost tracking (persisted via pickle, shared with OpenRouterClient)
        self.daily_tracker: DailyUsageTracker = self._load_daily_tracker()

        # Build health trackers for all supported provider fleets.
        self.model_health: Dict[str, ModelHealth] = {}
        for model_name, provider in self._all_fleets():
            key = self._model_key(model_name, provider)
            self.model_health[key] = ModelHealth(model=model_name, provider=provider)

        codex_cli_path = resolve_codex_cli_path()
        codex_available = self.codex_client is not None or bool(
            codex_cli_path and is_codex_authenticated(codex_cli_path)
        )
        openai_available = self.openai_client is not None or bool(
            getattr(settings.api, "openai_api_key", "")
        )
        openrouter_available = self.openrouter_client is not None or bool(
            getattr(settings.api, "openrouter_api_key", "")
        )

        self.logger.info(
            "ModelRouter initialized",
            default_provider=self.default_provider,
            openai_available=openai_available,
            openrouter_available=openrouter_available,
            codex_available=codex_available,
            codex_cli_path=codex_cli_path,
            fleet_size=len(self._active_fleet()),
        )

    # ------------------------------------------------------------------
    # Daily cost tracking
    # ------------------------------------------------------------------

    def _load_daily_tracker(self) -> DailyUsageTracker:
        """Load or create daily usage tracker (shared with OpenRouterClient)."""
        import os

        today = datetime.now().strftime("%Y-%m-%d")
        usage_file = "logs/daily_ai_usage.pkl"
        daily_limit = getattr(settings.trading, "daily_ai_cost_limit", 10.0)

        os.makedirs("logs", exist_ok=True)

        try:
            if os.path.exists(usage_file):
                with open(usage_file, "rb") as f:
                    tracker = load_daily_tracker_pickle(f)
                if tracker.date != today:
                    tracker = DailyUsageTracker(date=today, daily_limit=daily_limit)
                else:
                    tracker.daily_limit = daily_limit
                    if tracker.is_exhausted and tracker.total_cost < daily_limit:
                        tracker.is_exhausted = False
                return tracker
        except Exception as e:
            self.logger.warning(f"Failed to load daily tracker: {e}")

        return DailyUsageTracker(date=today, daily_limit=daily_limit)

    def _save_daily_tracker(self) -> None:
        import os
        import pickle

        try:
            os.makedirs("logs", exist_ok=True)
            with open("logs/daily_ai_usage.pkl", "wb") as f:
                pickle.dump(self.daily_tracker, f)
        except Exception as e:
            self.logger.error(f"Failed to save daily tracker: {e}")

    def _update_daily_cost(self, cost: float) -> None:
        """Update daily cost tracking."""
        self.daily_tracker.total_cost += cost
        self.daily_tracker.request_count += 1
        self._save_daily_tracker()

        if self.daily_tracker.total_cost >= self.daily_tracker.daily_limit:
            self.daily_tracker.is_exhausted = True
            self.daily_tracker.last_exhausted_time = datetime.now()
            self._save_daily_tracker()
            self.logger.warning(
                "Daily AI cost limit reached — trading paused until tomorrow.",
                daily_cost=self.daily_tracker.total_cost,
                daily_limit=self.daily_tracker.daily_limit,
            )

    async def check_daily_limits(self) -> bool:
        """
        Returns True if we can proceed with AI calls, False if daily limit reached.
        Beast-mode-bot calls this before each trading cycle.
        """
        # Reload to catch changes from other processes
        self.daily_tracker = self._load_daily_tracker()

        if self.daily_tracker.is_exhausted:
            now = datetime.now()
            if self.daily_tracker.date != now.strftime("%Y-%m-%d"):
                # New day — reset
                self.daily_tracker = DailyUsageTracker(
                    date=now.strftime("%Y-%m-%d"),
                    daily_limit=self.daily_tracker.daily_limit,
                )
                self._save_daily_tracker()
                self.logger.info("New day — daily AI limits reset")
                return True

            self.logger.info(
                "Daily AI limit reached — request skipped",
                daily_cost=self.daily_tracker.total_cost,
                daily_limit=self.daily_tracker.daily_limit,
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _model_key(model: str, provider: str) -> str:
        return f"{provider}::{model}"

    @staticmethod
    def _all_fleets() -> List[Tuple[str, str]]:
        """Return the union of supported model/provider pairs."""
        return list(dict.fromkeys(FULL_FLEET + OPENAI_FULL_FLEET + CODEX_FULL_FLEET))

    def _active_capability_map(self) -> Dict[str, List[Tuple[str, str]]]:
        """Return the capability map for the selected provider mode."""
        if self.default_provider == "codex":
            return CODEX_CAPABILITY_MAP
        if self.default_provider == "openai":
            return OPENAI_CAPABILITY_MAP
        return CAPABILITY_MAP

    def _active_fleet(self) -> List[Tuple[str, str]]:
        """Return the active fleet for the selected provider mode."""
        if self.default_provider == "codex":
            return CODEX_FULL_FLEET
        if self.default_provider == "openai":
            return OPENAI_FULL_FLEET
        return FULL_FLEET

    def _ensure_openrouter(self) -> OpenRouterClient:
        """Return the OpenRouter client, creating it on first use if needed."""
        if self.openrouter_client is None:
            self.openrouter_client = OpenRouterClient(db_manager=self.db_manager)
            self.logger.info("Lazily initialized OpenRouterClient")
        return self.openrouter_client

    def _ensure_openai(self) -> OpenAIClient:
        """Return the OpenAI client, creating it on first use if needed."""
        if self.openai_client is None:
            self.openai_client = OpenAIClient(db_manager=self.db_manager)
            self.logger.info("Lazily initialized OpenAIClient")
        return self.openai_client

    def _ensure_codex(self) -> CodexClient:
        """Return the Codex CLI client, creating it on first use if needed."""
        if self.codex_client is None:
            self.codex_client = CodexClient(db_manager=self.db_manager)
            self.logger.info("Lazily initialized CodexClient")
        return self.codex_client

    def _get_client(self, provider: str):
        """Return the provider-specific client instance."""
        if provider == "codex":
            return self._ensure_codex()
        if provider == "openai":
            return self._ensure_openai()
        return self._ensure_openrouter()

    def _infer_provider(self, model: str) -> str:
        """
        Infer the provider to use for a requested model.

        Explicit Codex models always go through the Codex CLI. When the active
        default is OpenRouter, non-Codex models stay on OpenRouter so callers
        can request `openai/...` models via OpenRouter-native routing. Direct
        OpenAI is used only for models in the OpenAI fleet when OpenRouter is
        not the active default. Everything else falls back to OpenRouter.
        """
        normalized = str(model or "").strip()
        bare_name = normalized.split("/", 1)[-1] if normalized else ""
        codex_models = {name for name, _ in CODEX_FULL_FLEET}
        openai_models = {name for name, _ in OPENAI_FULL_FLEET}
        codex_bare_models = {name.split("/", 1)[-1] for name in codex_models}
        openai_bare_models = {name.split("/", 1)[-1] for name in openai_models}

        if normalized in codex_models or normalized.startswith("codex/"):
            return "codex"
        if self.default_provider == "openrouter":
            return "openrouter"
        if (
            self.default_provider == "codex"
            and "/" not in normalized
            and bare_name in codex_bare_models
        ):
            return "codex"
        if normalized in openai_models or normalized.startswith("openai/"):
            return "openai"
        if bare_name in openai_bare_models:
            return "openai"
        if bare_name in codex_bare_models:
            return "codex"
        return "openrouter"

    def _fleet_for_provider(self, provider: str) -> List[Tuple[str, str]]:
        """Return the canonical fallback fleet for a provider."""
        if provider == "codex":
            return CODEX_FULL_FLEET
        if provider == "openai":
            return OPENAI_FULL_FLEET
        return FULL_FLEET

    def _resolve_targets(
        self,
        model: Optional[str] = None,
        capability: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """
        Produce an ordered list of (model, provider) tuples to attempt.

        Priority:
        1. Explicit *model* (+ fallback chain).
        2. *capability* mapping (+ fallback chain).
        3. Full fleet sorted by health / success rate.
        """
        targets: List[Tuple[str, str]] = []
        fleet = self._active_fleet()

        if model is not None:
            provider = self._infer_provider(model)
            targets.append((model, provider))
            fleet = self._fleet_for_provider(provider)
        elif capability is not None:
            cap_targets = self._active_capability_map().get(capability, [])
            targets.extend(cap_targets)
        else:
            targets = list(fleet)

        # Append remaining fleet members not yet in the list
        seen = set(targets)
        for entry in fleet:
            if entry not in seen:
                targets.append(entry)
                seen.add(entry)

        # Filter out unhealthy models (keep at least 2)
        healthy = [t for t in targets if self._is_model_healthy(t[0], t[1])]
        if len(healthy) >= 2:
            targets = healthy

        return targets

    def _is_model_healthy(self, model: str, provider: str) -> bool:
        key = self._model_key(model, provider)
        health = self.model_health.get(key)
        if health is None:
            return True
        return health.is_healthy

    def _record_success(self, model: str, provider: str, latency: float) -> None:
        key = self._model_key(model, provider)
        health = self.model_health.get(key)
        if health is None:
            health = ModelHealth(model=model, provider=provider)
            self.model_health[key] = health
        health.record_success(latency)

    def _record_failure(self, model: str, provider: str) -> None:
        key = self._model_key(model, provider)
        health = self.model_health.get(key)
        if health is None:
            health = ModelHealth(model=model, provider=provider)
            self.model_health[key] = health
        health.record_failure()

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------

    async def _dispatch_completion(
        self,
        prompt: str,
        model: str,
        provider: str,
        fallback_models: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        strategy: str = "unknown",
        query_type: str = "completion",
        role: Optional[str] = None,
        market_id: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        provider_preferences: Optional[Dict[str, Any]] = None,
        plugins: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        trace: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Send a completion request through the selected provider."""
        client = self._get_client(provider)
        return await client.get_completion(
            prompt=prompt,
            model=model,
            fallback_models=fallback_models,
            max_tokens=max_tokens,
            temperature=temperature,
            strategy=strategy,
            query_type=query_type,
            role=role,
            market_id=market_id,
            response_format=response_format,
            provider=provider_preferences,
            plugins=plugins,
            metadata=metadata,
            session_id=session_id,
            trace=trace,
        )

    async def _dispatch_trading_decision(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        news_summary: str,
        model: str,
        provider: str,
        role: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        provider_preferences: Optional[Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        plugins: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        trace: Optional[Dict[str, Any]] = None,
    ) -> Optional[TradingDecision]:
        """Request a trading decision through the selected provider."""
        client = self._get_client(provider)
        return await client.get_trading_decision(
            market_data=market_data,
            portfolio_data=portfolio_data,
            news_summary=news_summary,
            model=model,
            role=role,
            fallback_models=fallback_models,
            provider=provider_preferences,
            response_format=response_format,
            plugins=plugins,
            metadata=metadata,
            session_id=session_id,
            trace=trace,
        )

    # ------------------------------------------------------------------
    # Public API: get_completion
    # ------------------------------------------------------------------

    async def get_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        capability: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        strategy: str = "unknown",
        query_type: str = "completion",
        role: Optional[str] = None,
        market_id: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        provider_preferences: Optional[Dict[str, Any]] = None,
        plugins: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        trace: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Get a completion routed to the best available model/provider.

        Args:
            prompt: The user/system prompt.
            model: Explicit model identifier (e.g. ``"anthropic/claude-sonnet-4.5"``).
            capability: Capability hint — ``"fast"``, ``"reasoning"``,
                ``"balanced"``, or ``"cheap"``.  Ignored if *model* is given.
            temperature: Sampling temperature override.
            max_tokens: Max output tokens override.
            strategy: Strategy label for logging.
            query_type: Query type label for logging.
            market_id: Optional Kalshi market id for logging.

        Returns:
            Response text, or ``None`` if all models fail.
        """
        targets = self._resolve_targets(model=model, capability=capability)
        primary_model, primary_provider = targets[0]
        fallback_models = [target_model for target_model, _ in targets[1:]]
        start = time.time()

        try:
            result = await self._dispatch_completion(
                prompt=prompt,
                model=primary_model,
                provider=primary_provider,
                fallback_models=fallback_models,
                temperature=temperature,
                max_tokens=max_tokens,
                strategy=strategy,
                query_type=query_type,
                role=role,
                market_id=market_id,
                response_format=response_format,
                provider_preferences=provider_preferences,
                plugins=plugins,
                metadata=metadata,
                session_id=session_id,
                trace=trace,
            )

            if result is not None:
                client = self._get_client(primary_provider)
                actual_model = client.last_request_metadata.actual_model or primary_model
                actual_provider = self._infer_provider(actual_model)
                self._record_success(actual_model, actual_provider, time.time() - start)
                self.logger.debug(
                    "Completion routed successfully",
                    requested_model=primary_model,
                    actual_model=actual_model,
                    fallback_models=fallback_models,
                    latency=round(time.time() - start, 2),
                )
                return result

            self._record_failure(primary_model, primary_provider)
            self.logger.warning(
                "Provider returned no completion after fallbacks",
                requested_model=primary_model,
                provider=primary_provider,
                fallback_models=fallback_models,
            )
            return None

        except Exception as exc:
            self._record_failure(primary_model, primary_provider)
            self.logger.warning(
                "Completion routing failed",
                requested_model=primary_model,
                fallback_models=fallback_models,
                error=str(exc),
            )
            return None

    async def get_researched_completion(
        self,
        *,
        prompt: str,
        instructions: Optional[str] = None,
        model: Optional[str] = None,
        capability: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        text_format: Optional[Dict[str, Any]] = None,
        search_allowed_domains: Optional[List[str]] = None,
        search_context_size: str = "medium",
        strategy: str = "unknown",
        query_type: str = "researched_completion",
        market_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        use_web_research: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a researched completion.

        Direct OpenAI mode can use native web search via the Responses API.
        Other provider modes fall back to a standard completion on the same
        structured prompt, preserving the response shape without web search.
        """
        if self.default_provider == "openai":
            openai_client = self._ensure_openai()
            start = time.time()
            result = await openai_client.get_researched_completion(
                prompt=prompt,
                instructions=instructions,
                model=model,
                text_format=text_format,
                search_allowed_domains=search_allowed_domains,
                search_context_size=search_context_size,
                strategy=strategy,
                query_type=query_type,
                market_id=market_id,
                metadata=metadata,
                use_web_search=use_web_research,
            )

            if result is not None:
                actual_model = openai_client.last_request_metadata.actual_model or model or openai_client.default_model
                self._record_success(actual_model, "openai", time.time() - start)
                return result

            self._record_failure(model or openai_client.default_model, "openai")
            return None

        combined_prompt = prompt
        if instructions:
            combined_prompt = f"{instructions}\n\n{prompt}"

        content = await self.get_completion(
            prompt=combined_prompt,
            model=model,
            capability=capability or "reasoning",
            strategy=strategy,
            query_type=query_type,
            market_id=market_id,
            response_format=response_format,
            metadata=metadata,
        )
        if content is None:
            return None

        return {
            "content": content,
            "sources": [],
            "used_web_research": False,
        }

    # ------------------------------------------------------------------
    # Public API: get_trading_decision
    # ------------------------------------------------------------------

    async def get_trading_decision(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        news_summary: str = "",
        model: Optional[str] = None,
        capability: Optional[str] = None,
        provider_preferences: Optional[Dict[str, Any]] = None,
        role: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        plugins: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        trace: Optional[Dict[str, Any]] = None,
    ) -> Optional[TradingDecision]:
        """
        Get a trading decision from the best available model/provider.

        Args:
            market_data: Market information dict.
            portfolio_data: Portfolio / balance information dict.
            news_summary: Optional news context.
            model: Explicit model identifier.
            capability: Capability hint (see ``get_completion``).

        Returns:
            A ``TradingDecision`` or ``None`` if all models fail.
        """
        targets = self._resolve_targets(model=model, capability=capability)
        primary_model, primary_provider = targets[0]
        fallback_models = [target_model for target_model, _ in targets[1:]]
        start = time.time()

        try:
            decision = await self._dispatch_trading_decision(
                market_data=market_data,
                portfolio_data=portfolio_data,
                news_summary=news_summary,
                model=primary_model,
                provider=primary_provider,
                fallback_models=fallback_models,
                provider_preferences=provider_preferences,
                role=role,
                response_format=response_format,
                plugins=plugins,
                metadata=metadata,
                session_id=session_id,
                trace=trace,
            )

            if decision is not None:
                client = self._get_client(primary_provider)
                actual_model = client.last_request_metadata.actual_model or primary_model
                actual_provider = self._infer_provider(actual_model)
                self._record_success(actual_model, actual_provider, time.time() - start)
                self.logger.info(
                    "Trading decision routed successfully",
                    requested_model=primary_model,
                    actual_model=actual_model,
                    action=decision.action,
                    confidence=decision.confidence,
                    latency=round(time.time() - start, 2),
                )
                return decision

            self._record_failure(primary_model, primary_provider)
            self.logger.warning(
                "Provider returned no trading decision after fallbacks",
                requested_model=primary_model,
                provider=primary_provider,
                fallback_models=fallback_models,
            )
            return None

        except Exception as exc:
            self._record_failure(primary_model, primary_provider)
            self.logger.warning(
                "Trading decision routing failed",
                requested_model=primary_model,
                fallback_models=fallback_models,
                error=str(exc),
            )
            return None

    # ------------------------------------------------------------------
    # Aggregate cost tracking
    # ------------------------------------------------------------------

    def get_total_cost(self) -> float:
        """Return aggregate cost across all providers."""
        total = 0.0
        if self.openai_client:
            total += self.openai_client.total_cost
        if self.openrouter_client:
            total += self.openrouter_client.total_cost
        if self.codex_client:
            # Codex plan usage is $0 metered but we include it for parity.
            total += self.codex_client.total_cost
        return total

    def get_total_requests(self) -> int:
        """Return aggregate request count across all providers."""
        total = 0
        if self.openai_client:
            total += self.openai_client.request_count
        if self.openrouter_client:
            total += self.openrouter_client.request_count
        if self.codex_client:
            total += self.codex_client.request_count
        return total

    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Return a comprehensive cost and health summary.
        """
        self.daily_tracker = self._load_daily_tracker()
        summary: Dict[str, Any] = {
            "total_cost": round(self.get_total_cost(), 6),
            "total_requests": self.get_total_requests(),
            "providers": {},
            "model_health": {},
            "daily": {
                "cost": round(self.daily_tracker.total_cost, 6),
                "limit": self.daily_tracker.daily_limit,
                "requests": self.daily_tracker.request_count,
                "is_exhausted": self.daily_tracker.is_exhausted,
            },
        }

        if self.openai_client:
            summary["providers"]["openai"] = self.openai_client.get_cost_summary()
        if self.openrouter_client:
            summary["providers"]["openrouter"] = self.openrouter_client.get_cost_summary()
        if self.codex_client:
            summary["providers"]["codex"] = self.codex_client.get_cost_summary()

        for key, health in self.model_health.items():
            if health.total_requests > 0:
                summary["model_health"][key] = {
                    "model": health.model,
                    "provider": health.provider,
                    "total_requests": health.total_requests,
                    "success_rate": round(health.success_rate, 4),
                    "avg_latency": round(health.avg_latency, 3),
                    "consecutive_failures": health.consecutive_failures,
                    "is_healthy": health.is_healthy,
                }

        return summary

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Shut down all provider clients."""
        tasks = []
        if self.openai_client:
            tasks.append(self.openai_client.close())
        if self.openrouter_client:
            tasks.append(self.openrouter_client.close())
        if self.codex_client:
            tasks.append(self.codex_client.close())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self.logger.info(
            "ModelRouter closed",
            total_cost=round(self.get_total_cost(), 6),
            total_requests=self.get_total_requests(),
        )
