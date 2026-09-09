"""
Configuration settings for the Kalshi trading system.
Manages trading parameters, API configurations, and risk management settings.
"""

import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def _get_bool_env(name: str, default: bool = False) -> bool:
    """Return a boolean environment variable value."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _get_csv_env_list(name: str) -> List[str]:
    """Return a comma-separated environment variable as a cleaned list."""
    raw_value = os.getenv(name, "")
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _get_kalshi_env() -> str:
    """Return the configured Kalshi environment."""
    return os.getenv("KALSHI_ENV", "prod").strip().lower() or "prod"


def _get_kalshi_base_url() -> str:
    """Resolve the Kalshi REST base URL from env settings."""
    override = os.getenv("KALSHI_API_BASE_URL", "").strip()
    if override:
        return override.rstrip("/")

    env_name = _get_kalshi_env()
    if env_name == "demo":
        return "https://demo-api.kalshi.co"
    return "https://api.elections.kalshi.com"


def _get_llm_provider() -> str:
    """Return the requested LLM provider mode."""
    return os.getenv("LLM_PROVIDER", "auto").strip().lower() or "auto"


_CODEX_AUTH_CACHE: Dict[str, tuple[bool, float]] = {}
_CODEX_AUTH_CACHE_TTL_SECONDS = 30.0


def _resolve_codex_cli_path_from_env() -> Optional[str]:
    """Return the configured Codex CLI path without importing client modules."""
    override = os.getenv("CODEX_CLI_PATH", "").strip()
    if override:
        if os.path.isfile(override):
            return override
        resolved = shutil.which(override)
        return resolved
    return shutil.which("codex")


def _is_codex_cli_ready() -> bool:
    """
    Return ``True`` when the Codex CLI is both on PATH and authenticated.

    This helper intentionally avoids importing :mod:`src.clients.codex_client`.
    Settings are created while client modules may still be importing this
    module, so probing directly avoids a circular import during startup.
    """
    if os.getenv("CODEX_DISABLE_AUTH_PROBE", "").strip().lower() in {"1", "true", "yes"}:
        return False

    path = _resolve_codex_cli_path_from_env()
    if not path:
        return False

    now = time.time()
    cached = _CODEX_AUTH_CACHE.get(path)
    if cached is not None:
        value, expires_at = cached
        if expires_at > now:
            return value

    authenticated = False
    for argv in (
        (path, "login", "status"),
        (path, "auth", "status"),
    ):
        try:
            result = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue

        combined = f"{result.stdout}\n{result.stderr}".lower()
        if result.returncode != 0:
            continue
        if any(
            marker in combined
            for marker in (
                "not signed in",
                "not logged in",
                "logged out",
                "login required",
                "please log in",
                "please sign in",
                "unauthorized",
                "unauthenticated",
            )
        ):
            continue
        if combined.strip():
            authenticated = True
            break

    _CODEX_AUTH_CACHE[path] = (
        authenticated,
        now + _CODEX_AUTH_CACHE_TTL_SECONDS,
    )
    return authenticated


def _resolve_default_llm_provider() -> str:
    """
    Resolve the effective provider from environment state.

    `auto` prefers the Codex CLI (ChatGPT plan quota) when it is on PATH
    AND signed in, then direct OpenAI access when an API key is configured,
    and finally falls back to OpenRouter.
    """
    provider = _get_llm_provider()
    if provider != "auto":
        return provider
    if _is_codex_cli_ready():
        return "codex"
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai"
    return "openrouter"


def _get_default_primary_model() -> str:
    """Return the default primary model for the active provider."""
    env_override = os.getenv("PRIMARY_MODEL", "").strip()
    if env_override:
        return env_override

    provider = _resolve_default_llm_provider()
    if provider == "codex":
        return "codex/gpt-5.4"
    if provider == "openai":
        return "openai/gpt-5.4"
    return "anthropic/claude-sonnet-4.5"


def _get_default_fallback_model() -> str:
    """Return the default fallback model for the active provider."""
    env_override = os.getenv("FALLBACK_MODEL", "").strip()
    if env_override:
        return env_override

    provider = _resolve_default_llm_provider()
    if provider == "codex":
        return "codex/gpt-5.4-mini"
    if provider == "openai":
        return "openai/o3"
    return "deepseek/deepseek-v3.2"


def _get_default_sentiment_model() -> str:
    """Return the default sentiment model for the active provider."""
    env_override = os.getenv("SENTIMENT_MODEL", "").strip()
    if env_override:
        return env_override

    provider = _resolve_default_llm_provider()
    if provider == "codex":
        return "codex/gpt-5.4-mini"
    if provider == "openai":
        return "openai/gpt-4.1"
    return "google/gemini-3.1-flash-lite-preview"


@dataclass
class APIConfig:
    """API configuration settings."""
    kalshi_api_key: str = field(default_factory=lambda: os.getenv("KALSHI_API_KEY", ""))
    kalshi_env: str = field(default_factory=_get_kalshi_env)
    kalshi_base_url: str = field(default_factory=_get_kalshi_base_url)
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    llm_provider: str = field(default_factory=_get_llm_provider)
    openai_base_url: str = "https://api.openai.com/v1"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_http_referer: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    )
    openrouter_title: str = field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_TITLE", "Kalshi AI Trading Bot"
        ).strip()
    )

    # Codex CLI integration (ChatGPT plan quota)
    codex_cli_path: str = field(
        default_factory=lambda: os.getenv("CODEX_CLI_PATH", "").strip()
    )
    codex_plan_tier: str = field(
        default_factory=lambda: os.getenv("CODEX_PLAN_TIER", "plus").strip() or "plus"
    )

    # xai_api_key removed — all models now route through OpenRouter

    def get_openrouter_headers(self) -> Dict[str, str]:
        """Return optional OpenRouter attribution headers."""
        headers: Dict[str, str] = {}
        if self.openrouter_http_referer:
            headers["HTTP-Referer"] = self.openrouter_http_referer
        if self.openrouter_title:
            headers["X-OpenRouter-Title"] = self.openrouter_title
        return headers

    def resolve_llm_provider(self) -> str:
        """Return the effective provider after applying `auto` fallback rules."""
        if self.llm_provider != "auto":
            return self.llm_provider
        if _is_codex_cli_ready():
            return "codex"
        if self.openai_api_key:
            return "openai"
        return "openrouter"


@dataclass
class EnsembleConfig:
    """Multi-model ensemble configuration."""
    enabled: bool = True
    # Model roster for ensemble decisions — all via OpenRouter (April 2026)
    models: Dict[str, Dict] = field(default_factory=lambda: {
        "anthropic/claude-sonnet-4.5": {"provider": "openrouter", "role": "news_analyst", "weight": 0.30},
        "google/gemini-3.1-pro-preview": {"provider": "openrouter", "role": "forecaster", "weight": 0.30},
        "openai/gpt-5.4": {"provider": "openrouter", "role": "risk_manager", "weight": 0.20},
        "deepseek/deepseek-v3.2": {"provider": "openrouter", "role": "bull_researcher", "weight": 0.10},
        "x-ai/grok-4.1-fast": {"provider": "openrouter", "role": "bear_researcher", "weight": 0.10},
    })
    trader_model: str = "x-ai/grok-4.1-fast"
    min_models_for_consensus: int = 3
    disagreement_threshold: float = 0.25  # Std dev above this = low confidence
    parallel_requests: bool = True
    debate_enabled: bool = True
    calibration_tracking: bool = True
    max_ensemble_cost: float = 0.50  # Max cost per ensemble decision
    # Log-odds pooling extremization exponent (1.0 = plain pooling). Mild
    # extremization corrects the under-confidence of averaged forecasts.
    extremize_factor: float = field(
        default_factory=lambda: float(os.getenv("ENSEMBLE_EXTREMIZE_FACTOR", "1.2"))
    )
    # Weight on the pooled model probability when blending with the market
    # price in log-odds space; the remainder anchors to the market prior.
    market_blend_model_weight: float = field(
        default_factory=lambda: float(os.getenv("MARKET_BLEND_MODEL_WEIGHT", "0.65"))
    )

    @staticmethod
    def _normalize_role(role: str) -> str:
        """Normalize legacy role names to the current agent role set."""
        role = str(role or "").strip()
        if role == "lead_analyst":
            return "news_analyst"
        return role

    def normalized_models(self) -> Dict[str, Dict[str, Any]]:
        """Return the configured model map with normalized role names."""
        normalized: Dict[str, Dict[str, Any]] = {}
        for model_id, cfg in self.models.items():
            role = self._normalize_role(cfg.get("role", ""))
            normalized[model_id] = {
                **cfg,
                "role": role,
                "weight": float(cfg.get("weight", 0.0)),
            }
        return normalized

    def get_role_model_map(self) -> Dict[str, str]:
        """Return role -> model mapping used by the ensemble/debate system."""
        provider = settings.api.resolve_llm_provider()
        if provider == "codex":
            role_map = {
                "forecaster": "codex/gpt-5.4-mini",
                "news_analyst": "codex/gpt-5.4-mini",
                "bull_researcher": "codex/gpt-5.4-mini",
                "bear_researcher": "codex/gpt-5.4-mini",
                "risk_manager": "codex/gpt-5.4",
                "trader": "codex/gpt-5.4",
            }
        elif provider == "openai":
            role_map = {
                cfg["role"]: model_id
                for model_id, cfg in self.normalized_models().items()
                if cfg["role"] and cfg.get("provider") == "openai"
            }
            openai_defaults = {
                "forecaster": "openai/gpt-4.1",
                "news_analyst": "openai/o3",
                "bull_researcher": "openai/o3",
                "bear_researcher": "openai/gpt-4.1",
                "risk_manager": "openai/gpt-5.4",
                "trader": "openai/gpt-5.4",
            }
            role_map = {**openai_defaults, **role_map}
        else:
            role_map = {
                cfg["role"]: model_id
                for model_id, cfg in self.normalized_models().items()
                if cfg["role"]
            }

        role_map["trader"] = role_map.get("trader", self.trader_model)
        return role_map

    def get_role_weights(self) -> Dict[str, float]:
        """Return role -> weight mapping for weighted ensemble aggregation."""
        return {
            cfg["role"]: float(cfg.get("weight", 0.0))
            for cfg in self.normalized_models().values()
            if cfg["role"]
        }


def _get_rss_feeds() -> List[str]:
    """Return RSS feeds from the RSS_FEEDS env var or working defaults.

    The old Reuters endpoints (feeds.reuters.com) were discontinued and
    silently returned nothing, starving the sentiment pipeline. Defaults now
    cover general/business news plus sports and crypto, matching the live
    trading focus categories.
    """
    override = _get_csv_env_list("RSS_FEEDS")
    if override:
        return override
    return [
        "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://feeds.npr.org/1001/rss.xml",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://finance.yahoo.com/news/rssindex",
        "https://www.espn.com/espn/rss/news",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
    ]


@dataclass
class SentimentConfig:
    """News and sentiment analysis configuration."""
    enabled: bool = True
    rss_feeds: List[str] = field(default_factory=_get_rss_feeds)
    sentiment_model: str = field(default_factory=_get_default_sentiment_model)
    cache_ttl_minutes: int = 30
    max_articles_per_source: int = 10
    relevance_threshold: float = 0.3
    # Exponential recency half-life (hours) applied to news relevance ranking.
    # Live-trade markets resolve in hours, so a 3-day-old keyword match should
    # not outrank breaking news of equal overlap. A very large value disables
    # the decay (legacy overlap-only ordering). Dateless articles get a neutral
    # factor — never dropped — and the hard relevance gate stays on raw overlap.
    news_half_life_hours: float = field(
        default_factory=lambda: float(os.getenv("NEWS_HALF_LIFE_HOURS", "36.0"))
    )


@dataclass
class WeatherConfig:
    """
    Weather forecast model configuration.

    The weather pipeline (``src/data/weather_client.py`` +
    ``src/utils/weather_probability.py`` + ``src/data/weather_adapter.py``)
    turns free Open-Meteo ensemble forecasts and the official NWS point
    forecast into a deterministic P(bucket) for Kalshi temperature and
    precipitation contracts. These knobs control that model and how strongly
    its probability overrides the LLM estimate at the EV gate.
    """
    enabled: bool = field(default_factory=lambda: _get_bool_env("WEATHER_TRADING_ENABLED", True))

    # Open-Meteo ensemble models to pool (comma-separated env override).
    # gfs_seamless = 31 members, ecmwf_ifs025 = 51 members.
    ensemble_models: List[str] = field(
        default_factory=lambda: _get_csv_env_list("WEATHER_ENSEMBLE_MODELS")
        or ["gfs_seamless", "ecmwf_ifs025"]
    )

    # Weight of the deterministic weather-model probability when pooled with
    # the LLM fair probability before the EV gate (log-odds pooling). The
    # effective weight is scaled by the estimate's quality score.
    model_pool_weight: float = field(
        default_factory=lambda: float(os.getenv("WEATHER_MODEL_POOL_WEIGHT", "0.75"))
    )

    # Do not allow live weather entries further out than this many days —
    # ensemble skill degrades and the fee-adjusted edge is usually noise.
    max_lead_days: int = field(
        default_factory=lambda: int(os.getenv("WEATHER_MAX_LEAD_DAYS", "6"))
    )

    # Gaussian-kernel bandwidth model (deg F): sigma = max(floor, base +
    # per_day * lead_days), with `unverified_extra` added in quadrature when
    # the settlement station had to be geocoded instead of matched.
    sigma_floor_f: float = field(
        default_factory=lambda: float(os.getenv("WEATHER_SIGMA_FLOOR_F", "1.2"))
    )
    sigma_base_f: float = field(
        default_factory=lambda: float(os.getenv("WEATHER_SIGMA_BASE_F", "1.6"))
    )
    sigma_per_day_f: float = field(
        default_factory=lambda: float(os.getenv("WEATHER_SIGMA_PER_DAY_F", "0.5"))
    )
    unverified_station_extra_sigma_f: float = field(
        default_factory=lambda: float(os.getenv("WEATHER_UNVERIFIED_EXTRA_SIGMA_F", "1.5"))
    )

    # Kernel bandwidth for precipitation totals (inches).
    rain_sigma_in: float = field(
        default_factory=lambda: float(os.getenv("WEATHER_RAIN_SIGMA_IN", "0.08"))
    )
    snow_sigma_in: float = field(
        default_factory=lambda: float(os.getenv("WEATHER_SNOW_SIGMA_IN", "0.3"))
    )

    # Weight used to recenter the ensemble toward the official NWS point
    # forecast (settlement is an NWS product). 0 disables recentering.
    nws_blend_weight: float = field(
        default_factory=lambda: float(os.getenv("WEATHER_NWS_BLEND_WEIGHT", "0.35"))
    )

    # Margin (deg F) the observed running max/min must clear a bucket
    # boundary by before the model emits a hard 0/1 on same-day contracts.
    running_obs_margin_f: float = field(
        default_factory=lambda: float(os.getenv("WEATHER_RUNNING_OBS_MARGIN_F", "1.5"))
    )

    # Climatology fallback depth.
    climatology_years: int = field(
        default_factory=lambda: int(os.getenv("WEATHER_CLIMATOLOGY_YEARS", "10"))
    )

    # Minimum ensemble members before an estimate is allowed to override
    # the LLM at full strength (quality is scaled down below this).
    min_ensemble_members: int = field(
        default_factory=lambda: int(os.getenv("WEATHER_MIN_ENSEMBLE_MEMBERS", "8"))
    )

    # Minimum quality score (0-1) an estimate needs before it is pooled
    # into the live-trade EV gate at all.
    min_quality_to_pool: float = field(
        default_factory=lambda: float(os.getenv("WEATHER_MIN_QUALITY_TO_POOL", "0.35"))
    )

    # Allow geocoding fallback for cities outside the curated registry.
    allow_geocode_fallback: bool = field(
        default_factory=lambda: _get_bool_env("WEATHER_ALLOW_GEOCODE_FALLBACK", True)
    )

    # HTTP behaviour for the weather data client (ensemble payloads are
    # bigger than the 3s sports/crypto budget allows).
    request_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("WEATHER_REQUEST_TIMEOUT_SECONDS", "8.0"))
    )

    # ------------------------------------------------------------------
    # Weather scan job (`python cli.py weather-scan`): sweeps every open
    # weather event with the deterministic model — no LLM calls — and
    # surfaces/executes fee-positive divergences.
    # ------------------------------------------------------------------
    # Explicit series allowlist (CSV). Empty = auto-derive from the station
    # registry (KXHIGH/KXLOW x station codes).
    scan_series: List[str] = field(
        default_factory=lambda: _get_csv_env_list("WEATHER_SCAN_SERIES")
    )
    # Net edge after fees a bucket must clear before it becomes a scan
    # candidate. Slightly above the live-trade gate because the scanner has
    # no LLM second opinion.
    scan_min_net_edge: float = field(
        default_factory=lambda: float(os.getenv("WEATHER_SCAN_MIN_NET_EDGE", "0.03"))
    )
    # Minimum estimate quality (0-1) for scan candidates; stricter than the
    # pooling floor used when an LLM estimate is also present.
    scan_min_quality: float = field(
        default_factory=lambda: float(os.getenv("WEATHER_SCAN_MIN_QUALITY", "0.5"))
    )
    # Cap on events scanned per run (nearest expiries first) and on
    # positions opened per run when trading is enabled.
    scan_max_events: int = field(
        default_factory=lambda: int(os.getenv("WEATHER_SCAN_MAX_EVENTS", "16"))
    )
    scan_max_positions: int = field(
        default_factory=lambda: int(os.getenv("WEATHER_SCAN_MAX_POSITIONS", "5"))
    )
    # Execute scan candidates as paper positions when enabled. Live
    # execution additionally requires LIVE_TRADING_ENABLED and
    # WEATHER_SCAN_LIVE so real-money weather trading stays double opt-in.
    scan_trade_enabled: bool = field(
        default_factory=lambda: _get_bool_env("WEATHER_SCAN_TRADE_ENABLED", True)
    )
    scan_live: bool = field(
        default_factory=lambda: _get_bool_env("WEATHER_SCAN_LIVE")
    )


@dataclass
class SportsConfig:
    """
    Sportsbook-odds gate anchor configuration.

    ``src/data/sports_adapter.py`` already converts ESPN moneylines into
    de-vigged implied win probabilities — the single sharpest public prior for
    game-winner markets, and game-winner sports (NCAAB) is the bot's only
    proven-profitable niche. This config controls pooling that prior into the
    live-trade EV gate (``fair_yes_for_gate``) exactly the way the weather
    deterministic forecast is pooled, so the consensus anchors every sports
    trade deterministically rather than only reaching the LLM as prose.

    Ships default-on but at a deliberately conservative pool weight (Kalshi
    title->team mapping is fuzzier than the per-ticker weather mapping) with a
    kill switch. In-game pooling is disabled by default: ESPN's scoreboard
    odds block is a pre-game line that goes stale once the game tips off.
    """

    enabled: bool = field(
        default_factory=lambda: _get_bool_env("SPORTS_ODDS_GATE_ENABLED", True)
    )
    # Weight on the de-vigged sportsbook probability when pooled with the LLM
    # fair probability in log-odds space. Effective weight is scaled by the
    # estimate's quality. Conservative default vs weather's 0.75.
    model_pool_weight: float = field(
        default_factory=lambda: float(os.getenv("SPORTS_MODEL_POOL_WEIGHT", "0.55"))
    )
    # Minimum quality (0-1) before the sportsbook prior is pooled at all.
    min_quality_to_pool: float = field(
        default_factory=lambda: float(os.getenv("SPORTS_MIN_QUALITY_TO_POOL", "0.5"))
    )
    # Quality ceiling for in-game events. ESPN scoreboard moneylines are the
    # pre-game consensus and do not track the live score, so by default an
    # in-game market caps quality at 0 (never pooled). Raise only if a live
    # odds feed is wired in.
    in_game_max_quality: float = field(
        default_factory=lambda: float(os.getenv("SPORTS_IN_GAME_MAX_QUALITY", "0.0"))
    )


@dataclass
class AutoConfig:
    """
    Beast Auto V1 — unattended supervisor configuration.

    This governs only the autonomous loop wrapper (src/auto/runner.py). It
    does not change any decision, sizing, or risk-gate logic — those remain
    exactly as configured elsewhere. Auto reuses the existing live-trade
    decision loop and execution path; this config controls cadence, the
    fail-closed kill switch, and consecutive-failure shutdown behavior.
    """

    # Seconds to sleep between Auto decision-loop cycles.
    cadence_seconds: int = field(
        default_factory=lambda: int(os.getenv("AUTO_CADENCE_SECONDS", "60"))
    )
    # Consecutive cycle failures before Auto halts itself fail-closed.
    max_consecutive_failures: int = field(
        default_factory=lambda: int(os.getenv("AUTO_MAX_CONSECUTIVE_FAILURES", "3"))
    )
    # Path to the fail-closed kill-switch state file. If this file cannot be
    # read (missing is fine and means "not killed"; corrupt/unreadable is
    # NOT fine and is treated as killed), Auto blocks new buys.
    kill_switch_path: str = field(
        default_factory=lambda: os.getenv(
            "AUTO_KILL_SWITCH_PATH", "data/auto_kill_switch.json"
        )
    )
    # Strategy label Auto checks against the existing strategy_halts table
    # before running a cycle. Matches the label the live-trade loop already
    # uses so Auto respects halts raised by the existing pipeline.
    halt_strategy_label: str = field(
        default_factory=lambda: os.getenv("AUTO_HALT_STRATEGY_LABEL", "live_trade")
    )


# Trading strategy configuration — DISCIPLINED DEFAULTS (sane risk management)
# Beast mode is still available via --beast flag, but NOT the default.
# Discipline defaults based on live prediction market trading experience.
# NCAAB NO-side: 74% WR, +10% ROI — ONLY profitable category.
# Economic trades: -70% ROI, 78% of all losses.
@dataclass
class TradingConfig:
    """Trading strategy configuration."""
    # Position sizing and risk management — DISCIPLINED DEFAULTS
    max_position_size_pct: float = 3.0  # SANE: 3% per position (was 5% "beast mode")
    max_daily_loss_pct: float = 10.0    # SANE: 10% daily loss limit (was 15%)
    max_positions: int = 10              # SANE: 10 concurrent positions (was 15)
    min_balance: float = 100.0          # SANE: $100 minimum balance (was $50)
    
    # Market filtering criteria — DISCIPLINED
    min_volume: float = 500.0           # SANE: Higher volume requirement (was 200 beast mode)
    max_time_to_expiry_days: int = 14   # SANE: Shorter timeframes (was 30)
    
    # AI decision making — DATA-DRIVEN THRESHOLDS  
    min_confidence_to_trade: float = 0.45   # LOOSENED: 45% confidence minimum (was 60%, approved 2026-03-29)
                                           # Based on analysis: 65% was too conservative, bot finding 0 eligible markets
                                           # NCAAB NO-side showed 74% WR at +10% ROI, suggesting value at lower thresholds
    
    # Category-specific confidence adjustments (applied as multipliers to base threshold)
    category_confidence_adjustments: Dict[str, float] = field(default_factory=lambda: {
        "sports": 0.90,      # Sports showed best performance (NCAAB 74% WR), lower threshold
        "economics": 1.15,   # Economics showed -70% ROI, higher threshold required
        "politics": 1.05,    # Slight increase for political volatility
        "default": 1.0       # Base multiplier for other categories
    })

    # Category-specific multipliers on the live-trade NET-EDGE floor. Multiplies
    # ``live_trade_min_net_edge`` per category before the EV gate, so proven-weak
    # buckets must clear a higher edge bar and capital concentrates in proven
    # ones. Defaults are >= 1.0 by design: a sub-1.0 value LOWERS the floor and
    # could approve trades the flat floor blocks, so any loosening is an explicit
    # operator opt-in. Unlisted categories fall back to "default" (1.0 = no-op).
    category_min_net_edge_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "economics": 1.5,    # -70% ROI historically: demand much more edge
        "politics": 1.25,    # higher variance, weaker realized record
        "default": 1.0,      # no change for everything else (incl. sports)
    })

    # Event-level concentration cap (fraction of portfolio). The per-category
    # sector cap buckets every same-day NCAAB NO position into one "sports"
    # bucket; this caps exposure sharing a single Kalshi event/series root so an
    # upset-heavy slate cannot cluster correlated losses through one event. 1.0
    # disables the extra check (sector cap still applies).
    max_event_concentration_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_EVENT_CONCENTRATION_PCT", "0.12"))
    )

    # Total-portfolio usage cap (dry-powder floor) for the live/weather paths.
    # 1.0 = disabled (full deployment allowed). Set e.g. 0.90 to keep ~10% cash
    # free for the next high-edge opportunity. Enforced in PortfolioEnforcer.
    max_portfolio_usage_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_PORTFOLIO_USAGE_PCT", "1.0"))
    )

    # Route decide.py's main BUY path through the canonical evaluate_trade_intent
    # gate (the same one the live-trade loop and decide's high-confidence path
    # use) instead of the legacy EdgeFilter gate. OPT-IN (default off): it is a
    # decision-core behavior change — shadow/backtest before enabling live. When
    # on, it folds in the market-prior-calibrated anchor, category-aware
    # calibration slope, the settlement meta-model, the category net-edge
    # multiplier, and the coin-flip/weak-category surcharge.
    decide_use_canonical_gate: bool = field(
        default_factory=lambda: _get_bool_env("DECIDE_USE_CANONICAL_GATE", False)
    )

    # Cross-market (Polymarket) price pooled into the live-trade EV gate as a
    # second external prior. OPT-IN (default off): the LLM already sees the
    # Polymarket price as prose (partial double-count), the text-similarity
    # ticker mapping can be imprecise, and Polymarket's liquid coverage skews
    # to politics/crypto (weak overlap with the proven NCAAB niche). When on,
    # a fresh, liquid, high-confidence match is log-odds-pooled at a small
    # confidence-scaled weight; agreement shrinks claimed edge, divergence
    # pulls fair toward consensus. Fails closed on low confidence/volume/stale.
    cross_market_pool_enabled: bool = field(
        default_factory=lambda: _get_bool_env("CROSS_MARKET_POOL_ENABLED", False)
    )
    cross_market_min_mapping_confidence: float = field(
        default_factory=lambda: float(os.getenv("CROSS_MARKET_MIN_MAPPING_CONFIDENCE", "0.65"))
    )
    cross_market_min_volume_usd: float = field(
        default_factory=lambda: float(os.getenv("CROSS_MARKET_MIN_VOLUME_USD", "2000"))
    )
    cross_market_pool_weight_base: float = field(
        default_factory=lambda: float(os.getenv("CROSS_MARKET_POOL_WEIGHT_BASE", "0.20"))
    )
    cross_market_pool_weight_cap: float = field(
        default_factory=lambda: float(os.getenv("CROSS_MARKET_POOL_WEIGHT_CAP", "0.25"))
    )
    cross_market_max_age_seconds: float = field(
        default_factory=lambda: float(os.getenv("CROSS_MARKET_MAX_AGE_SECONDS", "900"))
    )
    
    scan_interval_seconds: int = 60      # SANE: 60-second scan interval (was 30)
    
    # AI model configuration
    primary_model: str = field(default_factory=_get_default_primary_model)
    fallback_model: str = field(default_factory=_get_default_fallback_model)
    ai_temperature: float = 0  # Lower temperature for more consistent JSON output
    ai_max_tokens: int = 8000    # Reasonable limit for reasoning-oriented models
    
    # Position sizing (LEGACY - now using Kelly-primary approach)
    default_position_size: float = 3.0  # REDUCED: Now using Kelly Criterion as primary method (was 5%, now 3%)
    position_size_multiplier: float = 1.0  # Multiplier for AI confidence
    
    # Kelly Criterion settings (PRIMARY position sizing method) — DISCIPLINED
    use_kelly_criterion: bool = True        # Use Kelly Criterion for position sizing (PRIMARY METHOD)
    kelly_fraction: float = 0.25            # SANE: Quarter-Kelly (was 0.75 beast mode — gambling)
    max_single_position: float = 0.03       # SANE: 3% max position cap (was 0.05 beast mode)
    
    # Live trading mode control
    live_trading_enabled: bool = field(default_factory=lambda: _get_bool_env("LIVE_TRADING_ENABLED"))
    paper_trading_mode: bool = field(default_factory=lambda: not _get_bool_env("LIVE_TRADING_ENABLED"))
    shadow_mode_enabled: bool = field(default_factory=lambda: _get_bool_env("SHADOW_MODE_ENABLED"))
    
    # Trading frequency - MORE FREQUENT
    market_scan_interval: int = 30          # DECREASED: Scan every 30 seconds (was 60)
    position_check_interval: int = 15       # DECREASED: Check positions every 15 seconds (was 30)
    max_trades_per_hour: int = 20           # INCREASED: Allow more trades per hour (was 10, now 20)
    run_interval_minutes: int = 10          # DECREASED: Run more frequently (was 15, now 10)
    num_processor_workers: int = 5      # Number of concurrent market processor workers

    # Unified strategy capital allocation
    market_making_allocation: float = field(
        default_factory=lambda: float(os.getenv("MARKET_MAKING_ALLOCATION", "0.40"))
    )
    directional_allocation: float = field(
        default_factory=lambda: float(os.getenv("DIRECTIONAL_ALLOCATION", "0.50"))
    )
    arbitrage_allocation: float = field(
        default_factory=lambda: float(os.getenv("ARBITRAGE_ALLOCATION", "0.10"))
    )
    
    # Market selection preferences
    preferred_categories: List[str] = field(default_factory=lambda: _get_csv_env_list("PREFERRED_CATEGORIES"))
    excluded_categories: List[str] = field(default_factory=lambda: _get_csv_env_list("EXCLUDED_CATEGORIES"))
    prefer_live_wagering: bool = field(default_factory=lambda: _get_bool_env("PREFER_LIVE_WAGERING"))
    live_wagering_max_hours_to_expiry: int = field(
        default_factory=lambda: int(os.getenv("LIVE_WAGERING_MAX_HOURS_TO_EXPIRY", "12"))
    )
    
    # High-confidence, near-expiry strategy
    enable_high_confidence_strategy: bool = True
    high_confidence_threshold: float = 0.95  # LLM confidence needed
    high_confidence_market_odds: float = 0.90 # Market price to look for
    high_confidence_expiry_hours: int = 24   # Max hours until expiry

    # AI trading criteria - MORE PERMISSIVE
    max_analysis_cost_per_decision: float = 0.15  # INCREASED: Allow higher cost per decision (was 0.10, now 0.15)
    min_confidence_threshold: float = 0.45  # DECREASED: Lower confidence threshold (was 0.55, now 0.45)

    # Quick flip strategy
    enable_quick_flip: bool = field(default_factory=lambda: _get_bool_env("ENABLE_QUICK_FLIP"))
    enable_live_quick_flip: bool = field(
        default_factory=lambda: _get_bool_env("ENABLE_LIVE_QUICK_FLIP")
    )
    quick_flip_disable_ai: bool = field(
        default_factory=lambda: _get_bool_env("QUICK_FLIP_DISABLE_AI")
    )
    quick_flip_allocation: float = field(
        default_factory=lambda: float(os.getenv("QUICK_FLIP_ALLOCATION", "0.00"))
    )
    quick_flip_max_market_checks: int = field(
        default_factory=lambda: int(os.getenv("QUICK_FLIP_MAX_MARKET_CHECKS", "100"))
    )
    quick_flip_target_opportunity_buffer: int = field(
        default_factory=lambda: int(os.getenv("QUICK_FLIP_TARGET_OPPORTUNITY_BUFFER", "6"))
    )
    quick_flip_min_entry_price: float = field(
        default_factory=lambda: float(os.getenv("QUICK_FLIP_MIN_ENTRY_PRICE", "0.01"))
    )
    quick_flip_max_entry_price: float = field(
        default_factory=lambda: float(os.getenv("QUICK_FLIP_MAX_ENTRY_PRICE", "0.20"))
    )
    quick_flip_min_profit_margin: float = field(
        default_factory=lambda: float(os.getenv("QUICK_FLIP_MIN_PROFIT_MARGIN", "0.10"))
    )
    quick_flip_max_position_size: int = field(
        default_factory=lambda: int(os.getenv("QUICK_FLIP_MAX_POSITION_SIZE", "100"))
    )
    quick_flip_max_concurrent_positions: int = field(
        default_factory=lambda: int(os.getenv("QUICK_FLIP_MAX_CONCURRENT_POSITIONS", "50"))
    )
    quick_flip_capital_per_trade: float = field(
        default_factory=lambda: float(os.getenv("QUICK_FLIP_CAPITAL_PER_TRADE", "50.0"))
    )
    quick_flip_daily_loss_budget_pct: float = field(
        default_factory=lambda: float(os.getenv("QUICK_FLIP_DAILY_LOSS_BUDGET_PCT", "0.05"))
    )
    quick_flip_max_open_positions: int = field(
        default_factory=lambda: int(os.getenv("QUICK_FLIP_MAX_OPEN_POSITIONS", "10"))
    )
    quick_flip_max_trades_per_hour: int = field(
        default_factory=lambda: int(os.getenv("QUICK_FLIP_MAX_TRADES_PER_HOUR", "60"))
    )
    quick_flip_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("QUICK_FLIP_CONFIDENCE_THRESHOLD", "0.6"))
    )
    quick_flip_max_hold_minutes: int = field(
        default_factory=lambda: int(os.getenv("QUICK_FLIP_MAX_HOLD_MINUTES", "30"))
    )
    quick_flip_min_market_volume: int = field(
        default_factory=lambda: int(os.getenv("QUICK_FLIP_MIN_MARKET_VOLUME", "1000"))
    )
    quick_flip_max_hours_to_expiry: int = field(
        default_factory=lambda: int(os.getenv("QUICK_FLIP_MAX_HOURS_TO_EXPIRY", "72"))
    )
    quick_flip_max_bid_ask_spread: float = field(
        default_factory=lambda: float(os.getenv("QUICK_FLIP_MAX_BID_ASK_SPREAD", "0.03"))
    )
    quick_flip_min_top_of_book_size: int = field(
        default_factory=lambda: int(os.getenv("QUICK_FLIP_MIN_TOP_OF_BOOK_SIZE", "10"))
    )
    quick_flip_min_net_profit: float = field(
        default_factory=lambda: float(os.getenv("QUICK_FLIP_MIN_NET_PROFIT", "0.10"))
    )
    quick_flip_min_net_roi: float = field(
        default_factory=lambda: float(os.getenv("QUICK_FLIP_MIN_NET_ROI", "0.03"))
    )
    quick_flip_recent_trade_window_seconds: int = field(
        default_factory=lambda: int(os.getenv("QUICK_FLIP_RECENT_TRADE_WINDOW_SECONDS", "3600"))
    )
    quick_flip_min_recent_trade_count: int = field(
        default_factory=lambda: int(os.getenv("QUICK_FLIP_MIN_RECENT_TRADE_COUNT", "5"))
    )
    quick_flip_max_target_vs_recent_trade_gap: float = field(
        default_factory=lambda: float(os.getenv("QUICK_FLIP_MAX_TARGET_VS_RECENT_TRADE_GAP", "0.02"))
    )
    quick_flip_min_recent_range_ticks: int = field(
        default_factory=lambda: int(os.getenv("QUICK_FLIP_MIN_RECENT_RANGE_TICKS", "2"))
    )
    quick_flip_min_recent_price_position: float = field(
        default_factory=lambda: float(os.getenv("QUICK_FLIP_MIN_RECENT_PRICE_POSITION", "0.4"))
    )
    quick_flip_max_entry_vs_recent_last_gap: float = field(
        default_factory=lambda: float(os.getenv("QUICK_FLIP_MAX_ENTRY_VS_RECENT_LAST_GAP", "0.02"))
    )
    quick_flip_maker_entry_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("QUICK_FLIP_MAKER_ENTRY_TIMEOUT_SECONDS", "180"))
    )
    quick_flip_maker_entry_poll_seconds: int = field(
        default_factory=lambda: int(os.getenv("QUICK_FLIP_MAKER_ENTRY_POLL_SECONDS", "5"))
    )
    quick_flip_maker_entry_reprice_seconds: int = field(
        default_factory=lambda: int(os.getenv("QUICK_FLIP_MAKER_ENTRY_REPRICE_SECONDS", "30"))
    )
    quick_flip_dynamic_exit_reprice_seconds: int = field(
        default_factory=lambda: int(os.getenv("QUICK_FLIP_DYNAMIC_EXIT_REPRICE_SECONDS", "60"))
    )
    quick_flip_stop_loss_pct: float = field(
        default_factory=lambda: float(os.getenv("QUICK_FLIP_STOP_LOSS_PCT", "0.08"))
    )
    # Quick-flip statistical gates. The EV gate converts each candidate's
    # reward (net profit at target) and risk (stop-loss distance plus taker
    # exit fees) into the minimum win probability that makes the scalp
    # positive expected value, and requires the movement confidence to clear
    # it (plus an optional margin). The freshness guard rejects candidates
    # whose most recent public trade is older than the configured age —
    # momentum heuristics on a stale tape are noise.
    quick_flip_ev_gate_enabled: bool = field(
        default_factory=lambda: _get_bool_env("QUICK_FLIP_EV_GATE_ENABLED", True)
    )
    quick_flip_ev_confidence_margin: float = field(
        default_factory=lambda: float(os.getenv("QUICK_FLIP_EV_CONFIDENCE_MARGIN", "0.0"))
    )
    quick_flip_max_last_trade_age_seconds: int = field(
        default_factory=lambda: int(os.getenv("QUICK_FLIP_MAX_LAST_TRADE_AGE_SECONDS", "900"))
    )
    live_trade_daily_loss_budget_pct: float = field(
        default_factory=lambda: float(os.getenv("LIVE_TRADE_DAILY_LOSS_BUDGET_PCT", "0.05"))
    )
    live_trade_max_open_positions: int = field(
        default_factory=lambda: int(os.getenv("LIVE_TRADE_MAX_OPEN_POSITIONS", "5"))
    )
    live_trade_max_trades_per_hour: int = field(
        default_factory=lambda: int(os.getenv("LIVE_TRADE_MAX_TRADES_PER_HOUR", "20"))
    )

    # Deterministic fee-aware EV gate for the live-trade loop. The final
    # intent's fair probability is calibration-shrunk, blended with the live
    # market price, and the trade must clear this many dollars of net edge
    # per contract after estimated fees before execution is allowed.
    live_trade_min_net_edge: float = field(
        default_factory=lambda: float(os.getenv("LIVE_TRADE_MIN_NET_EDGE", "0.02"))
    )
    live_trade_min_confidence: float = field(
        default_factory=lambda: float(os.getenv("LIVE_TRADE_MIN_CONFIDENCE", "0.55"))
    )
    # Apply settlement-calibration shrinkage to model probabilities before
    # EV gating (slope from realized outcomes; 1.0 = no shrink).
    calibration_shrink_enabled: bool = field(
        default_factory=lambda: _get_bool_env("CALIBRATION_SHRINK_ENABLED", True)
    )

    # Microstructure guards for the live-trade EV gate. Entries are refused
    # when the bid-ask spread exceeds the cap (midpoint blending is
    # unreliable across wide spreads) or when the top of book on the chosen
    # side rests fewer contracts than the minimum (exits would pay the
    # spread again). Set either to 0 to disable that guard.
    live_trade_max_spread_cents: float = field(
        default_factory=lambda: float(os.getenv("LIVE_TRADE_MAX_SPREAD_CENTS", "6"))
    )
    # Default matches the quick-flip depth guard (10 contracts at top of book).
    live_trade_min_top_depth_contracts: int = field(
        default_factory=lambda: int(os.getenv("LIVE_TRADE_MIN_TOP_DEPTH_CONTRACTS", "10"))
    )

    # Kelly sizing cap for the standard live-trade execution path. The LLM's
    # requested position_size_pct is clamped to the fractional-Kelly bankroll
    # fraction implied by the EV gate's blended win probability, so position
    # size can never exceed what the measured edge statistically supports.
    live_trade_kelly_sizing_enabled: bool = field(
        default_factory=lambda: _get_bool_env("LIVE_TRADE_KELLY_SIZING_ENABLED", True)
    )
    live_trade_kelly_multiplier: float = field(
        default_factory=lambda: float(os.getenv("LIVE_TRADE_KELLY_MULTIPLIER", "0.25"))
    )

    # Market-prior calibration. When fitted models exist (trained on settled
    # market snapshots via `python cli.py fit-market-prior`), the EV gate's
    # market anchor uses the calibrated probability instead of the raw mid,
    # correcting systematic price biases (favorite-longshot). Fails closed to
    # the raw mid whenever no validated model is available.
    market_prior_calibration_enabled: bool = field(
        default_factory=lambda: _get_bool_env("MARKET_PRIOR_CALIBRATION_ENABLED", True)
    )

    # Per-role ensemble skill weighting. Each debate member's probability is
    # persisted with executed decisions; at settlement the realized Brier
    # score per role accumulates (tagged with the market's category), and
    # pooling weights are scaled by inverse relative Brier — per category
    # when a role has enough settled observations there, shrunk toward its
    # global multiplier otherwise, and toward 1.0 until any evidence exists.
    model_skill_weighting_enabled: bool = field(
        default_factory=lambda: _get_bool_env("MODEL_SKILL_WEIGHTING_ENABLED", True)
    )

    # Settlement-result backfill. Periodically fetches final YES/NO results
    # for expired markets the snapshot collector observed, labelling the
    # historical dataset the market-prior calibration trains on.
    result_backfill_enabled: bool = field(
        default_factory=lambda: _get_bool_env("RESULT_BACKFILL_ENABLED", True)
    )
    result_backfill_interval_minutes: int = field(
        default_factory=lambda: int(os.getenv("RESULT_BACKFILL_INTERVAL_MINUTES", "60"))
    )
    result_backfill_max_tickers_per_run: int = field(
        default_factory=lambda: int(os.getenv("RESULT_BACKFILL_MAX_TICKERS_PER_RUN", "400"))
    )

    # Quick-flip momentum confirmation: require the resting bid depth to be
    # at least this fraction of the resting ask depth at top of book before
    # entering. A book stacked with sellers above and no buyers below is a
    # bad place to buy a momentum continuation. 0 disables the guard.
    quick_flip_min_bid_ask_size_ratio: float = field(
        default_factory=lambda: float(os.getenv("QUICK_FLIP_MIN_BID_ASK_SIZE_RATIO", "0.5"))
    )

    # Fee-avoidance: hold near-certain winners to settlement (fee-free $1
    # payout) instead of selling early and paying an exit fee plus spread.
    # Stop-losses still apply, so a genuine reversal is still cut.
    hold_winners_to_settlement: bool = field(
        default_factory=lambda: _get_bool_env("HOLD_WINNERS_TO_SETTLEMENT", True)
    )
    hold_winners_to_settlement_price: float = field(
        default_factory=lambda: float(os.getenv("HOLD_WINNERS_TO_SETTLEMENT_PRICE", "0.95"))
    )

    # ML meta-model: a statistical model (numpy logistic regression, or a
    # random forest when scikit-learn is installed and enough settlements
    # exist) trained on realized settlement outcomes. Its probability is
    # blended with the LLM ensemble's in log-odds space; the blend weight
    # ramps with training-set size and is capped below so the LLM signal is
    # never fully overridden.
    ml_meta_model_enabled: bool = field(
        default_factory=lambda: _get_bool_env("ML_META_MODEL_ENABLED", True)
    )
    ml_meta_model_max_blend_weight: float = field(
        default_factory=lambda: float(os.getenv("ML_META_MODEL_MAX_BLEND_WEIGHT", "0.35"))
    )

    # Category exploration. Unproven categories (fewer than 5 settled
    # trades) receive a small exploration score instead of a hard block so
    # they can accumulate the evidence the category scorer needs. Always
    # used in paper/shadow runs (unless disabled); live runs additionally
    # require CATEGORY_EXPLORATION_LIVE.
    category_exploration_enabled: bool = field(
        default_factory=lambda: _get_bool_env("CATEGORY_EXPLORATION_ENABLED", True)
    )
    category_exploration_live: bool = field(
        default_factory=lambda: _get_bool_env("CATEGORY_EXPLORATION_LIVE")
    )
    category_exploration_score: float = field(
        default_factory=lambda: float(os.getenv("CATEGORY_EXPLORATION_SCORE", "35"))
    )

    # Shadow drift auto-pause (W4 follow-up). Default OFF so existing runtime
    # behavior is preserved unless explicitly opted in via env.
    shadow_drift_auto_pause_enabled: bool = field(
        default_factory=lambda: os.getenv("SHADOW_DRIFT_AUTO_PAUSE_ENABLED", "false").lower() in ("1", "true", "yes")
    )
    shadow_drift_max_avg_abs_entry_delta_cents: float = field(
        default_factory=lambda: float(os.getenv("SHADOW_DRIFT_MAX_AVG_ABS_ENTRY_DELTA_CENTS", "2.0"))
    )
    shadow_drift_max_total_entry_cost_delta_usd: float = field(
        default_factory=lambda: float(os.getenv("SHADOW_DRIFT_MAX_TOTAL_ENTRY_COST_DELTA_USD", "25.0"))
    )
    shadow_drift_min_matched_entries: int = field(
        default_factory=lambda: int(os.getenv("SHADOW_DRIFT_MIN_MATCHED_ENTRIES", "5"))
    )

    # Cost control and market analysis frequency - MORE PERMISSIVE
    daily_ai_budget: float = 10.0  # INCREASED: Higher daily budget (was 5.0, now 10.0)
    max_ai_cost_per_decision: float = 0.08  # INCREASED: Higher per-decision cost (was 0.05, now 0.08)
    analysis_cooldown_hours: int = 3  # DECREASED: Shorter cooldown (was 6, now 3)
    max_analyses_per_market_per_day: int = 4  # INCREASED: More analyses per day (was 2, now 4)
    
    # Daily AI spending limits - SAFETY CONTROLS
    # Default is $10/day — conservative limit to prevent runaway API spend.
    # Raise via DAILY_AI_COST_LIMIT env var or by editing this value directly.
    # e.g. export DAILY_AI_COST_LIMIT=25  (for more aggressive scanning)
    daily_ai_cost_limit: float = field(default_factory=lambda: float(os.getenv("DAILY_AI_COST_LIMIT", "10.0")))
    enable_daily_cost_limiting: bool = True  # Enable daily cost limits
    sleep_when_limit_reached: bool = True  # Sleep until next day when limit reached

    # Enhanced market filtering to reduce analyses - MORE PERMISSIVE
    min_volume_for_ai_analysis: float = 200.0  # DECREASED: Much lower threshold (was 500, now 200)
    exclude_low_liquidity_categories: List[str] = field(default_factory=lambda: [
        # REMOVED weather and entertainment - trade all categories
    ])


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "DEBUG"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "logs/trading_system.log"
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    max_log_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


# BEAST MODE UNIFIED TRADING SYSTEM CONFIGURATION 🚀
# These settings control the advanced multi-strategy trading system

# === CAPITAL ALLOCATION ACROSS STRATEGIES ===
# Allocate capital across different trading approaches
market_making_allocation: float = 0.40  # 40% for market making (spread profits)
directional_allocation: float = 0.50    # 50% for directional trading (AI predictions) 
arbitrage_allocation: float = 0.10      # 10% for arbitrage opportunities

  # === PORTFOLIO OPTIMIZATION SETTINGS ===
# Kelly Criterion is now the PRIMARY position sizing method (moved to TradingConfig)
# total_capital: DYNAMICALLY FETCHED from Kalshi balance - never hardcoded!
use_risk_parity: bool = True            # Equal risk allocation vs equal capital
rebalance_hours: int = 6                # Rebalance portfolio every 6 hours
min_position_size: float = 5.0          # Minimum position size ($5 vs $10)
max_opportunities_per_batch: int = 50   # Limit opportunities to prevent optimization issues

# === RISK MANAGEMENT LIMITS ===
# Portfolio-level risk constraints — DISCIPLINED DEFAULTS
# Conservative defaults based on live trading experience. Beast mode available via CLI flag.
max_volatility: float = 0.40            # SANE: 40% volatility max (was 80%)
max_correlation: float = 0.70           # SANE: 70% correlation max (was 95%)
max_drawdown: float = 0.15              # SANE: 15% drawdown limit (was 50% — suicidal)
max_sector_exposure: float = 0.30       # SANE: 30% sector concentration (was 90%)

# === PERFORMANCE TARGETS ===
# System performance objectives - MORE AGGRESSIVE FOR MORE TRADES
target_sharpe: float = 0.3              # DECREASED: Lower Sharpe requirement (was 0.5, now 0.3)
target_return: float = 0.15             # INCREASED: Higher return target (was 0.10, now 0.15)
min_trade_edge: float = 0.08           # DECREASED: Lower edge requirement (was 0.15, now 8%)
min_confidence_for_large_size: float = 0.50  # DECREASED: Lower confidence requirement (was 0.65, now 50%)

# === DYNAMIC EXIT STRATEGIES ===
# Enhanced exit strategy settings - MORE AGGRESSIVE
use_dynamic_exits: bool = True
profit_threshold: float = 0.20          # DECREASED: Take profits sooner (was 0.25, now 0.20)
loss_threshold: float = 0.15            # INCREASED: Allow larger losses (was 0.10, now 0.15)
confidence_decay_threshold: float = 0.25  # INCREASED: Allow more confidence decay (was 0.20, now 0.25)
max_hold_time_hours: int = 240          # INCREASED: Hold longer (was 168, now 240 hours = 10 days)
volatility_adjustment: bool = True      # Adjust exits based on volatility

# === MARKET MAKING STRATEGY ===
# Settings for limit order market making - MORE AGGRESSIVE
enable_market_making: bool = True       # Enable market making strategy
min_spread_for_making: float = 0.01     # DECREASED: Accept smaller spreads (was 0.02, now 1¢)
max_inventory_risk: float = 0.15        # INCREASED: Allow higher inventory risk (was 0.10, now 15%)
order_refresh_minutes: int = 15         # Refresh orders every 15 minutes
max_orders_per_market: int = 4          # Maximum orders per market (2 each side)

# === MARKET SELECTION (ENHANCED FOR MORE OPPORTUNITIES) ===
# Removed time restrictions - trade ANY deadline with dynamic exits!
# max_time_to_expiry_days: REMOVED      # No longer used - trade any timeline!
min_volume_for_analysis: float = 200.0  # DECREASED: Much lower minimum volume (was 1000, now 200)
min_volume_for_market_making: float = 500.0  # DECREASED: Lower volume for market making (was 2000, now 500)
min_price_movement: float = 0.02        # DECREASED: Lower minimum range (was 0.05, now 2¢)
max_bid_ask_spread: float = 0.15        # INCREASED: Allow wider spreads (was 0.10, now 15¢)
min_confidence_long_term: float = 0.45  # DECREASED: Lower confidence for distant expiries (was 0.65, now 45%)

# === COST OPTIMIZATION (MORE GENEROUS) ===
# Enhanced cost controls for the beast mode system
daily_ai_budget: float = 15.0           # INCREASED: Higher budget for more opportunities (was 10.0, now 15.0)
max_ai_cost_per_decision: float = 0.12  # INCREASED: Higher per-decision limit (was 0.08, now 0.12)
analysis_cooldown_hours: int = 2        # DECREASED: Much shorter cooldown (was 4, now 2)
max_analyses_per_market_per_day: int = 6  # INCREASED: More analyses per day (was 3, now 6)
skip_news_for_low_volume: bool = True   # Skip expensive searches for low volume
news_search_volume_threshold: float = 1000.0  # News threshold

# === SYSTEM BEHAVIOR ===
# Overall system behavior settings
beast_mode_enabled: bool = True         # Enable the unified advanced system
fallback_to_legacy: bool = True         # Fallback to legacy system if needed
log_level: str = "INFO"                 # Logging level
performance_monitoring: bool = True     # Enable performance monitoring

# === ADVANCED FEATURES ===
# Cutting-edge features for maximum performance
cross_market_arbitrage: bool = False    # Enable when arbitrage module ready
multi_model_ensemble: bool = True       # Multi-model ensemble decisions (ENABLED)
sentiment_analysis: bool = True         # News sentiment analysis (ENABLED)
websocket_streaming: bool = True        # WebSocket real-time data (ENABLED)
options_strategies: bool = False        # Complex options strategies (future)
algorithmic_execution: bool = False     # Smart order execution (future)


@dataclass
class Settings:
    """Main settings class combining all configuration."""
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    sports: SportsConfig = field(default_factory=SportsConfig)
    auto: AutoConfig = field(default_factory=AutoConfig)

    def validate(self) -> bool:
        """Validate configuration settings."""
        if self.api.kalshi_env not in {"prod", "demo"}:
            raise ValueError("KALSHI_ENV must be 'prod' or 'demo'")

        if self.api.llm_provider not in {"auto", "openai", "openrouter", "codex"}:
            raise ValueError(
                "LLM_PROVIDER must be 'auto', 'codex', 'openai', or 'openrouter'"
            )

        if not self.api.kalshi_api_key:
            raise ValueError("KALSHI_API_KEY environment variable is required")

        effective_provider = self.api.resolve_llm_provider()

        if effective_provider == "openai" and not self.api.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER resolves to 'openai'"
            )

        if effective_provider == "openrouter" and not self.api.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required when LLM_PROVIDER resolves to 'openrouter'"
            )

        if effective_provider == "codex":
            # Only enforce CLI presence when the user explicitly asked for
            # codex. `auto` already falls back to openai/openrouter when the
            # CLI is missing, so we don't double-validate here.
            if self.api.llm_provider == "codex" and not _is_codex_cli_ready():
                raise ValueError(
                    "LLM_PROVIDER=codex requires the Codex CLI to be on PATH "
                    "and signed in via a ChatGPT plan. Set CODEX_CLI_PATH "
                    "or run `codex login` before starting."
                )

        if self.trading.max_position_size_pct <= 0 or self.trading.max_position_size_pct > 100:
            raise ValueError("max_position_size_pct must be between 0 and 100")

        if self.trading.min_confidence_to_trade <= 0 or self.trading.min_confidence_to_trade > 1:
            raise ValueError("min_confidence_to_trade must be between 0 and 1")

        if self.trading.quick_flip_allocation < 0 or self.trading.quick_flip_allocation > 1:
            raise ValueError("quick_flip_allocation must be between 0 and 1")

        if self.trading.quick_flip_max_market_checks <= 0:
            raise ValueError("quick_flip_max_market_checks must be positive")

        if self.trading.quick_flip_target_opportunity_buffer <= 0:
            raise ValueError("quick_flip_target_opportunity_buffer must be positive")

        if self.trading.quick_flip_confidence_threshold <= 0 or self.trading.quick_flip_confidence_threshold > 1:
            raise ValueError("quick_flip_confidence_threshold must be between 0 and 1")

        if self.trading.quick_flip_min_market_volume < 0:
            raise ValueError("quick_flip_min_market_volume must be non-negative")

        if self.trading.quick_flip_max_hours_to_expiry <= 0:
            raise ValueError("quick_flip_max_hours_to_expiry must be positive")

        if self.trading.quick_flip_max_bid_ask_spread <= 0 or self.trading.quick_flip_max_bid_ask_spread >= 1:
            raise ValueError("quick_flip_max_bid_ask_spread must be between 0 and 1")

        if self.trading.quick_flip_min_top_of_book_size <= 0:
            raise ValueError("quick_flip_min_top_of_book_size must be positive")

        if self.trading.quick_flip_min_net_profit < 0:
            raise ValueError("quick_flip_min_net_profit must be non-negative")

        if self.trading.quick_flip_min_net_roi < 0 or self.trading.quick_flip_min_net_roi >= 1:
            raise ValueError("quick_flip_min_net_roi must be between 0 and 1")

        if self.trading.quick_flip_recent_trade_window_seconds <= 0:
            raise ValueError("quick_flip_recent_trade_window_seconds must be positive")

        if self.trading.quick_flip_min_recent_trade_count < 0:
            raise ValueError("quick_flip_min_recent_trade_count must be non-negative")

        if (
            self.trading.quick_flip_max_target_vs_recent_trade_gap < 0
            or self.trading.quick_flip_max_target_vs_recent_trade_gap >= 1
        ):
            raise ValueError("quick_flip_max_target_vs_recent_trade_gap must be between 0 and 1")

        if self.trading.quick_flip_min_recent_range_ticks < 0:
            raise ValueError("quick_flip_min_recent_range_ticks must be non-negative")

        if (
            self.trading.quick_flip_min_recent_price_position < 0
            or self.trading.quick_flip_min_recent_price_position > 1
        ):
            raise ValueError("quick_flip_min_recent_price_position must be between 0 and 1")

        if (
            self.trading.quick_flip_max_entry_vs_recent_last_gap < 0
            or self.trading.quick_flip_max_entry_vs_recent_last_gap >= 1
        ):
            raise ValueError("quick_flip_max_entry_vs_recent_last_gap must be between 0 and 1")

        if self.trading.quick_flip_maker_entry_timeout_seconds <= 0:
            raise ValueError("quick_flip_maker_entry_timeout_seconds must be positive")

        if self.trading.quick_flip_maker_entry_poll_seconds <= 0:
            raise ValueError("quick_flip_maker_entry_poll_seconds must be positive")

        if self.trading.quick_flip_maker_entry_reprice_seconds <= 0:
            raise ValueError("quick_flip_maker_entry_reprice_seconds must be positive")

        if self.trading.quick_flip_dynamic_exit_reprice_seconds <= 0:
            raise ValueError("quick_flip_dynamic_exit_reprice_seconds must be positive")

        if self.trading.quick_flip_stop_loss_pct <= 0 or self.trading.quick_flip_stop_loss_pct >= 1:
            raise ValueError("quick_flip_stop_loss_pct must be between 0 and 1")

        for allocation_name in (
            "market_making_allocation",
            "directional_allocation",
            "arbitrage_allocation",
        ):
            allocation_value = getattr(self.trading, allocation_name)
            if allocation_value < 0 or allocation_value > 1:
                raise ValueError(f"{allocation_name} must be between 0 and 1")

        total_allocation = (
            self.trading.market_making_allocation +
            self.trading.directional_allocation +
            self.trading.arbitrage_allocation +
            self.trading.quick_flip_allocation
        )
        if total_allocation > 1.0:
            raise ValueError("strategy allocations must sum to 1.0 or less")

        if self.trading.live_wagering_max_hours_to_expiry <= 0:
            raise ValueError("live_wagering_max_hours_to_expiry must be positive")

        if not (0.0 <= self.weather.model_pool_weight <= 1.0):
            raise ValueError("WEATHER_MODEL_POOL_WEIGHT must be between 0 and 1")

        if not (0.0 <= self.weather.min_quality_to_pool <= 1.0):
            raise ValueError("WEATHER_MIN_QUALITY_TO_POOL must be between 0 and 1")

        if self.weather.max_lead_days < 0:
            raise ValueError("WEATHER_MAX_LEAD_DAYS must be non-negative")

        if not (0.0 <= self.weather.nws_blend_weight <= 1.0):
            raise ValueError("WEATHER_NWS_BLEND_WEIGHT must be between 0 and 1")

        if not (0.0 <= self.sports.model_pool_weight <= 1.0):
            raise ValueError("SPORTS_MODEL_POOL_WEIGHT must be between 0 and 1")

        if not (0.0 <= self.sports.min_quality_to_pool <= 1.0):
            raise ValueError("SPORTS_MIN_QUALITY_TO_POOL must be between 0 and 1")

        if not (0.0 <= self.sports.in_game_max_quality <= 1.0):
            raise ValueError("SPORTS_IN_GAME_MAX_QUALITY must be between 0 and 1")

        for sigma_name in (
            "sigma_floor_f",
            "sigma_base_f",
            "sigma_per_day_f",
            "rain_sigma_in",
            "snow_sigma_in",
        ):
            if getattr(self.weather, sigma_name) < 0:
                raise ValueError(f"weather.{sigma_name} must be non-negative")

        if self.weather.climatology_years < 1:
            raise ValueError("WEATHER_CLIMATOLOGY_YEARS must be at least 1")

        if not self.weather.ensemble_models:
            raise ValueError("WEATHER_ENSEMBLE_MODELS must list at least one model")

        required_roles = {
            "news_analyst",
            "forecaster",
            "risk_manager",
            "bull_researcher",
            "bear_researcher",
            "trader",
        }
        missing_roles = required_roles - set(self.ensemble.get_role_model_map())
        if missing_roles:
            missing = ", ".join(sorted(missing_roles))
            raise ValueError(f"ensemble config is missing required roles: {missing}")

        if self.auto.cadence_seconds <= 0:
            raise ValueError("AUTO_CADENCE_SECONDS must be positive")

        if self.auto.max_consecutive_failures <= 0:
            raise ValueError("AUTO_MAX_CONSECUTIVE_FAILURES must be positive")

        return True


# Global settings instance
settings = Settings()

# Validate settings on import
try:
    settings.validate()
except ValueError as e:
    print(f"Configuration validation error: {e}")
    print("Please check your environment variables and configuration.")
