"""
Configuration — single source of truth for all settings.
Everything is configurable via environment variables.
Per-session overrides via with_overrides().
"""

import copy
import os
from dataclasses import dataclass, field


PROVIDER_DEFAULTS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "",
        "think": True,
        "models": {
            "profiler": "google/gemini-2.5-flash",
            "scout": "google/gemini-2.5-flash",
            "coordinator": "openai/gpt-oss-20b:nitro",
            "worker": "openai/gpt-oss-20b:nitro",
            "worker_fallback": "openai/gpt-oss-20b:nitro",
        },
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "think": False,
        "models": {
            "profiler": "gpt-oss:20b",
            "scout": "gpt-oss:20b",
            "coordinator": "gpt-oss:20b",
            "worker": "gpt-oss:20b",
            "worker_fallback": "gpt-oss:20b",
        },
    },
}


@dataclass
class ModelConfig:
    """LLM model selection per agent role."""

    profiler: str = ""
    scout: str = ""
    coordinator: str = ""
    worker: str = ""
    worker_fallback: str = ""

    @classmethod
    def from_env(cls, provider: str = "openrouter") -> "ModelConfig":
        defaults = PROVIDER_DEFAULTS[provider]["models"]
        return cls(
            profiler=os.getenv("MODEL_PROFILER", defaults["profiler"]),
            scout=os.getenv("MODEL_SCOUT", defaults["scout"]),
            coordinator=os.getenv("MODEL_COORDINATOR", defaults["coordinator"]),
            worker=os.getenv("MODEL_WORKER", defaults["worker"]),
            worker_fallback=os.getenv("MODEL_WORKER_FALLBACK", defaults["worker_fallback"]),
        )


@dataclass
class TemperatureConfig:
    """Temperature per agent role."""

    profiler: float = 0.0
    scout: float = 0.7
    coordinator: float = 0.3
    worker: float = 0.0

    @classmethod
    def from_env(cls) -> "TemperatureConfig":
        return cls(
            profiler=float(os.getenv("TEMP_PROFILER", cls.profiler)),
            scout=float(os.getenv("TEMP_SCOUT", cls.scout)),
            coordinator=float(os.getenv("TEMP_COORDINATOR", cls.coordinator)),
            worker=float(os.getenv("TEMP_WORKER", cls.worker)),
        )


@dataclass
class AppConfig:
    """Top-level application config."""

    # LLM provider
    llm_provider: str = "openrouter"
    llm_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_think: bool = True
    app_name: str = "Latent Insights"
    app_url: str = "https://karthikbadam.github.io"

    # DuckDB
    data_dir: str = "data"

    # Threading
    default_seed_threads: int = 3
    num_scout_seed_questions: int = 8
    max_threads: int | None = None
    initial_questions: list[str] = field(default_factory=list)

    # Agents
    max_worker_retries: int = 3
    max_consecutive_errors: int = 5
    max_repeated_moves: int = 10
    llm_timeout: float = 120.0

    # Sub-configs
    models: ModelConfig = field(default_factory=ModelConfig)
    temperatures: TemperatureConfig = field(default_factory=TemperatureConfig)

    def with_overrides(self, overrides: dict) -> "AppConfig":
        """Return a copy with per-session overrides applied. None values are ignored."""
        cfg = copy.copy(self)
        cfg.models = copy.copy(self.models)
        cfg.temperatures = copy.copy(self.temperatures)

        model_fields = {
            "model_profiler": "profiler", "model_scout": "scout",
            "model_coordinator": "coordinator", "model_worker": "worker",
            "model_worker_fallback": "worker_fallback",
        }
        for key, attr in model_fields.items():
            if overrides.get(key) is not None:
                setattr(cfg.models, attr, overrides[key])

        temp_fields = {
            "temp_profiler": "profiler", "temp_scout": "scout",
            "temp_coordinator": "coordinator", "temp_worker": "worker",
        }
        for key, attr in temp_fields.items():
            if overrides.get(key) is not None:
                setattr(cfg.temperatures, attr, overrides[key])

        for key in ["max_worker_retries", "max_consecutive_errors", "max_repeated_moves",
                    "llm_timeout", "num_scout_seed_questions", "max_threads",
                    "initial_questions"]:
            if overrides.get(key) is not None:
                setattr(cfg, key, overrides[key])

        return cfg

    @classmethod
    def from_env(cls) -> "AppConfig":
        provider = os.getenv("LLM_PROVIDER", "openrouter")
        defaults = PROVIDER_DEFAULTS[provider]

        api_key = os.getenv("LLM_API_KEY") or defaults["api_key"]
        base_url = os.getenv("LLM_BASE_URL") or defaults["base_url"]
        think_env = os.getenv("LLM_THINK")
        think = think_env.lower() in ("1", "true") if think_env else defaults["think"]

        return cls(
            llm_provider=provider,
            llm_api_key=api_key,
            llm_base_url=base_url,
            llm_think=think,
            app_name=os.getenv("APP_NAME", cls.app_name),
            app_url=os.getenv("APP_URL", cls.app_url),
            data_dir=os.getenv("DATA_DIR", cls.data_dir),
            default_seed_threads=int(os.getenv("DEFAULT_SEED_THREADS", cls.default_seed_threads)),
            num_scout_seed_questions=int(os.getenv("NUM_SCOUT_SEED_QUESTIONS", cls.num_scout_seed_questions)),
            max_worker_retries=int(os.getenv("MAX_WORKER_RETRIES", cls.max_worker_retries)),
            max_consecutive_errors=int(os.getenv("MAX_CONSECUTIVE_ERRORS", cls.max_consecutive_errors)),
            max_repeated_moves=int(os.getenv("MAX_REPEATED_MOVES", cls.max_repeated_moves)),
            llm_timeout=float(os.getenv("LLM_TIMEOUT", cls.llm_timeout)),
            models=ModelConfig.from_env(provider),
            temperatures=TemperatureConfig.from_env(),
        )
