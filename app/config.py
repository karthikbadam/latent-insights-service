"""
Configuration — single source of truth for all settings.
Everything is configurable via environment variables.
"""

import os
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """LLM model selection per agent role. All OpenRouter model IDs."""

    profiler: str = "google/gemini-2.5-flash"
    scout: str = "google/gemini-2.5-flash"
    coordinator: str = "openai/gpt-oss-20b"
    worker: str = "openai/gpt-oss-20b"
    worker_fallback: str = "openai/gpt-oss-20b"

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            profiler=os.getenv("MODEL_PROFILER", cls.profiler),
            scout=os.getenv("MODEL_SCOUT", cls.scout),
            coordinator=os.getenv("MODEL_COORDINATOR", cls.coordinator),
            worker=os.getenv("MODEL_WORKER", cls.worker),
            worker_fallback=os.getenv("MODEL_WORKER_FALLBACK", cls.worker_fallback),
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
class CacheConfig:
    """Cache TTL per role in hours. Set to 0 to disable caching for a role."""

    profiler: int = 8760  # 1 year — same dataset = same profile
    worker: int = 24      # same instruction + schema = same SQL
    scout: int = 0        # no cache — want creative variation
    coordinator: int = 0  # no cache — depends on evolving history


@dataclass
class AppConfig:
    """Top-level application config."""

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    app_name: str = "Latent Insights"
    app_url: str = "https://yoursite.github.io"

    # DuckDB
    data_dir: str = "data"

    # Threading
    default_seed_threads: int = 5

    # Agents
    max_worker_retries: int = 3

    # Sub-configs
    models: ModelConfig = field(default_factory=ModelConfig)
    temperatures: TemperatureConfig = field(default_factory=TemperatureConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", cls.openrouter_base_url),
            app_name=os.getenv("APP_NAME", cls.app_name),
            app_url=os.getenv("APP_URL", cls.app_url),
            data_dir=os.getenv("DATA_DIR", cls.data_dir),
            default_seed_threads=int(os.getenv("DEFAULT_SEED_THREADS", "5")),
            max_worker_retries=int(os.getenv("MAX_WORKER_RETRIES", "3")),
            models=ModelConfig.from_env(),
            temperatures=TemperatureConfig.from_env(),
            cache=CacheConfig(),
        )
