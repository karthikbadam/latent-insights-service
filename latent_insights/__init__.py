"""Latent Insights — parallel-agent sensemaking tool for data analysis."""

from latent_insights.config import AppConfig
from latent_insights.models import (
    CoordinatorDecision,
    CoordinatorStatus,
    MoveType,
    ScoutOutput,
    ScoutQuestion,
    Session,
    Step,
    StreamEvent,
    Thread,
    ThreadStatus,
    WorkerResult,
)

__version__ = "0.1.0"

__all__ = [
    "AppConfig",
    "CoordinatorDecision",
    "CoordinatorStatus",
    "MoveType",
    "ScoutOutput",
    "ScoutQuestion",
    "Session",
    "Step",
    "StreamEvent",
    "Thread",
    "ThreadStatus",
    "WorkerResult",
]
