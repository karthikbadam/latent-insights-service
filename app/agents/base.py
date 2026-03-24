"""Base agent class — all agents inherit from this."""

from abc import ABC, abstractmethod

from app.core.llm import LLMClient


class Agent(ABC):
    """Base for all agents. Subclasses own their prompt, call logic, and result parsing."""

    def __init__(self, llm: LLMClient, model: str):
        self.llm = llm
        self.model = model

    @property
    @abstractmethod
    def role(self) -> str:
        """Agent role name (used for logging, LLM call tagging)."""
        ...
