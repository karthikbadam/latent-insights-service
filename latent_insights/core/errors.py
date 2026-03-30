"""Custom exceptions."""


class AgentError(Exception):
    """Base error for agent failures."""
    pass


class ParseError(AgentError):
    """LLM response could not be parsed."""
    pass


class QueryError(AgentError):
    """SQL query execution failed."""
    pass


class SessionNotFound(Exception):
    pass


class ThreadNotFound(Exception):
    pass
