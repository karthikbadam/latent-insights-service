"""
Pydantic schemas — used only at API boundaries for validation/serialization.
"""

from pydantic import BaseModel, Field


# --- Request schemas ---


class SessionConfig(BaseModel):
    """Optional per-session overrides. Omitted fields use server defaults."""
    model_profiler: str | None = None
    model_scout: str | None = None
    model_coordinator: str | None = None
    model_worker: str | None = None
    model_worker_fallback: str | None = None
    temp_profiler: float | None = None
    temp_scout: float | None = None
    temp_coordinator: float | None = None
    temp_worker: float | None = None
    max_threads: int | None = None
    max_worker_retries: int | None = None
    max_consecutive_errors: int | None = None
    max_repeated_moves: int | None = None
    llm_timeout: float | None = None
    num_scout_seed_questions: int | None = None
    initial_questions: list[str] | None = None


class CreateSessionRequest(BaseModel):
    dataset_path: str | None = None
    config: SessionConfig | None = None


class CreateThreadRequest(BaseModel):
    question: str = Field(min_length=1)
    motivation: str | None = None


class PostMessageRequest(BaseModel):
    content: str = Field(min_length=1)


# --- Response schemas ---


class StepEvent(BaseModel):
    type: str
    timestamp: float
    agent: str | None = None
    model: str | None = None
    duration_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    sql: str | None = None
    tool_result: str | None = None


class StepResponse(BaseModel):
    step_number: int
    move: str
    instruction: str
    result: str
    view_created: str | None = None
    duration_ms: int | None = None
    events: list[StepEvent] = []


class ThreadResponse(BaseModel):
    id: str
    seed_question: str
    motivation: str | None = None
    status: str
    summary: str | None = None
    running_summary: str | None = None
    error: str | None = None
    steps: list[StepResponse] = []
    updated_at: str


class SessionUrls(BaseModel):
    self: str
    events: str
    threads: str


class SessionResponse(BaseModel):
    id: str
    dataset_path: str
    schema_summary: str | None = None
    scout_questions: list[dict] | None = None
    threads: list[ThreadResponse]
    urls: SessionUrls
    created_at: str


class SessionSummary(BaseModel):
    id: str
    dataset_path: str
    table_name: str
    thread_count: int
    status_counts: dict[str, int] = {}
    created_at: str


class SystemStats(BaseModel):
    session_count: int
    thread_count: int
