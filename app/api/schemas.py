"""
Pydantic schemas — used only at API boundaries for validation/serialization.
"""

from pydantic import BaseModel, Field


class CreateThreadRequest(BaseModel):
    question: str = Field(min_length=1)
    motivation: str | None = None


class PostMessageRequest(BaseModel):
    content: str = Field(min_length=1)


class ThreadResponse(BaseModel):
    id: str
    seed_question: str
    status: str
    step_count: int
    updated_at: str


class SessionResponse(BaseModel):
    id: str
    thread_count: int
    schema_summary: str | None = None
    scout_questions: list[dict] | None = None


class DatasetInfo(BaseModel):
    name: str
    path: str
    size_bytes: int
    source: str  # "upload" or "sample"


class StepResponse(BaseModel):
    id: str
    step_number: int
    move: str
    instruction: str
    result: str | None = None
    view_created: str | None = None
    duration_ms: int | None = None
    llm_calls: list[dict] | None = None


class ThreadDetailResponse(BaseModel):
    id: str
    session_id: str
    seed_question: str
    motivation: str
    entry_point: str
    status: str
    summary: str | None = None
    error: str | None = None
    steps: list[StepResponse]


class SessionDetailResponse(BaseModel):
    id: str
    dataset_path: str
    schema_summary: str | None = None
    scout_output: dict | None = None
    threads: list[ThreadDetailResponse]
    created_at: str


class TaskResponse(BaseModel):
    task_id: str
    session_id: str
    thread_id: str | None = None
    description: str
    elapsed_seconds: float | None = None


class SystemStats(BaseModel):
    session_count: int
    thread_count: int
    step_count: int
    cache_entries: int


class CacheStats(BaseModel):
    role: str
    cached_calls: int
    input_tokens_saved: int
    output_tokens_saved: int
