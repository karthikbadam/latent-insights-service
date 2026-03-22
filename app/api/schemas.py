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
    summary: str | None = None
    error: str | None = None
    updated_at: str


class SessionResponse(BaseModel):
    id: str
    dataset_path: str
    schema_summary: str | None = None
    scout_questions: list[dict] | None = None
    threads: list[ThreadResponse]
    created_at: str


class SystemStats(BaseModel):
    session_count: int
    thread_count: int
