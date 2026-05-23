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
    # Alias for max_threads + num_scout_seed_questions: if set, caps both the
    # number of scout questions generated AND the total spawned thread count.
    # max_threads takes precedence when both are provided.
    seed_threads: int | None = None
    max_worker_retries: int | None = None
    max_consecutive_errors: int | None = None
    max_repeated_moves: int | None = None
    llm_timeout: float | None = None
    num_scout_seed_questions: int | None = None
    initial_questions: list[str] | None = None
    # Question source: "scout" (auto-discover), "human" (user-provided only), "both"
    question_source: str | None = None
    # Free-text context to guide scout question generation
    scout_context: str | None = None
    # Default pattern for session-spawned threads
    default_pattern: str | None = None
    # Summarize the thread history every N steps (0 or very large to disable)
    summarize_every_steps: int | None = None


class CreateSessionRequest(BaseModel):
    dataset_path: str | None = None
    config: SessionConfig | None = None


class CreateThreadRequest(BaseModel):
    question: str = Field(min_length=1)
    motivation: str | None = None


class PostMessageRequest(BaseModel):
    content: str = Field(min_length=1)
    # Only honored on ``POST /api/sessions/{id}/messages``. When True,
    # the message is used as the seed question for a brand-new thread
    # instead of being broadcast into existing threads. Ignored by
    # ``POST /api/threads/{id}/messages`` (always thread-scoped).
    as_new_thread: bool = False


# --- Response schemas ---


class StepEvent(BaseModel):
    type: str  # "llm_call" | "tool_call" | "human_message"
    timestamp: float
    agent: str | None = None
    model: str | None = None
    duration_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    sql: str | None = None
    tool_result: str | None = None
    response: str | None = None
    # Populated when type == "human_message":
    content: str | None = None
    target: str | None = None  # "thread" | "session"


class StepResponse(BaseModel):
    step_number: int
    move: str
    instruction: str
    assessment: str = ""
    rationale: str = ""
    result: str
    view_created: str | None = None
    duration_ms: int | None = None
    start_time: float | None = None
    end_time: float | None = None
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


# --- Pattern / graph schemas ---


class PatternInfo(BaseModel):
    name: str
    description: str
    input_schema: dict


class RunPatternRequest(BaseModel):
    inputs: dict = {}


class RunPatternResponse(BaseModel):
    thread_id: str
    pattern: str
    status: str


class GraphStateResponse(BaseModel):
    thread_id: str
    step_number: int
    current_node: str | None = None
    move_history: list[str] = []
    status: str
    decision: dict | None = None
