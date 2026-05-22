"""
FeedEntry — single render-ready wire/storage type.

Every SSE event and every entry in the session feed is a complete
``FeedEntry``: a flat row the frontend can append without reshaping,
parsing, or sorting. Long-form text fields (``full_message``,
``response_text``/``response_tables``, ``schema_summary_markdown``) are
pre-computed server-side so clients render straight from the wire.

``session_to_feed`` produces the same shape from a static
``SessionResponse`` snapshot. Canonical ordering:

  1. ``schema_summary_ready`` (if ``session.schema_summary`` is set)
  2. ``session_ready`` or ``scout_done`` (whichever applies)
  3. Per thread in ``session.threads`` order:
     - ``thread_start``
     - For each step in ``thread.steps`` order:
       - ``HUMAN_INPUT`` → one ``human_message`` row (no start/complete)
       - ``WAITING_FOR_HUMAN`` → skipped (trailing ``thread_waiting`` covers it)
       - otherwise: ``step_start`` → each event → ``step_complete``
     - ``thread_complete`` or ``thread_waiting``

``feed_index`` is assigned 0, 1, 2, … in emission order. Live SSE
emissions also use a per-session monotonic counter, so for a session
where threads start strictly after the session-level events fire, the
live and snapshot indices match.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

from latent_insights.api.schemas import SessionResponse
from latent_insights.core.parsing import extract_json


# ---------------------------------------------------------------------------
# FeedEntry
# ---------------------------------------------------------------------------


class FeedEntry(BaseModel):
    """One render-ready row in the session feed.

    All optional fields default to ``None`` — only fields relevant to the
    row's ``event_type`` are populated. Frontends switch on
    ``event_type`` and read the matching subset of fields.
    """

    # --- identity ---
    id: str
    feed_index: int
    event_type: str
    thread_id: str
    timestamp: float

    # --- common header ---
    message: str
    full_message: str | None = None

    # --- step / event identity ---
    step_number: int | None = None
    move: str | None = None
    agent: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_ms: int | None = None

    # --- tool_call payload ---
    sql: str | None = None
    tool_result: str | None = None

    # --- llm_call payload (raw + pre-parsed for the frontend) ---
    response: str | None = None
    response_text: str | None = None
    response_tables: dict[str, list] | None = None
    has_tool_calls: bool | None = None

    # --- human_message payload ---
    content: str | None = None
    target: str | None = None

    # --- thread_start / thread_complete / thread_waiting ---
    seed_question: str | None = None
    motivation: str | None = None
    entry_point: str | None = None
    thread_status: str | None = None
    reason: str | None = None
    running_summary: str | None = None
    summary: str | None = None
    result: str | None = None
    total_ms: int | None = None
    total_seconds: float | None = None
    step_count: int | None = None
    is_terminal: bool | None = None
    from_step: int | None = None

    # --- step_start coordinator extras ---
    instruction: str | None = None
    assessment: str | None = None
    rationale: str | None = None
    status: str | None = None

    # --- step_complete extras ---
    view_created: str | None = None

    # --- session-level (schema/scout/session_ready) ---
    schema_summary: str | None = None
    schema_summary_markdown: str | None = None
    dataset_path: str | None = None
    scout_questions: list[dict] | None = None
    question_source: str | None = None
    question_count: int | None = None
    source_threads: list[str] | None = None
    synthesis_thread: str | None = None


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------


_LLM_TEXT_KEYS = ("text", "answer", "summary", "response", "message", "details")


def parse_llm_response(
    raw: str | None,
) -> tuple[str | None, dict[str, list] | None]:
    """Return ``(response_text, response_tables)`` for an LLM response.

    Reuses :func:`latent_insights.core.parsing.extract_json` to handle
    markdown fences and first-``{...}`` fallback. If the parsed JSON has
    a known text key (``text`` / ``answer`` / ``summary`` / ``response`` /
    ``message`` / ``details``), that becomes ``response_text``. A dict
    under ``tables`` becomes ``response_tables``. Non-JSON input returns
    ``(raw, None)``.
    """
    if not raw:
        return None, None
    try:
        parsed = extract_json(raw)
    except (ValueError, json.JSONDecodeError):
        return raw, None
    if not isinstance(parsed, dict):
        return raw, None

    text: str | None = None
    for key in _LLM_TEXT_KEYS:
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            text = value
            break
    tables = parsed.get("tables")
    if not isinstance(tables, dict):
        tables = None
    if text is None and tables is None:
        return raw, None
    return text, tables


# ---------------------------------------------------------------------------
# Schema summary → markdown
# ---------------------------------------------------------------------------


def format_schema_summary(raw: str | None) -> str | None:
    """Rewrite the profiler's column-profile lines into a markdown table.

    Input shape (from ``Profiler``):
        ## Dataset summary
        - **Table:** foo
        ...

        ## Column profiles
        col_a | TYPE | n/m | stats...
        col_b | TYPE | n/m | stats...

        ## Notable patterns
        ...

    Output keeps every section verbatim except ``## Column profiles``,
    whose pipe-delimited rows are rewritten as a Markdown table with
    columns ``Column | Type | Filled | Stats``. Returns ``None`` if the
    input is empty.
    """
    if not raw:
        return None

    lines = raw.split("\n")
    out: list[str] = []
    in_profiles = False
    table_emitted = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## Column profiles"):
            in_profiles = True
            table_emitted = False
            out.append(line)
            continue
        if in_profiles and stripped.startswith("## "):
            in_profiles = False
            out.append(line)
            continue
        if in_profiles:
            if not stripped:
                if table_emitted:
                    out.append(line)
                continue
            parts = [p.strip() for p in line.split("|", 3)]
            if len(parts) == 4:
                if not table_emitted:
                    out.append("")
                    out.append("| Column | Type | Filled | Stats |")
                    out.append("|---|---|---|---|")
                    table_emitted = True
                stats = parts[3].replace("|", "\\|")
                out.append(f"| {parts[0]} | {parts[1]} | {parts[2]} | {stats} |")
            else:
                out.append(line)
            continue
        out.append(line)

    return "\n".join(out)


# ---------------------------------------------------------------------------
# session_to_feed mapper
# ---------------------------------------------------------------------------


def session_to_feed(session: SessionResponse) -> list[FeedEntry]:
    """Flatten a ``SessionResponse`` into ordered render-ready feed entries.

    Indices are assigned ``0, 1, 2, …`` in emission order. Timestamps
    are taken from the underlying records where available (events,
    thread.updated_at), falling back to ``session.created_at`` for
    rows we have to synthesize.
    """
    session_ts = _iso_to_ts(session.created_at)
    entries: list[FeedEntry] = []

    if session.schema_summary:
        entries.append(_make(
            event_type="schema_summary_ready",
            id=f"schema:{session.id}",
            thread_id="",
            timestamp=session_ts,
            message="Dataset profiled.",
            schema_summary=session.schema_summary,
            schema_summary_markdown=format_schema_summary(session.schema_summary),
            dataset_path=session.dataset_path,
        ))

    if session.scout_questions:
        entries.append(_make(
            event_type="scout_done",
            id=f"scout:{session.id}",
            thread_id="",
            timestamp=session_ts,
            message=f"Scout found {len(session.scout_questions)} questions",
            scout_questions=list(session.scout_questions),
            question_count=len(session.scout_questions),
        ))
    elif session.schema_summary:
        entries.append(_make(
            event_type="session_ready",
            id=f"session:{session.id}:ready",
            thread_id="",
            timestamp=session_ts,
            message="Session profiled. Waiting for human questions.",
            question_source="human",
            dataset_path=session.dataset_path,
        ))

    for thread in session.threads:
        thread_ts = _iso_to_ts(thread.updated_at) or session_ts
        entries.append(_make(
            event_type="thread_start",
            id=f"thread:{thread.id}:start",
            thread_id=thread.id,
            timestamp=thread_ts,
            message=thread.seed_question,
            full_message=thread.seed_question,
            seed_question=thread.seed_question,
            motivation=thread.motivation,
            step_number=0,
        ))

        for step in thread.steps:
            move = step.move or ""
            if move == "HUMAN_INPUT":
                entries.append(_make(
                    event_type="human_message",
                    id=f"human:{thread.id}:{step.step_number}",
                    thread_id=thread.id,
                    timestamp=thread_ts,
                    message=step.result or "",
                    full_message=step.result or "",
                    content=step.result or "",
                    target=step.instruction or "thread",
                    step_number=step.step_number,
                    move=move,
                ))
                continue
            if move == "WAITING_FOR_HUMAN":
                # Trailing thread_waiting row covers this step.
                continue

            # Pull the coordinator's call out of step.events so it folds
            # into the structured step_start row (assessment / rationale /
            # metrics). The raw coordinator llm_call is not emitted as a
            # separate row — its content is already exposed structurally.
            coord_ev, coord_idx = _find_coordinator_event(step.events)
            coord_assessment, coord_rationale = _extract_assessment_rationale(coord_ev)

            # See Recorder.step_start for the message-is-assessment-only
            # rationale. full_message is omitted so the frontend's
            # generic textContent fallback doesn't pick up the long
            # instruction as the row preview.
            entries.append(_make(
                event_type="step_start",
                id=f"step:{thread.id}:{step.step_number}:start",
                thread_id=thread.id,
                timestamp=thread_ts,
                message=coord_assessment or "",
                step_number=step.step_number,
                move=move,
                agent="coordinator",
                instruction=step.instruction or "",
                assessment=coord_assessment or "",
                rationale=coord_rationale or "",
                model=(coord_ev or {}).get("model"),
                input_tokens=(coord_ev or {}).get("input_tokens"),
                output_tokens=(coord_ev or {}).get("output_tokens"),
                duration_ms=(coord_ev or {}).get("duration_ms"),
            ))

            for ev_idx, ev in enumerate(step.events):
                if ev_idx == coord_idx:
                    continue
                entries.append(_event_to_entry(thread.id, step, ev_idx, ev))

            entries.append(_make(
                event_type="step_complete",
                id=f"step:{thread.id}:{step.step_number}:complete",
                thread_id=thread.id,
                timestamp=thread_ts,
                message=step.result or "",
                full_message=step.result or "",
                step_number=step.step_number,
                move=move,
                instruction=step.instruction or "",
                result=step.result or "",
                view_created=step.view_created,
                duration_ms=step.duration_ms,
            ))

        if thread.status == "complete":
            step_count = len(thread.steps)
            entries.append(_make(
                event_type="thread_complete",
                id=f"thread:{thread.id}:complete",
                thread_id=thread.id,
                timestamp=thread_ts,
                message=thread.summary or "",
                full_message=thread.summary or "",
                summary=thread.summary,
                result=thread.summary,
                step_count=step_count,
                step_number=step_count + 1,
                is_terminal=True,
                thread_status="complete",
            ))
        elif thread.status == "waiting":
            reason = thread.error or "waiting_for_human"
            question = ""
            for step in reversed(thread.steps):
                if step.move == "WAITING_FOR_HUMAN":
                    question = step.result or ""
                    break
            entries.append(_make(
                event_type="thread_waiting",
                id=f"thread:{thread.id}:waiting",
                thread_id=thread.id,
                timestamp=thread_ts,
                message=question,
                full_message=question,
                reason=reason,
                running_summary=thread.running_summary,
                step_number=len(thread.steps),
                is_terminal=True,
                thread_status="waiting",
            ))

    for index, entry in enumerate(entries):
        entry.feed_index = index
    return entries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make(**fields: Any) -> FeedEntry:
    """Build a FeedEntry with feed_index=0 (mapper assigns the real index)."""
    fields.setdefault("feed_index", 0)
    return FeedEntry(**fields)


def _iso_to_ts(iso: str | None) -> float:
    if not iso:
        return 0.0
    from datetime import datetime

    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return 0.0


def _event_to_entry(
    thread_id: str, step, ev_idx: int, ev,
) -> FeedEntry:
    """Convert a ``StepEvent`` snapshot record into a FeedEntry."""
    ev_dict = ev.model_dump() if hasattr(ev, "model_dump") else dict(ev)
    ev_type = ev_dict.get("type") or "llm_call"
    ts = float(ev_dict.get("timestamp") or 0.0)
    step_number = step.step_number
    move = step.move or ""
    entry_id = f"ev:{thread_id}:{step_number}:{ev_idx}"

    # Legacy saved snapshots labeled every worker event as ``llm_call``
    # even when the record carried ``sql``/``tool_result``. Discriminate
    # on ``sql`` presence so those rows render as tool_call entries (the
    # frontend's SQL panel keys off this type, not just the field set).
    has_sql = bool(ev_dict.get("sql"))

    if ev_type == "tool_call" or has_sql:
        sql = ev_dict.get("sql") or ""
        return _make(
            event_type="tool_call",
            id=entry_id,
            thread_id=thread_id,
            timestamp=ts,
            message=sql,
            full_message=sql,
            step_number=step_number,
            move=move,
            agent=ev_dict.get("agent") or "worker",
            sql=sql,
            tool_result=ev_dict.get("tool_result"),
            duration_ms=ev_dict.get("duration_ms"),
        )

    if ev_type == "human_message":
        content = ev_dict.get("content") or ""
        return _make(
            event_type="human_message",
            id=entry_id,
            thread_id=thread_id,
            timestamp=ts,
            message=content,
            full_message=content,
            step_number=step_number,
            move=move,
            content=content,
            target=ev_dict.get("target"),
        )

    # llm_call (default)
    response = ev_dict.get("response") or ""
    response_text, response_tables = parse_llm_response(response)
    agent = ev_dict.get("agent") or ""
    duration_ms = ev_dict.get("duration_ms")
    return _make(
        event_type="llm_call",
        id=entry_id,
        thread_id=thread_id,
        timestamp=ts,
        message=_llm_message(agent, duration_ms),
        full_message=response_text or response,
        step_number=step_number,
        move=move,
        agent=agent,
        model=ev_dict.get("model"),
        input_tokens=ev_dict.get("input_tokens"),
        output_tokens=ev_dict.get("output_tokens"),
        duration_ms=duration_ms,
        response=response,
        response_text=response_text,
        response_tables=response_tables,
    )


def _llm_message(agent: str, duration_ms: int | None) -> str:
    ms = f" ({duration_ms}ms)" if duration_ms is not None else ""
    label = agent.capitalize() if agent else "LLM"
    return f"{label} deciding{ms}"


def _find_coordinator_event(events: list) -> tuple[dict | None, int]:
    """Return the first coordinator llm_call event from a step's events.

    Returns ``(event_dict, index)`` or ``(None, -1)`` if absent. Used by
    ``session_to_feed`` to fold the coordinator's call into ``step_start``
    instead of emitting it as a separate row.
    """
    for idx, ev in enumerate(events):
        ev_dict = ev.model_dump() if hasattr(ev, "model_dump") else dict(ev)
        if ev_dict.get("agent") == "coordinator":
            return ev_dict, idx
    return None, -1


def _extract_assessment_rationale(
    ev_dict: dict | None,
) -> tuple[str | None, str | None]:
    """Pull ``assessment`` and ``rationale`` strings from a coordinator's
    LLM response JSON. Returns ``(None, None)`` if the response can't be
    parsed or the keys are missing.
    """
    if not ev_dict:
        return None, None
    raw = ev_dict.get("response") or ""
    if not raw:
        return None, None
    try:
        parsed = extract_json(raw)
    except (ValueError, json.JSONDecodeError):
        return None, None
    if not isinstance(parsed, dict):
        return None, None
    assessment = parsed.get("assessment")
    rationale = parsed.get("rationale")
    return (
        assessment if isinstance(assessment, str) else None,
        rationale if isinstance(rationale, str) else None,
    )
