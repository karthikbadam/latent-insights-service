"""Tests for the FeedEntry contract: mapper, LLM parser, schema formatter."""

import json
import os

import pytest

from latent_insights.api.feed import (
    FeedEntry,
    format_schema_summary,
    parse_llm_response,
    session_to_feed,
)
from latent_insights.api.schemas import SessionResponse, SessionUrls


SAVED_SESSIONS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "sessions",
)
SAVED_SESSION_IDS = ["846f0bbfefc0", "a59dfbbd0fee", "746fa2380425"]


def _load_session(session_id: str) -> SessionResponse:
    with open(os.path.join(SAVED_SESSIONS_DIR, f"{session_id}.json")) as f:
        data = json.load(f)
    data.setdefault("urls", SessionUrls(self="", events="", threads="").model_dump())
    data.setdefault("threads", [])
    return SessionResponse.model_validate(data)


# ---------------------------------------------------------------------------
# parse_llm_response
# ---------------------------------------------------------------------------


class TestParseLLMResponse:
    def test_empty_input(self):
        assert parse_llm_response("") == (None, None)
        assert parse_llm_response(None) == (None, None)

    def test_freeform_text_passthrough(self):
        text, tables = parse_llm_response("just a plain sentence with no JSON")
        assert text == "just a plain sentence with no JSON"
        assert tables is None

    def test_raw_json_with_text_field(self):
        text, tables = parse_llm_response('{"text": "hello world"}')
        assert text == "hello world"
        assert tables is None

    def test_json_with_summary_and_details(self):
        # The order is text > answer > summary > response > message > details.
        text, _ = parse_llm_response('{"summary": "short", "details": "long"}')
        assert text == "short"

    def test_json_with_tables(self):
        raw = json.dumps({
            "text": "results below",
            "tables": {"top_cars": [{"name": "Fiat", "mpg": 32}]},
        })
        text, tables = parse_llm_response(raw)
        assert text == "results below"
        assert tables == {"top_cars": [{"name": "Fiat", "mpg": 32}]}

    def test_fenced_json(self):
        raw = "Here are my findings:\n```json\n{\"text\": \"fenced\"}\n```"
        text, tables = parse_llm_response(raw)
        assert text == "fenced"
        assert tables is None

    def test_json_with_text_prefix(self):
        # extract_json's first-{...} fallback picks up the embedded JSON.
        raw = 'Some preamble text. {"text": "embedded"}'
        text, _ = parse_llm_response(raw)
        assert text == "embedded"

    def test_malformed_json_returns_raw(self):
        raw = "{not actually json"
        text, tables = parse_llm_response(raw)
        assert text == raw
        assert tables is None

    def test_non_dict_json_returns_raw(self):
        raw = "[1, 2, 3]"
        text, tables = parse_llm_response(raw)
        assert text == raw
        assert tables is None


# ---------------------------------------------------------------------------
# format_schema_summary
# ---------------------------------------------------------------------------


class TestFormatSchemaSummary:
    def test_empty_input(self):
        assert format_schema_summary("") is None
        assert format_schema_summary(None) is None

    def test_column_profiles_become_markdown_table(self):
        raw = (
            "## Dataset summary\n"
            "- **Table:** foo\n"
            "- **Rows:** 10\n"
            "\n"
            "## Column profiles\n"
            "a | INT | 10/10 | min=1, max=10\n"
            "b | VARCHAR | 8/10 | 5 unique\n"
            "\n"
            "## Notable patterns\n"
            "- something\n"
        )
        out = format_schema_summary(raw)
        assert "## Dataset summary" in out
        assert "- **Table:** foo" in out
        assert "| Column | Type | Filled | Stats |" in out
        assert "|---|---|---|---|" in out
        assert "| a | INT | 10/10 | min=1, max=10 |" in out
        assert "| b | VARCHAR | 8/10 | 5 unique |" in out
        # The Notable patterns section is preserved unchanged.
        assert "## Notable patterns" in out
        assert "- something" in out

    def test_pipes_in_stats_are_escaped(self):
        raw = (
            "## Column profiles\n"
            "x | INT | 1/1 | choices: a|b|c\n"
        )
        out = format_schema_summary(raw)
        assert "choices: a\\|b\\|c" in out

    def test_no_column_profiles_section(self):
        raw = "## Dataset summary\n- **Table:** foo\n"
        out = format_schema_summary(raw)
        assert out == raw

    def test_saved_session_roundtrip(self):
        session = _load_session("846f0bbfefc0")
        out = format_schema_summary(session.schema_summary)
        assert "| Column | Type | Filled | Stats |" in out
        assert "| mpg | DOUBLE |" in out


# ---------------------------------------------------------------------------
# session_to_feed
# ---------------------------------------------------------------------------


class TestSessionToFeed:
    @pytest.mark.parametrize("session_id", SAVED_SESSION_IDS)
    def test_indices_are_sequential(self, session_id):
        session = _load_session(session_id)
        entries = session_to_feed(session)
        assert entries, "expected at least one feed entry"
        for i, e in enumerate(entries):
            assert e.feed_index == i, f"index mismatch at position {i}"

    @pytest.mark.parametrize("session_id", SAVED_SESSION_IDS)
    def test_entry_ids_unique(self, session_id):
        session = _load_session(session_id)
        entries = session_to_feed(session)
        ids = [e.id for e in entries]
        assert len(ids) == len(set(ids)), "expected unique entry ids"

    @pytest.mark.parametrize("session_id", SAVED_SESSION_IDS)
    def test_canonical_order_session_then_threads(self, session_id):
        session = _load_session(session_id)
        entries = session_to_feed(session)
        # Session-scoped rows fire before any thread row.
        session_types = {"schema_summary_ready", "scout_done", "session_ready"}
        thread_seen = False
        for e in entries:
            if e.thread_id:
                thread_seen = True
            elif thread_seen and e.event_type in session_types:
                pytest.fail(
                    f"session-level row {e.event_type} appeared after a "
                    "thread-scoped row"
                )

    @pytest.mark.parametrize("session_id", SAVED_SESSION_IDS)
    def test_step_start_precedes_step_complete_per_step(self, session_id):
        session = _load_session(session_id)
        entries = session_to_feed(session)
        seen_start: set[tuple[str, int]] = set()
        for e in entries:
            if e.event_type == "step_start":
                seen_start.add((e.thread_id, e.step_number))
            elif e.event_type == "step_complete":
                assert (e.thread_id, e.step_number) in seen_start, (
                    f"step_complete without step_start at {e.id}"
                )

    @pytest.mark.parametrize("session_id", SAVED_SESSION_IDS)
    def test_step_start_precedes_llm_call_for_same_step(self, session_id):
        """Regression for the coordinator ordering bug."""
        session = _load_session(session_id)
        entries = session_to_feed(session)
        seen_start: set[tuple[str, int]] = set()
        for e in entries:
            if e.event_type == "step_start":
                seen_start.add((e.thread_id, e.step_number))
            elif e.event_type == "llm_call":
                key = (e.thread_id, e.step_number)
                # HUMAN_INPUT step has no step_start; skip llm_calls outside
                # an analytical step (none should exist in the mapper output).
                assert key in seen_start, (
                    f"llm_call for {key} appeared before its step_start"
                )

    def test_human_input_emits_single_human_message(self):
        session = SessionResponse(
            id="s1",
            dataset_path="/tmp/data.csv",
            schema_summary="## Dataset summary\n- **Table:** t",
            scout_questions=None,
            threads=[{
                "id": "t1",
                "seed_question": "Q?",
                "motivation": "m",
                "status": "complete",
                "summary": "S",
                "steps": [
                    {
                        "step_number": 1,
                        "move": "HUMAN_INPUT",
                        "instruction": "thread",
                        "result": "look at outliers",
                        "events": [],
                    },
                    {
                        "step_number": 2,
                        "move": "FORAGE",
                        "instruction": "explore",
                        "result": "found pattern",
                        "duration_ms": 1000,
                        "events": [],
                    },
                ],
                "updated_at": "",
            }],
            urls=SessionUrls(self="", events="", threads=""),
            created_at="",
        )
        entries = session_to_feed(session)
        types = [e.event_type for e in entries]
        # human_message appears exactly once; no step_start/complete for it.
        assert types.count("human_message") == 1
        human = next(e for e in entries if e.event_type == "human_message")
        assert human.content == "look at outliers"
        assert human.target == "thread"
        # FORAGE step produces step_start + step_complete.
        forage_starts = [e for e in entries if e.event_type == "step_start" and e.move == "FORAGE"]
        forage_completes = [e for e in entries if e.event_type == "step_complete" and e.move == "FORAGE"]
        assert len(forage_starts) == 1
        assert len(forage_completes) == 1

    def test_waiting_for_human_step_is_skipped(self):
        session = SessionResponse(
            id="s1",
            dataset_path="/tmp/data.csv",
            schema_summary=None,
            scout_questions=None,
            threads=[{
                "id": "t1",
                "seed_question": "Q?",
                "status": "waiting",
                "error": "coordinator_stuck",
                "steps": [
                    {
                        "step_number": 1,
                        "move": "WAITING_FOR_HUMAN",
                        "instruction": "context",
                        "result": "need help",
                        "events": [],
                    },
                ],
                "updated_at": "",
            }],
            urls=SessionUrls(self="", events="", threads=""),
            created_at="",
        )
        entries = session_to_feed(session)
        types = [e.event_type for e in entries]
        assert "step_start" not in types
        assert "step_complete" not in types
        waiting = [e for e in entries if e.event_type == "thread_waiting"]
        assert len(waiting) == 1
        assert waiting[0].reason == "coordinator_stuck"
        assert waiting[0].message == "need help"

    def test_step_complete_carries_full_message(self):
        session = _load_session("846f0bbfefc0")
        entries = session_to_feed(session)
        completes = [e for e in entries if e.event_type == "step_complete"]
        assert completes, "expected step_complete entries"
        for e in completes:
            assert e.full_message == e.result

    def test_llm_call_parses_response(self):
        session = _load_session("846f0bbfefc0")
        entries = session_to_feed(session)
        llm_calls = [e for e in entries if e.event_type == "llm_call"]
        # At least one llm_call should successfully parse JSON.
        with_text = [e for e in llm_calls if e.response_text]
        assert with_text, "expected at least one llm_call with parsed response_text"


# ---------------------------------------------------------------------------
# FeedEntry serialization
# ---------------------------------------------------------------------------


def test_feed_entry_serializes_with_excluded_nones():
    entry = FeedEntry(
        id="step:t1:1:start",
        feed_index=5,
        event_type="step_start",
        thread_id="t1",
        timestamp=1.0,
        message="explore",
        step_number=1,
        move="FORAGE",
        instruction="explore",
    )
    dump = entry.model_dump(exclude_none=True)
    assert dump["id"] == "step:t1:1:start"
    assert dump["feed_index"] == 5
    assert "sql" not in dump
    assert "response_text" not in dump
