"""Tests for app.core.parsing — JSON extraction and agent response parsing."""

import pytest

from app.core.parsing import (
    detect_degeneration,
    extract_json,
    parse_coordinator_response,
    parse_scout_response,
    parse_worker_response,
)
from app.models import CoordinatorStatus, MoveType


# --- detect_degeneration ---


def test_detect_degeneration_clean():
    assert detect_degeneration("This is a normal response with varied words.") is False


def test_detect_degeneration_short():
    assert detect_degeneration("pull pull pull") is False


def test_detect_degeneration_token_loop():
    text = "Gears pull a weaker " + "pull " * 25
    assert detect_degeneration(text) is True


def test_detect_degeneration_threshold():
    text = "word " * 19  # 19 consecutive repeats, below default 20
    assert detect_degeneration(text) is False
    text = "word " * 20  # exactly 20
    assert detect_degeneration(text) is True


def test_detect_degeneration_mixed():
    text = "a " * 10 + "b " * 10 + "c " * 10  # no single word repeats 20 times
    assert detect_degeneration(text) is False


# --- extract_json ---


def test_extract_json_raw():
    result = extract_json('{"key": "value"}')
    assert result == {"key": "value"}


def test_extract_json_markdown_wrapped():
    raw = '```json\n{"key": "value"}\n```'
    assert extract_json(raw) == {"key": "value"}


def test_extract_json_markdown_no_lang():
    raw = '```\n{"key": "value"}\n```'
    assert extract_json(raw) == {"key": "value"}


def test_extract_json_with_preamble():
    raw = 'Here is the result:\n{"key": "value"}\nDone.'
    assert extract_json(raw) == {"key": "value"}


def test_extract_json_nested_braces():
    raw = '{"outer": {"inner": [1, 2, 3]}}'
    result = extract_json(raw)
    assert result["outer"]["inner"] == [1, 2, 3]


def test_extract_json_invalid_raises():
    with pytest.raises(ValueError, match="Could not extract JSON"):
        extract_json("no json here at all")


def test_extract_json_empty_raises():
    with pytest.raises(ValueError):
        extract_json("")


# --- parse_coordinator_response ---


def test_parse_coordinator_response_full():
    raw = """```json
    {
        "assessment": "We need to explore more",
        "next_move": "FORAGE",
        "rationale": "Not enough data yet",
        "worker_instruction": "Run a GROUP BY query",
        "status": "CONTINUE"
    }
    ```"""
    result = parse_coordinator_response(raw)
    assert result.assessment == "We need to explore more"
    assert result.next_move == MoveType.FORAGE
    assert result.status == CoordinatorStatus.CONTINUE
    assert result.worker_instruction == "Run a GROUP BY query"


def test_parse_coordinator_response_stuck():
    raw = """{
        "assessment": "Stuck",
        "next_move": "INTERROGATE",
        "rationale": "Need help",
        "status": "STUCK",
        "question_for_human": "Is this pattern real?",
        "context": "Found bimodal distribution"
    }"""
    result = parse_coordinator_response(raw)
    assert result.status == CoordinatorStatus.STUCK
    assert result.question_for_human == "Is this pattern real?"
    assert result.context is not None


def test_parse_coordinator_response_defaults():
    raw = '{"assessment": "test"}'
    result = parse_coordinator_response(raw)
    assert result.next_move == MoveType.FORAGE  # default
    assert result.status == CoordinatorStatus.CONTINUE  # default
    assert result.worker_instruction is None


def test_parse_coordinator_response_garbage():
    with pytest.raises(ValueError):
        parse_coordinator_response("not json at all")


# --- parse_worker_response ---


def test_parse_worker_response_full():
    raw = """{
        "summary": "Found 100 rows",
        "details": "No NULLs found",
        "view_requested": {"name": "filtered", "sql": "SELECT * FROM t WHERE x > 0"}
    }"""
    result = parse_worker_response(raw)
    assert "Found 100 rows" in result.result
    assert "No NULLs found" in result.result
    assert result.view_requested is not None
    assert result.view_requested["name"] == "filtered"


def test_parse_worker_response_minimal():
    raw = '{"summary": "Nothing to report"}'
    result = parse_worker_response(raw)
    assert result.result == "Nothing to report"
    assert result.view_requested is None


def test_parse_worker_response_garbage():
    with pytest.raises(ValueError):
        parse_worker_response("garbage input")


# --- parse_scout_response ---


def test_parse_scout_response_full():
    raw = """{
        "exploration_notes": "Interesting dataset",
        "questions": [
            {
                "question": "Why are there gaps?",
                "motivation": "Could reveal bias",
                "entry_point": "Check distributions",
                "difficulty": "moderate"
            },
            {
                "question": "Is there clustering?",
                "motivation": "Formation theory",
                "entry_point": "Run K-means",
                "difficulty": "deep"
            }
        ]
    }"""
    result = parse_scout_response(raw)
    assert result.exploration_notes == "Interesting dataset"
    assert len(result.questions) == 2
    assert result.questions[0].difficulty == "moderate"
    assert result.questions[1].question == "Is there clustering?"


def test_parse_scout_response_empty_questions():
    raw = '{"exploration_notes": "Nothing notable", "questions": []}'
    result = parse_scout_response(raw)
    assert result.questions == []


def test_parse_scout_response_missing_fields_defaults():
    raw = '{"questions": [{"question": "Test?"}]}'
    result = parse_scout_response(raw)
    assert result.exploration_notes == ""
    assert result.questions[0].motivation == ""
    assert result.questions[0].difficulty == "moderate"  # default


def test_parse_scout_response_garbage():
    with pytest.raises(ValueError):
        parse_scout_response("not json")
