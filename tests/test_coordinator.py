"""Tests for app.agents.coordinator — thread coordination agent."""

import json
from unittest.mock import MagicMock


from app.agents.coordinator import Coordinator
from app.core.llm import LLMResponse
from app.models import CoordinatorStatus, MoveType
from tests.conftest import make_mock_llm


def _make_coordinator(mock_llm):
    return Coordinator(llm=mock_llm, model="test")


def test_coordinator_continue(schema_summary):
    mock = make_mock_llm("coordinator_response.json")
    coord = _make_coordinator(mock)
    result, _ = coord.call(
        seed_question="Why gaps?",
        motivation="Detection bias",
        entry_point="Compare distributions",
        schema_summary=schema_summary,
        thread_history="(No steps yet — this is the first move)",
    )

    assert result.status == CoordinatorStatus.CONTINUE
    assert result.next_move == MoveType.SCOPE
    assert result.worker_instruction is not None
    assert len(result.assessment) > 0


def test_coordinator_stuck(schema_summary):
    mock = make_mock_llm("coordinator_stuck_response.json")
    coord = _make_coordinator(mock)
    result, _ = coord.call(
        seed_question="Why eccentric?",
        motivation="Dynamics",
        entry_point="Plot eccentricity",
        schema_summary=schema_summary,
        thread_history="Step 1 [FORAGE]:\n  Instruction: ...\n  Result: ...",
    )

    assert result.status == CoordinatorStatus.STUCK
    assert result.question_for_human is not None
    assert result.context is not None


def test_coordinator_done(schema_summary):
    mock = make_mock_llm("coordinator_done_response.json")
    coord = _make_coordinator(mock)
    result, _ = coord.call(
        seed_question="Test",
        motivation="Test",
        entry_point="Test",
        schema_summary=schema_summary,
        thread_history="Step 1 [FORAGE]:...",
    )

    assert result.status == CoordinatorStatus.DONE
    assert result.next_move == MoveType.SYNTHESIZE
    assert result.worker_instruction is not None


def test_coordinator_validates_done_requires_synthesize(schema_summary):
    """If LLM returns DONE but wrong move, coordinator corrects to SYNTHESIZE."""
    mock = MagicMock()
    mock.call.return_value = LLMResponse(
        content=json.dumps({
            "assessment": "Done investigating",
            "next_move": "FORAGE",  # wrong — should be SYNTHESIZE
            "rationale": "wrap up",
            "worker_instruction": "final summary",
            "status": "DONE",
        }),
        model="test",
    )

    coord = _make_coordinator(mock)
    result, _ = coord.call(
        seed_question="Test", motivation="", entry_point="",
        schema_summary=schema_summary,
        thread_history="Step 1...",
    )

    assert result.status == CoordinatorStatus.DONE
    assert result.next_move == MoveType.SYNTHESIZE  # corrected


def test_coordinator_validates_stuck_has_question(schema_summary):
    """If LLM returns STUCK without question, coordinator adds default."""
    mock = MagicMock()
    mock.call.return_value = LLMResponse(
        content=json.dumps({
            "assessment": "stuck",
            "next_move": "INTERROGATE",
            "rationale": "need help",
            "status": "STUCK",
            # no question_for_human
        }),
        model="test",
    )

    coord = _make_coordinator(mock)
    result, _ = coord.call(
        seed_question="Test", motivation="", entry_point="",
        schema_summary=schema_summary,
        thread_history="Step 1...",
    )

    assert result.status == CoordinatorStatus.STUCK
    assert result.question_for_human is not None


def test_coordinator_prompt_contains_context(schema_summary):
    """Verify prompt includes seed question, schema, and history."""
    mock = make_mock_llm("coordinator_response.json")
    coord = _make_coordinator(mock)
    coord.call(
        seed_question="Are planets missing?",
        motivation="Bias analysis",
        entry_point="Check methods",
        schema_summary=schema_summary,
        thread_history="Step 1 [SCOPE]: filtered",
    )

    call_args = mock.call.call_args
    messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
    system_msg = messages[0]["content"]

    assert "Are planets missing?" in system_msg
    assert "Bias analysis" in system_msg
    assert "Step 1 [SCOPE]: filtered" in system_msg


def test_coordinator_retries_on_json_error(schema_summary):
    """Coordinator should retry once with a nudge if initial response isn't valid JSON."""
    mock = MagicMock()

    # First call: non-JSON response
    bad_response = LLMResponse(content="I think we should forage next", model="test")
    # Second call: valid JSON
    good_response = LLMResponse(
        content=json.dumps({
            "assessment": "Need more data",
            "next_move": "FORAGE",
            "rationale": "explore",
            "worker_instruction": "look at distributions",
            "status": "CONTINUE",
        }),
        model="test",
    )
    mock.call.side_effect = [bad_response, good_response]

    coord = _make_coordinator(mock)
    result, _ = coord.call(
        seed_question="Test", motivation="", entry_point="",
        schema_summary=schema_summary,
        thread_history="Step 1...",
    )

    assert result.status == CoordinatorStatus.CONTINUE
    assert mock.call.call_count == 2
