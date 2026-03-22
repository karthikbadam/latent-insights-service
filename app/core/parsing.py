"""
Parsing utilities — LLM JSON response → dataclasses.

Handles the messy reality of LLM outputs: markdown-wrapped JSON,
missing fields, extra whitespace.
"""

import json
import re

from app.models import (
    CoordinatorDecision,
    CoordinatorStatus,
    MoveType,
    ScoutOutput,
    ScoutQuestion,
    WorkerResult,
)


def detect_degeneration(text: str, max_repeats: int = 20) -> bool:
    """Detect LLM token loop degeneration (e.g., 'pull pull pull...')."""
    words = text.split()
    if len(words) < max_repeats:
        return False
    count = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            count += 1
            if count >= max_repeats:
                return True
        else:
            count = 1
    return False


def extract_json(raw: str) -> dict:
    """
    Extract JSON from an LLM response that might be wrapped in
    markdown code blocks or have extra text around it.
    """
    # Try direct parse first
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code blocks
    match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from response: {raw[:300]}")


def parse_coordinator_response(raw: str) -> CoordinatorDecision:
    data = extract_json(raw)
    return CoordinatorDecision(
        assessment=data.get("assessment", ""),
        next_move=MoveType(data.get("next_move", "FORAGE")),
        rationale=data.get("rationale", ""),
        status=CoordinatorStatus(data.get("status", "CONTINUE")),
        worker_instruction=data.get("worker_instruction"),
        question_for_human=data.get("question_for_human"),
        context=data.get("context"),
    )


def parse_worker_response(raw: str) -> WorkerResult:
    data = extract_json(raw)
    # Combine summary + details into a single result field
    raw_summary = data.get("summary", "")
    summary = json.dumps(raw_summary) if not isinstance(raw_summary, str) else raw_summary
    raw_details = data.get("details")
    details = json.dumps(raw_details) if raw_details and not isinstance(raw_details, str) else raw_details
    result = f"{summary}\n\n{details}" if details else summary
    return WorkerResult(
        result=result,
        view_requested=data.get("view_requested"),
    )


def parse_scout_response(raw: str) -> ScoutOutput:
    data = extract_json(raw)
    return ScoutOutput(
        exploration_notes=data.get("exploration_notes", ""),
        questions=[
            ScoutQuestion(
                question=q.get("question", ""),
                motivation=q.get("motivation", ""),
                entry_point=q.get("entry_point", ""),
                difficulty=q.get("difficulty", "moderate"),
            )
            for q in data.get("questions", [])
        ],
    )
