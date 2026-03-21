"""Tests for app.db.queries — CRUD operations and thread history formatting."""


from app.db.queries import (
    append_step,
    create_session,
    create_thread,
    format_thread_history,
    generate_id,
    get_session,
    get_steps,
    get_thread,
    get_threads,
    update_session_schema,
    update_session_scout,
    update_thread_status,
)
from app.models import MoveType, ThreadStatus


# --- generate_id ---


def test_generate_id_format():
    id_ = generate_id()
    assert len(id_) == 12
    assert all(c in "0123456789abcdef" for c in id_)


def test_generate_id_unique():
    ids = {generate_id() for _ in range(100)}
    assert len(ids) == 100


# --- Sessions ---


def test_create_and_get_session(test_db):
    session = create_session(test_db, "/data/test.csv")
    assert session.dataset_path == "/data/test.csv"
    assert session.schema_summary is None

    fetched = get_session(test_db, session.id)
    assert fetched is not None
    assert fetched.id == session.id
    assert fetched.dataset_path == "/data/test.csv"


def test_get_session_not_found(test_db):
    assert get_session(test_db, "nonexistent") is None


def test_update_session_schema(test_db):
    session = create_session(test_db, "/data/test.csv")
    update_session_schema(test_db, session.id, "## Schema\nTest summary")

    fetched = get_session(test_db, session.id)
    assert fetched.schema_summary == "## Schema\nTest summary"


def test_update_session_scout(test_db):
    session = create_session(test_db, "/data/test.csv")
    scout_data = {"exploration_notes": "test", "questions": [{"q": "why?"}]}
    update_session_scout(test_db, session.id, scout_data)

    fetched = get_session(test_db, session.id)
    assert fetched.scout_output == scout_data


# --- Threads ---


def test_create_and_get_thread(test_db):
    session = create_session(test_db, "/data/test.csv")
    thread = create_thread(test_db, session.id, "Why gaps?", "Detection bias", "Check distributions")

    assert thread.seed_question == "Why gaps?"
    assert thread.status == ThreadStatus.RUNNING

    fetched = get_thread(test_db, thread.id)
    assert fetched is not None
    assert fetched.seed_question == "Why gaps?"
    assert fetched.motivation == "Detection bias"


def test_get_thread_not_found(test_db):
    assert get_thread(test_db, "nonexistent") is None


def test_get_threads_ordered(test_db):
    session = create_session(test_db, "/data/test.csv")
    create_thread(test_db, session.id, "First")
    create_thread(test_db, session.id, "Second")
    create_thread(test_db, session.id, "Third")

    threads = get_threads(test_db, session.id)
    assert len(threads) == 3
    assert threads[0].seed_question == "First"
    assert threads[2].seed_question == "Third"


def test_update_thread_status(test_db):
    session = create_session(test_db, "/data/test.csv")
    thread = create_thread(test_db, session.id, "Test")

    update_thread_status(test_db, thread.id, ThreadStatus.WAITING)
    fetched = get_thread(test_db, thread.id)
    assert fetched.status == ThreadStatus.WAITING

    update_thread_status(test_db, thread.id, ThreadStatus.COMPLETE)
    fetched = get_thread(test_db, thread.id)
    assert fetched.status == ThreadStatus.COMPLETE


# --- Steps ---


def test_append_and_get_steps(test_db):
    session = create_session(test_db, "/data/test.csv")
    thread = create_thread(test_db, session.id, "Test")

    s1 = append_step(test_db, thread.id, MoveType.SCOPE, "Filter data", "100 rows filtered")
    s2 = append_step(test_db, thread.id, MoveType.FORAGE, "Explore dist", "Bimodal pattern")
    s3 = append_step(test_db, thread.id, MoveType.FRAME, "Hypothesis", "Proposed: bias effect")

    assert s1.step_number == 1
    assert s2.step_number == 2
    assert s3.step_number == 3

    steps = get_steps(test_db, thread.id)
    assert len(steps) == 3
    assert steps[0].move == MoveType.SCOPE
    assert steps[1].instruction == "Explore dist"
    assert steps[2].result_summary == "Proposed: bias effect"


def test_append_step_with_view(test_db):
    session = create_session(test_db, "/data/test.csv")
    thread = create_thread(test_db, session.id, "Test")

    step = append_step(
        test_db, thread.id, MoveType.SCOPE, "Create subset",
        "Filtered to 50 rows", view_created="mass_radius_complete",
    )
    assert step.view_created == "mass_radius_complete"

    steps = get_steps(test_db, thread.id)
    assert steps[0].view_created == "mass_radius_complete"


# --- Thread history formatting ---


def test_format_thread_history_empty():
    result = format_thread_history([])
    assert "No steps yet" in result


def test_format_thread_history_with_steps(test_db):
    session = create_session(test_db, "/data/test.csv")
    thread = create_thread(test_db, session.id, "Test")

    append_step(test_db, thread.id, MoveType.SCOPE, "Filter data", "100 rows")
    append_step(test_db, thread.id, MoveType.FORAGE, "Explore", "Found pattern")

    steps = get_steps(test_db, thread.id)
    result = format_thread_history(steps)

    assert "Step 1 [SCOPE]" in result
    assert "Step 2 [FORAGE]" in result
    assert '"Filter data"' in result
    assert "100 rows" in result


def test_format_thread_history_with_human_messages(test_db):
    session = create_session(test_db, "/data/test.csv")
    thread = create_thread(test_db, session.id, "Test")

    append_step(test_db, thread.id, MoveType.FORAGE, "Explore", "Found gap")

    steps = get_steps(test_db, thread.id)
    result = format_thread_history(steps, human_messages=["Check by stellar type"])

    assert "[Human input]" in result
    assert "Check by stellar type" in result
