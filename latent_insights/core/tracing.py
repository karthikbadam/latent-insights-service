"""
Lightweight OpenTelemetry-style trace store.

Spans accumulate in memory during the session. On thread completion,
they're flushed to JSONL trace files.
"""

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)


def _generate_span_id() -> str:
    return uuid.uuid4().hex[:16]


@dataclass
class Span:
    trace_id: str
    span_id: str = field(default_factory=_generate_span_id)
    parent_span_id: str | None = None
    name: str = ""
    kind: str = "step"
    attributes: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    status: str = "ok"
    status_message: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None


class TraceStore:
    def __init__(self, data_dir: str = "data"):
        self._traces: dict[str, list[Span]] = {}
        self._data_dir = data_dir

    def start_span(
        self,
        trace_id: str,
        name: str,
        kind: str = "step",
        parent_span_id: str | None = None,
        attributes: dict | None = None,
    ) -> Span:
        span = Span(
            trace_id=trace_id,
            name=name,
            kind=kind,
            parent_span_id=parent_span_id,
            attributes=attributes or {},
        )
        if trace_id not in self._traces:
            self._traces[trace_id] = []
        self._traces[trace_id].append(span)
        return span

    def end_span(
        self,
        span: Span,
        status: str = "ok",
        status_message: str | None = None,
    ):
        span.end_time = time.time()
        span.status = status
        span.status_message = status_message

    def add_event(self, span: Span, name: str, attributes: dict | None = None):
        span.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def get_spans(self, trace_id: str) -> list[Span]:
        return self._traces.get(trace_id, [])

    def get_step_spans(self, trace_id: str) -> list[Span]:
        return [s for s in self.get_spans(trace_id) if s.kind == "step"]

    def format_thread_history(
        self,
        trace_id: str,
        human_messages: list[str] | None = None,
        running_summary: str | None = None,
        full_window: int = 3,
    ) -> str:
        """Format thread history with windowing for long threads.

        - If running_summary exists, prepend it
        - Step 1 always in full
        - Middle steps: condensed (move + first sentence of result)
        - Last `full_window` steps: full detail
        - Human messages at end
        """
        steps = self.get_step_spans(trace_id)
        if not steps:
            preamble = f"Summary so far: {running_summary}\n\n" if running_summary else ""
            if human_messages:
                parts = [f'[Human input]: "{msg}"' for msg in human_messages]
                return preamble + "\n\n".join(parts) if preamble else "\n\n".join(parts)
            return preamble + "(No steps yet — this is the first move)"

        parts = []

        if running_summary:
            parts.append(f"**Summary of earlier analysis:**\n{running_summary}")

        total = len(steps)
        cutoff = max(1, total - full_window)

        for i, span in enumerate(steps, 1):
            move = span.attributes.get("move", "?")
            instruction = span.attributes.get("instruction", "")
            result = span.attributes.get("result", "")

            if i == 1 or i > cutoff:
                # Full detail for step 1 and last N steps
                parts.append(
                    f"Step {i} [{move}]:\n"
                    f'  Instruction: "{instruction}"\n'
                    f"  Result: {result}"
                )
            else:
                # Condensed for middle steps
                first_sentence = result.split(".")[0].strip() + "." if result else ""
                parts.append(f"Step {i} [{move}]: {first_sentence}")

        if human_messages:
            for msg in human_messages:
                parts.append(f'[Human input]: "{msg}"')

        return "\n\n".join(parts)

    def summarize_history(
        self,
        trace_id: str,
        llm,
        model: str,
        seed_question: str,
        threshold: int = 5,
    ) -> str | None:
        """Use LLM to compress thread history into a running summary.

        Returns None if step count is below threshold.
        """
        steps = self.get_step_spans(trace_id)
        if len(steps) < threshold:
            return None

        history_parts = []
        for i, span in enumerate(steps, 1):
            move = span.attributes.get("move", "?")
            result = span.attributes.get("result", "")
            history_parts.append(f"Step {i} [{move}]: {result}")

        history_text = "\n\n".join(history_parts)

        messages = [
            {"role": "system", "content": (
                "You are a research assistant. Summarize the analytical thread history below "
                "into 3-5 sentences. Preserve key findings, hypotheses tested, and data patterns "
                "discovered. Be specific about numbers and results."
            )},
            {"role": "user", "content": (
                f"Thread question: {seed_question}\n\n"
                f"History ({len(steps)} steps):\n\n{history_text}\n\n"
                "Summarize the progress so far."
            )},
        ]

        response = llm.call(
            model=model,
            messages=messages,
            role="summarizer",
            temperature=0.0,
            max_tokens=512,
        )
        return response.content

    def flush_to_file(self, trace_id: str, session_id: str):
        """Write all spans for a trace to a JSONL file."""
        spans = self.get_spans(trace_id)
        if not spans:
            return

        trace_dir = os.path.join(self._data_dir, "traces", session_id)
        os.makedirs(trace_dir, exist_ok=True)
        filepath = os.path.join(trace_dir, f"{trace_id}.jsonl")

        with open(filepath, "w") as f:
            for span in spans:
                f.write(json.dumps(asdict(span), default=str) + "\n")

        logger.info(f"Flushed {len(spans)} spans to {filepath}")
        return filepath

    def load_trace(self, trace_id: str, session_id: str) -> list[Span]:
        """Reload spans from a JSONL file into memory."""
        filepath = os.path.join(self._data_dir, "traces", session_id, f"{trace_id}.jsonl")
        if not os.path.exists(filepath):
            return []

        spans = []
        with open(filepath) as f:
            for line in f:
                data = json.loads(line)
                span = Span(
                    trace_id=data["trace_id"],
                    span_id=data["span_id"],
                    parent_span_id=data.get("parent_span_id"),
                    name=data.get("name", ""),
                    kind=data.get("kind", "step"),
                    attributes=data.get("attributes", {}),
                    events=data.get("events", []),
                    status=data.get("status", "ok"),
                    status_message=data.get("status_message"),
                    start_time=float(data.get("start_time", 0)),
                    end_time=float(data["end_time"]) if data.get("end_time") else None,
                )
                spans.append(span)

        self._traces[trace_id] = spans
        logger.info(f"Loaded {len(spans)} spans from {filepath}")
        return spans

    def clear_trace(self, trace_id: str):
        self._traces.pop(trace_id, None)
