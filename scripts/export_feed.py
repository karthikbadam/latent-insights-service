"""Export a saved SessionResponse JSON as a flat list of FeedEntries.

Usage:
    uv run python scripts/export_feed.py path/to/session.json > out.feed.json
"""

import json
import sys

from latent_insights.api.feed import session_to_feed
from latent_insights.api.schemas import SessionResponse, SessionUrls


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/export_feed.py path/to/session.json", file=sys.stderr)
        return 2

    src = sys.argv[1]
    with open(src) as f:
        data = json.load(f)

    # Saved sessions on disk drop ``urls`` (it's derived per-request) —
    # synthesize an empty SessionUrls so SessionResponse validates.
    data.setdefault("urls", SessionUrls(self="", events="", threads="").model_dump())
    data.setdefault("threads", [])

    session = SessionResponse.model_validate(data)
    entries = session_to_feed(session)
    payload = [e.model_dump(exclude_none=True) for e in entries]
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
