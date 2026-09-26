"""Pin the ADR-017 WebSocket protocol version the backend announces.

`backend/server.py` sends `PROTOCOL_VERSION` on `session_started`. The ADR moved to
1.2.0 on 2026-09-22 without this constant following, and to 1.3.0 with the additive
`extraction_status` message (MIS-171). A bump that is not also made in the ADR, or
the reverse, should fail here rather than drift silently again.
"""

import re

from backend import server

EXPECTED_PROTOCOL_VERSION = "1.3.0"


def test_protocol_version_matches_adr_017() -> None:
    assert server.PROTOCOL_VERSION == EXPECTED_PROTOCOL_VERSION


def test_protocol_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", server.PROTOCOL_VERSION)
