"""Standalone entry point: `python -m backend.extraction_service`.

Binds uvicorn to `EXTRACTION_PORT` (default 8090) and serves the app
`build_app_from_env()` wires from `EXTRACTION_*` env vars.
"""

from __future__ import annotations

import uvicorn

from backend.extraction_service.app import build_app_from_env
from backend.extraction_service.settings import ServiceSettings


def main() -> None:
    settings = ServiceSettings.from_env()
    app = build_app_from_env()
    # All interfaces inside the container: the backend reaches the service over
    # the compose network (local profile) or through the Tailscale sidecar's
    # shared network namespace (host profile). Exposure is decided by
    # docker-compose.extraction.yml, which publishes no host port in the host
    # profile and only 127.0.0.1 in the local one.
    uvicorn.run(app, host="0.0.0.0", port=settings.port)  # nosec B104


if __name__ == "__main__":
    main()
