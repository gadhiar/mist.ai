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
    uvicorn.run(app, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
