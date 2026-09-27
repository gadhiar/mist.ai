# MIST.AI extraction service -- minimal deploy image (T1b, goal mist-two-loop / MIS-171).
#
# This image ships ONLY the stateless extraction service
# (`backend/extraction_service/`) and the backend subpackages its import
# closure pulls in -- never the voice pipeline, Neo4j write path, or vault
# layer that `docker/backend/Dockerfile` builds for the main stack. It is
# built and run identically on both deploy targets; only the compose file
# that runs it differs: docker-compose.extraction.yml's `extraction-local`
# profile on the RTX 4070 SUPER workstation, and the standalone
# docker/extraction/compose.host.yml (no profiles) on the GTX 1070 remote
# host. They differ in env vars and network exposure, not in this image.
#
# Pinned by digest: tag `python:3.11-slim` (Python 3.11.16), resolved on the
# host by the lead on 2026-09-26. tests/unit/test_compose_extraction.py pins
# this exact string.
FROM python@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e

# First-run memory reservation for claude-worker containers (see
# ~/.claude CLAUDE.md "Running commands"). This image installs a short,
# pinned requirements.txt and copies backend/ only -- no torch, no CUDA
# devel toolchain -- so 2GiB clamps to the documented minimum rather than
# the 8GiB a label-less image would reserve on its first run.
LABEL claude-worker.first-run-memory=2GiB

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root user, UID 1000 -- same convention as docker/backend/Dockerfile
# (matches the conventional first user on Linux hosts and Docker Desktop's
# WSL2 UID mapping on Windows).
RUN groupadd -g 1000 appgroup \
    && useradd -m -u 1000 -g appgroup -s /usr/sbin/nologin appuser

WORKDIR /app

# Dependencies first (changes far less often than backend/ code).
COPY docker/extraction/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Application code. Explicit COPY of backend/ only -- no tests/, data/, or
# models/ (those live at the repo root, outside backend/, so this line
# cannot pull them in even by accident).
COPY backend/ /app/backend/

RUN chown -R appuser:appgroup /app

USER appuser
ENV HOME=/home/appuser

EXPOSE 8090

# python:3.11-slim has no curl; urllib is stdlib. Hardcodes the default
# EXTRACTION_PORT (8090, backend/extraction_service/settings.py) the same
# way docker/backend/Dockerfile's compose healthcheck hardcodes 8001 --
# override at the compose `healthcheck.test` level if EXTRACTION_PORT is
# ever overridden away from its default.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8090/v1/health')"]

ENTRYPOINT ["python", "-m", "backend.extraction_service"]
