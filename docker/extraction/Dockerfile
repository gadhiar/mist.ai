# MIST.AI extraction service -- minimal deploy image (T1b, goal mist-two-loop / MIS-171).
#
# This image ships ONLY the stateless extraction service
# (`backend/extraction_service/`) and the backend subpackages its import
# closure pulls in -- never the voice pipeline, Neo4j write path, or vault
# layer that `docker/backend/Dockerfile` builds for the main stack. It is
# built and run identically on both deploy targets (RTX 4070 SUPER local,
# GTX 1070 remote host); only docker-compose.extraction.yml's env vars
# differ per profile.
#
# TODO(lead): pin python:3.11-slim by digest once a build has been run and
# `docker buildx imagetools inspect python:3.11-slim` is available (this
# worker's container has no network to resolve it).
FROM python:3.11-slim

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
