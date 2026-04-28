# MIST.AI Backend -- Linux container with flash attention
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies (curl for health checks, libportaudio2 for sounddevice)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    libsndfile1 libportaudio2 ffmpeg git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Cluster 8 Phase 2 P0 follow-up (2026-04-22): non-root execution.
# Create appuser UID 1000 (matches the conventional first user on Linux hosts
# and Docker Desktop on Windows, so bind-mounted ./data, ./logs, ./mist-memory
# are owned by a UID the container can write to). Cache dirs precreated +
# chowned so the named volumes (mist-hf-cache, mist-torch-cache) inherit
# correct permissions on first mount.
RUN groupadd -g 1000 appgroup \
    && useradd -m -u 1000 -g appgroup -s /bin/bash appuser \
    && mkdir -p /home/appuser/.cache/huggingface /home/appuser/.cache/torch \
    && chown -R appuser:appgroup /home/appuser \
    && git config --system user.email "mist-backend@local" \
    && git config --system user.name "MIST Backend"

# Create venv inside container
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip

# --- Dependency layers (ordered for cache stability) ---
# These rarely change and are the slowest to install.
# PyTorch CUDA goes first as the heaviest download (~2.6GB).

# 1. PyTorch with CUDA 12.4 (flash attention kernels included in Linux wheels)
RUN pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 2. Chatterbox TTS (installs its own deps but torch is already pinned above)
RUN pip install chatterbox-tts soundfile

# 3. Backend requirements (requirements.txt changes rarely)
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Ensure openai and httpx are installed (safety net if requirements.txt cache is stale)
RUN pip install "openai>=1.30" "httpx>=0.27"

# 4. Re-pin PyTorch ONLY if chatterbox or requirements overwrote it
RUN TORCH_VER=$(python -c "import torch; print(torch.__version__)") && \
    if [ "$TORCH_VER" != "2.6.0+cu124" ]; then \
        echo "PyTorch was overwritten to $TORCH_VER -- re-pinning to 2.6.0+cu124" && \
        pip install --force-reinstall torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
            --index-url https://download.pytorch.org/whl/cu124; \
    else \
        echo "PyTorch 2.6.0+cu124 intact -- skipping re-pin"; \
    fi

# 5. flash-attn standalone (optional)
# Build deps must be installed BEFORE flash-attn because --no-build-isolation
# means pip will not create an ephemeral build env. flash-attn's setup.py
# imports psutil, packaging, and invokes ninja; wheel is needed for the bdist
# step. All four are required or the egg_info step fails before compilation
# even starts. Without these, the install silently failed for months
# (pre-2026-04-22).
RUN pip install ninja packaging wheel psutil
# Known issue (2026-04-27, attempted fix did not work): the resulting
# flash_attn_2_cuda*.so references the C++11 ABI version of
# c10::Error::Error (symbol: NSt7__cxx1112basic_string), but PyTorch 2.6.0+cu124
# from download.pytorch.org reports _GLIBCXX_USE_CXX11_ABI=False. Runtime
# import then fails with `undefined symbol`. Investigation found:
#   - Setting CXXFLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 during the source build did
#     not fix it (torch.utils.cpp_extension.BuildExtension overrides CXXFLAGS).
#   - Installing the upstream precompiled wheel
#     flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
#     produces a binary that ALSO references __cxx11 symbols despite the
#     filename claiming cxx11abi=FALSE -- looks like an upstream wheel-naming
#     inconsistency for v2.8.3.
# Net effect: flash-attn will fail at runtime import on the voice path; voice
# attention falls back to PyTorch SDPA (slower but functional). The build-time
# import check below makes this loud at image-build time so future operators
# don't ship without knowing.
# Followups: try flash-attn 2.7.x or 2.9.x wheels; or rebuild against a torch
# wheel with _GLIBCXX_USE_CXX11_ABI=True (cascades through chatterbox-tts).
RUN pip install flash-attn --no-build-isolation \
    || (echo "[FLASH-ATTN BUILD FAILED] See pip stderr above. Image build continues without flash-attn -- attention falls back to PyTorch SDPA." && exit 0)
RUN python -c "import flash_attn; print('[FLASH-ATTN IMPORT OK]', flash_attn.__version__)" \
    || echo "[FLASH-ATTN IMPORT FAILED at build time] symbol mismatch with torch's ABI -- voice attention will fall back to PyTorch SDPA at runtime"

# 6. Test dependencies
RUN pip install pytest>=8.0 pytest-asyncio>=0.24

# --- Application code ---
WORKDIR /app

# Copy application code (production mode -- dev mode overrides with volume mounts).
# NOTE: ./data is always bind-mounted at runtime via docker-compose.yml.
COPY backend/ /app/backend/
COPY src/ /app/src/
COPY dependencies/ /app/dependencies/
COPY scripts/ /app/scripts/

# Create data directories (populated by bind mount at runtime)
RUN mkdir -p /app/data/voice_profiles /app/data/vector_store /app/data/audio

# Cluster 8 Phase 2 P0 follow-up: hand off ownership to appuser AFTER all
# root-owned dependency installation is done. /opt/venv ownership transfer
# lets pip + python find their own site-packages without permission errors.
# /app ownership transfer lets the runtime write to /app/data /app/logs
# /app/mist-memory (bind mounts override these but the underlying mountpoint
# directories must be writable for Docker to mount over them).
RUN chown -R appuser:appgroup /opt/venv /app /home/appuser

USER appuser
ENV HOME=/home/appuser

EXPOSE 8001

CMD ["python", "backend/server.py"]
