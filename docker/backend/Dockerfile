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
RUN pip install flash-attn --no-build-isolation || echo "flash-attn optional install skipped"

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

EXPOSE 8001

CMD ["python", "backend/server.py"]
