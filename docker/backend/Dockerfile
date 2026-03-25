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
RUN pip install --no-cache-dir --upgrade pip

# Install Chatterbox TTS and backend deps FIRST (they pull their own torch)
RUN pip install --no-cache-dir chatterbox-tts soundfile

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# FORCE reinstall PyTorch with CUDA 12.4 AFTER all deps (overrides any
# torch version that chatterbox-tts or requirements.txt pulled in).
# Linux wheels include flash attention CUDA kernels.
RUN pip install --no-cache-dir --force-reinstall torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install flash-attn standalone (optional, additional optimization)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || echo "flash-attn optional install skipped"

# Install test dependencies
RUN pip install --no-cache-dir pytest>=8.0 pytest-asyncio>=0.24

# Set working directory (HARD REQUIREMENT -- relative sys.path in
# model_manager.py and voice_processor.py depends on this being /app)
WORKDIR /app

# Copy application code (production mode -- dev mode overrides with volume mounts).
# NOTE: ./data is always bind-mounted at runtime via docker-compose.yml.
# The COPY below provides code only; voice profiles and data come from the mount.
COPY backend/ /app/backend/
COPY src/ /app/src/
COPY dependencies/ /app/dependencies/
COPY scripts/ /app/scripts/

# Create data directories (populated by bind mount at runtime)
RUN mkdir -p /app/data/voice_profiles /app/data/vector_store /app/data/audio

EXPOSE 8001

CMD ["python", "backend/server.py"]
