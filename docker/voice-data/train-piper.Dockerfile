# mist-voice-train-piper: Piper fine-tuning from the OHF-Voice/piper1-gpl v1.8.0 source tree.
#
# FROM the base image of the same commit: `images build` tags it mist-voice-base:<first 12 hex>
# and passes only GIT_SHA, so BASE_IMAGE drops the last 28 characters of the 40-hex sha.
# The requirements install piper-tts 1.8.0 with its train extra; the v1.8.0 tree (tag 639388b631)
# is then installed over it in editable mode, without dependencies, and its monotonic-align
# extension built, as upstream's TRAINING.md does.
ARG GIT_SHA
ARG BASE_IMAGE=mist-voice-base:${GIT_SHA%????????????????????????????}
FROM ${BASE_IMAGE}

RUN apt-get update \
 && apt-get install -y --no-install-recommends cmake ninja-build \
 && rm -rf /var/lib/apt/lists/*

COPY docker/voice-data/requirements-piper.txt /opt/mist-voice/requirements/requirements-piper.txt
RUN python3.11 -m venv /opt/venv-train-piper \
 && /opt/venv-train-piper/bin/pip install --no-cache-dir -r /opt/mist-voice/requirements/requirements-piper.txt

RUN git clone --filter=blob:none https://github.com/OHF-Voice/piper1-gpl /opt/mist-voice/upstream/piper1-gpl \
 && git -C /opt/mist-voice/upstream/piper1-gpl checkout --detach v1.8.0 \
 && /opt/venv-train-piper/bin/pip install --no-cache-dir --no-deps -e /opt/mist-voice/upstream/piper1-gpl \
 && cd /opt/mist-voice/upstream/piper1-gpl \
 && PATH=/opt/venv-train-piper/bin:$PATH ./build_monotonic_align.sh

ARG GIT_SHA
RUN test -n "$GIT_SHA" \
 && python3.11 -I -c 'import hashlib, json, pathlib, sys; reqs = sorted(pathlib.Path("/opt/mist-voice/requirements").glob("*.txt")); record = {"family": sys.argv[1], "git_sha": sys.argv[2], "requirements": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reqs}}; pathlib.Path("/etc/mist-voice-image.json").write_text(json.dumps(record, indent=1) + "\n")' train-piper "$GIT_SHA"

COPY scripts/voice_data/ /opt/mist-voice/code/scripts/voice_data/
COPY scripts/model_bench/ /opt/mist-voice/code/scripts/model_bench/
RUN printf '%s\n' /opt/mist-voice/code > /opt/venv-train-piper/lib/python3.11/site-packages/mist-voice-code.pth
RUN /opt/venv-train-piper/bin/python -I -c 'import torch, piper, piper.train, tensorboard, tensorboardX, onnx, onnxruntime, librosa, numpy, scripts.voice_data'
