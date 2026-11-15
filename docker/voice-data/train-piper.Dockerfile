# mist-voice-train-piper: Piper fine-tuning from the OHF-Voice/piper1-gpl v1.8.0 source tree.
#
# FROM the code-free `runtime` stage of base.Dockerfile: `images build` tags it
# mist-voice-runtime:<first 12 hex> and passes it as BASE_IMAGE, so a new commit reuses the
# venv layers below and rebuilds only the last four steps.
# The requirements install piper-tts 1.8.0 with its train extra; the v1.8.0 tree, checked out by
# the commit its lightweight tag names (639388b6317fc4731e91d53da42aea68fd4166ff), is then
# installed over it in editable mode, without dependencies, and its monotonic-align extension
# built, as upstream's TRAINING.md does.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update \
 && apt-get install -y --no-install-recommends cmake ninja-build \
 && rm -rf /var/lib/apt/lists/*

COPY docker/voice-data/requirements-piper.txt /opt/mist-voice/requirements/requirements-piper.txt
RUN python3.11 -m venv /opt/venv-train-piper \
 && /opt/venv-train-piper/bin/pip install --no-cache-dir -r /opt/mist-voice/requirements/requirements-piper.txt
RUN printf '%s\n' /opt/mist-voice/code > /opt/venv-train-piper/lib/python3.11/site-packages/mist-voice-code.pth

RUN git clone --filter=blob:none https://github.com/OHF-Voice/piper1-gpl /opt/mist-voice/upstream/piper1-gpl \
 && git -C /opt/mist-voice/upstream/piper1-gpl checkout --detach 639388b6317fc4731e91d53da42aea68fd4166ff \
 && /opt/venv-train-piper/bin/pip install --no-cache-dir --no-deps -e /opt/mist-voice/upstream/piper1-gpl \
 && cd /opt/mist-voice/upstream/piper1-gpl \
 && PATH=/opt/venv-train-piper/bin:$PATH ./build_monotonic_align.sh

ARG GIT_SHA
RUN test -n "$GIT_SHA" \
 && python3.11 -I -c 'import hashlib, json, pathlib, sys; reqs = sorted(pathlib.Path("/opt/mist-voice/requirements").glob("*.txt")); record = {"family": sys.argv[1], "git_sha": sys.argv[2], "requirements": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reqs}}; pathlib.Path("/etc/mist-voice-image.json").write_text(json.dumps(record, indent=1) + "\n")' train-piper "$GIT_SHA"

COPY scripts/voice_data/ /opt/mist-voice/code/scripts/voice_data/
COPY scripts/model_bench/ /opt/mist-voice/code/scripts/model_bench/
RUN /opt/venv-train-piper/bin/python -I -c 'import torch, piper, piper.train, tensorboard, tensorboardX, onnx, onnxruntime, librosa, numpy, scripts.voice_data' \
 && /opt/venv-score/bin/python -I -c 'import scripts.voice_data'
