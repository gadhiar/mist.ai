# mist-voice-train-pocket: Pocket TTS fine-tuning. The package comes from PyPI; the training
# scripts (training/train.py and its configs) are not in it, so the repository is cloned at its
# pinned commit.
#
# FROM the code-free `runtime` stage of base.Dockerfile: `images build` tags it
# mist-voice-runtime:<first 12 hex> and passes it as BASE_IMAGE, so a new commit reuses the
# venv layers below and rebuilds only the last four steps.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY docker/voice-data/requirements-pocket.txt /opt/mist-voice/requirements/requirements-pocket.txt
RUN python3.11 -m venv /opt/venv-train-pocket \
 && /opt/venv-train-pocket/bin/pip install --no-cache-dir -r /opt/mist-voice/requirements/requirements-pocket.txt
RUN printf '%s\n' /opt/mist-voice/code > /opt/venv-train-pocket/lib/python3.11/site-packages/mist-voice-code.pth

RUN git clone --filter=blob:none https://github.com/kyutai-labs/pocket-tts /opt/mist-voice/upstream/pocket-tts \
 && git -C /opt/mist-voice/upstream/pocket-tts checkout --detach ee2ebc89aba9d0121d04d18e29c20eea75bb624e

ARG GIT_SHA
RUN test -n "$GIT_SHA" \
 && python3.11 -I -c 'import hashlib, json, pathlib, sys; reqs = sorted(pathlib.Path("/opt/mist-voice/requirements").glob("*.txt")); record = {"family": sys.argv[1], "git_sha": sys.argv[2], "requirements": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reqs}}; pathlib.Path("/etc/mist-voice-image.json").write_text(json.dumps(record, indent=1) + "\n")' train-pocket "$GIT_SHA"

COPY scripts/voice_data/ /opt/mist-voice/code/scripts/voice_data/
COPY scripts/model_bench/ /opt/mist-voice/code/scripts/model_bench/
RUN /opt/venv-train-pocket/bin/python -I -c 'import torch, torchaudio, pocket_tts, numpy, transformers, soundfile, yaml, jiwer, tqdm, scripts.voice_data' \
 && /opt/venv-score/bin/python -I -c 'import scripts.voice_data'
