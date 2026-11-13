# mist-voice-teacher-higgs3: Higgs TTS 3 (4B), served through vLLM-Omni from its pinned commit.
#
# FROM the base image of the same commit: `images build` tags it mist-voice-base:<first 12 hex>
# and passes only GIT_SHA, so BASE_IMAGE drops the last 28 characters of the 40-hex sha.
ARG GIT_SHA
ARG BASE_IMAGE=mist-voice-base:${GIT_SHA%????????????????????????????}
FROM ${BASE_IMAGE}

COPY docker/voice-data/requirements-higgs3.txt /opt/mist-voice/requirements/requirements-higgs3.txt
RUN python3.11 -m venv /opt/venv-teacher-higgs3 \
 && /opt/venv-teacher-higgs3/bin/pip install --no-cache-dir -r /opt/mist-voice/requirements/requirements-higgs3.txt

ARG GIT_SHA
RUN test -n "$GIT_SHA" \
 && python3.11 -I -c 'import hashlib, json, pathlib, sys; reqs = sorted(pathlib.Path("/opt/mist-voice/requirements").glob("*.txt")); record = {"family": sys.argv[1], "git_sha": sys.argv[2], "requirements": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reqs}}; pathlib.Path("/etc/mist-voice-image.json").write_text(json.dumps(record, indent=1) + "\n")' teacher-higgs3 "$GIT_SHA"

COPY scripts/voice_data/ /opt/mist-voice/code/scripts/voice_data/
COPY scripts/model_bench/ /opt/mist-voice/code/scripts/model_bench/
RUN printf '%s\n' /opt/mist-voice/code > /opt/venv-teacher-higgs3/lib/python3.11/site-packages/mist-voice-code.pth
RUN /opt/venv-teacher-higgs3/bin/python -I -c 'import torch, torchaudio, torchvision, vllm, vllm_omni, transformers, scripts.voice_data'
