# mist-voice-teacher-omnivoice: OmniVoice (timbre only), from PyPI.
#
# FROM the code-free `runtime` stage of base.Dockerfile: `images build` tags it
# mist-voice-runtime:<first 12 hex> and passes it as BASE_IMAGE, so a new commit reuses the
# venv layers below and rebuilds only the last four steps.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY docker/voice-data/requirements-omnivoice.txt /opt/mist-voice/requirements/requirements-omnivoice.txt
RUN python3.11 -m venv /opt/venv-teacher-omnivoice \
 && /opt/venv-teacher-omnivoice/bin/pip install --no-cache-dir -r /opt/mist-voice/requirements/requirements-omnivoice.txt
RUN printf '%s\n' /opt/mist-voice/code > /opt/venv-teacher-omnivoice/lib/python3.11/site-packages/mist-voice-code.pth

ARG GIT_SHA
RUN test -n "$GIT_SHA" \
 && python3.11 -I -c 'import hashlib, json, pathlib, sys; reqs = sorted(pathlib.Path("/opt/mist-voice/requirements").glob("*.txt")); record = {"family": sys.argv[1], "git_sha": sys.argv[2], "requirements": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reqs}}; pathlib.Path("/etc/mist-voice-image.json").write_text(json.dumps(record, indent=1) + "\n")' teacher-omnivoice "$GIT_SHA"

COPY scripts/voice_data/ /opt/mist-voice/code/scripts/voice_data/
COPY scripts/model_bench/ /opt/mist-voice/code/scripts/model_bench/
RUN /opt/venv-teacher-omnivoice/bin/python -I -c 'import torch, torchaudio, omnivoice, transformers, accelerate, gradio, librosa, numpy, pydub, soundfile, tensorboardX, webdataset, scripts.voice_data' \
 && /opt/venv-score/bin/python -I -c 'import scripts.voice_data'
