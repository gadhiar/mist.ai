# mist-voice-ref-voxtream2: the VoXtream2 streaming reference, from PyPI.
#
# FROM the code-free `runtime` stage of base.Dockerfile: `images build` tags it
# mist-voice-runtime:<first 12 hex> and passes it as BASE_IMAGE, so a new commit reuses the
# venv layers below and rebuilds only the last four steps.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY docker/voice-data/requirements-voxtream2.txt /opt/mist-voice/requirements/requirements-voxtream2.txt
RUN python3.11 -m venv /opt/venv-ref-voxtream2 \
 && /opt/venv-ref-voxtream2/bin/pip install --no-cache-dir -r /opt/mist-voice/requirements/requirements-voxtream2.txt
RUN printf '%s\n' /opt/mist-voice/code > /opt/venv-ref-voxtream2/lib/python3.11/site-packages/mist-voice-code.pth

ARG GIT_SHA
RUN test -n "$GIT_SHA" \
 && python3.11 -I -c 'import hashlib, json, pathlib, sys; reqs = sorted(pathlib.Path("/opt/mist-voice/requirements").glob("*.txt")); record = {"family": sys.argv[1], "git_sha": sys.argv[2], "requirements": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reqs}}; pathlib.Path("/etc/mist-voice-image.json").write_text(json.dumps(record, indent=1) + "\n")' ref-voxtream2 "$GIT_SHA"

COPY scripts/voice_data/ /opt/mist-voice/code/scripts/voice_data/
COPY scripts/model_bench/ /opt/mist-voice/code/scripts/model_bench/
RUN /opt/venv-ref-voxtream2/bin/python -I -c 'import torch, torchaudio, voxtream, torchtune, torchao, transformers, huggingface_hub, librosa, soundfile, inflect, nltk, gradio, starlette, pydantic, silero_vad, pyloudnorm, jiwer, whisper, scripts.voice_data' \
 && /opt/venv-score/bin/python -I -c 'import scripts.voice_data'
