# mist-voice-teacher-indextts2: IndexTTS-2, installed from index-tts at its pinned commit.
#
# FROM the code-free `runtime` stage of base.Dockerfile: `images build` tags it
# mist-voice-runtime:<first 12 hex> and passes it as BASE_IMAGE, so a new commit reuses the
# venv layers below and rebuilds only the last four steps.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY docker/voice-data/requirements-indextts2.txt /opt/mist-voice/requirements/requirements-indextts2.txt
RUN python3.11 -m venv /opt/venv-teacher-indextts2 \
 && /opt/venv-teacher-indextts2/bin/pip install --no-cache-dir -r /opt/mist-voice/requirements/requirements-indextts2.txt
RUN printf '%s\n' /opt/mist-voice/code > /opt/venv-teacher-indextts2/lib/python3.11/site-packages/mist-voice-code.pth

ARG GIT_SHA
RUN test -n "$GIT_SHA" \
 && python3.11 -I -c 'import hashlib, json, pathlib, sys; reqs = sorted(pathlib.Path("/opt/mist-voice/requirements").glob("*.txt")); record = {"family": sys.argv[1], "git_sha": sys.argv[2], "requirements": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reqs}}; pathlib.Path("/etc/mist-voice-image.json").write_text(json.dumps(record, indent=1) + "\n")' teacher-indextts2 "$GIT_SHA"

COPY scripts/voice_data/ /opt/mist-voice/code/scripts/voice_data/
COPY scripts/model_bench/ /opt/mist-voice/code/scripts/model_bench/
RUN /opt/venv-teacher-indextts2/bin/python -I -c 'import torch, torchaudio, accelerate, cn2an, audiotools, einops, ffmpeg, jieba, json5, librosa, matplotlib, modelscope, munch, numba, numpy, omegaconf, pandas, safetensors, sentencepiece, tensorboard, tokenizers, tqdm, transformers, whisper, indextts, scripts.voice_data' \
 && /opt/venv-score/bin/python -I -c 'import scripts.voice_data'
