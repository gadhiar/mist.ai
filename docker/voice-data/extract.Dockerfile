# mist-voice-extract: separation, dereverb, enhancement, diarization, ASR and forced alignment.
#
# FROM the code-free `runtime` stage of base.Dockerfile: `images build` tags it
# mist-voice-runtime:<first 12 hex> and passes it as BASE_IMAGE, so a new commit reuses the
# venv layers below and rebuilds only the last four steps.
# Upstream code bases that are not pip-installable are cloned at their pinned commits under
# /opt/mist-voice/upstream/.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY docker/voice-data/requirements-extract.txt /opt/mist-voice/requirements/requirements-extract.txt
RUN python3.11 -m venv /opt/venv-extract \
 && /opt/venv-extract/bin/pip install --no-cache-dir -r /opt/mist-voice/requirements/requirements-extract.txt
RUN printf '%s\n' /opt/mist-voice/code > /opt/venv-extract/lib/python3.11/site-packages/mist-voice-code.pth

RUN git clone --filter=blob:none https://github.com/JusperLee/TIGER /opt/mist-voice/upstream/TIGER \
 && git -C /opt/mist-voice/upstream/TIGER checkout --detach 9f18d4a10a7137e1ce8052cfb62215179f1287b6 \
 && git clone --filter=blob:none https://github.com/ZFTurbo/Music-Source-Separation-Training /opt/mist-voice/upstream/Music-Source-Separation-Training \
 && git -C /opt/mist-voice/upstream/Music-Source-Separation-Training checkout --detach 1f700c386e0dd763fe0a57242bb9420dce987354 \
 && git clone --filter=blob:none https://github.com/marianne-m/brouhaha-vad /opt/mist-voice/upstream/brouhaha-vad \
 && git -C /opt/mist-voice/upstream/brouhaha-vad checkout --detach 9132cbe62ac78f90abdbc21bcf6ec6cfe9bb4891 \
 && git clone --filter=blob:none https://github.com/sarulab-speech/Sidon /opt/mist-voice/upstream/Sidon \
 && git -C /opt/mist-voice/upstream/Sidon checkout --detach 45210939881b8a38a0752fb33451a79a9d3703ee

ARG GIT_SHA
RUN test -n "$GIT_SHA" \
 && python3.11 -I -c 'import hashlib, json, pathlib, sys; reqs = sorted(pathlib.Path("/opt/mist-voice/requirements").glob("*.txt")); record = {"family": sys.argv[1], "git_sha": sys.argv[2], "requirements": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reqs}}; pathlib.Path("/etc/mist-voice-image.json").write_text(json.dumps(record, indent=1) + "\n")' extract "$GIT_SHA"

COPY scripts/voice_data/ /opt/mist-voice/code/scripts/voice_data/
COPY scripts/model_bench/ /opt/mist-voice/code/scripts/model_bench/
RUN /opt/venv-extract/bin/python -I -c 'import torch, torchaudio, torchvision, torchcodec, whisperx, pyannote.audio, faster_whisper, ctranslate2, silero_vad, transformers, accelerate, huggingface_hub, numpy, pandas, nltk, omegaconf, scipy, librosa, soundfile, safetensors, einops, beartype, rotary_embedding_torch, ml_collections, yaml, scripts.voice_data' \
 && /opt/venv-score/bin/python -I -c 'import scripts.voice_data'
