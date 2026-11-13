# Voice-data stage images

One image per family in `scripts/voice_data/families.json`. Pins come from H2's pin sheet
(2026-11-12); H3 builds the images, and nothing here has been built yet.

## Build

```bash
python -m scripts.voice_data images build --sha <commit> --family <key or all>
```

`images build` exports `scripts/voice_data`, `scripts/model_bench` and `docker/voice-data` at the
commit with `git archive`, builds `base.Dockerfile` as `mist-voice-base:<first 12 hex>`, then each
family's `<image without mist-voice->.Dockerfile` with `--build-arg GIT_SHA=<commit>`. A family
Dockerfile finds its base as `mist-voice-base:${GIT_SHA%<28 x ?>}`, the 40-hex sha with its last
28 characters dropped; that string trim needs Dockerfile syntax 1.7 or later, which current
Docker Engine releases carry built in. `--build-arg BASE_IMAGE=<tag>` overrides it.

## Layout of every image

- `/opt/venv-score`: speaker similarity (WavLM-SV, ECAPA), Whisper large-v3-turbo for CER, the
  arousal monitor and librosa (`requirements-score.txt`).
- `/opt/venv-<family key>`: the family's own environment, with its own torch wheel and CUDA
  libraries (`requirements-<name>.txt`).
- `/opt/mist-voice/code/scripts/{voice_data,model_bench}`: the code at the image's commit, the
  last layer copied. `mist-voice-code.pth` in each venv's `site-packages` puts
  `/opt/mist-voice/code` on the path, because jobs run `<venv>/bin/python -I`.
- `/opt/mist-voice/upstream/<repo>`: upstream code that is not pip-installable, cloned at its
  pinned commit (TIGER, Music-Source-Separation-Training, brouhaha-vad, Sidon, CosyVoice with
  Matcha-TTS, knn-vc, seed-vc, piper1-gpl at tag v1.8.0, pocket-tts, StyleTTS2, neutts).
- `/etc/mist-voice-image.json`: the family, the git sha and the sha256 of each requirements file.
- Default command `remote-entry.sh`: on a pod with `PUBLIC_KEY` set, sshd with key-only login,
  then wait. Local jobs name their own command.

## Devices and VRAM (estimates)

`devices` and `vram_gb` in `families.json` are estimates from model sizes, not measurements:
score 6, extract 8, teacher-indextts2 16, teacher-cosyvoice3 8, teacher-omnivoice 8,
teacher-fish-s2 24, teacher-higgs3 24, convert 8, train-piper 12, train-pocket 16 (batch 16 with
4-step accumulation), train-styletts2 24, train-neutts 16, ref-voxtream2 8 GB. Replace each with a
measured peak once the family has run.

## Resolved at H3

The pin sheet left these unverified. They are not in the requirements files; pip resolves each as
a dependency where a pinned package pulls it in, and H3's `pip freeze` records the version. One
that nothing pulls in is added by H3 at the version it confirms.

- extract: `brouhaha-vad` (not on PyPI; its code is cloned, and whether it runs beside
  pyannote.audio 4.0.7 is open).
- teacher-indextts2: `fugashi`, `unidic-lite`, `textstat`, `requests`, `WeTextProcessing`.
- teacher-omnivoice: `huggingface-hub` (a 1.x release for transformers 5).
- teacher-fish-s2: `natsort`, `kui`, `loguru`, `pyaudio`, `ormsgpack`, `tiktoken`, `cachetools`,
  `protobuf`.
- teacher-higgs3: `vllm-omni`'s PyPI release (installed from the recipe's git commit instead).
- convert: `resemblyzer`, `python-dotenv`.
- train-piper: `lightning`, `jsonargparse`, `pysilero-vad`, `pathvalidate` (the train extra).
- train-pocket: `sphn`, `utmos-pytorch`, `whisper-normalizer`.
- train-styletts2: `typing-extensions`.
- train-neutts: `neucodec`, `datasets`, `accelerate`, `huggingface-hub`.
- ref-voxtream2: `moshi`.

Also open after H2:

- `sha256` is `"unverified"` for the three `url` models (`whisperx-align-en`,
  `knnvc-wavlm-large`, `knnvc-hifigan-prematched`); fetch hashes them at the first download.
- `enhancer-discriminative` is left out of `families.json`: no checkpoint was found.
- Licences marked `unverified` keep that value in `families.json`.
- `crisperwhisper`, `keras`, `opencv-python`, `deepspeed`, the TensorRT wheels, `FreeSimpleGUI`,
  `sounddevice` and `g2p-en` are installed but left out of the final import check, whose module
  names or import-time needs (a GPU, libGL, PortAudio, Tk, TensorFlow, or a download) were not
  confirmed.
- System packages come from Ubuntu 22.04 and the deadsnakes PPA without version pins.
