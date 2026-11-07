# mist-speech-bench-base: the running mist-backend image, with nothing added.
#
# The backend image already holds chatterbox-tts 0.1.7 (with Turbo), openai-whisper, onnxruntime
# and ffmpeg, so the speech bench's `turbo` and `original` engines, whisper-base and Smart Turn
# run here as built. The probe code is never copied in: it is mounted at run time as
# `-v <repo>:/work:ro`, so a code change needs no rebuild.
#
# Built by the lead with this directory as the context, given the image id the pin file records:
#   docker build -f docker/speech-bench/base.Dockerfile
#     --build-arg BACKEND_IMAGE=<mist-backend image id> -t mist-speech-bench-base docker/speech-bench/
# (one command, wrapped here).
ARG BACKEND_IMAGE
FROM ${BACKEND_IMAGE}
