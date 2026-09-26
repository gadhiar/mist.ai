"""Voice VRAM probe: run Whisper STT once and Chatterbox TTS once, report peak CUDA memory.

This is the only file in scripts/model_bench that is NOT stdlib-only (see
CLAUDE.md and the T3a brief): it runs INSIDE the mist-backend container,
piped in via `docker exec -i mist-backend python -` (see bench_host.py's
`voice` subcommand), and imports torch and the backend's own multimodal
modules -- lazily, after `sys.path` is set up, so importing this module on
the host (where torch is not installed) does not fail.

It is deliberately a single self-contained script, not a package import
from bench_host.py: `docker exec -i ... python -` only has this file's
source on stdin, nothing else from the repo is on that stream.

Eager-loading note (see README.md "Voice probe double-counting"): production
already loads both models at mist-backend startup
(`backend/server.py:500-507` awaits `VoiceProcessor.initialize()`, which
`backend/voice_processor.py:178` runs `ModelManager.load_all_models()`
(`backend/voice_models/model_manager.py:113-152`) in an executor). This
probe constructs its OWN WhisperSTT/ChatterboxTTS instances rather than
reusing the running backend's, so its peak-memory reading double-counts
whatever the backend already holds resident -- the vram_steps.json fit
estimate this feeds is therefore conservative (overstates VRAM headroom
pressure), not exact.

Prints exactly one JSON line on stdout:
    {"max_memory_reserved_mib": float|null, "max_memory_allocated_mib": float|null,
     "stt_ok": bool, "tts_ok": bool, "error": str|null}
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


def _mib(num_bytes: int) -> float:
    return round(num_bytes / (1024 * 1024), 3)


def main() -> None:
    result: dict[str, object] = {
        "max_memory_reserved_mib": None,
        "max_memory_allocated_mib": None,
        "stt_ok": False,
        "tts_ok": False,
        "error": None,
    }
    errors: list[str] = []

    try:
        import numpy as np
        import torch

        # Production's project root inside the container is fixed by
        # docker/backend/Dockerfile's `WORKDIR /app` and docker-compose.yml's
        # `./data:/app/data` mount, which is where voice_profiles live
        # (backend/voice_models/model_manager.py:171). Overridable for
        # container layouts other than mist-backend's own.
        project_root = os.environ.get("MIST_PROJECT_ROOT", "/app")
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        try:
            from src.multimodal.stt import WhisperSTT

            stt = WhisperSTT(model_size=os.environ.get("WHISPER_MODEL", "base"))
            silence = np.zeros(16000 * 2, dtype=np.float32)  # 2s @ 16kHz
            stt.transcribe_audio(silence, sample_rate=16000)
            result["stt_ok"] = True
        except Exception as exc:  # noqa: BLE001 -- probe reports both halves, never crashes on one
            errors.append(f"stt: {exc!r}")

        try:
            from src.multimodal.tts import ChatterboxTTS
            from src.multimodal.voice_profile import VoiceProfileRegistry

            data_dir = Path(project_root) / "data" / "voice_profiles"
            registry = VoiceProfileRegistry(data_dir)
            profile = registry.get_active()
            tts = ChatterboxTTS(profile=profile, device=None)
            tts.generate("Voice VRAM probe.")
            result["tts_ok"] = True
        except Exception as exc:  # noqa: BLE001 -- see stt comment above
            errors.append(f"tts: {exc!r}")

        if torch.cuda.is_available():
            result["max_memory_reserved_mib"] = _mib(torch.cuda.max_memory_reserved())
            result["max_memory_allocated_mib"] = _mib(torch.cuda.max_memory_allocated())

        if errors:
            result["error"] = "; ".join(errors)

    except Exception:  # noqa: BLE001 -- top-level: always print one JSON line, never a traceback
        result["error"] = traceback.format_exc()

    print(json.dumps(result))


if __name__ == "__main__":
    main()
