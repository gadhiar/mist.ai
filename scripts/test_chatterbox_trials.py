"""Chatterbox Turbo voice cloning trials with varying parameters.

Tests different reference configurations and generation settings
to find the optimal JARVIS voice clone.

Usage:
    python scripts/test_chatterbox_trials.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def concat_refs(paths: list[Path], target_sr: int = 24000) -> Path:
    """Concatenate reference WAVs into a single temp file."""
    segments = []
    for p in paths:
        audio, sr = sf.read(str(p))
        segments.append(audio)
        # Add 0.3s silence between clips
        segments.append(np.zeros(int(target_sr * 0.3)))

    combined = np.concatenate(segments)
    out_path = Path("data/chatterbox_eval/combined_ref.wav")
    sf.write(str(out_path), combined, target_sr)
    dur = len(combined) / target_sr
    print(f"  Combined reference: {dur:.1f}s from {len(paths)} clips")
    return out_path


def run_trial(
    model,
    trial_name: str,
    prompt: str,
    ref_path: Path,
    exaggeration: float = 0.5,
    temperature: float = 0.8,
    cfg_weight: float = 0.5,
    output_dir: Path = Path("data/chatterbox_eval"),
):
    """Run a single generation trial and report metrics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    wav = model.generate(
        text=prompt,
        audio_prompt_path=str(ref_path),
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfg_weight,
    )
    gen_time = time.perf_counter() - t0

    audio_dur = wav.shape[-1] / 24000
    rtf = gen_time / audio_dur
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    out_path = output_dir / f"{trial_name}.wav"
    sf.write(str(out_path), wav.squeeze().cpu().numpy(), 24000)

    print(
        f"  [{trial_name}] {audio_dur:.1f}s audio, {gen_time:.1f}s gen, "
        f"RTF={rtf:.2f}x, VRAM={peak_vram:.1f}GB -> {out_path.name}"
    )
    return rtf


def main():
    print("=" * 60)
    print("CHATTERBOX TURBO - JARVIS VOICE CLONING TRIALS")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("data/chatterbox_eval/trials")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model once
    print("\nLoading Chatterbox Turbo...")
    from chatterbox.tts import ChatterboxTTS

    model = ChatterboxTTS.from_pretrained(device=device)
    print("  Model loaded.")

    # Reference clips
    ref_dir = Path("data/voice_profiles/jarvis/reference")
    ref1 = ref_dir / "ref_001.wav"  # 5.0s - "scan the area for aim signatures"
    ref2 = ref_dir / "ref_002.wav"  # 5.1s - "something you should all see at the war table"
    ref3 = ref_dir / "ref_003.wav"  # 4.2s - "several scouting parties in the area"
    ref4 = ref_dir / "ref_004.wav"  # 5.4s - "locked out of the library"
    ref5 = ref_dir / "ref_005.wav"  # 4.7s - "elevator has been unlocked"

    # Create concatenated references
    print("\nPreparing references...")
    ref_short = ref1  # 5s single clip
    ref_medium = concat_refs([ref1, ref3, ref5])  # ~15s, 3 clips
    # Also try a longer one with all 5
    ref_long_path = Path("data/chatterbox_eval/combined_ref_long.wav")
    segments = []
    for p in [ref1, ref2, ref3, ref4, ref5]:
        audio, sr = sf.read(str(p))
        segments.append(audio)
        segments.append(np.zeros(int(24000 * 0.3)))
    combined = np.concatenate(segments)
    sf.write(str(ref_long_path), combined, 24000)
    ref_long = ref_long_path
    print(f"  Long reference: {len(combined)/24000:.1f}s from 5 clips")

    # Test prompts -- one short, one medium, one with emotion
    prompts = {
        "short": "Good evening, sir. All systems are operational.",
        "medium": "Sir, I must advise against this course of action. The probability of success is approximately twelve percent.",
        "emotional": "I'm afraid I have some troubling news. The reactor core is destabilizing, and we have less than four minutes to evacuate.",
    }

    # Warmup generation (discard)
    print("\nWarmup...")
    model.generate(text="Testing.", audio_prompt_path=str(ref1), exaggeration=0.5)
    print("  Done.")

    # ---------------------------------------------------------------
    # Trial 1: Reference length comparison (exag=0.5, default settings)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRIAL 1: Reference Length (exag=0.5)")
    print("=" * 60)

    for ref_name, ref_path in [("5s", ref_short), ("15s", ref_medium), ("25s", ref_long)]:
        for prompt_name, prompt in prompts.items():
            run_trial(
                model,
                trial_name=f"t1_{ref_name}_{prompt_name}",
                prompt=prompt,
                ref_path=ref_path,
                exaggeration=0.5,
                output_dir=output_dir,
            )

    # ---------------------------------------------------------------
    # Trial 2: Exaggeration sweep (using 15s ref, best balance)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRIAL 2: Exaggeration Sweep (15s ref)")
    print("=" * 60)

    for exag in [0.2, 0.3, 0.5, 0.7, 1.0]:
        run_trial(
            model,
            trial_name=f"t2_exag{exag:.1f}_medium",
            prompt=prompts["medium"],
            ref_path=ref_medium,
            exaggeration=exag,
            output_dir=output_dir,
        )

    # ---------------------------------------------------------------
    # Trial 3: Temperature sweep (15s ref, exag=0.3)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRIAL 3: Temperature Sweep (15s ref, exag=0.3)")
    print("=" * 60)

    for temp in [0.6, 0.8, 1.0]:
        run_trial(
            model,
            trial_name=f"t3_temp{temp:.1f}_medium",
            prompt=prompts["medium"],
            ref_path=ref_medium,
            exaggeration=0.3,
            temperature=temp,
            output_dir=output_dir,
        )

    # ---------------------------------------------------------------
    # Trial 4: CFG weight sweep (15s ref, exag=0.3, temp=0.8)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRIAL 4: CFG Weight Sweep (15s ref, exag=0.3)")
    print("=" * 60)

    for cfg in [0.3, 0.5, 0.7]:
        run_trial(
            model,
            trial_name=f"t4_cfg{cfg:.1f}_medium",
            prompt=prompts["medium"],
            ref_path=ref_medium,
            exaggeration=0.3,
            cfg_weight=cfg,
            output_dir=output_dir,
        )

    # Summary
    print("\n" + "=" * 60)
    print("ALL TRIALS COMPLETE")
    print(f"Output: {output_dir}")
    print()
    print("Evaluation guide:")
    print("  Trial 1: Compare 5s vs 15s vs 25s reference -- which sounds most like JARVIS?")
    print("  Trial 2: Compare exaggeration 0.2-1.0 -- which reduces accent drift?")
    print("  Trial 3: Compare temperature 0.6-1.0 -- which sounds most natural?")
    print("  Trial 4: Compare CFG 0.3-0.7 -- which balances quality vs adherence?")
    print("=" * 60)


if __name__ == "__main__":
    main()
