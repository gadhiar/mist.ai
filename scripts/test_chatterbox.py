"""Quick Chatterbox Turbo evaluation with JARVIS reference clip.

Usage:
    python scripts/test_chatterbox.py
"""

import sys
import time
from pathlib import Path

import soundfile as sf
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    print("=" * 60)
    print("CHATTERBOX TURBO - JARVIS VOICE EVAL")
    print("=" * 60)

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load reference clip
    ref_path = Path("data/voice_profiles/jarvis/reference/ref_001.wav")
    print(f"\nReference: {ref_path}")
    ref_audio, ref_sr = sf.read(str(ref_path))
    print(f"  Duration: {len(ref_audio)/ref_sr:.1f}s, Sample rate: {ref_sr}Hz")

    # Load Chatterbox
    print("\nLoading Chatterbox Turbo...")
    t0 = time.perf_counter()
    from chatterbox.tts import ChatterboxTTS

    model = ChatterboxTTS.from_pretrained(device=device)
    load_time = time.perf_counter() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    if device == "cuda":
        vram_used = torch.cuda.memory_allocated() / 1e9
        vram_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  VRAM allocated: {vram_used:.2f} GB")
        print(f"  VRAM reserved: {vram_reserved:.2f} GB")

    # Test prompts -- varying length and emotion
    test_prompts = [
        "Good evening, sir. All systems are operational.",
        "I've detected an anomaly in the lower levels. I recommend caution.",
        "Sir, I must advise against this course of action. The probability of success is approximately twelve percent.",
        "The Mark 42 suit is ready for deployment. Shall I initiate the launch sequence?",
    ]

    print("\n" + "-" * 60)
    print("GENERATION TESTS")
    print("-" * 60)

    output_dir = Path("data/chatterbox_eval")
    output_dir.mkdir(exist_ok=True)

    for i, prompt in enumerate(test_prompts):
        print(f'\n[{i+1}] "{prompt}"')

        # Reset VRAM tracking
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        wav = model.generate(
            text=prompt,
            audio_prompt_path=str(ref_path),
        )
        gen_time = time.perf_counter() - t0

        # Calculate RTF
        audio_dur = wav.shape[-1] / 24000  # Chatterbox outputs 24kHz
        rtf = gen_time / audio_dur

        print(f"  Generated: {audio_dur:.2f}s audio in {gen_time:.2f}s (RTF: {rtf:.3f}x)")

        if device == "cuda":
            peak_vram = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Peak VRAM: {peak_vram:.2f} GB")

        # Save output
        out_path = output_dir / f"jarvis_test_{i+1:02d}.wav"
        sf.write(str(out_path), wav.squeeze().cpu().numpy(), 24000)
        print(f"  Saved: {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"Output files in: {output_dir}")
    print("Listen to the WAV files to assess voice cloning quality.")
    print("=" * 60)


if __name__ == "__main__":
    main()
