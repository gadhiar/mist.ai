"""Quick test script for fine-tuned CSM-1B voice models.

Usage:
    python test_voice.py                          # Test with default lines
    python test_voice.py "Custom text to speak"   # Test with custom text
    python test_voice.py --profile cortana        # Use VoiceProfile system
    python test_voice.py --profile cortana "Hi"   # Profile + custom text
"""

import argparse
import sys
import os
import time
import torch
import torchaudio
import numpy as np
from safetensors.torch import load_file
from models import Model, ModelArgs
from generator import Generator, Segment
from huggingface_hub import hf_hub_download

MODEL_WEIGHTS = "finetuned_model/model.safetensors"
SPEAKER_ID = 0
OUTPUT_DIR = "test_output"

# Reference clips for voice anchoring -- model uses these to match the voice
REFERENCE_DIR = "D:/rajga/Downloads/Xbox 360 - Halo 3 - Character Voices - Cortana/Cortana"
REFERENCE_CLIPS = [
    # 050lb files -- AI-processed/radio-filtered Cortana voice
    ("050lb_030_cor.wav", "High Charity, the Prophet's Holy City is on-"),
    (
        "050lb_230_cor.wav",
        "High Charity, the Prophet's Holy City is on its way to Earth, with an army of flood.",
    ),
    (
        "050lb_240_cor.wav",
        "I can't tell you everything. It's not safe. The Gravemind, it knows I'm in the system.",
    ),
]

TEST_LINES = [
    "Good morning. I've reviewed the session notes from yesterday.",
    "The knowledge graph currently contains 847 nodes across 12 topic clusters.",
    "Understood. I'll update the self-model with that preference.",
    "Based on our recent conversations, you seem most productive in the late afternoon.",
    "The confidence score for that knowledge extraction is 0.73, which is above the threshold.",
]


def trim_silence(audio, sample_rate, threshold_db=-40, min_silence_ms=300):
    """Trim leading/trailing silence and collapse long internal pauses."""
    if audio.dim() == 1:
        audio_np = audio.cpu().numpy()
    else:
        audio_np = audio.squeeze(0).cpu().numpy()

    # Convert to dB
    amplitude = np.abs(audio_np)
    threshold = 10 ** (threshold_db / 20)

    # Find first and last non-silent sample
    non_silent = np.where(amplitude > threshold)[0]
    if len(non_silent) == 0:
        return audio

    start = max(0, non_silent[0] - int(0.05 * sample_rate))  # 50ms pad
    end = min(len(audio_np), non_silent[-1] + int(0.1 * sample_rate))  # 100ms pad
    trimmed = audio_np[start:end]

    # Collapse long internal silences
    min_silence_samples = int(min_silence_ms / 1000 * sample_rate)
    max_silence_samples = int(0.3 * sample_rate)  # Cap silence at 300ms

    result = []
    silence_count = 0
    for sample in trimmed:
        if abs(sample) < threshold:
            silence_count += 1
            if silence_count <= max_silence_samples:
                result.append(sample)
        else:
            silence_count = 0
            result.append(sample)

    return torch.tensor(np.array(result, dtype=np.float32))


def load_finetuned_model(weights_path, device="cuda"):
    """Load CSM-1B with fine-tuned weights, no torch.compile."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="model.safetensors")
    config = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )
    model = Model(config)
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict, strict=True)

    finetuned_dict = load_file(weights_path)
    model.load_state_dict(finetuned_dict, strict=False)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model.to(device=device, dtype=dtype)
    model.eval()

    return Generator(model)


def load_reference_segments(generator, ref_clips):
    """Load reference clips as context segments for voice anchoring."""
    segments = []
    for path, transcript in ref_clips:
        if not os.path.exists(path):
            print(f"  Reference clip not found: {path}")
            continue
        audio, sr = torchaudio.load(path)
        audio = torchaudio.functional.resample(
            audio.squeeze(0), orig_freq=sr, new_freq=generator.sample_rate
        )
        segments.append(
            Segment(
                text=transcript,
                speaker=SPEAKER_ID,
                audio=audio,
            )
        )
        print(
            f"  Loaded reference: {os.path.basename(path)} ({len(audio)/generator.sample_rate:.1f}s)"
        )
    return segments


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned CSM-1B voice models")
    parser.add_argument("text", nargs="?", help="Custom text to speak")
    parser.add_argument("--weights", default=None, help="Path to model weights")
    parser.add_argument(
        "--profile", default=None, help="Voice profile name (loads from data/voice_profiles/)"
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lines = [args.text] if args.text else TEST_LINES

    if args.profile:
        # Load via VoiceProfile system
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from src.multimodal.voice_profile import VoiceProfileRegistry
        from pathlib import Path

        project_root = Path(__file__).parent.parent.parent
        registry = VoiceProfileRegistry(project_root / "data" / "voice_profiles", project_root)
        profile = registry.get(args.profile)
        print(f"Using voice profile: {profile.name} ({profile.description})")

        weights_path = str(profile.weights_path)
        ref_clips = [(str(c.audio_path), c.transcription) for c in profile.reference_clips]
    else:
        # Standalone mode
        weights_path = args.weights or MODEL_WEIGHTS
        profile_ref_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "voice_profiles", "cortana"
        )
        if os.path.isdir(profile_ref_dir) and os.path.exists(
            os.path.join(profile_ref_dir, REFERENCE_CLIPS[0][0])
        ):
            ref_clips = [(os.path.join(profile_ref_dir, f), t) for f, t in REFERENCE_CLIPS]
        else:
            ref_clips = [(os.path.join(REFERENCE_DIR, f), t) for f, t in REFERENCE_CLIPS]

    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}")
        sys.exit(1)

    print(f"Loading fine-tuned model from {weights_path}...")
    start = time.time()
    generator = load_finetuned_model(weights_path)
    print(f"Model loaded in {time.time() - start:.1f}s")

    print("\nLoading reference audio for voice anchoring...")
    context = load_reference_segments(generator, ref_clips)
    print(f"Loaded {len(context)} reference segments.")

    for i, text in enumerate(lines):
        print(f'\nGenerating [{i+1}/{len(lines)}]: "{text}"')
        gen_start = time.time()
        audio = generator.generate(
            text=text,
            speaker=SPEAKER_ID,
            context=context,
            max_audio_length_ms=15_000,
        )
        elapsed = time.time() - gen_start

        # Trim silence
        audio_trimmed = trim_silence(audio, generator.sample_rate)
        trim_pct = (1 - len(audio_trimmed) / len(audio)) * 100

        out_path = os.path.join(OUTPUT_DIR, f"voice_test_{i:02d}.wav")
        torchaudio.save(out_path, audio_trimmed.unsqueeze(0).cpu(), generator.sample_rate)
        print(f"  Saved: {out_path} ({elapsed:.1f}s, trimmed {trim_pct:.0f}% silence)")

    print(f"\nAll files saved to {OUTPUT_DIR}/")
    print("Play them to evaluate voice quality.")


if __name__ == "__main__":
    main()
