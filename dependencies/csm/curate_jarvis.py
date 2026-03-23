"""JARVIS voice dataset curation pipeline.

Processes 1,550 extracted WAVs from Marvel's Avengers through resampling,
transcription, audio quality scoring, and text scoring to produce a curated
training dataset for CSM-1B LoRA fine-tuning.
"""

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torchaudio


SOURCE_DIR = Path(r"D:\Games\Marvel's Avengers\JarvisAudio\wav")
OUTPUT_DIR = Path(r"D:\Users\rajga\mist.ai\dependencies\csm\audio_data_jarvis")
REFERENCE_DIR = Path(r"D:\Users\rajga\mist.ai\data\voice_profiles\jarvis\reference")
REPORT_PATH = Path(r"D:\Games\Marvel's Avengers\JarvisAudio\curation_report.json")
TARGET_SR = 24000

# Thresholds
MIN_DURATION = 1.5
MAX_DURATION = 12.0
MIN_SNR_DB = 15.0
MAX_SILENCE_RATIO = 0.5


def load_and_resample(wav_path: Path) -> tuple[np.ndarray, int]:
    """Load WAV and resample to TARGET_SR. Returns (samples_float32, sample_rate)."""
    waveform, sr = torchaudio.load(str(wav_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    samples = waveform.squeeze(0).numpy()
    # Normalize to [-1, 1]
    peak = np.abs(samples).max()
    if peak > 0:
        samples = samples / peak
    return samples, TARGET_SR


def compute_duration(samples: np.ndarray, sr: int) -> float:
    """Duration in seconds."""
    return len(samples) / sr


def compute_snr_db(samples: np.ndarray, sr: int) -> float:
    """Estimate SNR by comparing speech RMS to noise floor RMS.

    Uses a simple energy-based VAD: frames above median energy are speech,
    frames below 25th percentile are noise.
    """
    frame_len = int(0.025 * sr)  # 25ms frames
    hop = int(0.010 * sr)  # 10ms hop
    frames = [samples[i : i + frame_len] for i in range(0, len(samples) - frame_len, hop)]
    if not frames:
        return 0.0
    energies = np.array([np.sqrt(np.mean(f**2)) for f in frames])
    energies = np.maximum(energies, 1e-10)

    median_energy = np.median(energies)
    speech_rms = np.mean(energies[energies >= median_energy])
    noise_floor = np.mean(energies[energies <= np.percentile(energies, 25)])

    if noise_floor < 1e-10:
        return 60.0  # Effectively silent noise floor
    return 20 * np.log10(speech_rms / noise_floor)


def compute_spectral_flatness(samples: np.ndarray, sr: int) -> float:
    """Mean spectral flatness. High = noise/effects, low = tonal/speech."""
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    flatness_values = []
    for i in range(0, len(samples) - frame_len, hop):
        frame = samples[i : i + frame_len]
        spectrum = np.abs(np.fft.rfft(frame))
        spectrum = np.maximum(spectrum, 1e-10)
        geo_mean = np.exp(np.mean(np.log(spectrum)))
        arith_mean = np.mean(spectrum)
        if arith_mean > 0:
            flatness_values.append(geo_mean / arith_mean)
    return float(np.mean(flatness_values)) if flatness_values else 0.0


def compute_rms_consistency(samples: np.ndarray, sr: int) -> float:
    """Std dev of frame-level RMS. High = dynamic/combat, low = steady speech."""
    frame_len = int(0.050 * sr)  # 50ms frames
    hop = int(0.025 * sr)
    rms_values = [
        np.sqrt(np.mean(samples[i : i + frame_len] ** 2))
        for i in range(0, len(samples) - frame_len, hop)
    ]
    return float(np.std(rms_values)) if rms_values else 0.0


def compute_silence_ratio(samples: np.ndarray, sr: int, threshold_db: float = -40.0) -> float:
    """Fraction of frames below threshold_db."""
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    threshold_linear = 10 ** (threshold_db / 20.0)
    silent = 0
    total = 0
    for i in range(0, len(samples) - frame_len, hop):
        rms = np.sqrt(np.mean(samples[i : i + frame_len] ** 2))
        if rms < threshold_linear:
            silent += 1
        total += 1
    return silent / total if total > 0 else 0.0


def score_audio_quality(
    duration: float,
    snr_db: float,
    spectral_flatness: float,
    rms_std: float,
    silence_ratio: float,
    sf_threshold: float,
    rms_threshold: float,
) -> tuple[float, list[str]]:
    """Compute normalized 0-1 audio quality score. Returns (score, rejection_reasons).

    A score of 0 means rejected (hard filter). Score > 0 means accepted with quality ranking.
    """
    reasons = []

    # Hard filters (reject)
    if duration < MIN_DURATION:
        reasons.append(f"too_short:{duration:.1f}s")
        return 0.0, reasons
    if duration > MAX_DURATION:
        reasons.append(f"too_long:{duration:.1f}s")
        return 0.0, reasons
    if snr_db < MIN_SNR_DB:
        reasons.append(f"low_snr:{snr_db:.1f}dB")
        return 0.0, reasons
    if silence_ratio > MAX_SILENCE_RATIO:
        reasons.append(f"too_silent:{silence_ratio:.2f}")
        return 0.0, reasons

    # Soft scoring (0-1 range for each metric, then average)
    # SNR: 15dB = 0, 40dB+ = 1
    snr_score = min(1.0, max(0.0, (snr_db - 15) / 25))
    # Spectral flatness: lower is better (0 = 1.0, threshold = 0)
    sf_score = max(0.0, 1.0 - (spectral_flatness / sf_threshold)) if sf_threshold > 0 else 1.0
    # RMS consistency: lower is better (0 = 1.0, threshold = 0)
    rms_score = max(0.0, 1.0 - (rms_std / rms_threshold)) if rms_threshold > 0 else 1.0
    # Silence: lower is better (0 = 1.0, 0.5 = 0)
    silence_score = max(0.0, 1.0 - (silence_ratio / MAX_SILENCE_RATIO))
    # Duration: prefer 2-8s (peak at 4s)
    if duration < 2:
        dur_score = (duration - MIN_DURATION) / (2 - MIN_DURATION)
    elif duration <= 8:
        dur_score = 1.0
    else:
        dur_score = max(0.0, 1.0 - (duration - 8) / (MAX_DURATION - 8))

    score = (snr_score + sf_score + rms_score + silence_score + dur_score) / 5.0
    return score, reasons


# ---------------------------------------------------------------------------
# Text content scoring
# ---------------------------------------------------------------------------

# Patterns indicating Whisper hallucination artifacts
_HALLUCINATION_PATTERNS = [
    re.compile(r"^\[.*\]$"),  # [music], [applause], etc.
    re.compile(r"^(\.+|,+|!+|\?+)$"),  # Punctuation-only
]


def _is_repeated_phrase(text: str) -> bool:
    """Detect Whisper hallucination loops (e.g. 'thank you thank you thank you')."""
    words = text.lower().split()
    if len(words) < 4:
        return False
    # Check if the same 1-3 word phrase repeats for >60% of the text
    for phrase_len in range(1, 4):
        if len(words) < phrase_len * 3:
            continue
        phrase = tuple(words[:phrase_len])
        repeats = 0
        for i in range(0, len(words) - phrase_len + 1, phrase_len):
            if tuple(words[i : i + phrase_len]) == phrase:
                repeats += 1
        total_groups = len(words) / phrase_len
        if total_groups > 0 and repeats / total_groups > 0.6:
            return True
    return False


def score_text_content(transcription: str) -> tuple[float, list[str]]:
    """Score transcription for completeness and clarity.

    Returns (score, reasons) where score 0.0 = rejected, >0.0 = accepted.
    Filters for utterance quality, NOT speaking style. JARVIS personality
    traits (dry humor, formal phrasing, wit) are all valid content.
    """
    reasons: list[str] = []
    text = transcription.strip()

    # Hard excludes
    if not text:
        reasons.append("empty_transcription")
        return 0.0, reasons

    words = text.split()
    word_count = len(words)

    if word_count == 1:
        reasons.append("single_word")
        return 0.0, reasons

    for pattern in _HALLUCINATION_PATTERNS:
        if pattern.match(text):
            reasons.append("hallucination_artifact")
            return 0.0, reasons

    if _is_repeated_phrase(text):
        reasons.append("repeated_phrase_hallucination")
        return 0.0, reasons

    # Soft scoring
    score = 0.5  # Baseline for accepted text

    # Penalize very short utterances (2-3 words)
    if word_count <= 3:
        score -= 0.15
        reasons.append(f"very_short:{word_count}w")

    # Boost complete sentences (>5 words)
    if word_count > 5:
        score += 0.15

    # Boost longer utterances (>10 words)
    if word_count > 10:
        score += 0.15

    # Boost multi-sentence utterances
    sentence_endings = len(re.findall(r"[.!?]", text))
    if sentence_endings >= 2:
        score += 0.1

    return min(1.0, max(0.0, score)), reasons


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------


def transcribe_all(wav_paths: list[Path], resampled_dir: Path) -> dict[str, str]:
    """Transcribe all WAVs using Whisper large-v3-turbo.

    Uses resampled 24kHz files for transcription. Returns a mapping of
    original filename (stem + .wav) to transcription text.
    """
    from transformers import pipeline as hf_pipeline

    print(f"[TRANSCRIBE] Loading whisper-large-v3-turbo model...")
    transcriber = hf_pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        device="cuda",
    )

    results: dict[str, str] = {}
    total = len(wav_paths)
    for i, wav_path in enumerate(wav_paths, 1):
        resampled_path = resampled_dir / wav_path.name
        if not resampled_path.exists():
            print(f"  [{i}/{total}] SKIP {wav_path.name} -- resampled file missing")
            continue
        if i % 50 == 0 or i == 1:
            print(f"  [{i}/{total}] Transcribing {wav_path.name}...")
        try:
            output = transcriber(str(resampled_path))
            text = output["text"].strip() if isinstance(output, dict) else ""
            results[wav_path.name] = text
        except Exception as e:
            print(f"  [{i}/{total}] ERROR {wav_path.name}: {e}")
            results[wav_path.name] = ""

    print(f"[TRANSCRIBE] Completed: {len(results)}/{total} files")
    return results


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------


def calibrate_thresholds(metrics: list[dict]) -> tuple[float, float]:
    """Compute soft ceilings from dataset distribution.

    Uses 90th percentile of spectral_flatness and rms_std as thresholds
    for the scoring function, so only the noisiest/most-inconsistent 10%
    receive heavy penalties.
    """
    sf_values = [m["spectral_flatness"] for m in metrics if "spectral_flatness" in m]
    rms_values = [m["rms_std"] for m in metrics if "rms_std" in m]

    sf_threshold = float(np.percentile(sf_values, 90)) if sf_values else 0.5
    rms_threshold = float(np.percentile(rms_values, 90)) if rms_values else 0.2

    print(f"[CALIBRATE] Spectral flatness 90th pctl: {sf_threshold:.4f}")
    print(f"[CALIBRATE] RMS std 90th pctl: {rms_threshold:.4f}")
    return sf_threshold, rms_threshold


# ---------------------------------------------------------------------------
# Reference clip selection
# ---------------------------------------------------------------------------


def select_reference_clips(ranked: list[dict], count: int = 5) -> list[dict]:
    """Select top reference clips for inference anchoring.

    Filters for clips between 3-6 seconds that do not end with a question
    mark, then picks the highest combined score entries.
    """
    candidates = [
        entry
        for entry in ranked
        if 3.0 <= entry["duration"] <= 6.0
        and not entry.get("transcription", "").strip().endswith("?")
        and entry["combined_score"] > 0
    ]
    # Sort by combined score descending
    candidates.sort(key=lambda x: x["combined_score"], reverse=True)
    selected = candidates[:count]
    print(
        f"[REFERENCE] Selected {len(selected)}/{count} reference clips "
        f"from {len(candidates)} candidates"
    )
    return selected


# ---------------------------------------------------------------------------
# LoRA cache generation
# ---------------------------------------------------------------------------


def generate_lora_cache(curated_dir: Path, transcriptions: dict[str, str]) -> None:
    """Generate transcription cache compatible with lora.py.

    lora.py builds cache keys as:
        f"{audio_file}_{file_stat.st_mtime}_{file_stat.st_size}"
    where audio_file comes from glob.glob(os.path.join(AUDIO_DIR, "*.wav")),
    producing relative paths like audio_data_jarvis/clip.wav (or backslash
    on Windows). We replicate that exact format.
    """
    cache: dict[str, str] = {}
    wav_files = sorted(curated_dir.glob("*.wav"))

    for wav_file in wav_files:
        transcription = transcriptions.get(wav_file.name, "")
        if not transcription:
            continue
        # Build key matching lora.py: os.path.join(AUDIO_DIR, "file.wav") + stats
        relative_path = os.path.join(curated_dir.name, wav_file.name)
        stat = wav_file.stat()
        cache_key = f"{relative_path}_{stat.st_mtime}_{stat.st_size}"
        cache[cache_key] = transcription

    cache_path = curated_dir / "transcription_cache.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    print(f"[CACHE] Wrote {len(cache)} entries to {cache_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full JARVIS curation pipeline.

    Stages:
        1. Resample all source WAVs to 24kHz (temp directory)
        2. Whisper transcription (with JSON cache)
        3. Audio quality metrics (on resampled files)
        4. Text content scoring and combined ranking
        5. Reference clip selection (separated from training)
        6. Copy curated files, generate lora cache, write report
    """
    source_files = sorted(SOURCE_DIR.glob("*.wav"))
    if not source_files:
        print(f"[ERROR] No WAV files found in {SOURCE_DIR}")
        return
    print(f"[PIPELINE] Found {len(source_files)} source WAVs in {SOURCE_DIR}")

    # ------------------------------------------------------------------
    # Stage 1: Resample all to 24kHz in a temp directory
    # ------------------------------------------------------------------
    resampled_dir = Path(tempfile.mkdtemp(prefix="jarvis_resampled_"))
    print(f"\n[STAGE 1] Resampling to {TARGET_SR}Hz -> {resampled_dir}")
    for i, src in enumerate(source_files, 1):
        dst = resampled_dir / src.name
        if i % 100 == 0 or i == 1:
            print(f"  [{i}/{len(source_files)}] Resampling {src.name}...")
        try:
            waveform, sr = torchaudio.load(str(src))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != TARGET_SR:
                waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
            torchaudio.save(str(dst), waveform, TARGET_SR)
        except Exception as e:
            print(f"  [{i}/{len(source_files)}] ERROR resampling {src.name}: {e}")
    resampled_count = len(list(resampled_dir.glob("*.wav")))
    print(f"[STAGE 1] Resampled {resampled_count}/{len(source_files)} files")

    # ------------------------------------------------------------------
    # Stage 2: Whisper transcription (with JSON caching)
    # ------------------------------------------------------------------
    cache_path = SOURCE_DIR.parent / "transcription_cache.json"
    transcriptions: dict[str, str] = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                transcriptions = json.load(f)
            print(f"\n[STAGE 2] Loaded {len(transcriptions)} cached transcriptions")
        except (json.JSONDecodeError, OSError) as e:
            print(f"\n[STAGE 2] Cache load failed ({e}), transcribing all")
            transcriptions = {}

    uncached = [p for p in source_files if p.name not in transcriptions]
    if uncached:
        print(f"[STAGE 2] Transcribing {len(uncached)} uncached files...")
        new_transcriptions = transcribe_all(uncached, resampled_dir)
        transcriptions.update(new_transcriptions)
        # Save updated cache
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(transcriptions, f, indent=2, ensure_ascii=False)
        print(f"[STAGE 2] Cache saved to {cache_path}")
    else:
        print(f"\n[STAGE 2] All {len(transcriptions)} transcriptions cached, skipping")

    # ------------------------------------------------------------------
    # Stage 3: Audio quality metrics
    # ------------------------------------------------------------------
    print(f"\n[STAGE 3] Computing audio quality metrics...")
    all_metrics: list[dict] = []
    resampled_files = sorted(resampled_dir.glob("*.wav"))
    for i, rfile in enumerate(resampled_files, 1):
        if i % 100 == 0 or i == 1:
            print(f"  [{i}/{len(resampled_files)}] Analyzing {rfile.name}...")
        try:
            samples, sr = load_and_resample(rfile)
            duration = compute_duration(samples, sr)
            snr_db = compute_snr_db(samples, sr)
            spectral_flatness = compute_spectral_flatness(samples, sr)
            rms_std = compute_rms_consistency(samples, sr)
            silence_ratio = compute_silence_ratio(samples, sr)
            all_metrics.append(
                {
                    "filename": rfile.name,
                    "duration": duration,
                    "snr_db": snr_db,
                    "spectral_flatness": spectral_flatness,
                    "rms_std": rms_std,
                    "silence_ratio": silence_ratio,
                }
            )
        except Exception as e:
            print(f"  [{i}/{len(resampled_files)}] ERROR {rfile.name}: {e}")
    print(f"[STAGE 3] Computed metrics for {len(all_metrics)} files")

    # ------------------------------------------------------------------
    # Stage 4: Score, rank, filter
    # ------------------------------------------------------------------
    print(f"\n[STAGE 4] Scoring and ranking...")
    sf_threshold, rms_threshold = calibrate_thresholds(all_metrics)

    ranked: list[dict] = []
    rejected_count = 0
    for entry in all_metrics:
        audio_score, audio_reasons = score_audio_quality(
            entry["duration"],
            entry["snr_db"],
            entry["spectral_flatness"],
            entry["rms_std"],
            entry["silence_ratio"],
            sf_threshold,
            rms_threshold,
        )
        transcription = transcriptions.get(entry["filename"], "")
        text_score, text_reasons = score_text_content(transcription)

        if audio_score == 0.0 or text_score == 0.0:
            rejected_count += 1
            continue

        combined = 0.6 * audio_score + 0.4 * text_score
        ranked.append(
            {
                **entry,
                "transcription": transcription,
                "audio_score": round(audio_score, 4),
                "text_score": round(text_score, 4),
                "combined_score": round(combined, 4),
                "audio_reasons": audio_reasons,
                "text_reasons": text_reasons,
            }
        )

    ranked.sort(key=lambda x: x["combined_score"], reverse=True)
    print(f"[STAGE 4] Accepted: {len(ranked)}, Rejected: {rejected_count}")

    if not ranked:
        print("[ERROR] No clips passed curation filters. Check thresholds.")
        return

    # ------------------------------------------------------------------
    # Stage 5: Reference clip selection
    # ------------------------------------------------------------------
    print(f"\n[STAGE 5] Selecting reference clips...")
    reference_clips = select_reference_clips(ranked)

    # Remove reference clips from training set
    ref_filenames = {clip["filename"] for clip in reference_clips}
    training_clips = [r for r in ranked if r["filename"] not in ref_filenames]
    print(f"[STAGE 5] Training: {len(training_clips)}, Reference: {len(reference_clips)}")

    # ------------------------------------------------------------------
    # Stage 6: Copy files, generate cache, write report
    # ------------------------------------------------------------------
    print(f"\n[STAGE 6] Copying curated files...")

    # Prepare output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    # Copy training clips
    training_transcriptions: dict[str, str] = {}
    for clip in training_clips:
        src = resampled_dir / clip["filename"]
        dst = OUTPUT_DIR / clip["filename"]
        if src.exists():
            shutil.copy2(str(src), str(dst))
            training_transcriptions[clip["filename"]] = clip["transcription"]
    print(f"  Copied {len(training_clips)} training clips to {OUTPUT_DIR}")

    # Copy reference clips
    for clip in reference_clips:
        src = resampled_dir / clip["filename"]
        dst = REFERENCE_DIR / clip["filename"]
        if src.exists():
            shutil.copy2(str(src), str(dst))
    print(f"  Copied {len(reference_clips)} reference clips to {REFERENCE_DIR}")

    # Generate lora-compatible transcription cache
    generate_lora_cache(OUTPUT_DIR, training_transcriptions)

    # Write curation report
    report = {
        "source_dir": str(SOURCE_DIR),
        "total_source_files": len(source_files),
        "total_resampled": resampled_count,
        "total_accepted": len(ranked),
        "total_rejected": rejected_count,
        "training_clips": len(training_clips),
        "reference_clips": len(reference_clips),
        "thresholds": {
            "spectral_flatness_90pctl": sf_threshold,
            "rms_std_90pctl": rms_threshold,
            "min_duration": MIN_DURATION,
            "max_duration": MAX_DURATION,
            "min_snr_db": MIN_SNR_DB,
            "max_silence_ratio": MAX_SILENCE_RATIO,
        },
        "score_distribution": {
            "min": ranked[-1]["combined_score"],
            "max": ranked[0]["combined_score"],
            "median": ranked[len(ranked) // 2]["combined_score"],
        },
        "reference_clip_details": [
            {
                "filename": c["filename"],
                "duration": c["duration"],
                "combined_score": c["combined_score"],
                "transcription": c["transcription"],
            }
            for c in reference_clips
        ],
        "all_clips": [
            {
                "filename": c["filename"],
                "duration": round(c["duration"], 2),
                "audio_score": c["audio_score"],
                "text_score": c["text_score"],
                "combined_score": c["combined_score"],
                "transcription": c["transcription"],
            }
            for c in ranked
        ],
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Report written to {REPORT_PATH}")

    # Spot-check: print top 5 and bottom 5
    print(f"\n{'=' * 70}")
    print("SPOT CHECK -- Top 5 clips:")
    print(f"{'=' * 70}")
    for clip in ranked[:5]:
        print(f"  {clip['filename']}")
        print(
            f"    Score: {clip['combined_score']} "
            f"(audio={clip['audio_score']}, text={clip['text_score']})"
        )
        print(f"    Duration: {clip['duration']:.1f}s | SNR: {clip['snr_db']:.1f}dB")
        print(f"    Text: {clip['transcription'][:80]}")

    print(f"\nBottom 5 clips:")
    for clip in ranked[-5:]:
        print(f"  {clip['filename']}")
        print(
            f"    Score: {clip['combined_score']} "
            f"(audio={clip['audio_score']}, text={clip['text_score']})"
        )
        print(f"    Duration: {clip['duration']:.1f}s | SNR: {clip['snr_db']:.1f}dB")
        print(f"    Text: {clip['transcription'][:80]}")

    print(f"\n{'=' * 70}")
    print(f"[DONE] Pipeline complete.")
    print(f"  Training clips: {len(training_clips)} in {OUTPUT_DIR}")
    print(f"  Reference clips: {len(reference_clips)} in {REFERENCE_DIR}")
    print(f"  Report: {REPORT_PATH}")
    print(f"{'=' * 70}")

    # Cleanup temp resampled dir
    shutil.rmtree(resampled_dir, ignore_errors=True)
    print(f"[CLEANUP] Removed temp directory {resampled_dir}")


if __name__ == "__main__":
    main()
