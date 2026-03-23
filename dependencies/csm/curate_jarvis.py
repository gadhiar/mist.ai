"""JARVIS voice dataset curation pipeline.

Processes 1,550 extracted WAVs from Marvel's Avengers through resampling,
transcription, audio quality scoring, and text scoring to produce a curated
training dataset for CSM-1B LoRA fine-tuning.
"""

import json
import os
import shutil
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
