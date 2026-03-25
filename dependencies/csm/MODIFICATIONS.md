# CSM Streaming Modifications

This document tracks all modifications made to the original CSM Streaming codebase from https://github.com/davidbrowne17/csm-streaming

**Original License:** Apache License 2.0
**Modified By:** Raj Gadhia
**Project:** M.I.S.T Voice AI Assistant
**Last Updated:** 2025-01-28

---

## Modified Files

### 1. `generator.py`

#### Modification 1: Disabled torchaudio import
**Lines:** 13-16
**Reason:** PyTorch 2.7 nightly has incompatibilities with torchaudio on Windows
**Impact:** AudioStreamWriter.close() no longer saves to file automatically

```python
# ORIGINAL:
import torchaudio

# MODIFIED:
# DO NOT import torch audio for our generators
# Torch audio is included but we specifically disallow it here due to
# INCOMPATABILITY: with PyTorch Nightly 2.7
TORCHAUDIO_AVAILABLE = False
```

#### Modification 2: Position overflow protection (sliding window)
**Lines:** 233-237
**Reason:** Prevents CUDA assertion error when generating audio >80 seconds
**Impact:** Enables infinite-length generation by resetting position counter

**Problem:** The `curr_pos` counter increments indefinitely during generation, but the backbone KV cache is limited to 2048 positions (max_seq_len). After ~2000 frames (80+ seconds of audio), the position exceeds cache bounds causing `index out of bounds` CUDA assertion.

**Solution:** Sliding window approach - reset position to safe middle point (1024) when approaching limit (2000).

```python
# ADDED:
# FIX: Prevent position overflow beyond KV cache bounds (max_seq_len=2048)
# When position approaches limit, reset to safe middle position
# This implements a sliding window to support infinite-length generation
if curr_pos[0, 0].item() >= 2000:  # Leave safety margin
    curr_pos = torch.tensor([[1024]], device=curr_pos.device, dtype=curr_pos.dtype)
```

**Testing:** Successfully generated 3+ minute responses without crashes.

#### Modification 3: Conditional torchaudio.save()
**Lines:** 536-538
**Reason:** Only save to file if torchaudio is available
**Impact:** Prevents crashes when torchaudio is disabled

```python
# ORIGINAL:
torchaudio.save(self.filename, audio.unsqueeze(0).cpu(), self.sample_rate)

# MODIFIED:
if TORCHAUDIO_AVAILABLE:
    torchaudio.save(self.filename, audio.unsqueeze(0).cpu(), self.sample_rate)
```

#### Modification 4: torch.compile error handling
**Lines:** 762-770
**Reason:** torch.compile may fail on some Windows configurations
**Impact:** Model continues to work without compilation (still streams correctly)

```python
# ORIGINAL:
model.backbone = torch.compile(model.backbone,mode='reduce-overhead', fullgraph=True, backend='inductor')
model.decoder = torch.compile(model.decoder,mode='reduce-overhead', fullgraph=True, backend='inductor')

# MODIFIED:
try:
    print("Compiling model with torch.compile (PyTorch nightly)...")
    model.backbone = torch.compile(model.backbone, mode='reduce-overhead', fullgraph=True, backend='inductor')
    model.decoder = torch.compile(model.decoder, mode='reduce-overhead', fullgraph=True, backend='inductor')
    print("[SUCCESS] Model compilation successful")
except Exception as e:
    print(f"[WARNING] torch.compile failed: {e}")
    print("Continuing without compilation (streaming still works)")
```

---

### 2. `lora.py`

#### Modification 1: Optimized training hyperparameters
**Lines:** 36-51
**Reason:** Tuned for voice fine-tuning with Cortana dataset (277 clips)
**Impact:** Higher quality voice cloning with rank-96 LoRA

```python
# ORIGINAL:
NUM_EPOCHS = 5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-6
MAX_GRAD_NORM = 0.1
WARMUP_STEPS = 50
R=32          # LoRA rank
APLHA=32      # LoRA alpha

# MODIFIED:
NUM_EPOCHS = 15              # Optimized for Cortana dataset
BATCH_SIZE = 2               # Full GPU utilization (12GB VRAM available)
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 8
LEARNING_RATE = 5e-7         # Lower LR for stable training
MAX_GRAD_NORM = 0.5          # Increased for better gradient flow
WARMUP_STEPS = 150           # More warmup for longer training
R=96                         # LoRA rank for voice quality
APLHA=96                     # Keep scaling ratio at 1.0
MAX_AUDIO_FILES = 0          # Use all 277 clips from Cortana dataset
```

**Results:**
- Voice fine-tuning with Cortana dataset
- High-quality voice consistency
- Natural prosody and intonation

#### Modification 2: Changed view() to reshape()
**Lines:** 1045, 1047-1048, 1063, 1066-1067
**Reason:** Better compatibility with PyTorch operations
**Impact:** Same functionality, more robust tensor operations

```python
# ORIGINAL:
active_mask = label_masks[..., 0].view(-1)
active_logits = c0_logits.view(-1, c0_logits.size(-1))[active_mask]
active_labels = c0_labels.view(-1)[active_mask]

# MODIFIED:
active_mask = label_masks[..., 0].reshape(-1)
active_logits = c0_logits.reshape(-1, c0_logits.size(-1))[active_mask]
active_labels = c0_labels.reshape(-1)[active_mask]
```

---

## Unmodified Files

The following files remain unchanged from the original repository:
- `config.py`
- `loadandmergecheckpoint.py`
- `llm_interface.py`
- `rag_system.py`
- `models.py`
- `main.py`
- `setup.py`
- `run_csm.py`
- `vad.py`
- `test.py`
- `LICENSE` (Original Apache 2.0 license preserved)

---

## Attribution

Original CSM Streaming: https://github.com/davidbrowne17/csm-streaming
Original Authors: David Browne and contributors
Original License: Apache License 2.0

Modifications: Raj Gadhia (2025)
Modified for: M.I.S.T Voice AI Assistant

---

## Compliance with Apache 2.0 License

Per Apache License 2.0 Section 4(b):
> "You must cause any modified files to carry prominent notices stating that You changed the files"

This MODIFICATIONS.md file serves as the prominent notice of changes.

All original copyright, patent, trademark, and attribution notices from the Source form of the Work have been retained in the modified files.

The original LICENSE file (Apache 2.0) is preserved in this directory.
