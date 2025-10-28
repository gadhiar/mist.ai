"""
Text-to-Speech using Sesame CSM
"""
import sys
import os
import torch
import warnings

# Suppress harmless PyTorch warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.distributed')
warnings.filterwarnings('ignore', category=UserWarning, module='torch._inductor')

# Configure torch.compile to suppress errors and fall back to eager mode on Windows
# This prevents inductor backend failures while still getting speedups where possible
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# Increase cache size to handle dynamic tensor shapes in streaming TTS
# Default is 8, but streaming generates varying sequence lengths
torch._dynamo.config.cache_size_limit = 128

# Add CSM to path
sys.path.insert(0, 'dependencies/csm')

from generator import load_csm_1b_local, Segment


class SesameTTS:
    """Optimized wrapper for Sesame CSM text-to-speech with context persistence"""

    def __init__(self, device: str = None, use_context: bool = True):
        """
        Initialize Sesame CSM TTS

        Args:
            device: 'cuda' or 'cpu'. Auto-detects if None.
            use_context: If True, maintain voice context for consistency and speed
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading FINE-TUNED Sesame CSM (Epoch 20) on {device}...")

        # Disable FX graph cache to avoid triton_key error on Windows
        # Only affects first-load compile time (~30s), NOT runtime performance
        import torch._inductor.config as inductor_config
        inductor_config.fx_graph_cache = False

        # Load fine-tuned model from epoch 20 (best validation loss: 6.110)
        # Need absolute path for model loading
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        model_path = str(project_root / "dependencies" / "csm" / "finetuned_model" / "checkpoint-epoch-20")
        self.generator = load_csm_1b_local(model_path, device=device)
        self.device = device
        self.use_context = use_context
        self.context = []  # Voice context for consistency
        self.speaker_id = 0

        print("Sesame CSM loaded successfully!")

        # Load reference audio from Elise dataset for maximum voice consistency
        self._load_reference_audio(project_root)

    def _load_reference_audio(self, project_root):
        """
        Load reference audio clips from Elise dataset to initialize context.
        This provides the voice signature for consistent generation.
        """
        import torchaudio
        import json
        from pathlib import Path

        audio_dir = project_root / "dependencies" / "csm" / "audio_data"
        metadata_path = audio_dir / "dataset_metadata.json"

        if not metadata_path.exists():
            print("Warning: Elise dataset metadata not found. Voice consistency may be reduced.")
            return

        # Load metadata to get transcriptions
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Select 3 diverse reference clips from the dataset
        # Use clips 0, 5, and 10 for variety in phonetic content
        reference_indices = [0, 5, 10]

        print(f"Loading {len(reference_indices)} reference audio clips from Elise dataset...")

        for idx in reference_indices:
            if idx >= len(metadata['samples']):
                continue

            sample = metadata['samples'][idx]
            audio_path = audio_dir / sample['audio_file']

            if not audio_path.exists():
                print(f"Warning: Reference audio {audio_path} not found, skipping.")
                continue

            try:
                # Load and resample to 24kHz (CSM's native sample rate)
                audio, sr = torchaudio.load(str(audio_path))
                audio = audio.squeeze(0)  # Remove channel dimension

                if sr != 24000:
                    audio = torchaudio.functional.resample(audio, sr, 24000)

                # Create segment with transcription (lowercase to match generation preprocessing)
                segment = Segment(
                    speaker=self.speaker_id,
                    text=sample['text'].lower(),
                    audio=audio
                )

                self.context.append(segment)
                print(f"  Loaded: {sample['audio_file']} - \"{sample['text'][:50]}...\"")

            except Exception as e:
                print(f"Warning: Failed to load {audio_path}: {e}")

        if len(self.context) > 0:
            print(f"Successfully loaded {len(self.context)} reference clips for voice consistency.")
        else:
            print("Warning: No reference audio loaded. Voice consistency may be reduced.")

    def _estimate_audio_length(self, text: str) -> int:
        """
        Estimate audio length in ms based on text
        Assumes ~150 words/minute speaking rate
        """
        words = len(text.split())
        # 150 words/min = 2.5 words/sec = 400ms/word
        # Add 20% buffer
        estimated_ms = int(words * 400 * 1.2)
        # Clamp between 2s and 15s
        return max(2000, min(estimated_ms, 15000))

    def speak(self, text: str, output_path: str = None, play: bool = False,
              temperature: float = 0.55, topk: int = 20, streaming: bool = True) -> torch.Tensor:
        """
        Generate speech from text with optimized settings

        Args:
            text: Text to speak
            output_path: Optional path to save audio file
            play: Whether to play audio immediately (requires sounddevice)
            temperature: Sampling temperature (lower = faster, more deterministic)
            topk: Top-k sampling (lower = faster, more focused)
            streaming: Use streaming generation for lower latency

        Returns:
            Audio tensor (concatenated from all chunks if streaming)
        """
        # Estimate appropriate max length based on text
        max_length = self._estimate_audio_length(text)

        if streaming:
            # Streaming generation - yields chunks as they're generated
            audio_chunks = []
            for chunk in self.generator.generate_stream(
                text=text,
                speaker=self.speaker_id,
                context=self.context if self.use_context else [],
                max_audio_length_ms=max_length,
                temperature=temperature,
                topk=topk,
            ):
                audio_chunks.append(chunk)

            # Concatenate all chunks
            audio = torch.cat(audio_chunks, dim=0) if audio_chunks else torch.tensor([])
        else:
            # Non-streaming generation (fallback)
            audio = self.generator.generate(
                text=text,
                speaker=self.speaker_id,
                context=self.context if self.use_context else [],
                max_audio_length_ms=max_length,
                temperature=temperature,
                topk=topk,
            )

        # Update context with this generation (keep reference + last 2 for voice consistency)
        if self.use_context and len(audio) > 0:
            segment = Segment(
                speaker=self.speaker_id,
                text=text,
                audio=audio
            )
            self.context.append(segment)
            # Keep 3 reference clips + 2 most recent utterances (5 total)
            # Reference clips are always first in the list, so keep them protected
            # Reduced from 8 to 5 to prevent CUDA out of bounds with long responses
            if len(self.context) > 5:
                # Keep first 3 (reference) + last 2 (recent conversation)
                self.context = self.context[:3] + self.context[-2:]

        # Save if path provided
        if output_path and len(audio) > 0:
            import torchaudio  # Import only when saving
            torchaudio.save(output_path, audio.unsqueeze(0).cpu(), self.generator.sample_rate)
            print(f"Audio saved to: {output_path}")

        # Play if requested
        if play and len(audio) > 0:
            self._play_audio(audio)

        return audio

    def reset_context(self):
        """Reset voice context (useful for starting a new conversation)"""
        self.context = []

    def _play_audio(self, audio: torch.Tensor):
        """Play audio using sounddevice"""
        try:
            import sounddevice as sd
            sd.play(audio.cpu().numpy(), self.generator.sample_rate)
            sd.wait()
        except ImportError:
            print("Install sounddevice to play audio: pip install sounddevice")
