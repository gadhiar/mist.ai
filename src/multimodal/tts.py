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

from generator import load_csm_1b, Segment


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

        print(f"Loading Sesame CSM on {device}...")

        # Disable FX graph cache to avoid triton_key error on Windows
        # Only affects first-load compile time (~30s), NOT runtime performance
        import torch._inductor.config as inductor_config
        inductor_config.fx_graph_cache = False

        self.generator = load_csm_1b(device=device)
        self.device = device
        self.use_context = use_context
        self.context = []  # Voice context for consistency
        self.speaker_id = 0

        print("Sesame CSM loaded successfully!")

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
              temperature: float = 0.7, topk: int = 30, streaming: bool = True) -> torch.Tensor:
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

        # Update context with this generation (keep last 3 for voice consistency)
        if self.use_context and len(audio) > 0:
            segment = Segment(
                speaker=self.speaker_id,
                text=text,
                audio=audio
            )
            self.context.append(segment)
            # Keep only last 3 utterances to avoid context getting too long
            if len(self.context) > 3:
                self.context = self.context[-3:]

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
