"""Text-to-Speech using Sesame CSM."""

import sys
import warnings

import torch

# Suppress harmless PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")

# Configure torch.compile to suppress errors and fall back to eager mode on Windows
# This prevents inductor backend failures while still getting speedups where possible
import torch._dynamo

torch._dynamo.config.suppress_errors = True
# Increase cache size to handle dynamic tensor shapes in streaming TTS
# Default is 8, but streaming generates varying sequence lengths
torch._dynamo.config.cache_size_limit = 128

# Add CSM to path
sys.path.insert(0, "dependencies/csm")

from generator import Generator, Segment

from models import Model, ModelArgs


def _load_finetuned_csm(weights_path: str, device: str = "cuda") -> Generator:
    """Load CSM-1B base model and overlay fine-tuned weights.

    Fine-tuned model.safetensors files are raw weight dumps (not HF-formatted
    checkpoints), so we load the base model first then overlay fine-tuned weights
    with strict=False to allow partial weight updates (LoRA-merged weights).
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    # Load base model architecture and weights from HuggingFace
    base_path = hf_hub_download(repo_id="sesame/csm-1b", filename="model.safetensors")
    config = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )
    model = Model(config)
    base_dict = load_file(base_path)
    model.load_state_dict(base_dict, strict=True)

    # Overlay fine-tuned weights (strict=False for LoRA-merged partial updates)
    finetuned_dict = load_file(weights_path)
    model.load_state_dict(finetuned_dict, strict=False)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model.to(device=device, dtype=dtype)
    model.eval()

    return Generator(model)


class SesameTTS:
    """Optimized wrapper for Sesame CSM text-to-speech with context persistence."""

    def __init__(self, profile, device: str = None, use_context: bool = True):
        """Initialize Sesame CSM TTS.

        Args:
            profile: VoiceProfile containing weights path, reference clips, and config.
            device: 'cuda' or 'cpu'. Auto-detects if None.
            use_context: If True, maintain voice context for consistency and speed.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.profile = profile
        print(f"Loading Sesame CSM ({profile.name}) on {device}...")

        import torch._inductor.config as inductor_config

        inductor_config.fx_graph_cache = False

        self.generator = _load_finetuned_csm(str(profile.weights_path), device=device)
        self.device = device
        self.use_context = use_context
        self.context = []
        self.speaker_id = profile.speaker_id

        print(f"Sesame CSM ({profile.name}) loaded successfully!")

        self._load_reference_audio()

    def _load_reference_audio(self):
        """Load reference audio clips from voice profile for voice anchoring."""
        import torchaudio

        if not self.profile.reference_clips:
            print("Warning: No reference clips in voice profile. Voice consistency may be reduced.")
            return

        print(
            f"Loading {len(self.profile.reference_clips)} reference clips"
            f" from {self.profile.name} profile..."
        )

        for clip in self.profile.reference_clips:
            audio, sr = torchaudio.load(str(clip.audio_path))
            audio = audio.squeeze(0)

            if sr != self.profile.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.profile.sample_rate)

            segment = Segment(
                speaker=self.profile.speaker_id,
                text=clip.transcription.lower(),
                audio=audio,
            )
            self.context.append(segment)
            print(f'  Loaded: {clip.audio_path.name} - "{clip.transcription[:50]}..."')

        print(f"Loaded {len(self.context)} reference clips for voice consistency.")

    def _estimate_audio_length(self, text: str) -> int:
        """Estimate audio length in ms based on text
        Assumes ~150 words/minute speaking rate.
        """
        words = len(text.split())
        # ~450ms/word (133 words/min, moderate pace)
        # Add 50% buffer for pauses and natural speech variation
        estimated_ms = int(words * 450 * 1.5)
        # Clamp between 2s and 15s
        return max(2000, min(estimated_ms, 15000))

    def speak(
        self,
        text: str,
        output_path: str = None,
        play: bool = False,
        temperature: float = None,
        topk: int = None,
        streaming: bool = True,
    ) -> torch.Tensor:
        """Generate speech from text with optimized settings.

        Args:
            text: Text to speak
            output_path: Optional path to save audio file
            play: Whether to play audio immediately (requires sounddevice)
            temperature: Sampling temperature. Defaults to profile value.
            topk: Top-k sampling. Defaults to profile value.
            streaming: Use streaming generation for lower latency

        Returns:
            Audio tensor (concatenated from all chunks if streaming)
        """
        temperature = temperature if temperature is not None else self.profile.temperature
        topk = topk if topk is not None else self.profile.topk

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
            segment = Segment(speaker=self.speaker_id, text=text, audio=audio)
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
        """Reset voice context (useful for starting a new conversation)."""
        self.context = []

    def _play_audio(self, audio: torch.Tensor):
        """Play audio using sounddevice."""
        try:
            import sounddevice as sd

            sd.play(audio.cpu().numpy(), self.generator.sample_rate)
            sd.wait()
        except ImportError:
            print("Install sounddevice to play audio: pip install sounddevice")
