import numpy as np
import torch
from typing import Callable, Dict, List
from collections import deque
class VoiceActivityDetector:
    def __init__(
        self,
        model,
        utils,
        sample_rate: int = 16000,
        threshold: float = 0.3,
        silence_duration: int = 45
    ):
        self.model = model
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.silence_duration = silence_duration
        
        # Get functions from utils
        self.get_speech_timestamps = utils[0]
        
        self.is_speaking = False
        self.silent_frames = 0
        self.frame_size = 512 if sample_rate == 16000 else 256  # Required by Silero VAD
        
        print(f"VAD initialized with threshold {threshold}, frame size {self.frame_size}, silence duration {silence_duration}")

    def reset(self) -> None:
        self.is_speaking   = False
        self.silent_frames = 0

        if hasattr(self.model, "reset_states"):
            self.model.reset_states()
        elif hasattr(self.model, "reset_state"):
            self.model.reset_state()
        else:
            for buf in ("h", "c"):
                if hasattr(self.model, buf):
                    getattr(self.model, buf).zero_()

    def process_audio_chunk(self, audio_chunk: np.ndarray) -> bool:
        # Prepare audio chunk
        if audio_chunk.ndim > 1:
            audio_chunk = np.mean(audio_chunk, axis=1)
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        
        # Process in chunks of the correct size
        speech_detected = False
        turn_ended = False
        
        speech_probs = []
        
        # Process audio in correct sized chunks for Silero VAD
        for i in range(0, len(audio_chunk), self.frame_size):
            # Get chunk of correct size
            chunk = audio_chunk[i:i+self.frame_size]
            
            # If we don't have enough samples, pad with zeros
            if len(chunk) < self.frame_size:
                chunk = np.pad(chunk, (0, self.frame_size - len(chunk)))
            
            # Convert to tensor
            audio_tensor = torch.tensor(chunk).to('cpu')
            
            # Get speech probability
            
            speech_prob = self.model(audio_tensor, self.sample_rate).item()
            
            speech_probs.append(speech_prob)
            
            # Update speaking state
            if speech_prob >= self.threshold:
                speech_detected = True
                self.silent_frames = 0
            else:
                if self.is_speaking:
                    self.silent_frames += 1
        
        # Print detailed speech detection information
        # print(f"Speech probabilities: {speech_probs}")
        # print(f"Speech detected: {speech_detected}, Current state: {self.is_speaking}")
        # print(f"Silent frames: {self.silent_frames}, Threshold: {self.silence_duration}")
        
        # Update speaking state based on all chunks
        if speech_detected:
            self.is_speaking = True
            self.silent_frames = 0
        elif self.is_speaking and self.silent_frames >= self.silence_duration:
            # Transition to not speaking if we've had enough silent frames
            self.is_speaking = False
            turn_ended = True
            print(f"Turn ended after {self.silent_frames} silent frames")
            self.silent_frames = 0
        
        return turn_ended


class AudioStreamProcessor:
    def __init__(
        self,
        model,
        utils,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        vad_threshold: float = 0.3,
        callbacks: Dict[str, Callable] = None,
        pre_speech_buffer_size: int = 10
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.pre_speech_buffer = deque(maxlen=pre_speech_buffer_size)
        # Ensure model is on CPU
        if hasattr(model, 'to'):
            model = model.to('cpu')
            
        self.vad = VoiceActivityDetector(
            model=model,
            utils=utils,
            sample_rate=sample_rate,
            threshold=vad_threshold,
            silence_duration=45  # Increased for better end detection
        )
        
        self.audio_buffer = []
        self.is_collecting = False
        self.callbacks = callbacks or {}
        self.silent_chunk_count = 0
        self.max_silent_chunks = 30  # Force end after this many silent chunks
        
        print(f"AudioStreamProcessor initialized with threshold: {vad_threshold}")
    
    def process_audio(self, audio_chunk: np.ndarray):
        # Always add to pre-speech buffer
        self.pre_speech_buffer.append(audio_chunk)
        
        if self.is_collecting:
            self.audio_buffer.append(audio_chunk)
        
        # Process with VAD
        is_turn_end = self.vad.process_audio_chunk(audio_chunk)
        
        # Start collecting on speech detection
        if self.vad.is_speaking and not self.is_collecting:
            self.is_collecting = True
            self.silent_chunk_count = 0
            # Include pre-speech buffer in the audio buffer
            self.audio_buffer = list(self.pre_speech_buffer)
            print(f"Speech started, beginning collection with {len(self.pre_speech_buffer)} pre-speech chunks")
            if "on_speech_start" in self.callbacks:
                self.callbacks["on_speech_start"]()
        
        # Count silent chunks when collecting but not speaking
        if self.is_collecting and not self.vad.is_speaking:
            self.silent_chunk_count += 1
            print(f"Silent chunk count: {self.silent_chunk_count}, max: {self.max_silent_chunks}")
            # Force end after too many silent chunks
            if self.silent_chunk_count >= self.max_silent_chunks:
                is_turn_end = True
                print(f"Forcing speech end after {self.silent_chunk_count} silent chunks")
        else:
            self.silent_chunk_count = 0
                
        # End collection on turn end
        if is_turn_end and self.is_collecting:
            print("Turn end detected, processing collected audio")
            self.is_collecting = False
            if self.audio_buffer:
                print(f"Audio buffer length: {len(self.audio_buffer)} chunks")
                print("Speech ended, processing collected audio")
                complete_audio = np.concatenate(self.audio_buffer)
                print(f"Complete audio length: {len(complete_audio)}")
                
                if "on_speech_end" in self.callbacks:
                    try:
                        print("Calling on_speech_end callback")
                        self.callbacks["on_speech_end"](complete_audio, self.sample_rate)
                        print("on_speech_end callback completed successfully")
                    except Exception as e:
                        print(f"Error in on_speech_end callback: {e}")
                
                # Clear buffer after processing
                self.audio_buffer = []
                self.silent_chunk_count = 0
    
    def reset(self):
        self.vad.reset()
        self.audio_buffer = []
        self.is_collecting = False
        self.silent_chunk_count = 0
        print("AudioStreamProcessor reset")