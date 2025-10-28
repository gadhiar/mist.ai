import time
from generator import Segment, load_csm_1b, generate_streaming_audio
import torchaudio

print(f"Starting script at: {time.strftime('%H:%M:%S')}")
start_time = time.time()

print("Downloading model...")
model_start = time.time()
print(f"Model download completed in {time.time() - model_start:.2f} seconds")

print("Loading model to CUDA...")
load_start = time.time()
generator = load_csm_1b("cuda")
print(f"Model loaded in {time.time() - load_start:.2f} seconds")

speakers = [0, 1, 0, 0]
transcripts = [
    "Hey how are you doing.",
    "Pretty good, pretty good.",
    "I'm great.",
    "So happy to be speaking to you.",
]
audio_paths = [
    "utterance_0.wav",
    "utterance_1.wav",
    "utterance_2.wav",
    "utterance_3.wav",
]

def load_audio(audio_path):
    print(f"Loading reference audio: {audio_path}")
    audio_load_start = time.time()
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    print(f"Audio loaded and resampled in {time.time() - audio_load_start:.2f} seconds")
    return audio_tensor

print("Creating segments with reference audio...")
segments_start = time.time()
segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]
print(f"Segments created in {time.time() - segments_start:.2f} seconds")

# Option 1: Regular generation with streaming internally enabled
print("Generating audio (with internal streaming)...")
gen_start = time.time()
audio = generator.generate(
    text="Me too, this is some cool stuff huh?",
    speaker=0,
    context=segments,
    max_audio_length_ms=10_000,
    stream=True  # Enable internal streaming
)
print(f"Audio generation completed in {time.time() - gen_start:.2f} seconds")

print("Saving audio file...")
save_start = time.time()
torchaudio.save("audio_regular.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
print(f"Audio saved in {time.time() - save_start:.2f} seconds")

# Option 2: Use the streaming helper function that saves as it goes
print("Generating audio using streaming API...")
generate_streaming_audio(
    generator=generator,
    text="Me too, this is some cool stuff huh?",
    speaker=0,
    context=segments,
    output_file="audio_streamed.wav",
    max_audio_length_ms=10_000,
    play_audio=True  # Set to True to play audio in real-time (requires sounddevice package)
)

total_time = time.time() - start_time
print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"Script completed at: {time.strftime('%H:%M:%S')}")