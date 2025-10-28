"""
CLI WebSocket Client for Voice AI Server

For testing the WebSocket server before building web frontend
"""
import asyncio
import json
import sys
import queue
import threading
import numpy as np
import sounddevice as sd
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import websockets
except ImportError:
    print("ERROR: websockets not installed")
    print("Run: pip install websockets")
    sys.exit(1)


class VoiceClient:
    """CLI client that connects to Voice AI WebSocket server"""

    def __init__(self, server_url="ws://localhost:8001/ws"):
        self.server_url = server_url
        self.websocket = None
        self.running = False

        # Audio settings
        self.mic_sample_rate = 16000
        self.mic_blocksize = 1024
        self.playback_sample_rate = 24000

        # Playback buffer (for incoming WebSocket messages)
        self.playback_queue = asyncio.Queue()

        # OutputStream pattern (like working version)
        self._playback_buffer = queue.Queue(maxsize=50)
        self._current_chunk = None
        self._chunk_position = 0
        self._stream = None
        self._interrupt_playback = threading.Event()  # Signal to stop playback

    async def connect(self):
        """Connect to WebSocket server"""
        print(f"Connecting to {self.server_url}...")
        self.websocket = await websockets.connect(self.server_url)
        print("Connected to server")

    async def send_audio_chunk(self, audio_data, sample_rate):
        """Send audio chunk to server"""
        await self.websocket.send(
            json.dumps(
                {
                    "type": "audio",
                    "audio": audio_data.tolist(),
                    "sample_rate": sample_rate,
                }
            )
        )

    async def send_text(self, text):
        """Send text message to server"""
        await self.websocket.send(json.dumps({"type": "text", "text": text}))

    async def send_interrupt(self):
        """Send interrupt signal"""
        await self.websocket.send(json.dumps({"type": "interrupt"}))

    async def receive_messages(self):
        """Receive and handle messages from server"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self.handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            print("\nConnection closed")
            self.running = False

    async def handle_message(self, data):
        """Handle different message types from server"""
        msg_type = data.get("type")

        if msg_type == "status":
            print(f"[Status] {data['message']}")

        elif msg_type == "vad_status":
            status = data.get("status")
            if status == "speech_started":
                print("[VAD] Speech detected - interrupting playback")
                # Stop current audio playback immediately
                self._interrupt_playback.set()

        elif msg_type == "transcription":
            text = data.get("text")
            print(f"\nYou: {text}")

        elif msg_type == "llm_token":
            token = data.get("token")
            print(token, end="", flush=True)

        elif msg_type == "llm_response":
            print()  # Newline after tokens

        elif msg_type == "audio_chunk":
            # Queue audio for playback
            audio_list = data.get("audio")
            sample_rate = data.get("sample_rate", 24000)
            chunk_num = data.get("chunk_num")

            if chunk_num == 1:
                print("\n[Audio] Playback starting...")
                # Signal new audio generation
                await self.playback_queue.put("START_NEW_AUDIO")

            audio_np = np.array(audio_list, dtype=np.float32)
            await self.playback_queue.put(audio_np)

        elif msg_type == "audio_complete":
            await self.playback_queue.put(None)  # Signal end
            print("Audio complete")
            print("-" * 60)

        elif msg_type == "error":
            print(f"\n[ERROR] {data.get('message')}")

        else:
            print(f"Unknown message type: {msg_type}")

    async def microphone_worker(self):
        """Capture audio from microphone and send to server"""
        print("[Mic] Starting...")

        # Save event loop reference for callback thread
        loop = asyncio.get_event_loop()

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Mic status: {status}")

            # Get audio chunk
            audio_chunk = indata[:, 0].copy()

            # Send to server (schedule in event loop)
            asyncio.run_coroutine_threadsafe(
                self.send_audio_chunk(audio_chunk, self.mic_sample_rate),
                loop,
            )

        # Start continuous input stream
        with sd.InputStream(
            samplerate=self.mic_sample_rate,
            channels=1,
            blocksize=self.mic_blocksize,
            callback=audio_callback,
            dtype=np.float32,
        ):
            print("[Mic] Active")
            while self.running:
                await asyncio.sleep(0.1)

        print("[Mic] Stopped")

    async def playback_worker(self):
        """
        Play audio chunks using OutputStream for gap-free playback
        Mimics the working voice_interface_interrupt.py pattern
        """
        print("[Playback] Starting OutputStream...")

        def audio_callback(outdata, frames, time_info, status):
            """Called by sounddevice when it needs more audio data"""
            if status:
                print(f"Stream status: {status}")

            # Check for interruption
            if self._interrupt_playback.is_set():
                # Stop playback immediately - fill with silence
                outdata[:, 0] = 0
                raise sd.CallbackStop()

            samples_needed = frames
            samples_written = 0

            while samples_written < samples_needed:
                # Get current chunk or fetch new one
                if self._current_chunk is None or self._chunk_position >= len(self._current_chunk):
                    try:
                        self._current_chunk = self._playback_buffer.get_nowait()
                        self._chunk_position = 0

                        # Check for stop signal
                        if self._current_chunk is None:
                            outdata[samples_written:, 0] = 0
                            raise sd.CallbackStop()

                    except queue.Empty:
                        # No audio available - fill with silence
                        outdata[samples_written:, 0] = 0
                        break

                # Copy audio from current chunk to output
                remaining_in_chunk = len(self._current_chunk) - self._chunk_position
                samples_to_copy = min(remaining_in_chunk, samples_needed - samples_written)

                outdata[samples_written:samples_written + samples_to_copy, 0] = \
                    self._current_chunk[self._chunk_position:self._chunk_position + samples_to_copy]

                self._chunk_position += samples_to_copy
                samples_written += samples_to_copy

        # Create OutputStream
        self._stream = sd.OutputStream(
            samplerate=self.playback_sample_rate,
            channels=1,
            blocksize=2048,
            callback=audio_callback,
            dtype=np.float32
        )

        print("[Playback] OutputStream created, waiting for audio...")

        # Track stream state per audio generation
        stream_started = False
        prebuffer_count = 0
        PREBUFFER_TARGET = 3  # Match working version

        while self.running:
            try:
                # Get audio chunk from WebSocket messages
                audio_chunk = await asyncio.wait_for(
                    self.playback_queue.get(), timeout=0.1
                )

                # Check for new audio generation signal (string, not array)
                if isinstance(audio_chunk, str) and audio_chunk == "START_NEW_AUDIO":
                    print("[Playback] New audio generation detected, resetting stream...")
                    # Stop previous stream if running
                    if stream_started and self._stream:
                        self._stream.stop()

                    # Reset state
                    stream_started = False
                    prebuffer_count = 0
                    self._interrupt_playback.clear()  # Clear interrupt flag for new audio

                    # Clear buffers
                    while not self._playback_buffer.empty():
                        try:
                            self._playback_buffer.get_nowait()
                        except queue.Empty:
                            break

                    self._current_chunk = None
                    self._chunk_position = 0
                    continue

                # Check for end signal
                if audio_chunk is None:
                    # End of audio signal
                    self._playback_buffer.put(None)
                    continue

                # Pre-buffer first 3 chunks before starting stream
                if not stream_started:
                    prebuffer_count += 1
                    self._playback_buffer.put(audio_chunk)

                    if prebuffer_count >= PREBUFFER_TARGET:
                        print(f"[Playback] Pre-buffered {prebuffer_count} chunks, starting stream...")
                        self._stream.start()
                        stream_started = True
                else:
                    # Stream already started, just feed chunks
                    self._playback_buffer.put(audio_chunk)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Playback error: {e}")

                # If stream was interrupted, handle gracefully
                if self._interrupt_playback.is_set():
                    print("[Playback] Stream interrupted by user speech")
                    # Clear buffers
                    while not self._playback_buffer.empty():
                        try:
                            self._playback_buffer.get_nowait()
                        except queue.Empty:
                            break
                    # Clear the playback queue too
                    while not self.playback_queue.empty():
                        try:
                            await asyncio.wait_for(self.playback_queue.get(), timeout=0.01)
                        except asyncio.TimeoutError:
                            break
                    # Stream will be restarted when new audio arrives

        # Cleanup
        if self._stream:
            self._stream.stop()
            self._stream.close()
        print("[Playback] Stopped")

    async def run(self):
        """Run the voice client"""
        try:
            await self.connect()
            self.running = True

            print("\n" + "=" * 60)
            print("VOICE AI CLIENT - WebSocket Mode")
            print("=" * 60)
            print("Speak to interact with the AI")
            print("Press Ctrl+C to exit")
            print("=" * 60 + "\n")

            # Start workers
            await asyncio.gather(
                self.receive_messages(),
                self.microphone_worker(),
                self.playback_worker(),
            )

        except KeyboardInterrupt:
            print("\n\nExiting...")
        except Exception as e:
            print(f"\n[ERROR] {e}")
        finally:
            self.running = False
            if self.websocket:
                await self.websocket.close()


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Voice AI WebSocket Client")
    parser.add_argument(
        "--server",
        default="ws://localhost:8000/ws",
        help="WebSocket server URL",
    )
    args = parser.parse_args()

    client = VoiceClient(server_url=args.server)
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())
