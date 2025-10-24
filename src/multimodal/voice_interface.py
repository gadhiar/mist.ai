"""
Complete voice interface: STT -> LLM -> TTS
"""
import ollama
from src.multimodal.stt import WhisperSTT
from src.multimodal.tts import SesameTTS
from src.utils.cleanup import register_cleanup


class VoiceInterface:
    """Voice conversation interface"""

    def __init__(self, llm_model: str = "qwen2.5:32b-instruct"):
        """
        Initialize voice interface

        Args:
            llm_model: Ollama model name
        """
        print("Initializing voice interface...")
        print("This will take 1-2 minutes on first run (loading all models)...\n")

        # Initialize in order of loading time
        print("1/3 Loading Whisper STT...")
        self.stt = WhisperSTT(model_size="base")

        print("2/3 Loading Sesame CSM TTS...")
        self.tts = SesameTTS()

        # Warmup TTS (loads models to GPU, caches everything)
        print("   Warming up TTS (first generation caches models)...")
        self.tts.speak("Initialization complete.", play=False)

        self.llm_model = llm_model

        # Register cleanup to stop Ollama on exit
        register_cleanup()

        # Warmup LLM (loads model into memory)
        print("3/3 Warming up LLM (Qwen 32B)...")
        ollama.chat(model=self.llm_model, messages=[{'role': 'user', 'content': 'hi'}])

        print("\n✓ Voice interface ready! All models loaded and cached.\n")

    def converse(self, duration: int = 5, play_audio: bool = True) -> tuple[str, str]:
        """
        One conversation turn: listen -> think -> speak

        Args:
            duration: Recording duration in seconds
            play_audio: Whether to play audio immediately

        Returns:
            (user_text, ai_response)
        """
        # Listen
        user_text = self.stt.listen(duration=duration)
        print(f"You: {user_text}")

        # Think
        print("Thinking...")
        response = ollama.chat(
            model=self.llm_model,
            messages=[{
                'role': 'user',
                'content': user_text
            }]
        )
        ai_text = response['message']['content']
        print(f"AI: {ai_text}")

        # Speak
        print("Speaking...")
        self.tts.speak(ai_text, play=play_audio)

        return user_text, ai_text

    def continuous_conversation(self):
        """Continuous conversation loop"""
        print("\n=== Voice Conversation Mode ===")
        print("Say 'goodbye' or 'exit' to end the conversation\n")

        while True:
            try:
                user_text, ai_text = self.converse(duration=5)

                # Check for exit
                if any(word in user_text.lower() for word in ['goodbye', 'exit', 'quit', 'stop']):
                    print("\nEnding conversation. Goodbye!")
                    break

            except KeyboardInterrupt:
                print("\nConversation interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                break
