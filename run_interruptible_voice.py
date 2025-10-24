"""
Interruptible Voice Conversation

Demonstrates true conversational flow where you can interrupt the AI at any time.

Features:
- Continuous VAD-based microphone listening
- Interrupt AI mid-response by speaking
- Graceful OutputStream interruption
- Natural conversation flow

Usage:
    python run_interruptible_voice.py [--debug] [--vad-threshold 0.5]

    Speak at ANY time - even while AI is talking
    Say "goodbye", "exit", "quit", or "stop" to end
    Press Ctrl+C for emergency stop
"""
import argparse
from src.multimodal.voice_interface_interrupt import InterruptibleVoiceInterface


def main():
    parser = argparse.ArgumentParser(description="Interruptible voice conversation")
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--vad-threshold', type=float, default=0.5,
                       help='VAD threshold 0.0-1.0 (default: 0.5, lower=more sensitive)')
    args = parser.parse_args()

    print("="*70)
    print("MIST.AI - INTERRUPTIBLE VOICE CONVERSATION")
    print("="*70)
    print()
    print("Revolutionary Feature: INTERRUPT AT ANY TIME!")
    print()
    print("Configuration:")
    print("  • Microphone: Continuously listening with VAD")
    print(f"  • VAD Threshold: {args.vad_threshold} (sensitivity)")
    print("  • Interruption: Immediate OutputStream stop")
    print("  • Audio: Gap-free playback when not interrupted")
    print("  • STT: Whisper Base")
    print("  • LLM: Qwen 2.5 7B Instruct")
    print("  • TTS: Sesame CSM 1B (streaming)")
    print()
    print("How It Works:")
    print("  1. Microphone is ALWAYS listening (VAD detects speech)")
    print("  2. Speak at ANY time - even while AI is talking")
    print("  3. AI stops immediately when you start speaking")
    print("  4. Your new input is processed right away")
    print()
    print("Commands:")
    print("  • Just speak naturally - no button press needed")
    print("  • Say 'goodbye', 'exit', 'quit', or 'stop' to end")
    print("  • Press Ctrl+C for emergency stop")
    print()
    if args.debug:
        print("⚙ Debug mode: Detailed logging enabled")
        print()
    print("="*70)
    print()

    try:
        # Initialize interruptible voice interface
        vi = InterruptibleVoiceInterface(vad_threshold=args.vad_threshold)

        print("="*70)
        print("STARTING INTERRUPTIBLE CONVERSATION")
        print("="*70)
        print("🎤 Microphone is now listening continuously...")
        print("Speak whenever you want - no need to wait!")
        print()

        # Run continuous conversation with interruption support
        vi.continuous_conversation(debug=args.debug)

    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("CONVERSATION INTERRUPTED")
        print("="*70)
        print("👋 Goodbye!")

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
