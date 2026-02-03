import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:logger/logger.dart';
import '../models/message_model.dart';
import '../models/websocket_message.dart';
import '../services/websocket_service.dart';
import '../services/audio_recording_service.dart';
import '../services/audio_playback_service.dart';
import 'websocket_provider.dart';
import 'audio_provider.dart';

/// Chat State
class ChatState {
  final List<ChatMessage> messages;
  final bool isProcessing;
  final String? currentAiResponse;

  ChatState({
    this.messages = const [],
    this.isProcessing = false,
    this.currentAiResponse,
  });

  ChatState copyWith({
    List<ChatMessage>? messages,
    bool? isProcessing,
    String? currentAiResponse,
  }) {
    return ChatState(
      messages: messages ?? this.messages,
      isProcessing: isProcessing ?? this.isProcessing,
      currentAiResponse: currentAiResponse ?? this.currentAiResponse,
    );
  }
}

/// Chat Notifier (using Riverpod 3.x Notifier class)
class ChatNotifier extends Notifier<ChatState> {
  late WebSocketService _wsService;
  late AudioRecordingService _audioRecordingService;
  late AudioPlaybackService _audioService;
  final Logger _logger = Logger();

  @override
  ChatState build() {
    // Get services from providers
    _wsService = ref.read(websocketServiceProvider);
    _audioRecordingService = ref.read(audioRecordingServiceProvider);
    _audioService = ref.read(audioPlaybackServiceProvider);

    // Listen to WebSocket messages
    _listenToWebSocket();

    // Listen for complete audio recordings
    _listenToCompleteAudio();

    return ChatState();
  }

  /// Listen for complete audio from recording service and send to backend
  void _listenToCompleteAudio() {
    _audioRecordingService.audioCompleteStream.listen((completeAudio) {
      // Send complete audio buffer to backend for Whisper transcription
      _wsService.sendAudioBytes(completeAudio);
      _logger.i('📤 Sent complete audio to backend: ${completeAudio.length} bytes');
    });
  }

  /// Listen to WebSocket messages
  void _listenToWebSocket() {
    _wsService.messageStream.listen((message) {
      _handleWebSocketMessage(message);
    });
  }

  /// Handle incoming WebSocket messages
  void _handleWebSocketMessage(WebSocketMessage message) {
    switch (message.type) {
      case WsMessageType.transcription:
        // User's transcribed speech
        if (message.text != null) {
          addMessage(ChatMessage.user(message.text!));
        }
        break;

      case WsMessageType.llmToken:
        // Streaming AI response token
        if (message.token != null) {
          _appendToCurrentResponse(message.token!);
        }
        break;

      case WsMessageType.llmResponse:
        // Complete AI response
        if (message.text != null) {
          _finalizeAiResponse(message.text!);
        }
        break;

      case WsMessageType.audioChunk:
        // Audio chunk received - play it
        if (message.audio != null && message.sampleRate != null) {
          // Backend sends float32 values [-1.0, 1.0] as a list
          // Keep them as doubles for proper conversion to PCM16
          final audioData = message.audio!.map((e) => (e as num).toDouble()).toList();
          _audioService.playAudioChunkFloat32(audioData, message.sampleRate!);
          _logger.d('Playing audio chunk: ${message.chunkNum} (${audioData.length} samples)');
        }
        break;

      case WsMessageType.audioComplete:
        // Audio generation complete
        _logger.d('Audio generation complete');
        state = state.copyWith(isProcessing: false);
        break;

      case WsMessageType.status:
        // Status message
        if (message.message != null) {
          addMessage(ChatMessage.system(message.message!));
        }
        break;

      case WsMessageType.vadStatus:
        // VAD status update
        _logger.d('VAD status: ${message.status}');
        if (message.status == 'speech_started') {
          state = state.copyWith(isProcessing: true);
        }
        break;

      case WsMessageType.error:
        // Error message
        if (message.message != null) {
          addMessage(ChatMessage.error(message.message!));
          state = state.copyWith(isProcessing: false);
        }
        break;

      default:
        _logger.w('Unknown message type: ${message.type}');
    }
  }

  /// Add a message to the chat
  void addMessage(ChatMessage message) {
    state = state.copyWith(
      messages: [...state.messages, message],
    );
  }

  /// Send a text message
  void sendTextMessage(String text) {
    if (text.trim().isEmpty) return;

    // Add user message to UI
    addMessage(ChatMessage.user(text));

    // Send to backend
    _wsService.sendText(text);

    // Set processing state
    state = state.copyWith(isProcessing: true, currentAiResponse: '');
  }

  /// Append token to current AI response (streaming)
  void _appendToCurrentResponse(String token) {
    final current = state.currentAiResponse ?? '';
    state = state.copyWith(
      currentAiResponse: current + token,
      isProcessing: true,
    );
  }

  /// Finalize AI response
  void _finalizeAiResponse(String fullResponse) {
    // Clear streaming state and add final message in one atomic update
    // This prevents showing both the streaming preview and final message simultaneously
    final updatedMessages = [...state.messages, ChatMessage.ai(fullResponse)];

    state = state.copyWith(
      messages: updatedMessages,
      currentAiResponse: null,
      isProcessing: false,
    );
  }

  /// Clear chat history
  void clearHistory() {
    state = ChatState();
  }

  /// Interrupt current processing
  void interrupt() {
    _wsService.sendInterrupt();
    _audioService.stop(); // Stop any playing audio
    _audioRecordingService.stopRecording(); // Stop recording
    state = state.copyWith(
      isProcessing: false,
      currentAiResponse: null,
    );
  }

  /// Start voice input (start recording and streaming audio to backend)
  Future<void> startVoiceInput() async {
    await _audioRecordingService.startRecording();
    _logger.i('Started recording audio for backend Whisper');
  }

  /// Stop voice input (stop recording and send to backend)
  Future<void> stopVoiceInput() async {
    // Stop recording - this will trigger audioCompleteStream and send to backend
    await _audioRecordingService.stopRecording();
    _logger.i('Stopped recording - audio will be sent to backend');
  }
}

/// Chat Provider
final chatProvider = NotifierProvider<ChatNotifier, ChatState>(() {
  return ChatNotifier();
});
