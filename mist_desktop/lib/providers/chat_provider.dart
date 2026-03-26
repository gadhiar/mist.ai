import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:logger/logger.dart';
import '../models/message_model.dart';
import '../models/websocket_message.dart';
import '../services/websocket_service.dart';
import 'websocket_provider.dart';

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
    bool clearCurrentAiResponse = false,
  }) {
    return ChatState(
      messages: messages ?? this.messages,
      isProcessing: isProcessing ?? this.isProcessing,
      currentAiResponse: clearCurrentAiResponse
          ? null
          : (currentAiResponse ?? this.currentAiResponse),
    );
  }
}

/// Chat Notifier (using Riverpod 3.x Notifier class)
class ChatNotifier extends Notifier<ChatState> {
  late WebSocketService _wsService;
  final Logger _logger = Logger();

  @override
  ChatState build() {
    _wsService = ref.read(websocketServiceProvider);

    final wsSub = _wsService.messageStream.listen((message) {
      _handleWebSocketMessage(message);
    });

    ref.onDispose(() {
      wsSub.cancel();
    });

    return ChatState();
  }

  /// Handle incoming WebSocket messages (non-audio types only)
  void _handleWebSocketMessage(WebSocketMessage message) {
    switch (message.type) {
      case WsMessageType.transcription:
        if (message.text != null) {
          addMessage(ChatMessage.user(message.text!));
        }
        break;

      case WsMessageType.llmToken:
        if (message.token != null) {
          _appendToCurrentResponse(message.token!);
        }
        break;

      case WsMessageType.llmResponse:
        if (message.text != null) {
          _finalizeAiResponse(message.text!);
        }
        break;

      case WsMessageType.status:
        if (message.message != null) {
          addMessage(ChatMessage.system(message.message!));
        }
        break;

      case WsMessageType.vadStatus:
        _logger.d('VAD status: ${message.status}');
        if (message.status == 'speech_started') {
          state = state.copyWith(isProcessing: true);
        }
        break;

      case WsMessageType.error:
        if (message.message != null) {
          addMessage(ChatMessage.error(message.message!));
          state = state.copyWith(isProcessing: false);
        }
        break;

      case WsMessageType.log:
      case WsMessageType.logConfigAck:
      case WsMessageType.logConfigError:
        // Handled by LogNotifier
        break;

      default:
        // audioChunk and audioComplete are handled by VoiceNotifier
        break;
    }
  }

  /// Add a message to the chat
  void addMessage(ChatMessage message) {
    state = state.copyWith(messages: [...state.messages, message]);
  }

  /// Send a text message
  void sendTextMessage(String text) {
    if (text.trim().isEmpty) return;

    addMessage(ChatMessage.user(text));
    _wsService.sendText(text);
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
    final updatedMessages = [...state.messages, ChatMessage.ai(fullResponse)];

    state = state.copyWith(
      messages: updatedMessages,
      clearCurrentAiResponse: true,
      isProcessing: false,
    );
  }

  /// Clear chat history
  void clearHistory() {
    state = ChatState();
  }

  /// Interrupt current processing (chat state only; caller should also stop voice)
  void interrupt() {
    _wsService.sendInterrupt();
    state = state.copyWith(isProcessing: false, clearCurrentAiResponse: true);
  }
}

/// Chat Provider
final chatProvider = NotifierProvider<ChatNotifier, ChatState>(() {
  return ChatNotifier();
});
