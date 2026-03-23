import 'package:uuid/uuid.dart';

/// Message Types
enum MessageType { user, ai, system, error }

/// Chat Message Model
class ChatMessage {
  final String id;
  final String text;
  final MessageType type;
  final DateTime timestamp;
  final Map<String, dynamic>? metadata;

  ChatMessage({
    String? id,
    required this.text,
    required this.type,
    DateTime? timestamp,
    this.metadata,
  }) : id = id ?? const Uuid().v4(),
       timestamp = timestamp ?? DateTime.now();

  /// Create a user message
  factory ChatMessage.user(String text) {
    return ChatMessage(text: text, type: MessageType.user);
  }

  /// Create an AI message
  factory ChatMessage.ai(String text) {
    return ChatMessage(text: text, type: MessageType.ai);
  }

  /// Create a system message
  factory ChatMessage.system(String text) {
    return ChatMessage(text: text, type: MessageType.system);
  }

  /// Create an error message
  factory ChatMessage.error(String text) {
    return ChatMessage(text: text, type: MessageType.error);
  }

  /// Copy with modifications
  ChatMessage copyWith({
    String? id,
    String? text,
    MessageType? type,
    DateTime? timestamp,
    Map<String, dynamic>? metadata,
  }) {
    return ChatMessage(
      id: id ?? this.id,
      text: text ?? this.text,
      type: type ?? this.type,
      timestamp: timestamp ?? this.timestamp,
      metadata: metadata ?? this.metadata,
    );
  }
}
