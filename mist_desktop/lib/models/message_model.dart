import 'package:uuid/uuid.dart';

/// Message Types
enum MessageType { user, ai, system, error }

/// Chat Message Model
class ChatMessage {
  final String id;
  final String text;
  final MessageType type;
  final DateTime timestamp;
  final List<double>? audioData;
  final int? audioSampleRate;
  final Map<String, dynamic>? metadata;

  ChatMessage({
    String? id,
    required this.text,
    required this.type,
    DateTime? timestamp,
    this.audioData,
    this.audioSampleRate,
    this.metadata,
  }) : id = id ?? const Uuid().v4(),
       timestamp = timestamp ?? DateTime.now();

  /// Create a user message
  factory ChatMessage.user(String text, {List<double>? audioData}) {
    return ChatMessage(
      text: text,
      type: MessageType.user,
      audioData: audioData,
    );
  }

  /// Create an AI message
  factory ChatMessage.ai(String text, {List<double>? audioData}) {
    return ChatMessage(text: text, type: MessageType.ai, audioData: audioData);
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
    List<double>? audioData,
    int? audioSampleRate,
    Map<String, dynamic>? metadata,
  }) {
    return ChatMessage(
      id: id ?? this.id,
      text: text ?? this.text,
      type: type ?? this.type,
      timestamp: timestamp ?? this.timestamp,
      audioData: audioData ?? this.audioData,
      audioSampleRate: audioSampleRate ?? this.audioSampleRate,
      metadata: metadata ?? this.metadata,
    );
  }
}
