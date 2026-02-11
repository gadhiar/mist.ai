import 'dart:convert';

/// WebSocket Message Types (matching backend)
class WsMessageType {
  // From client
  static const String audio = 'audio';
  static const String text = 'text';
  static const String interrupt = 'interrupt';
  static const String resetVad = 'reset_vad';

  // From server
  static const String status = 'status';
  static const String vadStatus = 'vad_status';
  static const String transcription = 'transcription';
  static const String llmToken = 'llm_token';
  static const String llmResponse = 'llm_response';
  static const String audioChunk = 'audio_chunk';
  static const String audioComplete = 'audio_complete';
  static const String error = 'error';
}

/// WebSocket Message Model
class WebSocketMessage {
  final String type;
  final Map<String, dynamic> data;

  WebSocketMessage({required this.type, required this.data});

  /// Create from JSON string
  factory WebSocketMessage.fromJson(String jsonStr) {
    final json = jsonDecode(jsonStr) as Map<String, dynamic>;
    return WebSocketMessage(type: json['type'] as String, data: json);
  }

  /// Create from Map
  factory WebSocketMessage.fromMap(Map<String, dynamic> map) {
    return WebSocketMessage(type: map['type'] as String, data: map);
  }

  /// Convert to JSON string
  String toJson() {
    final map = {'type': type, ...data};
    return jsonEncode(map);
  }

  /// Convert to Map
  Map<String, dynamic> toMap() {
    return {'type': type, ...data};
  }

  // Factory constructors for client messages

  /// Audio message
  factory WebSocketMessage.audio({
    required List<double> audio,
    required int sampleRate,
  }) {
    return WebSocketMessage(
      type: WsMessageType.audio,
      data: {
        'type': WsMessageType.audio,
        'audio': audio,
        'sample_rate': sampleRate,
      },
    );
  }

  /// Text message
  factory WebSocketMessage.text(String text) {
    return WebSocketMessage(
      type: WsMessageType.text,
      data: {'type': WsMessageType.text, 'text': text},
    );
  }

  /// Interrupt message
  factory WebSocketMessage.interrupt() {
    return WebSocketMessage(
      type: WsMessageType.interrupt,
      data: {'type': WsMessageType.interrupt},
    );
  }

  /// Reset VAD message
  factory WebSocketMessage.resetVad() {
    return WebSocketMessage(
      type: WsMessageType.resetVad,
      data: {'type': WsMessageType.resetVad},
    );
  }

  // Getters for common fields

  String? get text => data['text'] as String?;
  String? get message => data['message'] as String?;
  String? get token => data['token'] as String?;
  String? get status => data['status'] as String?;
  List<dynamic>? get audio => data['audio'] as List<dynamic>?;
  int? get sampleRate => data['sample_rate'] as int?;
  int? get chunkNum => data['chunk_num'] as int?;
}
