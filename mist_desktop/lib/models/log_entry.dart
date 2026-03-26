/// Structured log entry received from the backend via WebSocket.
class LogEntry {
  final DateTime timestamp;
  final String level;
  final int levelno;
  final String logger;
  final String? requestId;
  final String message;

  const LogEntry({
    required this.timestamp,
    required this.level,
    required this.levelno,
    required this.logger,
    required this.requestId,
    required this.message,
  });

  /// Parse a log entry from a WebSocket message JSON map.
  factory LogEntry.fromJson(Map<String, dynamic> json) {
    return LogEntry(
      timestamp: DateTime.parse(json['timestamp'] as String),
      level: json['level'] as String,
      levelno: json['levelno'] as int,
      logger: json['logger'] as String,
      requestId: json['request_id'] as String?,
      message: json['message'] as String,
    );
  }

  /// Returns the last segment after the last dot, or the full name if no dots.
  String get shortLoggerName {
    final lastDot = logger.lastIndexOf('.');
    if (lastDot < 0) return logger;
    return logger.substring(lastDot + 1);
  }
}
