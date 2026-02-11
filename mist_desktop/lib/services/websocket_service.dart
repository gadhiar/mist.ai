import 'dart:async';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:logger/logger.dart';
import '../config/app_config.dart';
import '../models/websocket_message.dart';

/// WebSocket Connection Status
enum ConnectionStatus { disconnected, connecting, connected, error }

/// WebSocket Service for communicating with the backend
class WebSocketService {
  final _logger = Logger();

  WebSocketChannel? _channel;
  ConnectionStatus _status = ConnectionStatus.disconnected;

  // Stream controllers
  final _statusController = StreamController<ConnectionStatus>.broadcast();
  final _messageController = StreamController<WebSocketMessage>.broadcast();

  // Getters
  ConnectionStatus get status => _status;
  Stream<ConnectionStatus> get statusStream => _statusController.stream;
  Stream<WebSocketMessage> get messageStream => _messageController.stream;

  /// Connect to WebSocket server
  Future<void> connect() async {
    if (_status == ConnectionStatus.connected) {
      _logger.w('Already connected to WebSocket');
      return;
    }

    try {
      _updateStatus(ConnectionStatus.connecting);
      _logger.i('Connecting to WebSocket: ${AppConfig.wsUrl}');

      _channel = WebSocketChannel.connect(Uri.parse(AppConfig.wsUrl));

      // Wait for connection to be established
      await _channel!.ready;

      _updateStatus(ConnectionStatus.connected);
      _logger.i('WebSocket connected successfully');

      // Listen to messages
      _channel!.stream.listen(
        _handleMessage,
        onError: _handleError,
        onDone: _handleClose,
        cancelOnError: false,
      );
    } catch (e) {
      _logger.e('Failed to connect to WebSocket: $e');
      _updateStatus(ConnectionStatus.error);
      rethrow;
    }
  }

  /// Disconnect from WebSocket server
  Future<void> disconnect() async {
    if (_status == ConnectionStatus.disconnected) {
      return;
    }

    _logger.i('Disconnecting from WebSocket');
    await _channel?.sink.close();
    _channel = null;
    _updateStatus(ConnectionStatus.disconnected);
  }

  /// Send a message to the server
  void sendMessage(WebSocketMessage message) {
    if (_status != ConnectionStatus.connected) {
      _logger.w('Cannot send message: not connected');
      throw Exception('WebSocket not connected');
    }

    try {
      final jsonStr = message.toJson();
      _channel?.sink.add(jsonStr);
      _logger.d('Sent message: ${message.type}');
    } catch (e) {
      _logger.e('Failed to send message: $e');
      rethrow;
    }
  }

  /// Send audio data from raw bytes (PCM16)
  void sendAudioBytes(List<int> audioBytes, {int sampleRate = 16000}) {
    // Convert PCM16 bytes to float32 array for backend
    final audioData = _convertPCM16ToFloat32(audioBytes);
    sendMessage(
      WebSocketMessage.audio(audio: audioData, sampleRate: sampleRate),
    );
  }

  /// Convert PCM16 bytes to float32 array
  List<double> _convertPCM16ToFloat32(List<int> bytes) {
    final samples = <double>[];
    for (int i = 0; i < bytes.length - 1; i += 2) {
      // Read 16-bit little-endian sample
      final sample = (bytes[i] | (bytes[i + 1] << 8)).toSigned(16);
      // Convert to float32 range [-1.0, 1.0]
      samples.add(sample / 32768.0);
    }
    return samples;
  }

  /// Send text message
  void sendText(String text) {
    sendMessage(WebSocketMessage.text(text));
  }

  /// Send interrupt signal
  void sendInterrupt() {
    sendMessage(WebSocketMessage.interrupt());
  }

  /// Reset VAD
  void resetVad() {
    sendMessage(WebSocketMessage.resetVad());
  }

  /// Handle incoming message
  void _handleMessage(dynamic data) {
    try {
      final jsonStr = data as String;
      final message = WebSocketMessage.fromJson(jsonStr);
      _logger.d('Received message: ${message.type}');
      _messageController.add(message);
    } catch (e) {
      _logger.e('Failed to parse message: $e');
    }
  }

  /// Handle WebSocket error
  void _handleError(dynamic error) {
    _logger.e('WebSocket error: $error');
    _updateStatus(ConnectionStatus.error);
  }

  /// Handle WebSocket close
  void _handleClose() {
    _logger.i('WebSocket connection closed');
    _updateStatus(ConnectionStatus.disconnected);
  }

  /// Update connection status
  void _updateStatus(ConnectionStatus newStatus) {
    _status = newStatus;
    _statusController.add(newStatus);
  }

  /// Dispose resources
  void dispose() {
    disconnect();
    _statusController.close();
    _messageController.close();
  }
}
