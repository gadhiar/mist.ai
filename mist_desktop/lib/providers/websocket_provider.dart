import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../services/websocket_service.dart';

/// WebSocket Service Provider
final websocketServiceProvider = Provider<WebSocketService>((ref) {
  final service = WebSocketService();

  // Dispose when provider is disposed
  ref.onDispose(() {
    service.dispose();
  });

  return service;
});

/// Connection Status Provider
final connectionStatusProvider = StreamProvider<ConnectionStatus>((ref) {
  final service = ref.watch(websocketServiceProvider);
  return service.statusStream;
});
