import 'dart:async';
import 'dart:typed_data';

import 'package:mist_desktop/models/binary_audio_frame.dart';
import 'package:mist_desktop/models/websocket_message.dart';
import 'package:mist_desktop/services/audio_playback_service.dart';
import 'package:mist_desktop/services/audio_recording_service.dart';
import 'package:mist_desktop/services/websocket_service.dart';

// ---------------------------------------------------------------------------
// FakeWebSocketService
// ---------------------------------------------------------------------------

class FakeWebSocketService implements WebSocketService {
  final List<WebSocketMessage> sentMessages = [];

  final _statusController = StreamController<ConnectionStatus>.broadcast();
  final _messageController = StreamController<WebSocketMessage>.broadcast();
  final _audioFrameController = StreamController<BinaryAudioFrame>.broadcast();

  ConnectionStatus _status = ConnectionStatus.disconnected;

  // Expose controllers so tests can push values in.
  StreamController<ConnectionStatus> get statusTestController =>
      _statusController;
  StreamController<WebSocketMessage> get messageTestController =>
      _messageController;
  StreamController<BinaryAudioFrame> get audioFrameTestController =>
      _audioFrameController;

  @override
  ConnectionStatus get status => _status;

  @override
  Stream<ConnectionStatus> get statusStream => _statusController.stream;

  @override
  Stream<WebSocketMessage> get messageStream => _messageController.stream;

  @override
  Stream<BinaryAudioFrame> get audioFrameStream => _audioFrameController.stream;

  // Convenience: last text message payload
  String? get lastSentText {
    for (final msg in sentMessages.reversed) {
      if (msg.type == WsMessageType.text) return msg.text;
    }
    return null;
  }

  @override
  Future<void> connect() async {
    _status = ConnectionStatus.connected;
    _statusController.add(_status);
  }

  @override
  Future<void> disconnect() async {
    _status = ConnectionStatus.disconnected;
    _statusController.add(_status);
  }

  @override
  void sendMessage(WebSocketMessage message) {
    if (_status != ConnectionStatus.connected) {
      throw Exception('WebSocket not connected');
    }
    sentMessages.add(message);
  }

  @override
  void sendText(String text) {
    sendMessage(WebSocketMessage.text(text));
  }

  @override
  void sendAudioBytes(List<int> audioBytes, {int sampleRate = 16000}) {
    // No-op: records call implicitly via sendMessage -> sentMessages
    sendMessage(
      WebSocketMessage.audio(audio: const [], sampleRate: sampleRate),
    );
  }

  @override
  void sendInterrupt() {
    sendMessage(WebSocketMessage.interrupt());
  }

  @override
  void resetVad() {
    sendMessage(WebSocketMessage.resetVad());
  }

  @override
  void cancelReconnect() {}

  @override
  void dispose() {
    _statusController.close();
    _messageController.close();
    _audioFrameController.close();
  }

  // Test helper: simulate incoming server message
  void simulateIncoming(WebSocketMessage message) {
    _messageController.add(message);
  }

  // Test helper: set status without connecting
  void setStatus(ConnectionStatus status) {
    _status = status;
    _statusController.add(_status);
  }
}

// ---------------------------------------------------------------------------
// FakeAudioRecordingService
// ---------------------------------------------------------------------------

class FakeAudioRecordingService implements AudioRecordingService {
  int startRecordingCallCount = 0;
  int stopRecordingCallCount = 0;

  bool _isRecording = false;

  final _recordingController = StreamController<bool>.broadcast();
  final _audioCompleteController = StreamController<Uint8List>.broadcast();

  StreamController<bool> get recordingTestController => _recordingController;
  StreamController<Uint8List> get audioCompleteTestController =>
      _audioCompleteController;

  @override
  bool get isRecording => _isRecording;

  @override
  Stream<bool> get recordingStream => _recordingController.stream;

  @override
  Stream<Uint8List> get audioCompleteStream => _audioCompleteController.stream;

  @override
  Future<bool> hasPermission() async => true;

  @override
  Future<void> startRecording() async {
    startRecordingCallCount++;
    _isRecording = true;
    _recordingController.add(true);
  }

  @override
  Future<void> stopRecording() async {
    stopRecordingCallCount++;
    _isRecording = false;
    _recordingController.add(false);
  }

  @override
  void dispose() {
    _recordingController.close();
    _audioCompleteController.close();
  }

  // Test helper: emit a completed audio buffer
  void simulateAudioComplete(Uint8List bytes) {
    _audioCompleteController.add(bytes);
  }
}

// ---------------------------------------------------------------------------
// FakeAudioPlaybackService
// ---------------------------------------------------------------------------

class WriteChunkCall {
  const WriteChunkCall({required this.pcm16Data, required this.sampleRate});

  final Uint8List pcm16Data;
  final int sampleRate;
}

class FakeAudioPlaybackService implements AudioPlaybackService {
  final List<WriteChunkCall> writeChunkCalls = [];
  int drainCallCount = 0;
  int stopImmediatelyCallCount = 0;
  int stopWithFadeCallCount = 0;
  int fadeAndCloseCallCount = 0;
  final List<int> validatedSequences = [];
  Uint8List? lastFadePayload;

  PlaybackState _state = PlaybackState.idle;
  final _playbackController = StreamController<bool>.broadcast();

  StreamController<bool> get playbackTestController => _playbackController;

  @override
  PlaybackState get state => _state;

  @override
  bool get isPlaying =>
      _state == PlaybackState.playing ||
      _state == PlaybackState.buffering ||
      _state == PlaybackState.draining;

  @override
  Stream<bool> get playbackStream => _playbackController.stream;

  @override
  int get underrunCount => 0;

  @override
  int get sequenceGapCount => 0;

  @override
  Future<void> initialize() async {}

  @override
  void writeChunk(Uint8List pcm16Data, int sampleRate) {
    writeChunkCalls.add(
      WriteChunkCall(pcm16Data: pcm16Data, sampleRate: sampleRate),
    );
    _state = PlaybackState.playing;
    _playbackController.add(true);
  }

  @override
  void drain() {
    drainCallCount++;
    _state = PlaybackState.draining;
  }

  @override
  void fadeAndClose(Uint8List fadePayload) {
    fadeAndCloseCallCount++;
    lastFadePayload = fadePayload;
    _state = PlaybackState.fading;
  }

  @override
  void stopImmediately() {
    stopImmediatelyCallCount++;
    _state = PlaybackState.idle;
    _playbackController.add(false);
  }

  @override
  void stopWithFade() {
    stopWithFadeCallCount++;
    _state = PlaybackState.idle;
    _playbackController.add(false);
  }

  @override
  void validateSequence(int chunkSeq) {
    validatedSequences.add(chunkSeq);
  }

  @override
  void dispose() {
    _playbackController.close();
  }
}
