import 'dart:async';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:logger/logger.dart';
import '../services/audio_recording_service.dart';
import '../services/audio_playback_service.dart';
import '../services/websocket_service.dart';
import '../models/websocket_message.dart';
import 'websocket_provider.dart';
import 'audio_provider.dart';

/// Voice state
class VoiceState {
  final bool isRecording;
  final bool isPlaying;

  const VoiceState({this.isRecording = false, this.isPlaying = false});

  VoiceState copyWith({bool? isRecording, bool? isPlaying}) {
    return VoiceState(
      isRecording: isRecording ?? this.isRecording,
      isPlaying: isPlaying ?? this.isPlaying,
    );
  }
}

/// Voice Notifier - owns recording, playback, and audio chunk routing
class VoiceNotifier extends Notifier<VoiceState> {
  late AudioRecordingService _audioRecordingService;
  late AudioPlaybackService _audioService;
  late WebSocketService _wsService;
  final Logger _logger = Logger();

  @override
  VoiceState build() {
    _audioRecordingService = ref.read(audioRecordingServiceProvider);
    _audioService = ref.read(audioPlaybackServiceProvider);
    _wsService = ref.read(websocketServiceProvider);

    // When recording completes, send audio bytes to backend
    final audioSub = _audioRecordingService.audioCompleteStream.listen((
      completeAudio,
    ) {
      try {
        _wsService.sendAudioBytes(completeAudio);
        _logger.i(
          'Sent complete audio to backend: ${completeAudio.length} bytes',
        );
      } catch (e) {
        _logger.w('Could not send audio: $e');
      }
    });

    // Route audio-related WebSocket messages
    final wsSub = _wsService.messageStream.listen((message) {
      _handleAudioMessage(message);
    });

    ref.onDispose(() {
      audioSub.cancel();
      wsSub.cancel();
    });

    return const VoiceState();
  }

  /// Handle audio-related incoming WebSocket messages
  void _handleAudioMessage(WebSocketMessage message) {
    switch (message.type) {
      case WsMessageType.audioChunk:
        if (message.audio != null && message.sampleRate != null) {
          final audioData = message.audio!
              .map((e) => (e as num).toDouble())
              .toList();
          _audioService.playAudioChunkFloat32(audioData, message.sampleRate!);
          _logger.d(
            'Playing audio chunk: ${message.chunkNum} (${audioData.length} samples)',
          );
        }
        break;

      case WsMessageType.audioComplete:
        _logger.d('Audio generation complete');
        state = state.copyWith(isPlaying: false);
        break;

      default:
        // Non-audio messages are handled by ChatNotifier
        break;
    }
  }

  /// Start voice input
  Future<void> startVoiceInput() async {
    try {
      await _audioRecordingService.startRecording();
      state = state.copyWith(isRecording: true);
      _logger.i('Started recording audio for backend Whisper');
    } catch (e) {
      _logger.w('Failed to start recording: $e');
      state = state.copyWith(isRecording: false);
      rethrow;
    }
  }

  /// Stop voice input
  Future<void> stopVoiceInput() async {
    await _audioRecordingService.stopRecording();
    state = state.copyWith(isRecording: false);
    _logger.i('Stopped recording - audio will be sent to backend');
  }

  /// Stop all audio (called during interrupt)
  Future<void> stopAll() async {
    await _audioService.stop();
    await _audioRecordingService.stopRecording();
    state = const VoiceState();
  }
}

/// Voice Provider
final voiceProvider = NotifierProvider<VoiceNotifier, VoiceState>(() {
  return VoiceNotifier();
});
