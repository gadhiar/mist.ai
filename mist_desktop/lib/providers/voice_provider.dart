import 'dart:async';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:logger/logger.dart';
import '../services/audio_recording_service.dart';
import '../services/audio_playback_service.dart';
import '../services/websocket_service.dart';
import '../models/binary_audio_frame.dart';
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

    // Route binary audio frames to playback service
    final audioFrameSub = _wsService.audioFrameStream.listen(_handleAudioFrame);

    // Stop audio cleanly on WebSocket disconnect or error
    final connectionSub = _wsService.statusStream.listen((status) {
      if (status == ConnectionStatus.disconnected ||
          status == ConnectionStatus.error) {
        _audioService.stopImmediately();
        state = state.copyWith(isPlaying: false);
      }
    });

    ref.onDispose(() {
      audioSub.cancel();
      wsSub.cancel();
      audioFrameSub.cancel();
      connectionSub.cancel();
    });

    return const VoiceState();
  }

  /// Handle audio-related incoming WebSocket messages.
  ///
  /// Audio is now routed via the binary frame stream (_handleAudioFrame).
  /// This method handles any remaining JSON control messages that are
  /// not audio-specific. Non-audio messages are forwarded to ChatNotifier.
  void _handleAudioMessage(WebSocketMessage message) {
    switch (message.type) {
      default:
        // Non-audio messages are handled by ChatNotifier
        break;
    }
  }

  /// Handle binary audio frames from the WebSocket binary stream.
  ///
  /// Routes each frame type to the appropriate AudioPlaybackService method:
  ///   0x01 (audio_chunk)    -> writeChunk()
  ///   0x02 (audio_complete) -> drain()
  ///   0x03 (interrupt_fade) -> fadeAndClose()
  void _handleAudioFrame(BinaryAudioFrame frame) {
    _audioService.validateSequence(frame.chunkSeq);

    if (frame.isAudioChunk) {
      // writeChunk() internally handles the case where a new chunk arrives
      // while the previous response is still draining or fading -- it calls
      // _forceCleanup() and starts a fresh stream automatically.
      _audioService.writeChunk(frame.payload, frame.sampleRate);
      state = state.copyWith(isPlaying: true);
    } else if (frame.isAudioComplete) {
      _audioService.drain();
      state = state.copyWith(isPlaying: false);
    } else if (frame.isInterruptFade) {
      _audioService.fadeAndClose(frame.payload);
      state = state.copyWith(isPlaying: false);
    } else {
      _logger.w('Unknown audio frame type: ${frame.messageType}');
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
    _audioService.stopImmediately();
    await _audioRecordingService.stopRecording();
    state = const VoiceState();
  }
}

/// Voice Provider
final voiceProvider = NotifierProvider<VoiceNotifier, VoiceState>(() {
  return VoiceNotifier();
});
