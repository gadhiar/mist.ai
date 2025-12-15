import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../services/audio_recording_service.dart';
import '../services/audio_playback_service.dart';

/// Audio Recording Service Provider
final audioRecordingServiceProvider = Provider<AudioRecordingService>((ref) {
  final service = AudioRecordingService();

  // Dispose when provider is disposed
  ref.onDispose(() {
    service.dispose();
  });

  return service;
});

/// Audio Playback Service Provider (moved from speech_provider)
final audioPlaybackServiceProvider = Provider<AudioPlaybackService>((ref) {
  final service = AudioPlaybackService();

  // Dispose when provider is disposed
  ref.onDispose(() {
    service.dispose();
  });

  return service;
});

/// Audio Recording State Provider
final isRecordingProvider = StreamProvider<bool>((ref) {
  final service = ref.watch(audioRecordingServiceProvider);
  return service.recordingStream;
});

/// Audio Playback State Provider
final isPlayingAudioProvider = StreamProvider<bool>((ref) {
  final service = ref.watch(audioPlaybackServiceProvider);
  return service.playbackStream;
});
