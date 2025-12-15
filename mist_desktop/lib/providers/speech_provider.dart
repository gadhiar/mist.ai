import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../services/speech_service.dart';
import '../services/audio_playback_service.dart';

/// Speech Service Provider
final speechServiceProvider = Provider<SpeechService>((ref) {
  final service = SpeechService();

  // Initialize on first use
  service.initialize();

  // Dispose when provider is disposed
  ref.onDispose(() {
    service.dispose();
  });

  return service;
});

/// Audio Playback Service Provider
final audioPlaybackServiceProvider = Provider<AudioPlaybackService>((ref) {
  final service = AudioPlaybackService();

  // Dispose when provider is disposed
  ref.onDispose(() {
    service.dispose();
  });

  return service;
});

/// Speech Listening State Provider
final isListeningProvider = StreamProvider<bool>((ref) {
  final service = ref.watch(speechServiceProvider);
  return service.listeningStream;
});

/// Audio Playback State Provider
final isPlayingAudioProvider = StreamProvider<bool>((ref) {
  final service = ref.watch(audioPlaybackServiceProvider);
  return service.playbackStream;
});
