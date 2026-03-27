// Tests for AudioPlaybackService state machine and diagnostics.
//
// AudioPlaybackService depends on flutter_soloud's SoLoud.instance, a
// singleton that requires platform (audio device) initialization. Methods that
// call SoLoud directly (writeChunk, startNewStream, drain with active source,
// fadeAndClose with active source, stopImmediately with active handle) cannot
// be unit-tested without mocking the static singleton. SoLoud is not designed
// for injection, so those paths are covered by integration/manual tests.
//
// What IS tested here:
//   - Initial state and diagnostic counters
//   - validateSequence() gap detection (pure logic, no platform dependency)
//   - Guard behavior of drain/fadeAndClose/stopImmediately/stopWithFade
//     when called from idle (no active source -- SoLoud never touched)
//   - dispose() from idle does not throw

import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:mist_desktop/services/audio_playback_service.dart';

void main() {
  group('AudioPlaybackService initial state', () {
    test('starts in idle state', () {
      final service = AudioPlaybackService();
      expect(service.state, equals(PlaybackState.idle));
    });

    test('isPlaying is false when idle', () {
      final service = AudioPlaybackService();
      expect(service.isPlaying, isFalse);
    });

    test('underrunCount is zero initially', () {
      final service = AudioPlaybackService();
      expect(service.underrunCount, equals(0));
    });

    test('sequenceGapCount is zero initially', () {
      final service = AudioPlaybackService();
      expect(service.sequenceGapCount, equals(0));
    });
  });

  group('AudioPlaybackService.validateSequence', () {
    test('detects single gap', () {
      final service = AudioPlaybackService();
      service.validateSequence(1);
      service.validateSequence(3); // gap: expected 2
      expect(service.sequenceGapCount, equals(1));
    });

    test('no gap for sequential chunks', () {
      final service = AudioPlaybackService();
      service.validateSequence(1);
      service.validateSequence(2);
      service.validateSequence(3);
      expect(service.sequenceGapCount, equals(0));
    });

    test('detects multiple gaps', () {
      final service = AudioPlaybackService();
      service.validateSequence(1);
      service.validateSequence(5); // gap
      service.validateSequence(6);
      service.validateSequence(10); // gap
      expect(service.sequenceGapCount, equals(2));
    });

    test('first chunk never counts as gap', () {
      final service = AudioPlaybackService();
      service.validateSequence(42); // first chunk, any number is fine
      expect(service.sequenceGapCount, equals(0));
    });
  });

  group('AudioPlaybackService guard behavior', () {
    test('drain from idle is no-op', () {
      final service = AudioPlaybackService();
      service.drain(); // should not throw
      expect(service.state, equals(PlaybackState.idle));
    });

    test('fadeAndClose from idle is no-op', () {
      final service = AudioPlaybackService();
      service.fadeAndClose(Uint8List(960)); // should not throw
      expect(service.state, equals(PlaybackState.idle));
    });

    test('stopImmediately from idle is no-op', () {
      final service = AudioPlaybackService();
      service.stopImmediately(); // should not throw
      expect(service.state, equals(PlaybackState.idle));
    });

    test('stopWithFade from idle is no-op', () {
      final service = AudioPlaybackService();
      service.stopWithFade(); // should not throw
      expect(service.state, equals(PlaybackState.idle));
    });

    test('dispose from idle does not throw', () {
      final service = AudioPlaybackService();
      service.dispose(); // should not throw
    });
  });
}
