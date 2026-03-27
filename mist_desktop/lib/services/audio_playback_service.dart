import 'dart:async';
import 'dart:typed_data';
import 'package:flutter_soloud/flutter_soloud.dart';
import 'package:logger/logger.dart';

/// Playback state machine states.
enum PlaybackState { idle, buffering, playing, draining, fading }

/// Audio Playback Service -- continuous PCM streaming via flutter_soloud.
///
/// Each backend response creates a new buffer stream source. The SoLoud
/// engine stays initialized for the lifetime of this service. PCM16 data
/// arrives as raw Uint8List bytes from binary WebSocket frames.
///
/// First-chunk immediate playback: the first chunk from pipeline
/// parallelism is a full sentence (~2-4s of audio), so playback starts
/// immediately on the first writeChunk() call with no buffering threshold.
/// Subsequent chunks feed directly into the playing stream.
///
/// If a new response arrives while draining or fading a previous one,
/// the old stream is stopped with a fade and the new one starts
/// immediately.
class AudioPlaybackService {
  final _logger = Logger();

  // Stream controller for playback status (true = playing/buffering)
  final _playbackController = StreamController<bool>.broadcast();

  PlaybackState _state = PlaybackState.idle;
  AudioSource? _activeSource;
  SoundHandle? _activeHandle;

  // Diagnostics
  int _underrunCount = 0;
  int _sequenceGapCount = 0;
  int _lastChunkSeq = -1;
  int _chunksReceived = 0;
  Timer? _drainTimer;

  // Underrun warning threshold per response
  static const int _underrunWarningThreshold = 3;

  // Getters
  PlaybackState get state => _state;
  bool get isPlaying =>
      _state == PlaybackState.playing ||
      _state == PlaybackState.buffering ||
      _state == PlaybackState.draining;
  Stream<bool> get playbackStream => _playbackController.stream;
  int get underrunCount => _underrunCount;
  int get sequenceGapCount => _sequenceGapCount;

  /// Initialize the SoLoud engine. Call once at app startup.
  Future<void> initialize() async {
    try {
      if (!SoLoud.instance.isInitialized) {
        await SoLoud.instance.init();
        _logger.i('SoLoud engine initialized');
      }
    } catch (e) {
      _logger.e('Failed to initialize SoLoud engine: $e');
      rethrow;
    }
  }

  /// Feed a PCM16 chunk into the active stream.
  ///
  /// On first call (idle state): creates a buffer stream source and starts
  /// playback immediately -- no buffering threshold. The first chunk from
  /// pipeline parallelism is a full sentence (~2-4s of audio), so there
  /// is no risk of underrun on the first chunk.
  ///
  /// If called while draining or fading a previous response, the old
  /// stream is stopped with a fade and a new stream starts for the
  /// incoming response.
  void writeChunk(Uint8List pcm16Data, int sampleRate) {
    if (_state == PlaybackState.draining || _state == PlaybackState.fading) {
      _logger.i(
        'New chunk arrived in $_state state -- '
        'stopping old stream and starting new one',
      );
      _forceCleanup();
    }

    try {
      if (_state == PlaybackState.idle) {
        _startNewStream(sampleRate);
      }

      SoLoud.instance.addAudioDataStream(_activeSource!, pcm16Data);
      _chunksReceived++;

      _logger.d(
        'Fed chunk #$_chunksReceived (${pcm16Data.length} bytes) '
        'in $_state state',
      );
    } on SoLoudPcmBufferFullCppException {
      _underrunCount++;
      _logger.w(
        'PCM buffer full (overfeed) -- chunk dropped '
        '(underrun count: $_underrunCount)',
      );
      if (_underrunCount > _underrunWarningThreshold) {
        _logger.w(
          'Underrun count ($_underrunCount) exceeds threshold '
          '($_underrunWarningThreshold) for this response',
        );
      }
    } catch (e) {
      _logger.e('Error writing chunk: $e');
    }
  }

  /// Create a new buffer stream source and start playback immediately.
  ///
  /// The source is created synchronously so data can be fed right away.
  /// The play() call is async but we fire-and-forget it -- data fed
  /// before play starts will be buffered by flutter_soloud.
  ///
  /// State transitions directly to playing (no buffering phase) because
  /// the first chunk is a full sentence with enough audio data to
  /// prevent underruns.
  void _startNewStream(int sampleRate) {
    _resetDiagnostics();

    _activeSource = SoLoud.instance.setBufferStream(
      sampleRate: sampleRate,
      channels: Channels.mono,
      format: BufferType.s16le,
      bufferingType: BufferingType.preserved,
      bufferingTimeNeeds: 0.15,
    );

    // Transition directly to playing -- first chunk is a full sentence
    // (~2-4s of audio), so no buffering threshold needed.
    _setState(PlaybackState.playing);

    // play() is async -- fire and store the handle when ready.
    // Data fed via addAudioDataStream before play resolves is buffered.
    SoLoud.instance
        .play(_activeSource!)
        .then((handle) {
          _activeHandle = handle;
          _logger.d('SoLoud play handle acquired');
        })
        .catchError((Object e) {
          _logger.e('Failed to start SoLoud playback: $e');
          _cleanup();
        });

    _logger.i(
      'Started new buffer stream (rate: $sampleRate, '
      'jitter buffer: 150ms) -- immediate playback',
    );
  }

  /// Signal that all audio data has been sent for this response.
  ///
  /// Calls setDataIsEnded on the source and transitions to draining.
  /// The stream will finish playing remaining buffered audio, then
  /// we clean up.
  void drain() {
    if (_activeSource == null || _state == PlaybackState.idle) {
      _logger.d('drain() called with no active source -- ignoring');
      return;
    }

    if (_state == PlaybackState.draining || _state == PlaybackState.fading) {
      _logger.d('drain() called in $_state state -- ignoring');
      return;
    }

    try {
      SoLoud.instance.setDataIsEnded(_activeSource!);
      _setState(PlaybackState.draining);
      _logger.i('Draining -- setDataIsEnded called');

      _logStreamDiagnostics();
      _startDrainMonitor();
    } catch (e) {
      _logger.e('Error during drain: $e');
      _cleanup();
    }
  }

  /// Feed a fade-out payload and close the stream.
  ///
  /// Used when an interrupt_fade frame arrives: we feed the short
  /// fade audio, signal end-of-data, and transition to fading.
  void fadeAndClose(Uint8List fadePayload) {
    if (_activeSource == null || _state == PlaybackState.idle) {
      _logger.d('fadeAndClose() called with no active source -- ignoring');
      return;
    }

    try {
      SoLoud.instance.addAudioDataStream(_activeSource!, fadePayload);
      SoLoud.instance.setDataIsEnded(_activeSource!);
      _setState(PlaybackState.fading);
      _logger.i('Fading -- fed ${fadePayload.length} bytes fade payload');

      _logStreamDiagnostics();
      _startDrainMonitor();
    } on SoLoudPcmBufferFullCppException {
      _logger.w('Buffer full during fadeAndClose -- forcing stop');
      _cleanup();
    } catch (e) {
      _logger.e('Error during fadeAndClose: $e');
      _cleanup();
    }
  }

  /// Hard stop -- dispose source and reset to idle immediately.
  void stopImmediately() {
    _logger.i('stopImmediately() called');
    _cleanup();
  }

  /// Stop with a brief silence ramp.
  ///
  /// Used when a new response starts while still draining the previous
  /// one. Feeds a short block of silence then disposes.
  void stopWithFade() {
    if (_activeSource == null || _state == PlaybackState.idle) {
      _logger.d('stopWithFade() called with no active source');
      return;
    }

    try {
      // 20ms of silence at 24kHz mono 16-bit = 960 bytes
      final silence = Uint8List(960);
      SoLoud.instance.addAudioDataStream(_activeSource!, silence);
      SoLoud.instance.setDataIsEnded(_activeSource!);
      _setState(PlaybackState.fading);
      _logger.i('stopWithFade -- fed 20ms silence ramp');

      // Clean up after the silence plays out
      Future.delayed(const Duration(milliseconds: 50), () {
        _cleanup();
      });
    } catch (e) {
      _logger.w('Error during stopWithFade: $e -- forcing cleanup');
      _cleanup();
    }
  }

  /// Track chunk sequence numbers and log gaps.
  void validateSequence(int chunkSeq) {
    if (_lastChunkSeq >= 0 && chunkSeq != _lastChunkSeq + 1) {
      final gap = chunkSeq - _lastChunkSeq - 1;
      _sequenceGapCount++;
      _logger.w(
        'Sequence gap detected: expected ${_lastChunkSeq + 1}, '
        'got $chunkSeq (gap: $gap, total gaps: $_sequenceGapCount)',
      );
    }
    _lastChunkSeq = chunkSeq;
  }

  /// Monitor draining/fading state and clean up when playback finishes.
  void _startDrainMonitor() {
    _drainTimer?.cancel();
    // Poll every 100ms to check if playback has finished.
    // SoLoud auto-disposes the source when buffer is drained after
    // setDataIsEnded, so we check if the source is still valid.
    _drainTimer = Timer.periodic(const Duration(milliseconds: 100), (timer) {
      if (_activeSource == null) {
        timer.cancel();
        return;
      }

      try {
        // Check if the sound handle is still valid by querying it.
        // If SoLoud has finished playing, the handle becomes invalid
        // and operations on it may throw.
        final valid = SoLoud.instance.getIsValidVoiceHandle(_activeHandle!);
        if (!valid) {
          _logger.d('Drain/fade complete -- source finished playing');
          timer.cancel();
          _cleanup();
        }
      } catch (e) {
        // Source no longer valid -- playback finished
        _logger.d('Drain monitor: source invalid -- cleaning up');
        timer.cancel();
        _cleanup();
      }
    });
  }

  void _setState(PlaybackState newState) {
    final oldState = _state;
    _state = newState;

    final wasPlaying =
        oldState == PlaybackState.playing ||
        oldState == PlaybackState.buffering ||
        oldState == PlaybackState.draining;
    final nowPlaying =
        newState == PlaybackState.playing ||
        newState == PlaybackState.buffering ||
        newState == PlaybackState.draining;

    if (wasPlaying != nowPlaying) {
      _playbackController.add(nowPlaying);
    }

    _logger.d('State: $oldState -> $newState');
  }

  void _cleanup() {
    _drainTimer?.cancel();
    _drainTimer = null;

    if (_activeHandle != null) {
      try {
        // Stop any active playback on this handle
        SoLoud.instance.stop(_activeHandle!);
      } catch (e) {
        // Handle may already be invalid -- that's fine
        _logger.d('Cleanup: handle already disposed');
      }
      _activeHandle = null;
    }
    _activeSource = null;

    _setState(PlaybackState.idle);
  }

  /// Force cleanup without SoLoud interaction for state transitions
  /// where we need to reset before starting a new stream.
  void _forceCleanup() {
    _drainTimer?.cancel();
    _drainTimer = null;

    if (_activeHandle != null) {
      try {
        SoLoud.instance.stop(_activeHandle!);
      } catch (e) {
        _logger.d('Force cleanup: handle already disposed');
      }
      _activeHandle = null;
    }
    _activeSource = null;

    // Reset to idle without emitting a playback stream event --
    // the new stream will immediately follow.
    _state = PlaybackState.idle;
  }

  void _resetDiagnostics() {
    _underrunCount = 0;
    _sequenceGapCount = 0;
    _lastChunkSeq = -1;
    _chunksReceived = 0;
  }

  void _logStreamDiagnostics() {
    _logger.i(
      'Stream diagnostics: '
      'chunks=$_chunksReceived, '
      'underruns=$_underrunCount, '
      'sequence_gaps=$_sequenceGapCount',
    );

    if (_underrunCount > _underrunWarningThreshold) {
      _logger.w(
        'High underrun count for this response: $_underrunCount '
        '(threshold: $_underrunWarningThreshold)',
      );
    }
  }

  /// Dispose resources. Called when the service is torn down.
  void dispose() {
    _cleanup();
    _playbackController.close();
  }
}
