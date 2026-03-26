import 'dart:async';
import 'dart:typed_data';
import 'package:audioplayers/audioplayers.dart';
import 'package:logger/logger.dart';

/// Audio Playback Service for TTS audio from backend
class AudioPlaybackService {
  final _logger = Logger();
  final AudioPlayer _player = AudioPlayer();

  bool _isPlaying = false;
  final List<(Uint8List, int)> _audioQueue = [];

  // PCM accumulation buffer for continuous playback
  final List<int> _pcmBuffer = [];
  int? _bufferSampleRate;
  int _chunksSinceFlush = 0;
  bool _preBuffering = true;

  // Pre-buffer threshold: accumulate this many seconds of audio before
  // starting playback. Eliminates gaps between sentences caused by TTS
  // generation time exceeding the previous sentence's audio duration.
  // On single-GPU, TTS runs ~1.3x slower due to LLM contention, so
  // buffering 2 sentences (~6-8s) covers the worst-case gap.
  static const double _preBufferSeconds = 6.0;

  // After pre-buffer fills, flush every chunk immediately for low latency.
  static const int _flushChunkThreshold = 1;

  // Stream controller for playback status
  final _playbackController = StreamController<bool>.broadcast();

  // Getters
  bool get isPlaying => _isPlaying;
  Stream<bool> get playbackStream => _playbackController.stream;

  /// Convert raw PCM int16 data to WAV format with headers
  Uint8List _addWavHeader(List<int> pcmData, int sampleRate) {
    final numChannels = 1; // Mono
    final bitsPerSample = 16; // 16-bit PCM
    final byteRate = sampleRate * numChannels * (bitsPerSample ~/ 8);
    final blockAlign = numChannels * (bitsPerSample ~/ 8);
    final dataSize = pcmData.length;
    final fileSize = 36 + dataSize;

    final header = ByteData(44);

    // RIFF header
    header.setUint8(0, 0x52); // 'R'
    header.setUint8(1, 0x49); // 'I'
    header.setUint8(2, 0x46); // 'F'
    header.setUint8(3, 0x46); // 'F'
    header.setUint32(4, fileSize, Endian.little);

    // WAVE header
    header.setUint8(8, 0x57); // 'W'
    header.setUint8(9, 0x41); // 'A'
    header.setUint8(10, 0x56); // 'V'
    header.setUint8(11, 0x45); // 'E'

    // fmt subchunk
    header.setUint8(12, 0x66); // 'f'
    header.setUint8(13, 0x6D); // 'm'
    header.setUint8(14, 0x74); // 't'
    header.setUint8(15, 0x20); // ' '
    header.setUint32(16, 16, Endian.little); // Subchunk1Size (16 for PCM)
    header.setUint16(20, 1, Endian.little); // AudioFormat (1 = PCM)
    header.setUint16(22, numChannels, Endian.little);
    header.setUint32(24, sampleRate, Endian.little);
    header.setUint32(28, byteRate, Endian.little);
    header.setUint16(32, blockAlign, Endian.little);
    header.setUint16(34, bitsPerSample, Endian.little);

    // data subchunk
    header.setUint8(36, 0x64); // 'd'
    header.setUint8(37, 0x61); // 'a'
    header.setUint8(38, 0x74); // 't'
    header.setUint8(39, 0x61); // 'a'
    header.setUint32(40, dataSize, Endian.little);

    // Combine header + PCM data
    final wavFile = Uint8List(44 + pcmData.length);
    wavFile.setRange(0, 44, header.buffer.asUint8List());
    wavFile.setRange(44, 44 + pcmData.length, pcmData);

    return wavFile;
  }

  AudioPlaybackService() {
    _initializePlayer();
  }

  /// Initialize audio player
  void _initializePlayer() {
    // Listen to player state changes
    _player.onPlayerStateChanged.listen((state) {
      final wasPlaying = _isPlaying;
      _isPlaying = state == PlayerState.playing;

      if (wasPlaying != _isPlaying) {
        _playbackController.add(_isPlaying);
      }

      // When audio completes, play next in queue
      if (state == PlayerState.completed) {
        _playNextInQueue();
      }
    });

    _logger.i('Audio playback service initialized');
  }

  /// Convert float32 audio samples to PCM16 bytes
  Uint8List _float32ToPCM16(List<double> audioFloats) {
    // Backend sends float32 values [-1.0, 1.0] via JSON
    final pcm16 = <int>[];

    for (final floatValue in audioFloats) {
      // Convert float32 [-1.0, 1.0] to PCM16 [-32768, 32767]
      final pcm16Value = (floatValue * 32767).round().clamp(-32768, 32767);

      // Write as little-endian 16-bit int
      pcm16.add(pcm16Value & 0xFF);
      pcm16.add((pcm16Value >> 8) & 0xFF);
    }

    _logger.d(
      'Converted ${audioFloats.length} float32 samples to ${pcm16.length} PCM16 bytes',
    );
    return Uint8List.fromList(pcm16);
  }

  /// Buffer audio chunk from float32 samples for continuous playback.
  ///
  /// During pre-buffering, accumulates audio until [_preBufferSeconds]
  /// worth of PCM data is collected, then flushes everything and starts
  /// playback. After the pre-buffer is satisfied, each subsequent chunk
  /// is flushed immediately for lowest latency.
  ///
  /// This eliminates gaps between sentences where TTS generation time
  /// exceeds the previous sentence's audio duration.
  Future<void> playAudioChunkFloat32(
    List<double> audioData,
    int sampleRate,
  ) async {
    try {
      _bufferSampleRate = sampleRate;

      // Convert float32 to PCM16 bytes and accumulate
      final pcm16Bytes = _float32ToPCM16(audioData);
      _pcmBuffer.addAll(pcm16Bytes);
      _chunksSinceFlush++;

      // Calculate buffered audio duration
      // PCM16 = 2 bytes per sample, mono
      final bufferedSeconds = _pcmBuffer.length / (sampleRate * 2);

      _logger.d(
        'Buffered chunk $_chunksSinceFlush '
        '(${pcm16Bytes.length} bytes, total: ${_pcmBuffer.length}, '
        '${bufferedSeconds.toStringAsFixed(1)}s)',
      );

      if (_preBuffering) {
        // During pre-buffer phase, wait until we have enough audio
        if (bufferedSeconds >= _preBufferSeconds) {
          _logger.i(
            'Pre-buffer filled (${bufferedSeconds.toStringAsFixed(1)}s, '
            '$_chunksSinceFlush chunks). Starting playback.',
          );
          _preBuffering = false;
          _flushBuffer();
        }
      } else {
        // After pre-buffer, flush every chunk immediately
        if (_chunksSinceFlush >= _flushChunkThreshold) {
          _flushBuffer();
        }
      }
    } catch (e) {
      _logger.e('Error buffering audio chunk: $e');
      _logger.e('Stack trace: ${StackTrace.current}');
      rethrow;
    }
  }

  /// Flush accumulated PCM buffer as a single WAV segment.
  void _flushBuffer() {
    if (_pcmBuffer.isEmpty) return;

    final pcmData = Uint8List.fromList(_pcmBuffer);
    _pcmBuffer.clear();
    _chunksSinceFlush = 0;

    final sampleRate = _bufferSampleRate ?? 24000;
    final wavData = _addWavHeader(pcmData, sampleRate);

    _logger.d(
      'Flushing buffer: ${pcmData.length} PCM bytes -> ${wavData.length} WAV bytes',
    );

    _queueWavForPlayback(wavData, sampleRate);
  }

  /// Called when the backend signals audio generation is complete.
  /// Flushes any remaining buffered PCM data as a final WAV segment.
  /// If still pre-buffering (short response), plays whatever we have.
  void flushAndFinalize() {
    if (_preBuffering && _pcmBuffer.isNotEmpty) {
      _logger.i(
        'Audio complete during pre-buffer. Flushing '
        '${_pcmBuffer.length} bytes immediately.',
      );
      _preBuffering = false;
    }
    _flushBuffer();
    _logger.d('Audio buffer finalized');
  }

  /// Queue a WAV segment for playback, or play immediately if idle.
  void _queueWavForPlayback(Uint8List wavData, int sampleRate) {
    if (_isPlaying) {
      _audioQueue.add((wavData, sampleRate));
      _logger.d('Queued WAV segment (queue size: ${_audioQueue.length})');
      return;
    }

    _playWav(wavData);
  }

  /// Play a complete WAV segment (already has header).
  Future<void> _playWav(Uint8List wavData) async {
    try {
      final source = BytesSource(wavData);
      await _player.play(source);
      _logger.d('Playing WAV segment (${wavData.length} bytes)');
    } catch (e) {
      _logger.e('Error in _playWav: $e');
      rethrow;
    }
  }

  /// Play next WAV segment from the queue.
  void _playNextInQueue() {
    if (_audioQueue.isEmpty) {
      return;
    }

    final (nextWav, _) = _audioQueue.removeAt(0);
    _logger.d(
      'Playing next WAV segment from queue (remaining: ${_audioQueue.length})',
    );

    _playWav(nextWav);
  }

  /// Stop playback and discard all buffered audio.
  Future<void> stop() async {
    try {
      await _player.stop();
      _audioQueue.clear();
      _pcmBuffer.clear();
      _chunksSinceFlush = 0;
      _preBuffering = true;
      _isPlaying = false;
      _playbackController.add(false);
      _logger.i('Stopped audio playback');
    } catch (e) {
      _logger.e('Error stopping playback: $e');
    }
  }

  /// Pause playback
  Future<void> pause() async {
    try {
      await _player.pause();
      _logger.i('Paused audio playback');
    } catch (e) {
      _logger.e('Error pausing playback: $e');
    }
  }

  /// Resume playback
  Future<void> resume() async {
    try {
      await _player.resume();
      _logger.i('Resumed audio playback');
    } catch (e) {
      _logger.e('Error resuming playback: $e');
    }
  }

  /// Clear audio queue and PCM buffer. Resets pre-buffer state.
  void clearQueue() {
    _audioQueue.clear();
    _pcmBuffer.clear();
    _chunksSinceFlush = 0;
    _preBuffering = true;
    _logger.d('Cleared audio queue and buffer');
  }

  /// Dispose resources.
  void dispose() {
    _player.dispose();
    _playbackController.close();
    _audioQueue.clear();
    _pcmBuffer.clear();
  }
}
