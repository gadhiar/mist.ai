import 'dart:async';
import 'dart:typed_data';
import 'package:record/record.dart';
import 'package:logger/logger.dart';

/// Audio Recording Service - Captures mic and collects complete audio
class AudioRecordingService {
  final _logger = Logger();
  final AudioRecorder _recorder = AudioRecorder();

  bool _isRecording = false;
  StreamSubscription<Uint8List>? _audioStreamSubscription;
  final List<int> _audioBuffer = []; // Collect all audio chunks

  // Stream controllers
  final _recordingController = StreamController<bool>.broadcast();
  final _audioCompleteController = StreamController<Uint8List>.broadcast();

  // Getters
  bool get isRecording => _isRecording;
  Stream<bool> get recordingStream => _recordingController.stream;
  Stream<Uint8List> get audioCompleteStream => _audioCompleteController.stream;

  /// Check if microphone permission is granted
  Future<bool> hasPermission() async {
    return await _recorder.hasPermission();
  }

  /// Start recording and streaming audio
  Future<void> startRecording() async {
    if (_isRecording) {
      _logger.w('Already recording');
      return;
    }

    try {
      // Check permission
      if (!await hasPermission()) {
        _logger.e('Microphone permission not granted');
        throw Exception('Microphone permission denied');
      }

      _logger.i('Starting audio recording...');

      // Start recording with streaming
      final stream = await _recorder.startStream(
        const RecordConfig(
          encoder: AudioEncoder.pcm16bits, // Raw PCM for backend
          sampleRate: 16000,                // Match backend expectation
          numChannels: 1,                   // Mono
          bitRate: 128000,
        ),
      );

      _isRecording = true;
      _recordingController.add(true);
      _logger.i(' Audio recording started (16kHz, mono, PCM16)');

      // Listen to audio stream and collect in buffer
      _audioStreamSubscription = stream.listen(
        (audioChunk) {
          _audioBuffer.addAll(audioChunk);
          _logger.d(' Collected audio chunk: ${audioChunk.length} bytes (total: ${_audioBuffer.length})');
        },
        onError: (error) {
          _logger.e('Audio stream error: $error');
          stopRecording();
        },
        onDone: () {
          _logger.i('Audio stream completed');
          stopRecording();
        },
      );
    } catch (e) {
      _logger.e('Failed to start recording: $e');
      _isRecording = false;
      _recordingController.add(false);
      rethrow;
    }
  }

  /// Stop recording and emit complete audio
  Future<void> stopRecording() async {
    if (!_isRecording) {
      return;
    }

    try {
      _logger.i('Stopping audio recording...');

      await _audioStreamSubscription?.cancel();
      await _recorder.stop();

      _isRecording = false;
      _recordingController.add(false);

      // Emit complete audio buffer
      if (_audioBuffer.isNotEmpty) {
        final completeAudio = Uint8List.fromList(_audioBuffer);
        _audioCompleteController.add(completeAudio);
        _logger.i(' Audio recording stopped - collected ${completeAudio.length} bytes total');

        // Clear buffer for next recording
        _audioBuffer.clear();
      } else {
        _logger.w('No audio data collected');
      }
    } catch (e) {
      _logger.e('Error stopping recording: $e');
      _isRecording = false;
      _recordingController.add(false);
    }
  }

  /// Dispose resources
  void dispose() {
    _audioStreamSubscription?.cancel();
    _recorder.dispose();
    _audioCompleteController.close();
    _recordingController.close();
  }
}
