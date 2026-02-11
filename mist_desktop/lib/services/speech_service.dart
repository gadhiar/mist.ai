import 'dart:async';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:logger/logger.dart';

/// Speech-to-Text Service using platform native STT
class SpeechService {
  final _logger = Logger();
  final stt.SpeechToText _speech = stt.SpeechToText();

  bool _isInitialized = false;
  bool _isListening = false;

  // Stream controllers
  final _transcriptionController = StreamController<String>.broadcast();
  final _listeningController = StreamController<bool>.broadcast();

  // Getters
  bool get isInitialized => _isInitialized;
  bool get isListening => _isListening;
  Stream<String> get transcriptionStream => _transcriptionController.stream;
  Stream<bool> get listeningStream => _listeningController.stream;

  /// Initialize speech recognition
  Future<bool> initialize() async {
    try {
      _logger.i('Initializing speech recognition...');
      _isInitialized = await _speech.initialize(
        onStatus: _handleStatus,
        onError: _handleError,
        debugLogging: false, // Disable debug logging to reduce noise
      );

      if (_isInitialized) {
        _logger.i('Speech recognition initialized successfully');
        try {
          final locales = await _speech.locales();
          _logger.d('Available locales: ${locales.length}');
        } catch (e) {
          // Ignore locale fetching errors (Windows plugin issue)
          _logger.w('Could not fetch locales (non-critical): $e');
        }
      } else {
        _logger.e('Failed to initialize speech recognition');
      }

      return _isInitialized;
    } catch (e) {
      _logger.e('Error initializing speech recognition: $e');
      return false;
    }
  }

  /// Start listening for speech
  Future<void> startListening() async {
    if (!_isInitialized) {
      _logger.w('Speech recognition not initialized');
      return;
    }

    if (_isListening) {
      _logger.w('Already listening');
      return;
    }

    try {
      await _speech.listen(
        onResult: _handleResult,
        listenFor: const Duration(seconds: 60), // Max listen duration (increased)
        pauseFor: const Duration(seconds: 5),   // Pause detection (increased to avoid early stop)
        listenOptions: stt.SpeechListenOptions(
          partialResults: true,
          cancelOnError: false, // Don't auto-cancel on errors (Windows plugin has issues)
          listenMode: stt.ListenMode.confirmation,
        ),
      );

      _isListening = true;
      _listeningController.add(true);
      _logger.i(' Started listening - speak now!');
    } catch (e) {
      _logger.e('Error starting listening: $e');
      _isListening = false;
      _listeningController.add(false);
    }
  }

  /// Stop listening
  Future<void> stopListening() async {
    if (!_isListening) {
      return;
    }

    try {
      await _speech.stop();
      _isListening = false;
      _listeningController.add(false);
      _logger.i('Stopped listening');
    } catch (e) {
      _logger.e('Error stopping listening: $e');
    }
  }

  /// Cancel listening (discard results)
  Future<void> cancel() async {
    if (!_isListening) {
      return;
    }

    try {
      await _speech.cancel();
      _isListening = false;
      _listeningController.add(false);
      _logger.i('Cancelled listening');
    } catch (e) {
      _logger.e('Error cancelling listening: $e');
    }
  }

  /// Handle speech recognition results
  void _handleResult(result) {
    try {
      _logger.d('Raw result received: $result');

      final text = result.recognizedWords as String;
      final isFinal = result.finalResult as bool;

      _logger.i(' STT Result: "$text" (final: $isFinal)');

      // Only send final results to avoid too many partial updates
      if (isFinal && text.isNotEmpty) {
        _logger.i(' Sending transcription: "$text"');
        _transcriptionController.add(text);
        _logger.i(' Transcription sent successfully!');
      } else if (!isFinal) {
        _logger.d('⏭  Skipping partial result: "$text"');
      }
    } catch (e) {
      _logger.e(' Error handling STT result: $e');
      _logger.e('Result object: $result');
    }
  }

  /// Handle status changes
  void _handleStatus(String status) {
    _logger.d('STT Status: $status');

    if (status == 'done' || status == 'notListening') {
      _isListening = false;
      _listeningController.add(false);
    }
  }

  /// Handle errors
  void _handleError(dynamic error) {
    // Filter out known Windows plugin errors that don't affect functionality
    final errorStr = error.toString();
    if (errorStr.contains('type \'Null\' is not a subtype') ||
        errorStr.contains('textRecognition')) {
      _logger.w('Non-critical STT error (Windows plugin): $error');
      return; // Don't stop listening for these errors
    }

    _logger.e('STT Error: $error');
    _isListening = false;
    _listeningController.add(false);
  }

  /// Dispose resources
  void dispose() {
    _speech.stop();
    _transcriptionController.close();
    _listeningController.close();
  }
}
