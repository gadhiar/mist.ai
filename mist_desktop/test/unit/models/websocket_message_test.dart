import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:mist_desktop/models/websocket_message.dart';

void main() {
  // ---------------------------------------------------------------------------
  // fromJson parsing
  // ---------------------------------------------------------------------------
  group('WebSocketMessage.fromJson', () {
    test('parses text message', () {
      final msg = WebSocketMessage.fromJson('{"type":"text","text":"hello"}');

      expect(msg.type, equals(WsMessageType.text));
      expect(msg.text, equals('hello'));
    });

    test('parses llm_token message', () {
      final msg = WebSocketMessage.fromJson(
        '{"type":"llm_token","token":"word"}',
      );

      expect(msg.type, equals(WsMessageType.llmToken));
      expect(msg.token, equals('word'));
    });

    test('parses llm_response message', () {
      final msg = WebSocketMessage.fromJson(
        '{"type":"llm_response","text":"full response"}',
      );

      expect(msg.type, equals(WsMessageType.llmResponse));
      expect(msg.text, equals('full response'));
    });

    test('parses transcription message', () {
      final msg = WebSocketMessage.fromJson(
        '{"type":"transcription","text":"user said this"}',
      );

      expect(msg.type, equals(WsMessageType.transcription));
      expect(msg.text, equals('user said this'));
    });

    test('parses audio_chunk message', () {
      final msg = WebSocketMessage.fromJson(
        '{"type":"audio_chunk","audio":[0.1,-0.2,0.3],"sample_rate":24000,"chunk_num":1}',
      );

      expect(msg.type, equals(WsMessageType.audioChunk));
      expect(msg.audio, isNotNull);
      expect(msg.audio!.length, equals(3));
      expect(msg.sampleRate, equals(24000));
      expect(msg.chunkNum, equals(1));
    });

    test('parses status message', () {
      final msg = WebSocketMessage.fromJson(
        '{"type":"status","message":"ready"}',
      );

      expect(msg.type, equals(WsMessageType.status));
      expect(msg.message, equals('ready'));
    });

    test('parses error message', () {
      final msg = WebSocketMessage.fromJson(
        '{"type":"error","message":"something failed"}',
      );

      expect(msg.type, equals(WsMessageType.error));
      expect(msg.message, equals('something failed'));
    });

    test('parses vad_status message', () {
      final msg = WebSocketMessage.fromJson(
        '{"type":"vad_status","status":"speech_started"}',
      );

      expect(msg.type, equals(WsMessageType.vadStatus));
      expect(msg.status, equals('speech_started'));
    });
  });

  // ---------------------------------------------------------------------------
  // toJson
  // ---------------------------------------------------------------------------
  group('WebSocketMessage.toJson', () {
    test('produces JSON string with type field', () {
      final msg = WebSocketMessage(
        type: WsMessageType.text,
        data: {'type': WsMessageType.text, 'text': 'hi'},
      );

      final jsonStr = msg.toJson();
      final decoded = jsonDecode(jsonStr) as Map<String, dynamic>;

      expect(decoded['type'], equals(WsMessageType.text));
      expect(decoded['text'], equals('hi'));
    });

    test('type field is always included even when not in original data', () {
      final msg = WebSocketMessage(
        type: WsMessageType.interrupt,
        data: {'type': WsMessageType.interrupt},
      );

      final decoded = jsonDecode(msg.toJson()) as Map<String, dynamic>;
      expect(decoded['type'], equals(WsMessageType.interrupt));
    });
  });

  // ---------------------------------------------------------------------------
  // Factory constructors
  // ---------------------------------------------------------------------------
  group('WebSocketMessage factory constructors', () {
    test('text() sets type and text field', () {
      final msg = WebSocketMessage.text('hello backend');

      expect(msg.type, equals(WsMessageType.text));
      expect(msg.text, equals('hello backend'));
    });

    test('audio() sets type, audio list, and sample_rate', () {
      final samples = [0.1, -0.1, 0.5];
      final msg = WebSocketMessage.audio(audio: samples, sampleRate: 16000);

      expect(msg.type, equals(WsMessageType.audio));
      expect(msg.audio, equals(samples));
      expect(msg.sampleRate, equals(16000));
    });

    test('interrupt() sets type to interrupt with no extra fields', () {
      final msg = WebSocketMessage.interrupt();

      expect(msg.type, equals(WsMessageType.interrupt));
    });

    test('resetVad() sets type to reset_vad', () {
      final msg = WebSocketMessage.resetVad();

      expect(msg.type, equals(WsMessageType.resetVad));
    });
  });

  // ---------------------------------------------------------------------------
  // Getter accessors
  // ---------------------------------------------------------------------------
  group('WebSocketMessage getters', () {
    test('text getter returns null when field absent', () {
      final msg = WebSocketMessage(
        type: WsMessageType.status,
        data: {'type': WsMessageType.status},
      );
      expect(msg.text, isNull);
    });

    test('token getter returns token field', () {
      final msg = WebSocketMessage.fromJson(
        '{"type":"llm_token","token":"chunk"}',
      );
      expect(msg.token, equals('chunk'));
    });

    test('message getter returns message field', () {
      final msg = WebSocketMessage.fromJson(
        '{"type":"status","message":"online"}',
      );
      expect(msg.message, equals('online'));
    });

    test('status getter returns status field', () {
      final msg = WebSocketMessage.fromJson(
        '{"type":"vad_status","status":"silence"}',
      );
      expect(msg.status, equals('silence'));
    });

    test('sampleRate getter returns sample_rate field', () {
      final msg = WebSocketMessage.audio(audio: [], sampleRate: 24000);
      expect(msg.sampleRate, equals(24000));
    });

    test('chunkNum getter returns chunk_num field', () {
      final msg = WebSocketMessage.fromJson(
        '{"type":"audio_chunk","audio":[],"sample_rate":24000,"chunk_num":5}',
      );
      expect(msg.chunkNum, equals(5));
    });
  });

  // ---------------------------------------------------------------------------
  // Round-trip: toJson -> fromJson
  // ---------------------------------------------------------------------------
  group('WebSocketMessage round-trip', () {
    test('text message survives toJson -> fromJson', () {
      final original = WebSocketMessage.text('round-trip test');
      final recovered = WebSocketMessage.fromJson(original.toJson());

      expect(recovered.type, equals(original.type));
      expect(recovered.text, equals(original.text));
    });

    test('audio message preserves audio list and sample_rate', () {
      final samples = [0.1, -0.5, 0.9];
      final original = WebSocketMessage.audio(
        audio: samples,
        sampleRate: 16000,
      );
      final recovered = WebSocketMessage.fromJson(original.toJson());

      expect(recovered.type, equals(WsMessageType.audio));
      expect(recovered.sampleRate, equals(16000));
      // Audio values are encoded/decoded through JSON as num, compare doubles
      final recoveredAudio = recovered.audio!
          .map((e) => (e as num).toDouble())
          .toList();
      expect(recoveredAudio, equals(samples));
    });

    test('interrupt message round-trips cleanly', () {
      final original = WebSocketMessage.interrupt();
      final recovered = WebSocketMessage.fromJson(original.toJson());

      expect(recovered.type, equals(WsMessageType.interrupt));
    });

    test('resetVad message round-trips cleanly', () {
      final original = WebSocketMessage.resetVad();
      final recovered = WebSocketMessage.fromJson(original.toJson());

      expect(recovered.type, equals(WsMessageType.resetVad));
    });
  });

  // ---------------------------------------------------------------------------
  // fromMap
  // ---------------------------------------------------------------------------
  group('WebSocketMessage.fromMap', () {
    test('parses from Map correctly', () {
      final map = {'type': 'text', 'text': 'from map'};
      final msg = WebSocketMessage.fromMap(map);

      expect(msg.type, equals(WsMessageType.text));
      expect(msg.text, equals('from map'));
    });
  });

  // ---------------------------------------------------------------------------
  // WsMessageType constants
  // ---------------------------------------------------------------------------
  group('WsMessageType constants', () {
    test('client-side message type values are correct', () {
      expect(WsMessageType.audio, equals('audio'));
      expect(WsMessageType.text, equals('text'));
      expect(WsMessageType.interrupt, equals('interrupt'));
      expect(WsMessageType.resetVad, equals('reset_vad'));
    });

    test('server-side message type values are correct', () {
      expect(WsMessageType.status, equals('status'));
      expect(WsMessageType.vadStatus, equals('vad_status'));
      expect(WsMessageType.transcription, equals('transcription'));
      expect(WsMessageType.llmToken, equals('llm_token'));
      expect(WsMessageType.llmResponse, equals('llm_response'));
      expect(WsMessageType.audioChunk, equals('audio_chunk'));
      expect(WsMessageType.audioComplete, equals('audio_complete'));
      expect(WsMessageType.error, equals('error'));
    });
  });
}
