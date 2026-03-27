import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:mist_desktop/models/binary_audio_frame.dart';

void main() {
  Uint8List buildTestFrame({
    int messageType = 0x01,
    int sessionId = 0,
    int chunkSeq = 1,
    int sampleRate = 24000,
    List<int> payload = const [],
  }) {
    final buffer = ByteData(16 + payload.length);
    buffer.setUint32(0, 0x4D495354, Endian.little);
    buffer.setUint16(4, messageType, Endian.little);
    buffer.setUint16(6, sessionId, Endian.little);
    buffer.setUint32(8, chunkSeq, Endian.little);
    buffer.setUint32(12, sampleRate, Endian.little);
    final bytes = buffer.buffer.asUint8List();
    for (int i = 0; i < payload.length; i++) {
      bytes[16 + i] = payload[i];
    }
    return bytes;
  }

  group('BinaryAudioFrame.parse', () {
    test('parses valid audio_chunk frame', () {
      final data = buildTestFrame(
        messageType: 0x01,
        sessionId: 0,
        chunkSeq: 5,
        sampleRate: 24000,
        payload: [0x00, 0x01, 0x02, 0x03],
      );
      final frame = BinaryAudioFrame.parse(data);
      expect(frame, isNotNull);
      expect(frame!.messageType, equals(0x01));
      expect(frame.sessionId, equals(0));
      expect(frame.chunkSeq, equals(5));
      expect(frame.sampleRate, equals(24000));
      expect(frame.payload.length, equals(4));
    });

    test('parses audio_complete with empty payload', () {
      final data = buildTestFrame(messageType: 0x02, chunkSeq: 10);
      final frame = BinaryAudioFrame.parse(data);
      expect(frame, isNotNull);
      expect(frame!.messageType, equals(0x02));
      expect(frame.payload.length, equals(0));
    });

    test('parses interrupt_fade frame', () {
      final data = buildTestFrame(
        messageType: 0x03,
        payload: List.filled(960, 0),
      );
      final frame = BinaryAudioFrame.parse(data);
      expect(frame, isNotNull);
      expect(frame!.messageType, equals(0x03));
      expect(frame.payload.length, equals(960));
    });

    test('returns null for invalid magic bytes', () {
      final data = Uint8List(20);
      final buffer = ByteData.view(data.buffer);
      buffer.setUint32(0, 0xDEADBEEF, Endian.little);
      final frame = BinaryAudioFrame.parse(data);
      expect(frame, isNull);
    });

    test('returns null for data shorter than header', () {
      final data = Uint8List(10);
      final frame = BinaryAudioFrame.parse(data);
      expect(frame, isNull);
    });

    test('isAudioChunk returns true for type 0x01', () {
      final data = buildTestFrame(messageType: 0x01);
      final frame = BinaryAudioFrame.parse(data)!;
      expect(frame.isAudioChunk, isTrue);
      expect(frame.isAudioComplete, isFalse);
      expect(frame.isInterruptFade, isFalse);
    });

    test('isAudioComplete returns true for type 0x02', () {
      final data = buildTestFrame(messageType: 0x02);
      final frame = BinaryAudioFrame.parse(data)!;
      expect(frame.isAudioComplete, isTrue);
    });

    test('isInterruptFade returns true for type 0x03', () {
      final data = buildTestFrame(messageType: 0x03);
      final frame = BinaryAudioFrame.parse(data)!;
      expect(frame.isInterruptFade, isTrue);
    });
  });
}
