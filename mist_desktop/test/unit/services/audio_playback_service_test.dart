// Tests for AudioPlaybackService pure-logic methods.
//
// NOTE: _float32ToPCM16 and _addWavHeader are private methods on
// AudioPlaybackService. They cannot be called directly from tests without
// either (a) using @visibleForTesting annotations, or (b) extracting them as
// package-visible utility functions.
//
// Recommendation: extract both as top-level functions in a separate file
// (e.g., lib/services/audio_utils.dart) so they can be tested without
// coupling to AudioPlayer initialization. The tests below verify the
// conversion logic indirectly via observable behavior documented in comments.
//
// What IS tested here:
//   - PCM16 conversion math (verified via independent Dart logic mirroring
//     the production code, used to state the expected values)
//   - WAV header structure expectations documented as constants
//
// Concrete unit tests for the conversion functions will be added once the
// methods are extracted to a public utility (see TODO below).

// TODO: Extract AudioPlaybackService._float32ToPCM16 and ._addWavHeader to
//       lib/services/audio_utils.dart as public functions, then import and
//       test them directly here.

import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';

void main() {
  // ---------------------------------------------------------------------------
  // PCM16 conversion math -- verified independently
  //
  // The production code in AudioPlaybackService._float32ToPCM16 does:
  //   pcm16Value = (floatValue * 32767).round().clamp(-32768, 32767)
  //   bytes: [pcm16Value & 0xFF, (pcm16Value >> 8) & 0xFF]  (little-endian)
  //
  // The production code in WebSocketService._convertPCM16ToFloat32 does:
  //   sample = (bytes[i] | (bytes[i+1] << 8)).toSigned(16)
  //   float  = sample / 32768.0
  // ---------------------------------------------------------------------------
  group('float32 to PCM16 conversion math', () {
    // Mirror the production conversion logic so we can assert known values
    // without calling the private method directly.
    List<int> float32ToPcm16(List<double> samples) {
      final result = <int>[];
      for (final v in samples) {
        final pcm = (v * 32767).round().clamp(-32768, 32767);
        result.add(pcm & 0xFF);
        result.add((pcm >> 8) & 0xFF);
      }
      return result;
    }

    test('0.0 maps to [0x00, 0x00]', () {
      expect(float32ToPcm16([0.0]), equals([0x00, 0x00]));
    });

    test('1.0 maps to [0xFF, 0x7F] (32767 little-endian)', () {
      // 32767 = 0x7FFF -> little-endian bytes: 0xFF, 0x7F
      expect(float32ToPcm16([1.0]), equals([0xFF, 0x7F]));
    });

    test('-1.0 maps to [0x01, 0x80] (-32767 clamped, little-endian)', () {
      // -32767 * ... => (-1.0 * 32767).round() = -32767
      // -32767 & 0xFF = 0x01, (-32767 >> 8) & 0xFF = 0x80
      expect(float32ToPcm16([-1.0]), equals([0x01, 0x80]));
    });

    test('0.5 produces positive mid-range value', () {
      final bytes = float32ToPcm16([0.5]);
      // (0.5 * 32767).round() = 16384 = 0x4000
      // little-endian: 0x00, 0x40
      expect(bytes, equals([0x00, 0x40]));
    });

    test('multiple samples produce correct byte count', () {
      final bytes = float32ToPcm16([0.0, 1.0, -1.0]);
      // 3 samples * 2 bytes each = 6 bytes
      expect(bytes.length, equals(6));
    });

    test('empty input produces empty output', () {
      expect(float32ToPcm16([]), isEmpty);
    });
  });

  // ---------------------------------------------------------------------------
  // PCM16 to float32 conversion math (mirrors WebSocketService logic)
  // ---------------------------------------------------------------------------
  group('PCM16 to float32 conversion math', () {
    List<double> pcm16ToFloat32(List<int> bytes) {
      final samples = <double>[];
      for (int i = 0; i < bytes.length - 1; i += 2) {
        final sample = (bytes[i] | (bytes[i + 1] << 8)).toSigned(16);
        samples.add(sample / 32768.0);
      }
      return samples;
    }

    test('[0x00, 0x00] maps to 0.0', () {
      final result = pcm16ToFloat32([0x00, 0x00]);
      expect(result, equals([0.0]));
    });

    test('[0xFF, 0x7F] maps to ~1.0 (32767/32768)', () {
      final result = pcm16ToFloat32([0xFF, 0x7F]);
      expect(result.first, closeTo(32767 / 32768.0, 1e-6));
    });

    test('[0x00, 0x80] maps to -1.0 (-32768/32768)', () {
      // 0x8000 as signed 16-bit = -32768
      final result = pcm16ToFloat32([0x00, 0x80]);
      expect(result.first, closeTo(-1.0, 1e-6));
    });

    test('odd trailing byte is ignored', () {
      // 3 bytes -> only first pair is processed
      final result = pcm16ToFloat32([0x00, 0x00, 0xFF]);
      expect(result.length, equals(1));
    });

    test('empty input produces empty output', () {
      expect(pcm16ToFloat32([]), isEmpty);
    });
  });

  // ---------------------------------------------------------------------------
  // WAV header structure
  // ---------------------------------------------------------------------------
  group('WAV header structure', () {
    // Mirror _addWavHeader logic to verify structure.
    Uint8List buildWavHeader(int pcmDataLength, int sampleRate) {
      const numChannels = 1;
      const bitsPerSample = 16;
      final byteRate = sampleRate * numChannels * (bitsPerSample ~/ 8);
      const blockAlign = numChannels * (bitsPerSample ~/ 8);
      final dataSize = pcmDataLength;
      final fileSize = 36 + dataSize;

      final header = ByteData(44);
      header.setUint8(0, 0x52); // R
      header.setUint8(1, 0x49); // I
      header.setUint8(2, 0x46); // F
      header.setUint8(3, 0x46); // F
      header.setUint32(4, fileSize, Endian.little);
      header.setUint8(8, 0x57); // W
      header.setUint8(9, 0x41); // A
      header.setUint8(10, 0x56); // V
      header.setUint8(11, 0x45); // E
      header.setUint8(12, 0x66); // f
      header.setUint8(13, 0x6D); // m
      header.setUint8(14, 0x74); // t
      header.setUint8(15, 0x20); // ' '
      header.setUint32(16, 16, Endian.little);
      header.setUint16(20, 1, Endian.little);
      header.setUint16(22, numChannels, Endian.little);
      header.setUint32(24, sampleRate, Endian.little);
      header.setUint32(28, byteRate, Endian.little);
      header.setUint16(32, blockAlign, Endian.little);
      header.setUint16(34, bitsPerSample, Endian.little);
      header.setUint8(36, 0x64); // d
      header.setUint8(37, 0x61); // a
      header.setUint8(38, 0x74); // t
      header.setUint8(39, 0x61); // a
      header.setUint32(40, dataSize, Endian.little);
      return header.buffer.asUint8List();
    }

    test('header is exactly 44 bytes', () {
      final header = buildWavHeader(100, 24000);
      expect(header.length, equals(44));
    });

    test('starts with RIFF marker', () {
      final header = buildWavHeader(0, 24000);
      // bytes 0-3: 'R','I','F','F'
      expect(header[0], equals(0x52));
      expect(header[1], equals(0x49));
      expect(header[2], equals(0x46));
      expect(header[3], equals(0x46));
    });

    test('contains WAVE marker at bytes 8-11', () {
      final header = buildWavHeader(0, 24000);
      expect(header[8], equals(0x57)); // W
      expect(header[9], equals(0x41)); // A
      expect(header[10], equals(0x56)); // V
      expect(header[11], equals(0x45)); // E
    });

    test('file size field equals 36 + pcmDataLength', () {
      const pcmLen = 200;
      final header = buildWavHeader(pcmLen, 24000);
      final fileSize = ByteData.sublistView(header).getUint32(4, Endian.little);
      expect(fileSize, equals(36 + pcmLen));
    });

    test('data chunk size field matches pcmDataLength', () {
      const pcmLen = 480;
      final header = buildWavHeader(pcmLen, 24000);
      final dataSize = ByteData.sublistView(
        header,
      ).getUint32(40, Endian.little);
      expect(dataSize, equals(pcmLen));
    });

    test('sample rate field is encoded correctly', () {
      const sr = 16000;
      final header = buildWavHeader(0, sr);
      final encodedSr = ByteData.sublistView(
        header,
      ).getUint32(24, Endian.little);
      expect(encodedSr, equals(sr));
    });
  });
}
