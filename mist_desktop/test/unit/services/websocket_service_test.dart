// Tests for WebSocketService logic that does not require a real WebSocket.
//
// NOTE: _convertPCM16ToFloat32 is a private method. The conversion logic is
// tested indirectly here by mirroring the same algorithm. For direct testing,
// extract _convertPCM16ToFloat32 to lib/services/audio_utils.dart (see TODO
// in audio_playback_service_test.dart).
//
// What IS tested:
//   - sendMessage throws when not connected
//   - sendText / sendInterrupt / resetVad delegate to sendMessage and throw
//     when not connected
//   - Status is disconnected on construction
//   - PCM16 -> float32 conversion math (mirrored logic)

import 'package:flutter_test/flutter_test.dart';
import 'package:mist_desktop/services/websocket_service.dart';
import 'package:mist_desktop/models/websocket_message.dart';

void main() {
  // ---------------------------------------------------------------------------
  // Initial state
  // ---------------------------------------------------------------------------
  group('WebSocketService initial state', () {
    test('status is disconnected on construction', () {
      final service = WebSocketService();
      expect(service.status, equals(ConnectionStatus.disconnected));
      service.dispose();
    });

    test('statusStream is a broadcast stream', () {
      final service = WebSocketService();
      expect(service.statusStream.isBroadcast, isTrue);
      service.dispose();
    });

    test('messageStream is a broadcast stream', () {
      final service = WebSocketService();
      expect(service.messageStream.isBroadcast, isTrue);
      service.dispose();
    });
  });

  // ---------------------------------------------------------------------------
  // sendMessage throws when not connected
  // ---------------------------------------------------------------------------
  group('WebSocketService.sendMessage', () {
    test('throws Exception when status is disconnected', () {
      final service = WebSocketService();

      expect(
        () => service.sendMessage(WebSocketMessage.text('hello')),
        throwsA(
          isA<Exception>().having(
            (e) => e.toString(),
            'message',
            contains('not connected'),
          ),
        ),
      );

      service.dispose();
    });

    test('sendText throws when not connected', () {
      final service = WebSocketService();

      expect(() => service.sendText('hello'), throwsException);

      service.dispose();
    });

    test('sendInterrupt throws when not connected', () {
      final service = WebSocketService();

      expect(() => service.sendInterrupt(), throwsException);

      service.dispose();
    });

    test('resetVad throws when not connected', () {
      final service = WebSocketService();

      expect(() => service.resetVad(), throwsException);

      service.dispose();
    });

    test('sendAudioBytes throws when not connected', () {
      final service = WebSocketService();

      expect(() => service.sendAudioBytes([0x00, 0x00]), throwsException);

      service.dispose();
    });
  });

  // ---------------------------------------------------------------------------
  // PCM16 -> float32 conversion math (mirrors production logic)
  // ---------------------------------------------------------------------------
  group('PCM16 to float32 conversion math', () {
    // This mirrors WebSocketService._convertPCM16ToFloat32 exactly.
    // When that method is extracted to audio_utils.dart, replace this
    // mirror with a direct import.
    List<double> convertPCM16ToFloat32(List<int> bytes) {
      final samples = <double>[];
      for (int i = 0; i < bytes.length - 1; i += 2) {
        final sample = (bytes[i] | (bytes[i + 1] << 8)).toSigned(16);
        samples.add(sample / 32768.0);
      }
      return samples;
    }

    test('silence [0x00, 0x00] produces 0.0', () {
      expect(convertPCM16ToFloat32([0x00, 0x00]), equals([0.0]));
    });

    test('max positive [0xFF, 0x7F] produces value close to 1.0', () {
      // 0x7FFF = 32767; 32767 / 32768 ≈ 0.9999695...
      final result = convertPCM16ToFloat32([0xFF, 0x7F]);
      expect(result.first, closeTo(1.0, 1e-4));
    });

    test('min negative [0x00, 0x80] produces -1.0', () {
      // 0x8000 as signed 16-bit = -32768; -32768 / 32768 = -1.0
      final result = convertPCM16ToFloat32([0x00, 0x80]);
      expect(result.first, equals(-1.0));
    });

    test('midpoint positive [0x00, 0x40] produces 0.5', () {
      // 0x4000 = 16384; 16384 / 32768 = 0.5
      final result = convertPCM16ToFloat32([0x00, 0x40]);
      expect(result.first, closeTo(0.5, 1e-6));
    });

    test('multiple sample pairs are all converted', () {
      // Two samples: silence + max positive
      final result = convertPCM16ToFloat32([0x00, 0x00, 0xFF, 0x7F]);
      expect(result.length, equals(2));
      expect(result[0], equals(0.0));
      expect(result[1], closeTo(1.0, 1e-4));
    });

    test('odd trailing byte is ignored', () {
      final result = convertPCM16ToFloat32([0x00, 0x00, 0xFF]);
      expect(result.length, equals(1));
    });

    test('empty input produces empty output', () {
      expect(convertPCM16ToFloat32([]), isEmpty);
    });
  });

  // ---------------------------------------------------------------------------
  // ConnectionStatus enum values
  // ---------------------------------------------------------------------------
  group('ConnectionStatus', () {
    test('all status values are distinct', () {
      const statuses = ConnectionStatus.values;
      expect(statuses.toSet().length, equals(statuses.length));
    });

    test('expected status values exist', () {
      expect(ConnectionStatus.values, contains(ConnectionStatus.disconnected));
      expect(ConnectionStatus.values, contains(ConnectionStatus.connecting));
      expect(ConnectionStatus.values, contains(ConnectionStatus.connected));
      expect(ConnectionStatus.values, contains(ConnectionStatus.error));
    });
  });
}
