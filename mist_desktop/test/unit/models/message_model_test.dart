import 'package:flutter_test/flutter_test.dart';
import 'package:mist_desktop/models/message_model.dart';

void main() {
  group('ChatMessage factory constructors', () {
    test('user() creates message with type user', () {
      final msg = ChatMessage.user('hello');

      expect(msg.type, equals(MessageType.user));
      expect(msg.text, equals('hello'));
    });

    test('ai() creates message with type ai', () {
      final msg = ChatMessage.ai('response text');

      expect(msg.type, equals(MessageType.ai));
      expect(msg.text, equals('response text'));
    });

    test('system() creates message with type system', () {
      final msg = ChatMessage.system('connected');

      expect(msg.type, equals(MessageType.system));
      expect(msg.text, equals('connected'));
    });

    test('error() creates message with type error', () {
      final msg = ChatMessage.error('something went wrong');

      expect(msg.type, equals(MessageType.error));
      expect(msg.text, equals('something went wrong'));
    });
  });

  group('ChatMessage unique IDs', () {
    test('two messages created without an explicit id get different ids', () {
      final a = ChatMessage.user('a');
      final b = ChatMessage.user('b');

      expect(a.id, isNotEmpty);
      expect(b.id, isNotEmpty);
      expect(a.id, isNot(equals(b.id)));
    });

    test('explicit id is preserved', () {
      const explicitId = 'test-id-123';
      final msg = ChatMessage(
        id: explicitId,
        text: 'x',
        type: MessageType.user,
      );

      expect(msg.id, equals(explicitId));
    });
  });

  group('ChatMessage.copyWith', () {
    test('returns a new instance with the updated field', () {
      final original = ChatMessage.user('original');
      final copy = original.copyWith(text: 'updated');

      expect(copy.text, equals('updated'));
      expect(copy.type, equals(MessageType.user));
      expect(copy.id, equals(original.id));
      expect(copy.timestamp, equals(original.timestamp));
    });

    test('preserves all fields when no arguments are passed', () {
      final ts = DateTime(2026, 1, 15, 10, 30);
      final original = ChatMessage(
        id: 'fixed-id',
        text: 'hello',
        type: MessageType.ai,
        timestamp: ts,
        metadata: {'key': 'value'},
      );

      final copy = original.copyWith();

      expect(copy.id, equals(original.id));
      expect(copy.text, equals(original.text));
      expect(copy.type, equals(original.type));
      expect(copy.timestamp, equals(original.timestamp));
      expect(copy.metadata, equals(original.metadata));
    });

    test('can override type', () {
      final original = ChatMessage.user('text');
      final copy = original.copyWith(type: MessageType.error);

      expect(copy.type, equals(MessageType.error));
      expect(copy.text, equals('text'));
    });
  });

  group('ChatMessage default timestamp', () {
    test('timestamp is set automatically when not provided', () {
      final before = DateTime.now();
      final msg = ChatMessage.user('now');
      final after = DateTime.now();

      expect(
        msg.timestamp.isAfter(before) || msg.timestamp.isAtSameMomentAs(before),
        isTrue,
      );
      expect(
        msg.timestamp.isBefore(after) || msg.timestamp.isAtSameMomentAs(after),
        isTrue,
      );
    });

    test('explicit timestamp is preserved', () {
      final ts = DateTime(2025, 6, 1);
      final msg = ChatMessage(
        text: 'x',
        type: MessageType.system,
        timestamp: ts,
      );

      expect(msg.timestamp, equals(ts));
    });
  });
}
