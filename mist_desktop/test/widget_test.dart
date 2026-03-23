// Basic widget test for MIST.AI desktop app.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:mist_desktop/main.dart';
import 'package:mist_desktop/providers/websocket_provider.dart';
import 'package:mist_desktop/providers/audio_provider.dart';

import 'helpers/fakes.dart';

void main() {
  testWidgets('App loads ChatScreen', (WidgetTester tester) async {
    // Use fake services to prevent real WebSocket/audio platform calls.
    final fakeWs = FakeWebSocketService();
    final fakeRecording = FakeAudioRecordingService();
    final fakePlayback = FakeAudioPlaybackService();

    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          websocketServiceProvider.overrideWithValue(fakeWs),
          audioRecordingServiceProvider.overrideWithValue(fakeRecording),
          audioPlaybackServiceProvider.overrideWithValue(fakePlayback),
        ],
        child: const MistAIApp(),
      ),
    );

    // Verify that the app loads without crashing.
    expect(find.byType(MaterialApp), findsOneWidget);
  });
}
