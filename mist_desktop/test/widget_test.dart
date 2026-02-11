// Basic widget test for MIST.AI desktop app.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:mist_desktop/main.dart';

void main() {
  testWidgets('App loads ChatScreen', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const ProviderScope(child: MistAIApp()));

    // Verify that the app loads without crashing.
    expect(find.byType(MaterialApp), findsOneWidget);
  });
}
