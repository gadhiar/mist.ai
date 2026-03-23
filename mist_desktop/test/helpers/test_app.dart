import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

/// Wraps [child] in a [ProviderScope] + [MaterialApp] suitable for widget
/// tests. Pass [overrides] to replace real providers with fakes.
///
/// The [overrides] list accepts values produced by provider.overrideWithValue()
/// or provider.overrideWith() -- the same list accepted by ProviderScope.overrides.
Widget buildTestApp({
  required Widget child,
  List<Object> overrides = const [],
}) {
  // ProviderScope.overrides is List<Override> (a sealed riverpod-internal type
  // not exposed in the public API). We cast here to satisfy the type system.
  // Callers should pass values from provider.overrideWith/overrideWithValue.
  return ProviderScope(
    overrides: overrides.cast(),
    child: MaterialApp(home: child),
  );
}
