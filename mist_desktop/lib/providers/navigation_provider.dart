import 'package:flutter_riverpod/flutter_riverpod.dart';

/// Whether the sidebar navigation rail is expanded.
class SidebarExpanded extends Notifier<bool> {
  @override
  bool build() => false;

  void toggle() => state = !state;
}

final sidebarExpandedProvider = NotifierProvider<SidebarExpanded, bool>(
  SidebarExpanded.new,
);

/// The currently selected destination index in the navigation rail.
class SelectedDestination extends Notifier<int> {
  @override
  int build() => 0;

  void select(int index) => state = index;
}

final selectedDestinationProvider = NotifierProvider<SelectedDestination, int>(
  SelectedDestination.new,
);
