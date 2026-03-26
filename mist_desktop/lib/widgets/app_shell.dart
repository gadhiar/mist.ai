import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/navigation_provider.dart';
import '../providers/log_provider.dart';
import '../screens/chat_screen.dart';
import '../screens/log_screen.dart';
import '../screens/stub_screen.dart';

/// Collapsed width of the navigation rail (icon only).
const double _railCollapsedWidth = 72;

/// Expanded width of the navigation rail (icon + label).
const double _railExpandedWidth = 200;

/// Duration of the sidebar expand/collapse animation.
const Duration _animationDuration = Duration(milliseconds: 200);

/// Root scaffold wrapping NavigationRail + content area.
///
/// Replaces ChatScreen as the MaterialApp home widget. Eagerly initializes
/// the log provider so log entries are captured from the moment of connection.
class AppShell extends ConsumerWidget {
  const AppShell({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final isExpanded = ref.watch(sidebarExpandedProvider);
    final selectedIndex = ref.watch(selectedDestinationProvider);

    // Eager init: ensure LogNotifier starts capturing immediately,
    // even before the user navigates to the Logs screen.
    ref.read(logProvider);

    return Scaffold(
      body: Row(
        children: [
          // Sidebar rail
          _SidebarRail(isExpanded: isExpanded, selectedIndex: selectedIndex),
          // Content area
          Expanded(child: _ContentArea(selectedIndex: selectedIndex)),
        ],
      ),
    );
  }
}

/// Animated navigation rail with chevron toggle.
class _SidebarRail extends ConsumerWidget {
  final bool isExpanded;
  final int selectedIndex;

  const _SidebarRail({required this.isExpanded, required this.selectedIndex});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final targetWidth = isExpanded ? _railExpandedWidth : _railCollapsedWidth;

    return AnimatedContainer(
      duration: _animationDuration,
      curve: Curves.easeInOut,
      width: targetWidth,
      color: const Color(0xFF161616),
      child: Column(
        children: [
          // Toggle button
          const SizedBox(height: 8),
          _buildToggleButton(ref),
          const SizedBox(height: 8),
          Divider(
            height: 1,
            thickness: 1,
            color: Colors.white.withValues(alpha: 0.06),
          ),
          // Destinations
          Expanded(
            child: NavigationRail(
              selectedIndex: selectedIndex,
              extended: isExpanded,
              minWidth: _railCollapsedWidth,
              minExtendedWidth: _railExpandedWidth,
              backgroundColor: Colors.transparent,
              indicatorColor: Theme.of(
                context,
              ).colorScheme.primary.withValues(alpha: 0.2),
              selectedIconTheme: IconThemeData(
                color: Theme.of(context).colorScheme.primary,
              ),
              selectedLabelTextStyle: TextStyle(
                color: Theme.of(context).colorScheme.primary,
                fontSize: 13,
                fontWeight: FontWeight.w500,
              ),
              unselectedIconTheme: IconThemeData(color: Colors.grey.shade500),
              unselectedLabelTextStyle: TextStyle(
                color: Colors.grey.shade500,
                fontSize: 13,
              ),
              labelType: NavigationRailLabelType.none,
              onDestinationSelected: (index) {
                ref.read(selectedDestinationProvider.notifier).select(index);
              },
              destinations: const [
                NavigationRailDestination(
                  icon: Icon(Icons.chat_bubble_outline),
                  selectedIcon: Icon(Icons.chat_bubble),
                  label: Text('Chat'),
                ),
                NavigationRailDestination(
                  icon: Icon(Icons.terminal),
                  selectedIcon: Icon(Icons.terminal),
                  label: Text('Logs'),
                ),
                NavigationRailDestination(
                  icon: Icon(Icons.record_voice_over),
                  selectedIcon: Icon(Icons.record_voice_over),
                  label: Text('Voice'),
                ),
                NavigationRailDestination(
                  icon: Icon(Icons.settings_outlined),
                  selectedIcon: Icon(Icons.settings),
                  label: Text('Settings'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildToggleButton(WidgetRef ref) {
    return Align(
      alignment: isExpanded ? Alignment.centerRight : Alignment.center,
      child: Padding(
        padding: EdgeInsets.only(right: isExpanded ? 8 : 0),
        child: IconButton(
          icon: Icon(
            isExpanded ? Icons.chevron_left : Icons.chevron_right,
            color: Colors.grey.shade500,
            size: 20,
          ),
          tooltip: isExpanded ? 'Collapse sidebar' : 'Expand sidebar',
          splashRadius: 16,
          onPressed: () {
            ref.read(sidebarExpandedProvider.notifier).toggle();
          },
        ),
      ),
    );
  }
}

/// Switches the visible content based on the selected destination index.
class _ContentArea extends StatelessWidget {
  final int selectedIndex;

  const _ContentArea({required this.selectedIndex});

  @override
  Widget build(BuildContext context) {
    return IndexedStack(
      index: selectedIndex,
      children: const [
        ChatScreen(),
        LogScreen(),
        StubScreen(destinationName: 'Voice Profiles'),
        StubScreen(destinationName: 'Settings'),
      ],
    );
  }
}
