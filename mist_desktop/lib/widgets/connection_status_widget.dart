import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../services/websocket_service.dart';
import '../providers/websocket_provider.dart';
import '../config/theme_config.dart';

/// Connection Status Widget
class ConnectionStatusWidget extends ConsumerWidget {
  const ConnectionStatusWidget({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final statusAsync = ref.watch(connectionStatusProvider);

    return statusAsync.when(
      data: (status) => _buildStatusIndicator(status),
      loading: () => _buildStatusIndicator(ConnectionStatus.connecting),
      error: (_, __) => _buildStatusIndicator(ConnectionStatus.error),
    );
  }

  Widget _buildStatusIndicator(ConnectionStatus status) {
    Color color;
    String label;
    IconData icon;

    switch (status) {
      case ConnectionStatus.connected:
        color = ThemeConfig.connectedColor;
        label = 'Connected';
        icon = Icons.check_circle;
        break;
      case ConnectionStatus.connecting:
        color = ThemeConfig.processingColor;
        label = 'Connecting...';
        icon = Icons.sync;
        break;
      case ConnectionStatus.disconnected:
        color = ThemeConfig.disconnectedColor;
        label = 'Disconnected';
        icon = Icons.cancel;
        break;
      case ConnectionStatus.error:
        color = ThemeConfig.errorColor;
        label = 'Error';
        icon = Icons.error;
        break;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.2),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color, width: 1),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            icon,
            size: 14,
            color: color,
          ),
          const SizedBox(width: 6),
          Text(
            label,
            style: TextStyle(
              fontSize: 12,
              color: color,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
}
