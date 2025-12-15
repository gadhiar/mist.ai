import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../models/message_model.dart';
import '../config/theme_config.dart';

/// Chat Message Widget
class ChatMessageWidget extends StatelessWidget {
  final ChatMessage message;

  const ChatMessageWidget({
    super.key,
    required this.message,
  });

  @override
  Widget build(BuildContext context) {
    final isUser = message.type == MessageType.user;
    final isSystem = message.type == MessageType.system;
    final isError = message.type == MessageType.error;

    Color backgroundColor;
    Color textColor;
    IconData? icon;

    if (isUser) {
      backgroundColor = ThemeConfig.userMessageBg;
      textColor = ThemeConfig.textPrimary;
      icon = Icons.person;
    } else if (isSystem) {
      backgroundColor = ThemeConfig.systemMessageBg;
      textColor = ThemeConfig.textPrimary;
      icon = Icons.info_outline;
    } else if (isError) {
      backgroundColor = ThemeConfig.errorColor;
      textColor = ThemeConfig.textPrimary;
      icon = Icons.error_outline;
    } else {
      backgroundColor = ThemeConfig.aiMessageBg;
      textColor = ThemeConfig.textPrimary;
      icon = Icons.smart_toy;
    }

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.7,
        ),
        margin: const EdgeInsets.symmetric(vertical: 4, horizontal: 8),
        child: Card(
          color: backgroundColor,
          elevation: 1,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                // Header with icon and timestamp
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      icon,
                      size: 16,
                      color: textColor.withOpacity(0.7),
                    ),
                    const SizedBox(width: 4),
                    Text(
                      _getLabel(),
                      style: TextStyle(
                        fontSize: 12,
                        color: textColor.withOpacity(0.7),
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    const Spacer(),
                    Text(
                      _formatTime(message.timestamp),
                      style: TextStyle(
                        fontSize: 11,
                        color: textColor.withOpacity(0.5),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                // Message text
                SelectableText(
                  message.text,
                  style: TextStyle(
                    fontSize: 14,
                    color: textColor,
                    height: 1.4,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  String _getLabel() {
    switch (message.type) {
      case MessageType.user:
        return 'You';
      case MessageType.ai:
        return 'MIST.AI';
      case MessageType.system:
        return 'System';
      case MessageType.error:
        return 'Error';
    }
  }

  String _formatTime(DateTime time) {
    return DateFormat('HH:mm').format(time);
  }
}
