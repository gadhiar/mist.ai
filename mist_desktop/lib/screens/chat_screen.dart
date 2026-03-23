import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/message_model.dart';
import '../providers/chat_provider.dart';
import '../providers/websocket_provider.dart';
import '../services/websocket_service.dart';
import '../widgets/chat_message_widget.dart';
import '../widgets/connection_status_widget.dart';
import '../widgets/voice_input_button.dart';
import '../config/app_config.dart';

/// Main Chat Screen
class ChatScreen extends ConsumerStatefulWidget {
  const ChatScreen({super.key});

  @override
  ConsumerState<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends ConsumerState<ChatScreen> {
  final TextEditingController _textController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  int _lastMessageCount = 0;

  @override
  void initState() {
    super.initState();
    // Connect to WebSocket when screen loads
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _connectToServer();
    });
  }

  Future<void> _connectToServer() async {
    final wsService = ref.read(websocketServiceProvider);
    try {
      await wsService.connect();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to connect: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  void _scrollToBottom() {
    if (_scrollController.hasClients) {
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeOut,
      );
    }
  }

  void _sendMessage() {
    final text = _textController.text.trim();
    if (text.isEmpty) return;

    // Send message via provider
    ref.read(chatProvider.notifier).sendTextMessage(text);

    // Clear input
    _textController.clear();

    // Scroll to bottom
    Future.delayed(const Duration(milliseconds: 100), _scrollToBottom);
  }

  @override
  Widget build(BuildContext context) {
    final chatState = ref.watch(chatProvider);
    final connectionStatus = ref.watch(connectionStatusProvider);

    // Auto-scroll only when the message count increases
    final totalItems =
        chatState.messages.length +
        (chatState.currentAiResponse != null ? 1 : 0);
    if (totalItems > _lastMessageCount) {
      _lastMessageCount = totalItems;
      WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToBottom());
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text(AppConfig.appName),
        actions: [
          const ConnectionStatusWidget(),
          const SizedBox(width: 8),
          // Reconnect button
          connectionStatus.when(
            data: (status) {
              if (status == ConnectionStatus.disconnected ||
                  status == ConnectionStatus.error) {
                return IconButton(
                  icon: const Icon(Icons.refresh),
                  tooltip: 'Reconnect',
                  onPressed: _connectToServer,
                );
              }
              return const SizedBox.shrink();
            },
            loading: () => const SizedBox.shrink(),
            error: (_, __) => const SizedBox.shrink(),
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: Column(
        children: [
          // Chat messages
          Expanded(
            child: chatState.messages.isEmpty
                ? _buildEmptyState()
                : ListView.builder(
                    controller: _scrollController,
                    padding: const EdgeInsets.all(8),
                    itemCount:
                        chatState.messages.length +
                        (chatState.currentAiResponse != null ? 1 : 0),
                    itemBuilder: (context, index) {
                      // Show streaming AI response
                      if (index == chatState.messages.length &&
                          chatState.currentAiResponse != null) {
                        return ChatMessageWidget(
                          message: ChatMessage.ai(chatState.currentAiResponse!),
                          isStreaming: true,
                        );
                      }

                      final message = chatState.messages[index];
                      return ChatMessageWidget(message: message);
                    },
                  ),
          ),

          // Processing indicator
          if (chatState.isProcessing)
            Container(
              padding: const EdgeInsets.symmetric(vertical: 8),
              child: const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  ),
                  SizedBox(width: 8),
                  Text('Processing...', style: TextStyle(fontSize: 12)),
                ],
              ),
            ),

          // Voice status indicator
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 8),
            child: Center(child: VoiceInputIndicator()),
          ),

          // Input area
          _buildInputArea(connectionStatus),
        ],
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.chat_bubble_outline,
            size: 64,
            color: Colors.grey.shade600,
          ),
          const SizedBox(height: 16),
          Text(
            'Start a conversation with MIST.AI',
            style: TextStyle(fontSize: 16, color: Colors.grey.shade600),
          ),
        ],
      ),
    );
  }

  Widget _buildInputArea(AsyncValue<ConnectionStatus> connectionStatus) {
    final isConnected = connectionStatus.value == ConnectionStatus.connected;

    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 4,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Row(
        children: [
          // Voice input button
          const VoiceInputButton(),
          const SizedBox(width: 8),
          // Text input
          Expanded(
            child: TextField(
              controller: _textController,
              enabled: isConnected,
              decoration: const InputDecoration(
                hintText: 'Type a message...',
                border: OutlineInputBorder(),
              ),
              maxLines: null,
              textInputAction: TextInputAction.send,
              onSubmitted: isConnected ? (_) => _sendMessage() : null,
            ),
          ),
          const SizedBox(width: 8),
          // Send button
          IconButton.filled(
            icon: const Icon(Icons.send),
            onPressed: isConnected ? _sendMessage : null,
            tooltip: 'Send',
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _textController.dispose();
    _scrollController.dispose();
    super.dispose();
  }
}
