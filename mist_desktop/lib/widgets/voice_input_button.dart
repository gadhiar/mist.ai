import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/audio_provider.dart';
import '../providers/chat_provider.dart';
import '../config/theme_config.dart';

/// Voice Input Button - Push to talk or toggle
class VoiceInputButton extends ConsumerWidget {
  const VoiceInputButton({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final isRecordingAsync = ref.watch(isRecordingProvider);

    return isRecordingAsync.when(
      data: (isRecording) => _buildButton(context, ref, isRecording),
      loading: () => _buildButton(context, ref, false),
      error: (_, __) => _buildButton(context, ref, false),
    );
  }

  Widget _buildButton(BuildContext context, WidgetRef ref, bool isRecording) {
    return IconButton.filled(
      icon: Icon(
        isRecording ? Icons.mic : Icons.mic_none,
        size: 28,
      ),
      style: IconButton.styleFrom(
        backgroundColor: isRecording
            ? ThemeConfig.processingColor
            : ThemeConfig.primaryColor,
        foregroundColor: ThemeConfig.textPrimary,
        padding: const EdgeInsets.all(16),
      ),
      onPressed: () => _handleButtonPress(ref),
      tooltip: isRecording ? 'Stop recording' : 'Start recording',
    );
  }

  void _handleButtonPress(WidgetRef ref) {
    final audioService = ref.read(audioRecordingServiceProvider);
    final chatNotifier = ref.read(chatProvider.notifier);

    // Use the service's actual state, not the stream state
    if (audioService.isRecording) {
      chatNotifier.stopVoiceInput();
    } else {
      chatNotifier.startVoiceInput();
    }
  }
}

/// Voice Input Indicator - Shows when recording/speaking
class VoiceInputIndicator extends ConsumerWidget {
  const VoiceInputIndicator({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final isRecordingAsync = ref.watch(isRecordingProvider);
    final isPlayingAsync = ref.watch(isPlayingAudioProvider);

    return isRecordingAsync.when(
      data: (isRecording) {
        if (isRecording) {
          return _buildRecordingIndicator();
        }

        return isPlayingAsync.when(
          data: (isPlaying) {
            if (isPlaying) {
              return _buildPlayingIndicator();
            }
            return const SizedBox.shrink();
          },
          loading: () => const SizedBox.shrink(),
          error: (_, __) => const SizedBox.shrink(),
        );
      },
      loading: () => const SizedBox.shrink(),
      error: (_, __) => const SizedBox.shrink(),
    );
  }

  Widget _buildRecordingIndicator() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: ThemeConfig.processingColor.withValues(alpha: 0.2),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: 16,
            height: 16,
            child: CircularProgressIndicator(
              strokeWidth: 2,
              color: ThemeConfig.processingColor,
            ),
          ),
          const SizedBox(width: 8),
          const Text(
            'Recording...',
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPlayingIndicator() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: ThemeConfig.primaryColor.withValues(alpha: 0.2),
        borderRadius: BorderRadius.circular(20),
      ),
      child: const Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            Icons.volume_up,
            size: 16,
            color: ThemeConfig.primaryColor,
          ),
          SizedBox(width: 8),
          Text(
            'Playing...',
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
}
