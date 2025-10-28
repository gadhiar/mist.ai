/**
 * WebSocket Middleware
 * Syncs WebSocket messages with Redux store
 * Handles incoming messages and dispatches appropriate actions
 */
import type { Middleware } from '@reduxjs/toolkit';
import { WebSocketService } from '../../services/websocket';
import { AudioPlaybackService } from '../../services/audioPlayback';

// Import actions from slices
import {
  addUserTranscription,
  startAssistantResponse,
  appendResponseChunk,
  completeAssistantResponse,
  setConversationState,
} from '../slices/conversationSlice';

import {
  addAudioChunk,
  clearAudioQueue,
  audioComplete,
  audioPlaybackFinished,
  startBuffering,
  play,
} from '../slices/audioSlice';

import {
  connecting,
  connected,
  disconnected,
  setVADStatus,
  incrementMessageCount,
  connectionError,
} from '../slices/connectionSlice';

import { ConversationState } from '../../types';
import { VADStatus } from '../slices/connectionSlice';

/**
 * Creates WebSocket middleware
 * @param wsService - WebSocket service instance
 */
export const createWebSocketMiddleware = (wsService: WebSocketService): Middleware => {
  // Create audio playback service instance
  const audioService = new AudioPlaybackService();

  return (store) => {
    // Set up audio playback callbacks
    audioService.onPlaybackStart(() => {
      console.log('[AudioPlayback] Playback started');
      store.dispatch(play());
    });

    audioService.onPlaybackEnd(() => {
      console.log('[AudioPlayback] All audio finished playing');
      store.dispatch(audioPlaybackFinished());
      store.dispatch(setConversationState(ConversationState.IDLE));
    });
    // Set up WebSocket message handlers
    // Backend sends: {type: "status", message: "Connected to Mist.AI Voice Server"}
    wsService.on('status', (msg: any) => {
      // Generate a temporary client ID since backend doesn't provide one
      const clientId = `client-${Date.now()}`;
      store.dispatch(connected({ clientId }));
      console.log('[WebSocket] Connected:', msg.message);
    });

    wsService.on('transcription', (msg: any) => {
      // Backend sends: {type: "transcription", text: "..."}
      const state = store.getState();

      // Check if this transcription was already added (prevent duplicates)
      const lastTurn = state.conversation.turns[state.conversation.turns.length - 1];
      const isDuplicate = lastTurn &&
                         lastTurn.role === 'user' &&
                         lastTurn.text === msg.text;

      if (!isDuplicate) {
        store.dispatch(addUserTranscription({
          text: msg.text || '',
          timestamp: msg.timestamp || Date.now(),
        }));
        store.dispatch(setConversationState(ConversationState.PROCESSING));
      }
      store.dispatch(incrementMessageCount());
    });

    // Backend sends "llm_token" for streaming chunks
    wsService.on('llm_token', (msg: any) => {
      // Backend sends: {type: "llm_token", token: "..."}
      const state = store.getState();

      // Check if we need to start a new assistant response
      // Look for an existing incomplete assistant turn
      const lastTurn = state.conversation.turns[state.conversation.turns.length - 1];
      const hasIncompleteAssistantTurn = lastTurn &&
                                         lastTurn.role === 'assistant' &&
                                         !lastTurn.isComplete;

      // Only start new response if there's no incomplete assistant turn
      if (!hasIncompleteAssistantTurn && !state.conversation.isStreaming) {
        store.dispatch(startAssistantResponse({ timestamp: Date.now() }));
        store.dispatch(setConversationState(ConversationState.SPEAKING));
      }

      store.dispatch(appendResponseChunk({ text: msg.token || '' }));
      store.dispatch(incrementMessageCount());
    });

    // Backend sends "llm_response" with full text when complete
    wsService.on('llm_response', (msg: any) => {
      // Backend sends: {type: "llm_response", text: "..."}
      store.dispatch(completeAssistantResponse({ fullText: msg.text || '' }));
      store.dispatch(incrementMessageCount());
    });

    wsService.on('audio_chunk', (msg: any) => {
      // Backend sends: {type: "audio_chunk", audio: [...], sample_rate: 24000}
      const state = store.getState();

      // Start buffering if this is the first chunk
      if (state.audio.audioQueue.length === 0 && !state.audio.isBuffering) {
        store.dispatch(startBuffering());
      }

      // Add to Redux store for tracking
      store.dispatch(addAudioChunk({
        audio: msg.audio || '',
        sampleRate: msg.sample_rate || 24000,
      }));

      // Add to audio playback service for actual playback
      audioService.addChunk({
        audio: msg.audio || [],
        sampleRate: msg.sample_rate || 24000,
      });

      store.dispatch(incrementMessageCount());
    });

    wsService.on('audio_complete', () => {
      store.dispatch(audioComplete());
      // Note: Don't set IDLE immediately - audio needs to finish playing
      // The audio playback system should set IDLE when playback actually completes
      // For now, we keep it in SPEAKING state until audio finishes
      store.dispatch(incrementMessageCount());
    });

    wsService.on('vad_status', (msg: any) => {
      // Backend sends: {type: "vad_status", status: "speech_started"}
      // Map backend VAD status to frontend enum
      let vadStatus: VADStatus;
      switch (msg.status) {
        case 'speech_started':
          vadStatus = VADStatus.SPEECH_STARTED;
          store.dispatch(setConversationState(ConversationState.LISTENING));
          // Interrupt audio playback (both Redux state and actual playback)
          store.dispatch(clearAudioQueue());
          audioService.stop();
          break;
        case 'speech_ended':
          vadStatus = VADStatus.SPEECH_ENDED;
          break;
        case 'processing':
          vadStatus = VADStatus.PROCESSING;
          store.dispatch(setConversationState(ConversationState.PROCESSING));
          break;
        default:
          vadStatus = VADStatus.IDLE;
      }

      store.dispatch(setVADStatus({
        status: vadStatus,
        timestamp: msg.timestamp || Date.now(),
      }));
      store.dispatch(incrementMessageCount());
    });

    wsService.on('error', (msg: any) => {
      // Backend sends: {type: "error", message: "..."}
      console.error('[WebSocket] Error:', msg.message);
      store.dispatch(connectionError({ error: msg.message || 'Unknown error' }));
      store.dispatch(incrementMessageCount());
    });

    // Handle all messages (for logging/debugging)
    wsService.on('*', (message: any) => {
      console.log('[WebSocket] Message received:', message.type, message);
    });

    return (next) => (action: any) => {
      // Handle connection actions
      if (action.type === 'websocket/connect') {
        store.dispatch(connecting());
        wsService.connect().catch((error) => {
          console.error('[WebSocket] Connection failed:', error);
          store.dispatch(connectionError({ error: error.message }));
        });
      }

      if (action.type === 'websocket/disconnect') {
        wsService.disconnect();
        audioService.stop();
        store.dispatch(disconnected());
      }

      // Pass action to next middleware
      return next(action);
    };
  };
};

// Action creators for WebSocket control
export const connectWebSocket = () => ({ type: 'websocket/connect' as const });
export const disconnectWebSocket = () => ({ type: 'websocket/disconnect' as const });
