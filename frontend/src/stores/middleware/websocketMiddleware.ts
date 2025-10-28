/**
 * WebSocket Middleware
 * Syncs WebSocket messages with Redux store
 * Handles incoming messages and dispatches appropriate actions
 */

/**
 * WebSocket Middleware
 * Syncs WebSocket messages with Redux store
 * Handles incoming messages and dispatches appropriate actions
 */
import type { Middleware } from '@reduxjs/toolkit';
import { WebSocketService } from '../../services/websocket';
import type { BackendMessage } from '../../types';

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
  startBuffering,
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
  return (store) => {
    // Set up WebSocket message handlers
    wsService.on('connection', (data: any) => {
      store.dispatch(connected({ clientId: data.client_id }));
      console.log('[WebSocket] Connected:', data.message);
    });

    wsService.on('transcription', (data: any) => {
      store.dispatch(addUserTranscription({
        text: data.text,
        timestamp: data.timestamp,
      }));
      store.dispatch(setConversationState(ConversationState.PROCESSING));
      store.dispatch(incrementMessageCount());
    });

    wsService.on('llm_response_chunk', (data: any) => {
      const state = store.getState();

      // If not streaming yet, start a new response
      if (!state.conversation.isStreaming) {
        store.dispatch(startAssistantResponse({ timestamp: Date.now() }));
        store.dispatch(setConversationState(ConversationState.SPEAKING));
      }

      store.dispatch(appendResponseChunk({ text: data.text }));
      store.dispatch(incrementMessageCount());
    });

    wsService.on('llm_response_complete', (data: any) => {
      store.dispatch(completeAssistantResponse({ fullText: data.full_text }));
      store.dispatch(incrementMessageCount());
    });

    wsService.on('audio_chunk', (data: any) => {
      const state = store.getState();

      // Start buffering if this is the first chunk
      if (state.audio.audioQueue.length === 0 && !state.audio.isBuffering) {
        store.dispatch(startBuffering());
      }

      store.dispatch(addAudioChunk({
        audio: data.audio,
        sampleRate: data.sample_rate,
      }));
      store.dispatch(incrementMessageCount());
    });

    wsService.on('audio_complete', () => {
      store.dispatch(audioComplete());
      store.dispatch(setConversationState(ConversationState.IDLE));
      store.dispatch(incrementMessageCount());
    });

    wsService.on('vad_status', (data: any) => {
      // Map backend VAD status to frontend enum
      let vadStatus: VADStatus;
      switch (data.status) {
        case 'speech_started':
          vadStatus = VADStatus.SPEECH_STARTED;
          store.dispatch(setConversationState(ConversationState.LISTENING));
          // Interrupt audio playback
          store.dispatch(clearAudioQueue());
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
        timestamp: data.timestamp,
      }));
      store.dispatch(incrementMessageCount());
    });

    wsService.on('error', (data: any) => {
      console.error('[WebSocket] Error:', data.message);
      store.dispatch(connectionError({ error: data.message }));
      store.dispatch(incrementMessageCount());
    });

    // Handle all messages (for logging/debugging)
    wsService.on('*', (message: BackendMessage) => {
      console.log('[WebSocket] Message received:', message.type, message.data);
    });

    return (next) => (action) => {
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
