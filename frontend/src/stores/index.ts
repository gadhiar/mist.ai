/**
 * Store Exports
 * Central export point for all Redux store-related functionality
 */

// Store and hooks
export { store, useAppDispatch, useAppSelector, wsService } from './store';
export type { RootState, AppDispatch } from './store';

// Conversation slice
export {
  addTurn,
  addUserTranscription,
  startAssistantResponse,
  appendResponseChunk,
  completeAssistantResponse,
  setConversationState,
  clearConversation,
  removeLastTurn,
} from './slices/conversationSlice';
export type { ConversationSliceState } from './slices/conversationSlice';

// Audio slice
export {
  addAudioChunk,
  clearAudioQueue,
  setPlaybackState,
  startBuffering,
  updateBufferProgress,
  finishBuffering,
  play,
  pause,
  stop,
  nextChunk,
  setVolume,
  toggleMute,
  setMute,
  updatePlaybackPosition,
  setAudioDuration,
  audioComplete,
  resetAudio,
} from './slices/audioSlice';
export type { AudioSliceState, AudioChunk } from './slices/audioSlice';

// Connection slice
export {
  connecting,
  connected,
  disconnected,
  reconnecting,
  connectionError,
  setVADStatus,
  updateLatency,
  incrementMessageCount,
  resetConnection,
  ConnectionStatus,
  VADStatus,
} from './slices/connectionSlice';
export type { ConnectionSliceState } from './slices/connectionSlice';

// Middleware
export { connectWebSocket, disconnectWebSocket } from './middleware/websocketMiddleware';
