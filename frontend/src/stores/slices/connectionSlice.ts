/**
 * Connection Slice
 * Manages WebSocket connection state and VAD status
 */

import { createSlice, type PayloadAction } from '@reduxjs/toolkit';

export enum ConnectionStatus {
  DISCONNECTED = 'disconnected',
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  RECONNECTING = 'reconnecting',
  ERROR = 'error',
}

export enum VADStatus {
  IDLE = 'idle',
  SPEECH_STARTED = 'speech_started',
  SPEECH_ENDED = 'speech_ended',
  PROCESSING = 'processing',
}

export interface ConnectionSliceState {
  // Connection status
  status: ConnectionStatus;
  clientId: string | null;

  // Connection metadata
  connectedAt: number | null;
  disconnectedAt: number | null;
  reconnectAttempts: number;
  lastError: string | null;

  // VAD status
  vadStatus: VADStatus;
  vadTimestamp: number | null;

  // Network quality (future)
  latency: number | null;
  messageCount: number;
}

const initialState: ConnectionSliceState = {
  status: ConnectionStatus.DISCONNECTED,
  clientId: null,
  connectedAt: null,
  disconnectedAt: null,
  reconnectAttempts: 0,
  lastError: null,
  vadStatus: VADStatus.IDLE,
  vadTimestamp: null,
  latency: null,
  messageCount: 0,
};

const connectionSlice = createSlice({
  name: 'connection',
  initialState,
  reducers: {
    // Connection lifecycle
    connecting: (state) => {
      state.status = ConnectionStatus.CONNECTING;
      state.lastError = null;
    },

    connected: (state, action: PayloadAction<{ clientId: string }>) => {
      state.status = ConnectionStatus.CONNECTED;
      state.clientId = action.payload.clientId;
      state.connectedAt = Date.now();
      state.disconnectedAt = null;
      state.reconnectAttempts = 0;
      state.lastError = null;
    },

    disconnected: (state) => {
      state.status = ConnectionStatus.DISCONNECTED;
      state.disconnectedAt = Date.now();
      state.clientId = null;
    },

    reconnecting: (state) => {
      state.status = ConnectionStatus.RECONNECTING;
      state.reconnectAttempts++;
    },

    connectionError: (state, action: PayloadAction<{ error: string }>) => {
      state.status = ConnectionStatus.ERROR;
      state.lastError = action.payload.error;
    },

    // VAD status updates
    setVADStatus: (state, action: PayloadAction<{ status: VADStatus; timestamp?: number }>) => {
      state.vadStatus = action.payload.status;
      state.vadTimestamp = action.payload.timestamp || Date.now();
    },

    // Network metrics
    updateLatency: (state, action: PayloadAction<number>) => {
      state.latency = action.payload;
    },

    incrementMessageCount: (state) => {
      state.messageCount++;
    },

    // Reset connection state
    resetConnection: (state) => {
      Object.assign(state, initialState);
    },
  },
});

export const {
  connecting,
  connected,
  disconnected,
  reconnecting,
  connectionError,
  setVADStatus,
  updateLatency,
  incrementMessageCount,
  resetConnection,
} = connectionSlice.actions;

export default connectionSlice.reducer;
