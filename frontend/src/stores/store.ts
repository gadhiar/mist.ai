/**
 * Redux Store Configuration
 * Configures the Redux store with slices and middleware
 */

import { configureStore } from '@reduxjs/toolkit';
import { type TypedUseSelectorHook, useDispatch, useSelector } from 'react-redux';

// Reducers
import conversationReducer from './slices/conversationSlice';
import audioReducer from './slices/audioSlice';
import connectionReducer from './slices/connectionSlice';

// Middleware
import { createWebSocketMiddleware } from './middleware/websocketMiddleware';

// WebSocket service
import { WebSocketService } from '../services/websocket';

// Create WebSocket service instance
// TODO: Move WebSocket URL to environment variable
const WS_URL = 'ws://localhost:8001/ws';
export const wsService = new WebSocketService(WS_URL);

// Create store
export const store = configureStore({
  reducer: {
    conversation: conversationReducer,
    audio: audioReducer,
    connection: connectionReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        // Ignore these action types for serialization checks
        // (WebSocket service is not serializable)
        ignoredActions: ['websocket/connect', 'websocket/disconnect'],
      },
    }).concat(createWebSocketMiddleware(wsService)),
  devTools: import.meta.env.MODE !== 'production',
});

// Infer types from the store
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Typed hooks for use throughout the app
export const useAppDispatch: () => AppDispatch = useDispatch;
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;
