/**
 * Conversation Slice
 * Manages conversation history, streaming state, and message management
 */

import { createSlice, type PayloadAction } from '@reduxjs/toolkit';
import { type ConversationTurn, ConversationState } from '../../types';

export interface ConversationSliceState {
  // Conversation history
  turns: ConversationTurn[];

  // Current state
  state: ConversationState;

  // Streaming state
  isStreaming: boolean;
  currentStreamingText: string;
  currentStreamingTurnId: string | null;

  // User input tracking
  lastUserInput: string | null;
  lastUserInputTimestamp: number | null;
}

const initialState: ConversationSliceState = {
  turns: [],
  state: ConversationState.IDLE,
  isStreaming: false,
  currentStreamingText: '',
  currentStreamingTurnId: null,
  lastUserInput: null,
  lastUserInputTimestamp: null,
};

const conversationSlice = createSlice({
  name: 'conversation',
  initialState,
  reducers: {
    // Add a new conversation turn
    addTurn: (state, action: PayloadAction<ConversationTurn>) => {
      state.turns.push(action.payload);
    },

    // Add user transcription
    addUserTranscription: (state, action: PayloadAction<{ text: string; timestamp: number }>) => {
      const { text, timestamp } = action.payload;

      // Create new user turn
      const userTurn: ConversationTurn = {
        id: `user-${timestamp}`,
        role: 'user',
        text,
        timestamp,
        isComplete: true,
      };

      state.turns.push(userTurn);
      state.lastUserInput = text;
      state.lastUserInputTimestamp = timestamp;
    },

    // Start streaming assistant response
    startAssistantResponse: (state, action: PayloadAction<{ timestamp: number }>) => {
      const { timestamp } = action.payload;

      // Create new assistant turn (incomplete)
      const assistantTurn: ConversationTurn = {
        id: `assistant-${timestamp}`,
        role: 'assistant',
        text: '',
        timestamp,
        isComplete: false,
      };

      state.turns.push(assistantTurn);
      state.isStreaming = true;
      state.currentStreamingText = '';
      state.currentStreamingTurnId = assistantTurn.id;
    },

    // Append chunk to streaming response
    appendResponseChunk: (state, action: PayloadAction<{ text: string }>) => {
      const { text } = action.payload;

      if (state.currentStreamingTurnId) {
        // Find the turn being streamed
        const turn = state.turns.find(t => t.id === state.currentStreamingTurnId);
        if (turn) {
          turn.text += text;
          state.currentStreamingText = turn.text;
        }
      }
    },

    // Complete streaming response
    completeAssistantResponse: (state, action: PayloadAction<{ fullText: string }>) => {
      const { fullText } = action.payload;

      if (state.currentStreamingTurnId) {
        // Find the turn being streamed
        const turn = state.turns.find(t => t.id === state.currentStreamingTurnId);
        if (turn) {
          turn.text = fullText;
          turn.isComplete = true;
        }
      }

      state.isStreaming = false;
      state.currentStreamingText = '';
      state.currentStreamingTurnId = null;
    },

    // Update conversation state
    setConversationState: (state, action: PayloadAction<ConversationState>) => {
      state.state = action.payload;
    },

    // Clear conversation history
    clearConversation: (state) => {
      state.turns = [];
      state.isStreaming = false;
      state.currentStreamingText = '';
      state.currentStreamingTurnId = null;
      state.state = ConversationState.IDLE;
    },

    // Remove last turn (for interruptions)
    removeLastTurn: (state) => {
      if (state.turns.length > 0) {
        state.turns.pop();
      }
      state.isStreaming = false;
      state.currentStreamingText = '';
      state.currentStreamingTurnId = null;
    },
  },
});

export const {
  addTurn,
  addUserTranscription,
  startAssistantResponse,
  appendResponseChunk,
  completeAssistantResponse,
  setConversationState,
  clearConversation,
  removeLastTurn,
} = conversationSlice.actions;

export default conversationSlice.reducer;
