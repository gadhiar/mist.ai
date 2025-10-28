/**
 * WebSocket Message Types
 * These match the backend protocol defined in backend/server.py
 */

// Message types from backend to frontend
export type BackendMessage =
  | ConnectionMessage
  | TranscriptionMessage
  | LLMResponseChunkMessage
  | LLMResponseCompleteMessage
  | AudioChunkMessage
  | AudioCompleteMessage
  | VADStatusMessage
  | ErrorMessage;

// Message types from frontend to backend
export type FrontendMessage = UserAudioMessage;

// Connection established
export interface ConnectionMessage {
  type: "connection";
  data: {
    client_id: string;
    message: string;
  };
}

// User speech transcription
export interface TranscriptionMessage {
  type: "transcription";
  data: {
    text: string;
    timestamp: number;
  };
}

// LLM response streaming chunk
export interface LLMResponseChunkMessage {
  type: "llm_response_chunk";
  data: {
    text: string;
  };
}

// LLM response complete
export interface LLMResponseCompleteMessage {
  type: "llm_response_complete";
  data: {
    full_text: string;
  };
}

// Audio chunk for playback
export interface AudioChunkMessage {
  type: "audio_chunk";
  data: {
    audio: string; // base64 encoded PCM float32 audio
    sample_rate: number;
  };
}

// Audio generation complete
export interface AudioCompleteMessage {
  type: "audio_complete";
  data: Record<string, never>; // Empty object
}

// VAD status updates
export interface VADStatusMessage {
  type: "vad_status";
  data: {
    status: "speech_started" | "speech_ended" | "processing";
    timestamp?: number;
  };
}

// Error message
export interface ErrorMessage {
  type: "error";
  data: {
    message: string;
    details?: string;
  };
}

// User audio data (frontend to backend)
export interface UserAudioMessage {
  type: "audio_data";
  data: {
    audio: string; // base64 encoded audio
    sample_rate: number;
  };
}

// UI State enums
export enum ConversationState {
  IDLE = "idle",
  LISTENING = "listening",
  PROCESSING = "processing",
  SPEAKING = "speaking",
}

export enum AudioPlaybackState {
  IDLE = "idle",
  BUFFERING = "buffering",
  PLAYING = "playing",
  PAUSED = "paused",
}

// Conversation turn for display
export interface ConversationTurn {
  id: string;
  role: "user" | "assistant";
  text: string;
  timestamp: number;
  isComplete: boolean;
}
