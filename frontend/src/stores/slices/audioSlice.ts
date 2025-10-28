/**
 * Audio Slice
 * Manages audio playback state, buffering, and audio queue
 */

import { createSlice, type PayloadAction } from '@reduxjs/toolkit';
import { AudioPlaybackState } from '../../types';

export interface AudioChunk {
  id: string;
  audio: string; // base64 encoded
  sampleRate: number;
  timestamp: number;
}

export interface AudioSliceState {
  // Playback state
  playbackState: AudioPlaybackState;

  // Audio queue
  audioQueue: AudioChunk[];
  currentChunkIndex: number;

  // Buffering
  isBuffering: boolean;
  bufferProgress: number; // 0-100

  // Volume
  volume: number; // 0-1
  isMuted: boolean;

  // Playback tracking
  isPlaying: boolean;
  currentAudioDuration: number;
  currentAudioPosition: number;
}

const initialState: AudioSliceState = {
  playbackState: AudioPlaybackState.IDLE,
  audioQueue: [],
  currentChunkIndex: 0,
  isBuffering: false,
  bufferProgress: 0,
  volume: 1.0,
  isMuted: false,
  isPlaying: false,
  currentAudioDuration: 0,
  currentAudioPosition: 0,
};

const audioSlice = createSlice({
  name: 'audio',
  initialState,
  reducers: {
    // Add audio chunk to queue
    addAudioChunk: (state, action: PayloadAction<Omit<AudioChunk, 'id' | 'timestamp'>>) => {
      const chunk: AudioChunk = {
        ...action.payload,
        id: `audio-${Date.now()}-${state.audioQueue.length}`,
        timestamp: Date.now(),
      };
      state.audioQueue.push(chunk);
    },

    // Clear audio queue (for interruptions)
    clearAudioQueue: (state) => {
      state.audioQueue = [];
      state.currentChunkIndex = 0;
      state.playbackState = AudioPlaybackState.IDLE;
      state.isPlaying = false;
      state.isBuffering = false;
      state.bufferProgress = 0;
    },

    // Set playback state
    setPlaybackState: (state, action: PayloadAction<AudioPlaybackState>) => {
      state.playbackState = action.payload;
      state.isPlaying = action.payload === AudioPlaybackState.PLAYING;
    },

    // Start buffering
    startBuffering: (state) => {
      state.isBuffering = true;
      state.playbackState = AudioPlaybackState.BUFFERING;
      state.bufferProgress = 0;
    },

    // Update buffer progress
    updateBufferProgress: (state, action: PayloadAction<number>) => {
      state.bufferProgress = Math.min(100, Math.max(0, action.payload));
    },

    // Finish buffering and start playback
    finishBuffering: (state) => {
      state.isBuffering = false;
      state.bufferProgress = 100;
      state.playbackState = AudioPlaybackState.PLAYING;
      state.isPlaying = true;
    },

    // Play audio
    play: (state) => {
      state.playbackState = AudioPlaybackState.PLAYING;
      state.isPlaying = true;
    },

    // Pause audio
    pause: (state) => {
      state.playbackState = AudioPlaybackState.PAUSED;
      state.isPlaying = false;
    },

    // Stop audio
    stop: (state) => {
      state.playbackState = AudioPlaybackState.IDLE;
      state.isPlaying = false;
      state.currentAudioPosition = 0;
    },

    // Advance to next chunk
    nextChunk: (state) => {
      if (state.currentChunkIndex < state.audioQueue.length - 1) {
        state.currentChunkIndex++;
      }
    },

    // Set volume
    setVolume: (state, action: PayloadAction<number>) => {
      state.volume = Math.min(1, Math.max(0, action.payload));
    },

    // Toggle mute
    toggleMute: (state) => {
      state.isMuted = !state.isMuted;
    },

    // Set mute
    setMute: (state, action: PayloadAction<boolean>) => {
      state.isMuted = action.payload;
    },

    // Update playback position
    updatePlaybackPosition: (state, action: PayloadAction<number>) => {
      state.currentAudioPosition = action.payload;
    },

    // Set audio duration
    setAudioDuration: (state, action: PayloadAction<number>) => {
      state.currentAudioDuration = action.payload;
    },

    // Audio generation complete (all chunks received)
    audioComplete: (state) => {
      // Don't change playback state yet - audio may still be playing
      state.isBuffering = false;
    },

    // Audio playback finished (all audio played)
    audioPlaybackFinished: (state) => {
      state.playbackState = AudioPlaybackState.IDLE;
      state.isPlaying = false;
      state.audioQueue = [];
      state.currentChunkIndex = 0;
      state.currentAudioPosition = 0;
      state.currentAudioDuration = 0;
    },

    // Reset audio state
    resetAudio: (state) => {
      Object.assign(state, initialState);
    },
  },
});

export const {
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
  audioPlaybackFinished,
  resetAudio,
} = audioSlice.actions;

export default audioSlice.reducer;
