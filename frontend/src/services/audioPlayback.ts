/**
 * Audio Playback Service
 * Handles audio playback using Web Audio API
 */

export interface AudioChunkData {
  audio: number[] | string; // Array of floats or base64
  sampleRate: number;
}

export type PlaybackCallback = () => void;

export class AudioPlaybackService {
  private audioContext: AudioContext | null = null;
  private audioQueue: AudioChunkData[] = [];
  private isPlaying = false;
  private currentSource: AudioBufferSourceNode | null = null;
  private volume = 1.0;
  private isMuted = false;
  private gainNode: GainNode | null = null;

  // Callbacks
  private onPlaybackStartCallback: PlaybackCallback | null = null;
  private onPlaybackEndCallback: PlaybackCallback | null = null;
  private onChunkPlayedCallback: PlaybackCallback | null = null;

  constructor() {
    this.initializeAudioContext();
  }

  private initializeAudioContext() {
    try {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      this.gainNode = this.audioContext.createGain();
      this.gainNode.connect(this.audioContext.destination);
      this.gainNode.gain.value = this.volume;
      console.log('[AudioPlayback] Audio context initialized');
    } catch (error) {
      console.error('[AudioPlayback] Failed to initialize audio context:', error);
    }
  }

  /**
   * Add audio chunk to playback queue
   */
  addChunk(chunk: AudioChunkData) {
    this.audioQueue.push(chunk);
    console.log(`[AudioPlayback] Added chunk (queue size: ${this.audioQueue.length})`);

    // Start playback if not already playing
    if (!this.isPlaying && this.audioQueue.length > 0) {
      this.playNext();
    }
  }

  /**
   * Convert audio data to AudioBuffer
   */
  private async convertToAudioBuffer(chunk: AudioChunkData): Promise<AudioBuffer> {
    if (!this.audioContext) {
      throw new Error('Audio context not initialized');
    }

    let audioData: Float32Array;

    // Handle different input formats
    if (typeof chunk.audio === 'string') {
      // Base64 encoded
      const binaryString = atob(chunk.audio);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      // Assuming 32-bit float PCM
      audioData = new Float32Array(bytes.buffer);
    } else {
      // Array of numbers
      audioData = new Float32Array(chunk.audio);
    }

    // Create AudioBuffer
    const audioBuffer = this.audioContext.createBuffer(
      1, // mono
      audioData.length,
      chunk.sampleRate
    );

    // Copy data to buffer
    audioBuffer.getChannelData(0).set(audioData);

    return audioBuffer;
  }

  /**
   * Play next chunk in queue
   */
  private async playNext() {
    if (this.audioQueue.length === 0) {
      console.log('[AudioPlayback] Queue empty, playback finished');
      this.isPlaying = false;
      if (this.onPlaybackEndCallback) {
        this.onPlaybackEndCallback();
      }
      return;
    }

    if (!this.audioContext) {
      console.error('[AudioPlayback] Audio context not available');
      return;
    }

    // Resume audio context if suspended (browser autoplay policy)
    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }

    const chunk = this.audioQueue.shift()!;
    this.isPlaying = true;

    // Call start callback on first chunk
    if (this.onPlaybackStartCallback && this.audioQueue.length === 0) {
      this.onPlaybackStartCallback();
    }

    try {
      // Convert chunk to AudioBuffer
      const audioBuffer = await this.convertToAudioBuffer(chunk);

      // Create source node
      this.currentSource = this.audioContext.createBufferSource();
      this.currentSource.buffer = audioBuffer;

      // Connect to gain node (for volume control)
      if (this.gainNode) {
        this.currentSource.connect(this.gainNode);
      } else {
        this.currentSource.connect(this.audioContext.destination);
      }

      // Set up completion handler
      this.currentSource.onended = () => {
        console.log('[AudioPlayback] Chunk finished playing');
        if (this.onChunkPlayedCallback) {
          this.onChunkPlayedCallback();
        }
        // Play next chunk
        this.playNext();
      };

      // Start playback
      this.currentSource.start(0);
      console.log(`[AudioPlayback] Playing chunk (${audioBuffer.duration.toFixed(2)}s, ${audioBuffer.sampleRate}Hz)`);

    } catch (error) {
      console.error('[AudioPlayback] Error playing chunk:', error);
      // Try next chunk
      this.playNext();
    }
  }

  /**
   * Stop playback and clear queue
   */
  stop() {
    console.log('[AudioPlayback] Stopping playback');

    if (this.currentSource) {
      try {
        this.currentSource.stop();
        this.currentSource.disconnect();
      } catch (error) {
        // Source may already be stopped
      }
      this.currentSource = null;
    }

    this.audioQueue = [];
    this.isPlaying = false;

    if (this.onPlaybackEndCallback) {
      this.onPlaybackEndCallback();
    }
  }

  /**
   * Set volume (0.0 to 1.0)
   */
  setVolume(volume: number) {
    this.volume = Math.min(1.0, Math.max(0.0, volume));
    if (this.gainNode) {
      this.gainNode.gain.value = this.isMuted ? 0 : this.volume;
    }
  }

  /**
   * Set mute state
   */
  setMute(muted: boolean) {
    this.isMuted = muted;
    if (this.gainNode) {
      this.gainNode.gain.value = muted ? 0 : this.volume;
    }
  }

  /**
   * Get current playing state
   */
  getIsPlaying(): boolean {
    return this.isPlaying;
  }

  /**
   * Get queue size
   */
  getQueueSize(): number {
    return this.audioQueue.length;
  }

  /**
   * Register callback for playback start
   */
  onPlaybackStart(callback: PlaybackCallback) {
    this.onPlaybackStartCallback = callback;
  }

  /**
   * Register callback for playback end (all chunks played)
   */
  onPlaybackEnd(callback: PlaybackCallback) {
    this.onPlaybackEndCallback = callback;
  }

  /**
   * Register callback for each chunk played
   */
  onChunkPlayed(callback: PlaybackCallback) {
    this.onChunkPlayedCallback = callback;
  }

  /**
   * Clean up resources
   */
  destroy() {
    this.stop();
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }
}
