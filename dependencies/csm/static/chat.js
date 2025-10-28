let ws;
let sessionStartTime = null;
let messageCount = 0;
let audioLevelsChart = null;
let isRecording = false;
let isAudioCurrentlyPlaying = false;
let configSaved = false;
let currentAudioSource = null; 
let interruptRequested = false; 
let interruptInProgress = false;
let audioContext = null;
let lastSeenGenId = 0;
let reconnecting = false;
let reconnectAttempts = 0;
let maxReconnectAttempts = 10;

const SESSION_ID = "default";
console.log("chat.js loaded");

let micStream;
let selectedMicId = null;
let selectedOutputId = null;

let audioPlaybackQueue = [];
let audioDataHistory = [];
let micAnalyser, micContext;
let activeGenId = 0;

function createPermanentVoiceCircle() {
  if (document.getElementById('voice-circle')) return;
  const style = document.createElement('style');
  style.textContent = `
    #voice-circle{
      position:fixed;top:50%;left:50%;
      width:180px;height:180px;border-radius:50%;
      background:rgba(99,102,241,.20);
      transform:translate(-50%,-50%) scale(var(--dynamic-scale,1));
      pointer-events:none;z-index:50;
      transition:background-color .35s ease;
    }
    #voice-circle.active{
      animation:pulse-circle 2s infinite alternate ease-in-out;
    }
    @keyframes pulse-circle{
      0%{background:rgba(99,102,241,.55)}
      100%{background:rgba(99,102,241,.20)}
    }`;
  document.head.appendChild(style);

  const c = document.createElement('div');
  c.id='voice-circle';
  document.body.appendChild(c);
  console.log("Created permanent voice circle");
}

function showVoiceCircle() {
  const c=document.getElementById('voice-circle')||createPermanentVoiceCircle();
  c.classList.add('active');
}

function hideVoiceCircle() {
  const c=document.getElementById('voice-circle');
  if (c){ c.classList.remove('active'); c.style.setProperty('--dynamic-scale',1); }
}

function showNotification(msg, type='info'){
  const n=document.createElement('div');
  n.className=`fixed bottom-4 right-4 px-4 py-3 rounded-lg shadow-lg z-50
               ${type==='success'?'bg-green-600':
                 type==='error'  ?'bg-red-600':'bg-indigo-600'}`;
  n.textContent=msg;
  document.body.appendChild(n);
  setTimeout(()=>{n.classList.add('opacity-0');
                  setTimeout(()=>n.remove(),500)},3000);
}

function addMessageToConversation(sender,text){
  const pane=document.getElementById('conversationHistory');
  if(!pane) return;
  const box=document.createElement('div');
  box.className=`p-3 mb-3 rounded-lg text-sm ${
            sender==='user'?'bg-gray-800 ml-2':'bg-indigo-900 mr-2'}`;
  box.innerHTML=`
      <div class="flex items-start mb-2">
        <div class="w-6 h-6 rounded-full flex items-center justify-center
             ${sender==='user'?'bg-gray-300 text-gray-800':'bg-indigo-500 text-white'}">
             ${sender==='user'?'U':'AI'}
        </div>
        <span class="text-xs text-gray-400 ml-2">${new Date().toLocaleTimeString()}</span>
      </div>
      <div class="text-white mt-1 text-sm">${text
            .replace(/&/g,'&amp;').replace(/</g,'&lt;')
            .replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>')
            .replace(/\*(.*?)\*/g,'<em>$1</em>')
            .replace(/```([^`]+)```/g,'<pre><code>$1</code></pre>')
            .replace(/`([^`]+)`/g,'<code>$1</code>')
            .replace(/\n/g,'<br>')}</div>`;
  pane.appendChild(box);
  pane.scrollTop=pane.scrollHeight;
}

function connectWebSocket() {
  if (reconnecting && reconnectAttempts >= maxReconnectAttempts) {
    console.error("Maximum reconnect attempts reached. Please refresh the page.");
    showNotification("Connection lost. Please refresh the page.", "error");
    return;
  }

  if (ws && ws.readyState !== WebSocket.CLOSED && ws.readyState !== WebSocket.CLOSING) {
    try {
      ws.close();
    } catch (e) {
      console.warn("Error closing existing WebSocket:", e);
    }
  }

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  window.ws = ws;

  const connLbl = document.getElementById('connectionStatus');
  if (connLbl) {
    connLbl.textContent = reconnecting ? 'Reconnecting…' : 'Connecting…';
    connLbl.className = 'text-yellow-500';
  }

  ws.onopen = () => {
    if (connLbl) {
      connLbl.textContent = 'Connected';
      connLbl.className = 'text-green-500';
    }
    
    reconnecting = false;
    reconnectAttempts = 0;
    
    ws.send(JSON.stringify({type: 'request_saved_config'}));
    
    if (!reconnecting) {
      addMessageToConversation('ai', 'WebSocket connected. Ready for voice or text.');
    } else {
      showNotification("Reconnected successfully", "success");
    }
  };

  ws.onclose = (event) => {
    console.log("WebSocket closed with code:", event.code);
    if (connLbl) {
      connLbl.textContent = 'Disconnected';
      connLbl.className = 'text-red-500';
    }

    // Clear audio state on disconnection
    clearAudioPlayback();
    
    // Don't auto-reconnect if this was a normal closure
    if (event.code !== 1000 && event.code !== 1001) {
      reconnecting = true;
      reconnectAttempts++;
      
      const delay = Math.min(1000 * Math.pow(1.5, reconnectAttempts), 1000);
      console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts})`);
      
      setTimeout(connectWebSocket, delay);
    }
  };

  ws.onerror = (error) => {
    console.error("WebSocket error:", error);
    if (connLbl) {
      connLbl.textContent = 'Error';
      connLbl.className = 'text-red-500';
    }
  };

  ws.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      handleWebSocketMessage(data);
    } catch (err) {
      console.error("Error handling WebSocket message:", err);
    }
  };
}

function sendTextMessage(txt) {
  if (!txt.trim()) return;
  
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    showNotification("Not connected", "error");
    return;
  }
  
  console.log("Force clearing all audio state before sending text message");
  
  // Stop any playing audio
  if (isAudioCurrentlyPlaying) {
    if (currentAudioSource) {
      try {
        if (currentAudioSource.disconnect) currentAudioSource.disconnect();
        if (currentAudioSource.stop) currentAudioSource.stop(0);
      } catch (e) {
        console.warn("Error stopping audio:", e);
      }
      currentAudioSource = null;
    }
    isAudioCurrentlyPlaying = false;
  }
  
  // Clear all flags and queues
  interruptRequested = false;
  interruptInProgress = false;
  activeGenId = 0;
  audioPlaybackQueue = [];
  
  // Always force interruption to be absolutely sure
  if (ws && ws.readyState === WebSocket.OPEN) {
    try {
      ws.send(JSON.stringify({type: 'interrupt', immediate: true}));
    } catch (e) {
      console.warn("Error sending interrupt:", e);
    }
  }
  
  // Wait a bit before sending the actual message
  setTimeout(() => {
    try {
      // Show visual feedback
      showVoiceCircle();
      
      // Send the message
      ws.send(JSON.stringify({
        type: 'text_message',
        text: txt,
        session_id: SESSION_ID
      }));
      
      const cnt = document.getElementById('messageCount');
      if (cnt) cnt.textContent = ++messageCount;
      
      document.getElementById('textInput').value = '';
      
      console.log("Text message sent successfully");
    } catch (error) {
      console.error("Error sending message:", error);
      showNotification("Error sending message", "error");
    }
  }, 300);
}

// Reset all audio state to ensure clean state for new interactions
function resetAudioState() {
  console.log("Resetting audio state");
  
  // Clear any stale generation information
  activeGenId = 0;
  lastSeenGenId = 0;
  
  // Clear any remaining flags
  interruptRequested = false;
  interruptInProgress = false;
  
  // Make sure we don't have any playing audio
  if (isAudioCurrentlyPlaying) {
    clearAudioPlayback();
  }
  
  // Clear any queued audio
  audioPlaybackQueue = [];
}

function clearAudioPlayback() {
  console.log("FORCEFULLY CLEARING AUDIO PLAYBACK");
  
  interruptRequested = true;
  interruptInProgress = true;
  
  try {
    // Empty the queue first - do this before stopping current source
    console.log(`Clearing queue with ${audioPlaybackQueue.length} items`);
    audioPlaybackQueue = [];
    
    activeGenId = 0;
    
    // Stop any currently playing audio
    if (currentAudioSource) {
      console.log("Stopping active audio source");
      
      try {
        if (currentAudioSource.disconnect) {
          currentAudioSource.disconnect();
        }
      } catch (e) {
        console.warn("Error disconnecting audio source:", e);
      }
      
      try {
        if (currentAudioSource.stop) {
          currentAudioSource.stop(0);
        }
      } catch (e) {
        console.warn("Error stopping audio source:", e);
      }
      
      currentAudioSource = null;
    }
    
    try {
      if (audioContext) {
        const oldContext = audioContext;
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        window.audioContext = audioContext;
        
        try {
          oldContext.close();
        } catch (closeError) {
          console.warn("Error closing old audio context:", closeError);
        }
      } else {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        window.audioContext = audioContext;
      }
    } catch (contextError) {
      console.error("Error recreating audio context:", contextError);
    }
  } catch (err) {
    console.error("Error clearing audio:", err);
  }
  
  // Reset state
  isAudioCurrentlyPlaying = false;
  hideVoiceCircle();
  
  console.log("Audio playback cleared successfully");
  
  // After a short delay, reset the interrupt flags to accept new audio
  setTimeout(() => {
    interruptInProgress = false;
    // Keep interruptRequested true until we get a new generation
  }, 300);
}


// Handle interruption request from user
function requestInterrupt() {
  console.log("User requested interruption");
  
  if (interruptInProgress) {
    console.log("Interrupt already in progress - force clearing again");
    clearAudioPlayback();
    return false;
  }
  
  // Set the flags immediately
  interruptRequested = true;
  interruptInProgress = true;
  
  // Show visual feedback
  showNotification("Interrupting...", "info");
  
  // Force clear all audio immediately on client side
  clearAudioPlayback();
  
  // Show visual feedback for the button
  const interruptBtn = document.getElementById('interruptBtn');
  if (interruptBtn) {
    interruptBtn.classList.add('bg-red-800');
    setTimeout(() => {
      interruptBtn.classList.remove('bg-red-800');
    }, 300);
  }
  
  // Then notify the server
  if (ws && ws.readyState === WebSocket.OPEN) {
    console.log("Sending interrupt request to server");
    try {
      ws.send(JSON.stringify({
        type: 'interrupt',
        immediate: true
      }));
    } catch (error) {
      console.error("Error sending interrupt request:", error);
    }
    
    // Set a timeout to reset interrupt flags if we don't get server confirmation
    setTimeout(() => {
      if (interruptInProgress) {
        console.log("No interrupt confirmation received from server, resetting state");
        interruptInProgress = false;
      }
    }, 2000);
    
    return true;
  } else {
    console.warn("WebSocket not available for interrupt request");
    // Reset flag after brief delay if we couldn't send to server
    setTimeout(() => {
      interruptInProgress = false;
    }, 500);
    return false;
  }
}

function handleWebSocketMessage(d) {
  console.log("Received message:", d.type, d);
  
  switch(d.type) {
    case 'transcription':
      addMessageToConversation('user', d.text);
      showVoiceCircle();
      break;
      
    case 'response':
      addMessageToConversation('ai', d.text);
      showVoiceCircle();
      
      console.log("NEW RESPONSE RECEIVED - FORCE RESETTING ALL AUDIO STATE");
      
      if (isAudioCurrentlyPlaying) {
        if (currentAudioSource) {
          try {
            if (currentAudioSource.disconnect) currentAudioSource.disconnect();
            if (currentAudioSource.stop) currentAudioSource.stop(0);
          } catch (e) {
            console.warn("Error stopping current audio:", e);
          }
          currentAudioSource = null;
        }
        isAudioCurrentlyPlaying = false;
      }
      
      interruptRequested = false;
      interruptInProgress = false;
      
      activeGenId = 0;
      
      audioPlaybackQueue = [];
      
      try {
        if (audioContext) {
          if (audioContext.state === 'suspended') {
            audioContext.resume().catch(e => console.warn("Error resuming audio context:", e));
          }
        } else {
          audioContext = new (window.AudioContext || window.webkitAudioContext)();
          window.audioContext = audioContext;
        }
      } catch (e) {
        console.warn("Error with audio context:", e);
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        window.audioContext = audioContext;
      }
      
      console.log("Audio state fully reset and ready for new audio");
      break;
      
    case 'audio_chunk':
      console.log("Audio chunk received, flags:", 
                 "interruptRequested:", interruptRequested, 
                 "interruptInProgress:", interruptInProgress,
                 "genId:", d.gen_id,
                 "activeGenId:", activeGenId);
      
      if (!isAudioCurrentlyPlaying && activeGenId === 0) {
        console.log("FIRST AUDIO CHUNK - FORCING FLAGS RESET");
        interruptRequested = false;
        interruptInProgress = false;
      }
      
      // Don't queue new audio if an interrupt was requested
      if (interruptRequested || interruptInProgress) {
        console.log("Interrupt active - ignoring new audio chunk");
        return;
      }
      
      // Set active generation ID on first chunk
      if (activeGenId === 0) {
        activeGenId = d.gen_id || 1;
        console.log("!!! Setting activeGenId to:", activeGenId);
      }
      
      // Only accept chunks that match our active generation
      if ((d.gen_id === activeGenId) || (activeGenId === 0)) {
        queueAudioForPlayback(d.audio, d.sample_rate, d.gen_id || 0);
        showVoiceCircle();
      } else {
        console.log(`Ignored stale chunk - current gen: ${activeGenId}, received: ${d.gen_id}`);
      }
      break;
      
    case 'audio_status':
      console.log("Audio status update:", d.status);
      
      if (d.status === 'generating') {
        console.log("GOT GENERATING STATUS - IMMEDIATELY CLEARING ALL INTERRUPT FLAGS");
        interruptRequested = false;
        interruptInProgress = false;
        
        // Capture the generation ID for new generations
        if (d.gen_id) {
          console.log(`New generation starting with ID: ${d.gen_id}`);
          activeGenId = d.gen_id;
        }
        
        showVoiceCircle();
      } 
      else if (d.status === 'complete') {
        console.log("Audio generation complete");
        if (!d.gen_id || d.gen_id === activeGenId) {
          activeGenId = 0; // Reset for next generation
        }
        if (!isAudioCurrentlyPlaying) {
          hideVoiceCircle();
        }
      } 
      else if (d.status === 'interrupted' || d.status === 'interrupt_acknowledged') {
        console.log("Server confirmed interrupt - clearing audio");
        clearAudioPlayback();
        
        setTimeout(() => {
          console.log("Resetting interrupt flags after server confirmation");
          interruptRequested = false;
          interruptInProgress = false;
        }, 300);
      }
      break;
      
    case 'status':
      if (d.message === 'Thinking...') {
        showVoiceCircle();
        
        interruptRequested = false;
        interruptInProgress = false;
        activeGenId = 0;
      }
      break;
      
    case 'error':
      showNotification(d.message, 'error');
      hideVoiceCircle();
      break;
      
    case 'vad_status':
      if (d.status === 'speech_started') {
        console.log(`[VAD] speech_started | should_interrupt=${d.should_interrupt}`);

        if (d.should_interrupt && isAudioCurrentlyPlaying) {
          console.log('[VAD] confirmed – sending interrupt');
          requestInterrupt();
        } else {
          console.log('[VAD] ignored (echo / early AI audio)');
        }
      }
      break;
  }
}

function queueAudioForPlayback(arr, sr, genId = 0) {
  if (activeGenId !== 0 && genId !== activeGenId) {
    console.log(`Stale chunk ignored (genId mismatch): ${genId} vs ${activeGenId}`);
    return;
  }
  
  // Don't queue if interrupting
  if (interruptRequested || interruptInProgress) {
    console.log("Interrupt active - skipping audio chunk");
    return;
  }
  
  console.log("Queueing audio chunk for playback");
  audioPlaybackQueue.push({arr, sr, genId});
  
  if (!isAudioCurrentlyPlaying) {
    console.log("▶Starting audio playback");
    processAudioPlaybackQueue();
  }
}

function queueAudioForPlayback(arr, sr, genId = 0) {
  // Extra logging for the first audio chunk
  if (!isAudioCurrentlyPlaying) {
    console.log("Queueing first audio chunk", 
               "interruptRequested:", interruptRequested, 
               "interruptInProgress:", interruptInProgress);
  }
  
  if (!isAudioCurrentlyPlaying && audioPlaybackQueue.length === 0) {
    console.log("First audio chunk - forcing clear of interrupt flags");
    interruptRequested = false;
    interruptInProgress = false;
  }
  
  // Don't queue audio from a different generation than our active one
  if (activeGenId !== 0 && genId !== activeGenId) {
    console.log(`Stale chunk ignored (genId mismatch): ${genId} vs ${activeGenId}`);
    return;
  }
  
  // Don't queue if interrupting - BUT CHECK AGAIN THAT FLAGS ARE VALID
  if (interruptRequested || interruptInProgress) {
    console.log("Interrupt active - skipping audio chunk");
    return;
  }
  
  console.log("Queueing audio chunk for playback");
  audioPlaybackQueue.push({arr, sr, genId});
  
  if (!isAudioCurrentlyPlaying) {
    console.log("STARTING AUDIO PLAYBACK - FIRST CHUNK");
    processAudioPlaybackQueue();
  }
}


// Modified to ensure first audio actually plays
function processAudioPlaybackQueue() {
  if (!isAudioCurrentlyPlaying && audioPlaybackQueue.length > 0) {
    console.log("Starting first audio chunk - force clearing interrupt flags");
    interruptRequested = false;
    interruptInProgress = false;
  }
  
  // Double-check interrupt flags AFTER clearling them
  if (interruptRequested || interruptInProgress) {
    console.log("Interrupt active - not processing audio queue");
    isAudioCurrentlyPlaying = false;
    hideVoiceCircle();
    return;
  }
  
  // Check if queue is empty
  if (!audioPlaybackQueue.length) {
    console.log("📭 Audio queue empty, stopping playback");
    isAudioCurrentlyPlaying = false;
    hideVoiceCircle();
    currentAudioSource = null;
    return;
  }
  
  // Enable the interrupt button when audio is playing
  const interruptBtn = document.getElementById('interruptBtn');
  if (interruptBtn) {
    interruptBtn.disabled = false;
    interruptBtn.classList.remove('opacity-50');
  }
  
  console.log("Processing next audio chunk");
  isAudioCurrentlyPlaying = true;
  
  // Get the genId from the chunk
  const {arr, sr, genId} = audioPlaybackQueue.shift();
  
  // Skip if it's a stale chunk
  if (activeGenId !== 0 && genId !== activeGenId) {
    console.log(`Skipping stale chunk playback (gen ${genId} vs active ${activeGenId})`);
    processAudioPlaybackQueue(); // Continue with next chunk
    return;
  }
  
  playAudioChunk(arr, sr)
    .then(() => {
      // Check interrupt status again after playback
      if (!interruptRequested && !interruptInProgress) {
        processAudioPlaybackQueue();
      } else {
        console.log("interrupt active - stopping queue processing");
        isAudioCurrentlyPlaying = false;
        hideVoiceCircle();
      }
    })
    .catch(err => {
      console.error("Error in audio playback:", err);
      isAudioCurrentlyPlaying = false;
      hideVoiceCircle();
      
      // Try to continue with next chunk despite errors
      setTimeout(() => {
        if (audioPlaybackQueue.length > 0 && !interruptRequested) {
          processAudioPlaybackQueue();
        }
      }, 200);
    });
}

async function playAudioChunk(audioArr, sampleRate) {
  // Skip playback if interrupt was requested
  if (interruptRequested || interruptInProgress) {
    console.log("Interrupt active - not playing audio chunk");
    return Promise.resolve();
  }
  
  try {
    // Ensure we have a valid audio context
    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      window.audioContext = audioContext;
    }
    
    // Make sure context is resumed
    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }
    
    const buf = audioContext.createBuffer(1, audioArr.length, sampleRate);
    buf.copyToChannel(new Float32Array(audioArr), 0);
    
    const src = audioContext.createBufferSource();
    src.buffer = buf;
    
    // Store reference to current source for potential interruption
    currentAudioSource = src;
    
    const an = audioContext.createAnalyser(); 
    an.fftSize = 256;
    src.connect(an); 
    an.connect(audioContext.destination); 
    src.start();
    
    console.log("🎵 Started playing audio chunk");

    const arr = new Uint8Array(an.frequencyBinCount);
    const circle = document.getElementById('voice-circle');
    
    // Animation function that respects interruption
    function pump() {
      // Stop animation if source is no longer current or interrupt requested
      if (src !== currentAudioSource || interruptRequested || interruptInProgress) {
        return;
      }
      
      try {
        an.getByteFrequencyData(arr);
        const avg = arr.reduce((a,b) => a+b, 0) / arr.length;
        if (circle) {
          circle.style.setProperty('--dynamic-scale', (1+avg/255*1.5).toFixed(3));
        }
      } catch (e) {
        console.warn("Error in animation pump:", e);
        return;
      }
      
      if (src.playbackState !== src.FINISHED_STATE) {
        requestAnimationFrame(pump);
      }
    }
    pump();
    
    return new Promise(resolve => {
      src.onended = () => {
        // Only resolve if this is still the current source and no interrupt
        if (src === currentAudioSource && !interruptRequested && !interruptInProgress) {
          resolve();
        } else {
          resolve(); // Still resolve to maintain chain
        }
      };
    });
  } catch (error) {
    console.error("Error playing audio chunk:", error);
    return Promise.resolve(); // Resolve anyway to keep chain going
  }
}

async function startRecording() {
  if (isRecording) return;
  try {
    const constraints = {
      audio: selectedMicId ? {deviceId:{exact:selectedMicId}} : true
    };
    micStream = await navigator.mediaDevices.getUserMedia(constraints);

    if (!audioContext) audioContext = new (AudioContext||webkitAudioContext)();
    const src = audioContext.createMediaStreamSource(micStream);
    const proc = audioContext.createScriptProcessor(4096,1,1);
    src.connect(proc); proc.connect(audioContext.destination);

    proc.onaudioprocess = e => {
      const samples = Array.from(e.inputBuffer.getChannelData(0));
      if (ws && ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(JSON.stringify({
            type:'audio',
            audio:samples,
            sample_rate:audioContext.sampleRate,
            session_id:SESSION_ID
          }));
        } catch (error) {
          console.error("Error sending audio data:", error);
          stopRecording();
        }
      }
    };

    window._micProcessor = proc;        
    isRecording = true;
    document.getElementById('micStatus').textContent = 'Listening…';
    showVoiceCircle();
  } catch (err) {
    console.error("Microphone access error:", err);
    showNotification('Microphone access denied','error');
  }
}

function stopRecording() {
  if (!isRecording) return;
  try {
    if (window._micProcessor) {
      window._micProcessor.disconnect();
      window._micProcessor = null;
    }
    if (micStream) {
      micStream.getTracks().forEach(t => t.stop());
      micStream = null;
    }
  } catch(e) {
    console.warn("Error stopping recording:", e);
  }
  isRecording = false;
  
  const micStatus = document.getElementById('micStatus');
  if (micStatus) {
    micStatus.textContent = 'Click to speak';
  }
  hideVoiceCircle();
}

async function setupChatUI() {
  document.documentElement.classList.add('bg-gray-950');
  document.documentElement.style.backgroundColor = '#030712';

  createPermanentVoiceCircle();
  connectWebSocket();

  initAudioLevelsChart();

  const txt = document.getElementById('textInput');
  const btn = document.getElementById('sendTextBtn');
  
  // Setup enhanced interrupt button
  const interruptBtn = document.createElement('button');
  interruptBtn.id = 'interruptBtn';
  interruptBtn.className = 'px-3 py-2 ml-2 bg-red-600 text-white rounded hover:bg-red-700 flex items-center transition duration-150';
  interruptBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-1" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clip-rule="evenodd" /></svg> Stop';
  interruptBtn.onclick = (e) => {
    e.preventDefault();
    try {
      requestInterrupt();
      interruptBtn.classList.add('bg-red-800', 'scale-95');
      setTimeout(() => interruptBtn.classList.remove('bg-red-800', 'scale-95'), 150);
    } catch (error) {
      console.error("Error in interrupt button handler:", error);
    }
  };
  interruptBtn.title = "Stop AI speech (Space or Esc)";
  interruptBtn.disabled = true; // Disabled by default
  interruptBtn.classList.add('opacity-50', 'cursor-not-allowed');
  
  if (btn && btn.parentElement) {
    btn.parentElement.appendChild(interruptBtn);
  }
  
  // Add debug button for easier debugging of interrupt issues
  const debugBtn = document.createElement('button');
  debugBtn.innerText = "Debug Audio";
  debugBtn.className = "px-3 py-2 ml-2 bg-blue-600 text-white rounded text-xs";
  debugBtn.onclick = () => {
    console.log("- Debug info:");
    console.log("- Audio playing:", isAudioCurrentlyPlaying);
    console.log("- Interrupt requested:", interruptRequested);
    console.log("- Interrupt in progress:", interruptInProgress);
    console.log("- Current source:", currentAudioSource);
    console.log("- Queue length:", audioPlaybackQueue.length);
    console.log("- Audio context state:", audioContext?.state);
    console.log("- Active generation ID:", activeGenId);
    console.log("- Last seen generation ID:", lastSeenGenId);
    console.log("- WebSocket state:", ws ? ws.readyState : "no websocket");
    showNotification("Debug info in console", "info");
  };
  
  if (btn && btn.parentElement) {
    btn.parentElement.appendChild(debugBtn);
  }
  
  // Run the update function periodically
  setInterval(() => {
    const interruptBtn = document.getElementById('interruptBtn');
    if (interruptBtn) {
      if (isAudioCurrentlyPlaying && !interruptRequested && !interruptInProgress) {
        interruptBtn.disabled = false;
        interruptBtn.classList.remove('opacity-50', 'cursor-not-allowed');
      } else {
        interruptBtn.disabled = true;
        interruptBtn.classList.add('opacity-50', 'cursor-not-allowed');
      }
    }
  }, 300);
  
  if (btn) {
    btn.onclick = () => {
      try {
        sendTextMessage(txt.value);
      } catch (error) {
        console.error("Error in send button handler:", error);
      }
    };
  }
  
  if (txt) {
    txt.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        try {
          sendTextMessage(txt.value);
        } catch (error) {
          console.error("Error in text input handler:", error);
        }
      }
    });
  }
  
  const micBtn = document.getElementById('micToggleBtn');
  if (micBtn) {
    micBtn.addEventListener('click', () => {
      try {
        if (isRecording) stopRecording();
        else startRecording();
      } catch (error) {
        console.error("Error in mic button handler:", error);
      }
    });
  }
  
  // Add event listeners to detect keyboard interruptions
  document.addEventListener('keydown', e => {
    // Allow space or escape to interrupt
    if ((e.code === 'Space' || e.code === 'Escape') && isAudioCurrentlyPlaying) {
      e.preventDefault();
      try {
        requestInterrupt();
        
        // Add visual feedback
        const interruptBtn = document.getElementById('interruptBtn');
        if (interruptBtn) {
          interruptBtn.classList.add('bg-red-800');
          setTimeout(() => {
            interruptBtn.classList.remove('bg-red-800');
          }, 200);
        }
      } catch (error) {
        console.error("Error in keyboard interrupt handler:", error);
      }
    }
  });
  
  // Initialize audio context
  if (!audioContext) {
    try {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      window.audioContext = audioContext;
    } catch (error) {
      console.error("Error creating audio context:", error);
      showNotification("Audio initialization failed. Please refresh the page.", "error");
    }
  }
  
  // Try to unlock audio context on user interaction
  ['click', 'touchstart', 'keydown'].forEach(ev =>
    document.addEventListener(ev, function unlock() {
      if (audioContext && audioContext.state === 'suspended') {
        try {
          audioContext.resume();
        } catch (error) {
          console.warn("Error resuming audio context:", error);
        }
      }
      document.removeEventListener(ev, unlock);
    })
  );

  console.log("Chat UI ready with enhanced interruption support");
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', setupChatUI);
} else {
  setupChatUI();
}

function initAudioLevelsChart() {
  const ctx = document.getElementById('audioLevels');
  if (!ctx) return;
  
  try {
    if (audioLevelsChart) audioLevelsChart.destroy();
    
    const grad = ctx.getContext('2d').createLinearGradient(0, 0, 0, 100);
    grad.addColorStop(0, 'rgba(79,70,229,.6)');
    grad.addColorStop(1, 'rgba(79,70,229,.1)');
    
    audioLevelsChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: Array(30).fill(''),
        datasets: [{
          data: Array(30).fill(0),
          backgroundColor: grad,
          borderColor: 'rgba(99,102,241,1)',
          borderWidth: 2,
          tension: .4,
          fill: true,
          pointRadius: 0
        }]
      },
      options: {
        animation: false,
        responsive: true,
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            ticks: {display: false},
            grid: {color: 'rgba(255,255,255,.1)'}
          },
          x: {display: false, grid: {display: false}}
        },
        plugins: {
          legend: {display: false},
          tooltip: {enabled: false}
        },
        elements: {point: {radius: 0}}
      }
    });
  } catch (error) {
    console.error("Error initializing audio chart:", error);
  }
}