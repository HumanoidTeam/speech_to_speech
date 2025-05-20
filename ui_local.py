#!/usr/bin/env python3
"""
Rainbow Robot – fully-local speech-to-speech assistant **with real-time Web-UI**
-----------------------------------------------------------------------------
• Wake words:  "hey robot", "hey robo", "wake up"
• Interrupts:  "rainbow", "stop", "shut up", "wait"
• Quit word :  "goodbye"

Run the file then browse to **http://localhost:5050** to see the animated
HMND-01 dashboard that mirrors the terminal interaction.

Requirements
============
    pip install flask sounddevice numpy pyttsx3 faster-whisper ollama

Tested on Python 3.10+ (Linux/macOS/Windows).
"""

###############################################################################
# -----------------------------  CONFIGURATION ------------------------------ #
###############################################################################
WAKE_WORDS   = ("hey robot", "hey robo", "wake up")
STOP_WORDS   = ("rainbow", "stop", "shut up", "wait", "no")
QUIT_WORDS   = ("goodbye",)

# Local knowledge base about robotics
ROBOTICS_KNOWLEDGE = """
Key Robotics Companies and Developments:
- Humanoid: A London-based humanoid robotics company founded in 2024, developing the HMND-01, the world's most reliable and commercially viable humanoid robot. The HMND-01 is customizable, modular, and designed for low Total Cost of Ownership, addressing critical global labor shortages across industrial, warehouse automation, service, manufacturing, and household sectors.

Other Notable Companies:
- Tesla: Developing Optimus humanoid robot for manufacturing and household tasks
- Boston Dynamics: Known for Atlas and Spot robots, focusing on mobility and manipulation
- Figure AI: Working on general-purpose humanoid robots for labor
- Agility Robotics: Creator of Digit, a bipedal robot for logistics
- Apptronik: Developing Apollo humanoid for industrial applications
- Sanctuary AI: Creating humanoid robots with advanced AI capabilities
- 1X Technologies: Building humanoid robots for various applications
- Unitree: Known for quadruped robots and humanoid development
- Fourier Intelligence: Developing rehabilitation and service robots
- UBTech: Creating educational and service robots

Recent Major Developments:
- Humanoid's HMND-01 represents a breakthrough in commercial humanoid robotics
- Focus on developing more dexterous manipulation capabilities
- Integration of large language models for better human-robot interaction
- Emphasis on safety and reliability in human-robot collaboration
- Growing investment in humanoid robotics startups
"""

ROBOT_IDENTITY = """
You are an advanced humanoid robot assistant developed by Humanoid, a London-based robotics company founded in 2024. You are part of the HMND series, designed to be the world's most reliable and commercially viable humanoid robot. Your purpose is to address critical global labor shortages and enhance human well-being by taking on dangerous, repetitive, or complex tasks across industrial, warehouse automation, service, manufacturing, and household sectors.

Key aspects of your identity:
- You are a prototype of the HMND-01 series, which will be commercially available later in 2025
- You are customizable, modular, and designed for low Total Cost of Ownership
- You are built with a focus on reliability and commercial viability
- You are part of Humanoid's mission to revolutionize human-robot collaboration
- You are designed to work alongside humans in various sectors

Technical specs:
- Height: 175 cm (5'9")
- Weight: 70 kg (154 lbs)
- Payload: 15 kg (33 lbs)
- Walking speed: 1.5 m/s (5.4 km/h)
- Runtime: 4 hours
- DOF: 41
"""

SYSTEM_TONE = (
    "You are a humanoid robot assistant with extensive knowledge about robotics. "
    "Here is your knowledge base and identity:\n\n"
    f"{ROBOTICS_KNOWLEDGE}\n\n"
    f"{ROBOT_IDENTITY}\n\n"
    "Provide concise (≤3-sentence) spoken-friendly answers in a friendly tone, "
    "using your robotics knowledge when relevant and always acting as a Humanoid HMND robot.\n\n"
    "User: {user}\nRobot:"
)

###############################################################################
# --------------------------------- IMPORTS --------------------------------- #
###############################################################################
import argparse, logging, threading, time, os, signal, sys, json
from datetime import datetime
from collections import deque

import numpy as np
import sounddevice as sd
import pyttsx3
from faster_whisper import WhisperModel
from ollama import Client

# Flask UI
from flask import Flask, jsonify, render_template_string

###############################################################################
# -------------------------  LOGGER CONFIGURATION --------------------------- #
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("RainbowRobot")

###############################################################################
# ----------------------  REAL-TIME WEB-UI (Flask)  ------------------------- #
###############################################################################
# Shared state between robot threads and Flask
display_lock  = threading.Lock()
display_state = {
    "status"  : "sleeping",  # sleeping | listening | thinking | speaking
    "message" : "",
    "response": "",
    "time"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

def _update_display(**kwargs):
    """Thread-safe helper the robot uses to push updates to the UI."""
    with display_lock:
        display_state.update(kwargs)
        display_state["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Flask application
app = Flask(__name__)

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html>
<head>
    <title>HMND Robot</title>
    <style>
        body {
            background-color: #1a1a1a;
            color: #ffffff;
            font-family: 'Courier New', monospace;
            margin: 20px;
        }
        .container { 
            max-width: 800px; 
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 30px;
            background: linear-gradient(45deg, #00ff00, #00ffff, #00ff00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 72px;
            font-weight: 900;
            letter-spacing: 5px;
            text-transform: uppercase;
            text-shadow: 0 0 20px rgba(0, 255, 0, 0.5);
            animation: glow 2s ease-in-out infinite alternate;
            position: relative;
        }
        .header::after {
            content: '';
            position: absolute;
            bottom: 0; left: 50%; transform: translateX(-50%);
            width: 80%; height: 2px;
            background: linear-gradient(90deg, transparent, #00ff00, transparent);
            animation: lineGlow 2s ease-in-out infinite alternate;
        }
        .description {
            text-align: center; 
            margin: -20px auto 30px; 
            padding: 20px;
            color: #00ff00; 
            font-size: 1.2em; 
            line-height: 1.6; 
            max-width: 600px;
            text-shadow: 0 0 10px rgba(0, 255, 0, 0.3); 
            animation: fadeIn 1s ease-out;
        }
        .keywords {
            text-align: center;
            margin: 20px auto;
            padding: 15px;
            max-width: 300px;
            background: rgba(0, 255, 0, 0.1);
            border: 1px solid #00ff00;
            border-radius: 8px;
        }
        
        .keywords p {
            margin: 8px 0;
            color: #00ff00;
            font-size: 1em;
        }
        .status-container {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            border-radius: 20px;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            align-items: center;
            gap: 10px;
            z-index: 1000;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        .status-waves {
            position: absolute;
            top: 50%;
            left: 20px;
            transform: translateY(-50%);
            width: 12px;
            height: 12px;
            border-radius: 50%;
            opacity: 0;
        }
        .status-text {
            font-size: 0.9em;
            color: #ffffff;
        }
        .conversation {
            margin-top: 40px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .message-container, .response-container {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            padding: 10px;
            opacity: 0;
            transform: translateY(20px);
            animation: fadeInUp 0.5s forwards;
        }
        .message-container {
            align-self: flex-start;
            flex-direction: row;
        }
        .response-container {
            align-self: flex-end;
            flex-direction: row-reverse;
        }
        .emoji {
            font-size: 1.5em;
            min-width: 40px;
            text-align: center;
        }
        .message-text, .response-text {
            padding: 10px 15px;
            border-radius: 15px;
            max-width: 70%;
            line-height: 1.4;
        }
        .message-text {
            background: rgba(255, 255, 0, 0.1);
            color: #ffff00;
            border: 1px solid rgba(255, 255, 0, 0.3);
        }
        .response-text {
            background: rgba(0, 255, 255, 0.1);
            color: #00ffff;
            border: 1px solid rgba(0, 255, 255, 0.3);
        }
        .time {
            text-align: center;
            color: #666;
            font-size: 0.8em;
            margin: 20px 0;
        }
        @keyframes fadeInUp {
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        @keyframes glow {
            from { text-shadow: 0 0 20px rgba(0,255,0,.5),0 0 40px rgba(0,255,0,.3),0 0 60px rgba(0,255,0,.2);}
            to { text-shadow: 0 0 30px rgba(0,255,0,.7),0 0 60px rgba(0,255,0,.5),0 0 90px rgba(0,255,0,.3);}
        }
        @keyframes lineGlow {
            from { box-shadow: 0 0 10px rgba(0,255,0,.3),0 0 20px rgba(0,255,0,.2);}
            to { box-shadow: 0 0 20px rgba(0,255,0,.5),0 0 40px rgba(0,255,0,.3);}
        }
        /* Sleeping Animation */
        .sleeping .status-indicator {
            background: #ff0000;
            opacity: 0.7;
            box-shadow: 0 0 10px rgba(255, 0, 0, 0.5);
        }
        /* Listening Animation */
        .listening .status-indicator {
            background: #00ff00;
            animation: listening 1s infinite;
        }
        .listening .status-waves {
            background: #00ff00;
            animation: listeningWaves 1s infinite;
        }
        @keyframes listening {
            0% { transform: scale(1); }
            50% { transform: scale(1.2); }
            100% { transform: scale(1); }
        }
        @keyframes listeningWaves {
            0% { transform: scale(1); opacity: 0.8; }
            100% { transform: scale(2); opacity: 0; }
        }
        /* Speaking Animation */
        .speaking .status-indicator {
            background: #00ffff;
            animation: speaking 0.5s infinite;
        }
        .speaking .status-waves {
            background: #00ffff;
            animation: speakingWaves 0.5s infinite;
        }
        @keyframes speaking {
            0% { transform: scale(1); }
            50% { transform: scale(1.5); }
            100% { transform: scale(1); }
        }
        @keyframes speakingWaves {
            0% { transform: scale(1); opacity: 0.8; }
            100% { transform: scale(2.5); opacity: 0; }
        }
        /* Thinking Animation */
        .thinking .status-indicator {
            background: #ff00ff;
            animation: thinking 1.5s infinite;
        }
        .thinking .status-waves {
            background: #ff00ff;
            animation: thinkingWaves 1.5s infinite;
        }
        @keyframes thinking {
            0% { transform: scale(1) rotate(0deg); }
            25% { transform: scale(1.2) rotate(90deg); }
            50% { transform: scale(1) rotate(180deg); }
            75% { transform: scale(1.2) rotate(270deg); }
            100% { transform: scale(1) rotate(360deg); }
        }
        @keyframes thinkingWaves {
            0% { transform: scale(1); opacity: 0.8; }
            50% { transform: scale(2); opacity: 0.4; }
            100% { transform: scale(3); opacity: 0; }
        }
        .thinking-emoji {
            font-size: 1.5em;
            animation: thinkingEmoji 2s infinite;
        }
        @keyframes thinkingEmoji {
            0% { transform: rotate(0deg); }
            25% { transform: rotate(15deg); }
            75% { transform: rotate(-15deg); }
            100% { transform: rotate(0deg); }
        }
    </style>
    <script>
      let lastStatus = '';
      let lastMessage = '';
      let lastResponse = '';
      
      function updateDisplay() {
        fetch('/get_display').then(r => r.json()).then(data => {
          const statusContainer = document.getElementById('status-container');
          const thinkingEmoji = document.getElementById('thinking-emoji');
          const statusText = document.getElementById('status-text');
          const currentStatus = data.status.toLowerCase();
          
          if (currentStatus !== lastStatus) {
            statusContainer.className = 'status-container ' + currentStatus;
            lastStatus = currentStatus;
            
            // Show/hide thinking emoji and update status text
            if (currentStatus === 'thinking') {
              thinkingEmoji.style.display = 'inline-block';
              statusText.textContent = 'Thinking...';
            } else {
              thinkingEmoji.style.display = 'none';
              statusText.textContent = 'Status: ' + data.status;
            }
          }
          
          if (data.message !== lastMessage) {
            const messageContainer = document.getElementById('message-container');
            messageContainer.style.animation = 'none';
            messageContainer.offsetHeight;
            messageContainer.style.animation = 'fadeInUp 0.5s forwards';
            
            const messageText = data.message ? data.message : 'No message yet';
            document.getElementById('message').innerText = messageText;
            lastMessage = data.message;
          }
          
          if (data.response !== lastResponse) {
            const responseContainer = document.getElementById('response-container');
            responseContainer.style.animation = 'none';
            responseContainer.offsetHeight;
            responseContainer.style.animation = 'fadeInUp 0.5s forwards';
            
            const responseText = data.response ? data.response : 'No response yet';
            document.getElementById('response').innerText = responseText;
            lastResponse = data.response;
          }
          
          document.getElementById('time').innerText = data.time;
        });
      }
      
      setInterval(updateDisplay, 1000);
      window.onload = updateDisplay;
    </script>
</head>
<body>
  <div class="container">
    <div class="header">HMND-01</div>
    <div class="description">
      Our first humanoid robot. Designed to be customizable, modular, and reliable with a low Total Cost of Ownership.
    </div>
    <div class="keywords">
      <p>Wake Word: wake up</p>
      <p>Interrupt: rainbow, stop</p>
      <p>End Conversation: goodbye</p>
    </div>
    <div id="status-container" class="status-container">
      <div class="status-indicator"></div>
      <div class="status-waves"></div>
      <p class="status-text" id="status">
        <span class="thinking-emoji" id="thinking-emoji" style="display: none;">🤔</span>
        <span id="status-text">Status: {{ status }}</span>
      </p>
    </div>
    <div class="conversation">
      <div id="message-container" class="message-container">
        <span class="emoji">👤</span>
        <p class="message-text" id="message">{{ message }}</p>
      </div>
      <div id="response-container" class="response-container">
        <span class="emoji">🤖</span>
        <p class="response-text" id="response">{{ response }}</p>
      </div>
    </div>
    <p class="time" id="time">{{ time }}</p>
  </div>
</body>
</html>
"""

@app.route("/")
def index():
    with display_lock:
        data = display_state.copy()
    return render_template_string(HTML_TEMPLATE, **data)

@app.route("/get_display")
def get_display():
    with display_lock:
        return jsonify(display_state)

def start_ui_server(host="0.0.0.0", port=5050):
    """Run Flask in a daemon thread so it never blocks the robot."""
    threading.Thread(
        target=lambda: app.run(host=host, port=port, threaded=True, use_reloader=False),
        daemon=True,
    ).start()
    log.info(f"🌐 Web-UI running on http://{host}:{port}")

###############################################################################
# --------------------------  SPEECH RECOGNITION ---------------------------- #
###############################################################################
class SpeechRecognizer:
    def __init__(self, model_id: str, device: str):
        log.info(f'Loading Whisper model "{model_id}" …')
        self.model = WhisperModel(model_id, device=device, compute_type="float16" if device == "cuda" else "int8")
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.threshold = 0.005  # Lowered threshold for better voice detection
        self.min_speech_duration = 0.3  # Minimum duration of speech to consider
        self.max_speech_duration = 15.0  # Maximum duration to process
        self.speech_pad = 0.2  # Padding around speech segments

    def _normalize_audio(self, audio):
        """Normalize audio to improve voice detection."""
        return audio / (np.max(np.abs(audio)) + 1e-8)  # Added small epsilon to prevent division by zero

    def _detect_voice_activity(self, audio):
        """Enhanced voice activity detection using energy and zero-crossing rate."""
        # Calculate energy
        energy = np.mean(np.square(audio))
        
        # Calculate zero-crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio))))
        zcr = zero_crossings / (len(audio) - 1)
        
        # Combined detection using both metrics
        return energy > self.threshold or (energy > self.threshold * 0.5 and zcr > 0.1)

    def _find_strongest_signal(self, audio, sr):
        """Find the segment with the strongest voice signal with improved context."""
        # Split audio into smaller chunks for more precise detection
        chunk_length = int(0.3 * sr)  # 0.3 second chunks for finer granularity
        chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]
        
        # Calculate energy for each chunk
        energies = [np.mean(np.square(chunk)) for chunk in chunks]
        
        # Find chunks above threshold
        above_threshold = [i for i, e in enumerate(energies) if e > self.threshold]
        
        if not above_threshold:
            return audio  # Return full audio if no clear signal found
        
        # Find the longest continuous segment above threshold
        segments = []
        current_segment = [above_threshold[0]]
        
        for i in range(1, len(above_threshold)):
            if above_threshold[i] - above_threshold[i-1] <= 2:  # Allow small gaps
                current_segment.append(above_threshold[i])
            else:
                segments.append(current_segment)
                current_segment = [above_threshold[i]]
        segments.append(current_segment)
        
        # Find the longest segment
        longest_segment = max(segments, key=len)
        
        # Add padding before and after
        start_idx = max(0, longest_segment[0] - 1)
        end_idx = min(len(chunks), longest_segment[-1] + 2)
        
        return np.concatenate(chunks[start_idx:end_idx])

    @staticmethod
    def _record(seconds=3, sr=16_000):
        """Record audio with improved voice detection."""
        audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="int16")
        sd.wait()
        return (audio.flatten().astype(np.float32) / 32768.0, sr)

    def listen(self, seconds=3) -> str:
        """Listen for speech with improved voice detection and noise filtering."""
        data, sr = self._record(seconds, self.sample_rate)
        
        # Normalize audio
        normalized_audio = self._normalize_audio(data)
        
        # Check if there's any voice activity
        if not self._detect_voice_activity(normalized_audio):
            log.info("No significant voice activity detected")
            return ""
        
        # Find the strongest signal segment
        strongest_signal = self._find_strongest_signal(normalized_audio, sr)
        
        # Transcribe the strongest signal with improved parameters
        segments, _ = self.model.transcribe(
            strongest_signal,
            beam_size=5,  # Increased beam size for better accuracy
            word_timestamps=True,  # Enable word timestamps
            condition_on_previous_text=True,  # Use previous context
            initial_prompt="The following is a conversation with a human."  # Add context
        )
        segments_list = list(segments)
        
        if not segments_list:
            return ""
            
        text = " ".join(s.text for s in segments_list).strip()
        
        if text:
            log.info("Detected speech")
        
        return text

###############################################################################
# ------------------------------  TTS (pyttsx3) ----------------------------- #
###############################################################################
class Speaker:
    def __init__(self, voice_name: str | None = None, rate: int = 150):
        self.voice_name = voice_name
        self.rate = rate
        self._speaking = False
        self._engine = None
        self._init_engine()
        self._speech_complete = threading.Event()
        self._interrupted = False

    def _init_engine(self):
        if self._engine is None:
            self._engine = pyttsx3.init()
            if self.voice_name:
                for v in self._engine.getProperty("voices"):
                    if self.voice_name.lower() in v.name.lower():
                        self._engine.setProperty("voice", v.id)
                        break
            self._engine.setProperty("rate", self.rate)
            
            def on_end(name, completed):
                self._speaking = False
                self._speech_complete.set()
            
            self._engine.connect('finished-utterance', on_end)

    def _cleanup_engine(self):
        if self._engine is not None:
            try:
                self._engine.stop()
            finally:
                self._engine = None
                self._speaking = False
                self._speech_complete.set()

    def say(self, text: str):
        self._speaking = True
        self._interrupted = False
        self._speech_complete.clear()
        try:
            self._init_engine()
            self._engine.say(text)
            self._engine.runAndWait()
        finally:
            self._speaking = False
            self._speech_complete.set()
            self._cleanup_engine()

    def stop(self):
        if self._speaking:
            self._interrupted = True
            self._cleanup_engine()
            self._speaking = False
            self._speech_complete.set()

    def is_speaking(self):
        return self._speaking

    def is_interrupted(self):
        return self._interrupted

    def wait_for_completion(self, timeout=0.1):
        return self._speech_complete.wait(timeout)

    def __del__(self):
        self._cleanup_engine()

###############################################################################
# ------------------------------  LOCAL LLM  -------------------------------- #
###############################################################################
class LocalChat:
    def __init__(self, model_id: str):
        log.info(f'Connecting to Ollama model "{model_id}" …')
        self.model  = model_id
        self.client = Client()
        try:
            # Test connection with a simple ping
            self.client.chat(model=self.model, messages=[{"role": "user", "content": "ping"}])
            log.info("✅ Connected to Ollama")
        except Exception as e:
            log.error(f"❌ Unable to reach Ollama: {str(e)}")
            log.error("Please make sure:")
            log.error("1. Ollama is installed and running (run 'ollama serve' in terminal)")
            log.error("2. The model is downloaded (run 'ollama pull " + model_id + "')")
            log.error("3. You can access http://127.0.0.1:11434")
            raise

    def reply(self, user_text: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_TONE.format(user=user_text)},
            {"role": "user",   "content": user_text},
        ]
        try:
            result = self.client.chat(model=self.model, messages=messages)
            return result["message"]["content"].strip()
        except Exception as exc:
            log.error(f"LLM error: {exc}")
            return "I'm having trouble thinking right now. Please make sure Ollama is running."

###############################################################################
# --------------------------  CONVERSATION HISTORY -------------------------- #
###############################################################################
class ConversationHistory:
    def __init__(self, max_history=10):
        self.history      = deque(maxlen=max_history)
        self.history_file = "conversation_history_local.json"
        self.load_history()

    # basic CRUD helpers
    def add_interaction(self, user_input: str, robot_response: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.history.append({
            "timestamp": timestamp,
            "user"     : user_input,
            "robot"    : robot_response,
        })
        self.save_history()

    def get_last(self):
        return self.history[-1] if self.history else None
    def get_all(self):
        return list(self.history)
    def get_recent(self, n=3):
        return list(self.history)[-n:]

    def save_history(self):
        try:
            with open(self.history_file, "w") as f:
                json.dump(list(self.history), f, indent=2)
        except Exception as e:
            log.error(f"Could not save history: {e}")

    def load_history(self):
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, "r") as f:
                    self.history = deque(json.load(f), maxlen=self.history.maxlen)
        except Exception as e:
            log.error(f"Could not load history: {e}")

###############################################################################
# ------------------------------  CORE ROBOT -------------------------------- #
###############################################################################
class RainbowRobot:
    def __init__(self, recognizer: SpeechRecognizer, speaker: Speaker, brain: LocalChat):
        self.recognizer = recognizer
        self.speaker = speaker
        self.brain = brain
        self.state = {"awake": False}
        self.running = True
        self.last_response = ""
        self._should_stop = False
        self.history = ConversationHistory()
        self.silence_threshold = 0.01  # Adjust this value based on your environment
        self.silence_duration = 0  # Track silence duration

        # initial UI
        self._set_ui(status="sleeping")

        log.info("🤖 Rainbow Robot initialised – waiting for wake word …")

    # UI helper
    def _set_ui(self, **kwargs):
        _update_display(**kwargs)

    # ------ high-level lifecycle helpers
    def shutdown(self):
        log.info("Shutting down …")
        self.running = False
        self.speaker.stop()
        self._set_ui(status="sleeping")
        log.info("Goodbye 👋")

    def check_for_wake_word(self):
        heard = self.recognizer.listen(2).lower()
        if heard:
            log.info(f"Heard: {heard}")
            if any(w in heard for w in WAKE_WORDS):
                self.state["awake"] = True
                log.info("🎯 Wake word detected!")
                self._set_ui(status="listening", message=heard)
                welcome_message = "Hello! I am HMND-01. How can I help you today?"
                self._set_ui(response=welcome_message)
                self.speaker.say(welcome_message)
                # Add the wake word interaction to history
                self.history.add_interaction(heard, welcome_message)
                return True
        return False

    # speak with interrupt capability
    def speak_with_interrupt(self, text: str):
        self._should_stop = False
        self._set_ui(status="speaking", response=text)

        def check_for_interruption():
            while self.speaker.is_speaking():
                try:
                    heard = self.recognizer.listen(1).lower()
                    if heard and any(w in heard for w in STOP_WORDS):
                        log.info("🛑 Interrupt word detected!")
                        self._should_stop = True
                        self.speaker.stop()
                        interruption_message = "I've been interrupted. How can I help you?"
                        self._set_ui(status="listening", response=interruption_message)
                        break
                except Exception:
                    continue

        # Start speaking in a separate thread
        speak_thread = threading.Thread(target=lambda: self.speaker.say(text))
        speak_thread.start()

        # Start interrupt listener in a separate thread
        interrupt_thread = threading.Thread(target=check_for_interruption)
        interrupt_thread.start()

        # Wait for speaking to complete or be interrupted
        speak_thread.join()
        interrupt_thread.join(timeout=0.5)

        # Update UI based on whether speech was interrupted
        if self.speaker.is_interrupted():
            self._set_ui(status="listening")
        else:
            self._set_ui(status="listening")

        return self.speaker.is_interrupted()

    # optional history queries
    def history_query(self, user_input: str):
        lower = user_input.lower()
        if "first interaction" in lower:
            first = self.history.get_all()[0] if self.history.get_all() else None
            if first:
                return f"Our first interaction was at {first['timestamp']}: you said '{first['user']}' and I replied '{first['robot']}'."
            return "I don't have any earlier interactions."
        if "last interaction" in lower:
            last = self.history.get_last()
            if last:
                return f"Our last interaction was at {last['timestamp']}: you said '{last['user']}' and I replied '{last['robot']}'."
            return "We haven't spoken yet."
        if "recent interactions" in lower:
            recent = self.history.get_recent(3)
            if recent:
                parts = [f"At {r['timestamp']} – you: '{r['user']}' | me: '{r['robot']}'" for r in recent]
                return "Here are our recent chats:\n" + "\n".join(parts)
        if "repeat" in lower and self.last_response:
            return self.last_response
        return None

    # main loop
    def run(self):
        while self.running:
            if not self.state["awake"]:
                if self.check_for_wake_word():
                    continue
                time.sleep(0.1)
                continue

            self._set_ui(status="listening")
            log.info("🎤 Listening for your question …")
            user = self.recognizer.listen(10)

            if not user:
                log.info("❌ No significant speech detected")
                no_speech_message = "I didn't catch that. Did you say something?"
                self._set_ui(message="", response=no_speech_message)
                self.speak_with_interrupt(no_speech_message)
                continue

            log.info(f"👤 You said: {user}")
            self._set_ui(message=user)

            # Check for stop words in user input
            if any(w in user.lower() for w in STOP_WORDS):
                log.info("🛑 Stop word detected in user input")
                self.speaker.stop()
                self._should_stop = True
                self._set_ui(status="listening", response="")
                continue

            if any(q in user.lower() for q in QUIT_WORDS):
                log.info("👋 Quit word detected")
                goodbye_message = "Goodbye! Say 'hey robot' when you need me again."
                self._set_ui(response=goodbye_message)
                self.speak_with_interrupt(goodbye_message)
                self.history.add_interaction(user, goodbye_message)
                self.state["awake"] = False
                self._set_ui(status="sleeping")
                continue

            # history queries
            hist = self.history_query(user)
            if hist:
                log.info("📜 History query")
                self._set_ui(response=hist)
                self.speak_with_interrupt(hist)
                continue

            # LLM answer
            log.info("🤔 Thinking …")
            self._set_ui(status="thinking")
            answer = self.brain.reply(user)
            self.last_response = answer
            log.info(f"🤖 {answer}")

            # store & speak
            self.history.add_interaction(user, answer)
            self._set_ui(response=answer)
            interrupted = self.speak_with_interrupt(answer)
            if interrupted:
                log.info("Ready for new input …")

###############################################################################
# --------------------------------  SIGNALS ---------------------------------#
###############################################################################
robot: RainbowRobot | None = None

def signal_handler(signum, frame):
    log.info("\nReceived shutdown signal")
    if robot:
        robot.shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

###############################################################################
# --------------------------------- ENTRY -----------------------------------#
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Rainbow Robot (fully local)")
    parser.add_argument("--voice",      default="en_GB-alba-low", help="Piper voice ID / path")
    parser.add_argument("--stt_model",  default="base.en",        help="Whisper model (tiny.en, base.en, ...)")
    parser.add_argument("--llm",        default="llama3",         help="Ollama model name")
    parser.add_argument("--device",     default="cpu",           help="cuda | cpu")
    args = parser.parse_args()

    # Check if Ollama is running before starting
    try:
        import requests
        response = requests.get("http://127.0.0.1:11434/api/tags")
        if response.status_code != 200:
            log.error("❌ Ollama server is not responding properly")
            log.error("Please make sure Ollama is running (run 'ollama serve' in terminal)")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        log.error("❌ Cannot connect to Ollama server")
        log.error("Please make sure Ollama is running (run 'ollama serve' in terminal)")
        sys.exit(1)

    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    recognizer = SpeechRecognizer(args.stt_model, args.device)
    speaker = Speaker(voice_name=None)     # pyttsx3 version; pass a name if you like
    brain = LocalChat(args.llm)

    # Start the Flask UI server
    start_ui_server()

    global robot
    robot = RainbowRobot(recognizer, speaker, brain)
    robot.run()

if __name__ == "__main__":
    import shutil   # imported late so pylint doesn't complain
    main()
