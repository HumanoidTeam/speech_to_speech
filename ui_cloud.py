"""
HMND-01 Humanoid Robot Assistant – Integrated Flask + Voice Assistant Script
==========================================================================
This single file runs two main components:
  • A Flask web‑server that shows live status, last user message, last robot
    response, and a timestamp using the neon‑green terminal theme you provided.
  • A voice‑driven assistant powered by OpenAI, speech‑to‑text, and pygame
    audio playback.  The assistant updates a shared data structure that the
    Flask UI polls once per second via /get_display.

Before running make sure you have the following installed:
  pip install flask openai SpeechRecognition pygame pyaudio python-dotenv
and create a .env file with your OpenAI key:
  OPENAI_API_KEY="sk‑..."

Run with:
  python ui_cloudpy [--message "Hello world"]

Visit http://localhost:5000 in your browser to view the UI.
"""
import argparse
import json
import logging
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime

import pygame
import speech_recognition as sr
from flask import Flask, jsonify, render_template_string
from logging.handlers import RotatingFileHandler
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

###############################################################################
# ------------------------------  CONFIGURATION  ---------------------------- #
###############################################################################

audio_model = "tts-1"
chat_model = "gpt-4o-mini"
voice_name = "alloy"

# Replace the following strings with your real content.
ROBOTICS_KNOWLEDGE = """[Previous robotics knowledge content …]"""
ROBOT_IDENTITY = """[Previous robot identity content …]"""

###############################################################################
# -------------------------------  LOGGING  --------------------------------- #
###############################################################################

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("RainbowRobot")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = RotatingFileHandler("rainbow_robot.log", maxBytes=1_048_576, backupCount=5)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()

###############################################################################
# ------------------------  GLOBAL STATE FOR DISPLAY  ----------------------- #
###############################################################################

# These values are read by Flask and updated by the assistant.
display_lock = threading.Lock()
display_data = {
    "status": "Sleeping",
    "message": "",
    "response": "",
    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

def update_display(*, status: str | None = None, message: str | None = None, response: str | None = None):
    """Thread‑safe helper to update the values shown on the web UI."""
    with display_lock:
        if status is not None:
            display_data["status"] = status
        if message is not None:
            display_data["message"] = message
        if response is not None:
            display_data["response"] = response
        display_data["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

###############################################################################
# -----------------------------  FLASK APP  --------------------------------- #
###############################################################################

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
            align-self: flex-end;
            flex-direction: row-reverse;
        }
        .response-container {
            align-self: flex-start;
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
    </style>
    <script>
      let lastStatus = '';
      let lastMessage = '';
      let lastResponse = '';
      
      function updateDisplay() {
        fetch('/get_display').then(r => r.json()).then(data => {
          const statusContainer = document.getElementById('status-container');
          const currentStatus = data.status.toLowerCase();
          
          if (currentStatus !== lastStatus) {
            statusContainer.className = 'status-container ' + currentStatus;
            lastStatus = currentStatus;
          }
          
          document.getElementById('status').innerText = data.status;
          
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
      <p class="status-text" id="status">Status: {{ status }}</p>
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
        data = display_data.copy()
    return render_template_string(HTML_TEMPLATE, **data)

@app.route("/get_display")
def get_display():
    with display_lock:
        return jsonify(display_data)

###############################################################################
# ------------------------  CONVERSATION HISTORY  --------------------------- #
###############################################################################

class ConversationHistory:
    def __init__(self, max_history: int = 10):
        self.history_file = "conversation_history.json"
        self.history: deque[dict[str, str]] = deque(maxlen=max_history)
        self._load()

    # ---------- persistence ---------- #
    def _load(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as fh:
                    self.history = deque(json.load(fh), maxlen=self.history.maxlen)
            except Exception as exc:
                logger.error(f"Failed to load history: {exc}")

    def _save(self):
        try:
            with open(self.history_file, "w", encoding="utf-8") as fh:
                json.dump(list(self.history), fh, indent=2)
        except Exception as exc:
            logger.error(f"Failed to save history: {exc}")

    # ---------- public API ---------- #
    def add(self, user: str, robot: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.history.append({"timestamp": ts, "user": user, "robot": robot})
        self._save()

    def first(self):
        return self.history[0] if self.history else None

    def last(self):
        return self.history[-1] if self.history else None

    def recent(self, n: int = 3):
        return list(self.history)[-n:]

conversation_history = ConversationHistory()

###############################################################################
# ---------------------------  OPENAI CLIENT  ------------------------------- #
###############################################################################

# Initialize OpenAI client with API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please create a .env file with your API key.")

client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")

###############################################################################
# -----------------------  SPEECH & AUDIO UTILS  ---------------------------- #
###############################################################################

pygame.mixer.init(frequency=44_100, size=-16, channels=2, buffer=4096)

interrupted = False  # set by check_for_interruption()

def check_for_interruption():
    global interrupted
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 200
    with sr.Microphone() as src:
        while True:
            try:
                audio = recognizer.listen(src, timeout=1, phrase_time_limit=1)
                text = recognizer.recognize_google(audio, language="en-US").lower()
                if any(k in text for k in ("rainbow", "stop")):
                    interrupted = True
                    pygame.mixer.music.stop()
                    logger.info("Interrupted by user – keyword detected.")
                    break
            except Exception:
                # swallow recognizer errors in this tight loop
                continue

def speak(text: str) -> bool:
    """Return True if speech was interrupted."""
    global interrupted
    interrupted = False
    update_display(status="Speaking", response=text)

    # start background interruption listener
    t = threading.Thread(target=check_for_interruption, daemon=True)
    t.start()

    try:
        response = client.audio.speech.create(model=audio_model, voice=voice_name, input=text)
        tmp_path = "temp_speech.mp3"
        with open(tmp_path, "wb") as fh:
            fh.write(response.content)
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() and not interrupted:
            time.sleep(0.1)
    except Exception as exc:
        logger.error(f"TTS failure: {exc}")
    finally:
        pygame.mixer.music.unload()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return interrupted

###############################################################################
# ------------------------  OPENAI CHAT RESPONSE  --------------------------- #
###############################################################################

def get_response(user_text: str) -> str:
    # quick shortcuts for history‑related queries
    low = user_text.lower()
    if "first interaction" in low or "first conversation" in low:
        first = conversation_history.first()
        if first:
            return f"Our first interaction was at {first['timestamp']}. You said: '{first['user']}' and I responded: '{first['robot']}'."
        return "I don't have any previous interactions recorded."
    if "last interaction" in low or "previous conversation" in low:
        last = conversation_history.last()
        if last:
            return f"Our last interaction was at {last['timestamp']}. You said: '{last['user']}' and I responded: '{last['robot']}'."
        return "I don't have any previous interactions recorded."
    if "recent interactions" in low or "recent conversations" in low:
        recent = conversation_history.recent()
        if recent:
            return "\n".join(
                f"At {r['timestamp']} – You: '{r['user']}'  Me: '{r['robot']}'" for r in recent
            )
        return "No recent interactions."
    if any(k in low for k in ("repeat", "say that again", "repeat that")):
        last = conversation_history.last()
        return last['robot'] if last else "Nothing to repeat yet."

    # regular chat completion
    try:
        completion = client.chat.completions.create(
            model=chat_model,
            temperature=0.5,
            max_tokens=500,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a humanoid robot assistant with extensive robotics knowledge. "
                        "Respond in 2‑3 concise sentences, friendly and engaging. Maintain identity as HMND‑01.\n\n"
                        f"{ROBOTICS_KNOWLEDGE}\n\n{ROBOT_IDENTITY}"
                    ),
                },
                {"role": "user", "content": user_text},
            ],
        )
        return completion.choices[0].message.content
    except Exception as exc:
        logger.error(f"OpenAI chat error: {exc}")
        return "I'm having trouble thinking right now. Please try again later."

###############################################################################
# -----------------------------  LISTENING  --------------------------------- #
###############################################################################

is_active = False  # becomes True after wake word

def listen_for_wake_word():
    global is_active
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 100
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.5

    logger.info("Robot is sleeping. Say 'Hey robot', 'Hey robo', or 'Wake up' to activate me!")
    update_display(status="Sleeping")

    with sr.Microphone() as src:
        recognizer.adjust_for_ambient_noise(src, duration=1)
        while True:
            try:
                audio = recognizer.listen(src, timeout=5, phrase_time_limit=3)
                text = recognizer.recognize_google(audio, language="en-US").lower()
                logger.info(f"Heard: {text}")  # Log what was heard
                
                # Check for wake words
                wake_words = ["hey robot", "hey robo", "wake up"]
                if any(wake_word in text for wake_word in wake_words):
                    is_active = True
                    logger.info("Wake word detected → robot active")
                    update_display(status="Active", message="Wake word")
                    speak("Hello! I am HMND‑01, your humanoid robot assistant. How can I help you today?")
                    break
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                continue
            except Exception as exc:
                logger.error(f"Wake‑word error: {exc}")


def get_speech_input(timeout: int = 20, phrase_time_limit: int = 15):
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 200
    recognizer.pause_threshold = 1.0
    recognizer.non_speaking_duration = 0.5
    recognizer.phrase_threshold = 0.3

    with sr.Microphone() as src:
        update_display(status="Listening")
        logger.info("Listening for user question …")
        try:
            recognizer.adjust_for_ambient_noise(src, duration=1)
            audio = recognizer.listen(src, timeout=timeout, phrase_time_limit=phrase_time_limit)
            text = recognizer.recognize_google(audio, language="en-US")
            logger.info(f"Heard user input: {text}")
            
            # Check if the input contains wake words
            wake_words = ["hey robot", "hey robo", "wake up"]
            if any(wake_word in text.lower() for wake_word in wake_words):
                logger.info("Wake word detected in user input - ignoring")
                return None
                
            update_display(message=text)
            return text
        except sr.WaitTimeoutError:
            logger.warning("No speech detected (timeout)")
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
        except Exception as exc:
            logger.error(f"Recognizer error: {exc}")
        return None

###############################################################################
# -------------------------------  MAIN  ------------------------------------ #
###############################################################################

def run_flask():
    logger.info("Starting Flask UI on http://localhost:5000 …")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


def main():
    global is_active
    parser = argparse.ArgumentParser(description="HMND‑01 Voice Assistant with Web UI")
    parser.add_argument("--message", type=str, help="Speak a single message then exit")
    args = parser.parse_args()

    # start Flask in background
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # optional single message mode (no voice):
    if args.message:
        update_display(status="Speaking", message=args.message)
        speak(args.message)
        return

    # launch wake‑word listener
    wake_thread = threading.Thread(target=listen_for_wake_word, daemon=True)
    wake_thread.start()

    silence_count = 0
    max_silence = 3

    logger.info("Say 'Hey robot' to wake me up. 'Rainbow' or 'Stop' to interrupt. 'Goodbye' to end.")

    while True:
        if not is_active:  # still sleeping
            time.sleep(0.1)
            continue

        user_text = get_speech_input()
        if not user_text:
            silence_count += 1
            if silence_count >= max_silence:
                speak("No speech detected for too long. Going back to sleep. Say 'Hey robot' to wake me up!")
                is_active = False
                wake_thread = threading.Thread(target=listen_for_wake_word, daemon=True)
                wake_thread.start()
            else:
                speak(f"I didn't catch that. Please try again. {max_silence - silence_count} attempts remaining.")
            continue

        silence_count = 0

        if "goodbye" in user_text.lower():
            speak("Goodbye! Have a great day! Say 'Hey robot' when you need me again!")
            is_active = False
            wake_thread = threading.Thread(target=listen_for_wake_word, daemon=True)
            wake_thread.start()
            continue

        robot_reply = get_response(user_text)
        interrupted_speech = speak(robot_reply)
        conversation_history.add(user_text, robot_reply)
        update_display(response=robot_reply)

        if interrupted_speech:
            logger.info("Speech was interrupted by the user.")


if __name__ == "__main__":
    main()
