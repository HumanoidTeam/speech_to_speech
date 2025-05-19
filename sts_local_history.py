#!/usr/bin/env python3
"""
This is the 100% offline speech-to-speech assistant with conversation history

"""

# ─────────── configuration ───────────
WAKE_WORDS   = ("hey robot", "hey robo", "wake up")
STOP_WORDS   = ("rainbow", "stop", "shut up", "wait")
QUIT_WORDS   = ("goodbye",)

# Local knowledge base about robotics
ROBOTICS_KNOWLEDGE = """
[Previous robotics knowledge content...]
"""

ROBOT_IDENTITY = """
[Previous robot identity content...]
"""

SYSTEM_TONE = (
    "You are a humanoid robot assistant with extensive knowledge about robotics. Here is your knowledge base and identity:\n\n"
    f"{ROBOTICS_KNOWLEDGE}\n\n"
    f"{ROBOT_IDENTITY}\n\n"
    "Please provide brief and concise responses (2-3 sentences maximum) that can be spoken naturally. "
    "Your answers must be in a friendly and engaging tone, and you should use your robotics knowledge when relevant to the conversation. "
    "Always maintain your identity as a Humanoid HMND series robot when appropriate.\n\n"
    "User: {user}\nRobot:"
)

# ─────────── imports ───────────
import argparse, logging, threading, time, queue
import numpy as np
import sounddevice as sd
import pyttsx3
from faster_whisper import WhisperModel
from ollama import Client
import signal
import sys
import json
from datetime import datetime
from collections import deque

# ─────────── utility: logger ───────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("RainbowRobot")

# ─────────── Conversation History ───────────
class ConversationHistory:
    def __init__(self, max_history=10):
        self.history = deque(maxlen=max_history)
        self.history_file = "conversation_history_local.json"
        self.load_history()

    def add_interaction(self, user_input, robot_response):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.history.append({
            "timestamp": timestamp,
            "user": user_input,
            "robot": robot_response
        })
        self.save_history()

    def get_last_interaction(self):
        if self.history:
            return self.history[-1]
        return None

    def get_recent_interactions(self, n=3):
        return list(self.history)[-n:]

    def get_all_interactions(self):
        return list(self.history)

    def save_history(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(list(self.history), f, indent=2)
        except Exception as e:
            log.error(f"Error saving conversation history: {e}")

    def load_history(self):
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.history = deque(json.load(f), maxlen=self.history.maxlen)
        except Exception as e:
            log.error(f"Error loading conversation history: {e}")

# ─────────── STT: Whisper ───────────
class SpeechRecognizer:
    def __init__(self, model_id: str, device: str):
        log.info(f'Loading Whisper model "{model_id}" ...')
        self.model = WhisperModel(model_id, device=device, compute_type="float16" if device == "cuda" else "int8")

    @staticmethod
    def _record(seconds=3, sr=16_000):
        audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="int16")
        sd.wait()
        return (audio.flatten().astype(np.float32) / 32768.0, sr)

    def listen(self, seconds=3) -> str:
        data, sr = self._record(seconds)
        segments, _ = self.model.transcribe(data, beam_size=2)
        return " ".join(s.text for s in segments).strip()

# ─────────── TTS: pyttsx3 ───────────
class Speaker:
    def __init__(self, voice_name: str | None = None, rate: int = 150):
        self.voice_name = voice_name
        self.rate = rate
        self._speaking = False
        self._engine = None
        self._init_engine()

    def _init_engine(self):
        if self._engine is None:
            self._engine = pyttsx3.init()
            if self.voice_name:
                for v in self._engine.getProperty("voices"):
                    if self.voice_name.lower() in v.name.lower():
                        self._engine.setProperty("voice", v.id)
                        break
            self._engine.setProperty("rate", self.rate)

    def _cleanup_engine(self):
        if self._engine is not None:
            try:
                self._engine.stop()
            except:
                pass
            self._engine = None

    def say(self, text: str):
        self._speaking = True
        try:
            self._init_engine()
            self._engine.say(text)
            self._engine.runAndWait()
        finally:
            self._speaking = False
            self._cleanup_engine()

    def stop(self):
        if self._speaking:
            self._cleanup_engine()
            self._speaking = False

    def is_speaking(self):
        return self._speaking

    def __del__(self):
        self._cleanup_engine()

# ─────────── LLM: Ollama ───────────
class LocalChat:
    def __init__(self, model_id: str):
        log.info(f'Loading Ollama model "{model_id}" ...')
        self.model = model_id
        self.client = Client()
        try:
            self.client.chat(model=self.model, messages=[{"role": "user", "content": "test"}])
            log.info("✅ Successfully connected to Ollama")
        except ConnectionError:
            log.error("❌ Failed to connect to Ollama!")
            log.error("Please make sure Ollama is installed and running:")
            log.error("1. Download from https://ollama.ai/download")
            log.error("2. Install and run 'ollama serve'")
            log.error("3. Run 'ollama pull llama3' to download the model")
            raise

    def reply(self, user_text: str) -> str:
        try:
            messages = [
                {"role": "system", "content": SYSTEM_TONE.format(user=user_text)},
                {"role": "user", "content": user_text}
            ]
            result = self.client.chat(model=self.model, messages=messages)
            return result["message"]["content"].strip()
        except ConnectionError:
            log.error("❌ Lost connection to Ollama!")
            log.error("Please make sure Ollama is running with 'ollama serve'")
            return "I'm having trouble connecting to my brain. Please make sure Ollama is running."

# ─────────── high-level assistant logic ───────────
class RainbowRobot:
    def __init__(self, recognizer, speaker, brain):
        self.recognizer = recognizer
        self.speaker = speaker
        self.brain = brain
        self.state = {"awake": False}
        self.running = True
        self.last_response = None
        self._should_stop = False
        self.conversation_history = ConversationHistory()
        log.info("🤖 Rainbow Robot initialized!")
        log.info(f"Wake words: {', '.join(WAKE_WORDS)}")
        log.info(f"Stop words: {', '.join(STOP_WORDS)}")
        log.info(f"Quit words: {', '.join(QUIT_WORDS)}")
        log.info("Waiting for wake word...")
        log.info("Press Ctrl+C to exit")

    def shutdown(self):
        log.info("Shutting down...")
        self.running = False
        self.speaker.stop()
        if self.speaker.is_speaking():
            time.sleep(0.5)
        log.info("Goodbye! 👋")

    def check_for_wake_word(self):
        heard = self.recognizer.listen(2).lower()
        if heard:
            log.info(f"Heard: {heard}")
            if any(w in heard for w in WAKE_WORDS):
                self.state["awake"] = True
                log.info("🎯 Wake word detected!")
                self.speaker.say("Hello! I am HMND-01, your humanoid robot assistant. How can I help you today?")
                return True
        return False

    def speak_with_interrupt(self, text):
        self._should_stop = False
        
        def speak():
            if not self._should_stop:
                self.speaker.say(text)

        speech_thread = threading.Thread(target=speak)
        speech_thread.start()

        while speech_thread.is_alive() and not self._should_stop:
            heard = self.recognizer.listen(1).lower()
            if heard:
                #log.info(f"Heard while speaking: {heard}")
                if any(w in heard for w in STOP_WORDS):
                    log.info("🛑 Stop word detected!")
                    self._should_stop = True
                    self.speaker.stop()
                    speech_thread.join(timeout=1.0)
                    return True

        return self._should_stop

    def handle_history_query(self, user_input: str) -> str | None:
        if "first interaction" in user_input.lower() or "first conversation" in user_input.lower():
            all_interactions = self.conversation_history.get_all_interactions()
            if all_interactions:
                first_interaction = all_interactions[0]
                return f"Our first interaction was at {first_interaction['timestamp']}. You said: '{first_interaction['user']}' and I responded: '{first_interaction['robot']}'"
            return "I don't have any previous interactions to recall."

        if "last interaction" in user_input.lower() or "previous conversation" in user_input.lower():
            last_interaction = self.conversation_history.get_last_interaction()
            if last_interaction:
                return f"Our last interaction was at {last_interaction['timestamp']}. You said: '{last_interaction['user']}' and I responded: '{last_interaction['robot']}'"
            return "I don't have any previous interactions to recall."

        if "recent interactions" in user_input.lower() or "recent conversations" in user_input.lower():
            recent = self.conversation_history.get_recent_interactions(3)
            if recent:
                response = "Here are our recent interactions:\n"
                for interaction in recent:
                    response += f"\nAt {interaction['timestamp']}:\nYou: '{interaction['user']}'\nMe: '{interaction['robot']}'\n"
                return response
            return "I don't have any recent interactions to recall."

        if "repeat" in user_input.lower() and self.last_response:
            return f"I'll repeat my last response: {self.last_response}"

        return None

    def run(self):
        while self.running:
            if not self.state["awake"]:
                if self.check_for_wake_word():
                    continue
                time.sleep(0.1)
                continue

            log.info("🎤 Listening for your question...")
            user = self.recognizer.listen(10)
            
            if not user:
                log.info("❌ No speech detected") 
                self.speak_with_interrupt("I didn't catch that. Did you say something?")
                continue

            log.info(f"👤 You said: {user}")
            
            if any(w in user.lower() for w in STOP_WORDS):
                log.info("🛑 Stop word detected in user input")
                self.speaker.stop()
                self._should_stop = True
                continue

            if any(q in user.lower() for q in QUIT_WORDS):
                log.info("👋 Quit word detected")
                self.speak_with_interrupt("Goodbye! Say 'Hey robot' when you need me again.")
                self.state["awake"] = False
                log.info("Waiting for wake word...")
                continue

            # Check for history-related queries
            history_response = self.handle_history_query(user)
            if history_response:
                log.info("📜 History query detected")
                self.speak_with_interrupt(history_response)
                continue

            # Get response from brain
            log.info("🤔 Thinking...")
            answer = self.brain.reply(user)
            self.last_response = answer
            log.info(f"🤖 Response: {answer}")

            # Store the interaction in history
            self.conversation_history.add_interaction(user, answer)

            # Speak the response
            was_interrupted = self.speak_with_interrupt(answer)
            if was_interrupted:
                log.info("Ready for new input...")
                self._should_stop = True

def signal_handler(signum, frame):
    log.info("\nReceived shutdown signal")
    if 'robot' in globals():
        robot.shutdown()
    sys.exit(0)

# ─────────── entry point ───────────
def main():
    parser = argparse.ArgumentParser(description="Rainbow Robot (fully local with history)")
    parser.add_argument("--voice",      default="en_GB-alba-low", help="Piper voice ID / path")
    parser.add_argument("--stt_model",  default="base.en",        help="Whisper model (tiny.en, base.en, ...)")
    parser.add_argument("--llm",        default="llama3",         help="Ollama model name")
    parser.add_argument("--device",     default="cpu",           help="cuda | cpu")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    recognizer = SpeechRecognizer(args.stt_model, args.device)
    speaker = Speaker(voice_name=None)
    brain = LocalChat(args.llm)

    global robot
    robot = RainbowRobot(recognizer, speaker, brain)
    robot.run()

if __name__ == "__main__":
    import os
    import shutil
    main() 
