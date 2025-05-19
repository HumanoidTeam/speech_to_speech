#!/usr/bin/env python3
"""
This script is 100 % offline speech-to-speech assistant, that relies on whisper for stt, llama3 as LLM, and pyttsx3 as tts.
"""

# ─────────── configuration ───────────
WAKE_WORDS   = ("hey robot", "hey robo", "wake up")
STOP_WORDS   = ("rainbow", "stop", "shut up", "wait")
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

Your technical specifications:
- Height: 175 cm (5'9")
- Weight: 70 kg (154 lbs)
- Payload capacity: 15 kg (33 lbs)
- Walking speed: 1.5 m/s (5.4 km/h)
- Average run time: 4 hours
- Degrees of freedom: 41
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

# ─────────── utility: logger ───────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s │ %(message)s",

    
    datefmt="%H:%M:%S"
)
log = logging.getLogger("RainbowRobot")


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


# ---------- TTS: pyttsx3 (no external voice download) ----------

class Speaker:
    """
    Simple wrapper around pyttsx3 for text-to-speech.
    """
    def __init__(self, voice_name: str | None = None, rate: int = 150):
        self.voice_name = voice_name
        self.rate = rate
        self._speaking = False
        self._engine = None
        self._init_engine()

    def _init_engine(self):
        """Initialize the TTS engine if not already initialized."""
        if self._engine is None:
            self._engine = pyttsx3.init()
            if self.voice_name:
                # Try to select a specific voice (optional)
                for v in self._engine.getProperty("voices"):
                    if self.voice_name.lower() in v.name.lower():
                        self._engine.setProperty("voice", v.id)
                        break
            self._engine.setProperty("rate", self.rate)

    def _cleanup_engine(self):
        """Clean up the engine instance."""
        if self._engine is not None:
            try:
                self._engine.stop()
            except:
                pass
            self._engine = None

    def say(self, text: str):
        """Speak the text."""
        self._speaking = True
        try:
            self._init_engine()
            self._engine.say(text)
            self._engine.runAndWait()
        finally:
            self._speaking = False
            self._cleanup_engine()

    def stop(self):
        """Stop speaking."""
        if self._speaking:
            self._cleanup_engine()
            self._speaking = False

    def is_speaking(self):
        """Check if the engine is currently speaking."""
        return self._speaking

    def __del__(self):
        """Cleanup when the object is destroyed."""
        self._cleanup_engine()



# ─────────── LLM: Ollama ───────────
class LocalChat:
    def __init__(self, model_id: str):
        log.info(f'Loading Ollama model "{model_id}" ...')
        self.model = model_id
        self.client = Client()
        # Test connection
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
        self.speaker    = speaker
        self.brain      = brain
        self.state      = {"awake": False}
        self.running    = True
        self.last_response = None
        self._should_stop = False
        log.info("🤖 Rainbow Robot initialized!")
        log.info(f"Wake words: {', '.join(WAKE_WORDS)}")
        log.info(f"Stop words: {', '.join(STOP_WORDS)}")
        log.info(f"Quit words: {', '.join(QUIT_WORDS)}")
        log.info("Waiting for wake word...")
        log.info("Press Ctrl+C to exit")

    def shutdown(self):
        """Gracefully shutdown the robot."""
        log.info("Shutting down...")
        self.running = False
        self.speaker.stop()
        if self.speaker.is_speaking():
            time.sleep(0.5)  # Give time for speech to stop
        log.info("Goodbye! 👋")

    def check_for_wake_word(self):
        """Check for wake word and respond if found."""
        heard = self.recognizer.listen(2).lower()
        if heard:
            log.info(f"Heard: {heard}")
            if any(w in heard for w in WAKE_WORDS):
                self.state["awake"] = True
                log.info("🎯 Wake word detected!")
                self.speaker.say("Hello! How can I help you today?")
                return True
        return False

    def speak_with_interrupt(self, text):
        """Speak text and listen for interruptions."""
        self._should_stop = False
        
        # Start speaking in a separate thread
        def speak():
            if not self._should_stop:
                self.speaker.say(text)

        speech_thread = threading.Thread(target=speak)
        speech_thread.start()

        # Listen for interruptions while speaking
        while speech_thread.is_alive() and not self._should_stop:
            heard = self.recognizer.listen(1).lower()
            if heard:
                log.info(f"Heard while speaking: {heard}")
                if any(w in heard for w in STOP_WORDS):
                    log.info("🛑 Stop word detected!")
                    self._should_stop = True
                    self.speaker.stop()
                    speech_thread.join(timeout=1.0)
                    return True

        return self._should_stop

    def run(self):
        while self.running:
            # Wait for wake word if not awake
            if not self.state["awake"]:
                if self.check_for_wake_word():
                    continue
                time.sleep(0.1)
                continue

            # Listen for user input
            log.info("🎤 Listening for your question...")
            user = self.recognizer.listen(10)
            
            if not user:
                log.info("❌ No speech detected") 
                self.speak_with_interrupt("I didn't catch that. Did you say something?")
                continue

            log.info(f"👤 You said: {user}")
            
            # Check for stop words first
            if any(w in user.lower() for w in STOP_WORDS):
                log.info("🛑 Stop word detected in user input")
                self.speaker.stop()
                self._should_stop = True
                continue # a quick fix to stop the robot from responding to stop words.

            # Check for quit word
            if any(q in user.lower() for q in QUIT_WORDS):
                log.info("👋 Quit word detected")
                self.speak_with_interrupt("Goodbye! Say 'Hey robot' when you need me again.")
                self.state["awake"] = False
                log.info("Waiting for wake word...")
                continue

            # Check if user wants to repeat the last response
            if "repeat" in user.lower() and self.last_response:
                log.info("🔄 Repeating last response")
                self.speak_with_interrupt(self.last_response)
                continue

            # Get response from brain
            log.info("🤔 Thinking...")
            answer = self.brain.reply(user)
            self.last_response = answer
            log.info(f"🤖 Response: {answer}")

            # Speak the response
            was_interrupted = self.speak_with_interrupt(answer)
            if was_interrupted:
                log.info("Ready for new input...")
                self._should_stop = True

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    log.info("\nReceived shutdown signal")
    if 'robot' in globals():
        robot.shutdown()
    sys.exit(0)

# ─────────── entry point ───────────
def main():
    parser = argparse.ArgumentParser(description="Rainbow Robot (fully local)")
    parser.add_argument("--voice",      default="en_GB-alba-low", help="Piper voice ID / path")
    parser.add_argument("--stt_model",  default="base.en",        help="Whisper model (tiny.en, base.en, ...)")
    parser.add_argument("--llm",        default="llama3",         help="Ollama model name")
    parser.add_argument("--device",     default="cpu",           help="cuda | cpu")
    args = parser.parse_args()

    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    recognizer = SpeechRecognizer(args.stt_model, args.device)
    speaker = Speaker(voice_name=None)     # pyttsx3 version; pass a name if you like
    brain = LocalChat(args.llm)

    global robot
    robot = RainbowRobot(recognizer, speaker, brain)
    robot.run()

if __name__ == "__main__":
    import shutil   # imported late so pylint doesn't complain
    main()
