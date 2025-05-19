import openai
import os
import speech_recognition as sr
import tempfile
import pygame
import time
import threading
from openai import OpenAI
import argparse
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import json
from collections import deque

# Configure logging
def setup_logger():
    logger = logging.getLogger('RainbowRobot')
    logger.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        'rainbow_robot.log',
        maxBytes=1024*1024,  # 1MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logger()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1"
)

# Initialize pygame mixer
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)

# Global flags
interrupted = False
is_active = False

# Conversation history
class ConversationHistory:
    def __init__(self, max_history=10):
        self.history = deque(maxlen=max_history)
        self.history_file = "conversation_history.json"
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
            logger.error(f"Error saving conversation history: {e}")

    def load_history(self):
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.history = deque(json.load(f), maxlen=self.history.maxlen)
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")

# Initialize conversation history
conversation_history = ConversationHistory()

def listen_for_wake_word():
    """Background thread to listen for wake word"""
    global is_active
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 100
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.5
    
    logger.info("Robot is sleeping. Say 'Hey robot', 'Hey robo', or 'Wake up' to activate me!")
    
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        while True:
            try:
                logger.info("Listening for wake word...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
                text = recognizer.recognize_google(audio, language="en-US").lower()
                logger.debug(f"Heard: {text}")
                
                if "hey robot" in text or "hey robo" in text or "wake up" in text:
                    is_active = True
                    logger.info("Wake word detected! Robot is now active.")
                    speak("Hello! I am HMND-01, your humanoid robot assistant. How can I help you today?")
                    break
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except Exception as e:
                logger.error(f"Error in wake word detection: {e}")
                continue

def check_for_interruption():
    """Background thread to check for interruption"""
    global interrupted
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 200
    
    with sr.Microphone() as source:
        while True:
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=1)
                text = recognizer.recognize_google(audio, language="en-US").lower()
                if "rainbow" in text or "stop" in text:
                    interrupted = True
                    pygame.mixer.music.stop()
                    logger.info(f"Interrupted by user saying '{text}'")
                    break
            except:
                continue

def speak(text):
    """Convert text to speech using OpenAI's text-to-speech API"""
    global interrupted
    interrupted = False
    
    interruption_thread = threading.Thread(target=check_for_interruption)
    interruption_thread.daemon = True
    interruption_thread.start()
    
    logger.info(f"Robot: {text}")
    
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        
        output_file = "temp_speech.mp3"
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy() and not interrupted:
            time.sleep(0.1)
            
        pygame.mixer.music.unload()
        os.remove(output_file)
        
        if interrupted:
            logger.info("Speech interrupted!")
            return True
            
    except Exception as e:
        logger.error(f"Error in speech synthesis: {e}")
        return False
    
    return False

def get_speech_input(timeout=20, phrase_time_limit=15):
    """Get speech input from the user"""
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 200
    recognizer.pause_threshold = 1.0
    recognizer.non_speaking_duration = 0.5
    recognizer.phrase_threshold = 0.3
    
    with sr.Microphone() as source:
        logger.info(f"Please speak now... (I'll wait for {timeout} seconds)")
        try:
            logger.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            
            logger.info("Listening...")
            audio = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_time_limit
            )
            
            text = recognizer.recognize_google(
                audio,
                language="en-US",
                show_all=False
            )
            
            if text:
                logger.info(f"You said: {text}")
                return text
            else:
                logger.warning("No speech detected")
                return None
                
        except sr.WaitTimeoutError:
            logger.warning(f"No speech detected for {timeout} seconds.")
            return None
        except sr.UnknownValueError:
            logger.warning("Sorry, I could not understand your speech")
            return None
        except sr.RequestError as e:
            logger.error(f"Could not request results; {e}")
            return None
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return None

def get_response(instruction: str) -> str:
    """Get a response from OpenAI's model"""
    try:
        # Check for history-related queries
        if "first interaction" in instruction.lower() or "first conversation" in instruction.lower():
            all_interactions = conversation_history.get_all_interactions()
            if all_interactions:
                first_interaction = all_interactions[0]
                return f"Our first interaction was at {first_interaction['timestamp']}. You said: '{first_interaction['user']}' and I responded: '{first_interaction['robot']}'"
            return "I don't have any previous interactions to recall."

        if "last interaction" in instruction.lower() or "previous conversation" in instruction.lower():
            last_interaction = conversation_history.get_last_interaction()
            if last_interaction:
                return f"Our last interaction was at {last_interaction['timestamp']}. You said: '{last_interaction['user']}' and I responded: '{last_interaction['robot']}'"
            return "I don't have any previous interactions to recall."

        if "recent interactions" in instruction.lower() or "recent conversations" in instruction.lower():
            recent = conversation_history.get_recent_interactions(3)
            if recent:
                response = "Here are our recent interactions:\n"
                for interaction in recent:
                    response += f"\nAt {interaction['timestamp']}:\nYou: '{interaction['user']}'\nMe: '{interaction['robot']}'\n"
                return response
            return "I don't have any recent interactions to recall."

        # Check for repeat request
        if any(phrase in instruction.lower() for phrase in ["repeat", "say that again", "repeat that", "repeat this"]):
            last_interaction = conversation_history.get_last_interaction()
            if last_interaction:
                return f"I'll repeat my last response: {last_interaction['robot']}"
            return "I don't have anything to repeat yet."

        # Regular response generation
        ROBOTICS_KNOWLEDGE = """
        [Previous robotics knowledge content...]
        """

        ROBOT_IDENTITY = """
        [Previous robot identity content...]
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"""You are a humanoid robot assistant with extensive knowledge about robotics. Here is your knowledge base and identity:

{ROBOTICS_KNOWLEDGE}

{ROBOT_IDENTITY}

Please provide brief and concise responses (2-3 sentences maximum) that can be spoken naturally. Your answers must be in a friendly and engaging tone, and you should use your robotics knowledge when relevant to the conversation. Always maintain your identity as a Humanoid HMND series robot when appropriate."""},
                {"role": "user", "content": instruction}
            ],
            temperature=0.5,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error getting response from OpenAI: {e}")
        return "I apologize, but I'm having trouble generating a response right now."

def main():
    global is_active
    
    parser = argparse.ArgumentParser(description="Voice Assistant with Conversation History")
    parser.add_argument("--message", type=str, help="Initial message to speak")
    args = parser.parse_args()
    
    logger.info("Starting Rainbow Robot Assistant with Conversation History...")
    
    if args.message:
        speak(args.message)
        return
        
    logger.info("Say 'Hey robot' to wake me up.")
    logger.info("Say 'Rainbow' or 'Stop' to interrupt my speech.")
    logger.info("Say 'goodbye' to end the conversation, or stay silent for 10 seconds.")
    logger.info("You can ask about our previous interactions by saying 'what was our last interaction' or 'tell me about our recent conversations'")
    
    wake_thread = threading.Thread(target=listen_for_wake_word)
    wake_thread.daemon = True
    wake_thread.start()
    
    silence_count = 0
    max_silence = 3

    while True:
        if not is_active:
            time.sleep(0.1)
            continue
            
        logger.info("Please speak your question:")
        speech_text = get_speech_input()
        
        if not speech_text:
            silence_count += 1
            if silence_count >= max_silence:
                speak("No speech detected for too long. Going back to sleep. Say 'Hey robot' to wake me up!")
                is_active = False
                wake_thread = threading.Thread(target=listen_for_wake_word)
                wake_thread.daemon = True
                wake_thread.start()
                continue
            speak(f"I didn't catch that. Please try again. {max_silence - silence_count} attempts remaining")
            continue
            
        silence_count = 0
            
        if "goodbye" in speech_text.lower():
            speak("Goodbye! Have a great day! Say 'Hey robot' when you need me again!")
            is_active = False
            wake_thread = threading.Thread(target=listen_for_wake_word)
            wake_thread.daemon = True
            wake_thread.start()
            continue
            
        response = get_response(speech_text)
        was_interrupted = speak(response)
        
        # Store the interaction in history
        conversation_history.add_interaction(speech_text, response)
        
        if was_interrupted:
            logger.info("Would you like to ask something else?")

if __name__ == "__main__":
    main() 
