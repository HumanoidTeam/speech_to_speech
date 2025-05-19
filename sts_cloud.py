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
    api_key="",
    base_url="https://api.openai.com/v1"
)

# Initialize pygame mixer
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)

# Global flags
interrupted = False
is_active = False

def listen_for_wake_word():
    """Background thread to listen for wake word"""
    global is_active
    recognizer = sr.Recognizer()
    # More sensitive settings for wake word detection
    recognizer.energy_threshold = 100  # Lower threshold to catch quieter speech
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.5   # Shorter pause threshold
    
    logger.info("Robot is sleeping. Say 'Hey robot', 'Hey robo', or 'Wake up' to activate me!")
    
    with sr.Microphone() as source:
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        while True:
            try:
                logger.info("Listening for wake word...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
                text = recognizer.recognize_google(audio, language="en-US").lower()
                logger.debug(f"Heard: {text}")  # Debug level for detailed logging
                
                if "hey robot" in text or "hey robo" in text or "wake up" in text:  # Added "wake up" as trigger
                    is_active = True
                    logger.info("Wake word detected! Robot is now active.")
                    speak("Hello! I am Rainbow, your humanoid robot assistant. How can I help you today?")
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
                if "rainbow" in text or "stop" in text:  # Accept both "Rainbow" and "Stop"
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
    
    # Start interruption listener in background
    interruption_thread = threading.Thread(target=check_for_interruption)
    interruption_thread.daemon = True
    interruption_thread.start()
    
    logger.info(f"Robot: {text}")
    
    try:
        # Generate speech
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        
        # Save to a file in the current directory
        output_file = "temp_speech.mp3"
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        # Play the audio
        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()
        
        # Wait for audio to finish or interruption
        while pygame.mixer.music.get_busy() and not interrupted:
            time.sleep(0.1)
            
        # Clean up
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
    # More robust recognition settings
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 200  # Even lower threshold for better sensitivity
    recognizer.pause_threshold = 1.0   # Increased pause threshold for better phrase detection
    recognizer.non_speaking_duration = 0.5  # Shorter non-speaking duration
    recognizer.phrase_threshold = 0.3  # Lower phrase threshold for better phrase detection
    
    with sr.Microphone() as source:
        logger.info(f"Please speak now... (I'll wait for {timeout} seconds)")
        try:
            # Longer ambient noise adjustment
            logger.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            
            # Listen with more generous timeouts
            logger.info("Listening...")
            audio = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_time_limit
            )
            
            # Use Google's speech recognition with more detailed settings
            text = recognizer.recognize_google(
                audio,
                language="en-US",
                show_all=False  # Get the most likely result
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

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"""You are a humanoid robot assistant with extensive knowledge about robotics. Here is your knowledge base and identity:

{ROBOTICS_KNOWLEDGE}

{ROBOT_IDENTITY}

Please provide brief and concise responses (2-3 sentences maximum) that can be spoken naturally. Your answers must be in a friendly and engaging tone, and you should use your robotics knowledge when relevant to the conversation. Always maintain your identity as a Humanoid HMND series robot when appropriate."""},
                {"role": "user", "content": instruction}
            ],
            temperature=0.5,  # Lower temperature for faster, more focused responses
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error getting response from OpenAI: {e}")
        return "I apologize, but I'm having trouble generating a response right now."

def main():
    global is_active
    
    parser = argparse.ArgumentParser(description="Voice Assistant")
    parser.add_argument("--message", type=str, help="Initial message to speak")
    args = parser.parse_args()
    
    logger.info("Starting Rainbow Robot Assistant...")
    
    if args.message:
        # If a message is provided, speak it and exit
        speak(args.message)
        return
        
    logger.info("Say 'Hey robot' to wake me up.")
    logger.info("Say 'Rainbow' or 'Stop' to interrupt my speech.")
    logger.info("Say 'goodbye' to end the conversation, or stay silent for 10 seconds.")
    
    # Start wake word listener
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
                # Restart wake word listener
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
            # Restart wake word listener
            wake_thread = threading.Thread(target=listen_for_wake_word)
            wake_thread.daemon = True
            wake_thread.start()
            continue
            
        response = get_response(speech_text)
        was_interrupted = speak(response)
        
        if was_interrupted:
            logger.info("Would you like to ask something else?")

if __name__ == "__main__":
    main() 