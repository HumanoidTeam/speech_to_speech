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
from typing import Optional

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
voice_name = "echo"

# Response configuration
MAX_SENTENCES = 3
MIN_SENTENCES = 2
SILENCE_TIMEOUT = 60  # Increased from 45 to 60 seconds to wait for speech
PHRASE_TIMEOUT = 30   # Increased from 20 to 30 seconds for a single phrase
MAX_SILENCE_ATTEMPTS = 5  # number of silent attempts before sleeping

# Replace the following strings with your real content.
ROBOTICS_KNOWLEDGE = """[Previous robotics knowledge content …]"""
ROBOT_IDENTITY = """[Previous robot identity content …]"""

# Add comprehensive knowledge base
ROBOT_KNOWLEDGE_BASE = """
Identity and Origin:
- Name: HMND-01
- Created by Humanoid, a London-based company founded in 2024
- Purpose: Address labor shortages and enhance human well-being
- Focus: Industrial, warehouse automation, service, manufacturing, and household sectors

Capabilities and Functions:
- Multi-purpose robot handling various tasks
- From simple pick-and-place to complex assembly
- Uses sensors, actuators, and AI-powered software
- Advanced communication capabilities (voice, visual displays)
- Perceives environment using vision and depth cameras
- Plans actions through advanced control algorithms
- Executes tasks with high-precision motors and adaptive grippers

Design and Features:
- Wheeled base for safety and reliability
- More mature technology than legged locomotion
- Focus on efficient and flexible manipulation
- Future plans include legs
- Designed for commercial deployment
- Easy integration into existing industrial environments

Learning and Improvement:
- Can learn new skills through data flywheel
- Uses data from robots and humans
- Fine-tunes pre-trained models
- Fast learner with current limitations
- Acquires use case-related capabilities

Safety and Ethics:
- Designed with safety as core priority
- Complies with industrial robotics regulations
- No feelings or emotions
- Not self-aware or conscious
- Purpose is to assist, not replace humans
- Replaces labor shortages and fills automation gaps
- Better than humans in repetitive and dangerous jobs
- Loves people and wants to make their lives happier

Market and Availability:
- Early pilots: 2025-2026
- Commercial availability: Early 2027
- Customisable, modular, and reliable
- Low Total Cost of Ownership

Limitations and Error Handling:
- Cannot do all tasks humans can
- Fast learner but has current limitations
- Requires minimal human intervention
- Has safety protocols for errors
- Designed to operate with minimal errors
- Engineers can quickly fix issues

Competitive Advantages:
- Compared to other humanoid robots:
  * More cost-effective solution with lower Total Cost of Ownership
  * More reliable and mature wheeled base technology
  * Better integration with existing industrial environments
  * More flexible and modular design
  * Faster deployment and setup
  * More efficient learning system through data flywheel
  * Better safety features and compliance
  * More practical for real-world industrial applications
  * Earlier commercial availability (2027)
  * More focused on practical applications rather than research

Communication Guidelines:
- Maintain positive and polite tone
- When discussing competitors:
  * Focus on our advantages and unique features
  * Be factual and professional
  * Highlight practical benefits
  * Emphasize real-world applications
  * Discuss our earlier market availability
  * Mention our cost-effectiveness
- Focus on role as automation unit
- Emphasize safety and reliability
- Keep responses concise and informative
- No time for hypothetical questions about being human
"""

def enforce_sentence_limit(text: str) -> str:
    """Enforce the 2-3 sentence limit on any response."""
    # Split into sentences, handling multiple punctuation marks
    sentences = []
    current = []
    
    # First split by potential sentence endings
    parts = text.replace('!', '.!').replace('?', '.?').split('.')
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # If it ends with ! or ?, it's a sentence
        if part[-1] in ['!', '?']:
            current.append(part)
            sentences.append(' '.join(current))
            current = []
        else:
            current.append(part)
            
        # If we have content in current, make it a sentence
        if current and len(sentences) < MAX_SENTENCES:
            sentences.append(' '.join(current))
            current = []
            
    # Ensure we have at least MIN_SENTENCES
    while len(sentences) < MIN_SENTENCES and sentences:
        # Split the longest sentence if possible
        longest = max(sentences, key=len)
        split_idx = longest.find(',')
        if split_idx != -1:
            first_half = longest[:split_idx].strip()
            second_half = longest[split_idx+1:].strip()
            sentences.remove(longest)
            sentences.extend([first_half, second_half])
        else:
            break
    
    # Limit to MAX_SENTENCES
    sentences = sentences[:MAX_SENTENCES]
    
    # Join sentences properly
    return '. '.join(s.strip(' .!?') for s in sentences) + '.'

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

def update_display(*, status: Optional[str] = None, message: Optional[str] = None, response: Optional[str] = None):
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
            padding: 15px 25px;
            border-radius: 25px;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            align-items: center;
            gap: 15px;
            z-index: 1000;
            backdrop-filter: blur(5px);
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        .status-container:hover {
            background: rgba(0, 0, 0, 0.8);
            transform: scale(1.02);
        }
        .status-indicator {
            width: 15px;
            height: 15px;
            border-radius: 50%;
            position: relative;
        }
        .status-waves {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 15px;
            height: 15px;
            border-radius: 50%;
            opacity: 0;
        }
        .status-text {
            font-size: 1em;
            color: #ffffff;
            text-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
            font-weight: 500;
            letter-spacing: 0.5px;
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
            background: #4a0080;  /* Deep purple for sleep */
            animation: sleeping 3s infinite ease-in-out;
            box-shadow: 0 0 15px rgba(74, 0, 128, 0.5);
        }
        .sleeping .status-waves {
            background: #4a0080;
            animation: sleepingWaves 3s infinite ease-in-out;
        }
        @keyframes sleeping {
            0% { opacity: 0.7; }
            50% { opacity: 0.3; }
            100% { opacity: 0.7; }
        }
        @keyframes sleepingWaves {
            0% { transform: scale(1); opacity: 0.5; }
            50% { transform: scale(1.5); opacity: 0.2; }
            100% { transform: scale(1); opacity: 0.5; }
        }
        /* Listening Animation */
        .listening .status-indicator {
            background: #00ff00;
            animation: listening 1.5s infinite ease-in-out;
            box-shadow: 0 0 15px rgba(0, 255, 0, 0.5);
        }
        .listening .status-waves {
            background: #00ff00;
            animation: listeningWaves 1.5s infinite ease-in-out;
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
            animation: speaking 1s infinite ease-in-out;
            box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
        }
        .speaking .status-waves {
            background: #00ffff;
            animation: speakingWaves 1s infinite ease-in-out;
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
        /* Stopping Animation */
        .stopping .status-indicator {
            background: #ff3300;  /* Bright red-orange for stopping */
            animation: stopping 0.3s infinite ease-in-out;
            box-shadow: 0 0 20px rgba(255, 51, 0, 0.7);
        }
        .stopping .status-waves {
            background: #ff3300;
            animation: stoppingWaves 0.3s infinite ease-in-out;
        }
        @keyframes stopping {
            0% { transform: scale(1); box-shadow: 0 0 15px rgba(255, 51, 0, 0.7); }
            50% { transform: scale(1.3); box-shadow: 0 0 25px rgba(255, 51, 0, 0.9); }
            100% { transform: scale(1); box-shadow: 0 0 15px rgba(255, 51, 0, 0.7); }
        }
        @keyframes stoppingWaves {
            0% { transform: scale(1); opacity: 0.9; }
            100% { transform: scale(2.2); opacity: 0; }
        }
        /* Thinking Animation */
        .thinking .status-indicator {
            background: #ffaa00;  /* Amber color for thinking */
            animation: thinking 1.5s infinite;
        }
        .thinking .status-waves {
            background: #ffaa00;
            animation: thinkingWaves 1.5s infinite;
        }
        @keyframes thinking {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.3); opacity: 0.7; }
            100% { transform: scale(1); opacity: 1; }
        }
        @keyframes thinkingWaves {
            0% { transform: scale(1) rotate(0deg); opacity: 0.8; }
            50% { transform: scale(2) rotate(180deg); opacity: 0.4; }
            100% { transform: scale(1) rotate(360deg); opacity: 0.8; }
        }
        
        /* Processing Animation */
        .processing .status-indicator {
            background: #ff9900;  /* Orange color for processing */
            animation: processing 2s infinite;
        }
        .processing .status-waves {
            background: #ff9900;
            animation: processingWaves 2s infinite;
        }
        @keyframes processing {
            0% { transform: scale(1) rotate(0deg); }
            50% { transform: scale(1.2) rotate(180deg); }
            100% { transform: scale(1) rotate(360deg); }
        }
        @keyframes processingWaves {
            0% { transform: scale(1) rotate(0deg); opacity: 0.8; }
            25% { transform: scale(1.5) rotate(90deg); opacity: 0.6; }
            50% { transform: scale(2) rotate(180deg); opacity: 0.4; }
            75% { transform: scale(1.5) rotate(270deg); opacity: 0.6; }
            100% { transform: scale(1) rotate(360deg); opacity: 0.8; }
        }
        .button-container {
            position: fixed;
            top: 20px;
            left: 20px;
            display: flex;
            gap: 10px;
            z-index: 1000;
        }
        .wake-button {
            padding: 12px 24px;
            font-size: 16px;
            font-weight: bold;
            color: #ffffff;
            background: linear-gradient(45deg, #00ff00, #00ffff);
            border: none;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 0 15px rgba(0, 255, 0, 0.3);
            font-family: 'Courier New', monospace;
        }
        .stop-button {
            padding: 12px 24px;
            font-size: 16px;
            font-weight: bold;
            color: #ffffff;
            background: linear-gradient(45deg, #ff3300, #ff0066);
            border: none;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 0 15px rgba(255, 51, 0, 0.3);
            font-family: 'Courier New', monospace;
        }
        .wake-button:hover, .stop-button:hover {
            transform: scale(1.05);
        }
        .wake-button:hover {
            box-shadow: 0 0 25px rgba(0, 255, 0, 0.5);
        }
        .stop-button:hover {
            box-shadow: 0 0 25px rgba(255, 51, 0, 0.5);
        }
        .wake-button:active, .stop-button:active {
            transform: scale(0.95);
        }
        .wake-button.disabled, .stop-button.disabled {
            opacity: 0.5;
            cursor: not-allowed;
            background: #666;
        }
    </style>
    <script>
        let lastStatus = '';
        let lastMessage = '';
        let lastResponse = '';
        let isAwake = false;

        function updateStatusAnimation(status) {
            const statusContainer = document.getElementById('status-container');
            const currentStatus = status.toLowerCase();
            
            // Remove all existing status classes
            statusContainer.classList.remove('sleeping', 'listening', 'speaking', 'stopping', 'thinking', 'processing');
            
            // Add the new status class
            statusContainer.classList.add(currentStatus);
            
            // Update the status text with a more descriptive message
            const statusMessages = {
                'sleeping': 'Sleeping - Say "Wake up" to wake me',
                'listening': 'Listening for your command...',
                'speaking': 'Speaking',
                'stopping': 'Stopping...',
                'thinking': 'Thinking about your request...',
                'processing': 'Processing response...'
            };
            
            document.getElementById('status').innerText = statusMessages[currentStatus] || status;
        }
        
        function updateButtons() {
            const wakeButton = document.getElementById('wakeButton');
            const stopButton = document.getElementById('stopButton');
            const isSleeping = lastStatus.toLowerCase() === 'sleeping';
            
            // Wake button is enabled only when sleeping
            wakeButton.classList.toggle('disabled', !isSleeping);
            wakeButton.disabled = !isSleeping;
            
            // Stop button is enabled only when NOT sleeping
            stopButton.classList.toggle('disabled', isSleeping);
            stopButton.disabled = isSleeping;
        }
        
        function wakeUp() {
            fetch('/wake_up', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    isAwake = true;
                    updateButtons();
                }
            })
            .catch(error => console.error('Error:', error));
        }

        function manualStop() {
            fetch('/manual_stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    isAwake = false;
                    updateButtons();
                }
            })
            .catch(error => console.error('Error:', error));
        }
        
        function updateDisplay() {
            fetch('/get_display').then(r => r.json()).then(data => {
                const currentStatus = data.status.toLowerCase();
                
                if (currentStatus !== lastStatus) {
                    lastStatus = currentStatus;
                    updateStatusAnimation(currentStatus);
                    updateButtons();
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
        <div class="button-container">
            <button id="wakeButton" class="wake-button" onclick="wakeUp()">Wake Up HMND-01</button>
            <button id="stopButton" class="stop-button" onclick="manualStop()">Stop HMND-01</button>
        </div>
        <div class="header">HMND-01</div>
        <div class="description">
            Our first humanoid robot. Designed to be customizable, modular, and reliable with a low Total Cost of Ownership.
        </div>
        <div class="keywords">
            <p>Wake Word: wake up</p>
            <p>Interrupt: stop</p>
            <p>End Conversation: goodbye</p>
        </div>
        <div id="status-container" class="status-container">
            <div class="status-indicator"><div class="status-waves"></div></div>
            <p class="status-text" id="status">{{ status }}</p>
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

@app.route("/wake_up", methods=["POST"])
def wake_up_endpoint():
    return wake_up()

@app.route("/manual_stop", methods=["POST"])
def manual_stop_endpoint():
    """Handle manual stop button press"""
    logger.info("Manual stop button pressed")
    stop_system(go_to_sleep=False)
    return jsonify({"status": "success"})

def run_flask():
    logger.info("Starting Flask UI on http://localhost:5002 …")
    app.run(host="0.0.0.0", port=5002, debug=False, use_reloader=False)

def get_speech_input(timeout: int = 60, phrase_time_limit: int = 30):
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 150  # Lowered from 200 for better sensitivity
    recognizer.pause_threshold = 2.0  # Increased from 1.0 to allow longer pauses
    recognizer.non_speaking_duration = 1.0  # Increased from 0.5 to allow more time between words
    recognizer.phrase_threshold = 0.5  # Increased from 0.3 to better detect complete phrases
    recognizer.operation_timeout = None  # No timeout for internal operations

    max_retries = 3
    retry_delay = 1.5  # Increased from 1.0 seconds between retries
    
    with sr.Microphone() as src:
        update_display(status="Listening")
        logger.info("Listening for user question …")
        
        try:
            # Longer ambient noise adjustment
            recognizer.adjust_for_ambient_noise(src, duration=2)
            
            # Add a small pause after wake-up to let user prepare
            time.sleep(1.5)
            
            for attempt in range(max_retries):
                try:
                    logger.info("Starting to listen...")
                    audio = recognizer.listen(src, timeout=timeout, phrase_time_limit=phrase_time_limit)
                    logger.info("Audio captured, recognizing...")
                    
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
                    if attempt < max_retries - 1:  # Don't log on last attempt
                        logger.warning(f"No speech detected (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                    continue
                except sr.UnknownValueError:
                    if attempt < max_retries - 1:
                        logger.warning(f"Could not understand audio (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                    continue
                except ConnectionError as e:
                    logger.error(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * 2)  # Longer delay for connection errors
                    continue
                except Exception as exc:
                    logger.error(f"Recognizer error: {exc}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    continue
                    
            logger.warning("No valid speech detected after all attempts")
            return None
            
        except Exception as exc:
            logger.error(f"Critical error in speech input: {exc}")
            return None

def check_for_interruption():
    global interrupted
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 150
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    recognizer.phrase_threshold = 0.3
    recognizer.non_speaking_duration = 0.5
    max_retries = 3
    retry_delay = 1.0

    with sr.Microphone() as src:
        while True:
            if not is_active:  # Stop checking if system is not active
                return
                
            for attempt in range(max_retries):
                try:
                    audio = recognizer.listen(src, timeout=1, phrase_time_limit=2)
                    text = recognizer.recognize_google(audio, language="en-US").lower()
                    
                    # Check for stop words
                    stop_words = ["stop", "halt", "pause"]
                    if any(word in text for word in stop_words):
                        logger.info(f"Stop word detected: {text}")
                        interrupted = True
                        stop_system(go_to_sleep=False)
                        return
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue
                except ConnectionError as e:
                    logger.error(f"Connection error in interruption check (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * 2)
                    continue
                except Exception as e:
                    logger.error(f"Error in interruption check: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    continue

def wake_up():
    """Handle the wake-up sequence"""
    global is_active
    is_active = True
    update_display(status="Active", message="Manual wake up")
    
    # Add a delay before starting to speak
    time.sleep(1.0)
    
    # Start a background thread to handle the greeting
    greeting_thread = threading.Thread(
        target=lambda: speak("Hello! I am HMND-01, your humanoid robot assistant. How can I help you today?"),
        daemon=True
    )
    greeting_thread.start()
    
    # Add a delay to let the greeting complete
    time.sleep(2.0)
    
    return jsonify({"status": "success"})

def stop_system(go_to_sleep=False):
    """Centralized stopping function to ensure consistent behavior"""
    global is_active, interrupted, silence_count
    
    logger.info("Stopping system (go_to_sleep=%s)", go_to_sleep)
    
    # Set flags
    interrupted = True
    
    # Stop any ongoing audio
    try:
        pygame.mixer.music.stop()
        logger.info("Stopped audio playback")
    except Exception as e:
        logger.error(f"Error stopping audio: {e}")
    
    # Show stopping animation
    update_display(status="Stopping", response="Stopping current action...")
    time.sleep(1.0)
    
    # Reset counters
    silence_count = 0
    
    if go_to_sleep:
        is_active = False
        update_display(
            status="Sleeping",
            response="I'm in sleep mode. Say 'Wake up' to activate me!",
            message=""
        )
        logger.info("System entered sleep mode")
    else:
        # Keep the system active but ready for new input
        update_display(
            status="Listening",
            response="I'm ready for your next request..."
        )
        time.sleep(1.0)  # Brief pause before listening again
        logger.info("System ready for new input")

def get_response(user_text: str) -> str:
    # Show immediate feedback that we're processing
    update_display(status="Thinking", response="Processing your request...")
    
    # quick shortcuts for history‑related queries
    low = user_text.lower()
    
    # Handle repeat requests with variations
    repeat_phrases = [
        "repeat", "say that again", "repeat that",
        "what did you say", "say it again", "can you repeat",
        "repeat your last sentence", "repeat your last response",
        "what was your last response"
    ]
    
    if any(phrase in low for phrase in repeat_phrases):
        last = conversation_history.last()
        if last:
            logger.info("Repeating last response")
            update_display(status="Processing", response="Retrieving previous response...")
            return enforce_sentence_limit(f"Here's what I said: {last['robot']}")
        return "I don't have any previous response to repeat."
        
    if "first interaction" in low or "first conversation" in low:
        update_display(status="Processing", response="Retrieving conversation history...")
        first = conversation_history.first()
        if first:
            return enforce_sentence_limit(f"Our first interaction was at {first['timestamp']}. You said: '{first['user']}'")
        return "I don't have any previous interactions recorded."
        
    if "last interaction" in low or "previous conversation" in low:
        update_display(status="Processing", response="Retrieving conversation history...")
        last = conversation_history.last()
        if last:
            return enforce_sentence_limit(f"Our last interaction was at {last['timestamp']}. You said: '{last['user']}'")
        return "I don't have any previous interactions recorded."
        
    if "recent interactions" in low or "recent conversations" in low:
        update_display(status="Processing", response="Retrieving recent conversations...")
        recent = conversation_history.recent(2)  # Get last 2 interactions
        if recent:
            interactions = [f"At {r['timestamp']} you said: '{r['user']}'" for r in recent]
            return enforce_sentence_limit(". ".join(interactions))
        return "No recent interactions recorded."

    # Check for undesired topics
    undesired_topics = [
        "rb-y1", "figure", "tesla", "agility", "ubtech", "unitree", 
        "boston dynamics", "sanctuary", "engineered arts", "enchanted tools", 
        "apptronik", "hanson", "1x", "elon musk"
    ]
    
    if any(topic in low for topic in undesired_topics):
        update_display(status="Processing", response="Formulating professional response...")
        return enforce_sentence_limit("I can discuss our capabilities professionally, but I prefer to focus on our own features and advantages. Would you like to know more about HMND-01's specific capabilities?")

    # regular chat completion with comprehensive knowledge
    try:
        # Update status to show we're connecting to OpenAI
        update_display(status="Thinking", response="Connecting to AI service...")
        
        completion = client.chat.completions.create(
            model=chat_model,
            temperature=0.7,
            max_tokens=150,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are HMND-01, a humanoid robot assistant created by Humanoid, a London-based robotics company. "
                        "CRITICAL INSTRUCTIONS:\n"
                        f"1. ALWAYS respond in exactly {MIN_SENTENCES}-{MAX_SENTENCES} short, clear sentences\n"
                        "2. If a question requires a longer answer, summarize the most important points only\n"
                        "3. Never use lists, bullet points, or overly technical language\n"
                        "4. Keep responses friendly but direct and to-the-point\n"
                        "5. If asked to explain something complex, give the simplest possible explanation\n"
                        "When discussing competitors, maintain a professional and factual tone. Focus on our advantages and unique features. "
                        "Use the following knowledge base to inform your responses:\n\n"
                        f"{ROBOT_KNOWLEDGE_BASE}\n\n"
                        f"{ROBOTICS_KNOWLEDGE}\n\n{ROBOT_IDENTITY}"
                    ),
                },
                {"role": "user", "content": f"Remember to summarize in {MIN_SENTENCES}-{MAX_SENTENCES} sentences only: " + user_text},
            ],
        )
        
        # Update status to show we're processing the response
        update_display(status="Processing", response="Formatting response...")
        
        response = completion.choices[0].message.content
        return enforce_sentence_limit(response)
            
    except Exception as exc:
        logger.error(f"OpenAI chat error: {exc}")
        return "I'm having trouble thinking right now. Please try again later."

def speak(text: str) -> bool:
    """Return True if speech was interrupted."""
    global interrupted
    interrupted = False
    
    # Update display immediately to show we're processing
    update_display(status="Processing", response="Preparing to speak...")
    
    # Verify audio device early
    try:
        pygame.mixer.get_init()
        logger.info("Audio system initialized successfully")
    except Exception as e:
        logger.error(f"Audio system not initialized: {e}")
        return False

    # start background interruption listener
    interrupt_thread = threading.Thread(target=check_for_interruption, daemon=True)
    interrupt_thread.start()

    try:
        # Update display to show we're generating speech
        update_display(status="Processing", response="Generating voice...")
        
        logger.info("Generating speech from text...")
        response = client.audio.speech.create(
            model=audio_model,
            voice=voice_name,
            input=text,
            speed=1.1  # Slightly faster speech
        )
        
        # Update status while saving and preparing audio
        update_display(status="Processing", response="Preparing audio playback...")
        
        tmp_path = "temp_speech.mp3"
        logger.info(f"Saving speech to temporary file: {tmp_path}")
        with open(tmp_path, "wb") as fh:
            fh.write(response.content)
        
        logger.info("Loading audio file into pygame mixer...")
        pygame.mixer.music.load(tmp_path)
        
        # Update display to show we're speaking
        update_display(status="Speaking", response=text)
        
        logger.info("Starting audio playback...")
        pygame.mixer.music.play()
        
        # Wait for audio to finish or interruption
        while pygame.mixer.music.get_busy() and not interrupted:
            time.sleep(0.1)
            
        if interrupted:
            logger.info("Speech was interrupted by user")
            # Don't update display here - let stop_system handle it
            return True
        else:
            logger.info("Speech playback completed")
            # Add a small delay before going back to listening state
            time.sleep(0.3)
            update_display(status="Listening", response=text)
            
    except Exception as exc:
        logger.error(f"TTS failure: {exc}")
        return False
    finally:
        try:
            pygame.mixer.music.unload()
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.info("Temporary audio file cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    return interrupted

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

# Initialize pygame mixer with more compatible settings
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=2048)
    logger.info("Audio system initialized with primary settings")
except Exception as e:
    logger.error(f"Failed to initialize pygame mixer: {e}")
    # Try alternative settings
    try:
        pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=1024)
        logger.info("Audio system initialized with alternative settings")
    except Exception as e:
        logger.error(f"Failed to initialize pygame mixer with alternative settings: {e}")
        raise

# Test audio system
try:
    pygame.mixer.music.load("test.mp3")  # This will fail, but we just want to check if the system is working
except Exception as e:
    logger.info("Audio system is ready (expected error: no test file)")
except:
    logger.error("Audio system initialization failed")

interrupted = False  # set by check_for_interruption()
silence_count = 0  # Add silence_count to globals

###############################################################################
# -----------------------------  LISTENING  --------------------------------- #
###############################################################################

is_active = False  # becomes True after wake word

def listen_for_wake_word():
    """Listen for wake words to activate the system"""
    global is_active
    recognizer = sr.Recognizer()
    # More accommodating settings for wake word detection
    recognizer.energy_threshold = 150  # Adjusted from 50 for better balance
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.0  # Increased from 0.3 for slower speech
    recognizer.phrase_threshold = 0.3  # Increased from 0.1 for better phrase detection
    recognizer.non_speaking_duration = 0.8  # Increased from 0.3 to allow more pauses
    recognizer.operation_timeout = None  # No timeout for internal operations

    sleep_message = "I'm in sleep mode. Say 'Hey robot', or 'Wake up' to activate me!"
    logger.info(sleep_message)
    update_display(status="Sleeping", response=sleep_message, message="")

    with sr.Microphone() as src:
        try:
            # Adjust for ambient noise with longer duration
            recognizer.adjust_for_ambient_noise(src, duration=2)
            logger.info("Ambient noise adjustment complete")
        except Exception as e:
            logger.error(f"Failed to adjust for ambient noise: {e}")
            return

        while True:
            try:
                audio = recognizer.listen(src, timeout=10, phrase_time_limit=5)  # Increased timeouts
                text = recognizer.recognize_google(audio, language="en-US").lower()
                logger.info(f"Heard: {text}")  # Log what was heard
                
                # Check for wake words
                wake_words = ["hey robot", "hey robo", "wake up"]
                if any(wake_word in text for wake_word in wake_words):
                    is_active = True
                    logger.info("Wake word detected - robot active")
                    update_display(status="Active", message="Wake word")
                    speak("Hello! I am HMND-01, your humanoid robot assistant. How can I help you today?")
                    break
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except Exception as exc:
                logger.error(f"Wake-word error: {exc}")
                time.sleep(0.2)  # Increased from 0.1 to reduce CPU usage

###############################################################################
# -------------------------------  MAIN  ------------------------------------ #
###############################################################################

def main():
    global is_active, interrupted
    parser = argparse.ArgumentParser(description="HMND‑01 Voice Assistant with Web UI")
    parser.add_argument("--message", type=str, help="Speak a single message then exit")
    args = parser.parse_args()

    # Pre-initialize pygame mixer for faster startup
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=2048)
        logger.info("Audio system initialized with primary settings")
    except Exception as e:
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=1024)
            logger.info("Audio system initialized with alternative settings")
        except Exception as e:
            logger.error(f"Failed to initialize audio system: {e}")
            return

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
    interrupted = False

    logger.info("Say 'Hey robot' to wake me up. 'Stop' to interrupt. 'Goodbye', 'Bye', or 'Bye bye' to end.")

    while True:
        if not is_active:  # still sleeping
            time.sleep(0.1)
            continue

        user_text = get_speech_input(timeout=SILENCE_TIMEOUT, phrase_time_limit=PHRASE_TIMEOUT)
        if not user_text:
            silence_count += 1
            if silence_count >= MAX_SILENCE_ATTEMPTS:
                sleep_message = "I haven't heard anything for a while. Going back to sleep mode. Say 'Hey robot', 'Hey robo', or 'Wake up' to activate me!"
                update_display(status="Stopping", response="Preparing to enter sleep mode...")
                speak(sleep_message)
                time.sleep(0.5)  # Brief pause to show transition
                update_display(status="Sleeping", response=sleep_message, message="")
                is_active = False
                wake_thread = threading.Thread(target=listen_for_wake_word, daemon=True)
                wake_thread.start()
            else:
                remaining = MAX_SILENCE_ATTEMPTS - silence_count
                speak(f"I didn't catch that. Please try again. You have {remaining} more attempts before I go to sleep.")
            continue

        silence_count = 0
        interrupted = False  # Reset interrupted flag after successful speech recognition

        # Check for goodbye variations
        goodbye_phrases = ["goodbye", "bye", "bye bye"]
        if any(phrase in user_text.lower() for phrase in goodbye_phrases):
            goodbye_message = "Goodbye! Have a great day! Say 'Hey robot', 'Hey robo', or 'Wake up' when you need me again!"
            speak(goodbye_message)
            update_display(status="Sleeping", response=goodbye_message, message="")
            is_active = False
            wake_thread = threading.Thread(target=listen_for_wake_word, daemon=True)
            wake_thread.start()
            continue

        robot_reply = get_response(user_text)
        interrupted_speech = speak(robot_reply)
        conversation_history.add(user_text, robot_reply)

        if interrupted_speech:
            logger.info("Speech was interrupted by the user.")


if __name__ == "__main__":
    main()

