# Speech-to-Speech Assistant

A voice-controlled AI assistant with both cloud and local implementations.

## Environment Setup

1. Create a `.env` file in the project root:

2. Add your OpenAI API key to the `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Cloud Version with History
```bash
python sts_cloud_history.py
```

### Local Version with History
```bash
python sts_local_history.py
```

## Features

- Wake word detection ("Hey robot", "Hey robo", "Wake up")
- Speech recognition and synthesis
- Conversation history
- Interruption handling
- Both cloud (OpenAI) and local (Ollama) implementations

## History Features

Both `sts_cloud_history.py` and `sts_local_history.py` include conversation history functionality:

- Stores the last 10 interactions in memory
- Persists conversations to JSON files:
  - Cloud version: `conversation_history.json`
  - Local version: `conversation_history_local.json`
- Timestamps each interaction
- Allows querying past conversations

You can ask the robot about previous conversations:
- "What was our first interaction?"
- "What was our last interaction?"
- "Tell me about our recent conversations"
- "Repeat that" or "Say that again"

## Security Note

Never commit your `.env` file or expose your API keys. The `.env` file is included in `.gitignore` to prevent accidental commits.

## Common Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Microphone and speakers

## Local Implementation with History (`sts_local_history.py`)

### Prerequisites
- Ollama installed (for local LLM support)
- CUDA support (optional, for faster processing)

### Installation

1. Install Ollama:
   - Download from: https://ollama.ai/download
   - Run `ollama serve` to start the server
   - Pull required models: `ollama pull llama3`

2. Install Python dependencies:
```bash
pip install -r requirements_local.txt
```

Required packages:
- torch
- transformers (>=4.40)
- faster-whisper (>=1.0)
- sounddevice
- numpy
- pyttsx3
- pypiwin32 (Windows only)
- accelerate
- llama-cpp-python

### Running the Local Version with History

```bash
python sts_local_history.py [options]
```

Command-line options:
- `--voice`: Voice ID (default: "en_GB-alba-low")
- `--stt_model`: Whisper model (default: "base.en")
- `--llm`: Ollama model name (default: "llama3")
- `--device`: Processing device (default: "cpu", options: "cuda" | "cpu")

### Voice Commands
Wake Words:
- "hey robot"
- "hey robo"
- "wake up"

Stop Words:
- "rainbow"
- "stop"
- "shut up"
- "wait"

Quit Words:
- "goodbye"

### Features
- Completely offline operation
- Lower latency
- No API costs
- Privacy-focused
- Local speech recognition
- Local text-to-speech
- Conversation history storage and retrieval

## Cloud Implementation with History (`sts_cloud_history.py`)

### Prerequisites
- OpenAI API key
- Internet connection

### Installation

1. Set up OpenAI API:
   - Get an API key from OpenAI
   - Add your API key to the environment variables or configuration

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- openai (>=1.0.0)
- pygame (>=2.6.1)
- SpeechRecognition (>=3.10.0)
- python-dotenv (==1.0.1)
- httpx (==0.26.0)

### Running the Cloud Version with History

```bash
python sts_cloud_history.py [options]
```

Command-line options:
- `--message`: Initial message to speak (optional)

### Voice Commands
Wake Words:
- "hey robot"
- "hey robo"
- "wake up"

Stop Words:
- "rainbow"
- "stop"

### Features
- Higher quality speech synthesis
- More advanced language model
- Better speech recognition
- Regular updates and improvements
- Cloud-based processing
- OpenAI's latest models
- Conversation history storage and retrieval

## Troubleshooting

### Local Version Issues
1. Ollama Connection Problems:
   - Ensure Ollama is installed and running (`ollama serve`)
   - Check if required models are downloaded (`ollama list`)
   - Verify system requirements are met

2. Speech Recognition Issues:
   - Check microphone permissions
   - Adjust ambient noise levels
   - Verify system audio settings

3. Python Package Issues:
   - Ensure all dependencies are installed
   - Check for CUDA compatibility if using GPU
   - Verify Python version compatibility

### Cloud Version Issues
1. API Connection Problems:
   - Verify OpenAI API key
   - Check internet connection
   - Ensure API quota is available

2. Speech Recognition Issues:
   - Check microphone permissions
   - Verify internet connection
   - Ensure proper audio input settings

3. Package Issues:
   - Ensure all dependencies are installed
   - Check for version conflicts
   - Verify Python version compatibility

## Logging

Both versions log their operation to `rainbow_robot.log`. The log includes:
- Wake word detection
- Speech recognition results
- System responses
- Error messages
- Conversation history operations

## Support

For issues and support:
1. Check the logs in `rainbow_robot.log`
2. Review the documentation
3. Open an issue in the repository
