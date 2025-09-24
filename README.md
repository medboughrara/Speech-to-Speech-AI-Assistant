# Speech-to-Speech AI Assistant

A voice-interactive AI assistant that uses state-of-the-art models for speech recognition, natural language processing, and speech synthesis.

## Features

- Wake word detection ("hi")
- Speech-to-text using Whisper model
- Natural language processing using Gemma 4B model
- Text-to-speech using Glow-TTS
- GPU acceleration support
- Error handling and recovery
- Continuous interaction loop

## Requirements

- Python 3.10 or later
- NVIDIA GPU with CUDA support (optional but recommended)
- Microphone for audio input
- Speakers for audio output
- Ollama for running the Gemma model

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd sts
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install sounddevice numpy requests faster-whisper TTS torch
```

4. Install and start Ollama:
- Follow instructions at https://ollama.ai to install Ollama
- Pull the Gemma model:
```bash
ollama pull gemma3:4b
```

## Usage

1. Start the Ollama server:
```bash
ollama serve
```

2. Run the assistant:
```bash
python s2s.py
```

3. Say "hi" to activate the assistant
4. After the wake word is detected, speak your question or command
5. The assistant will process your speech, generate a response, and speak it back to you

## Models Used

- **Speech Recognition**: Whisper (base model)
- **Language Processing**: Gemma 3 4B
- **Text-to-Speech**: Glow-TTS with LJSpeech voice

## Configuration

The system is configured for optimal performance with:
- 20-second input window
- High-accuracy speech recognition
- Detailed AI responses
- Natural-sounding speech synthesis

## Error Handling

The system includes robust error handling for:
- Audio device issues
- Model loading failures
- Network connectivity problems
- Speech recognition errors
- GPU/CPU fallback options

## License

[Your chosen license]

## Acknowledgments

- OpenAI for the Whisper model
- Google for the Gemma model
- Ollama team for the model serving infrastructure
- Coqui-ai for the TTS implementation