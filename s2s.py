import sounddevice as sd
import numpy as np
import requests
import json
from faster_whisper import WhisperModel
from TTS.api import TTS
import torch

print("Initializing models...")

# Initialize Whisper for Speech-to-Text
print("Loading Whisper model...")
try:
    stt_model = WhisperModel("base", device="cuda", compute_type="float16")  # Start with base model for reliability
    print("Successfully loaded Whisper model")
except Exception as e:
    print(f"Error loading Whisper model: {str(e)}")
    print("Trying CPU fallback...")
    try:
        stt_model = WhisperModel("base", device="cpu", compute_type="float32")  # CPU fallback
        print("Successfully loaded Whisper model on CPU")
    except Exception as e:
        print(f"Fatal error loading Whisper model: {str(e)}")
        exit(1)

# Initialize TTS model with GPU support
print("Loading TTS model...")
tts_model = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=True)
if torch.cuda.is_available():
    print("Moving TTS model to GPU...")
    tts_model.to("cuda")

def initialize_audio():
    """Initialize audio device settings"""
    try:
        # List available audio devices
        devices = sd.query_devices()
        default_input = sd.query_devices(kind='input')
        print("Using input device:", default_input['name'])
        
        # Set default device settings
        sd.default.device = default_input['index'], None  # Input device only
        sd.default.channels = 1, None  # Mono input
        sd.default.dtype = 'float32', None
        sd.default.latency = 'low', None
        return True
    except Exception as e:
        print(f"❌ Error initializing audio: {str(e)}")
        return False

def record_audio(duration=5, samplerate=16000):
    """Record audio with error handling"""
    try:
        print("🎙️ Recording for", duration, "seconds...")
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
        sd.wait()
        if audio is None or len(audio) == 0:
            print("❌ No audio recorded")
            return None
        return np.squeeze(audio)
    except KeyboardInterrupt:
        print("\n⏹️ Recording stopped by user")
        return None
    except Exception as e:
        print(f"❌ Error recording audio: {str(e)}")
        return None

def detect_wake_word():
    """Continuously listen for the wake word 'hi'"""
    print("🎧 Listening for wake word 'hi'...")
    while True:
        try:
            # Record in 5-second chunks for better wake word detection
            audio = record_audio(5)
            if audio is None:
                print("Retrying wake word detection...")
                continue

            # Process the audio with error handling
            try:
                segments, _ = stt_model.transcribe(audio, beam_size=5)  # Increased beam size for better accuracy
                text = " ".join([seg.text for seg in segments]).lower()
                if "hi" in text:
                    print("✨ Wake word detected!")
                    return True
            except Exception as e:
                print(f"Error processing audio: {str(e)}")
                continue
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            return False
        except Exception as e:
            print(f"Error in wake word detection: {str(e)}")
            continue
        
def check_ollama_status():
    """Check if Ollama is running and gemma3 is available"""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code != 200:
            return False, "Ollama server is not responding properly"
        
        models = response.json().get('models', [])
        if not any(model['name'] == 'gemma3:4b' for model in models):
            return False, "gemma3:4b model is not found in Ollama. Please run 'ollama pull gemma3:4b'"
        
        return True, "Ollama is running and gemma3 is available"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama. Please make sure 'ollama serve' is running"
    except Exception as e:
        return False, f"Error checking Ollama status: {str(e)}"

def get_gemma_response(prompt):
    """Get response from local Gemma model using Ollama API"""
    try:
        print("🤖 Thinking...")
        # Check Ollama status first
        status, message = check_ollama_status()
        if not status:
            return f"AI Assistant Error: {message}"

        response = requests.post('http://localhost:11434/api/generate',
                               json={
                                   'model': 'gemma3:4b',
                                   'prompt': prompt,
                                   'stream': False,
                                   'temperature': 0.7,  # Balanced between creativity and consistency
                                   'top_p': 0.9,  # High value for more focused and coherent responses
                                   'num_predict': 500  # Allow for longer responses
                               },
                               timeout=120)  # Increased timeout for longer responses
        
        if response.status_code == 200:
            result = response.json()
            return result['response'].strip()
        else:
            return f"Error: Failed to get response (Status code: {response.status_code})"
    except requests.exceptions.Timeout:
        return "Error: Request to Gemma timed out. The model is taking too long to respond."
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Please make sure 'ollama serve' is running."
    except Exception as e:
        return f"Error communicating with Ollama: {str(e)}"

def main():
    try:
        # Initialize audio system
        print("\nInitializing audio system...")
        if not initialize_audio():
            print("Failed to initialize audio system. Please check your microphone.")
            return

        # Check Ollama status before starting
        print("\nChecking Ollama status...")
        status, message = check_ollama_status()
        if not status:
            print(f"❌ {message}")
            print("Please start Ollama by running 'ollama serve' in a terminal")
            return

        print("✅ Ollama is ready with gemma3 model")
        
        while True:
            # Wait for wake word
            if detect_wake_word():
                # Step 1: Speech to Text
                print("\n1️⃣ Listening for your message...")
                audio = record_audio(20)  # Increased to 20 seconds for longer inputs
                print("Processing audio...")
                segments, _ = stt_model.transcribe(audio, beam_size=5, word_timestamps=True)  # Better transcription settings
                text = " ".join([seg.text for seg in segments])
                if not text.strip():
                    print("❌ No speech detected. Returning to standby...")
                    continue
                print("📝 Recognized:", text)

                # Step 2: Get response from Gemma
                print("\n2️⃣ Getting AI Response")
                system_prompt = """You are a helpful and friendly AI assistant with deep knowledge and expertise.
Provide detailed and accurate responses while maintaining a natural conversational tone.
Your responses should be comprehensive yet clear and well-structured.
Feel free to provide 4-5 sentences when the topic requires more detail.
Focus on accuracy and completeness while keeping the language natural for speech.
Do not use emojis or special characters in your responses as they will be spoken aloud."""
                
                full_prompt = f"{system_prompt}\n\nUser: {text}\nAssistant:"
                reply = get_gemma_response(full_prompt)
                print("🤖 AI Response:", reply)

                # Step 3: Text to Speech
                print("\n3️⃣ Converting to Speech")
                # Clean the text by removing emojis and special characters
                import re
                cleaned_text = re.sub(r'[^\x00-\x7F]+', '', reply)  # Remove non-ASCII characters
                cleaned_text = re.sub(r'[\n\r]+', ' ', cleaned_text)  # Replace newlines with spaces
                cleaned_text = cleaned_text.strip()
                print("Cleaned text:", cleaned_text)
                tts_model.tts_to_file(text=cleaned_text, file_path="reply.wav")
                print("🔊 Reply saved as reply.wav")
                print("\n🎧 Returning to standby mode...")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n🎧 Returning to standby mode...")

if __name__ == "__main__":
    main()
