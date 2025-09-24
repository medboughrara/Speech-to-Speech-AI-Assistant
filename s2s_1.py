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
stt_model = WhisperModel("base", device="cuda")

# Initialize TTS model with GPU support
print("Loading TTS model...")
tts_model = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=True)
if torch.cuda.is_available():
    print("Moving TTS model to GPU...")
    tts_model.to("cuda")

def record_audio(duration=5, samplerate=16000):
    print("🎙️ Recording for", duration, "seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    return np.squeeze(audio)

def detect_wake_word():
    """Continuously listen for the wake word 'hello my friend'"""
    print("🎧 Listening for wake word 'hello my friend'...")
    while True:
        # Record in 3-second chunks to check for wake word
        audio = record_audio(3)
        segments, _ = stt_model.transcribe(audio)
        text = " ".join([seg.text for seg in segments]).lower()
        if "hello my friend" in text:
            print("✨ Wake word detected!")
            return True
        
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
                                   'stream': False
                               },
                               timeout=60)  # 60 second timeout
        
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
        # Check Ollama status before starting
        print("Checking Ollama status...")
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
                audio = record_audio(10)
                segments, _ = stt_model.transcribe(audio)
                text = " ".join([seg.text for seg in segments])
                if not text.strip():
                    print("❌ No speech detected. Returning to standby...")
                    continue
                print("📝 Recognized:", text)

                # Step 2: Get response from Gemma
                print("\n2️⃣ Getting AI Response")
                system_prompt = """You are a helpful and friendly AI assistant.
Keep your responses concise, natural, and engaging.
Respond in a conversational way that sounds natural when spoken.
Keep responses under 3 sentences unless more detail is specifically requested.
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
