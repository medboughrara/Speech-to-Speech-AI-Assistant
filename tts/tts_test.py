from TTS.api import TTS
import torch
import os
import time

def get_device():
    print("\nCUDA Environment:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch CUDA enabled: {torch.backends.cuda.is_built()}")
    
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = "cpu"
        print("No GPU available, using CPU")
    
    return device

try:
    print("Initializing TTS...")
    
    # Using Glow-TTS model which is known to work well with GPU
    model_name = "tts_models/en/ljspeech/glow-tts"
    print(f"\nLoading model: {model_name}")
    
    # Force CUDA device if available
    device = get_device()
    print(f"\nInitializing model on {device}...")
    
    # Initialize TTS with GPU
    tts = TTS(model_name=model_name, progress_bar=True)
    if device == "cuda":
        print("Moving model to GPU...")
        tts.to(device)
    
    # Generate speech
    text = "Hello, this is a test of the text to speech system using GPU acceleration for faster processing and better quality output."
    output_file = "output.wav"
    print(f"\nGenerating speech for text: '{text}'")
    
    # Measure generation time
    start_time = time.time()
    tts.tts_to_file(text=text, file_path=output_file)
    generation_time = time.time() - start_time
    
    if os.path.exists(output_file):
        print(f"\nAudio generated successfully: {output_file}")
        print(f"Generation time: {generation_time:.2f} seconds")
        print(f"Using device: {device}")
    else:
        print("Error: Audio file was not generated")

except Exception as e:
    print(f"Error occurred: {str(e)}")