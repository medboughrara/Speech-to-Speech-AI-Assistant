from faster_whisper import WhisperModel
import time

print("Loading model...")
model = WhisperModel("base", device="cuda")  # use "cuda" for GPU acceleration
print("Model loaded successfully!")

print("Starting transcription...")
start_time = time.time()
segments, info = model.transcribe("sample.wav")
print(f"Transcription completed in {time.time() - start_time:.2f} seconds")

print("\nTranscription results:")
for seg in segments:
    print(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}")
