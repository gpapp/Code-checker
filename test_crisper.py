# /// script
# dependencies = [
#     "faster-whisper",
#     "torch",
#     "torchaudio",
#     "tqdm",
#     "pydub",
# ]
# ///

import json
import os
import time
from faster_whisper import WhisperModel
from tqdm import tqdm
from pydub import AudioSegment

# --- CONFIGURATION ---
FILE_PATH = "test_data/video_processing_work/2026-04-04--t08-13-10am--615e0811aa193a007ec6f9b7--gaganetz_gmail_com_a0.wav"  # Replace with your file name
MODEL_SIZE = "large-v3"   # Best for Hungarian accuracy
LANGUAGE = "hu"           # Hungarian
DEVICE = "cuda"           # Use your RTX 3060

# Common Hungarian filler words to highlight
HU_FILLERS = ["őő", "öö", "izé", "hát", "szóval", "ugye", "amúgy"]

def transcribe():
    if not os.path.exists(FILE_PATH):
        print(f"Error: {FILE_PATH} not found. Please place your MP3 in the folder.")
        return

    print(f"Loading model {MODEL_SIZE} on {DEVICE}...")
    # Initialize the model
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="float16")

    print(f"Transcribing {FILE_PATH} with VAD enabled (File duration: ~1 hour)...")
    
    # vad_filter=True enables Silero VAD to filter out non-speech
    segments, info = model.transcribe(
        FILE_PATH, 
        language=LANGUAGE, 
        word_timestamps=True,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        initial_prompt="Ööö, hát, szóval... őőő."
    )

    full_text = []
    word_timestamps = []
    found_fillers = []

    # Using tqdm to track progress for long files
    # Note: segments is a generator, so we track progress based on the audio duration
    pbar = tqdm(total=round(info.duration), unit="sec", desc="Processing Audio")
    last_ts = 0

    for segment in segments:
        full_text.append(segment.text)
        
        # Update progress bar
        pbar.update(segment.end - last_ts)
        last_ts = segment.end

        for word in segment.words:
            word_data = {
                "word": word.word.strip(),
                "start": round(word.start, 3),
                "end": round(word.end, 3),
                "probability": round(word.probability, 3)
            }
            word_timestamps.append(word_data)
            
            # Check if it's a filler
            clean_word = word.word.strip().lower().strip(".,!?")
            if clean_word in HU_FILLERS:
                found_fillers.append(word_data)
    
    pbar.close()

    # --- FILLER EXTRACTION ---
    if found_fillers:
        print(f"Extracting {len(found_fillers)} fillers to extracted_fillers.wav...")
        try:
            full_audio = AudioSegment.from_file(FILE_PATH)
            combined_fillers = AudioSegment.empty()
            
            # Padding in ms to avoid clipping the start/end of the word
            PADDING_MS = 50 
            
            for filler in found_fillers:
                start_ms = max(0, int(filler['start'] * 1000) - PADDING_MS)
                end_ms = min(len(full_audio), int(filler['end'] * 1000) + PADDING_MS)
                
                # Extract segment and add a tiny crossfade or silent gap
                filler_segment = full_audio[start_ms:end_ms]
                combined_fillers += filler_segment + AudioSegment.silent(duration=100)
            
            combined_fillers.export("extracted_fillers.wav", format="wav")
            print("Successfully generated extracted_fillers.wav")
        except Exception as e:
            print(f"Error during filler extraction: {e}")
    else:
        print("No fillers found, skipping extraction.")

    # Save Full Text
    with open("output_transcript.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(full_text))

    # Save All Word Timestamps
    with open("word_timestamps.json", "w", encoding="utf-8") as f:
        json.dump(word_timestamps, f, ensure_ascii=False, indent=2)

    # Save Found Fillers
    with open("detected_fillers.txt", "w", encoding="utf-8") as f:
        f.write("Detected Hungarian Filler Words:\n")
        for filler in found_fillers:
            f.write(f"[{filler['start']}s - {filler['end']}s] {filler['word']} (Prob: {filler['probability']})\n")

    print("\n--- Processing Complete ---")
    print(f"Detected Language: {info.language} ({info.language_probability:.2f})")
    print(f"Total Fillers Found: {len(found_fillers)}")
    print("Files generated: output_transcript.txt, word_timestamps.json, detected_fillers.txt, extracted_fillers.wav")

if __name__ == "__main__":
    transcribe()
