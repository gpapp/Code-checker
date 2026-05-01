import os
import glob
import torch
import torchaudio
from pydub import AudioSegment
from faster_whisper import WhisperModel
import pandas as pd
from tqdm import tqdm
import csv

# Constants for Hungarian Fillers - specialized for phonetic matching
TARGET_WORDS = ["őő", "öö", "eee", "hát", "szóval", "izé"]
OUTPUT_DIR = "fillers_hun"
CLIPS_DIR = os.path.join(OUTPUT_DIR, "clips")
CSV_PATH = os.path.join(OUTPUT_DIR, "PodcastFillers_HUN.csv")
NON_FILLER_CSV_PATH = os.path.join(OUTPUT_DIR, "NonFillers_HUN.csv")
MAX_CLIPS = 500
PADDING_MS = 100  # Amount of silence/buffer to add around the word

class HungarianExtractor:
    def __init__(self):
        # Using faster_whisper as the base engine.
        # To get CrisperWhisper-like performance, we use the large-v3 model 
        # which provides the best timestamp accuracy for Hungarian.
        self.model = WhisperModel(
            "large-v3", 
            device="cuda", 
            compute_type="float16"
        )
        
        if not os.path.exists(CLIPS_DIR):
            os.makedirs(CLIPS_DIR, exist_ok=True)

    def process_directory(self, root_dir):
        # Normalizáljuk az elérési utat, hogy Windows és Linux alatt is működjön
        root_dir = os.path.abspath(root_dir)
        
        # A glob rekurzív keresése néha érzékeny a perjelek irányára
        # A '**/MASTER/*.mp3' minta keres minden alkönyvtárban MASTER mappát és abban mp3-at
        search_pattern = os.path.join(root_dir, "**", "MASTER", "*.mp3")
        
        print(f"Keresési minta: {search_pattern}")
        files = glob.glob(search_pattern, recursive=True)
        
        # Ha nem talál fájlokat, megpróbáljuk kiterjeszteni a keresést (pl. alkönyvtárak mélysége miatt)
        if not files:
            # Alternatív keresés, ha a fenti nem hozna eredményt a Windows hálózati meghajtókon
            alternative_pattern = root_dir.rstrip("\\/") + "/**/*MASTER*/*.mp3"
            files = glob.glob(alternative_pattern, recursive=True)
        
        clip_count = 0
        non_filler_count = 0
        
        # Define CSV headers including 'probability'
        headers = [
            'pfID', 'consolidated_label', 'original_text', 
            'start_time', 'end_time', 'duration_ms', 
            'padded', 'probability'
        ]
        
        # Open both CSV files for immediate writing
        with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as f_fill, \
             open(NON_FILLER_CSV_PATH, mode='w', newline='', encoding='utf-8') as f_non:
            
            writer_fill = csv.DictWriter(f_fill, fieldnames=headers)
            writer_non = csv.DictWriter(f_non, fieldnames=headers)
            
            writer_fill.writeheader()
            writer_non.writeheader()
            
            f_fill.flush()
            f_non.flush()

            if not files:
                print(f"Hiba: Nem találtam mp3 fájlokat a megadott útvonalon: {root_dir}")
                return

            print(f"Talált fájlok száma: {len(files)}. Kivonatolás indítása...")

            for file_path in files:
                if clip_count >= MAX_CLIPS:
                    break
                    
                print(f"Feldolgozás: {file_path}")
                
                try:
                    # Use faster_whisper's transcription with word_timestamps enabled.
                    segments, info = self.model.transcribe(
                        file_path, 
                        language="hu", 
                        initial_prompt="Ööö, őőő, szóval, izé, hát...",
                        word_timestamps=True,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )

                    audio = AudioSegment.from_file(file_path)

                    for seg in segments:
                        if clip_count >= MAX_CLIPS:
                            break
                        
                        # Iterate through individual words in the segment
                        for word in seg.words:
                            word_text = word.word.lower().strip(".,!? ")
                            
                            # Skip empty results
                            if not word_text:
                                continue

                            # Identify if it's a filler
                            is_filler = any(target in word_text for target in TARGET_WORDS) and word_text != "tehát"
                            
                            # Common timing and metadata
                            start_ms = max(0, int(word.start * 1000) - PADDING_MS)
                            end_ms = min(len(audio), int(word.end * 1000) + PADDING_MS)
                            
                            row = {
                                'original_text': word_text,
                                'start_time': word.start,
                                'end_time': word.end,
                                'duration_ms': end_ms - start_ms,
                                'padded': True,
                                'probability': round(word.probability, 4) # Added probability
                            }

                            if is_filler and clip_count < MAX_CLIPS:
                                pfID = f"HUN_FILLER_{clip_count:04d}"
                                row['pfID'] = pfID
                                row['consolidated_label'] = self._get_label(word_text)
                                
                                # Export filler audio
                                clip = audio[start_ms:end_ms]
                                clip.export(os.path.join(CLIPS_DIR, f"{pfID}.wav"), format="wav")
                                
                                writer_fill.writerow(row)
                                f_fill.flush()
                                clip_count += 1
                            else:
                                # Log as a non-filler for the negative dataset
                                nfID = f"HUN_WORD_{non_filler_count:04d}"
                                row['pfID'] = nfID
                                row['consolidated_label'] = "Word"
                                
                                writer_non.writerow(row)
                                f_non.flush()
                                non_filler_count += 1
                                
                except Exception as e:
                    print(f"Hiba a fájl feldolgozása közben ({file_path}): {e}")
                    continue
                        
        print(f"Kész! {clip_count} töltelékszó kimentve, {non_filler_count} egyéb szó naplózva.")

    def _get_label(self, text):
        # Phonetic fillers mapped to 'Uh'
        if any(f in text for f in ["őő", "öö", "eee"]):
            return "Uh"
        # Word-based fillers mapped to 'Filler'
        if any(f in text for f in ["hát", "szóval", "izé"]):
            return "Filler"
        return "Word"

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Használat: python extract_hungarian_fillers.py /eleresi/ut/podcastokhoz")
    else:
        extractor = HungarianExtractor()
        extractor.process_directory(sys.argv[1])