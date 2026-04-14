#!/usr/bin/env python3
"""
Service layer for running ASR using a local Ollama endpoint with a Gemma model.

This module handles audio preprocessing (resampling, chunking) and communicates
with the local Ollama server API.
"""
import os
import logging
import tempfile
import soundfile as sf
import base64
import requests
import librosa
import json
import re
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)

# --- Constants ---
OLLAMA_BASE_URL = "http://localhost:11434"
TARGET_SAMPLE_RATE = 16000
# Assuming the user has pulled the model: ollama pull gemma:7b
# The model name must match what is pulled into Ollama.

def _preprocess_audio(audio_path: str, target_sr: int = 16000) -> Tuple[Optional[str], Optional[float]]:
    """Loads, resamples, and saves audio to a temporary 16kHz WAV file."""
    try:
        # Load and resample to the target rate in one go
        y_resampled, actual_sr = librosa.load(audio_path, sr=target_sr)
        
        # Limit duration to 30s for stability/memory safety in tests
        max_samples = 30 * actual_sr
        if len(y_resampled) > max_samples:
            logger.info(f"Audio too long ({len(y_resampled)/actual_sr:.1f}s), truncating to 30s for ASR.")
            y_resampled = y_resampled[:max_samples]
            
        duration = float(len(y_resampled) / actual_sr)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            temp_wav_path = tmp_file.name
            sf.write(temp_wav_path, y_resampled, actual_sr)

        logger.info(f"Audio preprocessed and saved to temp file: {temp_wav_path}")
        return temp_wav_path, duration
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        return None, None

def run_gemma_asr_test(
    audio_path: str | Dict[str, Any],
    model_name: str,
    language: str,
    test_type: str
) -> Dict[str, str]:
    """
    Unified function to test ASR against Ollama/Gemma.
    Handles different input types (path, raw array, dict).
    """
    logger.info(f"--- Starting ASR Test ({test_type}) using Model: {model_name} ---")
    temp_wav_path = None

    try:
        # --- 1. Determine the actual path/data to process ---
        if isinstance(audio_path, dict) and "raw" in audio_path:
            # Case: Raw Numpy Array (Passed as dict in the test)
            logger.info("Processing input as raw numpy array.")
            # For testing, we must save the raw array to disk for Ollama to read or for pre-processing tools.
            # This is a necessary limitation when simulating API calls.
            raw_audio = audio_path['raw']
            sampling_rate = audio_path.get('sampling_rate', TARGET_SAMPLE_RATE)
            
            # If sampling rate is different from target, resample
            if sampling_rate != TARGET_SAMPLE_RATE:
                raw_audio = librosa.resample(raw_audio, orig_sr=sampling_rate, target_sr=TARGET_SAMPLE_RATE)
            
            # Limit duration to 30s for stability/memory safety in tests
            max_samples = 30 * TARGET_SAMPLE_RATE
            if len(raw_audio) > max_samples:
                logger.info(f"Raw audio too long ({len(raw_audio)/TARGET_SAMPLE_RATE:.1f}s), truncating to 30s for ASR.")
                raw_audio = raw_audio[:max_samples]
                
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_wav_path = tmp_file.name
            sf.write(temp_wav_path, raw_audio, TARGET_SAMPLE_RATE)

        elif isinstance(audio_path, str):
            # Case: File path (This includes the original mp3 or a previously generated wav)
            # We use the WAV path if we generated one, otherwise we treat the input as the source.
            temp_wav_path, _ = _preprocess_audio(audio_path)

        else:
            raise TypeError("Unsupported audio_path type provided.")

        if not temp_wav_path:
            return {"text": "ERROR: Could not preprocess audio into a usable WAV file path."}

        # --- 2. Call the Ollama endpoint ---
        logger.info(f"Calling Ollama at {OLLAMA_BASE_URL} for transcription...")

        with open(temp_wav_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        # Construct payload for Gemma 4 multimodal ASR
        # We use the /api/chat endpoint which is standard for recent multimodal interactions
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": f"Transcribe this audio snippet. Language: {language}. Provide ONLY the transcribed text.",
                    "images": [audio_base64]
                }
            ],
            "options": {
                "num_ctx": 8192  # Reduced from 131k to save VRAM and prevent memory pool errors
            },
            "stream": False
        }

        response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=60)
        response.raise_for_status()
        
        result_data = response.json()
        transcribed_text = result_data.get('message', {}).get('content', '').strip()

        if not transcribed_text:
             logger.warning("Ollama returned an empty transcription.")
             transcribed_text = "[Empty Transcription]"

        return {"text": transcribed_text}

    except Exception as e:
        return {"text": f"ERROR: Failed during ASR testing. Ensure Ollama is running and the model '{model_name}' is pulled. Original Error: {str(e)}"}
    finally:
        # Cleanup the temporary WAV file
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

def refine_transcription_timed(words: list[dict], language: str = "Hungarian", cache_path: str = None, model_name: str = "gemma4") -> tuple[str, list[dict]]:
    """
    Cleans up transcription text using Gemma 4 while preserving timestamps.
    Uses sliding window context and sentence grouping.
    """
    if not words:
        return "", []

    logger.info(f"Starting ASR refinement with {model_name} (Language: {language})...")
    
    # 1. Group words into logical sentences based on silence gaps (>0.5s)
    sentences = []
    current_sentence = []
    for i, w in enumerate(words):
        current_sentence.append(w)
        if i < len(words) - 1:
            gap = words[i+1]['start'] - w['end']
            if gap > 0.5 or len(current_sentence) > 40:
                sentences.append(current_sentence)
                current_sentence = []
    if current_sentence:
        sentences.append(current_sentence)

    refined_words = []
    context_window = [] # Keep last 2 refined sentences as context
    
    # 4. Resume Logic
    tmp_path = cache_path + ".tmp" if cache_path else None
    if tmp_path and os.path.exists(tmp_path):
        try:
            with open(tmp_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
                refined_words = checkpoint.get("words", [])
                logger.info(f"Resuming refinement from checkpoint ({len(refined_words)} refined words).")
        except Exception as e:
            logger.warning(f"Failed to load refinement checkpoint {tmp_path}: {e}. Starting from scratch.")

    for idx, batch_words in enumerate(sentences):
        # Determine if we can skip this batch if resuming
        if refined_words:
            # Check if the last refined word's end time is past this batch's end time
            batch_end = batch_words[-1]["end"]
            if refined_words[-1]["end"] >= batch_end - 0.1:
                continue

        context_text = " ".join(context_window[-2:]) if context_window else "None (Start of transcript)"
        
        system_prompt = f"You are a professional {language} editor. Your task is to POST-PROCESS ASR transcripts. Fix all spelling, grammar, and punctuation while maintaining the original JSON structure. CAPITALIZE the start of sentences."
        
        user_content = f"""CONTEXT: {context_text}
INPUT JSON (to be corrected):
{json.dumps(batch_words, ensure_ascii=False)}
"""

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "stream": False
        }

        try:
            logger.info(f"Refining sentence {idx+1}/{len(sentences)}...")
            response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=300)
            response.raise_for_status()
            
            result_data = response.json()
            content = result_data.get('message', {}).get('content', '').strip()
            
            try:
                # Some models might wrap JSON in code blocks
                if "```json" in content:
                    content = re.search(r"```json\n(.*?)\n```", content, re.DOTALL).group(1)
                elif "```" in content:
                    content = re.search(r"```\n(.*?)\n```", content, re.DOTALL).group(1)
                
                r_words = json.loads(content)
                if isinstance(r_words, list):
                    # Append to master list
                    refined_words.extend(r_words)
                    context_window.append(" ".join([w.get('word', '') for w in r_words]))
                    
                    # Continuous output: save to tmp
                    if tmp_path:
                        try:
                            r_text = " ".join([w["word"] for w in refined_words])
                            current_state = {"text": r_text, "words": refined_words}
                            with open(tmp_path, "w", encoding="utf-8") as f:
                                json.dump(current_state, f, ensure_ascii=False)
                        except Exception as e:
                            logger.warning(f"Failed to save refinement checkpoint: {e}")
                else:
                    # Fallback to original if format is unexpected
                    refined_words.extend(batch_words)
            except Exception as pe:
                logger.warning(f"Failed to parse Gemma output for sentence {idx+1}: {pe}. Using original.")
                refined_words.extend(batch_words)
                
        except Exception as e:
            logger.error(f"Error calling Gemma for cleanup (sentence {idx+1}): {e}")
            # Fallback: keep original words for this batch to avoid losing timestamps
            refined_words.extend(batch_words)

    # Finalization: rename tmp to final cache_path
    if tmp_path and os.path.exists(tmp_path):
         if cache_path:
             if os.path.exists(cache_path): os.remove(cache_path)
             os.rename(tmp_path, cache_path)

    final_text = " ".join([w.get('word', '') for w in refined_words])
    return final_text.strip(), refined_words
