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
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
TARGET_SAMPLE_RATE = 16000
# Assuming the user has pulled the model: ollama pull gemma4:e4b
# The model name must match what is pulled into Ollama.

def refine_transcription_timed(words: list[dict], language: str = "Hungarian", cache_path: str = None, model_name: str = None) -> tuple[str, list[dict]]:
    """
    Cleans up transcription text using a sentence-level JSON structure while preserving timestamps.
    """
    model_name = model_name or os.getenv("OLLAMA_MODEL", "gemma4")
    if not words:
        return "", []

    logger.info(f"Starting ASR refinement with {model_name} (Language: {language})...")
    
    # 1. Group words into logical sentences
    sentence_objs = []
    current_sentence = []
    for i, w in enumerate(words):
        current_sentence.append(w)
        if i < len(words) - 1:
            gap = words[i+1]['start'] - w['end']
            # Split on natural pauses or punctuation
            if gap > 1.0 or w['word'].strip().endswith(('.', '?', '!', ':', '...')):
                sentence_objs.append({
                    "id": len(sentence_objs),
                    "text": " ".join(gw['word'].strip() for gw in current_sentence),
                    "start": current_sentence[0]['start'],
                    "end": current_sentence[-1]['end']
                })
                current_sentence = []
    if current_sentence:
        sentence_objs.append({
            "id": len(sentence_objs),
            "text": " ".join(gw['word'].strip() for gw in current_sentence),
            "start": current_sentence[0]['start'],
            "end": current_sentence[-1]['end']
        })

    # 2. Batching sentences to avoid context limits (e.g., 10 sentences per batch)
    BATCH_SIZE = 10
    refined_sentence_map = {}
    global_context = os.getenv("OLLAMA_REFINE_CONTEXT", "None (Start of transcript)")
    running_summary = "Starting first batch..."
    
    from openai import OpenAI
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

    system_prompt = (
        f"You are a professional {language} editor. Your task is to POST-PROCESS ASR transcripts. "
        "Fix spelling, grammar, and punctuation. You MUST return a JSON object where the keys match the input IDs and the values are the corrected strings. "
        "You MUST also include a 'batch_summary' key in the root of the JSON with a 1-sentence summary of this batch's content in the same input language. "
        "EVERY response must follow this format: {'batch_summary': '...', 'id1': 'text', 'id2': 'text', ...}. "
        "When a sentence logically continues on the next sentence ID, you must not add ellipses or capitalize the next word in the same sentence. "
        "Consider the provided context of the speech to correct errors."
    )

    for i in range(0, len(sentence_objs), BATCH_SIZE):
        batch = sentence_objs[i : i + BATCH_SIZE]
        logger.info(f"Refining batch {i // BATCH_SIZE + 1} ({len(batch)} sentences)...")
        
        # Compact structure: { "id": "text", ... }
        input_data = {str(s["id"]): s["text"] for s in batch}
        
        
        user_content = (
            f"GLOBAL CONTEXT: {global_context}\n"
            f"PREVIOUS BATCH SUMMARY: {running_summary}\n\n"
            f"DATA:\n{json.dumps(input_data, ensure_ascii=False)}"
        )
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
            )
            
            content = response.choices[0].message.content.strip()
            
            # Basic cleanup of markdown blocks if present
            if "```" in content:
                content = re.sub(r"^```(?:json)?\n|\n```$", "", content, flags=re.MULTILINE)
                
            result_data = json.loads(content)
            
            # Update running summary for next batch
            if "batch_summary" in result_data:
                running_summary = result_data["batch_summary"]
                logger.info(f"  [Rolling Summary]: {running_summary}")
            else:
                logger.warning(f"  [Warning]: Batch {i // BATCH_SIZE + 1} did not return a 'batch_summary'. Keeping previous summary.")

            # The result should be a flat dict of id -> text (plus the summary key)
            if isinstance(result_data, dict):
                for sid, stext in result_data.items():
                    if sid == "batch_summary":
                        continue
                    try:
                        sid_int = int(sid)
                        orig_text = input_data.get(sid, "")
                        if stext.strip() != orig_text.strip():
                            logger.info(f"Refinement change (ID {sid}):")
                            logger.info(f"  [Original]: {orig_text}")
                            logger.info(f"  [Refined ]: {stext}")
                        refined_sentence_map[sid_int] = stext
                    except (ValueError, TypeError):
                        continue
                        
        except Exception as e:
            logger.error(f"Error refining batch starting at {i}: {e}")
            # Fallback to original text for this batch
            for s in batch:
                refined_sentence_map[s["id"]] = s["text"]

    # 3. Reconstruct word list with interpolated timestamps
    final_refined_words = []
    for s in sentence_objs:
        corrected_text = refined_sentence_map.get(s["id"], s["text"])
        words_in_sentence = corrected_text.split()
        if not words_in_sentence:
            continue
            
        dur = s["end"] - s["start"]
        word_dur = dur / len(words_in_sentence)
        
        for j, w_text in enumerate(words_in_sentence):
            final_refined_words.append({
                "word": w_text,
                "start": s["start"] + (j * word_dur),
                "end": s["start"] + ((j + 1) * word_dur)
            })

    final_text = " ".join([w["word"] for w in final_refined_words])
    
    # Save cache if path provided
    if cache_path:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"text": final_text, "words": final_refined_words}, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save refinement cache: {e}")

    return final_text, final_refined_words
