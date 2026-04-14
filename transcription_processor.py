#!/usr/bin/env python3
"""
transcription_processor.py
Pass 2 transcription using faster-whisper (large-v3).
Reuses the cached WhisperModel from filler_processor to avoid loading the model twice.
"""
import os
import json
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

LANGUAGE_MAP = {
    "hu": "Hungarian",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "ro": "Romanian",
    "pl": "Polish",
}

def get_full_language_name(lang_code: str) -> str:
    """Maps ISO language codes to full names."""
    return LANGUAGE_MAP.get(lang_code.lower(), lang_code)


def process_transcription(
    audio_path: str,
    model_name: str,
    silence_intervals: list = None,
    language: str = "hu",
    cache_path: str = None,
) -> tuple[str, list[dict]]:
    """
    Runs ASR using faster-whisper (large-v3 by default).
    Reuses the shared WhisperModel cache from filler_processor if available,
    otherwise loads the model fresh.
    Returns (full_text, words) where words is a list of
    {word, start, end, probability} dicts.
    """
    try:
        from faster_whisper import WhisperModel
        import torch
    except ImportError as e:
        logger.error(f"faster-whisper not available: {e}")
        return "", []

    actual_model = model_name if model_name else "large-v3"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "int8_float16" if device == "cuda" else "float32"

    # Try to reuse the model that filler_processor already has in memory
    try:
        from filler_processor import _whisper_model_cache
        cache_key = (actual_model, device)
        if cache_key not in _whisper_model_cache:
            logger.info(f"Loading faster-whisper model for transcription: {actual_model} on {device}...")
            _whisper_model_cache[cache_key] = WhisperModel(actual_model, device=device, compute_type=compute_type)
        model = _whisper_model_cache[cache_key]
    except Exception:
        # Fallback: load our own instance
        logger.info(f"Loading standalone faster-whisper model: {actual_model} on {device}...")
        model = WhisperModel(actual_model, device=device, compute_type=compute_type)

    # Resume logic
    all_text_parts: list[str] = []
    all_words: list[dict] = []
    tmp_path = cache_path + ".tmp" if cache_path else None

    if tmp_path and os.path.exists(tmp_path):
        try:
            with open(tmp_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
                all_text_parts = [checkpoint.get("text", "")]
                all_words = checkpoint.get("words", [])
                logger.info(f"Resuming {os.path.basename(audio_path)} from checkpoint ({len(all_words)} words).")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {tmp_path}: {e}. Starting from scratch.")

    try:
        # Transcription prompt to help with Hungarian fillers
        prompt_map = {
            "hu": "Öö, izé, hát, szóval, ugye, amúgy, hmm, mhm.",
            "en": "I, uh, er, um, like stuttering, mhm.",
        }
        initial_prompt = prompt_map.get(language, "")

        logger.info(f"Transcribing {os.path.basename(audio_path)} with faster-whisper {actual_model}...")

        segments, info = model.transcribe(
            audio_path,
            language=language if language != "auto" else None,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            initial_prompt=initial_prompt if not all_words else None,  # skip if resuming
            condition_on_previous_text=True,
        )

        for segment in tqdm(segments, desc=f"  -> {os.path.basename(audio_path)[:25]}", leave=False, unit="seg"):
            # Skip segments already covered by a resumed checkpoint
            if all_words and segment.end <= all_words[-1]["end"] + 0.5:
                continue

            all_text_parts.append(segment.text.strip())

            if segment.words:
                for w in segment.words:
                    all_words.append({
                        "word": w.word.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 3),
                    })

            # Continuous checkpoint save
            if tmp_path:
                try:
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump({"text": " ".join(all_text_parts).strip(), "words": all_words}, f, ensure_ascii=False)
                except Exception as e:
                    logger.warning(f"Checkpoint save failed: {e}")

        final_text = " ".join(all_text_parts).strip()

        # Promote tmp -> final cache
        if tmp_path and os.path.exists(tmp_path) and cache_path:
            if os.path.exists(cache_path):
                os.remove(cache_path)
            os.rename(tmp_path, cache_path)

        logger.info(f"Transcription complete: {len(all_words)} words, lang={info.language} ({info.language_probability:.2f})")
        return final_text, all_words

    except Exception as e:
        logger.error(f"faster-whisper transcription error: {e}")
        import traceback
        traceback.print_exc()
        return "", []


def unload_transcription_model():
    """
    Unloads the shared faster-whisper model from VRAM.
    Since filler_processor and transcription_processor share the cache,
    this clears both.
    """
    try:
        from filler_processor import unload_crisper_model
        unload_crisper_model()
    except Exception:
        pass
