import logging
import os
import librosa

logger = logging.getLogger(__name__)

def run_vad(audio_path: str):
    """Runs NeMo VAD and returns active speech intervals."""
    try:
        import nemo.collections.asr as nemo_asr
        import torch
    except ImportError as e:
        logger.warning(f"NeMo VAD failed to initialize: {e}. Using librosa for VAD fallback.")
        y, sr = librosa.load(audio_path, sr=16000)
        non_silent = librosa.effects.split(y, top_db=30)
        return [(float(start)/sr, float(end)/sr) for start, end in non_silent]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained("vad_multilingual_marblenet").to(device)
        probs = vad_model.transcribe([audio_path], batch_size=1)
    except Exception as e:
        logger.error(f"NeMo transcription failed: {e}")
        return []

    window_length_in_sec = 0.02
    threshold = 0.5

    segments = []
    curr_start = None
    for i, p in enumerate(probs[0]):
        if p >= threshold:
            if curr_start is None:
                curr_start = i * window_length_in_sec
        else:
            if curr_start is not None:
                segments.append((curr_start, i * window_length_in_sec))
                curr_start = None
    if curr_start is not None:
        segments.append((curr_start, len(probs[0]) * window_length_in_sec))

    return segments

_whisper_model_cache = {}

def process_filler_detection_asr(audio_path: str, model_name: str, silence_intervals: list = None, language: str = "hu"):
    """
    CrisperWhisper powered by faster-whisper.
    Highly optimized for VRAM (int8_float16), prevents OOM crashes on 8GB GPUs, 
    and handles chunking perfectly without tensor mismatch errors.
    """
    try:
        from faster_whisper import WhisperModel
        import torch
        import gc
    except ImportError as e:
        logger.warning(f"Missing dependency for CrisperWhisper: {e}")
        return "", []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use provided model_name or default to nyrahealth/faster_CrisperWhisper
    actual_model_id = model_name if model_name else "nyrahealth/faster_CrisperWhisper"
    
    global _whisper_model_cache
    cache_key = (actual_model_id, device)
    
    if cache_key not in _whisper_model_cache:
        logger.info(f"Loading faster-whisper model: {actual_model_id} on {device}...")
        # Use float16 (matches test_crisper.py) for best filler detection quality.
        # large-v3 float16 uses ~3GB VRAM, well within RTX 3060 8GB headroom.
        _whisper_model_cache[cache_key] = WhisperModel(actual_model_id, device=device, compute_type="float16" if device == "cuda" else "float32")
    
    model = _whisper_model_cache[cache_key]
    
    try:
        # Improved filler-focused prompt with universal tokens used by CrisperWhisper
        prompt_map = {
            "en": "I, uh, er, um, like stuttering, [UH], [UM], [EH], [AH], mhm.",
            "hu": "Öö, izé, hát, [UH], [UM], [EH], [AH], szóval, hmm, mhm."
        }
        actual_prompt = prompt_map.get(language, "I, uh, er, um, [UH], [UM], [EH], [AH].")
        
        logger.info(f"Transcribing {audio_path} for fillers...")
        
        # We disable condition_on_previous_text so clean segments don't suppress noisy filler detection.
        # We disable VAD filter so short "uh" sounds at the edges aren't clipped out.
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=True,
            language=language if language != "auto" else None,
            condition_on_previous_text=True,  # Context helps recognize fillers (matches test_crisper.py)
            initial_prompt=actual_prompt,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        
        full_text = []
        all_words = []
        
        # We stream the results back
        for segment in segments:
            full_text.append(segment.text.strip())
            if segment.words:
                for w in segment.words:
                    all_words.append({
                        "word": w.word.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 3)
                    })
        
        logger.info(f"Pass 1 Raw Text: {' '.join(full_text)[:120]}...")
        logger.info(f"Found {len(all_words)} word chunks.")
        
        return " ".join(full_text), all_words
        
    except Exception as e:
        logger.error(f"CrisperWhisper error: {e}")
        import traceback
        traceback.print_exc()
        return "", []

def find_fillers(words: list[dict], filler_list: list[str]) -> list[tuple[float, float]]:
    """Identifies filler words and returns their time intervals with robust fuzzy matching."""
    fillers = []
    # Prepare clean versions of the filler words
    clean_fillers = [f.lower().strip(" .,?![]()-") for f in filler_list]
    
    for w in words:
        # Original word from ASR
        raw_word = w["word"].lower().strip()
        # Cleaned word for matching
        clean_word = raw_word.strip(" .,?![]()-")
        
        # Check if word is in our filler list or matches typical filler patterns
        is_filler = False
        if clean_word in clean_fillers:
            is_filler = True
        elif raw_word.startswith("-") and raw_word.strip("-") in clean_fillers:
            is_filler = True
        elif any(f in raw_word for f in ["[uh]", "[um]", "[ah]", "[eh]"]):
            is_filler = True
            
        if is_filler:
            fillers.append((w["start"], w["end"]))
    return fillers

def unload_crisper_model():
    """Explicitly unloads the CrisperWhisper model from VRAM."""
    global _whisper_model_cache
    if _whisper_model_cache:
        logger.info("Unloading faster-whisper model from memory...")
        _whisper_model_cache.clear()
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def find_overlaps(stream_speech_intervals: dict[str, list[tuple[float, float]]], min_duration: float) -> list[tuple[float, float]]:
    """Finds intervals where at least two streams have active speech simultaneously."""
    if len(stream_speech_intervals) < 2:
        return []

    timestamps = set()
    for intervals in stream_speech_intervals.values():
        for start, end in intervals:
            timestamps.add(start)
            timestamps.add(end)

    sorted_ts = sorted(list(timestamps))
    overlaps = []

    for i in range(len(sorted_ts) - 1):
        t1, t2 = sorted_ts[i], sorted_ts[i+1]
        mid = (t1 + t2) / 2

        active_count = 0
        for intervals in stream_speech_intervals.values():
            if any(start <= mid <= end for start, end in intervals):
                active_count += 1

        if active_count >= 2:
            if overlaps and overlaps[-1][1] == t1:
                overlaps[-1] = (overlaps[-1][0], t2)
            else:
                overlaps.append((t1, t2))

    return [o for o in overlaps if (o[1] - o[0]) >= min_duration]
