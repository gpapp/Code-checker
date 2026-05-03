import logging
import os
import librosa

logger = logging.getLogger(__name__)

def run_vad(audio_path: str):
    """Runs NeMo VAD and returns active speech intervals with robustness improvements."""
    duration = librosa.get_duration(path=audio_path)
    
_silero_model_cache = None

def run_vad(audio_path: str):
    """Runs Silero VAD and returns active speech intervals."""
    global _silero_model_cache
    duration = librosa.get_duration(path=audio_path)
    
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if _silero_model_cache is None:
            logger.info("Loading Silero VAD model via torch.hub...")
            model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                          model='silero_vad',
                                          force_reload=False,
                                          trust_repo=True)
            model = model.to(device)
            _silero_model_cache = (model, utils)
        
        model, utils = _silero_model_cache
        (get_speech_timestamps, _, read_audio, _, _) = utils
        
        # Silero read_audio handles various formats and ensures 16kHz
        wav = read_audio(audio_path, sampling_rate=16000).to(device)
        
        # get_speech_timestamps returns a list of dicts: {'start': sample, 'end': sample}
        speech_timestamps = get_speech_timestamps(
            wav, model, 
            sampling_rate=16000,
            threshold=0.4,
            min_silence_duration_ms=300,
            speech_pad_ms=100
        )
        
        segments = [(ts['start'] / 16000, ts['end'] / 16000) for ts in speech_timestamps]
        
        if not segments:
            raise ValueError("Silero VAD returned no segments.")
            
        return segments

    except Exception as e:
        logger.warning(f"Silero VAD failed ({e}). Falling back to librosa activity detection.")
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            non_silent = librosa.effects.split(y, top_db=35)
            return [(float(start)/sr, float(end)/sr) for start, end in non_silent]
        except Exception as e2:
            logger.error(f"VAD Fallback also failed: {e2}")
            return []

def unload_silero_model():
    """Unloads Silero VAD model from memory."""
    global _silero_model_cache
    if _silero_model_cache is not None:
        logger.info("Unloading Silero VAD model...")
        _silero_model_cache = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

_whisper_model_cache = {}


def unload_crisper_model():
    """Explicitly unloads the CrisperWhisper and Silero VAD models from VRAM."""
    global _whisper_model_cache
    
    unload_silero_model()
    
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
