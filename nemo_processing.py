import logging
from typing import List, Tuple, Dict
import librosa

logger = logging.getLogger(__name__)

def run_vad(audio_path: str):
    """Runs NeMo VAD and returns active speech intervals."""
    try:
        import nemo.collections.asr as nemo_asr
        import torch
    except ImportError:
        logger.warning("NeMo not installed. Using librosa for VAD fallback.")
        y, sr = librosa.load(audio_path, sr=16000)
        non_silent = librosa.effects.split(y, top_db=30)
        return [(float(start)/sr, float(end)/sr) for start, end in non_silent]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vad_model = nemo_asr.models.VADModel.from_pretrained("vad_multilingual_marblenet").to(device)

    probs = vad_model.transcribe([audio_path], batch_size=1)

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

def run_asr(audio_path: str, model_name: str):
    """Runs NeMo ASR and returns transcription and word-level timestamps."""
    try:
        import nemo.collections.asr as nemo_asr
        import torch
    except ImportError:
        logger.warning("NeMo or Torch not installed. Skipping ASR.")
        return "", []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model = nemo_asr.models.ASRModel.from_pretrained(model_name).to(device)

    res = model.transcribe([audio_path], return_hypotheses=True)
    if not res or len(res[0]) == 0:
        return "", []

    hyp = res[0][0]
    words = []

    if hasattr(hyp, 'word_timestamps') and hyp.word_timestamps:
        word_texts = hyp.text.split()
        for i, word_text in enumerate(word_texts):
            if i < len(hyp.word_timestamps):
                ts = hyp.word_timestamps[i]
                if isinstance(ts, (list, tuple)):
                    words.append({"word": word_text, "start": float(ts[0]), "end": float(ts[1])})
                else:
                    words.append({"word": word_text, "start": float(ts), "end": float(ts) + 0.3})
    elif hasattr(hyp, 'words') and hyp.words:
        for w in hyp.words:
            words.append({"word": w.word, "start": float(w.start_time), "end": float(w.end_time)})

    return hyp.text, words

def find_fillers(words: List[Dict], filler_list: List[str]) -> List[Tuple[float, float]]:
    """Identifies filler words and returns their time intervals."""
    fillers = []
    for w in words:
        clean_word = w["word"].lower().strip(".,?!")
        if clean_word in [f.lower().strip() for f in filler_list]:
            fillers.append((w["start"], w["end"]))
    return fillers

def find_overlaps(stream_speech_intervals: Dict[str, List[Tuple[float, float]]], min_duration: float) -> List[Tuple[float, float]]:
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
