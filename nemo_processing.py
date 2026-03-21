import logging
import os
import subprocess
import tempfile
import librosa
import time

logger = logging.getLogger(__name__)

def run_vad(audio_path: str):
    """Runs NeMo VAD and returns active speech intervals."""
    try:
        import nemo.collections.asr as nemo_asr
        import torch
    except Exception as e:
        logger.warning(f"NeMo VAD failed to initialize: {e}. Using librosa for VAD fallback.")
        y, sr = librosa.load(audio_path, sr=16000)
        non_silent = librosa.effects.split(y, top_db=30)
        return [(float(start)/sr, float(end)/sr) for start, end in non_silent]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained("vad_multilingual_marblenet").to(device)

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

def run_asr(audio_path: str, model_name: str, silence_intervals: list[tuple[float, float]] = None):
    """Runs ASR using Qwen3-ASR (primary) or NeMo (fallback) and returns transcription and word-level timestamps."""
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or "cuda" else "cpu"
    # Note: Torch check for cuda is more reliable
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    if "qwen" in model_name.lower():
        return _run_qwen_asr(audio_path, model_name, device, silence_intervals)
    else:
        return _run_nemo_asr(audio_path, model_name, device, silence_intervals)

def _run_qwen_asr(audio_path: str, model_name: str, device: str, silence_intervals: list[tuple[float, float]] = None):
    try:
        from qwen_asr import Qwen3ASRModel
        import torch
    except ImportError as e:
        logger.warning(f"qwen-asr or torch not found: {e}. Falling back to NeMo if available.")
        return _run_nemo_asr(audio_path, "nvidia/parakeet-tdt-0.6b-v3", device, silence_intervals)

    logger.info(f"Using Qwen3-ASR with model: {model_name} on {device}")
    
    # Initialize the model with the forced aligner for precision timestamps
    try:
        model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else "cpu",
            forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
            forced_aligner_kwargs=dict(
                dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else "cpu",
            ),
        )
    except Exception as e:
        logger.error(f"Failed to load Qwen3-ASR model: {e}")
        return "", []

    duration = librosa.get_duration(path=audio_path)
    # Qwen3 with forced aligner can handle long files better, but 300s (5m) is the recommended max for accuracy
    chunk_size = 120.0 
    
    full_text = []
    all_words = []
    
    # Use existing chunking logic for safety
    split_points = [0.0]
    curr = 0.0
    while curr < duration - chunk_size:
        target = curr + chunk_size
        found_silence = False
        if silence_intervals:
            best_s = None
            best_diff = 30.0
            for s_start, s_end in silence_intervals:
                mid = (s_start + s_end) / 2
                diff = abs(mid - target)
                if diff < best_diff:
                    best_diff = diff
                    best_s = mid
            if best_s:
                split_points.append(best_s)
                curr = best_s
                found_silence = True
        if not found_silence:
            curr += chunk_size
            split_points.append(curr)
    split_points.append(duration)

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i+1]
            if (end - start) < 0.1: continue
            
            chunk_path = os.path.join(tmpdir, f"chunk_{i}.wav")
            # Qwen3 handles various formats, but WAV is safest
            cmd = [
                "ffmpeg", "-i", audio_path, "-ss", str(start), "-to", str(end),
                "-ac", "1", "-ar", "16000", "-y", chunk_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            
            logger.info(f"Transcribing chunk {i+1}/{len(split_points)-1} with Qwen3: {start:.2f}s - {end:.2f}s")
            try:
                # transcribe returns a list of results
                res_list = model.transcribe(
                    audio=[chunk_path],
                    return_time_stamps=True
                )
                if res_list:
                    res = res_list[0]
                    full_text.append(res.text)
                    if hasattr(res, 'time_stamps'):
                        for ts in res.time_stamps:
                            all_words.append({
                                "word": ts.text,
                                "start": ts.start_time + start,
                                "end": ts.end_time + start
                            })
            except Exception as e:
                logger.error(f"Qwen3 transcription error on chunk {i}: {e}")
            
            if device == "cuda":
                torch.cuda.empty_cache()

    return " ".join(full_text), all_words

def _run_nemo_asr(audio_path: str, model_name: str, device: str, silence_intervals: list[tuple[float, float]] = None):
    try:
        import nemo.collections.asr as nemo_asr
        import torch
    except ImportError:
        logger.warning("NeMo not installed. Skipping NeMo ASR.")
        return "", []

    logger.info(f"Using NeMo ASR with model: {model_name} on {device}")
    # ... (rest of NeMo logic moved here)
    model = nemo_asr.models.ASRModel.from_pretrained(model_name).to(device)
    
    # Disable CUDA graphs if supported to improve memory stability/compatibility
    try:
        if hasattr(model, 'decoding') and hasattr(model.decoding, 'decoding') and hasattr(model.decoding.decoding, 'decoding_computer'):
            model.decoding.decoding.decoding_computer.disable_cuda_graphs()
            logger.info("CUDA graphs disabled for ASR decoding.")
    except Exception as e:
        logger.debug(f"Could not disable CUDA graphs: {e}")

    # Explicitly enable timestamps to ensure word-level metadata is generated
    try:
        if hasattr(model, 'change_decoding_strategy'):
            model.change_decoding_strategy(compute_timestamps=True)
            logger.info("Timestamps enabled in decoding strategy.")
    except Exception as e:
        logger.debug(f"Could not enable timestamps via change_decoding_strategy: {e}")

    duration = librosa.get_duration(path=audio_path)
    chunk_size = 30.0
    
    full_text = []
    all_words = []
    
    split_points = [0.0]
    curr = 0.0
    while curr < duration - chunk_size:
        target = curr + chunk_size
        found_silence = False
        if silence_intervals:
            best_s = None
            best_diff = 30.0
            for s_start, s_end in silence_intervals:
                mid = (s_start + s_end) / 2
                diff = abs(mid - target)
                if diff < best_diff:
                    best_diff = diff
                    best_s = mid
            if best_s:
                split_points.append(best_s)
                curr = best_s
                found_silence = True
        if not found_silence:
            curr += chunk_size
            split_points.append(curr)
    split_points.append(duration)

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i+1]
            if (end - start) < 0.1: continue
            
            chunk_path = os.path.join(tmpdir, f"chunk_{i}.wav")
            cmd = [
                "ffmpeg", "-i", audio_path, "-ss", str(start), "-to", str(end),
                "-c", "copy", "-y", chunk_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            
            logger.info(f"Transcribing chunk {i+1}/{len(split_points)-1} with NeMo: {start:.2f}s - {end:.2f}s")
            try:
                res = model.transcribe([chunk_path], return_hypotheses=True)
                text, words = _parse_hyp(res, start)
                if text: full_text.append(text)
                if words: all_words.extend(words)
            except Exception as e:
                logger.error(f"NeMo error: {e}")
            
            if device == "cuda":
                torch.cuda.empty_cache()
    
    return " ".join(full_text), all_words

def _parse_hyp(res, offset: float):
    if not res:
        return "", []

    # NeMo can return list of lists or just a list of hypotheses
    # Latest Parakeet/RNNT models return a list[Hypothesis] directly when return_hypotheses=True
    if isinstance(res, list) and len(res) > 0:
        hyp = res[0]
        # If it's a list of lists, take the first element of the inner list
        if isinstance(hyp, list) and len(hyp) > 0:
            hyp = hyp[0]
    else:
        return "", []

    if hyp is None:
        return "", []

    # Ensure hyp has the expected attributes
    words = []

    # Try different ways NeMo returns timestamps
    words_attr = getattr(hyp, 'words', None)
    timestep_attr = getattr(hyp, 'timestep', None)
    word_ts_attr = getattr(hyp, 'word_timestamps', None)
    timestamp_attr = getattr(hyp, 'timestamp', None)
    
    if words_attr is not None:
        for w in words_attr:
            start = getattr(w, 'start_time', getattr(w, 'start', None))
            end = getattr(w, 'end_time', getattr(w, 'end', None))
            text = getattr(w, 'word', getattr(w, 'text', ""))
            if start is not None and end is not None:
                words.append({"word": text, "start": float(start) + offset, "end": float(end) + offset})
    
    elif word_ts_attr is not None:
        word_texts = getattr(hyp, 'text', '')
        if word_texts is None: word_texts = ''
        word_texts = word_texts.split()
        
        ts_list = word_ts_attr
        for i, word_text in enumerate(word_texts):
            if i < len(ts_list):
                ts = ts_list[i]
                if isinstance(ts, (list, tuple)):
                    words.append({"word": word_text, "start": float(ts[0]) + offset, "end": float(ts[1]) + offset})
                else:
                    words.append({"word": word_text, "start": float(ts) + offset, "end": float(ts) + 0.3 + offset})
                    
    elif timestep_attr is not None or timestamp_attr is not None:
        # Some newer models return a timestamp/timestep dict or object
        ts_data = timestep_attr if timestep_attr is not None else timestamp_attr
        words_list = None
        if hasattr(ts_data, 'get'): # Dict-like
            words_list = ts_data.get('words', [])
        elif hasattr(ts_data, 'words'): # Object-like
            words_list = ts_data.words
            
        if words_list is not None:
            for w in words_list:
                # Handle both dict-like and object-like world structures
                if hasattr(w, 'get'):
                    w_text = w.get('word', w.get('text', ''))
                    w_start = w.get('start_offset', w.get('start', w.get('start_time', None)))
                    w_end = w.get('end_offset', w.get('end', w.get('end_time', None)))
                else:
                    w_text = getattr(w, 'word', getattr(w, 'text', ''))
                    w_start = getattr(w, 'start_offset', getattr(w, 'start', getattr(w, 'start_time', None)))
                    w_end = getattr(w, 'end_offset', getattr(w, 'end', getattr(w, 'end_time', None)))
                
                if w_start is not None and w_end is not None:
                    words.append({"word": w_text, "start": float(w_start) + offset, "end": float(w_end) + offset})

    if not words:
        logger.debug(f"No words extracted from hyp. Attributes: {dir(hyp)}")

    hyp_text = getattr(hyp, 'text', '')
    if hyp_text is None: hyp_text = ''
    return hyp_text, words

def find_fillers(words: list[dict], filler_list: list[str]) -> list[tuple[float, float]]:
    """Identifies filler words and returns their time intervals."""
    fillers = []
    for w in words:
        clean_word = w["word"].lower().strip(".,?!")
        if clean_word in [f.lower().strip() for f in filler_list]:
            fillers.append((w["start"], w["end"]))
    return fillers

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
