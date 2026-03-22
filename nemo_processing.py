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

def run_asr(audio_path: str, model_name: str, silence_intervals: list[tuple[float, float]] = None, language: str = "auto"):
    """Runs ASR using WhisperX (primary), Qwen3-ASR, or NeMo (fallback) and returns transcription and word-level timestamps."""
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
    elif "parakeet" in model_name.lower() or "nemo" in model_name.lower():
        return _run_nemo_asr(audio_path, model_name, device, silence_intervals)
    elif "crisper" in model_name.lower():
        return _run_crisper_whisper(audio_path, model_name, device, silence_intervals, language=language)
    else:
        return _run_whisperx_asr(audio_path, model_name, device, silence_intervals)

def adjust_pauses_for_hf_pipeline_output(pipeline_output, split_threshold=0.12):
    """
    Adjust pause timings by distributing pauses up to the threshold evenly between adjacent words.
    From nyrahealth/CrisperWhisper/utils.py
    """
    adjusted_chunks = pipeline_output["chunks"].copy()

    for i in range(len(adjusted_chunks) - 1):
        current_chunk = adjusted_chunks[i]
        next_chunk = adjusted_chunks[i + 1]

        current_start, current_end = current_chunk["timestamp"]
        next_start, next_end = next_chunk["timestamp"]
        pause_duration = next_start - current_end

        if pause_duration > 0:
            if pause_duration > split_threshold:
                distribute = split_threshold / 2
            else:
                distribute = pause_duration / 2

            # Adjust current chunk end time
            adjusted_chunks[i]["timestamp"] = (current_start, current_end + distribute)

            # Adjust next chunk start time
            adjusted_chunks[i + 1]["timestamp"] = (next_start - distribute, next_end)
    pipeline_output["chunks"] = adjusted_chunks

    return pipeline_output

def _run_crisper_whisper(audio_path: str, model_name: str, device: str, silence_intervals: list = None, language: str = "auto"):
    """
    CrisperWhisper with unified audio stream processing.
    
    Strategy: Instead of transcribing many small speech-only chunks (which causes
    model recalibration on each), we build unified ~5-minute audio files by
    concatenating speech segments with short silence stubs (0.3s) between them.
    After transcription, word timestamps are re-mapped back to the original
    timeline by expanding stubs back to their original silence durations.
    
    Uses 'int8_float16' for ~2GB VRAM footprint on RTX 3060.
    """
    try:
        from faster_whisper import WhisperModel
        import torch
        import gc
        import soundfile as sf
        import numpy as np
    except ImportError as e:
        logger.warning(f"Missing dependency for CrisperWhisper: {e}")
        return "", []

    if device == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    faster_model_id = "nyrahealth/faster_CrisperWhisper"
    logger.info(f"Using Unified-Stream CrisperWhisper: {faster_model_id} (language: {language})")
    
    SILENCE_THRESHOLD = 2.0   # Only truncate silences longer than this
    STUB_DURATION = 0.3       # Replace long silences with this short stub
    MAX_UNIFIED_CHUNK = 300.0 # Target ~5 min unified chunks
    SR = 16000                # Sample rate for processing
    
    try:
        model = WhisperModel(faster_model_id, device=device, compute_type="int8_float16")
        duration = librosa.get_duration(path=audio_path)
        
        # Load the full audio once at 16kHz mono
        full_audio, _ = librosa.load(audio_path, sr=SR, mono=True)
        
        # 1. BUILD A TIME MAP: list of (orig_start, orig_end, type)
        #    type = "speech" or "silence"
        #    Speech segments play as-is, long silences get truncated to STUB_DURATION
        time_map = []  # list of (orig_start, orig_end, is_long_silence)
        
        if not silence_intervals:
            time_map.append((0.0, duration, False))
        else:
            silences = sorted(silence_intervals)
            long_silences = [(s, e) for s, e in silences if (e - s) >= SILENCE_THRESHOLD]
            
            last_end = 0.0
            for s_start, s_end in long_silences:
                if s_start > last_end + 0.05:
                    time_map.append((last_end, s_start, False))    # speech
                time_map.append((s_start, s_end, True))            # long silence
                last_end = s_end
            if last_end < duration - 0.05:
                time_map.append((last_end, duration, False))       # trailing speech
        
        # 2. GROUP segments into unified chunks of ~MAX_UNIFIED_CHUNK (by unified duration)
        unified_chunks = []  # each is a list of (orig_start, orig_end, is_long_silence)
        current_group = []
        current_unified_dur = 0.0
        
        for orig_start, orig_end, is_long_silence in time_map:
            orig_dur = orig_end - orig_start
            unified_dur = STUB_DURATION if is_long_silence else orig_dur
            
            # If adding this segment would exceed the limit, and we already have content, start a new group
            if current_unified_dur + unified_dur > MAX_UNIFIED_CHUNK and current_group:
                unified_chunks.append(current_group)
                current_group = []
                current_unified_dur = 0.0
            
            current_group.append((orig_start, orig_end, is_long_silence))
            current_unified_dur += unified_dur
        
        if current_group:
            unified_chunks.append(current_group)
        
        logger.info(f"Built {len(unified_chunks)} unified chunks from {len(time_map)} segments.")
        
        # 3. FOR EACH UNIFIED CHUNK: build concatenated audio, transcribe, re-map timestamps
        full_text = []
        all_words = []
        stub_samples = np.zeros(int(STUB_DURATION * SR), dtype=np.float32)
        
        from tqdm import tqdm
        with tempfile.TemporaryDirectory() as tmpdir:
            for chunk_idx in tqdm(range(len(unified_chunks)), desc="ASR Progress"):
                group = unified_chunks[chunk_idx]
                
                # Build the concatenated audio and a mapping table
                # mapping_table: list of (unified_start, unified_end, orig_start, orig_end, is_silence)
                audio_pieces = []
                mapping_table = []
                unified_pos = 0.0
                
                for orig_start, orig_end, is_long_silence in group:
                    if is_long_silence:
                        # Insert a short silence stub
                        audio_pieces.append(stub_samples.copy())
                        unified_end = unified_pos + STUB_DURATION
                        mapping_table.append((unified_pos, unified_end, orig_start, orig_end, True))
                        unified_pos = unified_end
                    else:
                        # Insert the actual speech audio
                        s_sample = int(orig_start * SR)
                        e_sample = int(orig_end * SR)
                        speech_audio = full_audio[s_sample:e_sample]
                        audio_pieces.append(speech_audio)
                        unified_dur = len(speech_audio) / SR
                        unified_end = unified_pos + unified_dur
                        mapping_table.append((unified_pos, unified_end, orig_start, orig_end, False))
                        unified_pos = unified_end
                
                # Write the unified chunk
                concat_audio = np.concatenate(audio_pieces)
                chunk_path = os.path.join(tmpdir, f"unified_{chunk_idx}.wav")
                sf.write(chunk_path, concat_audio, SR)
                
                # Transcribe
                segments, info = model.transcribe(
                    chunk_path,
                    beam_size=5,
                    word_timestamps=True,
                    language=language if language != "auto" else None,
                    initial_prompt="I, uh, er, um, like stuttering, mhm."
                )
                
                # Re-map timestamps from unified back to original timeline
                def unified_to_original(t_unified):
                    """Map a timestamp in unified audio back to original timeline."""
                    for u_start, u_end, o_start, o_end, is_sil in mapping_table:
                        if t_unified < u_end + 0.001:  # small epsilon for boundary
                            if is_sil:
                                # Place it proportionally within the original silence
                                ratio = (t_unified - u_start) / max(u_end - u_start, 0.001)
                                return o_start + ratio * (o_end - o_start)
                            else:
                                # Direct offset within speech segment
                                offset = t_unified - u_start
                                return o_start + offset
                    # Past the end: return last original end
                    return mapping_table[-1][3] if mapping_table else t_unified
                
                for segment in segments:
                    full_text.append(segment.text.strip())
                    if segment.words:
                        for w in segment.words:
                            orig_start = unified_to_original(w.start)
                            orig_end = unified_to_original(w.end)
                            all_words.append({
                                "word": w.word,
                                "start": orig_start,
                                "end": orig_end
                            })
                
                os.remove(chunk_path)
        
        del model
        del full_audio
        if device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            
        return " ".join(full_text), all_words
        
    except Exception as e:
        logger.error(f"Unified-Stream CrisperWhisper error: {e}")
        import traceback
        traceback.print_exc()
        return "", []

def _run_whisperx_asr(audio_path: str, model_name: str, device: str, silence_intervals: list[tuple[float, float]] = None):
    try:
        import whisperx
    except ImportError as e:
        logger.warning(f"whisperx not found: {e}. Falling back to Qwen.")
        return _run_qwen_asr(audio_path, "Qwen/Qwen3-ASR-1.7B", device, silence_intervals)

    logger.info(f"Using WhisperX with model: {model_name} on {device}")
    
    try:
        model = whisperx.load_model(model_name, device, compute_type="int8")
        audio = whisperx.load_audio(audio_path)
        batch_size = 16 if device == "cuda" else 4
        
        logger.info("Transcribing audio with WhisperX...")
        result = model.transcribe(audio, batch_size=batch_size)
    except Exception as e:
        logger.error(f"Failed to transcribe with WhisperX: {e}")
        return "", []

    try:
        logger.info("Aligning transcription for precise timestamps...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    except Exception as e:
        logger.error(f"Failed to align with WhisperX: {e}")

    full_text = []
    all_words = []
    
    for segment in result.get("segments", []):
        full_text.append(segment.get("text", "").strip())
        if "words" in segment:
            for w in segment["words"]:
                if "start" in w and "end" in w:
                    all_words.append({
                        "word": w["word"],
                        "start": w["start"],
                        "end": w["end"]
                    })

    if device == "cuda":
        import torch
        import gc
        del model
        try:
            del model_a
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        
    return " ".join(full_text), all_words

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
