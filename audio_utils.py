import os
import subprocess
import json
import logging
import numpy as np
import librosa
from interval_utils import merge_intervals
import soundfile as sf

logger = logging.getLogger(__name__)

def has_video_stream(file_path: str) -> bool:
    """Returns True if the file contains at least one video stream."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v",
        "-show_entries", "stream=index", "-of", "json", file_path
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
        return len(data.get("streams", [])) > 0
    except Exception:
        return False

def get_video_stream_index(file_path: str) -> int:
    """Returns the absolute index of the first video stream."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=index", "-of", "json", file_path
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if streams:
            return int(streams[0]["index"])
    except Exception:
        pass
    return 0

def get_audio_stream_indices(file_path: str) -> list[int]:
    """Returns the absolute indices of all audio streams in the file."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=index", "-of", "json", file_path
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return [s["index"] for s in data.get("streams", [])]

def get_track_name_from_path(file_path: str) -> str:
    """Extracts a clean track name from a file path, stripping -- prefix convention."""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    if "--" in base_name:
        return base_name.split("--", 2)[2]
    return base_name

def process_streams_to_flac(file_path: str, output_dir: str) -> list[dict]:
    """Extracts all audio streams, compresses and levels to -14 LUFS, saves as FLAC.

    Uses two-pass loudnorm for accurate loudness normalization.
    Returns list of {flac, stream_idx} dicts.
    """
    file_path = os.path.abspath(file_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    stream_indices = get_audio_stream_indices(file_path)
    original_stem = os.path.splitext(os.path.basename(file_path))[0]
    flac_info = []

    for i, abs_idx in enumerate(stream_indices):
        output_path = os.path.join(output_dir, f"{original_stem}_a{i}.flac")

        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            logger.info(f"Using existing processed audio: {output_path}")
            flac_info.append({"flac": output_path, "stream_idx": abs_idx})
            continue

        logger.info(f"Pass 1/2: Measuring loudness for stream {i} from {os.path.basename(file_path)}...")
        measure_cmd = [
            "ffmpeg", "-i", file_path, "-map", f"0:a:{i}",
            "-ac", "1", "-ar", "16000",
            "-af", "loudnorm=I=-14:LRA=11:TP=-1.5:print_format=json",
            "-f", "null", "NUL"
        ]
        result = subprocess.run(measure_cmd, capture_output=True, text=True)

        # Parse measured values from loudnorm JSON output in stderr
        measured = {}
        for line in result.stderr.split("\n"):
            stripped = line.strip()
            if stripped.startswith("{"):
                try:
                    measured = json.loads(stripped)
                except json.JSONDecodeError:
                    pass

        measured_i = measured.get("input_i", "-14.0")
        measured_lra = measured.get("input_lra", "7.0")
        measured_tp = measured.get("input_tp", "-1.5")
        measured_thresh = measured.get("input_thresh", "-21.0")
        measured_offset = measured.get("target_offset", "0.0")

        logger.info(f"Pass 2/2: Applying compression and loudness normalization for stream {i}...")
        filter_chain = (
            f"loudnorm=I=-14:LRA=11:TP=-1.5:"
            f"measured_I={measured_i}:measured_LRA={measured_lra}:"
            f"measured_TP={measured_tp}:measured_thresh={measured_thresh}:"
            f"offset={measured_offset},"
            f"acompressor=ratio=4:attack=20:release=200:threshold=-24dB:makeup=3dB"
        )
        process_cmd = [
            "ffmpeg", "-i", file_path, "-map", f"0:a:{i}",
            "-ac", "1", "-ar", "16000",
            "-af", filter_chain,
            "-c:a", "flac",
            "-y", output_path
        ]
        subprocess.run(process_cmd, check=True, capture_output=True)

        flac_info.append({"flac": output_path, "stream_idx": abs_idx})

    return flac_info

def detect_silence_and_spikes(audio_path: str, threshold_db: float, min_silence_len: float, max_spike_len: float):
    """Detects silent intervals and short spikes in an audio file."""
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Apply a localized noise gate in-memory to improve silence detection accuracy
    # without affecting the ASR/Whisper source file on disk.
    # -45dB (0.0056) preserves quiet speech edges (fricatives, plosives, breath) while removing noise floor.
    y_gated = y.copy()
    y_gated[np.abs(y_gated) < 0.0056] = 0
    
    peak = 20 * np.log10(np.max(np.abs(y_gated)) + 1e-9)
    top_db = peak - threshold_db
    if top_db < 0: top_db = 0

    non_silent_intervals = librosa.effects.split(y_gated, top_db=top_db) / sr
    duration = len(y_gated) / sr

    # Separate spikes from real speech to avoid padding noise spikes
    spikes = []
    real_speech = []
    for start, end in non_silent_intervals:
        if (end - start) < max_spike_len:
            spikes.append((start, end))
        else:
            real_speech.append((start, end))

    # Apply padding only to real speech segments (attack/decay)
    padding = 0.3
    padded_speech = []
    for start, end in real_speech:
        padded_speech.append((max(0.0, start - padding), min(duration, end + padding)))
    
    # Final non-silent intervals for silence calculation
    from interval_utils import merge_intervals
    combined_non_silent = merge_intervals(padded_speech + spikes)

    silence_intervals = []
    last_end = 0.0
    for start, end in combined_non_silent:
        if start > last_end:
            silence_intervals.append((last_end, start))
        last_end = end
    if last_end < duration:
        silence_intervals.append((last_end, duration))

    spikes = []
    for start, end in non_silent_intervals:
        if (end - start) < max_spike_len:
            spikes.append((start, end))

    from interval_utils import merge_intervals, calculate_keep_segments

    return silence_intervals, spikes

def find_global_silence_f(stream_markers: dict[str, dict], min_duration_f: int, total_duration_f: int, fps: float) -> list[tuple[int, int]]:
    """Finds intervals where all streams are silent in the frame domain."""
    if not stream_markers:
        return []

    from interval_utils import calculate_keep_segments_f, merge_intervals_f
    
    all_speech_intervals_f = []
    for af, markers in stream_markers.items():
        dur_f = int(round(markers.get("duration", 0.0) * fps))
        silences_f = [(int(round(s * fps)), int(round(e * fps))) for s, e in markers.get("silence", [])]
        offset_f = int(round(markers.get("offset", 0.0) * fps))
        
        speech_f = calculate_keep_segments_f(silences_f, dur_f)
        # Shift in integer frame domain: no drift possible
        shifted_speech_f = [(s_f - offset_f, e_f - offset_f) for s_f, e_f in speech_f]
        all_speech_intervals_f.extend(shifted_speech_f)
    
    merged_speech_f = merge_intervals_f(all_speech_intervals_f)
    
    global_silence_f = []
    last_end_f = 0
    for s_f, e_f in merged_speech_f:
        if s_f > last_end_f:
            global_silence_f.append((last_end_f, s_f))
        last_end_f = max(last_end_f, e_f)
    
    if last_end_f < total_duration_f:
        global_silence_f.append((last_end_f, total_duration_f))

    return [s for s in global_silence_f if (s[1] - s[0]) >= min_duration_f]

def find_global_silence(stream_markers: dict[str, dict], min_duration: float, total_duration: float) -> list[tuple[float, float]]:
    """Finds intervals where all streams are silent for at least min_duration, accounting for track offsets."""
    if not stream_markers:
        return []

    all_speech_intervals = []
    for af, markers in stream_markers.items():
        dur = markers.get("duration", 0.0)
        silences = markers.get("silence", [])
        offset = markers.get("offset", 0.0)
        
        # Non-silent intervals for this track (track timeline)
        from interval_utils import calculate_keep_segments
        speech = calculate_keep_segments(silences, dur)
        
        # Shift speech to master timeline: T_master = T_track - offset
        shifted_speech = [(s - offset, e - offset) for s, e in speech]
        all_speech_intervals.extend(shifted_speech)
    
    # Union of all speech across all tracks in the master timeline
    merged_speech = merge_intervals(all_speech_intervals)
    
    # Global silence is the complement of merged_speech relative to total_duration
    global_silence = []
    last_end = 0.0
    for s, e in merged_speech:
        if s > last_end:
            global_silence.append((last_end, s))
        last_end = max(last_end, e)
    
    if last_end < total_duration:
        global_silence.append((last_end, total_duration))

    return [s for s in global_silence if (s[1] - s[0]) >= min_duration]

def find_repetitions(audio_path: str, window_size: float = 2.0, step_size: float = 1.0, threshold: float = 0.9) -> list[tuple[float, float]]:
    """Finds repeated audio segments using acoustic similarity."""
    try:
        y, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        logger.error(f"Failed to load {audio_path}: {e}")
        return []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T

    window_frames = int(window_size * sr / 512)
    step_frames = int(step_size * sr / 512)

    embeddings = []
    times = []
    for i in range(0, len(mfcc) - window_frames, step_frames):
        y_start = int(i * 512)
        y_end = int((i + window_frames) * 512)
        if y_end > len(y):
            break
            
        y_window = y[y_start:y_end]
        rms_energy = np.sqrt(np.mean(y_window**2))
        
        # Skip silent or extremely quiet frames which artificially inflate cosine similarity
        if rms_energy < 0.01:
            continue
            
        window = mfcc[i:i+window_frames]
        # Flatten preserves the temporal pattern; averaging destroys temporal structure
        embeddings.append(window.flatten())
        times.append(i * 512 / sr)

    if not embeddings:
        return []

    embeddings = np.array(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    embeddings = embeddings / norms

    sim_matrix = np.dot(embeddings, embeddings.T)
    repetitions = []
    num_windows = len(embeddings)
    for i in range(num_windows):
        for j in range(i + int(window_size / step_size) + 1, num_windows):
            if sim_matrix[i, j] > threshold:
                repetitions.append((times[i], times[i] + window_size))
                repetitions.append((times[j], times[j] + window_size))

    return merge_intervals(repetitions)

def get_video_duration(video_path: str) -> float:
    """Returns the duration of the file in seconds."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    val = result.stdout.strip()
    if val and val != "N/A":
        return float(val)
    # Fallback: try stream-level duration
    cmd2 = ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    result2 = subprocess.run(cmd2, check=True, capture_output=True, text=True)
    val2 = result2.stdout.strip()
    if val2 and val2 != "N/A":
        return float(val2)
    # Last resort: try video stream
    cmd3 = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    result3 = subprocess.run(cmd3, check=True, capture_output=True, text=True)
    val3 = result3.stdout.strip()
    if val3 and val3 != "N/A":
        return float(val3)
    return 0.0

def get_video_fps(video_path: str) -> float:
    """Returns the frame rate of the file. Defaults to 25.0 for audio-only."""
    if not has_video_stream(video_path):
        return 25.0
        
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    rate = result.stdout.strip()
    if "/" in rate:
        num, den = rate.split("/")
        return float(num) / float(den)
    try:
        return float(rate)
    except ValueError:
        return 25.0
