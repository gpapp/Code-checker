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

def get_audio_streams(video_path: str) -> int:
    """Returns the number of audio streams in the file."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=index", "-of", "json", video_path
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return len(data.get("streams", []))

def extract_audio_streams(file_path: str, working_dir: str, normalize: bool = True) -> list[str]:
    """Extracts all audio streams to WAV files and returns their paths."""
    file_path = os.path.abspath(file_path)
    working_dir = os.path.abspath(working_dir)
    num_streams = get_audio_streams(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    extracted_files = []

    for i in range(num_streams):
        output_path = os.path.join(working_dir, f"{base_name}_a{i}.wav")
        
        # Check if file already exists and has a valid size
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            logger.info(f"Using existing audio stream: {output_path}")
            extracted_files.append(output_path)
            continue

        # 1. Fast raw extraction to a temporary file (no slow filters here)
        temp_raw = os.path.join(working_dir, f"{base_name}_temp_a{i}.wav")
        extract_cmd = [
            "ffmpeg", "-i", file_path, "-map", f"0:a:{i}",
            "-ac", "1", "-ar", "16000", "-y", temp_raw
        ]
        
        logger.info(f"Extracting raw stream {i} from {file_path}...")
        subprocess.run(extract_cmd, check=True, capture_output=True)
        
        try:
            # 2. Fast in-memory processing with librosa (fulfills speed requirement)
            logger.info(f"Refining audio with librosa: {output_path}")
            y, sr = librosa.load(temp_raw, sr=16000)
            
            if normalize:
                # Fast Peak Normalization (replaces slow loudnorm)
                # Target peak level of -1dB (approx 0.89)
                max_val = np.max(np.abs(y))
                if max_val > 1e-6:
                    y = y * (0.89 / max_val)
                
                # NOTE: We no longer apply a hard noise gate here because it causes ASR gaps.
                # Silence detection handles noise thresholds internally.
            
            # Save final processed file
            sf.write(output_path, y, sr, subtype='PCM_16')
            extracted_files.append(output_path)
            
        except Exception as e:
            logger.error(f"Failed to process stream {i} with librosa: {e}")
            # Fallback: if librosa fails, move the raw file to output_path
            if os.path.exists(output_path): os.remove(output_path)
            os.rename(temp_raw, output_path)
            extracted_files.append(output_path)
        finally:
            if os.path.exists(temp_raw):
                try: os.remove(temp_raw)
                except: pass

    return extracted_files

def detect_silence_and_spikes(audio_path: str, threshold_db: float, min_silence_len: float, max_spike_len: float):
    """Detects silent intervals and short spikes in an audio file."""
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Apply a localized noise gate in-memory to improve silence detection accuracy
    # without affecting the ASR/Whisper source file on disk.
    # -35dB (0.017) is a safe threshold for normalized audio.
    y_gated = y.copy()
    y_gated[np.abs(y_gated) < 0.017] = 0
    
    peak = 20 * np.log10(np.max(np.abs(y_gated)) + 1e-9)
    top_db = peak - threshold_db
    if top_db < 0: top_db = 0

    non_silent_intervals = librosa.effects.split(y_gated, top_db=top_db) / sr
    duration = len(y_gated) / sr

    silence_intervals = []
    last_end = 0.0
    for start, end in non_silent_intervals:
        if start > last_end:
            silence_intervals.append((last_end, start))
        last_end = end
    if last_end < duration:
        silence_intervals.append((last_end, duration))

    spikes = []
    for start, end in non_silent_intervals:
        if (end - start) < max_spike_len:
            spikes.append((start, end))

    return silence_intervals, spikes

def find_global_silence(stream_markers: dict[str, dict], min_duration: float) -> list[tuple[float, float]]:
    """Finds intervals where all streams are silent for at least min_duration."""
    if not stream_markers:
        return []

    streams = list(stream_markers.keys())
    global_silence = stream_markers[streams[0]]["silence"]

    for audio_file in streams[1:]:
        new_global_silence = []
        stream_silence = stream_markers[audio_file]["silence"]
        for s1_start, s1_end in global_silence:
            for s2_start, s2_end in stream_silence:
                start = max(s1_start, s2_start)
                end = min(s1_end, s2_end)
                if start < end:
                    new_global_silence.append((start, end))
        global_silence = new_global_silence

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
