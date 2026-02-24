import os
import subprocess
import json
import logging
import numpy as np
import librosa
from typing import List, Tuple, Dict
from interval_utils import merge_intervals

logger = logging.getLogger(__name__)

def get_audio_streams(video_path: str) -> int:
    """Returns the number of audio streams in the video file."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=index", "-of", "json", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Failed to probe {video_path}: {result.stderr}")
        return 0
    data = json.loads(result.stdout)
    return len(data.get("streams", []))

def extract_audio_streams(video_path: str, working_dir: str) -> List[str]:
    """Extracts all audio streams to WAV files and returns their paths."""
    num_streams = get_audio_streams(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    extracted_files = []

    for i in range(num_streams):
        output_path = os.path.join(working_dir, f"{base_name}_a{i}.wav")
        cmd = [
            "ffmpeg", "-i", video_path, "-map", f"0:a:{i}",
            "-ac", "1", "-ar", "16000", "-y", output_path
        ]
        logger.info(f"Extracting stream {i} from {video_path} to {output_path}")
        subprocess.run(cmd, check=True, capture_output=True)
        extracted_files.append(output_path)

    return extracted_files

def detect_silence_and_spikes(audio_path: str, threshold_db: float, min_silence_len: float, max_spike_len: float):
    """Detects silent intervals and short spikes in an audio file."""
    y, sr = librosa.load(audio_path, sr=16000)
    peak = 20 * np.log10(np.max(np.abs(y)) + 1e-9)
    top_db = peak - threshold_db
    if top_db < 0: top_db = 0

    non_silent_intervals = librosa.effects.split(y, top_db=top_db) / sr
    duration = len(y) / sr

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

def find_global_silence(stream_markers: Dict[str, Dict], min_duration: float) -> List[Tuple[float, float]]:
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

def find_repetitions(audio_path: str, window_size: float = 2.0, step_size: float = 1.0, threshold: float = 0.9) -> List[Tuple[float, float]]:
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
        window = mfcc[i:i+window_frames]
        embeddings.append(np.mean(window, axis=0))
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
    """Returns the duration of the video file in seconds."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip()) if result.returncode == 0 else 0.0
