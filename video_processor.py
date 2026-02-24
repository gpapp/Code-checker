#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import json
import logging
import numpy as np
import librosa
import lxml.etree as ET
from typing import List, Tuple, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Process multiple videos based on audio streams for synchronized cutting and Kdenlive project generation.")
    parser.add_argument("inputs", nargs="+", help="Input MKV video files")
    parser.add_argument("--working-dir", help="Working directory (default: subfolder in input directory)")
    parser.add_argument("--silence-threshold", type=float, default=-30.0, help="Silence threshold in dB (default: -30.0)")
    parser.add_argument("--silence-duration", type=float, default=2.0, help="Minimum silence duration in seconds for cutting (default: 2.0)")
    parser.add_argument("--spike-duration", type=float, default=0.2, help="Maximum duration in seconds for a spike to be silenced (default: 0.2)")
    parser.add_argument("--overlap-duration", type=float, default=5.0, help="Minimum duration in seconds for overlapping talk to be marked (default: 5.0)")
    parser.add_argument("--model-name", default="nvidia/parakeet-tdt-0.6b-v3", help="NeMo ASR model name (default: nvidia/parakeet-tdt-0.6b-v3)")
    parser.add_argument("--filler-words", default="er,ő", help="Comma-separated filler words to cut (default: er,ő)")
    parser.add_argument("--output-prefix", default="processed_", help="Prefix for output video files")

    return parser.parse_args()

def get_working_dir(inputs: List[str], working_dir: str = None) -> str:
    if working_dir:
        return working_dir
    # Use the directory of the first input file
    base_dir = os.path.dirname(os.path.abspath(inputs[0]))
    target_dir = os.path.join(base_dir, "video_processing_work")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return target_dir

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

    # Perform VAD
    # Marblenet returns frame-level probabilities
    probs = vad_model.transcribe([audio_path], batch_size=1)

    # Post-process probabilities into segments
    # Default parameters for Marblenet: 0.02s window
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

    # Transcribe with word timestamps if supported
    res = model.transcribe([audio_path], return_hypotheses=True)
    if not res or len(res[0]) == 0:
        return "", []

    hyp = res[0][0]
    words = []

    # Extract timestamps from hypothesis
    # NeMo models vary in where they store timestamps. TDT models usually have word_timestamps or timestep.
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

def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merges overlapping or adjacent intervals."""
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for curr_start, curr_end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if curr_start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append((curr_start, curr_end))
    return merged

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

def get_video_duration(video_path: str) -> float:
    """Returns the duration of the video file in seconds."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip()) if result.returncode == 0 else 0.0

def calculate_keep_segments(cut_segments: List[Tuple[float, float]], total_duration: float) -> List[Tuple[float, float]]:
    """Inverts cut segments to find segments to keep."""
    keep = []
    last_end = 0.0
    for start, end in sorted(cut_segments):
        if start > last_end:
            keep.append((last_end, start))
        last_end = max(last_end, end)
    if last_end < total_duration:
        keep.append((last_end, total_duration))
    return keep

def adjust_timestamps(segments: List[Tuple[float, float]], keep_segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Adjusts timestamps from original timeline to the cut timeline."""
    adjusted = []
    for start, end in segments:
        new_start = None
        new_end = None
        current_new_time = 0.0
        for ks, ke in sorted(keep_segments):
            duration = ke - ks
            if new_start is None:
                if ks <= start < ke:
                    new_start = current_new_time + (start - ks)
                elif start < ks:
                    new_start = current_new_time
            if new_end is None:
                if ks <= end < ke:
                    new_end = current_new_time + (end - ks)
                elif end < ks and new_start is not None:
                    new_end = current_new_time
            current_new_time += duration
        if new_start is not None:
            if new_end is None: new_end = current_new_time
            adjusted.append((new_start, new_end))
    return adjusted

def process_video(input_path: str, output_path: str, keep_segments: List[Tuple[float, float]], stream_spikes: List[List[Tuple[float, float]]]):
    """Cuts the video and mutes spikes using FFmpeg complex filters."""
    if not keep_segments:
        return
    v_select = "+".join([f"between(t,{s},{e})" for s, e in keep_segments])
    filter_complex = [f"[0:v]select='{v_select}',setpts=N/FRAME_RATE/TB[v]"]
    audio_outputs = []
    for i, spikes in enumerate(stream_spikes):
        a_select = "+".join([f"between(t,{s},{e})" for s, e in keep_segments])
        af = f"aselect='{a_select}',asetpts=N/SR/TB"
        if spikes:
            mute_expr = "+".join([f"between(t,{ss},{se})" for ss, se in spikes])
            af += f",volume=enable='{mute_expr}':volume=0"
        filter_complex.append(f"[0:a:{i}]{af}[a{i}]")
        audio_outputs.append(f"[a{i}]")
    cmd = ["ffmpeg", "-i", input_path, "-filter_complex", ";".join(filter_complex), "-map", "[v]"]
    for ao in audio_outputs: cmd.extend(["-map", ao])
    cmd.extend(["-c:v", "libx264", "-c:a", "aac", "-y", output_path])
    subprocess.run(cmd, capture_output=True, text=True)

def generate_kdenlive_project(video_files: List[str], output_path: str, overlaps: List[Tuple[float, float]], repetitions: List[Tuple[float, float]]):
    """Generates a .kdenlive XML file."""
    root = ET.Element("mlt", version="7.13", producer="main_bin")
    ET.SubElement(root, "profile", description="HD 1080p 25fps", frame_rate_num="25", frame_rate_den="1", width="1920", height="1080", progressive="1", sample_aspect_num="1", sample_aspect_den="1", display_aspect_num="16", display_aspect_den="9", colorspace="709")
    for i, f in enumerate(video_files):
        p = ET.SubElement(root, "producer", id=f"producer_{i}", resource=os.path.abspath(f))
        ET.SubElement(p, "property", name="kdenlive:clipname").text = os.path.basename(f)
    bin_playlist = ET.SubElement(root, "playlist", id="main_bin")
    for i in range(len(video_files)): ET.SubElement(bin_playlist, "entry", producer=f"producer_{i}")
    tractor = ET.SubElement(root, "tractor", id="main_tractor", global_feed="1")
    multitrack = ET.SubElement(tractor, "multitrack")
    for i, f in enumerate(video_files):
        playlist = ET.SubElement(multitrack, "playlist", id=f"video_track_{i}")
        ET.SubElement(playlist, "entry", producer=f"producer_{i}")
    kdenlive_root = ET.SubElement(root, "kdenlive_project")
    for s, e in overlaps: ET.SubElement(kdenlive_root, "guide", time=str(s), comment="Overlap", type="1")
    for s, e in repetitions: ET.SubElement(kdenlive_root, "guide", time=str(s), comment="Repetition", type="2")
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True, pretty_print=True)

def format_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def generate_ass_file(words: List[Dict], output_path: str):
    header = "[Script Info]\nScriptType: v4.00+\nPlayResX: 384\nPlayResY: 288\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,16,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for w in words:
            f.write(f"Dialogue: 0,{format_ass_time(w['start'])},{format_ass_time(w['end'])},Default,,0,0,0,,{w['word']}\n")

def main():
    args = parse_args()
    working_dir = get_working_dir(args.inputs, args.working_dir)
    video_to_audio_map = {v: extract_audio_streams(v, working_dir) for v in args.inputs}
    all_audio_files = [f for files in video_to_audio_map.values() for f in files]

    stream_markers = {}
    for af in all_audio_files:
        silence, spikes = detect_silence_and_spikes(af, args.silence_threshold, args.silence_duration, args.spike_duration)
        stream_markers[af] = {"silence": silence, "spikes": spikes}

    cutting_segments = find_global_silence(stream_markers, args.silence_duration)

    asr_results = {}
    for af in all_audio_files:
        text, words = run_asr(af, args.model_name)
        asr_results[af] = {"text": text, "words": words}
        cutting_segments.extend(find_fillers(words, args.filler_words.split(",")))

    cutting_segments = merge_intervals(cutting_segments)
    total_dur = get_video_duration(args.inputs[0])
    keep_segments = calculate_keep_segments(cutting_segments, total_dur)

    speech_intervals = {af: run_vad(af) for af in all_audio_files}
    overlap_segments = find_overlaps(speech_intervals, args.overlap_duration)
    repetition_segments = [rep for af in all_audio_files for rep in find_repetitions(af)]

    output_files = []
    for video in args.inputs:
        out = os.path.join(os.path.dirname(video), args.output_prefix + os.path.basename(video))
        process_video(video, out, keep_segments, [stream_markers[af]["spikes"] for af in video_to_audio_map[video]])
        output_files.append(out)

    new_overlaps = adjust_timestamps(overlap_segments, keep_segments)
    new_reps = adjust_timestamps(repetition_segments, keep_segments)
    generate_kdenlive_project(output_files, os.path.join(working_dir, "project.kdenlive"), new_overlaps, new_reps)

    for af, data in asr_results.items():
        if data["words"]:
            adj_words = []
            for w in data["words"]:
                adj = adjust_timestamps([(w["start"], w["end"])], keep_segments)
                if adj: adj_words.append({"word": w["word"], "start": adj[0][0], "end": adj[0][1]})
            generate_ass_file(adj_words, os.path.join(working_dir, os.path.splitext(os.path.basename(af))[0] + ".ass"))

if __name__ == "__main__":
    main()
