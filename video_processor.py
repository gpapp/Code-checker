#!/usr/bin/env python3
import argparse
import os
import logging
from typing import List

from audio_utils import (
    extract_audio_streams,
    detect_silence_and_spikes,
    find_global_silence,
    find_repetitions,
    get_video_duration
)
from nemo_processing import run_vad, run_asr, find_fillers, find_overlaps
from interval_utils import merge_intervals, calculate_keep_segments, adjust_timestamps
from exporter import process_video, generate_kdenlive_project, generate_ass_file

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
    base_dir = os.path.dirname(os.path.abspath(inputs[0]))
    target_dir = os.path.join(base_dir, "video_processing_work")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return target_dir

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
        logger.info(f"Running ASR on {af}...")
        text, words = run_asr(af, args.model_name)
        asr_results[af] = {"text": text, "words": words}
        fillers = find_fillers(words, args.filler_words.split(","))
        cutting_segments.extend(fillers)

    cutting_segments = merge_intervals(cutting_segments)
    total_dur = get_video_duration(args.inputs[0])
    keep_segments = calculate_keep_segments(cutting_segments, total_dur)

    speech_intervals = {af: run_vad(af) for af in all_audio_files}
    overlap_segments = find_overlaps(speech_intervals, args.overlap_duration)

    repetition_segments = []
    for af in all_audio_files:
        repetition_segments.extend(find_repetitions(af))

    output_files = []
    for video in args.inputs:
        out = os.path.join(os.path.dirname(video), args.output_prefix + os.path.basename(video))
        logger.info(f"Processing video {video} to {out}")
        process_video(video, out, keep_segments, [stream_markers[af]["spikes"] for af in video_to_audio_map[video]])
        output_files.append(out)

    new_overlaps = adjust_timestamps(overlap_segments, keep_segments)
    new_reps = adjust_timestamps(repetition_segments, keep_segments)

    kdenlive_path = os.path.join(working_dir, "project.kdenlive")
    generate_kdenlive_project(output_files, kdenlive_path, new_overlaps, new_reps)

    for af, data in asr_results.items():
        if data["words"]:
            adj_words = []
            for w in data["words"]:
                adj = adjust_timestamps([(w["start"], w["end"])], keep_segments)
                if adj:
                    adj_words.append({"word": w["word"], "start": adj[0][0], "end": adj[0][1]})
            ass_path = os.path.join(working_dir, os.path.splitext(os.path.basename(af))[0] + ".ass")
            generate_ass_file(adj_words, ass_path)

if __name__ == "__main__":
    main()
