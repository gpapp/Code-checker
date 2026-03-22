#!/usr/bin/env python3
import argparse
import os
import logging
import json
from tqdm import tqdm

from audio_utils import (
    extract_audio_streams,
    detect_silence_and_spikes,
    find_global_silence,
    find_repetitions,
    get_video_duration,
    get_video_fps,
    has_video_stream
)
from nemo_processing import run_vad, run_asr, find_fillers, find_overlaps
from interval_utils import merge_intervals, calculate_keep_segments, adjust_timestamps, compress_global_silence
from exporter import process_video, generate_kdenlive_project, generate_ass_file, generate_srt_file

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Process multiple videos based on audio streams for synchronized cutting and Kdenlive project generation.")
    parser.add_argument("inputs", nargs="+", help="Input MKV/MP3/M4A/etc. files")
    parser.add_argument("--working-dir", help="Working directory (default: subfolder in input directory)")
    parser.add_argument("--silence-threshold", type=float, default=-30.0, help="Silence threshold in dB (default: -30.0)")
    parser.add_argument("--silence-duration", type=float, default=2.0, help="Minimum silence duration in seconds for cutting (default: 2.0)")
    parser.add_argument("--spike-duration", type=float, default=0.2, help="Maximum duration in seconds for a spike to be silenced (default: 0.2)")
    parser.add_argument("--overlap-duration", type=float, default=5.0, help="Minimum duration in seconds for overlapping talk to be marked (default: 5.0)")
    parser.add_argument("--model-name", default="nyrahealth/CrisperWhisper", help="ASR model name (default: nyrahealth/CrisperWhisper for WhisperX, Qwen/Qwen3-ASR-1.7B for Qwen)")
    parser.add_argument("--language", default="auto", help="Language code (e.g. 'hu', 'en'). Default is 'auto' for automatic detection.")
    parser.add_argument("--filler-words", default="[UH],[UM],-hm,-hm.,-hmm,", help="Comma-separated filler words to cut (default: er,ő)")
    parser.add_argument("--output-prefix", default="processed_", help="Prefix for output video files")
    parser.add_argument("--render", action="store_true", help="Render the processed videos into new files (slow and space consuming). Default: virtual cut in Kdenlive only.")
    parser.add_argument("--video-offsets", default="", help="Comma-separated list of +/- second offsets for video tracks (e.g. 0.5,-0.2,0)")

    return parser.parse_args()

def get_working_dir(inputs: list[str], working_dir: str = None) -> str:
    if working_dir:
        return os.path.abspath(working_dir)
    # Place working dir in the same directory as the first input
    first_input_abs = os.path.abspath(inputs[0])
    parent_dir = os.path.dirname(first_input_abs)
    target_dir = os.path.join(parent_dir, "video_processing_work")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return target_dir

def main():
    args = parse_args()
    
    # Process inputs: if directory, scan for supported files. Normalize to absolute paths.
    final_inputs = []
    supported_extensions = {".mkv", ".mp4", ".avi", ".mp3", ".wav", ".m4a"}
    
    for item in tqdm(args.inputs, desc="Scanning inputs"):
        # Strip potential literal quotes (common shell boundary issue on Windows)
        item = item.strip("'").strip('"')
        abs_item = os.path.abspath(item)
        if not os.path.exists(abs_item):
            logger.warning(f"Input path does not exist: {abs_item}")
            continue
            
        if os.path.isdir(abs_item):
            files = os.listdir(abs_item)
            for f in files:
                if os.path.splitext(f)[1].lower() in supported_extensions:
                    final_inputs.append(os.path.join(abs_item, f))
        elif os.path.isfile(abs_item):
            final_inputs.append(abs_item)
        else:
            logger.warning(f"Input item is neither file nor directory: {abs_item}")
            
    if not final_inputs:
        logger.error("No valid input files found.")
        return

    working_dir = get_working_dir(final_inputs, args.working_dir)

    # Parse video offsets
    offsets = [0.0] * len(final_inputs)
    if args.video_offsets:
        parsed_vals = [x.strip() for x in args.video_offsets.split(",")]
        if len(parsed_vals) == 1:
            # Apply same offset to all
            val = float(parsed_vals[0])
            offsets = [val] * len(final_inputs)
        else:
            for i, val in enumerate(parsed_vals):
                if i < len(offsets):
                    offsets[i] = float(val)

    fps = get_video_fps(final_inputs[0])

    # Extract audio streams (normalization is now always enabled)
    logger.info("Step 1/7: Extracting audio streams...")
    
    # If there are standalone audio files (MP3, WAV, M4A), ignore on-camera audio from video files
    audio_exts = {".mp3", ".wav", ".m4a"}
    has_external_audio = any(os.path.splitext(v)[1].lower() in audio_exts for v in final_inputs)
    
    if has_external_audio:
        logger.info("Standalone audio detected. On-camera video audio will be ignored for analysis.")

    video_to_audio_map = {}
    for v in tqdm(final_inputs, desc="Extracting"):
        if has_external_audio and has_video_stream(v):
            video_to_audio_map[v] = []
        else:
            video_to_audio_map[v] = extract_audio_streams(v, working_dir)
    all_audio_files = [f for files in video_to_audio_map.values() for f in files]

    logger.info("Step 2/7: Detecting silence and spikes...")
    stream_markers = {}
    for af in tqdm(all_audio_files, desc="Analyzing Audio"):
        cache_path = af + ".markers.json"
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                silence = [tuple(x) for x in data["silence"]]
                spikes = [tuple(x) for x in data["spikes"]]
        else:
            silence, spikes = detect_silence_and_spikes(af, args.silence_threshold, args.silence_duration, args.spike_duration)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"silence": silence, "spikes": spikes}, f, ensure_ascii=False)
        
        stream_markers[af] = {"silence": silence, "spikes": spikes}

    logger.info("Step 3/7: Running ASR and filler detection...")
    cutting_segments = []
    asr_results = {}
    for af in tqdm(all_audio_files, desc="ASR Processing"):
        cache_path = af + ".asr.json"
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                text = data["text"]
                words = data["words"]
        else:
            text, words = run_asr(af, args.model_name, silence_intervals=stream_markers[af]["silence"], language=args.language)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"text": text, "words": words}, f, ensure_ascii=False)
        
        asr_results[af] = {"text": text, "words": words}
        fillers = find_fillers(words, args.filler_words.split(","))
        cutting_segments.extend(fillers)

    # Compress global silence: instead of cutting it entirely, keep a compressed portion
    # Rules: >1s truncated to 1s, then >0.2s compressed to half
    global_silence = find_global_silence(stream_markers, args.silence_duration)
    silence_excess_cuts = compress_global_silence(global_silence)
    cutting_segments.extend(silence_excess_cuts)

    # Cut detected spikes (short pops/clicks) from the timeline
    for af in all_audio_files:
        cutting_segments.extend(stream_markers[af]["spikes"])

    cutting_segments = merge_intervals(cutting_segments)
    
    # Use max duration of all inputs as total duration
    total_dur = max([get_video_duration(v) for v in final_inputs])
    keep_segments = calculate_keep_segments(cutting_segments, total_dur)

    logger.info("Step 4/7: VAD and Overlap detection...")
    speech_intervals = {}
    for af in tqdm(all_audio_files, desc="VAD"):
        cache_path = af + ".vad.json"
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            with open(cache_path, "r", encoding="utf-8") as f:
                intervals = [tuple(x) for x in json.load(f)]
        else:
            intervals = run_vad(af)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(intervals, f, ensure_ascii=False)
        speech_intervals[af] = intervals
    
    overlap_segments = find_overlaps(speech_intervals, args.overlap_duration)

    logger.info("Step 5/7: Repetition detection...")
    repetition_segments = []
    for af in tqdm(all_audio_files, desc="Repetitions"):
        cache_path = af + ".reps.json"
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            with open(cache_path, "r", encoding="utf-8") as f:
                reps = [tuple(x) for x in json.load(f)]
        else:
            reps = find_repetitions(af)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(reps, f, ensure_ascii=False)
        repetition_segments.extend(reps)

    output_files = []
    # Output rendered files to parent directory of work dir
    output_base_dir = os.path.dirname(working_dir)

    if args.render:
        logger.info("Step 6/7: Rendering processed videos...")
        for i, video in enumerate(tqdm(final_inputs, desc="Rendering")):
            out = os.path.join(output_base_dir, args.output_prefix + os.path.basename(video))
            process_video(
                video, 
                out, 
                keep_segments, 
                [stream_markers[af]["spikes"] for af in video_to_audio_map[video]], 
                fps=fps,
                video_offset=offsets[i]
            )
            output_files.append(out)
    else:
        logger.info("Step 6/7: Skipping render (virtual cut mode)...")
        output_files = final_inputs

    logger.info("Step 7/7: Exporting project files...")
    
    # 1. Generate Subtitles (ASS/SRT) adjusted for the cut timeline
    ass_files = []
    for video_path in final_inputs:
        af_list = video_to_audio_map.get(video_path, [])
        if not af_list: continue
        af = af_list[0]
        if af not in asr_results: continue
        
        words = asr_results[af]["words"]
        word_segments = [(w["start"], w["end"]) for w in words]
        adj_segments = adjust_timestamps(word_segments, keep_segments)
        
        adj_words = []
        for i, (new_s, new_e) in enumerate(adj_segments):
            if i < len(words):
                adj_words.append({
                    "word": words[i]["word"],
                    "start": new_s,
                    "end": new_e
                })
        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        ass_path = os.path.join(output_base_dir, base_name + ".ass")
        srt_path = os.path.join(output_base_dir, base_name + ".srt")
        
        generate_ass_file(adj_words, ass_path)
        generate_srt_file(adj_words, srt_path)
        ass_files.append(ass_path)
        
    # 2. Adjust Overlaps and Repetitions
    adj_overlaps = adjust_timestamps(overlap_segments, keep_segments)
    adj_reps = adjust_timestamps(repetition_segments, keep_segments)

    # 3. Final Kdenlive Project Generation
    kdenlive_path = os.path.join(output_base_dir, "project.kdenlive")
    stream_spikes_map = {v: [stream_markers[af]["spikes"] for af in video_to_audio_map.get(v, []) if af in stream_markers] for v in final_inputs}
    
    # Collect source-aligned words for Kdenlive's internal speech view
    source_asr_words = []
    for v in final_inputs:
        af_list = video_to_audio_map.get(v, [])
        if af_list and af_list[0] in asr_results:
            source_asr_words.append(asr_results[af_list[0]].get("words", []))
        else:
            source_asr_words.append([])

    generate_kdenlive_project(
        output_files, 
        kdenlive_path, 
        keep_segments, 
        stream_spikes_map, 
        adj_overlaps, 
        adj_reps, 
        fps=fps, 
        is_rendered=args.render,
        video_offsets=offsets,
        ass_paths=ass_files,
        asr_words=source_asr_words,
        stream_markers_global=stream_markers,
        video_to_audio_map=video_to_audio_map
    )

if __name__ == "__main__":
    main()
