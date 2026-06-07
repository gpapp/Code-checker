#!/usr/bin/env python3
import argparse
import os
import logging
import json
import glob
from tqdm import tqdm
from filler_processor import unload_crisper_model
from transcription_processor import unload_transcription_model

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from audio_utils import (
    process_streams_to_flac,
    detect_silence_and_spikes,
    find_global_silence,
    find_repetitions,
    get_track_name_from_path,
    get_video_duration,
    has_video_stream
)
from filler_processor import run_vad, find_overlaps
from interval_utils import merge_intervals, calculate_keep_segments, adjust_timestamps, compress_global_silence
from exporter import generate_kdenlive_project, generate_ass_file, generate_srt_file, render_processed_video, generate_kdenlive_from_rendered
from transcription_processor import process_transcription, get_full_language_name
from PodcastFillerLib import PodcastFillerLib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Process multiple videos based on audio streams for synchronized cutting and Kdenlive project generation.")
    parser.add_argument("inputs", nargs="+", help="Input MKV/MP3/M4A/etc. files")
    parser.add_argument("--working-dir", help="Working directory (default: subfolder in input directory)")
    parser.add_argument("--silence-threshold", type=float, default=-40.0, help="Silence threshold in dB (default: -40.0)")
    parser.add_argument("--silence-duration", type=float, default=2.0, help="Minimum silence duration in seconds for cutting (default: 2.0)")
    parser.add_argument("--spike-duration", type=float, default=0.2, help="Maximum duration in seconds for a spike to be silenced (default: 0.2)")
    parser.add_argument("--overlap-duration", type=float, default=5.0, help="Minimum duration in seconds for overlapping talk to be marked (default: 5.0)")
    parser.add_argument("--language", default="hu", help="Language code (e.g. 'hu', 'en'). Default is 'hu' for Hungarian.")
    parser.add_argument("--output-prefix", default="processed_", help="Prefix for output video files")
    parser.add_argument("--video-offsets", default="", help="Comma-separated list of +/- second offsets for video tracks (e.g. 0.5,-0.2,0)")
    parser.add_argument("--restart", action="store_true", help="If set, deletes temporary artifact files (*.json, *.markers.json, etc.) from the working directory before processing.")
    parser.add_argument("--noclear", action="store_true", help="If set, skips cleanup of temporary artifact files after processing.")
    parser.add_argument("--refine", action="store_true", help="If set, uses Gemma 4 via Ollama to clean up the transcription (Step 3c).")
    parser.add_argument("--no-asr", action="store_true", help="If set, skips all ASR/Whisper steps (filler detection and transcription).")
    parser.add_argument("--filler-threshold", type=float, default=0.9, help="Confidence threshold for filler detection (0.0 to 1.0). Default: 0.9")
    parser.add_argument("--filler-merge-gap", type=float, default=0.05, help="Maximum gap in seconds between fillers to merge them. Default: 0.05")

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
    supported_extensions = {".mkv", ".mov", ".mp4", ".avi", ".mp3", ".wav", ".m4a", ".flac"}
    
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

    if args.restart:
        run_cleanup(working_dir, args, include_outputs=True)
        # Ensure directory exists after cleanup
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

    output_base_dir = os.path.dirname(working_dir)
    grandparent_dir = os.path.dirname(output_base_dir)
    grandparent_name = os.path.basename(grandparent_dir)
    rendered_dir = os.path.join(grandparent_dir, "PROCESSED")
    
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

    # Process audio streams: compress and level to -14 LUFS, save as FLAC
    logger.info("Step 1/7: Processing audio streams to FLAC (-14 LUFS)...")
    
    audio_exts = {".mp3", ".wav", ".m4a", ".flac"}
    has_external_audio = any(os.path.splitext(v)[1].lower() in audio_exts for v in final_inputs)
    
    if has_external_audio:
        logger.info("Standalone audio detected. On-camera video audio will be ignored for analysis.")

    video_to_audio_map = {}
    audio_info_map = {}

    for v in tqdm(final_inputs, desc="Processing Audio"):
        is_audio_ext = os.path.splitext(v)[1].lower() in audio_exts
        if has_external_audio and has_video_stream(v) and not is_audio_ext:
            video_to_audio_map[v] = []
        else:
            processed = process_streams_to_flac(v, working_dir)
            video_to_audio_map[v] = [e["flac"] for e in processed]
            for e in processed:
                audio_info_map[e["flac"]] = {"original": v, "stream_idx": e["stream_idx"]}

    all_audio_files = [f for files in video_to_audio_map.values() for f in files]

    # Build cache key map: flac_path -> cache path in working_dir
    _ck = {}
    for af in all_audio_files:
        stem = os.path.splitext(os.path.basename(af))[0]
        _ck[af] = os.path.join(working_dir, stem)

    logger.info("Step 2/7: Detecting silence and spikes...")
    stream_markers = {}
    for af in tqdm(all_audio_files, desc="Analyzing Audio"):
        cache_path = _ck[af] + ".markers.json"
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                silence = [tuple(x) for x in data["silence"]]
                spikes = [tuple(x) for x in data["spikes"]]
        else:
            silence, spikes = detect_silence_and_spikes(af, args.silence_threshold, args.silence_duration, args.spike_duration)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"silence": silence, "spikes": spikes}, f, ensure_ascii=False)
        
        stream_markers[af] = {
            "silence": silence, 
            "spikes": spikes, 
            "duration": get_video_duration(af),
            "offset": offsets[final_inputs.index(audio_info_map[af]["original"])] if af in audio_info_map else 0.0
        }


    logger.info("Step 3/7: Running ASR and filler detection...")
    cutting_segments = []
    asr_results = {}
    transcription_results = {}
    # Whisper large-v3 used for both filler detection (Pass 1b) and final transcription (Pass 2)
    transcription_model = "large-v3"
    
    # 1. Pass 1: CNN model for acoustic filler detection (Fast)
    filler_lib = PodcastFillerLib()
    if not os.path.exists(filler_lib.model_path):
        logger.warning(f"PodcastFillerLib model {filler_lib.model_path} not found. Running setup.bat strongly recommended.")
        logger.info("Falling back to silence-only cutting (Step 1-2).")
    else:
        for af in tqdm(all_audio_files, desc="Pass 1: CNN Filler Detection"):
            # Check for CNN results cache
            cnn_cache = _ck[af] + ".filler_cnn.json"
            if os.path.exists(cnn_cache) and os.path.getsize(cnn_cache) > 0:
                with open(cnn_cache, "r", encoding="utf-8") as f:
                    fillers = [tuple(x) for x in json.load(f)]
            else:
                fillers = filler_lib.detect_fillers(af, threshold=args.filler_threshold, gap_sec=args.filler_merge_gap)
                with open(cnn_cache, "w", encoding="utf-8") as f:
                    json.dump(fillers, f, ensure_ascii=False)
            
            # Shift fillers to master timeline
            offset = stream_markers[af]["offset"]
            silences = stream_markers[af]["silence"]
            
            # Filter out fillers that fall entirely within silent intervals (false positives)
            def is_entirely_silent(start, end):
                for s_start, s_end in silences:
                    if s_start <= start + 0.05 and end - 0.05 <= s_end:
                        return True
                return False

            # Add fillers to markers instead of cutting them
            valid_fillers = [f for f in fillers if not is_entirely_silent(f[0], f[1])]
            stream_markers[af]["fillers"] = valid_fillers
            
            # shifted_fillers = [(s - offset, e - offset) for s, e in valid_fillers]
            # cutting_segments.extend(shifted_fillers) # No longer cutting fillers

    # Pass 1b: Whisper Filler Detection has been removed. Filler detection now relies solely on Step 1 (CNN).
    
    # Whisper model stays loaded; Pass 2 below handles files not covered above (edge case)
    # unload_crisper_model() — deliberately NOT called here; shared model reused in process_transcription

    # 2. Pass 2: Final Transcription — skipped for files already transcribed by Pass 1b
    if not args.no_asr:
        logger.info("Step 3b/7: Running Pass 2: Final Transcription (faster-whisper large-v3)...")
        for af in tqdm(all_audio_files, desc="Pass 2: Transcription"):
            transcription_cache = _ck[af] + ".asr.json"
    
            # Load from disk cache first
            if os.path.exists(transcription_cache) and os.path.getsize(transcription_cache) > 0:
                with open(transcription_cache, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    transcription_results[af] = {"text": data["text"], "words": data["words"]}
                continue
    
            # Reuse Pass 1b result if available (same Whisper run)
            if af in transcription_results:
                with open(transcription_cache, "w", encoding="utf-8") as f:
                    json.dump(transcription_results[af], f, ensure_ascii=False)
                logger.info(f"  (Pass 2 skipped for {os.path.basename(af)}: reusing Pass 1b result)")
                continue
    
            # Fallback: run Whisper large-v3 explicitly for this file
            c_text, c_words = process_transcription(
                af, transcription_model,
                silence_intervals=stream_markers[af]["silence"],
                language=args.language,
                cache_path=transcription_cache
            )
            transcription_results[af] = {"text": c_text, "words": c_words}
    
        # Free memory after Pass 2
        unload_transcription_model()
    else:
        logger.info("Step 3b/7: Skipping Pass 2 (Final Transcription) due to --no-asr.")

    # 3. Pass 3: Gemma Cleanup (Step 3c)
    if args.refine:
        logger.info("Step 3c/7: Running Pass 3: Gemma Cleanup...")
        from gemma_asr_service import refine_transcription_timed
        
        for af in tqdm(all_audio_files, desc="Pass 3: Cleanup"):
            # Check for refined cache
            refined_cache = _ck[af] + ".refined.json"
            if os.path.exists(refined_cache) and os.path.getsize(refined_cache) > 0:
                with open(refined_cache, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    r_text, r_words = data["text"], data["words"]
            else:
                # Use current transcription results as input
                if af not in transcription_results:
                    logger.warning(f"Skipping refinement for {os.path.basename(af)}: No ASR data available.")
                    continue
                
                raw_words = transcription_results[af]["words"]
                full_lang = get_full_language_name(args.language)
                r_text, r_words = refine_transcription_timed(raw_words, full_lang, cache_path=refined_cache)
            
            # Update results with refined version
            transcription_results[af] = {"text": r_text, "words": r_words}
    else:
        logger.info("Skipping Step 3c (Gemma Cleanup). Use --refine to enable.")

    # Use max duration of all inputs as total duration, adjusted by their offsets
    # Master timeline T = track_time - offset. So track end at duration_i is at master time duration_i - offset_i.
    total_duration = 0.0
    for i, v in enumerate(final_inputs):
        total_duration = max(total_duration, get_video_duration(v) - offsets[i])

    # 1. Collect and shift per-stream spikes in seconds
    cutting_segments = []
    for af, markers in stream_markers.items():
        offset = markers["offset"]
        for s, e in markers.get("spikes", []):
            cutting_segments.append((s - offset, e - offset))

    # 2. Find and compress global silence using time-based utilities
    global_silence = find_global_silence(stream_markers, 0.2, total_duration)
    global_silence_cuts = compress_global_silence(global_silence)
    cutting_segments.extend(global_silence_cuts)

    cutting_segments = merge_intervals(cutting_segments)
    keep_segments = calculate_keep_segments(cutting_segments, total_duration)

    logger.info("Step 4/7: VAD and Overlap detection...")
    speech_intervals = {}
    for af in tqdm(all_audio_files, desc="VAD"):
        cache_path = _ck[af] + ".vad.json"
        
        # Bypass NeMo VAD if we have accurate word timestamps from Pass 2 (Transcription)
        if af in transcription_results:
            words = transcription_results[af].get("words", [])
            intervals = []
            if words:
                curr_start = words[0]["start"]
                curr_end = words[0]["end"]
                for w in words[1:]:
                    if w["start"] - curr_end <= 0.5:
                        curr_end = max(curr_end, w["end"])
                    else:
                        intervals.append((curr_start, curr_end))
                        curr_start = w["start"]
                        curr_end = w["end"]
                intervals.append((curr_start, curr_end))
            
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(intervals, f, ensure_ascii=False)
            speech_intervals[af] = intervals
            continue
            
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            with open(cache_path, "r", encoding="utf-8") as f:
                intervals = [tuple(x) for x in json.load(f)]
        else:
            intervals = run_vad(af)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(intervals, f, ensure_ascii=False)
        speech_intervals[af] = intervals
    
    overlap_segments = find_overlaps(speech_intervals, args.overlap_duration)

    logger.info("Step 5/7: Repetition detection (IGNORED)...")
    repetition_segments = []
    # for af in tqdm(all_audio_files, desc="Repetitions"):
    #     cache_path = af + ".reps.json"
    #     if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
    #         with open(cache_path, "r", encoding="utf-8") as f:
    #             reps = [tuple(x) for x in json.load(f)]
    #     else:
    #         reps = find_repetitions(af)
    #         with open(cache_path, "w", encoding="utf-8") as f:
    #             json.dump(reps, f, ensure_ascii=False)
    #     repetition_segments.extend(reps)

    output_files = final_inputs

    logger.info("Step 6/7: Rendering processed video...")

    # Prepare audio files configuration for rendering
    audio_files_config = {}
    for i, out_v in enumerate(output_files):
        orig_v = final_inputs[i]
        af_list = video_to_audio_map.get(orig_v, [])
        sources = []
        for af in af_list:
            info = audio_info_map.get(af, {"original": af, "stream_idx": 0})
            sources.append({
                "original": af,
                "stream_idx": info["stream_idx"],
                "temp_path": af
            })
        audio_files_config[out_v] = sources

    rendered_files = render_processed_video(
        video_files=output_files,
        audio_files=audio_files_config,
        keep_segments=keep_segments,
        output_dir=rendered_dir,
        video_offsets=offsets,
    )

    logger.info("Step 7/7: Exporting project files...")
    
    # 1. Generate Subtitles (ASS/SRT) adjusted for the cut timeline
    ass_files = []
    for video_path in final_inputs:
        af_list = video_to_audio_map.get(video_path, [])
        if not af_list: continue
        af = af_list[0]
        if af not in transcription_results: continue
        
        words = transcription_results[af]["words"]
        word_segments = [(w["start"], w["end"]) for w in words]
        adj_segments = adjust_timestamps(word_segments, keep_segments)
        
        adj_words = []
        for i, (new_s, new_e) in enumerate(adj_segments):
            if i < len(words):
                if (new_e - new_s) > 0.05:
                    adj_words.append({
                        "word": words[i]["word"],
                        "start": new_s,
                        "end": new_e
                    })
        
        track_name = get_track_name_from_path(video_path)
        ass_path = os.path.join(output_base_dir, track_name + ".ass")
        srt_path = os.path.join(output_base_dir, track_name + ".srt")
        
        generate_ass_file(adj_words, ass_path)
        generate_srt_file(adj_words, srt_path)
        ass_files.append(ass_path)
        
    # 2. Kdenlive project referencing rendered processed files
    kdenlive_path = os.path.join(grandparent_dir, f"{grandparent_name}.kdenlive")

    generate_kdenlive_from_rendered(
        rendered_files=rendered_files,
        output_path=kdenlive_path,
        ass_paths=ass_files,
    )
    # Clean up temporary artifacts by default (use --noclear to skip)
    if not args.noclear:
        run_cleanup(working_dir, args, include_outputs=False)

def run_cleanup(working_dir: str, args, include_outputs: bool = False):
    """Deletes temporary artifact files from the working directory."""
    logger.info("Running cleanup routine: Deleting temporary artifact files.")

    # Define patterns for temporary files based on observed extensions
    temp_patterns = [
        "*.filler.json",
        "*.filler_cnn.json",
        "*.filler_whisper.json",
        "*.asr.json",
        "*.refined.json",
        "*.markers.json",
        "*.vad.json",
        "*.reps.json",
        "*.wav",
    ]

    cleaned_count = 0
    for pattern in temp_patterns:
        for file in glob.glob(working_dir + os.path.sep + pattern):
            try:
                os.remove(file)
                cleaned_count += 1
            except OSError as e:
                logger.warning(f"Failed to remove {file}: {e}")
                
    if include_outputs:
        output_base_dir = os.path.dirname(working_dir)
        grandparent = os.path.dirname(output_base_dir)
        output_patterns = [
            "*.srt",
            "*.ass",
            "*.kdenlive",
            f"{args.output_prefix}*"
        ]
        logger.info("Restart specified: Deleting potential output files.")
        # Clean from output_base_dir (subtitles)
        for pattern in output_patterns:
            for file in glob.glob(output_base_dir + os.path.sep + pattern):
                try:
                    os.remove(file)
                    cleaned_count += 1
                except OSError as e:
                    logger.warning(f"Failed to remove output file {file}: {e}")
        # Clean kdenlive from grandparent dir (moved one up)
        for file in glob.glob(grandparent + os.path.sep + "*.kdenlive"):
            try:
                os.remove(file)
                cleaned_count += 1
            except OSError as e:
                logger.warning(f"Failed to remove kdenlive file {file}: {e}")
        # Clean FLACs from working_dir (pre-processed audio)
        for file in glob.glob(working_dir + os.path.sep + "*.flac"):
            try:
                os.remove(file)
                cleaned_count += 1
            except OSError as e:
                logger.warning(f"Failed to remove FLAC {file}: {e}")
        # Clean rendered files from PROCESSED dir (grandparent level)
        rendered_dir = os.path.join(grandparent, "PROCESSED")
        if os.path.isdir(rendered_dir):
            for pattern in ("*_processed.mp4", "*_processed.flac"):
                for file in glob.glob(rendered_dir + os.path.sep + pattern):
                    try:
                        os.remove(file)
                        cleaned_count += 1
                    except OSError as e:
                        logger.warning(f"Failed to remove rendered file {file}: {e}")

    logger.info(f"Cleanup finished. Removed {cleaned_count} files/artifacts.")
    
    # Remove working directory if empty
    try:
        if os.path.exists(working_dir) and not os.listdir(working_dir):
            os.rmdir(working_dir)
            logger.info(f"Removed empty working directory: {working_dir}")
    except OSError as e:
        logger.warning(f"Failed to remove working directory {working_dir}: {e}")


if __name__ == "__main__":
    main()
