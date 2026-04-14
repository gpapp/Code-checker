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
    extract_audio_streams,
    detect_silence_and_spikes,
    find_global_silence,
    find_repetitions,
    get_video_duration,
    get_video_fps,
    has_video_stream
)
from filler_processor import run_vad, process_filler_detection_asr, find_fillers, find_overlaps
from interval_utils import merge_intervals, calculate_keep_segments, adjust_timestamps, compress_global_silence
from exporter import generate_kdenlive_project, generate_ass_file, generate_srt_file
from transcription_processor import process_transcription, get_full_language_name
from PodcastFillerLib import PodcastFillerLib

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
    parser.add_argument("--language", default="hu", help="Language code (e.g. 'hu', 'en'). Default is 'hu' for Hungarian.")
    parser.add_argument("--filler-words", default="[UH],[UM],[EH],[AH],-hm,-mhm,öö,őő,öhm,hmm,izé,hát,szóval,ugye,amúgy", help="Comma-separated filler words to cut (default: includes common Hungarian fillers)")
    parser.add_argument("--output-prefix", default="processed_", help="Prefix for output video files")
    parser.add_argument("--video-offsets", default="", help="Comma-separated list of +/- second offsets for video tracks (e.g. 0.5,-0.2,0)")
    parser.add_argument("--restart", action="store_true", help="If set, deletes temporary artifact files (*.json, *.markers.json, etc.) from the working directory before processing.")
    parser.add_argument("--clean", action="store_true", help="If set, deletes temporary artifact files (*.json, *.markers.json, etc.) from the working directory after processing.")
    parser.add_argument("--refine", action="store_true", help="If set, uses Gemma 4 via Ollama to clean up the transcription (Step 3c).")

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

    if args.restart:
        run_cleanup(working_dir, args)
    
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
            cnn_cache = af + ".filler_cnn.json"
            if os.path.exists(cnn_cache) and os.path.getsize(cnn_cache) > 0:
                with open(cnn_cache, "r", encoding="utf-8") as f:
                    fillers = [tuple(x) for x in json.load(f)]
            else:
                fillers = filler_lib.detect_fillers(af)
                with open(cnn_cache, "w", encoding="utf-8") as f:
                    json.dump(fillers, f, ensure_ascii=False)
            
            cutting_segments.extend(fillers)

    # 1b. Pass 1b: Whisper model for semantic filler detection (Medium)
    logger.info("Step 3a/7: Running Pass 1b: Whisper Filler Detection...")
    # Using large-v3 as in test_crisper.py for best Hungarian accuracy
    whisper_filler_model = "large-v3" 
    for af in tqdm(all_audio_files, desc="Pass 1b: Whisper Filler Detection"):
        # Check for Whisper results cache
        whisper_cache = af + ".filler_whisper.json"
        if os.path.exists(whisper_cache) and os.path.getsize(whisper_cache) > 0:
            with open(whisper_cache, "r", encoding="utf-8") as f:
                data = json.load(f)
                w_text, w_words = data["text"], data["words"]
        else:
            w_text, w_words = process_filler_detection_asr(af, whisper_filler_model, language=args.language)
            with open(whisper_cache, "w", encoding="utf-8") as f:
                json.dump({"text": w_text, "words": w_words}, f, ensure_ascii=False)
        
        # Identify fillers from Whisper words
        filler_list = args.filler_words.split(",")
        w_fillers = find_fillers(w_words, filler_list)
        logger.info(f"  Filler Detection: {len(w_fillers)} filler(s) found in {os.path.basename(af)}: "
                    f"{[(round(s,2), round(e,2)) for s, e in w_fillers]}")
        cutting_segments.extend(w_fillers)

        # Build a lookup set of filler (start, end) timestamps so we can strip them
        # from the transcription word list — the subtitle/Kdenlive output should show
        # only clean speech, not the filler tokens that are going to be cut anyway.
        filler_times = {(round(s, 3), round(e, 3)) for s, e in w_fillers}
        clean_words = [
            w for w in w_words
            if (round(w["start"], 3), round(w["end"], 3)) not in filler_times
        ]
        clean_text = " ".join(w["word"] for w in clean_words).strip()

        # Reuse the cleaned Whisper transcription as Pass 2 result — avoids running Whisper twice
        transcription_results[af] = {"text": clean_text, "words": clean_words}
    
    # Whisper model stays loaded; Pass 2 below handles files not covered above (edge case)
    # unload_crisper_model() — deliberately NOT called here; shared model reused in process_transcription

    # 2. Pass 2: Final Transcription — skipped for files already transcribed by Pass 1b
    logger.info("Step 3b/7: Running Pass 2: Final Transcription (faster-whisper large-v3)...")
    for af in tqdm(all_audio_files, desc="Pass 2: Transcription"):
        transcription_cache = af + ".asr.json"

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

    # 3. Pass 3: Gemma Cleanup (Step 3c)
    if args.refine:
        logger.info("Step 3c/7: Running Pass 3: Gemma Cleanup...")
        from gemma_asr_service import refine_transcription_timed
        
        for af in tqdm(all_audio_files, desc="Pass 3: Cleanup"):
            # Check for refined cache
            refined_cache = af + ".refined.json"
            if os.path.exists(refined_cache) and os.path.getsize(refined_cache) > 0:
                with open(refined_cache, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    r_text, r_words = data["text"], data["words"]
            else:
                # Use current transcription results as input
                raw_words = transcription_results[af]["words"]
                full_lang = get_full_language_name(args.language)
                r_text, r_words = refine_transcription_timed(raw_words, full_lang, cache_path=refined_cache)
            
            # Update results with refined version
            transcription_results[af] = {"text": r_text, "words": r_words}
    else:
        logger.info("Skipping Step 3c (Gemma Cleanup). Use --refine to enable.")

    # Compress global silence: instead of cutting it entirely, keep a compressed portion
    # Rules: >1s truncated to 1s, then >0.2s compressed to half
    # We use a 0.2s threshold here because we want to capture ALL silences that are candidates for compression.
    global_silence = find_global_silence(stream_markers, 0.2)
    silence_excess_cuts = compress_global_silence(global_silence)
    cutting_segments.extend(silence_excess_cuts)

    cutting_segments = merge_intervals(cutting_segments)
    
    # Use max duration of all inputs as total duration
    total_dur = max([get_video_duration(v) for v in final_inputs])
    keep_segments = calculate_keep_segments(cutting_segments, total_dur)

    logger.info("Step 4/7: VAD and Overlap detection...")
    speech_intervals = {}
    for af in tqdm(all_audio_files, desc="VAD"):
        cache_path = af + ".vad.json"
        
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

    output_files = final_inputs
    # Output rendered files to parent directory of work dir
    output_base_dir = os.path.dirname(working_dir)

    logger.info("Step 6/7: Skipping render (virtual cut mode)...")

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
                # Only keep words that have a meaningful duration after adjustment
                # (Words entirely in a cut segment will have new_s == new_e)
                if (new_e - new_s) > 0.05:
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
        if af_list and af_list[0] in transcription_results:
            source_asr_words.append(transcription_results[af_list[0]].get("words", []))
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
        video_offsets=offsets,
        ass_paths=ass_files,
        asr_words=source_asr_words,
        stream_markers_global=stream_markers,
        video_to_audio_map=video_to_audio_map
    )
    # Run cleanup routine if the flag is set
    if args.clean:
        run_cleanup(working_dir, args)

def run_cleanup(working_dir: str, args):
    """Deletes all temporary artifact files from the working directory."""
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
    ]

    cleaned_count = 0
    for pattern in temp_patterns:
        for file in glob.glob(working_dir + os.path.sep + pattern):
            try:
                os.remove(file)
                cleaned_count += 1
            except OSError as e:
                logger.warning(f"Failed to remove {file}: {e}")
                
    if getattr(args, 'restart', False):
        output_base_dir = os.path.dirname(working_dir)
        output_patterns = [
            "*.srt",
            "*.ass",
            "project.kdenlive",
            f"{args.output_prefix}*"
        ]
        logger.info("Restart specified: Deleting potential output files.")
        for pattern in output_patterns:
            for file in glob.glob(output_base_dir + os.path.sep + pattern):
                try:
                    os.remove(file)
                    cleaned_count += 1
                except OSError as e:
                    logger.warning(f"Failed to remove output file {file}: {e}")

    logger.info(f"Cleanup finished. Removed {cleaned_count} files/artifacts.")


if __name__ == "__main__":
    main()
