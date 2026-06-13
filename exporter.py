import os
import math
import logging
import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Set
from audio_utils import get_video_duration, has_video_stream, get_video_stream_index, get_audio_stream_indices, get_track_name_from_path

from mlt_python.project import MLTProject
from interval_utils import merge_intervals, intersect_intervals, calculate_keep_segments, adjust_timestamps


logger = logging.getLogger(__name__)


def generate_kdenlive_speech_html(prod_id: int, words: List[Dict]) -> str:
    if not words:
        return ""
    html = [
        '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">',
        '<html><head><meta name="qrichtext" content="1" /><meta charset="utf-8" /><style type="text/css">',
        'p, li { white-space: pre-wrap; }',
        'hr { height: 1px; border-width: 0; }',
        'li.unchecked::marker { content: "\\2610"; }',
        'li.checked::marker { content: "\\2612"; }',
        '</style></head><body style=" font-family:\'Segoe UI\'; font-size:9pt; font-weight:400; font-style:normal;">'
    ]
    p_style = ' margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;'
    
    current_p = f'<p style="{p_style}">'
    last_end = 0.0
    for w in words:
        start = w['start']
        end = w['end']
        word = w['word']
        if start - last_end > 1.0 and last_end > 0:
            html.append(current_p + '</p>')
            html.append(f'<p style="{p_style}"><a href="{prod_id}#{last_end:.2f}:{start:.2f}">No speech</a></p>')
            current_p = f'<p style="{p_style}">'
        elif last_end == 0.0 and start > 0.5:
            html.append(f'<p style="{p_style}"><a href="{prod_id}#0.00:{start:.2f}">No speech</a></p>')
            
        current_p += f'<a href="{prod_id}#{start:.2f}:{end:.2f}"> {word}</a> '
        last_end = end
        
    if current_p != f'<p style="{p_style}">':
        html.append(current_p + '</p>')
        
    html.append('</body></html>')
    return '\n'.join(html)

def generate_kdenlive_project(
    video_files: List[str],
    audio_files: Dict[str, List[Dict]],
    output_path: str,
    keep_segments: List[Tuple[float, float]],
    stream_spikes: Dict = None,
    video_offsets: List[float] = None,
    ass_paths: List[str] = None,
    asr_words: List[List[Dict]] = None,
    stream_markers: Dict = None,
):
    if not video_offsets:
        video_offsets = [0.0] * len(video_files)
    if audio_files is None:
        audio_files = {f: [{"original": f, "stream_idx": -1, "temp_path": f}] for f in video_files}
    if stream_markers is None:
        stream_markers = {}
    if ass_paths is None:
        ass_paths = []
    if asr_words is None:
        asr_words = []

    proj = MLTProject(profile="hd1080_25")
    fps = proj.profile.fps

    # Frame-align ALL keep segments to prevent drift between tracks.
    # Blanks serialize to frames, clips serialize to ms — rounding differences
    # between these units accumulate and cause tracks with more blanks (audio)
    # to drift from tracks with fewer blanks (video).
    keep_contrib_frames = [int(round((ke - ks) * fps)) for ks, ke in keep_segments]
    cumul_keep_frames = [0]
    for cf in keep_contrib_frames:
        cumul_keep_frames.append(cumul_keep_frames[-1] + cf)
    total_timeline_frames = cumul_keep_frames[-1]
    total_timeline_duration = total_timeline_frames / fps

    # Centralized storage for all producers (raw files and their specific stream representations)
    required_chains = []
    
    # Determine if we have audio overrides for videos
    has_override = {}
    kdenlive_clip_producers: Dict[Tuple[str, str], Producer] = {}
    primary_producer_for_file: Dict[str, Producer] = {}

    # Shared Bin ID per physical file to prevent replication in Kdenlive Bin
    file_to_bin_id: Dict[str, str] = {}
    def get_shared_bin_id(path: str) -> str:
        if path not in file_to_bin_id:
            file_to_bin_id[path] = str(abs(hash(path)) % 1000000)
        return file_to_bin_id[path]

    # First pass: Create all necessary Producer objects for Kdenlive clips in the bin
    for v_path in video_files:
        has_vid_stream = has_video_stream(v_path)
        
        associated_audio_sources = audio_files.get(v_path, [])

        if has_vid_stream:
            video_producer_id = f"clip_vid_{len(kdenlive_clip_producers)}"
            video_props = {"video_index": "0", "audio_index": "-1"}
            video_props["kdenlive:id"] = get_shared_bin_id(v_path)
            video_producer = proj.add_producer(v_path, id=video_producer_id, mlt_service="avformat", properties=video_props)
            kdenlive_clip_producers[(v_path, "video")] = video_producer
            primary_producer_for_file[v_path] = video_producer

        if associated_audio_sources:
            for src_info in associated_audio_sources:
                original_audio_path = src_info["original"]
                stream_idx = src_info["stream_idx"]
                temp_path = src_info["temp_path"]
                
                audio_producer_id = f"clip_aud_{len(kdenlive_clip_producers)}"
                audio_props = {"audio_index": str(stream_idx), "video_index": "-1"}
                audio_props["kdenlive:id"] = get_shared_bin_id(original_audio_path)
                audio_producer = proj.add_producer(original_audio_path, id=audio_producer_id, mlt_service="avformat", properties=audio_props)
                kdenlive_clip_producers[(original_audio_path, f"audio_{stream_idx}")] = audio_producer
                
                if v_path not in primary_producer_for_file and not has_vid_stream:
                    primary_producer_for_file[v_path] = audio_producer

        elif not has_vid_stream and v_path not in primary_producer_for_file:
            audio_producer_id = f"clip_aud_{len(kdenlive_clip_producers)}"
            audio_props = {"audio_index": "0", "video_index": "-1"}
            audio_props["kdenlive:id"] = get_shared_bin_id(v_path)
            audio_producer = proj.add_producer(v_path, id=audio_producer_id, mlt_service="avformat", properties=audio_props)
            kdenlive_clip_producers[(v_path, "audio_0")] = audio_producer
            primary_producer_for_file[v_path] = audio_producer

    # Add speech info
    paired_inputs = []
    for i in range(len(video_files)):
        paired_inputs.append({
            "path": video_files[i],
            "offset": video_offsets[i] if i < len(video_offsets) else 0.0,
            "asr": asr_words[i] if asr_words and i < len(asr_words) else []
        })
    paired_inputs.sort(key=lambda x: (not has_video_stream(x["path"]), video_files.index(x["path"])))

    for i, item in enumerate(paired_inputs):
        file_path = item["path"]
        producer = primary_producer_for_file.get(file_path)
        if producer and item["asr"]:
            speech_html = generate_kdenlive_speech_html(i + 1, item["asr"])
            producer.set_property("kdenlive:speech", speech_html)

    # 2. Add Tracks and Clips
    added_video_tracks: Set[str] = set()
    added_audio_tracks: Set[str] = set()
    av_track_mapping: Dict[str, Dict] = {}

    # --- Pass 1: Audio Tracks (Forward Order) ---
    for item in paired_inputs:
        video_path = item["path"]
        offset = item["offset"]
        is_vid = has_video_stream(video_path)
        
        current_audio_sources = audio_files.get(video_path, [])
        if not current_audio_sources and not is_vid:
            current_audio_sources = [{"original": video_path, "stream_idx": 0, "temp_path": video_path}]

        for src_info in current_audio_sources:
            original_audio_path = src_info["original"]
            stream_idx = src_info["stream_idx"]
            temp_path = src_info["temp_path"]

            audio_clip_key = (original_audio_path, f"audio_{stream_idx}")
            audio_clip_producer = kdenlive_clip_producers.get(audio_clip_key)

            if audio_clip_producer and audio_clip_key not in added_audio_tracks:
                track_idx = len(proj.playlists) + 1
                playlist = proj.add_track("audio", id=f"track_aud_{track_idx}")
                playlist.set_property("kdenlive:track_name", get_track_name_from_path(original_audio_path))
                
                if video_path not in av_track_mapping:
                    av_track_mapping[video_path] = {"v": None, "a": []}
                av_track_mapping[video_path]["a"].append(track_idx)
            
                a_dur = get_video_duration(original_audio_path)  # seconds
                markers_data = (stream_markers or {}).get(temp_path, {})
                silences = markers_data.get("silence", [])  # list of (start_s, end_s)

                # Non-silence intervals in producer timeline (seconds)
                nsa = calculate_keep_segments(silences, a_dur)

                # Smoothing (in seconds)
                # Asymmetric roll-off: speech onsets are sharp (0.05s),
                # trailing words fade out gradually (0.3s)
                pad_pre = 0.05
                pad_post = 0.3
                gap = 0.4
                if nsa:
                    padded_nsa = [(max(0.0, s - pad_pre), min(a_dur, e + pad_post)) for s, e in nsa]
                    merged_nsa = merge_intervals(padded_nsa)
                    if merged_nsa:
                        smoothed_nsa = [merged_nsa[0]]
                        for start_s, end_s in merged_nsa[1:]:
                            if start_s - smoothed_nsa[-1][1] < gap:
                                smoothed_nsa[-1] = (smoothed_nsa[-1][0], end_s)
                            else:
                                smoothed_nsa.append((start_s, end_s))
                        nsa = smoothed_nsa

                # Map keep_segments to audio producer timeline (shifted by offset)
                audio_keep_intervals = []
                for ks, ke in keep_segments:
                    s = max(0.0, ks + offset)
                    e = min(a_dur, ke + offset)
                    if s < e:
                        audio_keep_intervals.append((s, e))

                # Intersect in seconds domain
                final_audio_segments = intersect_intervals(nsa, audio_keep_intervals)
                
                # Track position in frames to match XML serialization exactly
                curr_frame = 0
                
                # Group final audio segments by keep segments they fall into
                keep_seg_audio = {i: [] for i in range(len(keep_segments))}
                for as_start, as_end in final_audio_segments:
                    master_start = as_start - offset
                    # Find which keep segment contains this master position
                    for i, (ks, ke) in enumerate(keep_segments):
                        if ks <= master_start < ke:
                            keep_seg_audio[i].append((as_start, as_end))
                            break

                for i, (ks, ke) in enumerate(keep_segments):
                    prev_frame = cumul_keep_frames[i]
                    target_end_frame = cumul_keep_frames[i+1]
                    
                    segs = keep_seg_audio[i]
                    if not segs:
                        # Keep segment is silent for this track, add a blank matching the keep segment duration
                        blank_frames = target_end_frame - curr_frame
                        if blank_frames > 0:
                            playlist.add_blank(blank_frames / fps)
                            curr_frame = target_end_frame
                        continue
                    
                    # Align the first and last segments to keep segment boundaries
                    first_as_start, first_as_end = segs[0]
                    last_as_start, last_as_end = segs[-1]
                    
                    # Stretch first and last to keep segment boundaries
                    segs[0] = (max(0.0, ks + offset), first_as_end)
                    segs[-1] = (last_as_start, min(a_dur, ke + offset))
                    if len(segs) == 1:
                        segs[0] = (max(0.0, ks + offset), min(a_dur, ke + offset))
                        
                    # Frame-align all segments to prevent timecode rounding discrepancies
                    aligned_segs = []
                    for s, e in segs:
                        s_aligned = round(s * fps) / fps
                        e_aligned = round(e * fps) / fps
                        aligned_segs.append((s_aligned, e_aligned))
                    segs = aligned_segs
                        
                    for as_start, as_end in segs:
                        master_start = as_start - offset
                        start_offset_frames = int(round((master_start - ks) * fps))
                        timeline_frame = prev_frame + start_offset_frames
                        
                        master_end = min(as_end - offset, ke)
                        end_offset_frames = int(round((master_end - ks) * fps))
                        dur_frames = end_offset_frames - start_offset_frames
                        
                        if timeline_frame > curr_frame:
                            blank_frames = timeline_frame - curr_frame
                            playlist.add_blank(blank_frames / fps)
                            curr_frame = timeline_frame
                        
                        if dur_frames > 0:
                            playlist.add_clip(
                                audio_clip_producer.id,
                                in_point=as_start,
                                duration=dur_frames / fps,
                                fps=fps,
                            )
                            curr_frame += dur_frames
                            
                    # Add any remaining blank to match keep segment end exactly
                    if curr_frame < target_end_frame:
                        playlist.add_blank((target_end_frame - curr_frame) / fps)
                        curr_frame = target_end_frame

                # Add filler markers on this audio chain (visible as colored bars in timeline)
                filler_intervals = markers_data.get("fillers", [])
                for fs, fe in filler_intervals:
                    if fe - fs > 0.01:
                        proj.add_marker(
                            fs, comment="Filler", marker_type=4,
                            duration=fe - fs, producer_id=audio_clip_producer.id
                        )

                added_audio_tracks.add(audio_clip_key)

    # --- Pass 2: Video Tracks (Reverse Order to Mirror Audio) ---
    for item in reversed(paired_inputs):
        video_path = item["path"]
        offset = item["offset"]
        is_vid = has_video_stream(video_path)
        track_name = get_track_name_from_path(video_path)

        if is_vid and video_path not in added_video_tracks:
            video_clip_producer = kdenlive_clip_producers.get((video_path, "video"))
            if video_clip_producer:
                track_idx = len(proj.playlists) + 1
                playlist = proj.add_track("video", id=f"track_vid_{track_idx}")
                playlist.set_property("kdenlive:track_name", track_name)
                
                if video_path not in av_track_mapping:
                    av_track_mapping[video_path] = {"v": None, "a": []}
                av_track_mapping[video_path]["v"] = track_idx

                for ks, ke in keep_segments:
                    dur_frames = int(round((ke - ks) * fps))
                    dur_aligned = dur_frames / fps
                    if dur_aligned > 0:
                        playlist.add_clip(
                            video_clip_producer.id, 
                            in_point=ks + offset,
                            duration=dur_aligned,
                            fps=fps,
                        )
                added_video_tracks.add(video_path)

    # 3. Link subtitles
    for ass_file in (ass_paths or []):
        if os.path.exists(ass_file):
            proj.add_subtitle(ass_file)

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(proj.to_xml(kdenlive_format=True))

def detect_best_intra_frame_encoder() -> str:
    """Detect best available intra-frame encoder (prores_ks > prores > ffv1 > libx264)."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=True
        )
        encoders = result.stdout
        if "prores_ks" in encoders:
            return "prores_ks"
        elif "prores" in encoders:
            return "prores"
        elif "ffv1" in encoders:
            return "ffv1"
    except Exception:
        pass
    return "libx264"


def _video_extension(encoder: str) -> str:
    """Return the recommended container extension for a video encoder."""
    if encoder in ("prores_ks", "prores"):
        return ".mov"
    elif encoder == "ffv1":
        return ".mkv"
    return ".mp4"


def _edit_friendly_opts(encoder: str) -> List[str]:
    """Common options for edit-friendly output (30fps, intra-frame, CFR, broad compatibility)."""
    if encoder in ("prores_ks", "prores"):
        return ["-profile:v", "hq", "-pix_fmt", "yuv422p10le", "-r", "30", "-vsync", "cfr"]
    elif encoder == "ffv1":
        return ["-level", "3", "-pix_fmt", "yuv422p", "-r", "30", "-vsync", "cfr"]
    return ["-g", "1", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", "30", "-vsync", "cfr"]


def _frame_align_segments(
    segments: List[Tuple[float, float]],
    fps: float,
) -> List[Tuple[float, float]]:
    """Snap segment boundaries to the nearest video frame boundaries (inward).

    Ensures audio cuts align exactly with video frames, preventing
    audio/video duration mismatch.
    """
    frame_dur = 1.0 / fps
    aligned = []
    for s, e in segments:
        s_aligned = math.ceil(s / frame_dur) * frame_dur
        e_aligned = math.floor(e / frame_dur) * frame_dur
        if s_aligned < e_aligned:
            aligned.append((s_aligned, e_aligned))
    return aligned


def _get_video_bitrate(video_path: str) -> int:
    """Get the video stream bitrate in bps, or 0 if unknown."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=bit_rate",
             "-of", "default=noprint_wrappers=1:nokey=1",
             video_path],
            capture_output=True, text=True, check=True
        )
        val = result.stdout.strip()
        return int(val) if val else 0
    except (ValueError, subprocess.CalledProcessError):
        return 0


def render_processed_video_lossless_cut(
    video_files: List[str],
    audio_files: Dict[str, List[Dict]],
    keep_segments: List[Tuple[float, float]],
    output_dir: str,
    video_offsets: List[float] = None,
) -> Dict[str, Dict[str, str]]:
    """Lossless-cut rendering using -ss/-to per segment + concat filter.

    Uses -ss/-to before each input to seek fast (decodes only keep segments),
    then concatenates with the concat filter. No temp files needed.
    Encodes with best available intra-frame codec (ProRes > FFV1 > libx264 All-I)
    at constant frame rate for editing compatibility.
    """
    if video_offsets is None:
        video_offsets = [0.0] * len(video_files)

    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)
    rendered: Dict[str, Dict[str, str]] = {}
    log = ["-loglevel", "error", "-hide_banner"]

    for vf_idx, v_path in enumerate(tqdm(video_files, desc="Rendering sources", unit="source")):
        offset = video_offsets[vf_idx] if vf_idx < len(video_offsets) else 0.0
        is_video = has_video_stream(v_path)
        associated = audio_files.get(v_path, [])
        track_name = get_track_name_from_path(v_path)
        if not is_video and not associated:
            continue
        src_dur = get_video_duration(v_path)

        valid_segs = []
        for ks, ke in keep_segments:
            s = max(0.0, ks + offset)
            e = min(src_dur, ke + offset)
            if s < e:
                valid_segs.append((s, e))
        if not valid_segs:
            continue

        if is_video:
            valid_segs = _frame_align_segments(valid_segs, 30.0)

        flac_path = associated[0]["original"] if associated else None
        a_out = os.path.join(output_dir, f"{track_name}_processed.flac")
        entry: Dict[str, str] = {}

        if is_video:
            encoder = detect_best_intra_frame_encoder()
            ext = _video_extension(encoder)
            v_out = os.path.join(output_dir, f"{track_name}_processed{ext}")
            ef_opts = _edit_friendly_opts(encoder)
            src_bitrate = _get_video_bitrate(v_path)
            if src_bitrate:
                cap = 10_000_000
                bv = min(src_bitrate, cap)
                maxrate = min(int(src_bitrate * 1.5), cap)
                nvenc_opts = ["-preset", "p2", "-rc", "vbr_hq", "-b:v", str(bv), "-maxrate", str(maxrate), *ef_opts]
            else:
                nvenc_opts = ["-preset", "p2", "-cq", "23", *ef_opts]
            enc_opts = {"h264_nvenc": nvenc_opts,
                        "h264_qsv": ["-preset", "veryfast", "-global_quality", "23", "-b:v", "50M", *ef_opts],
                        "libx264": ["-preset", "superfast", "-crf", "23", *ef_opts],
                        "prores_ks": ef_opts, "prores": ef_opts,
                        "ffv1": ef_opts}.get(encoder, ef_opts)
            concat_script = os.path.join(output_dir, f"_concat_{track_name}.txt")
            with open(concat_script, "w", encoding="utf-8") as f:
                f.write("ffconcat version 1.0\n")
                for s, e in valid_segs:
                    f.write(f"file '{v_path.replace(chr(92), '/')}'\n")
                    f.write(f"inpoint {s}\n")
                    f.write(f"outpoint {e}\n")
            try:
                subprocess.run([
                    "ffmpeg",
                    "-f", "concat", "-safe", "0",
                    "-i", concat_script,
                    "-c:v", encoder, *enc_opts,
                    *log, "-y", v_out
                ], check=True)
            finally:
                if os.path.exists(concat_script):
                    os.remove(concat_script)
            entry["video"] = v_out

        if flac_path:
            select_parts = "+".join(f"between(t,{s},{e})" for s, e in valid_segs)
            filter_file = os.path.join(output_dir, f"_filter_{track_name}.txt")
            filter_graph = f"aselect='{select_parts}',asetpts=N/SR/TB"
            with open(filter_file, "w", encoding="utf-8") as f:
                f.write(filter_graph)
            try:
                subprocess.run([
                    "ffmpeg", "-i", flac_path,
                    "-filter_complex_script", filter_file,
                    "-c:a", "flac",
                    *log, "-y", a_out
                ], check=True)
            finally:
                if os.path.exists(filter_file):
                    os.remove(filter_file)
            entry["audio"] = a_out

        if entry:
            rendered[v_path] = entry

    return rendered


def render_processed_video(
    video_files: List[str],
    audio_files: Dict[str, List[Dict]],
    keep_segments: List[Tuple[float, float]],
    output_dir: str,
    video_offsets: List[float] = None,
) -> Dict[str, Dict[str, str]]:
    """Renders processed files per source with keep segments concatenated.

    Produces two files per source: video (ProRes/FFV1/H.264 + FLAC audio) and FLAC (audio-only).
    Video and audio are cut together in a single filter graph so output durations match exactly.
    Uses ffmpeg's select/aselect filters with filter_complex_script to avoid
    Windows cmd length limits. Encodes with best available intra-frame codec
    (ProRes > FFV1 > libx264 All-I) at constant frame rate for editing compatibility.
    Returns dict mapping source path -> {video: video_path, audio: flac_path}.
    """
    if video_offsets is None:
        video_offsets = [0.0] * len(video_files)

    encoder = detect_best_intra_frame_encoder()
    ext = _video_extension(encoder)
    ef_opts = _edit_friendly_opts(encoder)
    enc_opts = {"h264_nvenc": ["-preset", "p2", "-rc", "vbr_hq", "-b:v", "10M", "-maxrate", "15M", *ef_opts],
                "h264_qsv": ["-preset", "veryfast", "-global_quality", "23", "-b:v", "10M", *ef_opts],
                "libx264": ["-preset", "superfast", "-crf", "23", *ef_opts],
                "prores_ks": ef_opts, "prores": ef_opts,
                "ffv1": ef_opts}.get(encoder, ef_opts)

    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)
    rendered: Dict[str, Dict[str, str]] = {}

    for vf_idx, v_path in enumerate(tqdm(video_files, desc="Rendering sources", unit="source")):
        offset = video_offsets[vf_idx] if vf_idx < len(video_offsets) else 0.0
        is_video = has_video_stream(v_path)
        associated = audio_files.get(v_path, [])
        track_name = get_track_name_from_path(v_path)
        if not is_video and not associated:
            continue
        src_dur = get_video_duration(v_path)

        valid_segs = []
        for ks, ke in keep_segments:
            s = max(0.0, ks + offset)
            e = min(src_dur, ke + offset)
            if s < e:
                valid_segs.append((s, e))
        if not valid_segs:
            continue

        if is_video:
            valid_segs = _frame_align_segments(valid_segs, 30.0)

        flac_path = associated[0]["original"] if associated else None
        select_parts = "+".join(f"between(t,{s},{e})" for s, e in valid_segs)
        log = ["-loglevel", "error", "-hide_banner"]
        filter_file = os.path.join(output_dir, f"_filter_{track_name}.txt")
        entry: Dict[str, str] = {}
        v_out = os.path.join(output_dir, f"{track_name}_processed{ext}")
        a_out = os.path.join(output_dir, f"{track_name}_processed.flac")

        if is_video and flac_path:
            filter_graph = (
                f"[0:v:0]select='{select_parts}',setpts=N/FRAME_RATE/TB[v];\n"
                f"[1:a:0]aselect='{select_parts}',asetpts=N/SR/TB[a]"
            )
            with open(filter_file, "w", encoding="utf-8") as f:
                f.write(filter_graph)
            try:
                subprocess.run([
                    "ffmpeg", "-i", v_path, "-i", flac_path,
                    "-filter_complex_script", filter_file,
                    "-map", "[v]", "-map", "[a]",
                    "-c:v", encoder, *enc_opts,
                    "-c:a", "flac",
                    *log, "-y", v_out
                ], check=True)
            finally:
                if os.path.exists(filter_file):
                    os.remove(filter_file)
            entry["video"] = v_out
            subprocess.run([
                "ffmpeg", "-i", v_out,
                "-vn", "-c:a", "copy",
                *log, "-y", a_out
            ], check=True)
            entry["audio"] = a_out
        elif is_video:
            filter_graph = f"[0:v:0]select='{select_parts}',setpts=N/FRAME_RATE/TB[v]"
            with open(filter_file, "w", encoding="utf-8") as f:
                f.write(filter_graph)
            try:
                subprocess.run([
                    "ffmpeg", "-i", v_path,
                    "-filter_complex_script", filter_file,
                    "-map", "[v]",
                    "-c:v", encoder, *enc_opts,
                    *log, "-y", v_out
                ], check=True)
            finally:
                if os.path.exists(filter_file):
                    os.remove(filter_file)
            entry["video"] = v_out
        elif flac_path:
            filter_graph = f"aselect='{select_parts}',asetpts=N/SR/TB"
            with open(filter_file, "w", encoding="utf-8") as f:
                f.write(filter_graph)
            try:
                subprocess.run([
                    "ffmpeg", "-i", flac_path,
                    "-filter_complex_script", filter_file,
                    "-c:a", "flac",
                    *log, "-y", a_out
                ], check=True)
            finally:
                if os.path.exists(filter_file):
                    os.remove(filter_file)
            entry["audio"] = a_out

        if entry:
            rendered[v_path] = entry

    return rendered


def generate_kdenlive_from_rendered(
    rendered_files: Dict[str, Dict[str, str]],
    output_path: str,
    ass_paths: List[str] = None,
    filler_intervals: Dict[str, List[Tuple[float, float]]] = None,
):
    """Generate a simple kdenlive project from pre-rendered processed files.

    Each source gets separate video and/or audio tracks referencing the
    rendered .mp4 (H.264+FLAC) and .flac (audio-only) files.
    Filler markers are added as colored bars on audio chains when
    filler_intervals is provided (keyed by source path).
    """
    from mlt_python.project import MLTProject

    if filler_intervals is None:
        filler_intervals = {}

    proj = MLTProject(profile="hd1080_25")
    fps = proj.profile.fps

    for source_path, rend_entry in rendered_files.items():
        track_name = get_track_name_from_path(source_path)

        v_path = rend_entry.get("video")
        a_path = rend_entry.get("audio")
        intervals = filler_intervals.get(source_path, [])

        if v_path:
            playlist = proj.add_track("video", id=f"track_{track_name}_video")
            playlist.set_property("kdenlive:track_name", track_name)
            producer = proj.add_producer(
                v_path, id=f"clip_{track_name}_video", mlt_service="avformat",
            )
            dur = get_video_duration(v_path)
            if dur > 0:
                playlist.add_clip(producer.id, in_point=0.0, duration=dur, fps=fps)

        if a_path:
            playlist = proj.add_track("audio", id=f"track_{track_name}_audio")
            playlist.set_property("kdenlive:track_name", f"{track_name} (audio)")
            producer = proj.add_producer(
                a_path, id=f"clip_{track_name}_audio", mlt_service="avformat",
            )
            dur = get_video_duration(a_path)
            if dur > 0:
                playlist.add_clip(producer.id, in_point=0.0, duration=dur, fps=fps)

                for fs, fe in intervals:
                    if fe - fs > 0.01:
                        proj.add_marker(
                            fs, comment="Filler", marker_type=4,
                            duration=fe - fs, producer_id=producer.id,
                        )

    for ass_file in (ass_paths or []):
        if os.path.exists(ass_file):
            proj.add_subtitle(ass_file)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(proj.to_xml(kdenlive_format=True))


def format_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def group_words_into_segments(words: List[Dict], max_words: int = 5, max_gap: float = 1.0) -> List[Dict]:
    """Groups individual words into readable segments (sentences or small chunks)."""
    if not words:
        return []
    
    segments = []
    current_group = []
    
    for w in words:
        word_text = w['word'].strip()
        
        should_start_new = False
        if not current_group:
            should_start_new = False
        else:
            if len(current_group) >= max_words:
                should_start_new = True
            elif w['start'] - current_group[-1]['end'] > max_gap:
                should_start_new = True
            prev_word = current_group[-1]['word'].strip()
            if prev_word.endswith(('.', '?', '!', '...', ':')):
                should_start_new = True
        
        if should_start_new:
            segments.append({
                'start': current_group[0]['start'],
                'end': current_group[-1]['end'],
                'word': ' '.join(gw['word'].strip() for gw in current_group)
            })
            current_group = [w]
        else:
            current_group.append(w)
            
    if current_group:
        segments.append({
            'start': current_group[0]['start'],
            'end': current_group[-1]['end'],
            'word': ' '.join(gw['word'].strip() for gw in current_group)
        })
        
    return segments

def generate_ass_file(words: List[Dict], output_path: str):
    header = "[Script Info]\nScriptType: v4.00+\nPlayResX: 384\nPlayResY: 288\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,16,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    segments = group_words_into_segments(words)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for seg in segments:
            f.write(f"Dialogue: 0,{format_ass_time(seg['start'])},{format_ass_time(seg['end'])},Default,,0,0,0,,{seg['word']}\n")

def generate_srt_file(words: List[Dict], output_path: str):
    segments = group_words_into_segments(words)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n")
            f.write(f"{seg['word']}\n\n")
