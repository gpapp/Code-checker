import os
import logging
import sys
import uuid
import json
from pathlib import Path
import xml.etree.ElementTree as ET # Keep for generate_kdenlive_speech_html
from typing import List, Tuple, Dict, Set
from audio_utils import get_video_duration, has_video_stream, get_video_fps, get_video_stream_index, get_audio_stream_indices, get_track_name_from_path

from mlt_python.project import MLTProject
from mlt_python.timecode import Timecode
from mlt_python.marker import Marker, markers_to_json
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
    keep_segments: List[Tuple[float, float]], # Float seconds in the master timeline
    stream_spikes: Dict = None,
    video_offsets: List[float] = None,
    ass_paths: List[str] = None,
    asr_words: List[List[Dict]] = None,
    stream_markers: Dict = None,
    fps: float = 25.0
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
       

    # Detect profile from first video
    if video_files and os.path.exists(video_files[0]):
        try:
            detected_fps = get_video_fps(video_files[0])
        except:
            detected_fps = fps
    else:
        detected_fps = fps

    # Map FPS to profile preset
    profile_name = "hd1080_30"
    if abs(detected_fps - 25.0) < 0.1: profile_name = "hd1080_25"
    elif abs(detected_fps - 29.97) < 0.1: profile_name = "hd1080_2997"
    elif abs(detected_fps - 24.0) < 0.1: profile_name = "hd1080_24"
    elif abs(detected_fps - 30.0) < 0.1: profile_name = "hd1080_30"
    elif abs(detected_fps - 23.976) < 0.1: profile_name = "hd1080_2398"
    elif abs(detected_fps - 50.0) < 0.1: profile_name = "hd1080_50"
    elif abs(detected_fps - 60.0) < 0.1: profile_name = "hd1080_60"

    proj = MLTProject(profile=profile_name)
    fps = proj.profile.fps

    def to_tc(seconds: float) -> str:
        """Convert float seconds to HH:MM:SS:FF timecode string."""
        return str(Timecode.from_seconds(seconds, fps))

    total_timeline_duration = sum(ke - ks for ks, ke in keep_segments)

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
                
                # Add markers for fillers if available
                if temp_path and stream_markers and temp_path in stream_markers:
                    fillers = stream_markers[temp_path].get("fillers", [])
                    if fillers:
                        markers = []
                        for start, end in fillers:
                            frame = int(start * fps)
                            duration_frames = int((end - start) * fps)
                            markers.append(Marker(pos=frame, comment="Filler", marker_type=0, duration=duration_frames))
                        if markers:
                            proj.clip_markers[audio_producer.id] = markers
                
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
    paired_inputs.sort(key=lambda x: (not has_video_stream(x["path"]), x["path"]))

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

    # --- Pass 1: Audio Tracks (Reverse Order) ---
    for item in reversed(paired_inputs):
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
                pad = 0.1
                gap = 0.4
                if nsa:
                    padded_nsa = [(max(0.0, s - pad), min(a_dur, e + pad)) for s, e in nsa]
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
                
                curr_timeline = 0.0
                for as_start, as_end in final_audio_segments:
                    master_start = as_start - offset
                    
                    # Find timeline position by locating the corresponding keep segment
                    timeline_start = None
                    accum = 0.0
                    for ks, ke in keep_segments:
                        if ks <= master_start < ke:
                            timeline_start = accum + (master_start - ks)
                            break
                        accum += (ke - ks)
                    
                    if timeline_start is None: continue
                    
                    dur = as_end - as_start
                    if timeline_start > curr_timeline:
                        playlist.add_blank(to_tc(timeline_start - curr_timeline), fps=fps)
                    
                    playlist.add_clip(
                        audio_clip_producer.id,
                        in_point=to_tc(as_start),
                        duration=to_tc(dur),
                        fps=fps
                    )
                    curr_timeline = timeline_start + dur

                # Ensure audio track length matches total keep duration exactly
                if curr_timeline < total_timeline_duration:
                    playlist.add_blank(to_tc(total_timeline_duration - curr_timeline), fps=fps)

                added_audio_tracks.add(audio_clip_key)

    # --- Pass 2: Video Tracks (Forward Order to Mirror Audio) ---
    for item in paired_inputs:
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
                    if ke > ks:
                        playlist.add_clip(
                            video_clip_producer.id, 
                            in_point=to_tc(ks + offset),
                            duration=to_tc(ke - ks),
                            fps=fps
                        )
                added_video_tracks.add(video_path)

    # --- Pass 3: AV Split Groups ---
    groups = []
    for video_path, tracks in av_track_mapping.items():
        v_idx = tracks["v"]
        a_indices = tracks["a"]
        if v_idx is not None and a_indices:
            for i in range(len(keep_segments)):
                group = {
                    "type": "AVSplit",
                    "children": [
                        {
                            "type": "Leaf", "leaf": "clip", 
                            "data": f"{v_idx}:{i}:-1"
                        }
                    ]
                }
                for a_idx in a_indices:
                    group["children"].append({
                        "type": "Leaf", "leaf": "clip",
                        "data": f"{a_idx}:{i}:-1"
                    })
                groups.append(group)
    
    if groups:
        proj.kdenlive.set_doc_property("sequenceproperties.groups", json.dumps(groups, indent=4))

    # 3. Link subtitles
    for ass_file in (ass_paths or []):
        if os.path.exists(ass_file):
            proj.add_subtitle(ass_file)

    # Save
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
