import os
import logging
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
from audio_utils import get_video_duration, has_video_stream, get_video_fps

from mlt_python.project import MLTProject
from mlt_python.filter import Filter
from interval_utils import merge_intervals

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
    output_path: str,
    keep_segments: List[Tuple[float, float]],
    stream_spikes_map: Dict[str, List[List[Tuple[float, float]]]],
    overlaps: List[Tuple[float, float]],
    repetitions: List[Tuple[float, float]],
    fps: float = 25.0,
    video_offsets: List[float] = None,
    ass_paths: List[str] = None,
    asr_words: List[List[Dict]] = None,
    stream_markers_global: Dict = None,
    video_to_audio_map: Dict = None,
    audio_info_map: Dict = None
):
    if not video_offsets:
        video_offsets = [0.0] * len(video_files)
    if video_to_audio_map is None:
        video_to_audio_map = {f: [f] for f in video_files}
    if stream_markers_global is None:
        stream_markers_global = {}
    if ass_paths is None:
        ass_paths = []
    if asr_words is None:
        asr_words = []
        
    import sys
    import uuid
    from pathlib import Path
       

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

    proj = MLTProject(profile=profile_name)
    fps = proj.profile.fps

    # 1. Add Files to Bin (Producers)
    required_chains = []
    for v in video_files:
        if has_video_stream(v):
            required_chains.append((v, -1)) # -1 for video+audio stream 1

    for temp_wav, info in (audio_info_map or {}).items():
        required_chains.append((info["original"], info["stream_idx"]))

    for v, audios in (video_to_audio_map or {}).items():
        for a in audios:
            if a not in video_files or not has_video_stream(a):
                required_chains.append((a, 0))

    required_chains = list(set(required_chains))
    producer_map = {} # (original_file_path, s_idx) -> producer_id

    for orig_path, s_idx in required_chains:
        is_vid = has_video_stream(orig_path)
        
        props = {}
        if s_idx == -1:
            props["audio_index"] = "1" if is_vid else "0"
            props["video_index"] = "0" if is_vid else "-1"
        else:
            props["audio_index"] = str(s_idx)
            props["video_index"] = "-1"
        props["astream"] = "0"

        producer = proj.add_producer(orig_path, properties=props)
        producer_map[(orig_path, s_idx)] = producer.id
        
        if s_idx != -1 or not is_vid:
            producer.add_filter(Filter("ladspa.1073", properties={
                "internal_added": "237",
                "0": "1", "1": "0.5", "2": "0.1", "3": "0.1",
                "wetness": "1", "instances": "2", "disable": "0"
            }))
            
            producer.add_filter(Filter("dynamic_loudness", properties={
                "internal_added": "237",
                "target_loudness": "-23",
                "window": "3",
                "max_gain": "15",
                "min_gain": "-15",
                "max_rate": "3",
                "discontinuity_reset": "1",
                "disable": "0"
            }))

    # Add speech info
    paired_inputs = []
    for i in range(len(video_files)):
        paired_inputs.append({
            "path": video_files[i],
            "offset": video_offsets[i] if i < len(video_offsets) else 0.0,
            "asr": asr_words[i] if i < len(asr_words) else []
        })
    paired_inputs.sort(key=lambda x: not has_video_stream(x["path"]))

    for i, item in enumerate(paired_inputs):
        v_path = item["path"]
        p_id = producer_map.get((v_path, -1))
        if not p_id:
            for (p, s), pid in producer_map.items():
                if p == v_path:
                    p_id = pid
                    break
        if p_id and item["asr"]:
            producer = proj.get_producer(p_id)
            if producer:
                speech_html = generate_kdenlive_speech_html(i+1, item["asr"])
                producer.set_property("kdenlive:speech", speech_html)

    # 2. Add Tracks and Clips
    for item in paired_inputs:
        video_path = item["path"]
        offset = item["offset"]
        is_vid = has_video_stream(video_path)

        audio_sources = video_to_audio_map.get(video_path, [])
        if not audio_sources and not is_vid:
            audio_sources = [video_path]

        vid_prod_id = producer_map.get((video_path, -1))

        if is_vid:
            playlist = proj.add_track("video")
            for ks, ke in keep_segments:
                if ke > ks:
                    in_frame = int(round((ks + offset) * fps))
                    dur_frames = int(round((ke - ks) * fps))
                    playlist.add_clip(vid_prod_id, in_point=in_frame, length=dur_frames)

        for a_source in audio_sources:
            playlist = proj.add_track("audio")
            a_prod_id = None
            if audio_info_map and a_source in audio_info_map:
                info = audio_info_map[a_source]
                a_prod_id = producer_map.get((info["original"], info["stream_idx"]))
            else:
                a_prod_id = producer_map.get((a_source, 0)) or producer_map.get((a_source, -1))
            if not a_prod_id: continue

            markers = stream_markers_global.get(a_source, {})
            silences = markers.get("silence", [])
            spikes = markers.get("spikes", [])
            mute_intervals = merge_intervals(silences + spikes)

            for ks, ke in keep_segments:
                if ke <= ks: continue
                in_frame = int(round((ks + offset) * fps))
                dur_frames = int(round((ke - ks) * fps))
                
                clip = playlist.add_clip(a_prod_id, in_point=in_frame, length=dur_frames)
                
                # Volume mutes
                kfs = []
                rel_mutes = []
                clip_start = ks + offset
                clip_end = ke + offset
                for ms, me in mute_intervals:
                    is_s = max(ms, clip_start)
                    is_e = min(me, clip_end)
                    if is_e > is_s:
                        rs = int(round((is_s - clip_start) * fps))
                        re = int(round((is_e - clip_start) * fps))
                        rel_mutes.append((rs, re))
                
                if rel_mutes:
                    rel_mutes.sort()
                    merged = []
                    if rel_mutes:
                        merged.append(list(rel_mutes[0]))
                        for curr in rel_mutes[1:]:
                            if curr[0] <= merged[-1][1]:
                                merged[-1][1] = max(merged[-1][1], curr[1])
                            else:
                                merged.append(list(curr))
                    
                    if merged[0][0] > 0: kfs.append("0=1")
                    for idx, (rs, re) in enumerate(merged):
                        re = min(dur_frames - 1, re)
                        if re < rs: continue
                        if rs > 0: kfs.append(f"{max(0, rs-1)}=1")
                        kfs.append(f"{rs}=0")
                        kfs.append(f"{re}=0")
                        if idx + 1 < len(merged):
                            next_rs = merged[idx+1][0]
                            if next_rs > re + 1: kfs.append(f"{re+1}=1")
                        elif re < dur_frames - 1: kfs.append(f"{re+1}=1")
                    
                    if kfs:
                        clip.filters.append(Filter("volume", properties={"gain": ";".join(kfs), "mlt_service": "volume", "kdenlive_id": "volume"}))

    # 3. Sequence-level Filters
    proj.add_filter("volume", properties={
        "window": "75", "max_gain": "20dB", "channel_mask": "-1", "internal_added": "237", "disable": "1"
    })
    proj.add_filter("panner", properties={
        "internal_added": "237", "start": "0.5", "disable": "1"
    })

    # 4. Link subtitles
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
        
        # Determine if we should start a new segment
        should_start_new = False
        if not current_group:
            should_start_new = False
        else:
            # Too many words?
            if len(current_group) >= max_words:
                should_start_new = True
            # Too long a gap?
            elif w['start'] - current_group[-1]['end'] > max_gap:
                should_start_new = True
            # Previous word ended a sentence?
            prev_word = current_group[-1]['word'].strip()
            if prev_word.endswith(('.', '?', '!', '...', ':')):
                should_start_new = True
        
        if should_start_new:
            # Finalize current segment
            segments.append({
                'start': current_group[0]['start'],
                'end': current_group[-1]['end'],
                'word': ' '.join(gw['word'].strip() for gw in current_group)
            })
            current_group = [w]
        else:
            current_group.append(w)
            
    # Final segment
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
