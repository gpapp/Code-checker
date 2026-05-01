import os
import subprocess
import logging
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
from audio_utils import get_video_duration, has_video_stream, get_video_fps

logger = logging.getLogger(__name__)

def frames_to_tc(frames: int, fps: float = 25.0) -> str:
    seconds = frames / fps
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    f = int(frames % fps)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"

def secs_to_tc(seconds: float) -> str:
    # Deprecated for timeline usage, use frames_to_tc for project elements
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

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
    video_to_audio_map: Dict = None
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
        
    from kdenlive.kdenlive_lib import KdenliveProject
    
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kdenlive", "empty.kdenlive")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found at {template_path}")
        
    proj = KdenliveProject(template_path)
    
    import uuid
    from interval_utils import merge_intervals, adjust_timestamps

    max_dur_frames = sum(int(round(ke * proj.fps)) - int(round(ks * proj.fps)) for ks, ke in keep_segments)
    max_dur_tc = frames_to_tc(max_dur_frames - 1, fps) if max_dur_frames > 0 else "00:00:00:00"

    # Track assignments based on the template's 4 tracks (A1, A2, V1, V2)
    vid_tracks = ["V1", "V2"]
    aud_tracks = ["A1", "A2"]
    v_idx = 0
    a_idx = 0
    
    chain_map = {} # video_path -> chain_id
    
    # 1. Add Files to Bin (Chains) - use SOURCE files, not temporary WAV
    for i, fpath in enumerate(video_files):
        duration_s = get_video_duration(fpath)
        total_f = int(round(duration_s * fps))
        is_vid = has_video_stream(fpath)
        
        # Add source file to bin (NOT the temporary WAV)
        chain_id = proj.addFileToBin(fpath, duration_frames=total_f, clip_type="0" if is_vid else "1")
        chain_map[fpath] = chain_id
        
        # Set extra chain properties
        chain = proj.root.find(f".//chain[@id='{chain_id}']")
        ET.SubElement(chain, "property", name="audio_index").text = "1" if is_vid else "0"
        ET.SubElement(chain, "property", name="video_index").text = "0" if is_vid else "-1"
        ET.SubElement(chain, "property", name="astream").text = "0"
        
        # Add audio filters to chain (not track)
        if not is_vid:
            proj.addFilterToChain(chain_id, "ladspa.1073", {
                "internal_added": "237",
                "0": "1", "1": "0.5", "2": "0.1", "3": "0.1",
                "wetness": "1", "instances": "2", "disable": "0"
            })
            
            proj.addFilterToChain(chain_id, "dynamic_loudness", {
                "internal_added": "237",
                "target_loudness": "-23",
                "window": "3",
                "max_gain": "15",
                "min_gain": "-15",
                "max_rate": "3",
                "discontinuity_reset": "1",
                "disable": "0"
            })
        
        if i < len(asr_words) and asr_words[i]:
            speech_html = generate_kdenlive_speech_html(i+1, asr_words[i])
            ET.SubElement(chain, "property", name="kdenlive:speech").text = speech_html
    
    # 2. Add Clips to Tracks using keep_segments and handle silences
    # For each video file, add timeline items to its assigned tracks
    for i, video_path in enumerate(video_files):
        is_vid = has_video_stream(video_path)
        vid_track = None
        aud_track = None

        if is_vid:
            vid_track = vid_tracks[i] if i < len(vid_tracks) else vid_tracks[-1]
            aud_track = aud_tracks[i] if i < len(aud_tracks) else aud_tracks[-1]
        else:
            aud_track = aud_tracks[i] if i < len(aud_tracks) else aud_tracks[-1]

        # Determine which chain to use for audio track
        # If video_to_audio_map provides separate audio, use that chain
        audio_path = video_path
        if video_to_audio_map and video_path in video_to_audio_map and video_to_audio_map[video_path]:
            audio_path = video_to_audio_map[video_path][0]  # Use first audio file

        vid_chain_id = chain_map.get(video_path)
        aud_chain_id = chain_map.get(audio_path, vid_chain_id)

        # Get silences for this stream (use audio path for silence detection)
        stream_markers = stream_markers_global.get(audio_path, {})
        silences = stream_markers.get("silence", [])

        # Build timeline items: list of (type, start_frame, end_frame)
        timeline_items = []
        for ks, ke in keep_segments:
            ks_frame = int(round(ks * proj.fps))
            ke_frame = int(round(ke * proj.fps))

            # Get silences that fall within [ks, ke]
            segment_silences = [(int(round(s * proj.fps)), int(round(e * proj.fps)))
                               for s, e in silences if s < ke and e > ks]
            segment_silences.sort()

            current_pos = ks_frame
            for s_frame, e_frame in segment_silences:
                if s_frame > current_pos:
                    timeline_items.append(('clip', current_pos, s_frame))
                timeline_items.append(('blank', s_frame, e_frame))
                current_pos = e_frame

            if current_pos < ke_frame:
                timeline_items.append(('clip', current_pos, ke_frame))

        # Now add to tracks using library methods
        # Track timeline position for each track
        track_timeline_pos = {}
        for track_name in [t for t in [vid_track, aud_track] if t]:
            track_timeline_pos[track_name] = 0

        for item_type, item_start_frame, item_end_frame in timeline_items:
            for track_name in [t for t in [vid_track, aud_track] if t]:
                if item_type == 'clip':
                    # Determine which chain to use
                    if track_name == vid_track and vid_chain_id:
                        cid = vid_chain_id
                    elif track_name == aud_track and aud_chain_id:
                        cid = aud_chain_id
                    else:
                        continue

                    proj.addClipToTrack(track_name, cid, item_start_frame, item_end_frame,
                                        track_timeline_pos[track_name])

                elif item_type == 'blank':
                    blank_frames = item_end_frame - item_start_frame
                    proj.addBlankToTrack(track_name, blank_frames)

                # Update timeline position for this track
                track_timeline_pos[track_name] = item_end_frame
    
    # 3. Add Track-level Filters (audio tracks A1, A2)
    for track_name in aud_tracks:
        if track_name in proj.tracks:
            proj.addFilterToTrack(track_name, "volume", {
                "window": "75",
                "max_gain": "20dB",
                "channel_mask": "-1",
                "mlt_service": "volume",
                "internal_added": "237",
                "disable": "1"
            })
            proj.addFilterToTrack(track_name, "panner", {
                "channel": "-1",
                "mlt_service": "panner",
                "internal_added": "237",
                "start": "0.5",
                "disable": "1"
            })
            proj.addFilterToTrack(track_name, "audiolevel", {
                "iec_scale": "0",
                "mlt_service": "audiolevel",
                "internal_added": "237",
                "dbpeak": "1",
                "disable": "1"
            })
    
    # 4. Add Sequence-level Filters and Transitions (tractor4)
    seq_tractor = proj.root.find(".//tractor[@id='tractor4']")
    if seq_tractor is not None:
        # Update sequence duration
        seq_tractor.set("out", max_dur_tc)
        
        # Add sequence filters (master level)
        filt = ET.SubElement(seq_tractor, "filter", id=proj._get_next_filter_id())
        ET.SubElement(filt, "property", name="mlt_service").text = "volume"
        ET.SubElement(filt, "property", name="internal_added").text = "237"
        ET.SubElement(filt, "property", name="window").text = "75"
        ET.SubElement(filt, "property", name="max_gain").text = "20dB"
        ET.SubElement(filt, "property", name="channel_mask").text = "-1"
        ET.SubElement(filt, "property", name="disable").text = "1"
        
        filt2 = ET.SubElement(seq_tractor, "filter", id=proj._get_next_filter_id())
        ET.SubElement(filt2, "property", name="mlt_service").text = "panner"
        ET.SubElement(filt2, "property", name="internal_added").text = "237"
        ET.SubElement(filt2, "property", name="start").text = "0.5"
        ET.SubElement(filt2, "property", name="disable").text = "1"
        
        # Add transitions for audio tracks (blend against track 0)
        for idx, track_name in enumerate(aud_tracks):
            if idx == 0:
                continue
            trans = ET.SubElement(seq_tractor, "transition", id=proj._get_next_filter_id().replace("filter", "transition"))
            ET.SubElement(trans, "property", name="a_track").text = "0"
            ET.SubElement(trans, "property", name="b_track").text = str(idx + 1)
            ET.SubElement(trans, "property", name="mlt_service").text = "mix"
            ET.SubElement(trans, "property", name="kdenlive_id").text = "mix"
            ET.SubElement(trans, "property", name="internal_added").text = "237"
            ET.SubElement(trans, "property", name="always_active").text = "1"
            ET.SubElement(trans, "property", name="accepts_blanks").text = "1"
            ET.SubElement(trans, "property", name="sum").text = "1"
        
        # Add transitions for video tracks
        vid_offset = len(aud_tracks) + 1  # +1 for black track (track 0)
        for idx, track_name in enumerate(vid_tracks):
            trans = ET.SubElement(seq_tractor, "transition", id=proj._get_next_filter_id().replace("filter", "transition"))
            ET.SubElement(trans, "property", name="a_track").text = "0"
            ET.SubElement(trans, "property", name="b_track").text = str(vid_offset + idx)
            ET.SubElement(trans, "property", name="mlt_service").text = "qtblend"
            ET.SubElement(trans, "property", name="kdenlive_id").text = "qtblend"
            ET.SubElement(trans, "property", name="internal_added").text = "237"
            ET.SubElement(trans, "property", name="always_active").text = "1"
            ET.SubElement(trans, "property", name="compositing").text = "0"
            ET.SubElement(trans, "property", name="distort").text = "0"
            ET.SubElement(trans, "property", name="rotate_center").text = "0"
    
    # 5. Adjust sequence duration properties
    seq_tr = proj.seq_tractor
    ET.SubElement(seq_tr, "property", name="kdenlive:duration").text = max_dur_tc
    ET.SubElement(seq_tr, "property", name="kdenlive:maxduration").text = str(max_dur_frames)
    
    # Update project tractor (tractor4 is the sequence)
    proj.save(output_path)
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
