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
        
    from kdenlive.kdenlive_lib import KdenliveProject
    
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kdenlive", "empty.kdenlive")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found at {template_path}")
        
    proj = KdenliveProject(template_path)
    
    import uuid
    from interval_utils import merge_intervals, adjust_timestamps

    max_dur_frames = sum(int(round(ke * proj.fps)) - int(round(ks * proj.fps)) for ks, ke in keep_segments)
    max_dur_tc = frames_to_tc(max_dur_frames - 1, proj.fps) if max_dur_frames > 0 else "00:00:00:00"

    # Track assignments based on the template's 4 tracks (A1, A2, V1, V2)
    vid_tracks = ["V1", "V2"]
    aud_tracks = ["A1", "A2"]
    v_idx = 0
    a_idx = 0
    
    chain_map = {} # (original_file_path, stream_idx) -> chain_id
    
    # 1. Add Files to Bin (Chains)
    # We use ORIGINAL files in the Kdenlive project.
    # audio_info_map: temp_wav -> {"original": path, "stream_idx": idx}

    # Track which (original_file, stream_idx) pairs we need
    required_chains = []
    # Video chains
    for v in video_files:
        if has_video_stream(v):
            required_chains.append((v, -1)) # -1 for video+audio stream 1

    # Audio chains from extracted WAVs
    for temp_wav, info in (audio_info_map or {}).items():
        required_chains.append((info["original"], info["stream_idx"]))

    # Deduplicate
    required_chains = list(set(required_chains))

    for orig_path, s_idx in required_chains:
        duration_s = get_video_duration(orig_path)
        total_f = int(round(duration_s * proj.fps))
        is_vid = has_video_stream(orig_path)
        
        chain_id = proj.addFileToBin(orig_path, duration_frames=total_f, clip_type="0" if is_vid else "1")
        chain_map[(orig_path, s_idx)] = chain_id
        
        # Set extra chain properties
        chain = proj.root.find(f".//chain[@id='{chain_id}']")

        # If s_idx is -1, it's the primary video/audio stream
        if s_idx == -1:
            ET.SubElement(chain, "property", name="audio_index").text = "1" if is_vid else "0"
            ET.SubElement(chain, "property", name="video_index").text = "0" if is_vid else "-1"
        else:
            ET.SubElement(chain, "property", name="audio_index").text = str(s_idx)
            ET.SubElement(chain, "property", name="video_index").text = "-1"

        ET.SubElement(chain, "property", name="astream").text = "0"
        
        # Add audio filters to chain (not track) - apply to any chain being used for audio
        if s_idx != -1 or not is_vid:
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
        
    # Pair video files with their offsets and asr_words to maintain correspondence
    paired_inputs = []
    for i in range(len(video_files)):
        paired_inputs.append({
            "path": video_files[i],
            "offset": video_offsets[i] if i < len(video_offsets) else 0.0,
            "asr": asr_words[i] if i < len(asr_words) else []
        })

    # Sort: prioritize those WITH video streams
    paired_inputs.sort(key=lambda x: not has_video_stream(x["path"]))

    # Add speech info
    for i, item in enumerate(paired_inputs):
        v_path = item["path"]
        vid_chain_id = chain_map.get((v_path, -1))
        if vid_chain_id and item["asr"]:
            chain = proj.root.find(f".//chain[@id='{vid_chain_id}']")
            if chain is not None:
                speech_html = generate_kdenlive_speech_html(i+1, item["asr"])
                ET.SubElement(chain, "property", name="kdenlive:speech").text = speech_html

    # 2. Add Clips to Tracks using keep_segments
    video_track_idx = 0
    audio_track_idx = 0

    for item in paired_inputs:
        video_path = item["path"]
        offset = item["offset"]
        is_vid = has_video_stream(video_path)

        audio_sources = video_to_audio_map.get(video_path, [])
        if not audio_sources and not is_vid:
            audio_sources = [video_path]

        vid_chain_id = chain_map.get((video_path, -1))

        # Add video track if exists
        if is_vid and video_track_idx < len(vid_tracks):
            v_track_name = vid_tracks[video_track_idx]
            video_track_idx += 1

            timeline_pos = 0
            for ks, ke in keep_segments:
                ks_f = int(round((ks + offset) * proj.fps))
                ke_f = int(round((ke + offset) * proj.fps))
                if ke_f > ks_f:
                    proj.addClipToTrack(v_track_name, vid_chain_id, ks_f, ke_f, timeline_pos)
                    timeline_pos += (ke_f - ks_f)

        # Add audio tracks
        for a_source in audio_sources:
            if audio_track_idx >= len(aud_tracks):
                break

            a_track_name = aud_tracks[audio_track_idx]
            audio_track_idx += 1

            # Find chain for this audio source
            if audio_info_map and a_source in audio_info_map:
                info = audio_info_map[a_source]
                a_chain_id = chain_map.get((info["original"], info["stream_idx"]))
            else:
                a_chain_id = chain_map.get((a_source, -1))
            if not a_chain_id:
                continue

            # Get silences and spikes for this stream
            markers = stream_markers_global.get(a_source, {})
            silences = markers.get("silence", [])
            spikes = markers.get("spikes", [])

            timeline_pos = 0
            for ks, ke in keep_segments:
                ks_f = int(round((ks + offset) * proj.fps))
                ke_f = int(round((ke + offset) * proj.fps))
                if ke_f <= ks_f:
                    continue

                # Add continuous clip
                entry = proj.addClipToTrack(a_track_name, a_chain_id, ks_f, ke_f, timeline_pos)

                # Build volume keyframes for muting
                # Mute if:
                # 1. It is a silence on THIS stream
                # 2. It is a spike on THIS stream

                mute_intervals = merge_intervals(silences + spikes)
                # Clip-relative intervals
                clip_mutes = []
                for ms, me in mute_intervals:
                    # Intersect with [ks, ke]
                    is_s = max(ms, ks)
                    is_e = min(me, ke)
                    if is_e > is_s:
                        # Convert to clip-relative frames
                        rel_s = int(round((is_s - ks) * proj.fps))
                        rel_e = int(round((is_e - ks) * proj.fps))
                        clip_mutes.append((rel_s, rel_e))

                if clip_mutes:
                    # Generate volume keyframes: frame=level;frame=level
                    # level 0 for mute, 1 for normal
                    kfs = []
                    clip_len_f = ke_f - ks_f
                    # Sort and merge just in case
                    clip_mutes = merge_intervals([(float(s), float(e)) for s, e in clip_mutes])
                    clip_mutes = [(int(s), int(e)) for s, e in clip_mutes]

                    if clip_mutes[0][0] > 0:
                        kfs.append("0=1")

                    for idx, (rs, re) in enumerate(clip_mutes):
                        # Clamp to clip boundaries
                        rs = max(0, rs)
                        re = min(clip_len_f - 1, re)
                        if re < rs: continue

                        if rs > 0:
                            # If there was space before this mute, ensure we were 1
                            if not kfs or not kfs[-1].split('=')[0] == str(rs-1):
                                kfs.append(f"{rs-1}=1")
                        kfs.append(f"{rs}=0")
                        kfs.append(f"{re}=0")

                        # Look ahead to see if we should unmute
                        if idx + 1 < len(clip_mutes):
                            next_rs = clip_mutes[idx+1][0]
                            if next_rs > re + 1:
                                kfs.append(f"{re+1}=1")
                        else:
                            # Last mute
                            if re < clip_len_f - 1:
                                kfs.append(f"{re+1}=1")

                    if kfs:
                        proj.addFilterToEntry(entry, "volume", {"gain": ";".join(kfs)})

                timeline_pos += (ke_f - ks_f)
    
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
    if seq_tr is not None:
        duration_prop = seq_tr.find("property[@name='kdenlive:duration']")
        if duration_prop is not None:
            duration_prop.text = max_dur_tc
        else:
            ET.SubElement(seq_tr, "property", name="kdenlive:duration").text = max_dur_tc

        max_dur_prop = seq_tr.find("property[@name='kdenlive:maxduration']")
        if max_dur_prop is not None:
            max_dur_prop.text = str(max_dur_frames)
        else:
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
