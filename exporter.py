import os
import subprocess
import logging
import lxml.etree as ET
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

def secs_to_tc(seconds: float) -> str:
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
    if video_offsets is None:
        video_offsets = []
    if ass_paths is None:
        ass_paths = []
    if asr_words is None:
        asr_words = []
        
    import uuid
    from audio_utils import get_video_duration, has_video_stream
    import xml.etree.ElementTree as ET

    max_dur = sum(e - s for s, e in keep_segments)
    if not keep_segments:
        logger.warning("No keep segments, setting max_dur to 0")

    from interval_utils import merge_intervals

    seq_uuid = f"{{{uuid.uuid4()}}}"

    # root attribute tells MLT where to resolve relative resource paths from
    media_root = os.path.dirname(os.path.abspath(video_files[0])) if video_files else os.path.dirname(os.path.abspath(output_path))
    root = ET.Element("mlt", LC_NUMERIC="C", version="7.28.0", title="Anonymous Submission", producer="main_bin", root=media_root)
    ET.SubElement(root, "profile", description=f"HD 1080p {fps}fps", frame_rate_num=str(int(fps)), 
                  frame_rate_den="1", width="1920", height="1080", progressive="1", 
                  sample_aspect_num="1", sample_aspect_den="1", display_aspect_num="16", 
                  display_aspect_den="9", colorspace="709")

    # Generators
    for i, fpath in enumerate(video_files):
        prod_num = i + 1
        duration_s = get_video_duration(fpath)
        is_vid = has_video_stream(fpath)
        file_size = os.path.getsize(fpath)
        clip_uuid = f"{{{uuid.uuid4()}}}"
        bname = os.path.basename(fpath)

        for use_tl in [False, True]:
            chain_id = f"chain{prod_num}_tl" if use_tl else f"chain{prod_num}"
            t = ET.SubElement(root, "chain", id=chain_id, out=secs_to_tc(duration_s))
            ET.SubElement(t, "property", name="length").text = str(int(duration_s * fps))
            ET.SubElement(t, "property", name="eof").text = "pause"
            ET.SubElement(t, "property", name="resource").text = bname if use_tl else fpath
            ET.SubElement(t, "property", name="mlt_service").text = "avformat-novalidate" if use_tl else "avformat"
            ET.SubElement(t, "property", name="seekable").text = "1"
            ET.SubElement(t, "property", name="audio_index").text = "1" if is_vid else "0"
            ET.SubElement(t, "property", name="video_index").text = "0" if is_vid else "-1"
            ET.SubElement(t, "property", name="astream").text = "0"
            ET.SubElement(t, "property", name="kdenlive:control_uuid").text = clip_uuid
            ET.SubElement(t, "property", name="kdenlive:proxy").text = "-"
            ET.SubElement(t, "property", name="kdenlive:originalurl").text = bname
            ET.SubElement(t, "property", name="kdenlive:id").text = str(prod_num)
            ET.SubElement(t, "property", name="kdenlive:clip_type").text = "0" if is_vid else "1"
            ET.SubElement(t, "property", name="kdenlive:file_size").text = str(file_size)
            ET.SubElement(t, "property", name="kdenlive:folderid").text = "-1"
            if use_tl and not is_vid:
                ET.SubElement(t, "property", name="set.test_audio").text = "0"
                ET.SubElement(t, "property", name="set.test_image").text = "1"
            if not use_tl and i < len(asr_words) and asr_words[i]:
                speech_html = generate_kdenlive_speech_html(prod_num, asr_words[i])
                ET.SubElement(t, "property", name="kdenlive:speech").text = speech_html

    # Pre-calculate tracks to generate correct tractor IDs
    num_video_tracks = sum(1 for f in video_files if has_video_stream(f))
    # Only count audio tracks for files that actually have audio
    num_audio_tracks = sum(1 for f in video_files if video_to_audio_map is None or video_to_audio_map.get(f, []))
    total_track_pairs = num_video_tracks + num_audio_tracks
    
    seq_tractor_id = f"tractor{total_track_pairs}"
    proj_tractor_id = f"tractor{total_track_pairs + 1}"

    # main_bin
    bin_playlist = ET.SubElement(root, "playlist", id="main_bin")
    for i in range(len(video_files)):
        prod_num = i + 1
        fpath = video_files[i]
        duration_s = get_video_duration(fpath)
        ent = ET.SubElement(bin_playlist, "entry", **{
            "in": "00:00:00.000",
            "out": secs_to_tc(duration_s),
            "producer": f"chain{prod_num}"
        })
        ET.SubElement(ent, "property", name="kdenlive:id").text = str(prod_num)
        
    seq_ent = ET.SubElement(bin_playlist, "entry", **{
        "in": "00:00:00.000",
        "out": secs_to_tc(max_dur),
        "producer": seq_tractor_id
    })
    ET.SubElement(bin_playlist, "property", name="kdenlive:docproperties.activetimeline").text = seq_uuid
    ET.SubElement(bin_playlist, "property", name="kdenlive:docproperties.audioChannels").text = "2"
    ET.SubElement(bin_playlist, "property", name="kdenlive:docproperties.version").text = "1.1"
    ET.SubElement(bin_playlist, "property", name="kdenlive:docproperties.decimalPoint").text = "."
    ET.SubElement(bin_playlist, "property", name="xml_retain").text = "1"
    
    # black_track
    bt = ET.SubElement(root, "producer", id="black_track", **{"in":"00:00:00.000", "out":secs_to_tc(max_dur)})
    ET.SubElement(bt, "property", name="length").text = "2147483647"
    ET.SubElement(bt, "property", name="eof").text = "continue"
    ET.SubElement(bt, "property", name="resource").text = "black"
    ET.SubElement(bt, "property", name="mlt_service").text = "color"
    ET.SubElement(bt, "property", name="mlt_image_format").text = "rgba"

    # Playlists and Tractors
    pl_idx = 0
    t_idx = 0
    multitrack_producers = ["black_track"]
    
    num_video_tracks = 0
    num_audio_tracks = 0

    # PASS 1: VIDEO TRACKS
    for i, video_path in enumerate(video_files):
        if not has_video_stream(video_path):
            continue
            
        prod_num = i + 1
        num_video_tracks += 1
        offset = video_offsets[i] if i < len(video_offsets) else 0.0
        
        main_pl = ET.SubElement(root, "playlist", id=f"playlist{pl_idx}")
        aux_pl = ET.SubElement(root, "playlist", id=f"playlist{pl_idx+1}")
        
        for ks, ke in keep_segments:
            src_in = ks + offset
            src_out = ke + offset
            if src_out <= 0:
                ET.SubElement(main_pl, "blank", length=secs_to_tc(ke - ks))
                continue
                
            blank_dur = 0.0
            if src_in < 0:
                blank_dur = -src_in
                src_in = 0.0
            
            if blank_dur > 0:
                ET.SubElement(main_pl, "blank", length=secs_to_tc(blank_dur))
                
            ent = ET.SubElement(main_pl, "entry", producer=f"chain{prod_num}_tl", **{
                "in": secs_to_tc(src_in),
                "out": secs_to_tc(src_out)
            })
            ET.SubElement(ent, "property", name="kdenlive:id").text = str(prod_num)

        tr_id = f"tractor{t_idx}"
        tr = ET.SubElement(root, "tractor", id=tr_id, **{"in":"00:00:00.000"})
        ET.SubElement(tr, "property", name="kdenlive:trackheight").text = "61"
        ET.SubElement(tr, "track", hide="audio", producer=f"playlist{pl_idx}")
        ET.SubElement(tr, "track", hide="audio", producer=f"playlist{pl_idx+1}")
            
        multitrack_producers.append(tr_id)
        pl_idx += 2
        t_idx += 1

    # PASS 2: AUDIO TRACKS (skip files with no audio mapping)
    for i, video_path in enumerate(video_files):
        # Skip video files whose on-camera audio is unused (external audio present)
        af_list = video_to_audio_map.get(video_path, []) if video_to_audio_map else []
        if not af_list:
            continue
            
        prod_num = i + 1
        is_vid = has_video_stream(video_path)
        
        # Audio tracks are the reference for cuts, so they don't get shifted.
        offset = 0.0
        
        main_pl = ET.SubElement(root, "playlist", id=f"playlist{pl_idx}")
        aux_pl = ET.SubElement(root, "playlist", id=f"playlist{pl_idx+1}")
        
        ET.SubElement(main_pl, "property", name="kdenlive:audio_track").text = "1"
        ET.SubElement(aux_pl, "property", name="kdenlive:audio_track").text = "1"
            
        filter_counter = 0

        # Get silence intervals for this stream
        silence_list = []
        if af_list[0] in stream_markers_global:
            silence_list = sorted(stream_markers_global[af_list[0]]["silence"])
            
        # 1. FIND SPEECH SEGMENTS (Invert silence with 2.0s threshold to keep short pauses together)
        SILENCE_THRESHOLD = 2.0
        long_silences = [s for s in silence_list if (s[1] - s[0]) >= SILENCE_THRESHOLD]
        
        # Spikes (pops/clicks)
        spikes_list = stream_markers_global.get(af_list[0], {}).get("spikes", []) if af_list else []
            
        # FIRST PASS: Generate base audio entries perfectly identical to the video track logic
        audio_entries = []
        for ks, ke in keep_segments:
            src_in = ks + offset
            src_out = ke + offset
            if src_out <= 0:
                audio_entries.append({"type": "blank", "in": ks, "out": ke})
                continue
                
            blank_dur = 0.0
            if src_in < 0:
                audio_entries.append({"type": "blank", "in": src_in, "out": 0.0})
                src_in = 0.0
            
            if src_in < src_out:
                audio_entries.append({"type": "entry", "in": src_in, "out": src_out})

        def frames_to_tc(frames):
            return secs_to_tc(frames / fps)

        # SECOND PASS: Sub-divide entries by silences natively in frames
        # This completely removes and trims silent portions as physical cuts!
        processed_elements = []
        for item in audio_entries:
            if item["type"] == "blank":
                dur_frames = int(round(item["out"] * fps)) - int(round(item["in"] * fps))
                processed_elements.append({"type": "blank", "dur_frames": dur_frames})
                continue
                
            e_in = item["in"]
            e_out = item["out"]
            e_in_frame = int(round(e_in * fps))
            e_out_frame = int(round(e_out * fps))
            
            # Since offset is 0 for audio, e_in and e_out match source timeline
            overlap_silences = [s for s in long_silences if s[0] < e_out and s[1] > e_in]
            
            if not overlap_silences:
                processed_elements.append({"type": "entry", "in_frames": e_in_frame, "out_frames": e_out_frame})
                continue
                
            # Slicing e_in_frame to e_out_frame using overlap silences guarantees exact sum
            curr_frame = e_in_frame
            
            for s in overlap_silences:
                s_in_frame = int(round(s[0] * fps))
                s_out_frame = int(round(s[1] * fps))
                
                # Constrain to current entry bounds
                s_in_frame = max(curr_frame, s_in_frame)
                s_out_frame = min(e_out_frame, s_out_frame)
                
                if s_in_frame > curr_frame:
                    # Speech before the silence
                    processed_elements.append({"type": "entry", "in_frames": curr_frame, "out_frames": s_in_frame})
                    
                if s_out_frame > s_in_frame:
                    # The silence itself
                    processed_elements.append({"type": "blank", "dur_frames": s_out_frame - s_in_frame})
                    
                curr_frame = max(curr_frame, s_out_frame)
                
            if curr_frame < e_out_frame:
                # Remaining speech
                processed_elements.append({"type": "entry", "in_frames": curr_frame, "out_frames": e_out_frame})

        # BUILD THE PLAYLIST
        for item in processed_elements:
            if item["type"] == "blank":
                ET.SubElement(main_pl, "blank", length=frames_to_tc(item["dur_frames"]))
            else:
                s_in_frame = item["in_frames"]
                s_out_frame = item["out_frames"]
                ss_in = s_in_frame / fps
                ss_out = s_out_frame / fps
                
                ent = ET.SubElement(main_pl, "entry", producer=f"chain{prod_num}_tl", **{
                    "in": frames_to_tc(s_in_frame),
                    "out": frames_to_tc(s_out_frame)
                })
                ET.SubElement(ent, "property", name="kdenlive:id").text = str(prod_num)
                ET.SubElement(ent, "property", name="audio_index").text = "1" if is_vid else "0"
                
                # Setup spike filters inside this chopped entry
                spikes_list = stream_markers_global.get(af_list[0], {}).get("spikes", []) if af_list else []
                rel_spikes = [s for s in spikes_list if s[0] < ss_out and s[1] > ss_in]
                if rel_spikes:
                    filter_id = f"filter_{pl_idx}_{filter_counter}"
                    filter_counter += 1
                    f_node = ET.SubElement(ent, "filter", id=filter_id)
                    ET.SubElement(f_node, "property", name="mlt_service").text = "volume"
                    ET.SubElement(f_node, "property", name="kdenlive_id").text = "volume"
                    kf = ["0=1.0"]
                    dur_frames = s_out_frame - s_in_frame
                    
                    for spike_s, spike_e in rel_spikes:
                        c_start = max(0, spike_s - ss_in)
                        c_end = min(ss_out, spike_e) - ss_in
                        rs = int(c_start * fps)
                        re = int(c_end * fps)
                        if rs > 0: kf.append(f"{rs-1}=1.0")
                        kf.append(f"{rs}=0.0")
                        kf.append(f"{re}=0.0")
                        kf.append(f"{re+1}=1.0")
                        
                    last_frame = dur_frames - 1
                    if last_frame < 0: last_frame = 0
                    kf.append(f"{last_frame}=1.0")
                    ET.SubElement(f_node, "property", name="level").text = ";".join(kf)
        tr_id = f"tractor{t_idx}"
        tr = ET.SubElement(root, "tractor", id=tr_id, **{"in":"00:00:00.000"})
        ET.SubElement(tr, "property", name="kdenlive:audio_track").text = "1"
        ET.SubElement(tr, "property", name="kdenlive:trackheight").text = "61"
        ET.SubElement(tr, "track", hide="video", producer=f"playlist{pl_idx}")
        ET.SubElement(tr, "track", hide="video", producer=f"playlist{pl_idx+1}")
        
        # Track-level audio effects (apply to all content on this track)
        # Compressor (ladspa.1073)
        f_comp = ET.SubElement(tr, "filter", id=f"trackfilter_comp_{t_idx}")
        ET.SubElement(f_comp, "property", name="mlt_service").text = "ladspa.1073"
        ET.SubElement(f_comp, "property", name="kdenlive_id").text = "ladspa.1073"
        ET.SubElement(f_comp, "property", name="0").text = "1.0"
        ET.SubElement(f_comp, "property", name="1").text = "0.5"
        ET.SubElement(f_comp, "property", name="2").text = "0.25"
        ET.SubElement(f_comp, "property", name="3").text = "0.0"
        ET.SubElement(f_comp, "property", name="disable").text = "0"
        
        # Dynamic Loudness Normalizer
        f_dyn = ET.SubElement(tr, "filter", id=f"trackfilter_dyn_{t_idx}")
        ET.SubElement(f_dyn, "property", name="mlt_service").text = "dynamic_loudness"
        ET.SubElement(f_dyn, "property", name="target").text = "-14"
        ET.SubElement(f_dyn, "property", name="kdenlive_id").text = "dynamic_loudness"
        ET.SubElement(f_dyn, "property", name="disable").text = "0"
            
        multitrack_producers.append(tr_id)
        pl_idx += 2
        t_idx += 1

    # Main sequence tractor
    seq_tr = ET.SubElement(root, "tractor", id=seq_tractor_id, **{"in":"00:00:00.000", "out":secs_to_tc(max_dur)})
    ET.SubElement(seq_tr, "property", name="kdenlive:clipname").text = "Sequence 1"
    ET.SubElement(seq_tr, "property", name="kdenlive:uuid").text = seq_uuid
    ET.SubElement(seq_tr, "property", name="kdenlive:clip_type").text = "0"
    ET.SubElement(seq_tr, "property", name="kdenlive:producer_type").text = "17"
    ET.SubElement(seq_tr, "property", name="kdenlive:sequenceproperties.hasAudio").text = "1" if num_audio_tracks > 0 else "0"
    ET.SubElement(seq_tr, "property", name="kdenlive:sequenceproperties.hasVideo").text = "1" if num_video_tracks > 0 else "0"
    ET.SubElement(seq_tr, "property", name="kdenlive:sequenceproperties.tracksCount").text = str(num_video_tracks + num_audio_tracks)
    ET.SubElement(seq_tr, "property", name="kdenlive:sequenceproperties.tracks").text = str(num_video_tracks)
    
    for mtp in multitrack_producers:
        ET.SubElement(seq_tr, "track", producer=mtp)
        
    # Write Kdenlive-native subtitle file (project.kdenlive.srt)
    # Kdenlive expects a companion .srt file named exactly "<project_file>.srt"
    if ass_paths:
        combined_srt_path = output_path + ".srt"
        try:
            # Read and combine all SRT content
            combined_words = []
            for ap in ass_paths:
                srt_companion = os.path.splitext(ap)[0] + ".srt"
                if os.path.exists(srt_companion):
                    import shutil
                    shutil.copy2(srt_companion, combined_srt_path)
                    break  # Use first available SRT as the project subtitle
        except Exception as e:
            logger.warning(f"Could not create Kdenlive subtitle file: {e}")

    for ti in range(1, len(multitrack_producers)):
        trans = ET.SubElement(seq_tr, "transition", id=f"transition{ti-1}")
        ET.SubElement(trans, "property", name="a_track").text = "0"
        ET.SubElement(trans, "property", name="b_track").text = str(ti)
        is_audio = (ti > num_video_tracks) # Assumes top tracks are video
        srv = "mix" if is_audio else "qtblend"
        ET.SubElement(trans, "property", name="mlt_service").text = srv
        ET.SubElement(trans, "property", name="kdenlive_id").text = srv
        ET.SubElement(trans, "property", name="internal_added").text = "237"
        ET.SubElement(trans, "property", name="always_active").text = "1"

    # Project tractor
    proj_tr = ET.SubElement(root, "tractor", id=proj_tractor_id, **{"in":"00:00:00.000", "out":secs_to_tc(max_dur)})
    ET.SubElement(proj_tr, "property", name="kdenlive:projectTractor").text = "1"
    ET.SubElement(proj_tr, "track", producer=seq_tractor_id, **{"in":"00:00:00.000", "out":secs_to_tc(max_dur)})

    ET.indent(root, space=" ", level=0)
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

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
