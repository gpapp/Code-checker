import os
import subprocess
import logging
import lxml.etree as ET
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

def process_video(input_path: str, output_path: str, keep_segments: List[Tuple[float, float]], stream_spikes: List[List[Tuple[float, float]]], fps: float = 25.0):
    """Cuts the video and mutes spikes using FFmpeg complex filters."""
    if not keep_segments:
        logger.warning(f"No keep segments for {input_path}")
        return

    # Video filter: use numeric fps
    v_select = "+".join([f"between(t,{s},{e})" for s, e in keep_segments])
    vf = f"select='{v_select}',setpts=N/({fps})/TB"
    filter_complex = [f"[0:v]{vf}[v]"]

    audio_outputs = []
    for i, spikes in enumerate(stream_spikes):
        af = ""
        if spikes:
            mute_expr = "+".join([f"between(t,{ss},{se})" for ss, se in spikes])
            af += f"volume=enable='{mute_expr}':volume=0,"

        a_select = "+".join([f"between(t,{s},{e})" for s, e in keep_segments])
        af += f"aselect='{a_select}',asetpts=N/(16000)/TB" # Assuming 16k SR from extraction

        filter_complex.append(f"[0:a:{i}]{af}[a{i}]")
        audio_outputs.append(f"[a{i}]")

    cmd = ["ffmpeg", "-i", input_path, "-filter_complex", ";".join(filter_complex), "-map", "[v]"]
    for ao in audio_outputs:
        cmd.extend(["-map", ao])

    cmd.extend(["-c:v", "libx264", "-c:a", "aac", "-y", output_path])
    logger.info(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=True, text=True)

def generate_kdenlive_project(
    video_files: List[str],
    output_path: str,
    keep_segments: List[Tuple[float, float]],
    stream_spikes_map: Dict[str, List[List[Tuple[float, float]]]],
    overlaps: List[Tuple[float, float]],
    repetitions: List[Tuple[float, float]],
    fps: float = 25.0,
    is_rendered: bool = False
):
    """Generates a .kdenlive XML file."""
    root = ET.Element("mlt", version="7.13", producer="main_bin")
    ET.SubElement(root, "profile", description=f"HD 1080p {fps}fps", frame_rate_num=str(int(fps)), frame_rate_den="1", width="1920", height="1080", progressive="1", sample_aspect_num="1", sample_aspect_den="1", display_aspect_num="16", display_aspect_den="9", colorspace="709")

    for i, f in enumerate(video_files):
        p = ET.SubElement(root, "producer", id=f"producer_{i}", resource=os.path.abspath(f))
        ET.SubElement(p, "property", name="kdenlive:clipname").text = os.path.basename(f)

    bin_playlist = ET.SubElement(root, "playlist", id="main_bin")
    for i in range(len(video_files)):
        ET.SubElement(bin_playlist, "entry", producer=f"producer_{i}")

    tractor = ET.SubElement(root, "tractor", id="main_tractor", global_feed="1")
    multitrack = ET.SubElement(tractor, "multitrack")

    for i, video_path in enumerate(video_files):
        v_playlist = ET.SubElement(multitrack, "playlist", id=f"video_track_{i}")

        if is_rendered:
            # If rendered, the file is already cut. Just one entry.
            entry = ET.SubElement(v_playlist, "entry", producer=f"producer_{i}")
        else:
            for ks, ke in keep_segments:
                entry = ET.SubElement(v_playlist, "entry", producer=f"producer_{i}")
                entry.set("in", str(int(ks * fps)))
                entry.set("out", str(int(ke * fps) - 1))

        spikes_list = stream_spikes_map.get(video_path, [])
        for a_idx, spikes in enumerate(spikes_list):
            a_playlist = ET.SubElement(multitrack, "playlist", id=f"audio_track_{i}_{a_idx}")

            if is_rendered:
                entry = ET.SubElement(a_playlist, "entry", producer=f"producer_{i}")
                ET.SubElement(entry, "property", name="audio_index").text = str(a_idx)
                ET.SubElement(entry, "property", name="video_index").text = "-1"
                # Spikes are already muted in rendered output, so no filters needed here.
            else:
                for ks, ke in keep_segments:
                    entry = ET.SubElement(a_playlist, "entry", producer=f"producer_{i}")
                    entry.set("in", str(int(ks * fps)))
                    entry.set("out", str(int(ke * fps) - 1))
                    ET.SubElement(entry, "property", name="audio_index").text = str(a_idx)
                    ET.SubElement(entry, "property", name="video_index").text = "-1"

                    # Mute spikes with proper keyframes
                    relevant_spikes = [s for s in spikes if s[0] < ke and s[1] > ks]
                    if relevant_spikes:
                        f_node = ET.SubElement(entry, "filter")
                        ET.SubElement(f_node, "property", name="mlt_service").text = "volume"
                        # Keyframe string: start with 1.0
                        kf = ["0=1.0"]
                        for ss, se in relevant_spikes:
                            rel_s = max(0, int((ss - ks) * fps))
                            rel_e = int((min(ke, se) - ks) * fps)
                            if rel_s > 0: kf.append(f"{rel_s-1}=1.0")
                            kf.append(f"{rel_s}=0.0")
                            kf.append(f"{rel_e}=0.0")
                            kf.append(f"{rel_e+1}=1.0")
                        # Ensure it ends with 1.0 if not already
                        last_frame = int((ke - ks) * fps) - 1
                        kf.append(f"{last_frame}=1.0")
                        ET.SubElement(f_node, "property", name="level").text = ";".join(kf)

    kdenlive_root = ET.SubElement(root, "kdenlive_project")
    for s, e in overlaps:
        ET.SubElement(kdenlive_root, "guide", time=str(s), comment="Overlap", type="1")
    for s, e in repetitions:
        ET.SubElement(kdenlive_root, "guide", time=str(s), comment="Repetition", type="2")

    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True, pretty_print=True)

def format_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def generate_ass_file(words: List[Dict], output_path: str):
    header = "[Script Info]\nScriptType: v4.00+\nPlayResX: 384\nPlayResY: 288\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,16,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for w in words:
            f.write(f"Dialogue: 0,{format_ass_time(w['start'])},{format_ass_time(w['end'])},Default,,0,0,0,,{w['word']}\n")
