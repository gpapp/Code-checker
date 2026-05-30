import sys
import xml.etree.ElementTree as ET
from pathlib import Path

def parse_timecode(tc, fps):
    if tc is None:
        return 0
    if ":" not in tc:
        try:
            return int(tc)
        except:
            return 0
    parts = tc.split(":")
    try:
        if len(parts) == 4:  # HH:MM:SS:FF
            h, m, s, f = map(int, parts)
            return int(round((h * 3600 + m * 60 + s) * fps + f))
        elif len(parts) == 3:  # HH:MM:SS.mmm
            h, m, s_ms = parts
            total_seconds = int(h) * 3600 + int(m) * 60 + float(s_ms)
            return int(round(total_seconds * fps))
    except:
        pass
    return 0

def analyze_drift(file_path):
    print(f"Analyzing: {file_path}")
    tree = ET.parse(file_path)
    root = tree.getroot()

    profile = root.find("profile")
    fps = 25.0
    if profile is not None:
        try:
            num = float(profile.get("frame_rate_num", 25))
            den = float(profile.get("frame_rate_den", 1))
            fps = num / den
        except:
            pass
    print(f"FPS: {fps}")

    seq_tractor = None
    for t in root.findall("tractor"):
        if t.find("property[@name='kdenlive:sequenceproperties.tracks']") is not None:
            seq_tractor = t
            break

    if seq_tractor is None:
        print("Error: Could not find sequence tractor.")
        return

    # Let's map tracks
    id_map = {child.get("id"): child for child in root if child.get("id")}
    tracks = seq_tractor.findall("track")

    track_clips = {}
    for i, track in enumerate(tracks):
        prod_id = track.get("producer")
        if prod_id == "producer0" or not prod_id:
            continue

        track_node = id_map.get(prod_id)
        if track_node is None:
            continue

        is_audio = track_node.findtext("property[@name='kdenlive:audio_track']") == "1"
        track_name = track_node.findtext("property[@name='kdenlive:track_name']", default="Unnamed")
        track_type = "Audio" if is_audio else "Video"

        inner_tracks = track_node.findall("track")
        if not inner_tracks:
            continue

        playlist_id = inner_tracks[0].get("producer")
        playlist = id_map.get(playlist_id)
        if playlist is None:
            continue

        # Get all entries/blanks
        timeline = []
        current_frame = 0
        for child in playlist:
            if child.tag == "blank":
                length = int(child.get("length", 0))
                current_frame += length
            elif child.tag == "entry":
                in_f = parse_timecode(child.get("in"), fps)
                out_f = parse_timecode(child.get("out"), fps)
                duration = out_f - in_f + 1
                timeline.append({
                    "start": current_frame,
                    "end": current_frame + duration,
                    "duration": duration,
                    "in": in_f,
                    "out": out_f,
                    "producer": child.get("producer")
                })
                current_frame += duration
        
        track_clips[i] = {
            "name": track_name,
            "type": track_type,
            "timeline": timeline,
            "total_length": current_frame
        }

    video_tracks = {idx: info for idx, info in track_clips.items() if info["type"] == "Video"}
    audio_tracks = {idx: info for idx, info in track_clips.items() if info["type"] == "Audio"}

    if not video_tracks:
        print("No video tracks found.")
        return
    
    ref_idx, ref_track = list(video_tracks.items())[0]
    print(f"\nReference Video Track: {ref_track['name']} ({len(ref_track['timeline'])} clips)")

    for a_idx, a_track in audio_tracks.items():
        print(f"\nAnalyzing Audio Track: {a_track['name']}")
        
        # We want to map each keep segment (video clip) to its corresponding audio track content.
        # For each video clip: (v_start, v_end), check what audio clips or blanks exist in that range.
        mismatches = []
        perfect_alignments = 0
        
        for j, v_clip in enumerate(ref_track["timeline"]):
            v_start, v_end = v_clip["start"], v_clip["end"]
            
            # Find audio clips that overlap with this video clip's timeline range [v_start, v_end]
            overlapping_audios = []
            for a_clip in a_track["timeline"]:
                # If audio clip overlaps with [v_start, v_end]
                if not (a_clip["end"] <= v_start or a_clip["start"] >= v_end):
                    overlapping_audios.append(a_clip)
            
            if not overlapping_audios:
                # Video segment is entirely silent (no audio clips)
                # Let's check if the silence is expected
                pass
            else:
                a_first = overlapping_audios[0]
                a_last = overlapping_audios[-1]
                
                # Check if first audio starts exactly at v_start, and last ends exactly at v_end
                starts_aligned = (a_first["start"] == v_start)
                ends_aligned = (a_last["end"] == v_end)
                
                if starts_aligned and ends_aligned and len(overlapping_audios) == 1:
                    perfect_alignments += 1
                else:
                    mismatches.append({
                        "video_idx": j,
                        "video_range": (v_start, v_end),
                        "audio_clips": [(a["start"], a["end"]) for a in overlapping_audios],
                        "starts_aligned": starts_aligned,
                        "ends_aligned": ends_aligned
                    })
                    
        print(f"  Perfect alignments (exactly 1 audio clip matching video clip boundaries): {perfect_alignments} / {len(ref_track['timeline'])}")
        print(f"  Mismatched segments: {len(mismatches)}")
        
        if mismatches:
            print("\n  First 10 mismatches:")
            for m in mismatches[:10]:
                print(f"    Video Clip {m['video_idx']}: Timeline [{m['video_range'][0]}, {m['video_range'][1]}] (dur={m['video_range'][1]-m['video_range'][0]})")
                print(f"      Audio clips: {m['audio_clips']}")
                print(f"      Starts aligned: {m['starts_aligned']}, Ends aligned: {m['ends_aligned']}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_drift(sys.argv[1])
    else:
        print("Please provide the path to the Kdenlive project.")
