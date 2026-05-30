import xml.etree.ElementTree as ET
import sys

def parse_timecode(tc, fps):
    if tc is None:
        return 0
    if ":" not in tc:
        try: return int(tc)
        except: return 0
    parts = tc.split(":")
    try:
        if len(parts) == 4:
            h, m, s, f = map(int, parts)
            return int(round((h * 3600 + m * 60 + s) * fps + f))
        elif len(parts) == 3:
            h, m, s_ms = parts
            total_seconds = int(h) * 3600 + int(m) * 60 + float(s_ms)
            return int(round(total_seconds * fps))
    except: pass
    return 0

def check_mismatches(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    profile = root.find("profile")
    fps = 25.0
    if profile is not None:
        try:
            num = float(profile.get("frame_rate_num", 25))
            den = float(profile.get("frame_rate_den", 1))
            fps = num / den
        except: pass

    seq_tractor = next((t for t in root.findall("tractor") 
                       if t.find("property[@name='kdenlive:sequenceproperties.tracks']") is not None), None)
    id_map = {child.get("id"): child for child in root if child.get("id")}
    
    # Get reference video timeline
    ref_track = None
    for track in seq_tractor.findall("track"):
        prod_id = track.get("producer")
        if prod_id == "producer0" or not prod_id: continue
        track_node = id_map.get(prod_id)
        if track_node is None: continue
        is_audio = track_node.findtext("property[@name='kdenlive:audio_track']") == "1"
        if not is_audio:
            ref_track = track_node
            break
            
    inner_tracks = ref_track.findall("track")
    playlist_id = inner_tracks[0].get("producer")
    playlist = id_map.get(playlist_id)
    
    ref_timeline = []
    current_frame = 0
    for child in playlist:
        if child.tag == "blank":
            current_frame += int(child.get("length", 0))
        elif child.tag == "entry":
            in_f = parse_timecode(child.get("in"), fps)
            out_f = parse_timecode(child.get("out"), fps)
            duration = out_f - in_f + 1
            ref_timeline.append((current_frame, current_frame + duration))
            current_frame += duration

    # Check audio track mismatches from start
    for track in seq_tractor.findall("track"):
        prod_id = track.get("producer")
        if prod_id == "producer0" or not prod_id: continue
        track_node = id_map.get(prod_id)
        if track_node is None: continue
        is_audio = track_node.findtext("property[@name='kdenlive:audio_track']") == "1"
        if not is_audio: continue
        
        track_name = track_node.findtext("property[@name='kdenlive:track_name']", default="Unnamed")
        inner_tracks = track_node.findall("track")
        if not inner_tracks: continue
        playlist_id = inner_tracks[0].get("producer")
        playlist = id_map.get(playlist_id)
        
        print(f"\nChecking track: {track_name}")
        current_frame = 0
        clip_idx = 0
        
        for child in playlist:
            if child.tag == "blank":
                current_frame += int(child.get("length", 0))
            elif child.tag == "entry":
                in_f = parse_timecode(child.get("in"), fps)
                out_f = parse_timecode(child.get("out"), fps)
                duration = out_f - in_f + 1
                
                # Let's find which video keep segment contains this audio clip's center
                clip_center = current_frame + duration // 2
                matching_vid_idx = None
                for idx, (vs, ve) in enumerate(ref_timeline):
                    if vs <= clip_center < ve:
                        matching_vid_idx = idx
                        break
                
                if matching_vid_idx is not None:
                    vs, ve = ref_timeline[matching_vid_idx]
                    # Check start and end mismatches
                    # Since this is grouped by keep segments:
                    # In our exporter, we stretched the first clip of keep segment to vs, and last to ve.
                    # Let's see if the very first clip in this keep segment starts at vs, and very last ends at ve.
                    # We can just check the first mismatch on the timeline!
                    diff_start = current_frame - vs
                    # If this is the first clip in the keep segment (it shouldn't have preceding blanks in same keep segment)
                    # Let's check the mismatch
                    if current_frame < vs or current_frame > ve:
                        print(f"  Clip {clip_idx} timeline range: [{current_frame}, {current_frame+duration}]")
                        print(f"    Mismatched: center falls in Video Clip {matching_vid_idx} [{vs}, {ve}]!")
                        print(f"    This audio clip is OUTSIDE its matching video clip timeline boundaries!")
                        return

                current_frame += duration
                clip_idx += 1

if __name__ == "__main__":
    check_mismatches(sys.argv[1])
