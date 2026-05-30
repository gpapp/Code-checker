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

def dump_near(file_path, target_frame):
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
    if not seq_tractor:
        print("No sequence tractor")
        return

    id_map = {child.get("id"): child for child in root if child.get("id")}
    
    for i, track in enumerate(seq_tractor.findall("track")):
        prod_id = track.get("producer")
        if prod_id == "producer0" or not prod_id: continue
        
        track_node = id_map.get(prod_id)
        if track_node is None: continue
        
        is_audio = track_node.findtext("property[@name='kdenlive:audio_track']") == "1"
        track_name = track_node.findtext("property[@name='kdenlive:track_name']", default="Unnamed")
        track_type = "Audio" if is_audio else "Video"
        
        inner_tracks = track_node.findall("track")
        if not inner_tracks: continue
        playlist_id = inner_tracks[0].get("producer")
        playlist = id_map.get(playlist_id)
        if playlist is None: continue
        
        print(f"\n--- {track_name} ({track_type}) ---")
        current_frame = 0
        for child in playlist:
            if child.tag == "blank":
                length = int(child.get("length", 0))
                if current_frame <= target_frame <= current_frame + length:
                    print(f"  [BLANK] Timeline: {current_frame} to {current_frame+length} (len={length})")
                current_frame += length
            elif child.tag == "entry":
                in_f = parse_timecode(child.get("in"), fps)
                out_f = parse_timecode(child.get("out"), fps)
                duration = out_f - in_f + 1
                if current_frame - 200 <= target_frame <= current_frame + duration + 200:
                    print(f"  [ENTRY] Timeline: {current_frame} to {current_frame+duration} (dur={duration}) | Source in={in_f} out={out_f}")
                current_frame += duration

if __name__ == "__main__":
    dump_near(sys.argv[1], int(sys.argv[2]))
