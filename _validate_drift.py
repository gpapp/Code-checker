"""Validate generated project.kdenlive for zero drift between all tracks.

Computes exact frame counts from serialized XML for every track and
reports any mismatch."""
import sys
import xml.etree.ElementTree as ET

FILE = 'test_data/project.kdenlive'

tree = ET.parse(FILE)
root = tree.getroot()

profile = root.find('profile')
num = float(profile.get('frame_rate_num', 25))
den = float(profile.get('frame_rate_den', 1))
fps = num / den

def parse_tc(tc):
    if tc is None:
        return 0.0
    if ':' not in tc:
        try:
            return float(tc)
        except:
            return 0.0
    parts = tc.split(':')
    if len(parts) == 4:  # HH:MM:SS:FF
        h, m, s, f = map(int, parts)
        return (h * 3600 + m * 60 + s) + f / fps
    elif len(parts) == 3:
        h, m, s_ms = parts
        return int(h) * 3600 + int(m) * 60 + float(s_ms)
    return 0.0

# Find main sequence
seq = None
for t in root.findall('tractor'):
    if t.find("property[@name='kdenlive:sequenceproperties.tracks']") is not None:
        seq = t
        break

print(f'Profile FPS: {fps}')
print()

track_totals = {}

for track_elem in seq.findall('track'):
    prod_id = track_elem.get('producer')
    if prod_id == 'producer0':
        continue

    track_node = root.find(f".//tractor[@id='{prod_id}']")
    if track_node is None:
        track_node = root.find(f".//playlist[@id='{prod_id}']")
    if track_node is None:
        continue

    is_audio = track_node.findtext("property[@name='kdenlive:audio_track']") == '1'
    track_name = track_node.findtext("property[@name='kdenlive:track_name']", default='Unnamed')
    track_type = 'AUDIO' if is_audio else 'VIDEO'

    inner_tracks = track_node.findall('track')
    if not inner_tracks:
        continue
    playlist_id = inner_tracks[0].get('producer')
    playlist = root.find(f".//playlist[@id='{playlist_id}']")
    if playlist is None:
        continue

    total_frames = 0
    clip_count = 0
    blank_count = 0

    for child in playlist:
        if child.tag == 'blank':
            length = int(child.get('length', 0))
            total_frames += length
            blank_count += 1
        elif child.tag == 'entry':
            in_t = parse_tc(child.get('in'))
            out_t = parse_tc(child.get('out'))
            in_f = int(round(in_t * fps))
            out_f = int(round(out_t * fps))
            duration = out_f - in_f + 1  # inclusive out
            total_frames += duration
            clip_count += 1

    track_totals[track_type] = total_frames
    print(f'  [{track_type:5}] "{track_name}": {total_frames:6d} frames ({total_frames/fps:.3f}s), {clip_count} clips, {blank_count} blanks')

print()
durations = list(track_totals.values())
if len(set(durations)) == 1:
    print(f'  [OK] All {len(durations)} tracks match: {durations[0]} frames ({durations[0]/fps:.3f}s)')
else:
    max_dur = max(durations)
    for name, dur in track_totals.items():
        diff = dur - max_dur if dur != max_dur else 0
        print(f'  [DRIFT] {name} is {dur} frames, off by {diff} frames ({diff/fps:.3f}s) from max')
    sys.exit(1)

# Also verify no track exceeds the sequence tractor out point
seq_out = parse_tc(seq.get('out', '0'))
seq_frames = int(round(seq_out * fps)) + 1
print(f'  Sequence out: {seq_frames} frames ({seq_out:.3f}s)')
for name, dur in track_totals.items():
    if dur > seq_frames:
        print(f'  [WARN] {name} ({dur}f) exceeds sequence ({seq_frames}f) by {dur-seq_frames}f')
