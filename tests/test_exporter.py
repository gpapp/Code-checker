import pytest
import os
import xml.etree.ElementTree as ET
from unittest.mock import patch
from exporter import generate_kdenlive_project

@patch("exporter.get_video_duration", return_value=10.0)
@patch("exporter.has_video_stream", return_value=True)
@patch("os.path.getsize", return_value=1024)
def test_kdenlive_xml_sync(mock_size, mock_has_vid, mock_dur, working_dir):
    output_path = os.path.join(working_dir, "test_kdenlive_xml_sync.kdenlive")
    video_files = ["dummy.mp4"]
    # 25 fps
    # Keep 0-2s and 4-6s
    keep_segments = [(0.0, 2.0), (4.0, 6.0)]
    
    generate_kdenlive_project(
        video_files=video_files,
        output_path=output_path,
        keep_segments=keep_segments,
        stream_spikes_map={},
        overlaps=[],
        repetitions=[],
        fps=25.0
    )
    
    assert os.path.exists(output_path)
    tree = ET.parse(output_path)
    root = tree.getroot()
    
    # Check if entries use the correct frame-based TC
    playlist_entries = root.findall(".//playlist/entry")
    
    found_in_out = []
    for ent in playlist_entries:
        prod = ent.get("producer")
        # We just want entries that have a producer and aren't black track
        if prod and prod != "black_track":
            found_in_out.append((ent.get("in"), ent.get("out")))
            
    # Segment 1: 0 to 2s. 2s * 25 = 50 frames (0 to 49).
    assert found_in_out[0] == ("00:00:00:00", "00:00:01:24")
    # Segment 2: 4 to 6s. 6s * 25 = 150. Out is 149.
    assert found_in_out[1] == ("00:00:04:00", "00:00:05:24")

