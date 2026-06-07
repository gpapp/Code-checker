import pytest
import os
import xml.etree.ElementTree as ET
from unittest.mock import patch
import sys
sys.path.insert(0, '.')
sys.path.insert(0, r'c:\Users\gerge\source\repos\mlt-python\src')
from exporter import generate_kdenlive_project


def tc_to_seconds(tc_str):
    """Convert HH:MM:SS.mmm or HH:MM:SS:FF string to float seconds."""
    if ":" not in tc_str:
        return float(tc_str)
    parts = tc_str.split(':')
    if '.' in parts[-1]:
        # HH:MM:SS.mmm format
        sec_parts = parts[-1].split('.')
        seconds = float(sec_parts[0]) + float(sec_parts[1]) / 1000.0
        h, m = int(parts[0]), int(parts[1])
        return h * 3600 + m * 60 + seconds
    else:
        # HH:MM:SS:FF format
        h, m, s, f = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        return h * 3600 + m * 60 + s + f / 30.0


@patch("exporter.get_video_duration", return_value=10.0)
@patch("exporter.has_video_stream", return_value=True)
@patch("os.path.getsize", return_value=1024)
def test_kdenlive_xml_sync(mock_size, mock_has_vid, mock_dur, working_dir):
    output_path = os.path.join(working_dir, "test_kdenlive_xml_sync.kdenlive")
    video_files = ["dummy.mp4"]
    keep_segments = [(0.0, 2.0), (4.0, 6.0)]
    
    generate_kdenlive_project(
        video_files=video_files,
        audio_files=None,
        output_path=output_path,
        keep_segments=keep_segments,
        stream_spikes={},
    )
    
    assert os.path.exists(output_path)
    tree = ET.parse(output_path)
    root = tree.getroot()
    
    # Check if entries use the correct time-based TC (HH:MM:SS.mmm)
    playlist_entries = []
    for pl in root.findall(".//playlist"):
        if pl.get("id") != "main_bin":
            playlist_entries.extend(pl.findall("entry"))
    
    found_in_out = []
    for ent in playlist_entries:
        prod = ent.get("producer")
        if prod and prod != "black_track":
            found_in_out.append((ent.get("in"), ent.get("out")))
    
    # Segment 1: 0 to 2s -> in="00:00:00.000" out="00:00:01.960" (inclusive: 2s - 1/25)
    assert found_in_out[0][0] == "00:00:00.000"
    assert found_in_out[0][1] == "00:00:01.960"
    # Segment 2: 4 to 6s -> in="00:00:04.000" out="00:00:05.960" (inclusive: 6s - 1/25)
    assert found_in_out[1][0] == "00:00:04.000"
    assert found_in_out[1][1] == "00:00:05.960"
