import pytest
import os
import xml.etree.ElementTree as ET
import json
from unittest.mock import patch
import sys
sys.path.insert(0, '.')
sys.path.insert(0, r'c:\Users\gerge\source\repos\mlt-python\src')
from exporter import generate_kdenlive_project
from mlt_python.marker import Marker, markers_from_json


@patch("exporter.get_video_duration", return_value=30.0)
@patch("exporter.has_video_stream", return_value=True)
@patch("os.path.exists", return_value=True)
def test_filler_markers_use_mlt_python_api(mock_exists, mock_has_vid, mock_dur, working_dir):
    """Test that no filler markers exist in output (marker support was removed)."""
    output_path = os.path.join(working_dir, "test_markers.kdenlive")
    
    video_files = ["video.mkv"]
    audio_path = "audio.wav"
    
    fillers = [(1.5, 2.5), (5.0, 6.0), (10.0, 12.0)]
    
    stream_markers = {
        audio_path: {
            "fillers": fillers,
            "silence": [],
            "spikes": []
        }
    }
    
    audio_files = {
        "video.mkv": [{"original": audio_path, "stream_idx": 0, "temp_path": audio_path}]
    }
    
    generate_kdenlive_project(
        video_files=video_files,
        audio_files=audio_files,
        output_path=output_path,
        keep_segments=[(0.0, 30.0)],
        stream_spikes={},
        video_offsets=[0.0],
        ass_paths=[],
        asr_words=[[]],
        stream_markers=stream_markers,
    )
    
    assert os.path.exists(output_path)
    tree = ET.parse(output_path)
    root = tree.getroot()
    
    for chain in root.findall(".//chain"):
        markers_prop = chain.find("property[@name='kdenlive:markers']")
        assert markers_prop is None, "No markers should exist in output"


@patch("exporter.get_video_duration", return_value=30.0)
@patch("exporter.has_video_stream", return_value=True)
@patch("os.path.exists", return_value=True)
def test_marker_json_format_valid(mock_exists, mock_has_vid, mock_dur, working_dir):
    """Test that no markers exist in output (marker support was removed)."""
    output_path = os.path.join(working_dir, "test_marker_format.kdenlive")
    
    video_files = ["video.mkv"]
    audio_path = "audio.wav"
    
    fillers = [(3.0, 4.0)]
    
    stream_markers = {
        audio_path: {
            "fillers": fillers,
            "silence": [],
            "spikes": []
        }
    }
    
    audio_files = {
        "video.mkv": [{"original": audio_path, "stream_idx": 0, "temp_path": audio_path}]
    }
    
    generate_kdenlive_project(
        video_files=video_files,
        audio_files=audio_files,
        output_path=output_path,
        keep_segments=[(0.0, 30.0)],
        stream_spikes={},
        video_offsets=[0.0],
        ass_paths=[],
        asr_words=[[]],
        stream_markers=stream_markers,
    )
    
    tree = ET.parse(output_path)
    root = tree.getroot()
    
    for chain in root.findall(".//chain"):
        markers_prop = chain.find("property[@name='kdenlive:markers']")
        assert markers_prop is None, "No markers should exist in output"


def test_marker_api_direct():
    """Test mlt-python Marker API directly without exporter."""
    from mlt_python.marker import markers_to_json
    
    # Create markers using the API
    markers = [
        Marker(pos=37, comment="Filler", marker_type=0, duration=25),
        Marker(pos=125, comment="Filler", marker_type=0, duration=25)
    ]
    
    # Serialize to JSON
    json_str = markers_to_json(markers)
    
    # Verify it's valid JSON
    parsed = json.loads(json_str)
    assert len(parsed) == 2
    
    # Deserialize back
    restored = markers_from_json(json_str)
    assert len(restored) == 2
    assert restored[0].pos == 37
    assert restored[0].comment == "Filler"
    assert restored[1].pos == 125
    
    # Test with duration=0 (point marker)
    point_marker = Marker(pos=100, comment="Point", marker_type=1, duration=0)
    json_str = markers_to_json([point_marker])
    restored = markers_from_json(json_str)
    assert restored[0].duration == 0
    assert restored[0].marker_type == 1
