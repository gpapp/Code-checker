import pytest
import os
import xml.etree.ElementTree as ET
import json
from unittest.mock import patch
import sys
sys.path.insert(0, '.')
sys.path.insert(0, r'c:\Users\gerge\source\repos\mlt-python\src')
from exporter import generate_kdenlive_project
from mlt_python.marker import Marker, markers_from_json, markers_to_json


@patch("exporter.get_video_duration", return_value=30.0)
@patch("exporter.has_video_stream", return_value=True)
@patch("os.path.exists", return_value=True)
def test_filler_markers_use_mlt_python_api(mock_exists, mock_has_vid, mock_dur, working_dir):
    """Test that filler markers are added as kdenlive:markers on audio chains."""
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
    
    # Find the chain for audio.wav
    audio_chain = None
    for chain in root.findall(".//chain"):
        res = chain.find("property[@name='resource']")
        if res is not None and audio_path in res.text:
            audio_chain = chain
            break
    assert audio_chain is not None, "Audio chain should exist"
    
    markers_prop = audio_chain.find("property[@name='kdenlive:markers']")
    assert markers_prop is not None, "Filler markers should exist on audio chain"
    
    parsed_markers = json.loads(markers_prop.text)
    assert len(parsed_markers) == 3
    
    # Verify marker positions (in frames at 25fps, banker's rounding)
    assert parsed_markers[0]["pos"] == 38  # round(1.5 * 25) = round(37.5) = 38
    assert parsed_markers[0]["duration"] == 25  # round(1.0 * 25) = 25
    assert parsed_markers[0]["comment"] == "Filler"
    assert parsed_markers[1]["pos"] == 125  # 5.0s * 25 = 125
    assert parsed_markers[2]["pos"] == 250  # 10.0s * 25 = 250
    assert parsed_markers[2]["duration"] == 50  # 2.0s * 25 = 50


@patch("exporter.get_video_duration", return_value=30.0)
@patch("exporter.has_video_stream", return_value=True)
@patch("os.path.exists", return_value=True)
def test_filler_marker_no_fillers(mock_exists, mock_has_vid, mock_dur, working_dir):
    """Test that no markers exist when no fillers are provided."""
    output_path = os.path.join(working_dir, "test_marker_format.kdenlive")
    
    video_files = ["video.mkv"]
    audio_path = "audio.wav"
    
    stream_markers = {
        audio_path: {
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
        assert markers_prop is None, "No markers should exist when no fillers provided"


def test_marker_api_direct():
    """Test mlt-python Marker API directly without exporter."""
    
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
