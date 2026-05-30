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
    """Test that filler markers are added using mlt-python Marker API (JSON format)."""
    output_path = os.path.join(working_dir, "test_markers.kdenlive")
    
    video_files = ["video.mkv"]
    audio_path = "audio.wav"
    
    # Define fillers as (start, end) in seconds
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
        fps=25.0
    )
    
    assert os.path.exists(output_path)
    tree = ET.parse(output_path)
    root = tree.getroot()
    
    # Find any producer that has kdenlive:markers property
    marker_producer = None
    for producer in root.findall(".//producer"):
        markers_prop = producer.find("property[@name='kdenlive:markers']")
        if markers_prop is not None:
            marker_producer = producer
            break
    
    assert marker_producer is not None, "No producer found with kdenlive:markers property"
    
    # Get the markers JSON
    markers_prop = marker_producer.find("property[@name='kdenlive:markers']")
    markers_json = markers_prop.text
    assert markers_json is not None, "Markers JSON is empty"
    
    # Parse the JSON and verify it matches mlt-python Marker format
    markers = markers_from_json(markers_json)
    
    assert len(markers) == len(fillers), f"Expected {len(fillers)} markers, got {len(markers)}"
    
    # Verify each marker
    for i, (start, end) in enumerate(fillers):
        marker = markers[i]
        expected_frame = int(start * 25.0)  # 25 fps
        expected_duration = int((end - start) * 25.0)
        
        assert marker.pos == expected_frame, f"Marker {i}: expected pos={expected_frame}, got {marker.pos}"
        assert marker.comment == "Filler", f"Marker {i}: expected comment='Filler', got '{marker.comment}'"
        assert marker.marker_type == 0, f"Marker {i}: expected type=0, got {marker.marker_type}"
        assert marker.duration == expected_duration, f"Marker {i}: expected duration={expected_duration}, got {marker.duration}"


@patch("exporter.get_video_duration", return_value=30.0)
@patch("exporter.has_video_stream", return_value=True)
@patch("os.path.exists", return_value=True)
def test_marker_json_format_valid(mock_exists, mock_has_vid, mock_dur, working_dir):
    """Test that the marker JSON format is valid and parseable by mlt-python."""
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
        fps=25.0
    )
    
    tree = ET.parse(output_path)
    root = tree.getroot()
    
    # Find producer with markers
    for producer in root.findall(".//producer"):
        markers_prop = producer.find("property[@name='kdenlive:markers']")
        if markers_prop is not None:
            markers_json = markers_prop.text
            
            # Verify it's valid JSON
            parsed = json.loads(markers_json)
            assert isinstance(parsed, list), "Markers JSON should be a list"
            assert len(parsed) == 1, "Should have 1 marker"
            
            # Verify structure matches mlt-python Marker dict format
            marker_dict = parsed[0]
            assert "pos" in marker_dict, "Marker dict should have 'pos' key"
            assert "comment" in marker_dict, "Marker dict should have 'comment' key"
            assert "type" in marker_dict, "Marker dict should have 'type' key"
            assert "duration" in marker_dict, "Marker dict should have 'duration' key"
            
            # Verify markers_from_json can parse it
            markers = markers_from_json(markers_json)
            assert len(markers) == 1
            assert isinstance(markers[0], Marker)
            break


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
