import pytest
import os
import xml.etree.ElementTree as ET
from unittest.mock import patch
from exporter import generate_kdenlive_project

def test_synthetic_audio_cutting(synthetic_audio, working_dir):
    """
    Test that synthetic audio with a 5s gap is correctly cut into two entries 
    with a blank in between in the Kdenlive XML.
    """
    output_path = os.path.join(working_dir, "synthetic.kdenlive")
    # Fixture provides a 15s file: 0-5s speech, 5-10s silence, 10-15s speech
    video_files = [synthetic_audio]
    
    # We want to keep the whole 15s
    keep_segments = [(0.0, 15.0)]
    
    # Mock markers: silence from 5s to 10s
    stream_markers = {
        synthetic_audio: {
            "silence": [(5.0, 10.0)],
            "spikes": []
        }
    }
    
    with patch("audio_utils.get_video_duration", return_value=15.0), \
         patch("audio_utils.has_video_stream", return_value=False):
        
        generate_kdenlive_project(
            video_files=video_files,
            output_path=output_path,
            keep_segments=keep_segments,
            stream_spikes_map={},
            overlaps=[],
            repetitions=[],
            stream_markers_global=stream_markers,
            fps=25.0
        )
    
    assert os.path.exists(output_path)
    tree = ET.parse(output_path)
    root = tree.getroot()
    
    # Find the audio playlist (the one with kdenlive:audio_track=1)
    # Since it's a non-video file, num_video_tracks=0, so it should be the first playlist (playlist0)
    # Wait, black_track is first...
    
    audio_pl = None
    for pl in root.findall("playlist"):
        props = pl.findall("property[@name='kdenlive:audio_track']")
        if any(p.text == "1" for p in props):
            audio_pl = pl
            break
            
    assert audio_pl is not None, "Audio playlist not found"
    
    # Expected structure:
    # 1. Entry: 0-5s (0 to 124 frames)
    # 2. Blank: 5s (125 frames)
    # 3. Entry: 10-15s (250 to 374 frames)
    
    elements = list(audio_pl)
    # First elements might be properties, skip them
    pl_items = [e for e in elements if e.tag in ["entry", "blank"]]
    
    assert len(pl_items) == 3
    
    # Item 1: Entry (0-5s)
    assert pl_items[0].tag == "entry"
    assert pl_items[0].get("in") == "00:00:00:00"
    assert pl_items[0].get("out") == "00:00:04:24" # Frame 124
    
    # Item 2: Blank (5s)
    assert pl_items[1].tag == "blank"
    # Duration of 5s is 125 frames
    assert pl_items[1].get("length") == "00:00:05:00"
    
    # Item 3: Entry (10-15s)
    assert pl_items[2].tag == "entry"
    assert pl_items[2].get("in") == "00:00:10:00"
    assert pl_items[2].get("out") == "00:00:14:24" # Frame 374 (15*25 - 1)

def test_chained_transitions(working_dir):
    """Test that transitions use a_track=ti-1 for chaining."""
    output_path = os.path.join(working_dir, "transitions.kdenlive")
    # 2 files -> 2 audio tracks (if no video)
    video_files = ["v1.mp3", "v2.mp3"]
    keep_segments = [(0.0, 10.0)]
    
    with patch("audio_utils.get_video_duration", return_value=10.0), \
         patch("audio_utils.has_video_stream", return_value=False), \
         patch("os.path.getsize", return_value=1000):
        
        generate_kdenlive_project(
            video_files=video_files,
            output_path=output_path,
            keep_segments=keep_segments,
            stream_spikes_map={},
            overlaps=[],
            repetitions=[],
            fps=25.0
        )
        
    tree = ET.parse(output_path)
    root = tree.getroot()
    
    # Sequence tractor should have transitions
    seq_tr = root.find(".//tractor[@id='tractor2']") # 2 audio tracks + black_track = tractor2? 
    # Wait, num_video_tracks=0, num_audio_tracks=2. tractor_idx: 0, 1. seq_tr = tractor2. Correct.
    
    transitions = seq_tr.findall("transition")
    assert len(transitions) == 2
    
    # Transition 1: Black (0) to Audio 1 (1)
    assert transitions[0].find("property[@name='a_track']").text == "0"
    assert transitions[0].find("property[@name='b_track']").text == "1"
    
    # Transition 2: Audio 1 (1) to Audio 2 (2)
    assert transitions[1].find("property[@name='a_track']").text == "1"
    assert transitions[1].find("property[@name='b_track']").text == "2"
