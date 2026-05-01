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
    
    with patch("exporter.get_video_duration", return_value=15.0), \
         patch("exporter.has_video_stream", return_value=False):
        
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
    """Test that transitions blend against track 0 (star model)."""
    output_path = os.path.join(working_dir, "transitions.kdenlive")
    # 2 files -> 2 audio tracks (if no video)
    video_files = ["v1.mp3", "v2.mp3"]
    keep_segments = [(0.0, 10.0)]
    
    with patch("exporter.get_video_duration", return_value=10.0), \
         patch("exporter.has_video_stream", return_value=False), \
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
    
    # Find the sequence tractor from main_bin's activetimeline property
    main_bin = root.find(".//playlist[@id='main_bin']")
    active_timeline = main_bin.find("property[@name='kdenlive:docproperties.activetimeline']")
    seq_tr_id = active_timeline.text
    seq_tr = root.find(f".//tractor[@id='{seq_tr_id}']")
    
    # The sequence tractor should have transitions blending against track 0
    transitions = seq_tr.findall("transition")
    # Template has 4 transitions (black to A1, A2, V1, V2)
    assert len(transitions) >= 4
    
    # Check that first two transitions (black to tracks) have a_track=0
    for i in range(min(2, len(transitions))):
        assert transitions[i].find("property[@name='a_track']").text == "0"
