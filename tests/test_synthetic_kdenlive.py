import pytest
import os
import xml.etree.ElementTree as ET
from unittest.mock import patch
from exporter import generate_kdenlive_project

def test_synthetic_audio_cutting(synthetic_audio, working_dir):
    """
    Test that synthetic audio with a 5s gap has volume muting for silence.
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
    audio_pl = None
    for pl in root.findall("playlist"):
        props = pl.findall("property[@name='kdenlive:audio_track']")
        if any(p.text == "1" for p in props):
            audio_pl = pl
            break

    assert audio_pl is not None, "Audio playlist not found"

    # Should have entries with volume muting for silence (not blanks)
    entries = audio_pl.findall('entry')
    assert len(entries) > 0, "Expected audio entries"

    # Check for volume filter with mute keyframes at 5-10s (frames 125-249)
    volume_found = False
    for entry in entries:
        for filt in entry.findall('filter'):
            service = filt.find("property[@name='mlt_service']")
            if service is not None and service.text == 'volume':
                gain_prop = filt.find("property[@name='gain']")
                if gain_prop is not None:
                    gain_text = gain_prop.text
                    # Should mute at 5s (frame 125) and unmute at 10s (frame 250)
                    if '125=0' in gain_text or '124=0' in gain_text:
                        volume_found = True
                        # Verify unmute happens after silence (at frame 250 or 251)
                        assert '250=0' in gain_text or '251=1' in gain_text or '250=1' in gain_text, \
                            f"Expected unmute after silence, got: {gain_text}"

    assert volume_found, "No volume filter with mute keyframes found for 5-10s silence"

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
    # 2 audio tracks + black track = 2 transitions (black to A1, black to A2)
    assert len(transitions) == 2, f"Expected 2 transitions, got {len(transitions)}"

    # Check that transitions have a_track=0 (star model)
    for trans in transitions:
        a_track = trans.find("property[@name='a_track']")
        assert a_track is not None, "a_track property not found"
        assert a_track.text == "0", f"Expected a_track=0, got {a_track.text}"
