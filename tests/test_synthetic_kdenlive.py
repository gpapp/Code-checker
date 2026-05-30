import pytest
import os
import xml.etree.ElementTree as ET
from unittest.mock import patch
import sys
sys.path.insert(0, '.')
sys.path.insert(0, r'c:\Users\gerge\source\repos\mlt-python\src')
from exporter import generate_kdenlive_project

def tc_to_frames(tc_str, fps=25.0):
    if ":" not in tc_str:
        return int(tc_str)
    parts = tc_str.split(':')
    h, m, s, f = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    return h * 3600 * int(fps) + m * 60 * int(fps) + s * int(fps) + f

def test_synthetic_audio_cutting(synthetic_audio, working_dir):
    """
    Test that synthetic audio with a 5s gap is cut into two entries (0-5, 10-15) with a blank between.
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
            audio_files=None,
            output_path=output_path,
            keep_segments=keep_segments,
            stream_spikes={},
            stream_markers=stream_markers,
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

    # Audio is cut into 2 entries (0-5s, 10-15s) with a blank for the 5-10s silence
    all_children = list(audio_pl)
    entries = audio_pl.findall('entry')
    assert len(entries) == 2, f"Expected 2 audio entries for non-silent intervals, got {len(entries)}"

    # Verify no volume filters on chains (silence is cut, not muted)
    for chain in root.findall('chain'):
        for filt in chain.findall('filter'):
            service = filt.find("property[@name='mlt_service']")
            if service is not None and service.text == 'volume':
                raise AssertionError("Unexpected volume filter on chain (pre-processed FLAC replaces filters)")

    # Verify total timeline = 15s (375 frames at 25fps)
    total_frames = 0
    for child in all_children:
        if child.tag == 'entry':
            in_f = tc_to_frames(child.get('in'), 25.0)
            out_f = tc_to_frames(child.get('out'), 25.0)
            total_frames += out_f - in_f + 1
        elif child.tag == 'blank':
            total_frames += tc_to_frames(child.get('length'), 25.0)

    assert abs(total_frames - int(15 * 25.0)) <= 2, \
        f"Audio timeline mismatch: got {total_frames}, expected {375}"


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
            audio_files=None,
            output_path=output_path,
            keep_segments=keep_segments,
            stream_spikes={},
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
