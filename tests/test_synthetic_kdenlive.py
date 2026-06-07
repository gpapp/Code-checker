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
        sec_parts = parts[-1].split('.')
        seconds = float(sec_parts[0]) + float(sec_parts[1]) / 1000.0
        h, m = int(parts[0]), int(parts[1])
        return h * 3600 + m * 60 + seconds
    else:
        h, m, s, f = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        return h * 3600 + m * 60 + s + f / 30.0


def test_synthetic_audio_cutting(synthetic_audio, working_dir):
    """
    Test that synthetic audio with a 5s gap is cut into two entries (0-5, 10-15) with a blank between.
    """
    output_path = os.path.join(working_dir, "synthetic.kdenlive")
    video_files = [synthetic_audio]

    keep_segments = [(0.0, 15.0)]

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
        )

    assert os.path.exists(output_path)
    tree = ET.parse(output_path)
    root = tree.getroot()

    audio_pl = None
    for pl in root.findall("playlist"):
        props = pl.findall("property[@name='kdenlive:audio_track']")
        if any(p.text == "1" for p in props):
            audio_pl = pl
            break

    assert audio_pl is not None, "Audio playlist not found"

    all_children = list(audio_pl)
    entries = audio_pl.findall('entry')
    assert len(entries) == 2, f"Expected 2 audio entries for non-silent intervals, got {len(entries)}"

    for chain in root.findall('chain'):
        for filt in chain.findall('filter'):
            service = filt.find("property[@name='mlt_service']")
            if service is not None and service.text == 'volume':
                raise AssertionError("Unexpected volume filter on chain (pre-processed FLAC replaces filters)")

    # Verify total timeline = 15s (compute from timecodes)
    root_profile = root.find("profile")
    fps = float(root_profile.get("frame_rate_num")) / float(root_profile.get("frame_rate_den")) if root_profile is not None else 30.0

    total_seconds = 0.0
    for child in all_children:
        if child.tag == 'entry':
            in_s = tc_to_seconds(child.get('in'))
            out_s = tc_to_seconds(child.get('out'))
            total_seconds += out_s - in_s
        elif child.tag == 'blank':
            frame_len = int(child.get('length', '0'))
            total_seconds += frame_len / fps

    assert abs(total_seconds - 15.0) <= 0.1, \
        f"Audio timeline mismatch: got {total_seconds}, expected {15.0}"


def test_chained_transitions(working_dir):
    """Test that transitions blend against track 0 (star model)."""
    output_path = os.path.join(working_dir, "transitions.kdenlive")
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
        )

    tree = ET.parse(output_path)
    root = tree.getroot()

    main_bin = root.find(".//playlist[@id='main_bin']")
    active_timeline = main_bin.find("property[@name='kdenlive:docproperties.activetimeline']")
    seq_tr_id = active_timeline.text
    seq_tr = root.find(f".//tractor[@id='{seq_tr_id}']")

    transitions = seq_tr.findall("transition")
    assert len(transitions) == 2, f"Expected 2 transitions, got {len(transitions)}"

    for trans in transitions:
        a_track = trans.find("property[@name='a_track']")
        assert a_track is not None, "a_track property not found"
        assert a_track.text == "0", f"Expected a_track=0, got {a_track.text}"
