import pytest
import os
import xml.etree.ElementTree as ET
from unittest.mock import patch, MagicMock
from exporter import generate_kdenlive_project, frames_to_tc

@patch("exporter.get_video_duration", return_value=10.0)
@patch("exporter.has_video_stream")
@patch("os.path.exists", return_value=True)
def test_multiple_audio_tracks_assignment(mock_exists, mock_has_vid, mock_dur, working_dir):
    output_path = os.path.join(working_dir, "test_multi_audio.kdenlive")
    # One video and two separate audio files
    video_files = ["video.mkv", "audio1.mp3", "audio2.mp3"]

    # Mock video.mkv as having video, others not
    mock_has_vid.side_effect = lambda x: x == "video.mkv"

    # audio_info_map: temp_wav -> {"original": path, "stream_idx": idx}
    audio_info_map = {
        "temp_a1.wav": {"original": "audio1.mp3", "stream_idx": 0},
        "temp_a2.wav": {"original": "audio2.mp3", "stream_idx": 0}
    }

    video_to_audio_map = {
        "video.mkv": [],
        "audio1.mp3": ["temp_a1.wav"],
        "audio2.mp3": ["temp_a2.wav"]
    }

    keep_segments = [(0.0, 5.0)]

    # Silence only on audio2
    stream_markers = {
        "temp_a1.wav": {"silence": [], "spikes": []},
        "temp_a2.wav": {"silence": [(1.0, 2.0)], "spikes": []}
    }

    generate_kdenlive_project(
        video_files=video_files,
        output_path=output_path,
        keep_segments=keep_segments,
        stream_spikes_map={},
        overlaps=[],
        repetitions=[],
        fps=25.0,
        video_to_audio_map=video_to_audio_map,
        stream_markers_global=stream_markers,
        audio_info_map=audio_info_map
    )

    tree = ET.parse(output_path)
    root = tree.getroot()

    # V1 (playlist4) should have video.mkv
    v1_pl = root.find(".//playlist[@id='playlist4']")
    v1_entry = v1_pl.find("entry")
    v1_chain_id = v1_entry.get("producer")
    v1_chain = root.find(f".//chain[@id='{v1_chain_id}']")
    assert "video.mkv" in v1_chain.find("property[@name='resource']").text

    # Check all audio tracks
    a1_pl = root.find(".//playlist[@id='playlist0']")
    a2_pl = root.find(".//playlist[@id='playlist2']")

    a1_entry = a1_pl.find("entry")
    a2_entry = a2_pl.find("entry")

    a1_chain = root.find(f".//chain[@id='{a1_entry.get('producer')}']")
    a2_chain = root.find(f".//chain[@id='{a2_entry.get('producer')}']")

    resources = [
        a1_chain.find("property[@name='resource']").text,
        a2_chain.find("property[@name='resource']").text
    ]

    # Both audio1 and audio2 (ORIGINAL) should be present on audio tracks
    assert any("audio1.mp3" in r for r in resources)
    assert any("audio2.mp3" in r for r in resources)

    # Identify which track is audio2 to check filters
    target_entry = None
    if a1_chain is not None and "audio2.mp3" in a1_chain.find("property[@name='resource']").text:
        target_entry = a1_entry
    elif a2_chain is not None and "audio2.mp3" in a2_chain.find("property[@name='resource']").text:
        target_entry = a2_entry

    # Check volume filter on the entry that has audio2
    vol_filter = target_entry.find("filter") if target_entry is not None else None
    assert vol_filter is not None
    gain_prop = vol_filter.find("property[@name='gain']")
    assert gain_prop is not None
    # Silence was at 1.0-2.0s. At 25fps, that's frames 25-50.
    assert "25=0" in gain_prop.text
    assert "50=0" in gain_prop.text

    # Check for bin effects (filters on chains)
    def find_filter(chain, service):
        for f in chain.findall("filter"):
            for p in f.findall("property"):
                if p.get("name") == "mlt_service" and p.text == service:
                    return f
        return None

    assert find_filter(a1_chain, "ladspa.1073") is not None
    assert find_filter(a1_chain, "dynamic_loudness") is not None

@patch("exporter.get_video_duration", return_value=10.0)
@patch("exporter.has_video_stream")
@patch("os.path.exists", return_value=True)
def test_video_offsets(mock_exists, mock_has_vid, mock_dur, working_dir):
    output_path = os.path.join(working_dir, "test_offsets.kdenlive")
    video_files = ["video1.mkv", "video2.mkv"]
    mock_has_vid.return_value = True

    # 0.5s offset for video1, -0.2s for video2
    video_offsets = [0.5, -0.2]
    keep_segments = [(1.0, 3.0)] # Global time

    generate_kdenlive_project(
        video_files=video_files,
        output_path=output_path,
        keep_segments=keep_segments,
        stream_spikes_map={},
        overlaps=[],
        repetitions=[],
        fps=25.0,
        video_offsets=video_offsets
    )

    tree = ET.parse(output_path)
    root = tree.getroot()

    # Check entries for both videos
    entries = root.findall(".//playlist/entry")
    # Should find entries for video1 and video2 (and their audio if not ignored)
    # Our mocks don't ignore audio since no separate audio was provided.

    # Video 1: 1.0+0.5 = 1.5s to 3.0+0.5 = 3.5s. Frames: 37.5 (38) to 87.5 (88).
    # Video 2: 1.0-0.2 = 0.8s to 3.0-0.2 = 2.8s. Frames: 20 to 70.

    found_in_out = []
    for ent in entries:
        prod = ent.get("producer")
        chain = root.find(f".//chain[@id='{prod}']")
        if chain is not None:
            res = chain.find("property[@name='resource']").text
            found_in_out.append((res, ent.get("in"), ent.get("out")))

    assert any("video1.mkv" in r and i == "00:00:01:13" and o == "00:00:03:12" for r, i, o in found_in_out)
    assert any("video2.mkv" in r and i == "00:00:00:20" and o == "00:00:02:19" for r, i, o in found_in_out)
