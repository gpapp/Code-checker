import pytest
import os
import xml.etree.ElementTree as ET
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, '.')
sys.path.insert(0, r'c:\Users\gerge\source\repos\mlt-python\src')
from exporter import generate_kdenlive_project

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

    audio_files = {
        "video.mkv": [],
        "audio1.mp3": [{"original": "audio1.mp3", "stream_idx": 0, "temp_path": "temp_a1.wav"}],
        "audio2.mp3": [{"original": "audio2.mp3", "stream_idx": 0, "temp_path": "temp_a2.wav"}]
    }
    generate_kdenlive_project(
        video_files=video_files,
        audio_files=audio_files,
        output_path=output_path,
        keep_segments=keep_segments,
        stream_spikes={},
        video_offsets=None,
        ass_paths=None,
        asr_words=None,
        stream_markers=stream_markers,
        fps=25.0
    )

    tree = ET.parse(output_path)
    root = tree.getroot()

    # V1 should have video.mkv
    # With dynamic tracks, V1 is the first video track created
    # We find playlist IDs by looking for "A" or "V" properties or just looking at chain resources
    playlists = [pl for pl in root.findall(".//playlist") if pl.get("id") != "main_bin"]

    def find_pl_by_resource(res_name):
        for pl in playlists:
            for ent in pl.findall("entry"):
                prod = ent.get("producer")
                chain = root.find(f".//chain[@id='{prod}']")
                if chain is not None and res_name in chain.find("property[@name='resource']").text:
                    # In dynamic tracks, entries point to chains.
                    return pl
        return None

    def find_chain_by_resource(res_name):
        for chain in root.findall(".//chain"):
            res = chain.find("property[@name='resource']")
            if res is not None and res_name in res.text:
                return chain
        return None

    v1_chain = find_chain_by_resource("video.mkv")
    assert v1_chain is not None
    v1_chain_id = v1_chain.get("id")

    # Find entry pointing to this chain
    v1_entry = None
    for ent in root.findall(".//playlist/entry"):
        if ent.get("producer") == v1_chain_id:
            v1_entry = ent
            break
    assert v1_entry is not None

    # Check all audio tracks
    a1_pl = find_pl_by_resource("audio1.mp3")
    a2_pl = find_pl_by_resource("audio2.mp3")

    a1_entry = a1_pl.find("entry")
    a2_entry = a2_pl.find("entry")

    a1_chain = root.find(f".//chain[@id='{a1_entry.get('producer')}']")
    # a1_entry might be a2_entry if resources were found in different order.
    # Recalculate a1_chain and a2_chain based on entries
    def get_chain_for_entry(entry):
        return root.find(f".//chain[@id='{entry.get('producer')}']")

    # Identify chain with audio1.mp3
    target_a1_chain = None
    if "audio1.mp3" in get_chain_for_entry(a1_entry).find("property[@name='resource']").text:
        target_a1_chain = get_chain_for_entry(a1_entry)
    else:
        target_a1_chain = get_chain_for_entry(a2_entry)

    resources = [
        get_chain_for_entry(a1_entry).find("property[@name='resource']").text,
        get_chain_for_entry(a2_entry).find("property[@name='resource']").text
    ]

    # Both audio1 and audio2 (ORIGINAL) should be present on audio tracks
    all_chains = root.findall(".//chain")
    all_resources = [c.find("property[@name='resource']").text for c in all_chains]

    assert any("audio1.mp3" in r for r in all_resources)
    assert any("audio2.mp3" in r for r in all_resources)

    # Identify which chain is audio2 to check filters
    target_chain = None
    for chain in root.findall(".//chain"):
        res = chain.find("property[@name='resource']")
        if res is not None and "audio2.mp3" in res.text:
            target_chain = chain
            break

    # Check volume filter on the chain that has audio2
    # Check for bin effects (filters on chains)
    def find_filter(chain, service):
        if chain is None: return None
        for f in chain.findall("filter"):
            for p in f.findall("property"):
                if p.get("name") == "mlt_service" and p.text == service:
                    return f
        return None

    vol_filter = find_filter(target_chain, "volume")
    assert vol_filter is not None, "Volume filter not found on audio2 chain"
    gain_prop = vol_filter.find("property[@name='gain']")
    assert gain_prop is not None
    # Silence was at 1.0-2.0s. 
    assert "00:00:01:00=0" in gain_prop.text
    assert "00:00:02:00=0" in gain_prop.text



    assert find_filter(target_a1_chain, "ladspa.1073") is not None
    assert find_filter(target_a1_chain, "dynamic_loudness") is not None


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
        audio_files=None,
        output_path=output_path,
        keep_segments=keep_segments,
        stream_spikes={},
        video_offsets=video_offsets,
        fps=25.0
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
