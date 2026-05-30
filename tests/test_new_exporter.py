import pytest
import os
import xml.etree.ElementTree as ET
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, '.')
sys.path.insert(0, r'c:\Users\gerge\source\repos\mlt-python\src')
from exporter import generate_kdenlive_project


def tc_to_seconds(tc_str):
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


@patch("exporter.get_video_duration", return_value=10.0)
@patch("exporter.has_video_stream")
@patch("os.path.exists", return_value=True)
def test_multiple_audio_tracks_assignment(mock_exists, mock_has_vid, mock_dur, working_dir):
    output_path = os.path.join(working_dir, "test_multi_audio.kdenlive")
    video_files = ["video.mkv", "audio1.mp3", "audio2.mp3"]

    mock_has_vid.side_effect = lambda x: x == "video.mkv"

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
    )

    tree = ET.parse(output_path)
    root = tree.getroot()

    playlists = [pl for pl in root.findall(".//playlist") if pl.get("id") != "main_bin"]

    def find_pl_by_resource(res_name):
        for pl in playlists:
            for ent in pl.findall("entry"):
                prod = ent.get("producer")
                chain = root.find(f".//chain[@id='{prod}']")
                if chain is not None and res_name in chain.find("property[@name='resource']").text:
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

    v1_entry = None
    for ent in root.findall(".//playlist/entry"):
        if ent.get("producer") == v1_chain_id:
            v1_entry = ent
            break
    assert v1_entry is not None

    a1_pl = find_pl_by_resource("audio1.mp3")
    a2_pl = find_pl_by_resource("audio2.mp3")

    a1_entry = a1_pl.find("entry")
    a2_entry = a2_pl.find("entry")

    a1_chain = root.find(f".//chain[@id='{a1_entry.get('producer')}']")

    def get_chain_for_entry(entry):
        return root.find(f".//chain[@id='{entry.get('producer')}']")

    target_a1_chain = None
    if "audio1.mp3" in get_chain_for_entry(a1_entry).find("property[@name='resource']").text:
        target_a1_chain = get_chain_for_entry(a1_entry)
    else:
        target_a1_chain = get_chain_for_entry(a2_entry)

    resources = [
        get_chain_for_entry(a1_entry).find("property[@name='resource']").text,
        get_chain_for_entry(a2_entry).find("property[@name='resource']").text
    ]

    all_chains = root.findall(".//chain")
    all_resources = [c.find("property[@name='resource']").text for c in all_chains]

    assert any("audio1.mp3" in r for r in all_resources)
    assert any("audio2.mp3" in r for r in all_resources)

    target_chain = None
    for chain in root.findall(".//chain"):
        res = chain.find("property[@name='resource']")
        if res is not None and "audio2.mp3" in res.text:
            target_chain = chain
            break

    def find_filter(chain, service):
        if chain is None: return None
        for f in chain.findall("filter"):
            for p in f.findall("property"):
                if p.get("name") == "mlt_service" and p.text == service:
                    return f
        return None

    for chain in [target_chain, target_a1_chain]:
        for filt in chain.findall("filter"):
            for p in filt.findall("property"):
                if p.get("name") == "mlt_service" and p.text in ("volume", "ladspa.1073", "dynamic_loudness"):
                    raise AssertionError(f"Unexpected {p.text} filter on chain (pre-processed FLAC replaces filters)")


@patch("exporter.get_video_duration", return_value=10.0)
@patch("exporter.has_video_stream")
@patch("os.path.exists", return_value=True)
def test_video_offsets(mock_exists, mock_has_vid, mock_dur, working_dir):
    output_path = os.path.join(working_dir, "test_offsets.kdenlive")
    video_files = ["video1.mkv", "video2.mkv"]
    mock_has_vid.return_value = True

    video_offsets = [0.5, -0.2]
    keep_segments = [(1.0, 3.0)]

    generate_kdenlive_project(
        video_files=video_files,
        audio_files=None,
        output_path=output_path,
        keep_segments=keep_segments,
        stream_spikes={},
        video_offsets=video_offsets,
    )

    tree = ET.parse(output_path)
    root = tree.getroot()

    entries = root.findall(".//playlist/entry")

    # Video 1: in=1.5s, out=1.5+2.0-1/25=3.460s -> 00:00:03.460
    # Video 2: in=0.8s, out=0.8+2.0-1/25=2.760s -> 00:00:02.760

    found_in_out = []
    for ent in entries:
        prod = ent.get("producer")
        chain = root.find(f".//chain[@id='{prod}']")
        if chain is not None:
            res = chain.find("property[@name='resource']").text
            found_in_out.append((res, ent.get("in"), ent.get("out")))

    assert any("video1.mkv" in r and i == "00:00:01.500" and o == "00:00:03.460" for r, i, o in found_in_out)
    assert any("video2.mkv" in r and i == "00:00:00.800" and o == "00:00:02.760" for r, i, o in found_in_out)
