"""
Test that verifies the generated kdenlive project has correct structure:
- A1 and A2 both have audio clips
- Video clips are properly placed
- Clip lengths match expected values
- Silence gaps are correct
"""
import pytest
import sys
import os
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import soundfile as sf

sys.path.insert(0, '.')
sys.path.insert(0, r'c:\Users\gerge\source\repos\mlt-python\src')

from exporter import generate_kdenlive_project


def create_synthetic_wav(path, duration=60.0, sr=16000):
    """Create a synthetic WAV file with speech and silence."""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    y = np.zeros_like(t)
    
    y[0:int(10*sr)] = 0.5 * np.sin(2 * np.pi * 440 * t[0:int(10*sr)])
    y[int(15*sr):int(25*sr)] = 0.5 * np.sin(2 * np.pi * 880 * t[int(15*sr):int(25*sr)])
    y[int(30*sr):int(60*sr)] = 0.5 * np.sin(2 * np.pi * 660 * t[int(30*sr):int(60*sr)])
    
    sf.write(path, y, sr)
    return path


def create_synthetic_mkv_mock(path):
    with open(path, 'w') as f:
        f.write("mock video file")
    return path


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


def test_audio_clips_on_both_tracks():
    """Test that audio files are placed on both A1 and A2 tracks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio1 = os.path.join(tmpdir, "audio1.wav")
        audio2 = os.path.join(tmpdir, "audio2.wav")
        create_synthetic_wav(audio1, duration=60.0)
        create_synthetic_wav(audio2, duration=60.0)
        
        video1 = os.path.join(tmpdir, "video1.mkv")
        video2 = os.path.join(tmpdir, "video2.mkv")
        create_synthetic_mkv_mock(video1)
        create_synthetic_mkv_mock(video2)
        
        output = os.path.join(tmpdir, "project.kdenlive")
        keep_segments = [(0, 60)]
        
        video_to_audio_map = {
            video1: [audio1],
            video2: [audio2]
        }
        
        import unittest.mock as mock
        with mock.patch('exporter.has_video_stream', return_value=True), \
             mock.patch('exporter.get_video_duration', return_value=60.0):
            
            audio_files = {
                video1: [{"original": audio1, "stream_idx": 0, "temp_path": audio1}],
                video2: [{"original": audio2, "stream_idx": 0, "temp_path": audio2}]
            }
            generate_kdenlive_project(
                video_files=[video1, video2],
                audio_files=audio_files,
                output_path=output,
                keep_segments=keep_segments,
                stream_spikes={},
                video_offsets=[0.0, 0.0],
                ass_paths=[],
                asr_words=[[], []],
                stream_markers={},
            )
        
        assert os.path.exists(output), "Kdenlive file not created"
        tree = ET.parse(output)
        root = tree.getroot()

        main_bin = root.find(".//playlist[@id='main_bin']")
        active_timeline = main_bin.find("property[@name='kdenlive:docproperties.activetimeline']")
        seq_tr_id = active_timeline.text
        seq_tr = root.find(f".//tractor[@id='{seq_tr_id}']")

        audio_tractors = []
        for tr in root.findall("tractor"):
            props = tr.findall("property[@name='kdenlive:audio_track']")
            if any(p.text == "1" for p in props):
                audio_tractors.append(tr)

        a1_pl = None
        a2_pl = None

        for tr in audio_tractors:
            tracks = tr.findall("track")
            if tracks:
                pl1_id = tracks[0].get("producer")
                pl1 = root.find(f".//playlist[@id='{pl1_id}']")
                if a1_pl is None:
                    a1_pl = pl1
                elif a2_pl is None:
                    a2_pl = pl1

        a1_entries = a1_pl.findall('entry') if a1_pl is not None else []
        a2_entries = a2_pl.findall('entry') if a2_pl is not None else []

        assert len(a1_entries) > 0, f"A1 track has no entries"
        assert len(a2_entries) > 0, f"A2 track has no entries"

        print(f"✓ A1 has {len(a1_entries)} entries")
        print(f"✓ A2 has {len(a2_entries)} entries")


def test_video_clip_length():
    """Test that video clips have correct length in timeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio1 = os.path.join(tmpdir, "audio1.wav")
        create_synthetic_wav(audio1, duration=60.0)
        
        video1 = os.path.join(tmpdir, "video1.mkv")
        create_synthetic_mkv_mock(video1)
        
        output = os.path.join(tmpdir, "project.kdenlive")
        keep_segments = [(0, 10), (20, 40)]
        
        video_to_audio_map = {video1: [audio1]}
        
        import unittest.mock as mock
        with mock.patch('exporter.has_video_stream', return_value=True), \
             mock.patch('exporter.get_video_duration', return_value=60.0):
            
            audio_files = {
                video1: [{"original": audio1, "stream_idx": 0, "temp_path": audio1}]
            }
            generate_kdenlive_project(
                video_files=[video1],
                audio_files=audio_files,
                output_path=output,
                keep_segments=keep_segments,
                stream_spikes={},
                video_offsets=[0.0],
                ass_paths=[],
                asr_words=[[]],
                stream_markers={},
            )
        
        tree = ET.parse(output)
        root = tree.getroot()
        
        v1_pl = None
        for pl in root.findall("playlist"):
            props = pl.findall("property[@name='kdenlive:audio_track']")
            if not any(p.text == "1" for p in props):
                pl_id = pl.get("id")
                for tr in root.findall("tractor"):
                    track_elem = tr.find(f"track[@producer='{pl_id}']")
                    if track_elem is not None and track_elem.get("hide") != "video":
                        v1_pl = pl
                        break
                if v1_pl is not None:
                    break

        entries = v1_pl.findall('entry') if v1_pl is not None else []
        assert len(entries) > 0, "V1 track has no entries"
        
        total_seconds = 0.0
        for entry in entries:
            in_tc = entry.get('in')
            out_tc = entry.get('out')
            in_s = tc_to_seconds(in_tc)
            out_s = tc_to_seconds(out_tc)
            total_seconds += out_s - in_s
        
        expected_seconds = 30.0
        print(f"✓ Video total seconds: {total_seconds}, expected: {expected_seconds}")
        assert abs(total_seconds - expected_seconds) <= 0.1, \
            f"Video clip length mismatch: got {total_seconds}, expected {expected_seconds}"


def test_silence_gaps_in_audio():
    """Test that silence gaps are cut out via interval-based audio entries (not volume filters)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio1 = os.path.join(tmpdir, "audio1.wav")
        create_synthetic_wav(audio1, duration=60.0)

        video1 = os.path.join(tmpdir, "video1.mkv")
        create_synthetic_mkv_mock(video1)

        output = os.path.join(tmpdir, "project.kdenlive")
        keep_segments = [(0, 60)]

        stream_markers_global = {
            audio1: {
                "silence": [(10.0, 15.0), (25.0, 30.0)],
                "spikes": []
            }
        }

        video_to_audio_map = {video1: [audio1]}

        import unittest.mock as mock
        with mock.patch('exporter.has_video_stream', return_value=True), \
             mock.patch('exporter.get_video_duration', return_value=60.0):

            audio_files = {
                video1: [{"original": audio1, "stream_idx": 0, "temp_path": audio1}]
            }
            generate_kdenlive_project(
                video_files=[video1],
                audio_files=audio_files,
                output_path=output,
                keep_segments=keep_segments,
                stream_spikes={},
                video_offsets=[0.0],
                ass_paths=[],
                asr_words=[[]],
                stream_markers=stream_markers_global,
            )

        tree = ET.parse(output)
        root = tree.getroot()

        a1_pl = None
        for pl in root.findall("playlist"):
            props = pl.findall("property[@name='kdenlive:audio_track']")
            if any(p.text == "1" for p in props):
                a1_pl = pl
                break

        all_children = list(a1_pl) if a1_pl is not None else []
        entries = a1_pl.findall('entry') if a1_pl is not None else []
        assert len(entries) > 0, "A1 track has no entries"

        for chain in root.findall('chain'):
            for filt in chain.findall('filter'):
                service = filt.find("property[@name='mlt_service']")
                if service is not None and service.text == 'volume':
                    raise AssertionError("Unexpected volume filter on chain (silence should be cut, not muted)")

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

        expected_seconds = 60.0
        assert abs(total_seconds - expected_seconds) <= 0.2, \
            f"Audio timeline mismatch: got {total_seconds}, expected {expected_seconds}"

        print(f"✓ A1 has {len(entries)} entries covering non-silent intervals ({total_seconds}s)")


if __name__ == "__main__":
    test_audio_clips_on_both_tracks()
    print()
    test_video_clip_length()
    print()
    test_silence_gaps_in_audio()
