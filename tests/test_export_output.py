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

from exporter import generate_kdenlive_project
from kdenlive.kdenlive_lib import KdenliveProject
from audio_utils import get_video_fps


def create_synthetic_wav(path, duration=60.0, sr=16000):
    """Create a synthetic WAV file with speech and silence."""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    y = np.zeros_like(t)
    
    # Add some "speech" (sine waves) in segments
    y[0:int(10*sr)] = 0.5 * np.sin(2 * np.pi * 440 * t[0:int(10*sr)])
    y[int(15*sr):int(25*sr)] = 0.5 * np.sin(2 * np.pi * 880 * t[int(15*sr):int(25*sr)])
    y[int(30*sr):int(60*sr)] = 0.5 * np.sin(2 * np.pi * 660 * t[int(30*sr):int(60*sr)])
    
    sf.write(path, y, sr)
    return path


def create_synthetic_mkv_mock(path):
    """Create a file that ffmpeg/ffprobe will recognize as having video."""
    with open(path, 'w') as f:
        f.write("mock video file")
    return path


def tc_to_frames(tc_str, fps=25.0):
    """Convert HH:MM:SS:FF to frames."""
    parts = tc_str.split(':')
    h, m, s, f = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    return h * 3600 * int(fps) + m * 60 * int(fps) + s * int(fps) + f


def test_audio_clips_on_both_tracks():
    """Test that audio files are placed on both A1 and A2 tracks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic audio files
        audio1 = os.path.join(tmpdir, "audio1.wav")
        audio2 = os.path.join(tmpdir, "audio2.wav")
        create_synthetic_wav(audio1, duration=60.0)
        create_synthetic_wav(audio2, duration=60.0)
        
        # Create mock video files
        video1 = os.path.join(tmpdir, "video1.mkv")
        video2 = os.path.join(tmpdir, "video2.mkv")
        create_synthetic_mkv_mock(video1)
        create_synthetic_mkv_mock(video2)
        
        # Generate kdenlive project
        output = os.path.join(tmpdir, "project.kdenlive")
        keep_segments = [(0, 60)]  # Keep entire 60s
        
        video_to_audio_map = {
            video1: [audio1],
            video2: [audio2]
        }
        
        import unittest.mock as mock
        with mock.patch('exporter.has_video_stream', return_value=True), \
             mock.patch('exporter.get_video_duration', return_value=60.0), \
             mock.patch('exporter.get_video_fps', return_value=25.0):
            
            generate_kdenlive_project(
                [video1, video2], output, keep_segments,
                stream_spikes_map={},
                overlaps=[],
                repetitions=[],
                fps=25.0,
                video_offsets=[0.0, 0.0],
                ass_paths=[],
                asr_words=[[], []],
                stream_markers_global={},
                video_to_audio_map=video_to_audio_map
            )
        
        # Load and verify (check XML directly without triggering _reset_timelines)
        assert os.path.exists(output), "Kdenlive file not created"
        tree = ET.parse(output)
        root = tree.getroot()

        # Find sequence tractor from main_bin's activetimeline property
        main_bin = root.find(".//playlist[@id='main_bin']")
        active_timeline = main_bin.find("property[@name='kdenlive:docproperties.activetimeline']")
        seq_tr_id = active_timeline.text
        seq_tr = root.find(f".//tractor[@id='{seq_tr_id}']")

        # Check A1 and A2 playlists by finding audio tracks
        # Each track has 2 playlists (pl1 with entries, pl2 without)
        # Find the correct pl1 playlists by looking at the tractors
        a1_pl = None
        a2_pl = None

        # Find audio tractors (those with kdenlive:audio_track=1)
        audio_tractors = []
        for tr in root.findall("tractor"):
            props = tr.findall("property[@name='kdenlive:audio_track']")
            if any(p.text == "1" for p in props):
                audio_tractors.append(tr)

        # Sort by track order in sequence tractor
        seq_tracks = []
        if seq_tr is not None:
            for t in seq_tr.findall("track"):
                seq_tracks.append(t.get("producer"))

        # Get pl1 for each audio tractor (first track element)
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
        # Keep segments that total 30 seconds
        keep_segments = [(0, 10), (20, 40)]  # Total 30s
        
        video_to_audio_map = {video1: [audio1]}
        
        import unittest.mock as mock
        with mock.patch('exporter.has_video_stream', return_value=True), \
             mock.patch('exporter.get_video_duration', return_value=60.0), \
             mock.patch('exporter.get_video_fps', return_value=25.0):
            
            generate_kdenlive_project(
                [video1], output, keep_segments,
                stream_spikes_map={},
                overlaps=[],
                repetitions=[],
                fps=25.0,
                video_offsets=[0.0],
                ass_paths=[],
                asr_words=[[]],
                stream_markers_global={},
                video_to_audio_map=video_to_audio_map
            )
        
        # Check XML directly without reloading (which would trigger _reset_timelines)
        tree = ET.parse(output)
        root = tree.getroot()
        
        # Check V1 track by finding video track (no kdenlive:audio_track=1)
        v1_pl = None
        for pl in root.findall("playlist"):
            props = pl.findall("property[@name='kdenlive:audio_track']")
            if not any(p.text == "1" for p in props):
                # Check if this playlist is used in a tractor with video track
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
        
        # Calculate total duration of video clips
        total_frames = 0
        for entry in entries:
            in_tc = entry.get('in')
            out_tc = entry.get('out')
            in_f = tc_to_frames(in_tc, 25.0)  # proj.fps = 25
            out_f = tc_to_frames(out_tc, 25.0)
            duration = out_f - in_f + 1
            total_frames += duration
        
        expected_frames = int(30 * 25.0)  # 30 seconds at 25fps = 750 frames
        print(f"✓ Video total frames: {total_frames}, expected: {expected_frames}")
        assert abs(total_frames - expected_frames) <= 2, \
            f"Video clip length mismatch: got {total_frames}, expected {expected_frames}"


def test_silence_gaps_in_audio():
    """Test that silence gaps are correctly muted via volume keyframes in audio tracks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio1 = os.path.join(tmpdir, "audio1.wav")
        create_synthetic_wav(audio1, duration=60.0)

        video1 = os.path.join(tmpdir, "video1.mkv")
        create_synthetic_mkv_mock(video1)

        output = os.path.join(tmpdir, "project.kdenlive")
        keep_segments = [(0, 60)]  # Keep entire 60s

        # Silences at 10-15s and 25-30s
        stream_markers_global = {
            audio1: {
                "silence": [(10.0, 15.0), (25.0, 30.0)],
                "spikes": []
            }
        }

        video_to_audio_map = {video1: [audio1]}

        import unittest.mock as mock
        with mock.patch('exporter.has_video_stream', return_value=True), \
             mock.patch('exporter.get_video_duration', return_value=60.0), \
             mock.patch('exporter.get_video_fps', return_value=25.0):

            generate_kdenlive_project(
                [video1], output, keep_segments,
                stream_spikes_map={},
                overlaps=[],
                repetitions=[],
                fps=25.0,
                video_offsets=[0.0],
                ass_paths=[],
                asr_words=[[]],
                stream_markers_global=stream_markers_global,
                video_to_audio_map=video_to_audio_map
            )

        # Check XML directly without reloading (which would trigger _reset_timelines)
        tree = ET.parse(output)
        root = tree.getroot()

        # Check A1 track - find audio playlist dynamically
        a1_pl = None
        for pl in root.findall("playlist"):
            props = pl.findall("property[@name='kdenlive:audio_track']")
            if any(p.text == "1" for p in props):
                a1_pl = pl
                break

        entries = a1_pl.findall('entry') if a1_pl is not None else []
        assert len(entries) > 0, "A1 track has no entries"

        # Check that volume filter with mute keyframes exists
        # The silence intervals should be muted (volume=0)
        volume_found = False
        for entry in entries:
            for filt in entry.findall('filter'):
                service = filt.find("property[@name='mlt_service']")
                if service is not None and service.text == 'volume':
                    gain_prop = filt.find("property[@name='gain']")
                    if gain_prop is not None:
                        gain_text = gain_prop.text
                        # Should have mute keyframes (volume=0) at silence intervals
                        # 10-15s = frames 250-374, 25-30s = frames 625-749
                        assert '250=0' in gain_text or '249=0' in gain_text, \
                            f"Expected mute at 10s (frame ~250), got: {gain_text}"
                        assert '625=0' in gain_text or '624=0' in gain_text, \
                            f"Expected mute at 25s (frame ~625), got: {gain_text}"
                        volume_found = True

        assert volume_found, "No volume filter with mute keyframes found for silences"
        print(f"✓ A1 has {len(entries)} entries with volume muting for silences")


if __name__ == "__main__":
    test_audio_clips_on_both_tracks()
    print()
    test_video_clip_length()
    print()
    test_silence_gaps_in_audio()
