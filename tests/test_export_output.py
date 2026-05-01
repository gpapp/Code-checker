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
        
        # Check A1 and A2 playlists
        a1_pl = root.find(".//playlist[@id='playlist0']")
        a2_pl = root.find(".//playlist[@id='playlist2']")
        
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
        
        # Check V1 track (playlist4)
        v1_pl = root.find(".//playlist[@id='playlist4']")
        
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
    """Test that silence gaps are correctly represented as blanks in audio tracks."""
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
        
        # Check A1 track for blanks (silences)
        a1_pl = root.find(".//playlist[@id='playlist0']")
        
        blanks = a1_pl.findall('blank') if a1_pl is not None else []
        entries = a1_pl.findall('entry') if a1_pl is not None else []
        
        print(f"✓ A1 has {len(entries)} entries and {len(blanks)} blanks for silences")
        
        # We expect 2 blanks (for the 2 silences) and 3 clips (before, between, after silences)
        assert len(blanks) == 2, f"Expected 2 blanks for silences, got {len(blanks)}"
        assert len(entries) == 3, f"Expected 3 clips around silences, got {len(entries)}"
        
        # Check blank durations (should be 5s each = 125 frames at 25fps)
        for blank in blanks:
            length_tc = blank.get('length')
            length_frames = tc_to_frames(length_tc, 25.0)  # fps = 25
            expected = int(5 * 25.0)  # 5 seconds
            print(f"  Blank length: {length_frames} frames (expected {expected})")
            assert abs(length_frames - expected) <= 2, \
                f"Blank length mismatch: got {length_frames}, expected {expected}"


if __name__ == "__main__":
    test_audio_clips_on_both_tracks()
    print()
    test_video_clip_length()
    print()
    test_silence_gaps_in_audio()
