import pytest
import os
from audio_utils import detect_silence_and_spikes

def test_silence_detection(synthetic_audio):
    # synthetic_audio has silence from 5s to 10s, split by a spike at 7.5s
    markers = detect_silence_and_spikes(synthetic_audio, threshold_db=-30, min_silence_len=2.0, max_spike_len=0.3)
    silences, spikes = markers
    
    assert len(silences) >= 2
    # Check if any detected silence covers the expected ranges
    # Tolerance 0.6s accounts for asymmetric padding (0.3s attack, 0.5s release)
    assert any(abs(s - 5.0) < 0.6 for s, e in silences)
    assert any(abs(e - 10.0) < 0.6 for s, e in silences)

def test_spike_detection(synthetic_audio):
    # synthetic_audio has a spike at 7.5s (inside silence block)
    # The spike is ~0.1s long but might be detected slightly longer due to windows
    markers = detect_silence_and_spikes(synthetic_audio, threshold_db=-30, min_silence_len=2.0, max_spike_len=0.3)
    _, spikes = markers
    
    assert len(spikes) >= 1
    found = False
    for s, e in spikes:
        if abs(s - 7.5) < 0.5:
            found = True
            break
    assert found, f"Expected spike around 7.5s, got {spikes}"
