import pytest
import numpy as np
import soundfile as sf
import os
import tempfile

@pytest.fixture
def synthetic_audio():
    """Generates a synthetic WAV with 5s speech, 5s silence, 5s speech, and a spike."""
    sr = 16000
    duration = 15
    t = np.linspace(0, duration, duration * sr, endpoint=False)
    
    # Simple sine waves for speech
    y = np.zeros_like(t)
    y[0:5*sr] = 0.5 * np.sin(2 * np.pi * 440 * t[0:5*sr])
    y[5*sr:10*sr] = 0.0 # Silence
    y[10*sr:15*sr] = 0.5 * np.sin(2 * np.pi * 440 * t[10*sr:15*sr])
    
    # Add a spike (0.1s pop) at 7.5s (inside silence block)
    spike_start = int(7.5 * sr)
    spike_end = int(7.6 * sr)
    y[spike_start:spike_end] = 0.9
    
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, y, sr)
    
    yield path
    
    if os.path.exists(path):
        os.remove(path)

@pytest.fixture
def working_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
