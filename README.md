# Video Processing Tool

A modular Python tool for synchronized multi-video editing based on audio stream analysis. Optimized for Hungarian language processing and NVIDIA 3060 GPUs.

## Operation Sequence

```mermaid
sequenceDiagram
    participant User
    participant Main as video_processor.py
    participant AU as audio_utils.py
    participant NP as nemo_processing.py
    participant IU as interval_utils.py
    participant EX as exporter.py

    User->>Main: Execute with MKV/Audio inputs
    Main->>AU: extract_audio_streams(inputs)
    Note over AU: Compression + Normalization (-14 LUFS)
    AU-->>Main: mono WAV files

    loop Each Audio File
        Main->>AU: detect_silence_and_spikes(wav)
        Note over AU: Significant silences list (>=2s)
        AU-->>Main: .markers.json (silence/spikes)
        
        Main->>NP: run_asr(wav, silence_intervals)
        Note over NP: Unified-Stream CrisperWhisper (int8)
        Note over NP: Concatenate speech + 0.3s stubs
        NP-->>Main: .asr.json (transcript & words)
        
        Main->>NP: find_fillers(words)
        Note over NP: Check against [UH],[UM], etc.
        NP-->>Main: filler intervals
    end

    Main->>AU: find_global_silence(all_silence)
    AU-->>Main: global cut intervals (all streams quiet)

    Main->>IU: merge_intervals(global_silence + fillers)
    IU-->>Main: final cut_segments

    Main->>IU: calculate_keep_segments(cut_segments, total_duration)
    IU-->>Main: synchronized keep_segments

    loop Each Video Input
        Main->>EX: generate_kdenlive_project(...)
        Note over EX: Silence-Aware timeline placement
        Note over EX: Apply Volume/Comp/Norm filters
        EX-->>Main: project.kdenlive
    end

    Main-->>User: Done (Ready for Kdenlive)
```

## Features

- **Unified-Stream Transcription**: Uses `nyrahealth/CrisperWhisper` with a custom concatenation strategy. Speech segments are unified with 0.3s silence stubs to maximize model context and prevent recalibration drift.
- **VRAM Optimizations**: Uses `int8_float16` quantization via `faster-whisper`, reducing VRAM footprint to ~2GB (perfect for RTX 3060) while maintaining verbatim accuracy.
- **Precision Timestamps**: Re-maps transcript timestamps from unified audio back to the original timeline with millisecond precision.
- **High-Quality Signal Processing**: Automatically applies compression and targets -14 LUFS for consistent levels across all streams.
- **Fast Execution via Caching**: Metadata (ASR, repetitions, silence) is cached in JSON files for instant re-runs.
- **Silence-Aware Export**: Kdenlive project generator matches ASR logic, skipping long silences (>=2s) to create a clean, synchronized multi-track timeline.
- **Automatic Audio Filters**: Every audio clip on the Kdenlive timeline automatically receives high-quality Compressor, Limiter, and Loudness Normalization filters.
- **Filler Detection**: Automatically identifies and cuts Hungarian/English filler words like `[UH]`, `[UM]`, `er`, and `ő`.
- **Overlap & Repetition Detection**: Highlights active discussions and marks duplicate acoustic patterns for easy editing.
- **ASR Subtitles**: Generates synchronized `.ass` and `.srt` files with timestamps adjusted for the final cut.

## Modular Structure

- `video_processor.py`: Main entry point, CLI logic, and batch orchestration.
- `audio_utils.py`: Audio extraction, signal processing, silence/spike/repetition detection using `librosa`.
- `nemo_processing.py`: NVIDIA NeMo integration for VAD and ASR.
- `interval_utils.py`: Mathematical logic for merging, inverting, and adjusting time intervals.
- `exporter.py`: FFmpeg processing and file generation (Kdenlive XML, ASS).

## Prerequisites

- [Python 3.10+](https://www.python.org/)
- [uv](https://github.com/astral-sh/uv) (for environment and package management)
- [FFmpeg](https://ffmpeg.org/) and `ffprobe` (must be in system PATH)
- NVIDIA GPU with CUDA support (e.g., RTX 3060)

### Installation

#### Windows (Recommended)
Simply run `setup.bat`. This will create a virtual environment, install PyTorch with CUDA support, and all dependencies using `uv`.

#### Linux/macOS
Using `uv`:
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Usage

#### Basic Command
Use `process.bat` with a file or a directory:
```batch
process.bat "C:\MyRecording\Source" --video-offsets 0.5
```

#### Advanced Options
- `--no-render`: Generates the Kdenlive project using original files (lossless and fast).
- `--video-offsets`: Comma-separated or single value in seconds (e.g., `1.2` or `0.5,-0.2,0`) to align tracks.
- `--silence-threshold`: Set dB level for silence (default: `-30.0`).
- `--filler-words`: Comma-separated list of Hungarian filler words to cut (default: `er,ő`).

### Automatic File Discovery
When a directory is provided, the tool automatically finds:
- **Video**: `.mkv`, `.mp4`, `.avi`
- **Audio**: `.mp3`, `.wav`, `.m4a`

### Directory Structure
- **Working Directory**: The tool creates a `video_processing_work` folder in the **parent directory** of your input source to store cache files (`.asr.json`, etc.) and extracted audio.
- **Output Position**: Rendered videos and the `project.kdenlive` file are placed in the **parent directory** of the input source.

## Project Output

The `video_processing_work` directory will contain:
- Extracted and normalized mono WAV files.
- Transcribed `.ass` files.
- A `project.kdenlive` file ready for further editing.
- Cache files: `*.markers.json`, `*.asr.json`, `*.vad.json`, `*.reps.json`.
