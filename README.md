# Video Processing Tool

A modular Python tool for synchronized multi-video editing based on audio stream analysis. Optimized for Hungarian language processing and NVIDIA 3060 GPUs.

## Features

- **Dual-Pass ASR Pipeline**: 
    - **Pass 1 (Filler Detection)**: Uses `nyrahealth/CrisperWhisper` via `faster-whisper` for high-speed precision detection of speech boundaries and fillers.
    - **Pass 2 (High-Fidelity Transcription)**: Uses `Qwen3-ASR-1.7B` with `Qwen3-ForcedAligner` for state-of-the-art Hungarian transcription and word-level timestamp accuracy.
- **VRAM Optimizations**: Orchestrated to stay within 8GB VRAM (RTX 3060). Pass 1 uses `int8_float16` quantization; Pass 2 uses `bfloat16`.
- **Precision Timestamps**: Re-maps transcript timestamps from unified audio back to the original timeline with millisecond precision.
- **High-Quality Signal Processing**: Automatically applies compression and targets -14 LUFS for consistent levels across all streams.
- **Fast Execution via Caching**: Metadata (ASR, repetitions, silence) is cached in JSON files for instant re-runs.
- **Silence-Aware Export**: Cuts long silences (>=2s) and renders each source into a single gap-free file using hardware encoding (NVENC/QSV). A simple Kdenlive project is generated referencing the rendered files.
- **Filler Detection**: Automatically identifies and cuts Hungarian/English filler words like `[UH]`, `[UM]`, `er`, and `ő`.
- **Overlap & Repetition Detection**: Highlights active discussions and marks duplicate acoustic patterns for easy editing.
- **ASR Subtitles**: Generates synchronized `.ass` and `.srt` files with timestamps adjusted for the final cut.

## Modular Structure

- `video_processor.py`: Main entry point, CLI logic, and batch orchestration.
- `audio_utils.py`: Audio extraction, signal processing, silence/spike/repetition detection using `librosa`.
- `filler_processor.py`: Speech boundary detection and filler word identification using CrisperWhisper.
- `transcription_processor.py`: Main ASR engine for high-fidelity transcription using Qwen3-ASR.
- `interval_utils.py`: Mathematical logic for merging, inverting, and adjusting time intervals.
- `exporter.py`: FFmpeg processing (hardware-accelerated rendering) and file generation (Kdenlive XML, ASS).

## Prerequisites

- [Python 3.10+](https://www.python.org/)
- [uv](https://github.com/astral-sh/uv) (for environment and package management)
- [FFmpeg](https://ffmpeg.org/) and `ffprobe` (must be in system PATH)
- NVIDIA GPU with CUDA support (e.g., RTX 3060)

## Dataset Setup (Optional but Recommended)

For advanced filler word detection, this tool uses a custom CNN model. To train or re-train this model, you need the **PodcastFillers** dataset:

1. **Download**: Visit [Zenodo (PodcastFillers)](https://zenodo.org/records/7121457).
2. **Files Needed**:
   - `PodcastFillers.csv`
   - `clips.tar.gz` (or the unpacked `clip_wav` folder)
3. **Structure**: Place these files into a folder named `PodcastFillerDataset` in the project root:
   ```text
   video-processing-tool/
   ├── PodcastFillerDataset/
   │   ├── PodcastFillers.csv
   │   └── clip_wav/ (folder containing .wav files)
   ```
4. **Trigger Training**: Run `setup.bat`. It will automatically detect the dataset and train the model if `filler_detector.pth` is missing.

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
- `--no-render`: Skip hardware-accelerated rendering. Kdenlive project references original source files (lossless and fast).
- `--video-offsets`: Comma-separated or single value in seconds (e.g., `1.2` or `0.5,-0.2,0`) to align tracks.
- `--silence-threshold`: Set dB level for silence (default: `-30.0`).
- `--filler-words`: Comma-separated list of Hungarian filler words to cut (default: `er,ő`).
- `--no-asr`: Skip ASR transcription pass (for fast testing).

### Automatic File Discovery
When a directory is provided, the tool automatically finds:
- **Video**: `.mkv`, `.mp4`, `.avi`
- **Audio**: `.mp3`, `.wav`, `.m4a`

### Directory Structure

```
grandparent_dir/                          # e.g. videos/
├── {grandparent_name}.kdenlive           # Kdenlive project referencing PROCESSED files
├── PROCESSED/                            # Final rendered files (gap-free, per source)
│   ├── {track}_processed.mp4
│   ├── {track}_processed.flac
│   └── {track}_processed.flac
└── input_source_dir/                     # e.g. recordings/
    ├── source1.mkv
    ├── source2.mkv
    └── video_processing_work/            # Cache + pre-processed audio
        ├── {track}_a0.flac               # Normalized FLAC (-14 LUFS, compressed)
        ├── *.asr.json
        ├── *.markers.json
        ├── *.vad.json
        ├── *.reps.json
        ├── *.filler_*.json
        ├── *.ass / *.srt                 # Subtitles with cut-adjusted timestamps
        └── ...
```

- **Working Directory**: `video_processing_work/` in the source's parent dir for cache JSONs and pre-processed FLACs.
- **Rendered Output**: `PROCESSED/` at the grandparent level contains one gap-free file per source, re-encoded with the best available hardware encoder (NVENC > QSV > libx264). Presets favor speed: `p2` (NVENC), `veryfast` (QSV), `superfast` (libx264).
- **Kdenlive Project**: `{grandparent_name}.kdenlive` at the grandparent level, referencing files in `PROCESSED/`.

## Kdenlive / MLT Reference

See [AGENTS.md](AGENTS.md) for MLT XML formatting rules used by `exporter.py`.
