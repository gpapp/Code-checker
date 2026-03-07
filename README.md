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

    User->>Main: Execute with MKV inputs
    Main->>AU: extract_audio_streams(inputs)
    Note over AU: Compression + DynAudNorm + Loudnorm
    AU-->>Main: mono WAV files

    loop Each Audio File
        Main->>AU: detect_silence_and_spikes(wav)
        Note over AU: Check .markers.json cache
        AU-->>Main: silence & spike intervals
        Main->>NP: run_asr(wav)
        Note over NP: Check .asr.json cache
        NP-->>Main: transcript & words
        Main->>NP: find_fillers(words)
        NP-->>Main: filler intervals
    end

    Main->>AU: find_global_silence(all_silence)
    AU-->>Main: global cut intervals

    Main->>IU: merge_intervals(global_silence + fillers)
    IU-->>Main: final cut_segments

    Main->>AU: get_video_duration(input)
    AU-->>Main: total_duration

    Main->>IU: calculate_keep_segments(cut_segments, total_duration)
    IU-->>Main: keep_segments

    loop Each Audio File
        Main->>NP: run_vad(wav)
        Note over NP: Check .vad.json cache
        NP-->>Main: speech intervals
        Main->>AU: find_repetitions(wav)
        Note over AU: Check .reps.json cache
        AU-->>Main: repetition intervals
    end

    Main->>NP: find_overlaps(all_speech)
    NP-->>Main: overlap segments

    alt Render Processed Videos
        loop Each Video Input
            Main->>EX: process_video(input, output, keep_segments, spikes)
            Note over EX: FFmpeg complex filters
            EX-->>Main: Processed MKV
        end
    else Kdenlive-Native (no-render)
        Main->>AU: get_video_fps(input)
        AU-->>Main: fps
    end

    Main->>IU: adjust_timestamps(overlaps, keep_segments)
    IU-->>Main: adjusted overlaps
    Main->>IU: adjust_timestamps(repetitions, keep_segments)
    IU-->>Main: adjusted repetitions

    Main->>EX: generate_kdenlive_project(files, keep_segments, spikes, overlaps, repetitions, fps, is_rendered)
    Note over EX: Generates timeline with segments or single clip
    EX-->>Main: project.kdenlive

    loop Each ASR Result
        Main->>IU: adjust_timestamps(word_times, keep_segments)
        IU-->>Main: adjusted word_times
        Main->>EX: generate_ass_file(adj_words, output)
        EX-->>Main: .ass file
    end

    Main-->>User: Processing Complete
```

## Features

- **High-Quality Signal Processing**: Automatically applies compression, dynamic normalization, and targets -14 LUFS to all audio streams for consistent levels.
- **Fast Execution via Caching**: Metadata (ASR, VAD, repetitions, silence) is cached in JSON files, allowing near-instant re-runs if analysis parameters haven't changed.
- **Standalone Audio Support**: Handles `.mp3`, `.wav`, and `.m4a` files. If standalone audio is provided with video files, on-camera audio is ignored during analysis.
- **Kdenlive-Native Editing**: Optionally avoids re-encoding by generating a Kdenlive timeline with virtual segments from original files using the `--no-render` flag.
- **Synchronized Multi-Stream Cutting**: Processes multiple inputs simultaneously, maintaining perfect sync across all tracks.
- **Silence Detection**: Automatically cuts segments where all audio streams are below -30dB (default) for more than 2 seconds.
- **Spike Muting**: Identifies and mutes minor audio spikes (clicks, coughs) under 0.2s.
- **Hungarian Filler Detection**: Uses NVIDIA Parakeet TDT ASR to find and cut "er" and "ő" filler sounds.
- **Overlap Marking**: Identifies active discussions (>=5s) on multiple streams and marks them as Guide regions in Kdenlive.
- **Repetition Detection**: Identifies repeated phrases or sounds using acoustic similarity (MFCC-based).
- **ASR Subtitles**: Generates `.ass` subtitle files for ASR transcriptions with timestamps adjusted for the final edit.

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
