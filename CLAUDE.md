# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚀 Overview
This tool is a modular Python pipeline designed for synchronized multi-video editing, specifically tailored for audio stream analysis. It automates the complex process of extracting speech, detecting speech events (silence, spikes, fillers), generating transcripts, and preparing synchronized assets for a professional NLE like Kdenlive. The system leverages external APIs (Cohere) and advanced time-stamping utilities for maximum accuracy.

## 🛠️ 1. Build & Setup Commands

### Prerequisites (Must be done before setup)
*   **Python:** Python 3.10+ is required.
*   **System Tools:** **FFmpeg** and `ffprobe` must be installed and available in the system's PATH.
*   **Hardware:** An NVIDIA GPU with CUDA support is highly recommended for optimal performance.

### Setup Procedure

**A. Windows (Recommended)**
Run the provided `setup.bat` script. This script automates the entire process:
1.  Creates a virtual environment (`.venv`).
2.  Installs base dependencies listed in `requirements.txt`.
3.  Installs PyTorch with specific CUDA support (e.g., `cu126`), ensuring GPU utilization.

**B. Linux/macOS (Using `uv`)**
1.  **Create/Activate Environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```
2.  **Install Dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```
3.  **Install PyTorch (CUDA):** *(The exact CUDA version must match the system environment.)*
    ```bash
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ```
*Note: API Keys (e.g., COHERE_API_KEY) must be set as environment variables before running any processing.*

## ▶️ 2. How to Run the Tool

### Running the Core Tool
The primary execution command is handled by `process.bat` (on Windows).

**Example Execution:**
```batch
process.bat "C:\Videos\Recording\Source" --video-offsets 0.5
```
*   **Function:** This calls `video_processor.py` within the activated virtual environment to run the entire pipeline.
*   **Output:** Cache files (`.asr.json`, `.markers.json`) and pre-processed FLACs in `video_processing_work/` under the source's parent. Rendered gap-free files in `PROCESSED/` at the grandparent level. Kdenlive project at `grandparent/{grandparent_name}.kdenlive`.

### Testing
*   **Recommendation:** Testing should involve running `process.bat` with sample data from the `test_data/` directory and manually validating the output artifacts against expected results.

## 🏗️ 3. Architectural Components and Interactions (The Big Picture)

The system operates as a linear, state-passing pipeline, orchestrated by `video_processor.py`.

### Core Components

| File/Module | Primary Responsibility | Key Dependencies / Changes |
| :--- | :--- | :--- |
| **`video_processor.py`** | **Orchestrator:** Manages the entire workflow: audio extraction $\rightarrow$ ASR $\rightarrow$ VAD/Overlap Detection $\rightarrow$ Interval Merging $\rightarrow$ Exporter execution. | Depends on all other modules. |
| **`transcription_processor.py`** | **Main ASR:** High-fidelity transcription engine using Qwen3-ASR and Forced Aligner for precise word timestamps. | `qwen-asr`, `torch`. |
| **`filler_processor.py`** | **Speech Analysis Core:** Handles VAD and initial filler detection using CrisperWhisper. | `librosa`, `faster-whisper`, `nemo`. |
| **`audio_utils.py`** | **Signal Processing:** Handles audio manipulation. Tasks include: audio extraction, normalization, silence/spike detection, and finding acoustic repetitions. | `librosa`, Signal Processing. |
| **`interval_utils.py`** | **Time Logic:** The mathematical core. Responsible for merging, inverting, and adjusting all detected time boundaries (silences, fillers, overlaps) to create the final "keep" segments. | Interval Algebra. |
| **`exporter.py`** | **Output Generation:** Renders gap-free per-source video/audio files with hardware encoding (NVENC/QSV). Creates the **`.kdenlive` project file** referencing rendered files, and generates time-synced **`.ass` / `.srt`** subtitle files. | FFmpeg, Kdenlive API/Format. |

### Interaction Flow (Pipeline Stages)

1.  **Input $\rightarrow$ Audio Extraction (`audio_utils.py`):** Raw video/audio is separated into normalized FLAC files (two-pass loudnorm, -14 LUFS) in `video_processing_work/`.
2.  **Analysis Loop (Iterative):** For every audio file:
    *   **Silence/Spike Detection (`audio_utils.py`):** Generates initial markers.
    *   **ASR & Transcription (`filler_processor.py` & `transcription_processor.py`):** Transcribes audio using the dual-pass engine (CrisperWhisper $\rightarrow$ Qwen3-ASR).
    *   **Filler Detection (`filler_processor.py`):** Identifies non-speech elements like `[UH]` or `er`.
3.  **Global Cleanup & Structuring:**
    *   Global silence intervals are calculated and merged with filler intervals.
    *   The system calculates the final, synchronized segments that should be *kept*.
    *   Overlaps and repetitions are analyzed to mark areas of redundant discussion.
4.  **Export & Finalization:**
    *   The core loop calls `exporter.py` for each video input, generating a **Kdenlive project** that skips silent/redundant areas.
    *   Finally, `exporter.py` uses the refined timestamps to generate synchronized `.ass` subtitle files.