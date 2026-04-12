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
process.bat "C:\MyRecording\Source" --video-offsets 0.5
```
*   **Function:** This calls `video_processor.py` within the activated virtual environment to run the entire pipeline.
*   **Output:** The tool will generate cache files (e.g., `.asr.json`, `.markers.json`) in a `video_processing_work` folder and place the final `project.kdenlive` file in the parent directory of the source.

### Testing
*   **Recommendation:** Testing should involve running `process.bat` with sample data from the `test_data/` directory and manually validating the output artifacts against expected results.

## 🏗️ 3. Architectural Components and Interactions (The Big Picture)

The system operates as a linear, state-passing pipeline, orchestrated by `video_processor.py`.

### Core Components

| File/Module | Primary Responsibility | Key Dependencies / Changes |
| :--- | :--- | :--- |
| **`video_processor.py`** | **Orchestrator:** Manages the entire workflow: audio extraction $\rightarrow$ ASR $\rightarrow$ VAD/Overlap Detection $\rightarrow$ Interval Merging $\rightarrow$ Exporter execution. | Depends on all other modules. |
| **`cohere_asr.py`** | **Primary ASR:** Handles API interaction with the Cohere API. Must be checked first in the pipeline. | Cohere SDK. |
| **`nemo_processing.py`** | **Speech Analysis Core:** Now acts as a unified utility hub. It contains `process_filler_detection_asr` which is the mandatory fallback ASR engine using CrisperWhisper. | `librosa`, NeMo/WhisperX. |
| **`audio_utils.py`** | **Signal Processing:** Handles audio manipulation. Tasks include: audio extraction, normalization, silence/spike detection, and finding acoustic repetitions. | `librosa`, Signal Processing. |
| **`interval_utils.py`** | **Time Logic:** The mathematical core. Responsible for merging, inverting, and adjusting all detected time boundaries (silences, fillers, overlaps) to create the final "keep" segments. | Interval Algebra. |
| **`exporter.py`** | **Output Generation:** Creates the final, editable assets. Interfaces with industry standards: generating the **`.kdenlive` project file**, and creating time-synced **`.ass` / `.srt`** subtitle files. | FFmpeg, Kdenlive API/Format. |

### Interaction Flow (Pipeline Stages)

1.  **Input $\rightarrow$ Audio Extraction (`audio_utils.py`):** Raw video/audio is separated into normalized mono WAV files.
2.  **Analysis Loop (Iterative):** For every audio file:
    *   **Silence/Spike Detection (`audio_utils.py`):** Generates initial markers.
    *   **ASR & Transcription (`nemo_processing.py`):** Transcribes audio using the primary engine (Cohere $\rightarrow$ CrisperWhisper).
    *   **Filler Detection (`nemo_processing.py`):** Identifies non-speech elements like `[UH]` or `er`.
3.  **Global Cleanup & Structuring:**
    *   Global silence intervals are calculated and merged with filler intervals.
    *   The system calculates the final, synchronized segments that should be *kept*.
    *   Overlaps and repetitions are analyzed to mark areas of redundant discussion.
4.  **Export & Finalization:**
    *   The core loop calls `exporter.py` for each video input, generating a **Kdenlive project** that skips silent/redundant areas.
    *   Finally, `exporter.py` uses the refined timestamps to generate synchronized `.ass` subtitle files.