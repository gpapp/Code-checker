# Agent Instructions for Video Processing Tool

## Essential Setup & Execution

- **Environment Setup**: Run `setup.bat` FIRST to create venv, install dependencies, and train filler model if needed
- **Processing**: Use `process.bat <input_path> [options]` to run the tool on files/directories
- **Virtual Environment**: All Python commands must run within `.venv` (activated by setup/process.bat)
- **GPU Requirement**: NVIDIA GPU with CUDA 12.6 required for reasonable performance (Pass 2 ASR)

## Core Principles (VERIFIED)

- **Maintain Synchronization**: All cuts MUST happen at same timestamps across videos. Always use `calculate_keep_segments` on consolidated `cutting_segments` to generate `keep_segments` for all streams
- **Virtual Cutting**: Tool generates Kdenlive projects with virtual cuts (`in`/`out` points) - faster and lossless
- **Audio Filter Order**: In `exporter.py`, volume filters (muting spikes) MUST be applied BEFORE selection filters (`aselect`) because spikes detected on original un-cut audio
- **Timestamp Adjustment**: When generating sidecar files (ASS/Kdenlive), use `adjust_timestamps` from `interval_utils.py` to map original timestamps to cut timeline

## Critical Architecture

- **Entry Point**: `video_processor.py` - main CLI logic and batch orchestration
- **Processing Flow**: 
  1. `audio_utils.py`: Extract audio, detect silence/spikes/repetitions
  2. `filler_processor.py`: Pass 1 ASR (CrisperWhisper) for boundaries/fillers
  3. `transcription_processor.py`: Pass 2 ASR (Qwen3-ASR) for transcription
  4. `interval_utils.py`: Merge/invert/adjust time intervals
  5. `exporter.py`: Generate Kdenlive/XML outputs
- **Cache**: Metadata stored as JSON files in `video_processing_work` directory (parent of input source)
- **Output**: `project.kdenlive` and rendered videos placed in parent directory of input source

## Filler Detection System

- **Training**: Automatic via `setup.bat` when `filler_detector.pth` missing
- **Dataset**: Requires `PodcastFillerDataset\` with `PodcastFillers.csv` and `clip_wav\` folder
- **Hungarian Support**: Additional training on `fillers_hun\ZO_Hungarian.csv` with 5x weight when detected
- **Model**: Overwrites `filler_detector.pth` (not separate file)

## Testing

- **Unit Tests**: Use `test_modular_processor.py` (mock-based) to verify changes without large MKV/GPU
- **Dependencies**: Managed via `uv`; requirements.txt specifies exact versions
- **FFmpeg**: Must be in system PATH for audio/video processing

## Key Constraints

- **Librosa Usage**: Loads audio into memory; consider block-based processing for large files
- **NeMo**: Still used for VAD/filler detection in some configurations
- **Qwen-ASR**: Pass 2 transcription requires GPU for reasonable performance

## File & Directory Conventions (OBSERVED)

- **Input**: Accepts `.mp3`, `.wav`, `.m4a`, `.mkv`, `.mp4`, `.avi` files
- **Work Directory**: Creates `video_processing_work` in PARENT directory of input source for:
  - Extracted/normalized mono WAV files (`*_a0.wav`)
  - Cache files: `*.markers.json` (silence/spikes), `*.asr.json` (transcription), `*.filler_*.json` (filler detection), `*.reps.json` (repetitions), `*.vad.json` (voice activity)
  - Generated subtitle files: `*.ass`, `*.srt`
- **Output**: Places `project.kdenlive` and rendered videos in PARENT directory of input source
- **Naming Pattern**: Test files follow timestamp-description-uuid--email format (observed in test_data)
