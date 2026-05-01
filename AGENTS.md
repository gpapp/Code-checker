# Agent Instructions for Video Processing Tool

## Setup & Execution

- **Setup**: Run `setup.bat` to create `.venv` (Python 3.13), install deps via `uv`, and train filler model if `filler_detector.pth` missing
- **Run**: `process.bat <input>` or `uv run python video_processor.py <input>` (activates `.venv` automatically)
- **GPU**: Pass 2 ASR (Qwen3-ASR) requires NVIDIA GPU with CUDA 12.6; use `--no-asr` to skip ASR for testing

## Critical Architecture

- **Entry Point**: `video_processor.py` orchestrates the pipeline
- **Processing Flow**: `audio_utils.py` → `filler_processor.py` (CrisperWhisper) → `transcription_processor.py` (Qwen3-ASR) → `interval_utils.py` → `exporter.py`
- **Synchronized Cuts**: ALL cuts must use same timestamps across videos. Call `calculate_keep_segments` on consolidated `cutting_segments` for all streams
- **Cache**: JSON files (`*.asr.json`, `*.markers.json`, etc.) stored in `video_processing_work/` (parent of input)
- **Output**: `project.kdenlive` placed in parent directory of input source

## Key Constraints

- **Audio Filter Order** (`exporter.py`): Apply volume filters (muting spikes) BEFORE `aselect` filters—spikes detected on original un-cut audio
- **Timestamp Adjustment**: Use `adjust_timestamps` from `interval_utils.py` when generating ASS/Kdenlive sidecars to map original timestamps to cut timeline
- **Librosa**: Loads entire audio into memory; monitor memory for large files
- **NeMo**: Used as VAD fallback in `filler_processor.py` when primary method fails

## File Conventions

- **Input**: `.mkv`, `.mp4`, `.avi`, `.mp3`, `.wav`, `.m4a`
- **Work files** (in `video_processing_work/`): `*_a0.wav` (extracted audio), `*.markers.json`, `*.asr.json`, `*.filler_*.json`, `*.reps.json`, `*.vad.json`
- **Filler model**: `filler_detector.pth` (single file, overwritten on retrain); Hungarian dataset in `fillers_hun/` trains with 5x weight

## Testing

- **Command**: `uv run pytest tests/ -q` (24 tests across 8 files)
- **Tests**: `tests/test_intervals.py`, `tests/test_exporter.py`, `tests/test_audio.py`, `tests/test_kdenlive_lib.py`, `tests/test_synthetic_kdenlive.py`, `tests/test_refinement.py`, `tests/test_export_output.py`
- **Mock-based**: No GPU/MKV needed for unit tests; `conftest.py` provides synthetic audio fixtures
- **New tests**: `test_export_output.py` verifies kdenlive structure (A1/A2 tracks, clip lengths, silence gaps)

## MLT/Kdenlive XML (for `exporter.py`)

See [`kdenlive/README.md`](kdenlive/README.md) for full reference. Critical rules:
- **Profile FPS**: Read from `<profile>` element (`frame_rate_num`/`frame_rate_den`), NOT hardcoded 25.0
- **Timecodes**: `HH:MM:SS:FF` format; `out` points are **inclusive** (`out = total_frames - 1`)
- **Transitions**: MUST blend against track 0 (`a_track="0"`) using `qtblend` (video) / `mix` (audio); never cascade transitions
- **Filters**: `volume` keyframes use `frame=level` format; clip effects on `<chain>`, track effects on `<tractor>`
- **Persistence**: `_reset_timelines()` IS called in `KdenliveProject.__init__()` to clear template bins/tracks; don't reload saved projects (re-opening clears entries)
