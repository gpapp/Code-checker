# Agent Instructions for Video Processing Tool

## Setup & Execution

- **Setup**: Run `setup.bat` to create `.venv` (Python 3.13), install deps via `uv`, and train filler model if `filler_detector.pth` missing
- **Run**: `process.bat <input>` or `uv run python video_processor.py <input>` (activates `.venv` automatically)
- **GPU**: Pass 2 ASR (Qwen3-ASR) requires NVIDIA GPU with CUDA 12.6; use `--no-asr` to skip ASR for testing

## Critical Architecture

- **Entry Point**: `video_processor.py` orchestrates the pipeline
- **Processing Flow**: `audio_utils.py` (two-pass loudnorm → FLAC) → `filler_processor.py` (CrisperWhisper) → `transcription_processor.py` (Qwen3-ASR) → `interval_utils.py` → `exporter.py`
- **Synchronized Cuts**: ALL cuts must use same timestamps across videos. Call `calculate_keep_segments` on consolidated `cutting_segments` for all streams
- **Audio Processing**: All audio streams are processed to FLAC in `video_processing_work/` with two-pass loudnorm (-14 LUFS) + compression. Analysis runs on FLAC files; no raw WAVs are stored.
- **Cache**: JSON files (`*.asr.json`, `*.markers.json`, etc.) stored in `video_processing_work/` keyed by source file basename
- **Rendering**: `render_processed_video()` (step 6) concatenates keep segments per source into one file using best hardware encoder (`h264_nvenc` > `h264_qsv` > `libx264`). Presets favor speed: `p2` (NVENC), `veryfast` (QSV), `superfast` (libx264). Output goes to `grandparent_dir/PROCESSED/`.
- **Output**: `{grandparent_name}.kdenlive` placed in `grandparent_dir/` (no sequence-level audio filters). References rendered files in `PROCESSED/` by absolute path.
- **Timecode Architecture** (`mlt-python` + `exporter.py`): The MLT XML format uses `HH:MM:SS:FF` timecodes for `entry` in/out points. All time positions in `Clip`, `Blank`, `Filter`, `Transition` are stored as timecode strings. Frame conversion happens only at XML serialization boundary (`Blank.to_xml(fps)`) and arithmetic boundaries (`add_clip` computes inclusive `out_point` from exclusive `end`). The exporter works in float-seconds internally, converting to timecodes only when calling `Playlist.add_clip`/`add_blank`.

## Key Constraints

- **Audio Filter Order** (`exporter.py`): Apply volume filters (muting spikes) BEFORE `aselect` filters—spikes detected on original un-cut audio
- **Timestamp Adjustment**: Use `adjust_timestamps` from `interval_utils.py` when generating ASS/Kdenlive sidecars to map original timestamps to cut timeline
- **Librosa**: Loads entire audio into memory; monitor memory for large files
- **NeMo**: Used as VAD fallback in `filler_processor.py` when primary method fails
- **No Post-Processing Filters**: All compression and loudness normalization is baked into the PROCESSED FLAC files. `exporter.py` adds zero audio filters at the sequence level.

## File Conventions

- **Input**: `.mkv`, `.mp4`, `.avi`, `.mp3`, `.wav`, `.m4a`
- **Work files** (in `video_processing_work/`): `*.markers.json`, `*.asr.json`, `*.filler_*.json`, `*.reps.json`, `*.vad.json`
- **Filler model**: `filler_detector.pth` (single file, overwritten on retrain); Hungarian dataset in `fillers_hun/` trains with 5x weight
- **Rendered files** (in `grandparent_dir/PROCESSED/`): `*_processed.mp4` (video-only H.264), `*_processed.flac` (audio-only)
- **Kdenlive project**: `grandparent_dir/{grandparent_name}.kdenlive`

## Testing

- **Command**: `uv run pytest tests/ -q` (18 tests across 8 files)
- **Tests**: `tests/test_intervals.py`, `tests/test_exporter.py`, `tests/test_audio.py`, `tests/test_kdenlive_lib.py`, `tests/test_synthetic_kdenlive.py`, `tests/test_refinement.py`, `tests/test_export_output.py`, `tests/test_new_exporter.py`, `tests/test_marker_api.py`
- **Mock-based**: No GPU/MKV needed for unit tests; `conftest.py` provides synthetic audio fixtures

## MLT/Kdenlive XML (for `exporter.py`)

See `kdenlive/README.md` for full reference (if available). Critical rules:
- **Profile FPS**: Read from `<profile>` element (`frame_rate_num`/`frame_rate_den`), NOT hardcoded 25.0
- **Timecodes**: `HH:MM:SS:FF` format; `out` points are **inclusive** (`out = total_frames - 1`)
- **Transitions**: MUST blend against track 0 (`a_track="0"`) using `qtblend` (video) / `mix` (audio); never cascade transitions
- **Filters**: `volume` keyframes use `frame=level` format; clip effects on `<chain>`, track effects on `<tractor>`
- **Persistence**: `_reset_timelines()` IS called in `KdenliveProject.__init__()` to clear template bins/tracks; don't reload saved projects (re-opening clears entries)
- **Timecode Storage** (`mlt-python`): `Clip` stores `in_point`/`out_point` as timecode strings; `Blank` stores `length` as timecode string; `Filter`/`Transition` store in/out as timecode strings. Frame conversion happens only at XML serialization (`Blank.to_xml(fps)`) and when computing inclusive `out_point` from exclusive `end`/`duration`. Never work with frame numbers in `exporter.py`—use `Timecode.from_seconds(s, fps)` to build timecodes from float seconds.
