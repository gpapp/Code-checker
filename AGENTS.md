# Agent Instructions for Video Processing Tool

This project automates video editing based on audio features. When working on this codebase, please follow these guidelines:

## Core Principles

- **Maintain Synchronization**: All cuts MUST happen at the same timestamps across all input videos. Always use `calculate_keep_segments` on the consolidated `cutting_segments` to generate the `keep_segments` used by all streams.
- **Kdenlive-Native Logic**: When generating a Kdenlive project without rendering, the tool creates multiple `<entry>` tags per track with `in` and `out` points in frames.
- **Timestamp Adjustment**: When generating sidecar files (ASS, Kdenlive project), use `adjust_timestamps` from `interval_utils.py` to map original timestamps to the newly cut timeline.
- **FFmpeg Filter Order**: In `exporter.py`, ensure volume filters (for muting spikes) are applied *before* the selection filters (`aselect`), because spikes are detected on the original un-cut audio.

## Dependencies

- **NeMo**: This project relies on `nemo_toolkit[asr]`. Be aware that loading these models can be memory-intensive.
- **Librosa**: Used for acoustic similarity and basic silence detection. It loads audio into memory; for very large files, consider block-based processing if memory becomes an issue.

## Development

- **Testing**: Use `test_modular_processor.py` (or similar mock-based tests) to verify changes without requiring large MKV files or a GPU.
- **FFmpeg**: Always verify that the complex filter strings generated in `exporter.py` are valid, especially when dealing with multiple audio streams.
