# Agent Instructions for Video Processing Tool

This project automates video editing based on audio features. When working on this codebase, please follow these guidelines:

## Core Principles

- **Maintain Synchronization**: All cuts MUST happen at the same timestamps across all input videos. Always use `calculate_keep_segments` on the consolidated `cutting_segments` to generate the `keep_segments` used by all streams.
- **Virtual Cutting (Preferred)**: The tool defaults to generating Kdenlive project files that reference original media with virtual cuts (`in`/`out` points). This is faster and lossless.
- **Audio Processing**: High-quality filters (acompressor, loudnorm, dynaudnorm) must be applied to Kdenlive virtual tracks using `avfilter.*` MLT services to ensure audio quality matches rendered output.
- **Timestamp Adjustment**: When generating sidecar files (ASS, Kdenlive project), use `adjust_timestamps` from `interval_utils.py` to map original timestamps to the newly cut timeline.
- **FFmpeg Filter Order**: In `exporter.py`, ensure volume filters (for muting spikes) are applied *before* the selection filters (`aselect`), because spikes are detected on the original un-cut audio.

## Next Tasks

- [ ] **Grouping**: In Kdenlive XML, group the video and its corresponding audio tracks together so they move as one in the timeline.
- [ ] **Track Naming**: Implement descriptive track names in Kdenlive based on file names or stream metadata.
- [ ] **Validation Tool**: Create a small script to validate the generated `.kdenlive` XML against MLT schema or common pitfalls (e.g. mismatched track lengths).
- [ ] **GPU Acceleration for Analysis**: Explore using GPU for `librosa` or `librosa`-like features (e.g. `torch-audio`) to speed up repetition detection.

## Dependencies

- **NeMo**: This project relies on `nemo_toolkit[asr]`. Be aware that loading these models can be memory-intensive.
- **Librosa**: Used for acoustic similarity and basic silence detection. It loads audio into memory; for very large files, consider block-based processing if memory becomes an issue.

## Development

- **Testing**: Use `test_modular_processor.py` (or similar mock-based tests) to verify changes without requiring large MKV files or a GPU.
- **FFmpeg/MLT**: Always verify that the complex filter strings or XML properties are valid. Use `melt` (MLT player) to check if the project opens correctly in a headless environment.
