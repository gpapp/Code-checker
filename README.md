# Video Processing Tool

A Python tool that automatically edits multi-track recordings by analyzing audio to detect silence, filler words, and overlapping speech — then produces gap-free renders and a Kdenlive project ready for fine-tuning.

## Why

Recording multi-track podcasts or interviews leaves you with hours of footage containing dead air, filler words ("öö", "izé", "um"), and overlapping talk. Manually scrubbing through each track to find the keepers is tedious and error-prone. This tool automates that entire first pass: it listens to every audio stream, identifies what to keep and what to cut, and outputs cleaned-up files plus a timeline you can open directly in Kdenlive.

## How It Works

The pipeline runs in 7 steps on one or more input files (video or audio):

1. **Audio Extraction & Normalization** — Each source is decoded to FLAC with two-pass loudness normalization (-14 LUFS) and dynamic-range compression, ensuring consistent levels across all tracks.
2. **Silence & Spike Detection** — `librosa`-based analysis finds extended silences (>= 2 s) and short audio spikes (clicks, plosives) to cut or suppress.
3. **ASR & Filler Detection** (optional, requires GPU):
   - *Pass 1* — A custom CNN (`PodcastFillerLib`) detects acoustic filler sounds in ~1 s.
   - *Pass 2* — Word-level ASR transcription. Choose between:
     - **Whisper** (`large-v3` via `faster-whisper`) — fast, accurate, good for Hungarian.
     - **Granite Speech** (`ibm-granite/granite-speech-4.1-2b-plus` via `transformers`) — IBM's speech model with built-in word-level timestamps.
   - *Pass 3* (optional) — Gemma 4 via Ollama cleans up the transcript text.
4. **VAD & Overlap Detection** — ASR word timestamps (or Silero VAD fallback) produce speech intervals; overlapping talk across tracks is flagged.
5. **Cut Planning** — Silence cuts, spike suppressions, and VAD intervals are merged into a unified "keep" timeline shared across all sources.
6. **Rendering** — Each source is cut to its keep segments and concatenated into a single gap-free file using the best available hardware encoder (`h264_nvenc` > `h264_qsv` > `libx264`).
7. **Export** — ASS/SRT subtitles with cut-adjusted timestamps and a Kdenlive project referencing the rendered files.

## Installation

### Prerequisites

- [Python 3.13](https://www.python.org/)
- [uv](https://github.com/astral-sh/uv) — environment and package manager
- [FFmpeg](https://ffmpeg.org/) and `ffprobe` — must be on system PATH
- NVIDIA GPU with CUDA support (e.g. RTX 3060) — required for ASR; optional if you skip ASR

### Setup

#### Windows (Recommended)

```
setup.bat
```

This creates a `.venv` (Python 3.13), installs all dependencies via `uv`, installs PyTorch with CUDA 12.6 support, and trains the filler detection model if `filler_detector.pth` is missing and training data is present.

#### Linux / macOS

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install torch torchvision torchaudio torchcodec --upgrade --index-url https://download.pytorch.org/whl/cu126
```

### Optional: Filler Detection Dataset

For filler word detection, a custom CNN model is used. To train it:

1. Download [PodcastFillers](https://zenodo.org/records/7121457) from Zenodo.
2. Place `PodcastFillers.csv` and the `clip_wav/` folder into `PodcastFillerDataset/` at the project root.
3. Re-run `setup.bat` — it will train automatically if `filler_detector.pth` is missing.

## Usage

```bash
process.bat "C:\Recordings\episode42"
```

Or directly with `uv`:

```bash
uv run python video_processor.py "C:\Recordings\episode42" --video-offsets 0.5
```

### CLI Reference

| Flag | Description |
|------|-------------|
| `--asr-engine {whisper,granite}` | ASR backend. If omitted, ASR is skipped entirely. |
| `--language CODE` | Language code for ASR (default: `hu` for Hungarian). |
| `--refine` | Enable Pass 3 — Gemma 4 transcript cleanup via Ollama. |
| `--video-offsets LIST` | Comma-separated per-track offsets in seconds (e.g. `0.5,-0.2,0`). |
| `--silence-threshold DB` | Silence detection threshold in dB (default: `-40.0`). |
| `--silence-duration SEC` | Minimum silence length to cut (default: `2.0`). |
| `--spike-duration SEC` | Maximum spike length to suppress (default: `0.2`). |
| `--overlap-duration SEC` | Minimum overlap to flag (default: `5.0`). |
| `--filler-threshold FLOAT` | CNN filler confidence threshold 0.0–1.0 (default: `0.9`). |
| `--filler-merge-gap SEC` | Max gap between fillers to merge (default: `0.05`). |
| `--full-render` | Fully re-encode instead of lossless-cut hybrid. |
| `--restart` | Delete all cached artifacts before processing. |
| `--noclear` | Keep temporary cache files after processing. |

### Examples

```bash
# Whisper ASR with Hungarian language
uv run python video_processor.py input.mkv --asr-engine whisper --language hu

# Granite Speech ASR with refinement
uv run python video_processor.py input.mkv --asr-engine granite --refine

# Skip ASR, use silence-only cutting
uv run python video_processor.py input.mkv

# Full pipeline with offset alignment
process.bat "C:\Recordings" --asr-engine whisper --video-offsets 0.3,-0.1 --refine
```

## Output Structure

```
grandparent_dir/                          # e.g. videos/
├── {grandparent_name}.kdenlive           # Kdenlive project referencing PROCESSED files
├── PROCESSED/                            # Rendered gap-free files
│   ├── {track}_processed.mp4             # H.264 video + FLAC audio
│   └── {track}_processed.flac            # Audio-only
└── input_source_dir/                     # e.g. recordings/
    ├── source1.mkv
    ├── source2.mkv
    └── video_processing_work/            # Cache + pre-processed audio
        ├── {track}_a0.flac               # Normalized FLAC (-14 LUFS)
        ├── *.asr.json                    # ASR word timestamps
        ├── *.markers.json                # Silence/spike markers
        ├── *.vad.json                    # Speech intervals
        ├── *.filler_cnn.json             # CNN filler detection
        ├── *.ass / *.srt                 # Subtitles (cut-adjusted)
        └── ...
```

## Architecture

| Module | Role |
|--------|------|
| `video_processor.py` | Entry point — CLI, pipeline orchestration |
| `audio_utils.py` | Audio extraction, loudness normalization, silence/spike detection |
| `transcription_processor.py` | Whisper ASR backend (faster-whisper large-v3) |
| `granite_asr.py` | Granite Speech ASR backend (ibm-granite/granite-speech-4.1-2b-plus) |
| `filler_processor.py` | Silero VAD + shared model cache |
| `PodcastFillerLib.py` | CNN acoustic filler detector |
| `gemma_asr_service.py` | Gemma 4 transcript refinement via Ollama |
| `interval_utils.py` | Interval merging, keep-segment calculation, timestamp adjustment |
| `exporter.py` | FFmpeg rendering, Kdenlive XML generation, ASS/SRT export |

## Testing

```bash
uv run pytest tests/ -q
```

Unit tests cover interval math, exporter logic, audio utilities, and refinement — no GPU or input files required.
