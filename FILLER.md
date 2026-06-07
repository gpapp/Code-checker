# Filler Processing

Filler words ("hát", "szóval", "izé", "uh", "um") are detected using a lightweight CNN that runs on sliding 1-second windows at 16kHz. The model is trained on English (PodcastFillers) + Hungarian (ZO) datasets and produces probability scores per window.

## Detection Flow

```
FLAC → PodcastFillerLib.detect_fillers(threshold=0.9, gap_sec=0.05)
         │
         ├─ torchaudio.load → mono 16kHz
         ├─ 1s window, 0.25s stride
         ├─ MelSpectrogram(64 bins) + CNN → sigmoid probability
         └─ merge adjacent detections → [(start_s, end_s), ...]
```

- Results cached per FLAC as `<stem>.filler_cnn.json`
- Fillers falling within silent intervals (0.05s tolerance) are discarded
- Detected fillers are stored as markers and are **not cut** from output (informational only)

## Training

### Model: `FillerDetectionCNN`

| Layer | Shape |
|---|---|
| MelSpectrogram | n_fft=1024, hop_length=256, n_mels=64 |
| Conv2D + BN + ReLU + MP | 1→16, 3×3, pool 2 |
| Conv2D + BN + ReLU + MP | 16→32, 3×3, pool 2 |
| Conv2D + BN + ReLU + AAP | 32→64, 3×3, pool 1×1 |
| Linear + ReLU + Drop(0.3) | 64→32 |
| Linear + Sigmoid | 32→1 |

### Datasets

| Dataset | CSV | Clips Dir | Label Rule | Weight |
|---|---|---|---|---|
| English | `PodcastFillerDataset/PodcastFillers.csv` | `PodcastFillerDataset/clip_wav/` (has subdirs: train/test/validation/extra) | Uh/Um → 1, else 0 | 1.0 |
| Hungarian (ZO) | `fillers_hun/ZO_Hungarian.csv` | `fillers_hun/zo_clips/` (flat) | Filler/Uh/Um → 1, else 0 | 5.0 |

### Training command (from `setup.bat`)

```
python PodcastFillerLib.py --mode train ^
    --csv PodcastFillerDataset\PodcastFillers.csv ^
    --clips_dir PodcastFillerDataset\clip_wav ^
    --hun_csv fillers_hun\ZO_Hungarian.csv ^
    --hun_clips_dir fillers_hun\zo_clips ^
    --hun_weight 5.0 --epochs 5 --batch_size 256
```

### Window extraction for long clips (`ingest_zo.py`)

Clips >1s are split into overlapping 1-second windows (0.5s stride, 50% overlap) to match the inference window and generate more training data. Clips <1s are zero-padded to 1s. Clips <50ms are discarded.

## Adding New Hungarian Data

### From Audacity AUP3 projects

```
uv run python fillers_hun/extract_fillers_from_aup.py path/to/fillers.aup3
```

Reads label regions from each label track, pairs them with the preceding audio track (by track order), extracts each segment from the SQLite-embedded PCM data, resamples to 16kHz mono, and writes:

- `fillers_hun/training/{track_name}_{counter:04d}.flac`
- `fillers_hun/training/metadata.csv` (columns: `clip_name`, `consolidated_label`)

Labels with non-empty title → `Filler`, empty title → `NonFiller`.

### From ZO WAV files

```
uv run python fillers_hun/ingest_zo.py
```

Processes `fillers_hun/ZO/` (48kHz WAVs) → `fillers_hun/zo_clips/` + `ZO_Hungarian.csv`.

## Directory Structure

```
fillers_hun/
├── ZO/                      # Raw 48kHz WAV source (Bad1-*, Bad2-*, False-*)
├── zo_clips/                # Processed 16kHz clips from ZO (used for training)
├── training/                # AUP3-extracted clips (FLAC + metadata.csv)
├── clips/                   # OLD podcast clips (HUN_FILLER_*.wav, NOT used)
├── ZO_Hungarian.csv         # Training CSV for ZO dataset
├── PodcastFillers_HUN.csv   # CSV for old clips/ (NOT used for training)
├── NonFillers_HUN.csv       # Non-filler words (NOT used for training)
├── extract_fillers_from_aup.py
└── ingest_zo.py

PodcastFillerDataset/         # English dataset (from Zenodo 7121457)
├── PodcastFillers.csv
└── clip_wav/ (train/, test/, validation/, extra/ subdirs)

filler_detector.pth           # Trained model weights
```

## CLI Parameters

| Flag | Default | Effect |
|---|---|---|
| `--filler-threshold` | 0.9 | CNN confidence threshold |
| `--filler-merge-gap` | 0.05 | Max gap (s) between windows to merge |
| `--no-asr` | false | Skip ASR (Step 4) but still run filler detection |
