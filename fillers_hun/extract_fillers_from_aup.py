#!/usr/bin/env python3
"""
Extract every wave clip from every audio track in an Audacity AUP3 project
into 16kHz mono FLAC files, named {track_name}_{seq:04d}.flac.

Usage:
    uv run python fillers_hun/extract_fillers_from_aup.py path/to/fillers.aup3
"""

import argparse
import csv
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from aup3_python import Aup3Project

TARGET_SR = 16000


def main():
    parser = argparse.ArgumentParser(
        description="Extract all wave clips from Audacity AUP3 project"
    )
    parser.add_argument("aup3_path", type=str, help="Path to .aup3 file")
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output directory for filler clips (default: fillers_hun/training/)",
    )
    parser.add_argument(
        "--non-filler-dir", type=str, default=None,
        help="Export last audio track here instead of --output",
    )
    args = parser.parse_args()

    src = Path(args.aup3_path).resolve(strict=True)
    out_dir = Path(args.output or Path(__file__).resolve().parent / "training")
    out_dir.mkdir(parents=True, exist_ok=True)
    non_filler_dir = Path(args.non_filler_dir).resolve() if args.non_filler_dir else None

    proj = Aup3Project(str(src))
    tracks = proj.wave_tracks
    print(f"Wave tracks: {proj.list_wave_tracks()}")

    csv_rows = []

    for i, track in enumerate(tqdm(tracks, desc="Tracks")):
        dest = non_filler_dir if (non_filler_dir and i == len(tracks) - 1) else out_dir
        dest.mkdir(parents=True, exist_ok=True)

        for idx, clip in enumerate(track.clips):
            clip_dur = proj._clip_duration(clip)
            if clip_dur <= 0.01 or clip_dur > 60:
                continue

            audio = proj._extract_from_track(
                track, clip.offset, clip.offset + clip_dur
            )
            if audio is None or len(audio) == 0:
                continue

            src_sr = track.sample_rate
            if src_sr != TARGET_SR:
                tensor = torch.from_numpy(audio).float().unsqueeze(0)
                resampler = torchaudio.transforms.Resample(
                    orig_freq=int(src_sr), new_freq=TARGET_SR
                )
                tensor = resampler(tensor)
                audio = tensor.squeeze(0).numpy()

            np.clip(audio, -1.0, 1.0, out=audio)
            audio_int16 = (audio * 32767).astype(np.int16)

            clip_name = f"{track.name}_{idx:04d}.flac"
            out_path = dest / clip_name
            sf.write(str(out_path), audio_int16, TARGET_SR)

            csv_rows.append({
                "clip_name": clip_name,
                "track": track.name,
                "dest": str(dest),
            })

    proj.close()

    print(f"\nExtracted {len(csv_rows)} clips total.")
    for dest_name, dest_dir in [("filler", out_dir), ("non-filler", non_filler_dir)]:
        if dest_dir:
            clips_in_dir = [r for r in csv_rows if r["dest"] == str(dest_dir)]
            csv_path = dest_dir / "wave_clips.csv"
            with open(str(csv_path), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["clip_name", "track"])
                w.writeheader()
                w.writerows({"clip_name": r["clip_name"], "track": r["track"]} for r in clips_in_dir)
            print(f"  {dest_name}: {len(clips_in_dir)} clips -> {csv_path}")


if __name__ == "__main__":
    main()
