import os
import argparse
from pathlib import Path
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from pydub import AudioSegment
from tqdm import tqdm
import soundfile as sf

# ==========================================
# 1. CORE MODEL ARCHITECTURE
# ==========================================
class FillerDetectionCNN(nn.Module):
    """
    A lightweight CNN optimized for local RTX 3060 execution.
    Converts raw audio to Mel Spectrograms and classifies filler words.
    """
    def __init__(self, sample_rate=16000):
        super(FillerDetectionCNN, self).__init__()
        self.sample_rate = sample_rate
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=64
        )
        
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.mel_spectrogram(x)
        x = torch.log(x + 1e-9) 
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ==========================================
# 2. LIBRARY LOGIC CLASS
# ==========================================
class PodcastFillerLib:
    def __init__(self, model_path="filler_detector.pth", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.sample_rate = 16000
        self.model = FillerDetectionCNN(sample_rate=self.sample_rate).to(self.device)
        
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()

    def train(self, csv_path, clips_dir, hun_csv=None, hun_clips_dir=None, hun_weight=1.0, epochs=5, batch_size=256):
        """Trains the model on the PodcastFillers dataset and saves weights.
        
        Args:
            csv_path: Path to English CSV file
            clips_dir: Path to English clips directory
            hun_csv: Path to Hungarian CSV file (optional)
            hun_clips_dir: Path to Hungarian clips directory (optional)
            hun_weight: Weight multiplier for Hungarian samples (default 1.0)
            epochs: Number of training epochs
            batch_size: Batch size for training (default 1024 for RTX 3060)
        """
        print(f"Starting training on {self.device}...")
        
        # Build datasets
        datasets = []
        dataset_weights = []

        # English dataset (optional — skip if csv_path points to Hungarian-format CSV)
        eng_dataset = None
        try:
            eng_dataset = PodcastFillersDataset(csv_path, clips_dir, target_sr=self.sample_rate)
        except Exception as e:
            print(f"[INFO] Skipping English dataset: {e}")

        if eng_dataset is not None and len(eng_dataset) > 0:
            datasets.append(eng_dataset)
            dataset_weights.extend([1.0] * len(eng_dataset))
            print(f"English dataset: {len(eng_dataset)} samples")
        else:
            print("English dataset: none")

        # Hungarian dataset (if provided)
        if hun_csv and hun_clips_dir and os.path.exists(hun_csv) and os.path.exists(hun_clips_dir):
            hun_dataset = HungarianFillersDataset(hun_csv, hun_clips_dir, target_sr=self.sample_rate)
            datasets.append(hun_dataset)
            dataset_weights.extend([hun_weight] * len(hun_dataset))
            print(f"Hungarian dataset: {len(hun_dataset)} samples (weight: {hun_weight})")
        elif hun_csv or hun_clips_dir:
            print("[WARNING] Hungarian CSV or clips directory provided but not found. Training English-only.")

        if not datasets:
            print("[ERROR] No datasets available for training.")
            return

        num_workers = 0

        if len(datasets) > 1:
            combined_dataset = CombinedDataset(datasets)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=dataset_weights,
                num_samples=len(dataset_weights),
                replacement=True
            )
            dataloader = DataLoader(combined_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
            print(f"Combined dataset: {len(combined_dataset)} samples with weighted sampling")
        else:
            dataset = datasets[0]
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            print(f"Local dataset: {len(dataset)} samples")
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for waveforms, labels in pbar:
                waveforms, labels = waveforms.to(self.device), labels.to(self.device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = self.model(waveforms)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})
            
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Training complete. Model saved to {self.model_path}")

    def _load_audio_tensor(self, audio_path):
        """Helper to load audio tensor reliably across platforms, falling back to soundfile if needed."""
        try:
            return torchaudio.load(audio_path)
        except Exception:
            # Fallback for Windows or systems without torchcodec/ffmpeg backends
            data, sr = sf.read(audio_path)
            waveform = torch.from_numpy(data).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T
            return waveform, sr

    def detect_fillers(self, audio_path, threshold=0.7, window_sec=1.0, stride_sec=0.25, gap_sec=0.3):
        """Public API to detect fillers. Returns intervals in seconds [(start, end), ...]."""
        self.model.eval()
        waveform, sr = self._load_audio_tensor(audio_path)

        # Convert to mono and resample if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)

        total_samples = waveform.shape[1]
        win_samples = int(window_sec * self.sample_rate)
        stride_samples = int(stride_sec * self.sample_rate)

        # Coarse pass: detect rough filler regions
        intervals = []
        with torch.no_grad():
            for start in tqdm(range(0, total_samples - win_samples, stride_samples), desc="Analyzing Audio Features", leave=False):
                chunk = waveform[:, start : start + win_samples].to(self.device).unsqueeze(0)
                if self.model(chunk).item() > threshold:
                    intervals.append((start, start + win_samples))
        if not intervals:
            return []

        # Merge overlapping coarse intervals (in samples)
        gap_samples = int(gap_sec * self.sample_rate)
        intervals.sort(key=lambda x: x[0])
        merged = [[intervals[0][0], intervals[0][1]]]
        for s, e in intervals[1:]:
            if s <= merged[-1][1] + gap_samples:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])

        # Refine boundaries with fine stride for accurate duration
        return self._refine_boundaries(waveform, merged, threshold, win_samples, stride_samples)

    def _refine_boundaries(self, waveform, coarse_intervals, threshold, win_samples, stride_samples):
        """Refine start/end of each coarse interval with fine boundary scan.
        
        Returns refined intervals in seconds.
        """
        total_samples = waveform.shape[1]
        fine_stride = stride_samples // 4  # ~0.0625s at 16kHz
        # Base the search window on the detection parameters
        margin = int(0.5 * self.sample_rate)  # search 0.5s beyond each boundary

        refined = []
        with torch.no_grad():
            for start_samp, end_samp in coarse_intervals:
                # --- Refine start ---
                search_start = max(0, start_samp - margin)
                ref_start = start_samp
                found_start = False
                for t in range(search_start, min(start_samp + win_samples, total_samples - win_samples), fine_stride):
                    chunk = waveform[:, t : t + win_samples].to(self.device).unsqueeze(0)
                    if self.model(chunk).item() > threshold:
                        ref_start = t
                        found_start = True
                        break
                if not found_start:
                    continue  # skip if boundary no longer confirmed

                # --- Refine end ---
                search_end = min(total_samples - win_samples, end_samp + margin)
                ref_end = end_samp
                last_above = end_samp
                for t in range(max(0, end_samp - win_samples), search_end + fine_stride, fine_stride):
                    chunk = waveform[:, t : t + win_samples].to(self.device).unsqueeze(0)
                    if self.model(chunk).item() > threshold:
                        last_above = t + win_samples
                ref_end = max(last_above, ref_end)

                duration_sec = (ref_end - ref_start) / self.sample_rate
                if duration_sec > 0.01:
                    refined.append((ref_start / self.sample_rate, ref_end / self.sample_rate))

        return refined

    def _merge_intervals_seconds(self, intervals, gap_sec=0.3):
        if not intervals: return []
        intervals.sort(key=lambda x: x[0])
        merged = [[intervals[0][0], intervals[0][1]]]
        for curr_s, curr_e in intervals[1:]:
            prev_s, prev_e = merged[-1]
            if curr_s <= prev_e + gap_sec:
                merged[-1] = [prev_s, max(prev_e, curr_e)]
            else:
                merged.append([curr_s, curr_e])
        return [tuple(x) for x in merged]

    def process_file(self, input_path, output_path, threshold=0.7):
        """Detects and removes fillers from a long audio file."""
        intervals_ms = self._detect_ms(input_path, threshold)
        self._remove_and_save(input_path, output_path, intervals_ms)

    def _detect_ms(self, audio_path, threshold, window_sec=1.0, stride_sec=0.25):
        # Kept for backward compatibility with process_file/remove_and_save
        intervals_sec = self.detect_fillers(audio_path, threshold, window_sec, stride_sec)
        return [[s * 1000, e * 1000] for s, e in intervals_sec]

    def _remove_and_save(self, input_path, output_path, intervals):
        audio = AudioSegment.from_file(input_path)
        merged = self._merge_intervals(intervals)
        
        keep = []
        last_end = 0
        for start, end in merged:
            keep.append((last_end, start))
            last_end = end
        keep.append((last_end, len(audio)))
        
        clean = AudioSegment.empty()
        for s, e in keep:
            if e > s: clean += audio[s:e].fade_in(20).fade_out(20)
            
        clean.export(output_path, format="wav")
        print(f"Cleaned file saved to: {output_path}")

    def _merge_intervals(self, intervals, gap_ms=300):
        if not intervals: return []
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for curr in intervals[1:]:
            prev = merged[-1]
            if curr[0] <= prev[1] + gap_ms:
                merged[-1] = [prev[0], max(prev[1], curr[1])]
            else:
                merged.append(curr)
        return merged

# ==========================================
# 3. HELPER DATASET CLASS
# ==========================================
class PodcastFillersDataset(Dataset):
    def __init__(self, csv_file, clips_dir, target_sr=16000):
        self.df = pd.read_csv(csv_file)
        self.df['is_filler'] = self.df['label_consolidated_vocab'].apply(lambda x: 1 if x in ['Uh', 'Um'] else 0)
        self.clips_dir = clips_dir
        self.target_sr = target_sr
        self._preload()

    def _preload(self):
        """Preload all audio into RAM to eliminate I/O bottleneck."""
        print(f"  Preloading {len(self.df)} samples into RAM...")
        self.wavs = []
        self.labels = []
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="  Loading", leave=False):
            path = os.path.join(self.clips_dir, row['clip_split_subset'], row['clip_name'])
            try:
                data, sr = sf.read(path)
                wav = torch.from_numpy(data).float()
                if wav.ndim == 1: wav = wav.unsqueeze(0)
                else: wav = wav.T
                if sr != self.target_sr:
                    wav = torchaudio.transforms.Resample(sr, self.target_sr)(wav)
            except Exception:
                wav = torch.zeros(1, self.target_sr)
            if wav.shape[1] > self.target_sr: wav = wav[:, :self.target_sr]
            else: wav = torch.nn.functional.pad(wav, (0, self.target_sr - wav.shape[1]))
            self.wavs.append(wav)
            self.labels.append(torch.tensor(row['is_filler'], dtype=torch.float32))
        print(f"  Preloaded {len(self.wavs)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.wavs[idx], self.labels[idx]


class HungarianFillersDataset(Dataset):
    """
    Dataset for Hungarian filler detection with flat CSV+clips structure.
    Expects CSV with columns: clip_name, consolidated_label
    And clips_dir containing all .wav files directly (no subdirs).
    """
    def __init__(self, csv_file, clips_dir, target_sr=16000):
        self.df = pd.read_csv(csv_file)
        self.df['is_filler'] = self.df['consolidated_label'].apply(
            lambda x: 1 if x in ['Filler', 'Uh', 'Um'] else 0
        )
        self.clips_dir = clips_dir
        self.target_sr = target_sr
        self._preload()

    def _preload(self):
        """Preload all audio into RAM to eliminate I/O bottleneck."""
        print(f"  Preloading {len(self.df)} samples into RAM...")
        self.wavs = []
        self.labels = []
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="  Loading", leave=False):
            path = os.path.join(self.clips_dir, row['clip_name'])
            try:
                data, sr = sf.read(path)
                wav = torch.from_numpy(data).float()
                if wav.ndim == 1: wav = wav.unsqueeze(0)
                else: wav = wav.T
                if sr != self.target_sr:
                    wav = torchaudio.transforms.Resample(sr, self.target_sr)(wav)
            except Exception:
                wav = torch.zeros(1, self.target_sr)
            if wav.shape[1] > self.target_sr: wav = wav[:, :self.target_sr]
            else: wav = torch.nn.functional.pad(wav, (0, self.target_sr - wav.shape[1]))
            self.wavs.append(wav)
            self.labels.append(torch.tensor(row['is_filler'], dtype=torch.float32))
        print(f"  Preloaded {len(self.wavs)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.wavs[idx], self.labels[idx]


class CombinedDataset(Dataset):
    """
    Combines multiple datasets for joint training.
    Provides unified interface for mixed dataset training.
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_sizes = self._get_cumulative_sizes()
        
    def _get_cumulative_sizes(self):
        sizes = [len(dataset) for dataset in self.datasets]
        cumulative = []
        total = 0
        for size in sizes:
            total += size
            cumulative.append(total)
        return cumulative
        
    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)
        
    def __getitem__(self, idx):
        # Map global index to dataset-local index
        dataset_idx = 0
        while dataset_idx < len(self.cumulative_sizes) and idx >= self.cumulative_sizes[dataset_idx]:
            dataset_idx += 1
            
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            
        return self.datasets[dataset_idx][sample_idx]

# ==========================================
# 4. CLI INTERFACE
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'process', 'train_fillers'], required=True)
    parser.add_argument("--csv", type=str)
    parser.add_argument("--clips_dir", type=str)
    parser.add_argument("--hun_csv", type=str, help="Path to Hungarian CSV file")
    parser.add_argument("--hun_clips_dir", type=str, help="Path to Hungarian clips directory")
    parser.add_argument("--hun_weight", type=float, default=1.0, help="Weight multiplier for Hungarian samples")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", default="cleaned.wav")
    parser.add_argument("--model", default="filler_detector.pth")
    parser.add_argument("--filler-dir", type=str, help="Directory of positive (filler) FLAC clips")
    parser.add_argument("--non-filler-dir", type=str, help="Directory of negative (non-filler) FLAC clips")
    parser.add_argument("--eng_csv", type=str, help="Path to English PodcastFillers CSV (optional in train_fillers mode)")
    parser.add_argument("--eng_clips_dir", type=str, help="Path to English clips directory (optional in train_fillers mode)")
    args = parser.parse_args()

    filler_lib = PodcastFillerLib(model_path=args.model)

    if args.mode == 'train_fillers':
        import csv
        import glob
        pos_dir = Path(args.filler_dir).resolve()
        neg_dir = Path(args.non_filler_dir).resolve()
        parent = pos_dir.parent  # e.g. fillers_hun/
        tmp_csv = parent / "_train_metadata.csv"

        pos_name = pos_dir.name  # e.g. "training"
        neg_name = neg_dir.name  # e.g. "non_filler"

        rows = []
        for fpath in glob.glob(str(pos_dir / "*.flac")):
            rows.append({"clip_name": f"{pos_name}/{Path(fpath).name}", "consolidated_label": "Filler"})
        for fpath in glob.glob(str(neg_dir / "*.flac")):
            rows.append({"clip_name": f"{neg_name}/{Path(fpath).name}", "consolidated_label": "NonFiller"})

        with open(str(tmp_csv), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["clip_name", "consolidated_label"])
            w.writeheader()
            w.writerows(rows)
        print(f"Built training metadata: {len(rows)} samples ({sum(1 for r in rows if r['consolidated_label']=='Filler')} filler, {sum(1 for r in rows if r['consolidated_label']=='NonFiller')} non-filler)")

        eng_csv = args.eng_csv or str(tmp_csv)
        eng_clips = args.eng_clips_dir or str(parent)

        filler_lib.train(
            csv_path=eng_csv,
            clips_dir=eng_clips,
            hun_csv=str(tmp_csv),
            hun_clips_dir=str(parent),
            hun_weight=5.0,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        tmp_csv.unlink(missing_ok=True)

    elif args.mode == 'train':
        filler_lib.train(
            csv_path=args.csv,
            clips_dir=args.clips_dir,
            hun_csv=args.hun_csv,
            hun_clips_dir=args.hun_clips_dir,
            hun_weight=args.hun_weight,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    elif args.mode == 'process':
        filler_lib.process_file(args.input, args.output)