import os
import argparse
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

    def train(self, csv_path, clips_dir, epochs=5, batch_size=128):
        """Trains the model on the PodcastFillers dataset and saves weights."""
        print(f"Starting training on {self.device}...")
        dataset = PodcastFillersDataset(csv_path, clips_dir, target_sr=self.sample_rate)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
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

    def detect_fillers(self, audio_path, threshold=0.7, window_sec=1.0, stride_sec=0.25):
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
        
        intervals = []
        with torch.no_grad():
            for start in tqdm(range(0, total_samples - win_samples, stride_samples), desc="Analyzing Audio Features", leave=False):
                chunk = waveform[:, start : start + win_samples].to(self.device).unsqueeze(0)
                if self.model(chunk).item() > threshold:
                    # Return in SECONDS for compatibility with video_processor
                    intervals.append((start / self.sample_rate, (start + win_samples) / self.sample_rate))
        
        return self._merge_intervals_seconds(intervals)

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
        # The column is 'label_consolidated_vocab' and we want 'Uh' or 'Um'
        self.df['is_filler'] = self.df['label_consolidated_vocab'].apply(lambda x: 1 if x in ['Uh', 'Um'] else 0)
        self.clips_dir = clips_dir
        self.target_sr = target_sr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Dataset structure: clip_wav/{split}/{clip_name}
        path = os.path.join(self.clips_dir, row['clip_split_subset'], row['clip_name'])
        try:
            # Using basic soundfile loading for dataset to be robust
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
        
        return wav, torch.tensor(row['is_filler'], dtype=torch.float32)

# ==========================================
# 4. CLI INTERFACE
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'process'], required=True)
    parser.add_argument("--csv", type=str)
    parser.add_argument("--clips_dir", type=str)
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", default="cleaned.wav")
    parser.add_argument("--model", default="filler_detector.pth")
    args = parser.parse_args()

    # Using the library
    filler_lib = PodcastFillerLib(model_path=args.model)

    if args.mode == 'train':
        filler_lib.train(args.csv, args.clips_dir)
    elif args.mode == 'process':
        filler_lib.process_file(args.input, args.output)