#!/usr/bin/env python3
"""
Ingest script for Hungarian ZO filler dataset.
Resamples ZO files from 48kHz to 16kHz and generates combined CSV.

For clips longer than 1 second, extracts multiple overlapping 1-second windows
to preserve all audio content and generate more training data. This is critical
because the filler detection model uses 1-second sliding windows during inference.

Source data structure (fillers_hun/ZO/):
  - Bad1-xxx.wav, Bad2-xxx.wav: True filler clips (varying length 0-2.1s)
  - False-xxx.wav: Non-filler clips (varying length 0.3-3.8s, often >1s)

NOTE: fillers_hun/clips/ (HUN_FILLER_xxxx.wav from extraxt_fillers_from_podcast.py)
      is NOT used for training. Only ZO clips + English PodcastFillers dataset are used.
"""
import os
import torch
import torchaudio
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# Window size must match inference window (1 second at 16kHz)
WINDOW_SEC = 1.0
# Overlap stride for extracting windows from long clips  
WINDOW_STRIDE_SEC = 0.5  # 50% overlap gives good coverage


def process_zo_files(zo_dir, output_csv_dir, output_clips_dir, target_sr=16000):
    """
    Process ZO files: resample from 48kHz to 16kHz and generate CSV.
    
    For clips longer than WINDOW_SEC, extracts multiple overlapping windows
    to preserve all content. For shorter clips, pads with silence.
    
    Args:
        zo_dir: Directory containing ZO files (Bad1-xx, Bad2-xx, False-xx)
        output_csv_dir: Directory to save output CSV
        output_clips_dir: Directory to save resampled audio clips
        target_sr: Target sample rate (default 16000)
    """
    # Create output directories
    os.makedirs(output_clips_dir, exist_ok=True)
    os.makedirs(output_csv_dir, exist_ok=True)
    
    target_length = int(WINDOW_SEC * target_sr)  # samples per window
    stride_length = int(WINDOW_STRIDE_SEC * target_sr)
    
    # Get all files
    all_files = os.listdir(zo_dir)
    
    # Categorize files
    bad1_files = sorted([f for f in all_files if f.startswith('Bad1-') or f.startswith('1Bad1-')])
    bad2_files = sorted([f for f in all_files if f.startswith('Bad2-')])
    false_files = sorted([f for f in all_files if f.startswith('False-')])
    
    print(f"Found {len(bad1_files)} Bad1 files (true fillers)")
    print(f"Found {len(bad2_files)} Bad2 files (true fillers)")  
    print(f"Found {len(false_files)} False files (non-fillers)")
    
    # Combine true fillers
    filler_files = bad1_files + bad2_files
    non_filler_files = false_files
    
    # Prepare data for CSV
    data_rows = []
    filler_clip_id = 0
    nonfiller_clip_id_start = 10000  # offset to avoid ID collisions
    nonfiller_clip_count = 0
    
    # Stats tracking
    stats = {'filler_clips': 0, 'filler_windows': 0, 
             'nonfiller_clips': 0, 'nonfiller_windows': 0,
             'filler_truncated': 0, 'filler_padded': 0,
             'nonfiller_truncated': 0, 'nonfiller_padded': 0}
    
    def load_audio_file(filepath):
        """Load audio file with fallback for Windows compatibility."""
        try:
            # Try torchaudio first
            waveform, sr = torchaudio.load(filepath)
        except Exception:
            # Fallback to soundfile
            data, sr = sf.read(filepath)
            waveform = torch.from_numpy(data).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T
        return waveform, sr
    
    def extract_windows(waveform, target_length, stride_length):
        """Extract overlapping windows from a waveform.
        
        For short clips (<= target_length): returns one padded window.
        For long clips: returns multiple overlapping windows.
        """
        num_samples = waveform.shape[1]
        windows = []
        
        if num_samples <= target_length:
            # Short clip: pad to target_length
            padding = target_length - num_samples
            padded = torch.nn.functional.pad(waveform, (0, padding))
            windows.append(padded)
        else:
            # Long clip: extract overlapping windows
            start = 0
            while start + target_length <= num_samples:
                window = waveform[:, start:start + target_length]
                windows.append(window)
                start += stride_length
            
            # If the last window didn't reach the end, add a final window
            # aligned to the end of the clip (ensures we don't miss tail content)
            if start < num_samples and (num_samples - target_length) != (start - stride_length):
                window = waveform[:, num_samples - target_length:num_samples]
                windows.append(window)
        
        return windows
    
    # Process filler files
    print(f"\nProcessing filler files (window={WINDOW_SEC}s, stride={WINDOW_STRIDE_SEC}s)...")
    for filename in tqdm(filler_files, desc="Fillers"):
        src_path = os.path.join(zo_dir, filename)
        
        # Load audio
        waveform, sr = load_audio_file(src_path)
        
        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Skip near-empty clips
        if waveform.shape[1] < target_sr * 0.05:  # < 50ms
            continue
        
        stats['filler_clips'] += 1
        was_long = waveform.shape[1] > target_length
        if was_long:
            stats['filler_truncated'] += 1
        else:
            stats['filler_padded'] += 1
        
        # Extract windows
        windows = extract_windows(waveform, target_length, stride_length)
        stats['filler_windows'] += len(windows)
        
        for window in windows:
            clip_name = f"ZO_FILLER_{filler_clip_id:04d}.wav"
            dst_path = os.path.join(output_clips_dir, clip_name)
            audio_np = window.squeeze(0).numpy()
            sf.write(dst_path, audio_np, target_sr)
            
            data_rows.append({
                'clip_name': clip_name,
                'consolidated_label': 'Filler',
                'source_file': filename,
            })
            filler_clip_id += 1
    
    # Process non-filler files
    print("\nProcessing non-filler files...")
    nf_id = nonfiller_clip_id_start
    for filename in tqdm(non_filler_files, desc="Non-fillers"):
        src_path = os.path.join(zo_dir, filename)
        
        # Load audio
        waveform, sr = load_audio_file(src_path)
        
        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Skip near-empty clips
        if waveform.shape[1] < target_sr * 0.05:  # < 50ms
            continue
            
        stats['nonfiller_clips'] += 1
        was_long = waveform.shape[1] > target_length
        if was_long:
            stats['nonfiller_truncated'] += 1
        else:
            stats['nonfiller_padded'] += 1
        
        # Extract windows
        windows = extract_windows(waveform, target_length, stride_length)
        stats['nonfiller_windows'] += len(windows)
        
        for window in windows:
            clip_name = f"ZO_NONFILLER_{nf_id:04d}.wav"
            dst_path = os.path.join(output_clips_dir, clip_name)
            audio_np = window.squeeze(0).numpy()
            sf.write(dst_path, audio_np, target_sr)
            
            data_rows.append({
                'clip_name': clip_name,
                'consolidated_label': 'NonFiller',
                'source_file': filename,
            })
            nf_id += 1
    nonfiller_clip_count = nf_id - nonfiller_clip_id_start
    
    # Create DataFrame and save CSV (drop source_file column for training CSV)
    df = pd.DataFrame(data_rows)
    csv_path = os.path.join(output_csv_dir, "ZO_Hungarian.csv")
    df[['clip_name', 'consolidated_label']].to_csv(csv_path, index=False)
    
    print(f"\nProcessing complete!")
    print(f"{'='*50}")
    print(f"Source files:")
    print(f"  Filler source clips:     {stats['filler_clips']} ({stats['filler_padded']} padded, {stats['filler_truncated']} windowed)")
    print(f"  Non-filler source clips: {stats['nonfiller_clips']} ({stats['nonfiller_padded']} padded, {stats['nonfiller_truncated']} windowed)")
    print(f"Output (after windowing):")
    print(f"  Filler training clips:     {filler_clip_id}")
    print(f"  Non-filler training clips: {nonfiller_clip_count}")
    print(f"  Total training clips:      {filler_clip_id + nonfiller_clip_count}")
    print(f"  Expansion ratio:           {(filler_clip_id + nonfiller_clip_count) / (stats['filler_clips'] + stats['nonfiller_clips']):.1f}x")
    print(f"CSV saved to: {csv_path}")
    print(f"Clips saved to: {output_clips_dir}")

if __name__ == "__main__":
    # Configuration
    ZO_DIR = r"./ZO"
    OUTPUT_CSV_DIR = r"."
    OUTPUT_CLIPS_DIR = r"./zo_clips"
    
    process_zo_files(ZO_DIR, OUTPUT_CSV_DIR, OUTPUT_CLIPS_DIR)