#!/usr/bin/env python3
"""
Ingest script for Hungarian ZO filler dataset.
Resamples ZO files from 48kHz to 16kHz and generates combined CSV.
"""
import os
import torch
import torchaudio
import pandas as pd
import soundfile as sf
from tqdm import tqdm

def process_zo_files(zo_dir, output_csv_dir, output_clips_dir, target_sr=16000):
    """
    Process ZO files: resample from 48kHz to 16kHz and generate CSV.
    
    Args:
        zo_dir: Directory containing ZO files (Bad1-xx, Bad2-xx, False-xx)
        output_csv_dir: Directory to save output CSV
        output_clips_dir: Directory to save resampled audio clips
        target_sr: Target sample rate (default 16000)
    """
    # Create output directories
    os.makedirs(output_clips_dir, exist_ok=True)
    os.makedirs(output_csv_dir, exist_ok=True)
    
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
    clip_id = 0
    
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
    
    # Process filler files
    print("\nProcessing filler files...")
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
        
        # Ensure exactly 1 second (pad or truncate)
        target_length = target_sr  # 1 second of audio
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]  # truncate
        elif waveform.shape[1] < target_length:
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))  # pad
        
        # Save resampled clip using soundfile to avoid torchcodec dependency
        clip_name = f"ZO_FILLER_{clip_id:04d}.wav"
        dst_path = os.path.join(output_clips_dir, clip_name)
        # Convert to numpy and transpose for soundfile (expects samples x channels)
        audio_np = waveform.squeeze(0).numpy()
        sf.write(dst_path, audio_np, target_sr)
        
        # Add to dataset
        data_rows.append({
            'clip_name': clip_name,
            'consolidated_label': 'Filler'  # All fillers map to 'Filler' label
        })
        clip_id += 1
    
    # Process non-filler files
    print("\nProcessing non-filler files...")
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
        
        # Ensure exactly 1 second (pad or truncate)
        target_length = target_sr  # 1 second of audio
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]  # truncate
        elif waveform.shape[1] < target_length:
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))  # pad
        
        # Save resampled clip using soundfile to avoid torchcodec dependency
        clip_name = f"ZO_NONFILLER_{clip_id:04d}.wav"
        dst_path = os.path.join(output_clips_dir, clip_name)
        # Convert to numpy and transpose for soundfile (expects samples x channels)
        audio_np = waveform.squeeze(0).numpy()
        sf.write(dst_path, audio_np, target_sr)
        
        # Add to dataset
        data_rows.append({
            'clip_name': clip_name,
            'consolidated_label': 'NonFiller'  # Non-fillers map to 'NonFiller' label
        })
        clip_id += 1
    
    # Create DataFrame and save CSV
    df = pd.DataFrame(data_rows)
    csv_path = os.path.join(output_csv_dir, "ZO_Hungarian.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\nProcessing complete!")
    print(f"Total clips: {len(data_rows)}")
    print(f"Filler clips: {len(filler_files)}")
    print(f"Non-filler clips: {len(non_filler_files)}")
    print(f"CSV saved to: {csv_path}")
    print(f"Clips saved to: {output_clips_dir}")

if __name__ == "__main__":
    # Configuration
    ZO_DIR = r"C:\Users\gerge\source\repos\video-processing-tool\fillers_hun\ZO"
    OUTPUT_CSV_DIR = r"C:\Users\gerge\source\repos\video-processing-tool\fillers_hun"
    OUTPUT_CLIPS_DIR = r"C:\Users\gerge\source\repos\video-processing-tool\fillers_hun\zo_clips"
    
    process_zo_files(ZO_DIR, OUTPUT_CSV_DIR, OUTPUT_CLIPS_DIR)