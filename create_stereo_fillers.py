import os
import sys
import json
import numpy as np
import librosa
from pydub import AudioSegment
from pydub.generators import Sine
from PodcastFillerLib import PodcastFillerLib

# Initialize the filler detector
pfl = PodcastFillerLib()

# Use test_data directory by default
master_dir = os.path.join(os.path.dirname(__file__), 'test_data')
test_files = [os.path.join(master_dir, f) for f in sorted(os.listdir(master_dir)) if f.endswith('.mp3')]

# Configuration
CONTEXT_DURATION = 0.5  # seconds of context before/after each segment
BEEP_DURATION = 0.1     # seconds for the beep
BEEP_FREQUENCY = 200    # Hz - low pitched beep
SEPARATOR_DURATION = 0.5 # seconds of silence between segments
BEEP_VOLUME = -20       # dB - not too loud

# ============================================
# TIGHTER DETECTION PARAMETERS
# ============================================
DETECTION_THRESHOLD = 0.995   # Raised from 0.99 - require very high confidence
MIN_FILLER_DURATION = 0.3     # seconds - minimum filler duration
MAX_FILLER_DURATION = 1.5     # seconds - real fillers rarely exceed 1.5s
WINDOW_SEC = 0.75             # seconds - shorter window catches actual fillers better
STRIDE_SEC = 0.15             # seconds - finer stride for more precise boundaries
MERGE_GAP_SEC = 0.15          # seconds - reduced from 0.3, less eager merging

# Require minimum consecutive positive windows to confirm a filler
# This prevents isolated false positives from passing through
MIN_CONSECUTIVE_WINDOWS = 3   # Need at least 3 adjacent windows to confirm


def detect_fillers_strict(pfl, audio_path, threshold=DETECTION_THRESHOLD,
                          window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC,
                          merge_gap=MERGE_GAP_SEC, min_consecutive=MIN_CONSECUTIVE_WINDOWS):
    """
    Stricter filler detection that requires multiple consecutive positive windows
    before accepting an interval. This reduces false positives significantly.
    """
    import torch
    import torchaudio
    
    pfl.model.eval()
    waveform, sr = pfl._load_audio_tensor(audio_path)
    
    # Convert to mono and resample if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != pfl.sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=pfl.sample_rate)(waveform)

    total_samples = waveform.shape[1]
    win_samples = int(window_sec * pfl.sample_rate)
    stride_samples = int(stride_sec * pfl.sample_rate)
    
    # Step 1: Collect all positive windows with their confidence scores
    positive_windows = []
    with torch.no_grad():
        for start in range(0, total_samples - win_samples, stride_samples):
            chunk = waveform[:, start : start + win_samples].to(pfl.device).unsqueeze(0)
            score = pfl.model(chunk).item()
            if score > threshold:
                t_start = start / pfl.sample_rate
                t_end = (start + win_samples) / pfl.sample_rate
                positive_windows.append((t_start, t_end, score))
    
    if not positive_windows:
        return []
    
    # Step 2: Group consecutive positive windows
    groups = []
    current_group = [positive_windows[0]]
    
    for i in range(1, len(positive_windows)):
        prev_end = positive_windows[i-1][1]
        curr_start = positive_windows[i][0]
        # Windows are consecutive if they overlap or are within merge_gap
        if curr_start <= prev_end + merge_gap:
            current_group.append(positive_windows[i])
        else:
            groups.append(current_group)
            current_group = [positive_windows[i]]
    groups.append(current_group)
    
    # Step 3: Filter groups by minimum consecutive windows
    intervals = []
    for group in groups:
        if len(group) >= min_consecutive:
            group_start = group[0][0]
            group_end = group[-1][1]
            avg_confidence = sum(w[2] for w in group) / len(group)
            duration = group_end - group_start
            
            # Only accept if duration is within bounds
            if MIN_FILLER_DURATION <= duration <= MAX_FILLER_DURATION:
                intervals.append((group_start, group_end, avg_confidence, len(group)))
    
    return intervals


def mute_fillers(audio, intervals):
    """Mute filler intervals from audio."""
    result = audio
    for item in intervals:
        start, end = item[0], item[1]
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        silence = AudioSegment.silent(duration=end_ms - start_ms, frame_rate=audio.frame_rate)
        if audio.channels == 2:
            silence = silence.set_channels(2)
        result = result[:start_ms] + silence + result[end_ms:]
    return result


def export_individual_segments(audio, intervals, output_dir, file_prefix, padded=True):
    """Export each detected segment as an individual WAV file for inspection.
    
    Args:
        audio: pydub AudioSegment
        intervals: list of (start, end, ...) tuples in seconds
        output_dir: directory to write segments
        file_prefix: prefix for filenames
        padded: if True, add CONTEXT_DURATION padding around each segment
    
    Returns:
        list of (output_path, start, end, duration) tuples
    """
    os.makedirs(output_dir, exist_ok=True)
    exported = []
    
    for i, item in enumerate(intervals):
        start, end = item[0], item[1]
        confidence = item[2] if len(item) > 2 else 0
        
        if padded:
            seg_start = max(0, start - CONTEXT_DURATION)
            seg_end = min(len(audio) / 1000.0, end + CONTEXT_DURATION)
            suffix = "padded"
        else:
            seg_start = start
            seg_end = end
            suffix = "exact"
        
        start_ms = int(seg_start * 1000)
        end_ms = int(seg_end * 1000)
        segment = audio[start_ms:end_ms]
        
        duration = (end_ms - start_ms) / 1000.0
        filename = f"{file_prefix}_{i:04d}_{suffix}_conf{confidence:.3f}.wav"
        output_path = os.path.join(output_dir, filename)
        segment.export(output_path, format="wav")
        exported.append((output_path, seg_start, seg_end, duration))
    
    return exported


def run_asr_verification(segment_paths, max_segments=50):
    """Run ASR on extracted segments to verify they don't contain real Hungarian words.
    
    Uses faster_whisper with Hungarian language model.
    Returns a report of what was transcribed.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[WARN] faster_whisper not installed, skipping ASR verification")
        return []
    
    print(f"\n{'='*60}")
    print(f"ASR VERIFICATION: Checking {min(len(segment_paths), max_segments)} segments")
    print(f"{'='*60}")
    
    # Use small model for quick verification
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    
    # Known filler words/sounds that are acceptable
    ACCEPTABLE_FILLERS = {
        "hát", "hat", "szóval", "izé", "ize",
        "öö", "őő", "ööö", "eee", "ee", "ő", "ö",
        "öhm", "őőő", "ühm", "ähm",
        "mm", "mhm", "hmm", "hm", "ah", "eh",
        "na", "nos", "nah", "nahát",
        "a", "az",  # often just noise/breath transcribed as articles
    }
    
    # Minimum ASR word probability to consider a word "real" (not hallucinated)
    # Whisper frequently hallucinates on short segments with low probability
    MIN_WORD_PROB = 0.7
    # Need at least this many confident non-filler words to flag a segment
    MIN_FLAGGING_WORDS = 2
    
    results = []
    flagged = []
    
    for i, (path, start, end, duration) in enumerate(segment_paths[:max_segments]):
        segments_iter, info = model.transcribe(
            path,
            language="hu",
            word_timestamps=True,
            vad_filter=False,
        )
        
        full_text = ""
        words = []
        no_speech_prob = 0.0
        for seg in segments_iter:
            full_text += seg.text + " "
            no_speech_prob = max(no_speech_prob, seg.no_speech_prob)
            if seg.words:
                words.extend([(w.word.strip(), w.probability) for w in seg.words])
        
        full_text = full_text.strip()
        
        # Check if transcription contains non-filler Hungarian words
        # Only count words with HIGH probability to filter out hallucinations
        confident_non_filler = []
        for word_text, prob in words:
            clean_word = word_text.lower().strip(".,!? ")
            if (clean_word and clean_word not in ACCEPTABLE_FILLERS 
                    and len(clean_word) > 1 and prob >= MIN_WORD_PROB):
                confident_non_filler.append(f"{clean_word}({prob:.2f})")
        
        # Only flag if we have multiple confident non-filler words
        # (single words are often hallucinated on short filler segments)
        is_clean = len(confident_non_filler) < MIN_FLAGGING_WORDS
        
        status = "✓ CLEAN" if is_clean else "✗ FLAGGED"
        nsp_tag = f" nsp={no_speech_prob:.2f}" if no_speech_prob > 0.3 else ""
        basename = os.path.basename(path)
        print(f"  [{status}] {basename}: \"{full_text}\"{nsp_tag}")
        if confident_non_filler:
            print(f"           Confident non-filler words: {', '.join(confident_non_filler)}")
        
        result = {
            'path': path,
            'text': full_text,
            'is_clean': is_clean,
            'non_filler_words': confident_non_filler,
            'no_speech_prob': no_speech_prob,
            'duration': duration,
        }
        results.append(result)
        if not is_clean:
            flagged.append(result)
    
    print(f"\n--- ASR Summary (hallucination-aware) ---")
    print(f"Total checked: {len(results)}")
    print(f"Clean (filler/silence/hallucination): {sum(1 for r in results if r['is_clean'])}")
    print(f"Flagged (≥{MIN_FLAGGING_WORDS} confident non-filler words): {len(flagged)}")
    
    if flagged:
        print(f"\nFlagged segments that may contain real speech:")
        for f in flagged:
            print(f"  {os.path.basename(f['path'])}: \"{f['text']}\" -> {f['non_filler_words']}")
    
    return results


# ============================================
# MAIN PROCESSING
# ============================================

# Collect all segments as raw numpy arrays
all_left_samples = []
all_right_samples = []
frame_rate = None

# Clean versions
clean_outputs = []

# Track all exported segments for ASR verification
all_exported_padded = []
all_exported_exact = []

# Per-file statistics
stats = []

for file_path in test_files:
    print(f"\nProcessing {file_path}")
    
    # Get filler intervals with STRICT detection
    intervals = detect_fillers_strict(pfl, file_path)
    
    file_basename = os.path.splitext(os.path.basename(file_path))[0]
    
    print(f"  Found {len(intervals)} filler intervals (strict mode)")
    for i, (s, e, conf, n_wins) in enumerate(intervals):
        print(f"    [{i:3d}] {s:8.2f}s - {e:8.2f}s  dur={e-s:.2f}s  conf={conf:.4f}  wins={n_wins}")
    
    stat = {
        'file': os.path.basename(file_path),
        'intervals': len(intervals),
        'total_filler_duration': sum(e - s for s, e, _, _ in intervals),
        'avg_confidence': sum(c for _, _, c, _ in intervals) / len(intervals) if intervals else 0,
        'details': [(s, e, c, n) for s, e, c, n in intervals]
    }
    stats.append(stat)
    
    # Load the audio file
    audio = AudioSegment.from_mp3(file_path)
    
    if frame_rate is None:
        frame_rate = audio.frame_rate
    
    # Create clean version with fillers muted
    clean_audio = mute_fillers(audio, intervals)
    clean_outputs.append((clean_audio, os.path.basename(file_path).replace('.mp3', '_clean.mp3')))
    
    # Export individual segments: PADDED and UNPADDED (EXACT)
    padded_dir = os.path.join('test_data', 'segments_padded')
    exact_dir = os.path.join('test_data', 'segments_exact')
    
    exported_padded = export_individual_segments(
        audio, intervals, padded_dir, file_basename, padded=True
    )
    exported_exact = export_individual_segments(
        audio, intervals, exact_dir, file_basename, padded=False
    )
    
    all_exported_padded.extend(exported_padded)
    all_exported_exact.extend(exported_exact)
    
    print(f"  Exported {len(exported_padded)} padded + {len(exported_exact)} exact segments")
    
    # Generate beep as float array
    beep_duration_ms = int(BEEP_DURATION * 1000)
    beep = Sine(BEEP_FREQUENCY).to_audio_segment(duration=beep_duration_ms).apply_gain(BEEP_VOLUME)
    beep_samples = np.array(beep.get_array_of_samples(), dtype=np.float32) / 32768.0
    
    # Extract each interval with context, add beep, and apply VAD
    for i, (start, end, confidence, n_wins) in enumerate(intervals):
        # Add context padding
        context_start = max(0, start - CONTEXT_DURATION)
        context_end = min(len(audio) / 1000.0, end + CONTEXT_DURATION)
        
        # Extract the segment with context
        start_ms = int(context_start * 1000)
        end_ms = int(context_end * 1000)
        segment_with_context = audio[start_ms:end_ms]
        
        # Apply VAD to trim silence from the context-padded segment
        samples = np.array(segment_with_context.get_array_of_samples())
        if segment_with_context.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        
        # Normalize samples to float32 in range [-1, 1] for librosa
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        elif samples.dtype == np.int32:
            samples = samples.astype(np.float32) / 2147483648.0
        elif samples.dtype == np.uint8:
            samples = (samples.astype(np.float32) - 128) / 128.0
        
        # Apply VAD using librosa
        non_silent = librosa.effects.split(samples, top_db=30)
        
        # Extract non-silent chunks as float arrays
        clean_samples = []
        for (start_sample, end_sample) in non_silent:
            chunk = samples[start_sample:end_sample]
            clean_samples.append(chunk)
        
        # Skip if no audio after VAD
        if not clean_samples:
            continue
        
        # Combine chunks
        clean_segment = np.concatenate(clean_samples)
        filler_duration = len(clean_segment)
        
        # Left channel: filler + 0.5s context
        left_segment = np.concatenate([clean_segment, np.zeros(int(CONTEXT_DURATION * frame_rate), dtype=np.float32)])
        
        # Right channel: beep at start, beep at end, silence in between
        right_segment = np.concatenate([beep_samples, np.zeros(filler_duration, dtype=np.float32), beep_samples])
        
        # Pad right to match left if needed
        if len(right_segment) < len(left_segment):
            right_segment = np.concatenate([right_segment, np.zeros(len(left_segment) - len(right_segment), dtype=np.float32)])
        elif len(right_segment) > len(left_segment):
            right_segment = right_segment[:len(left_segment)]
        
        # Add separator between segments
        if i < len(intervals) - 1:
            separator_samples = np.zeros(int(SEPARATOR_DURATION * frame_rate), dtype=np.float32)
            left_segment = np.concatenate([left_segment, separator_samples])
            right_segment = np.concatenate([right_segment, separator_samples])
            
        all_left_samples.append(left_segment)
        all_right_samples.append(right_segment)

# Print summary statistics
print(f"\n{'='*60}")
print(f"DETECTION SUMMARY")
print(f"{'='*60}")
print(f"Detection parameters:")
print(f"  Threshold:        {DETECTION_THRESHOLD}")
print(f"  Window:           {WINDOW_SEC}s")
print(f"  Stride:           {STRIDE_SEC}s")
print(f"  Merge gap:        {MERGE_GAP_SEC}s")
print(f"  Min consecutive:  {MIN_CONSECUTIVE_WINDOWS} windows")
print(f"  Duration range:   {MIN_FILLER_DURATION}s - {MAX_FILLER_DURATION}s")
print(f"")
for stat in stats:
    print(f"  {stat['file']}: {stat['intervals']} fillers, "
          f"total={stat['total_filler_duration']:.1f}s, "
          f"avg_conf={stat['avg_confidence']:.4f}")

total_fillers = sum(s['intervals'] for s in stats)
total_duration = sum(s['total_filler_duration'] for s in stats)
print(f"\n  TOTAL: {total_fillers} fillers, {total_duration:.1f}s of filler audio")
print(f"  Exported: {len(all_exported_padded)} padded + {len(all_exported_exact)} exact segments")

# Export clean versions
for clean_audio, filename in clean_outputs:
    output_path = os.path.join('test_data', filename)
    clean_audio.export(output_path, format="mp3", bitrate="192k")
    print(f"Exported clean version: {output_path} ({len(clean_audio)/1000.0:.2f}s)")

# Concatenate all segments
if all_left_samples and all_right_samples:
    left_channel = np.concatenate(all_left_samples)
    right_channel = np.concatenate(all_right_samples)
    
    # Ensure same length
    min_len = min(len(left_channel), len(right_channel))
    left_channel = left_channel[:min_len]
    right_channel = right_channel[:min_len]
    
    # Create stereo by interleaving
    stereo = np.empty((min_len * 2,), dtype=np.float32)
    stereo[0::2] = left_channel
    stereo[1::2] = right_channel
    
    # Convert to int16 for pydub
    stereo_int16 = np.clip(stereo * 32768, -32768, 32767).astype(np.int16)
    
    # Create AudioSegment from raw data
    stereo_audio = AudioSegment(
        stereo_int16.tobytes(),
        frame_rate=frame_rate,
        sample_width=2,
        channels=2
    )
    
    # Export the stereo filler segments to MP3
    output_path = 'test_data/fillers_stereo_ZO241.mp3'
    stereo_audio.export(output_path, format="mp3", bitrate="192k")
    print(f"\nExported stereo fillers to {output_path}")
    print(f"Total duration: {len(stereo_audio) / 1000.0:.2f} seconds")
    print(f"Left channel: Source audio with context")
    print(f"Right channel: Beep at start and end of filler, silence in between")
else:
    print("No audio segments to export")

# Save stats as JSON for later analysis
stats_path = 'test_data/detection_stats.json'
os.makedirs('test_data', exist_ok=True)
with open(stats_path, 'w', encoding='utf-8') as f:
    json.dump({
        'parameters': {
            'threshold': DETECTION_THRESHOLD,
            'window_sec': WINDOW_SEC,
            'stride_sec': STRIDE_SEC,
            'merge_gap_sec': MERGE_GAP_SEC,
            'min_consecutive_windows': MIN_CONSECUTIVE_WINDOWS,
            'min_filler_duration': MIN_FILLER_DURATION,
            'max_filler_duration': MAX_FILLER_DURATION,
        },
        'per_file': stats,
        'total_fillers': total_fillers,
        'total_filler_duration': total_duration,
    }, f, indent=2, default=str)
print(f"Stats saved to {stats_path}")

# ============================================
# ASR VERIFICATION (Optional)
# ============================================
# Set to True to run ASR on the EXACT (unpadded) segments to check for real words
RUN_ASR_VERIFICATION = False

if RUN_ASR_VERIFICATION:
    print("\n\nRunning ASR verification on EXACT (unpadded) segments...")
    asr_results = run_asr_verification(all_exported_exact)
else:
    print("\n\nASR verification skipped (RUN_ASR_VERIFICATION = False)")
    asr_results = []

# Save ASR results
if asr_results:
    asr_path = 'test_data/asr_verification.json'
    with open(asr_path, 'w', encoding='utf-8') as f:
        json.dump(asr_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"ASR results saved to {asr_path}")