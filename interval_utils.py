# Interval utilities for merging and adjusting timestamps

def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merges overlapping or adjacent intervals."""
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for curr_start, curr_end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if curr_start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append((curr_start, curr_end))
    return merged

def compress_global_silence(global_silence: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Compresses global silence intervals (where ALL tracks are quiet).
    
    Rules (applied in order):
    1. If silence > 1s, truncate to 1s
    2. If silence > 0.2s (after truncation), compress to half
    
    Returns CUT segments representing the EXCESS silence to remove.
    The remaining (uncut) portion of each silence is kept in the timeline.
    
    Example:
      5.0s silence -> keep 0.6s, cut 4.4s  (5->1s->0.6s, cut [start+0.6, end])
      0.8s silence -> keep 0.5s, cut 0.3s  (0.8->0.8->0.5s, cut [start+0.5, end])
      0.15s silence -> keep 0.15s, cut 0s  (unchanged)
    """
    cuts = []
    for s_start, s_end in sorted(global_silence):
        orig_dur = s_end - s_start
        keep_dur = orig_dur
        
        # 1. Truncate > 1s to 1s
        if keep_dur > 1.0:
            keep_dur = 1.0
        # 2. Compress > 0.2s to half
        if keep_dur > 0.2:
            keep_dur = 0.2 + (keep_dur - 0.2) / 2.0
        
        # The portion we CUT is everything after the kept portion
        if keep_dur < orig_dur - 0.01:
            cuts.append((s_start + keep_dur, s_end))
    
    return cuts

def calculate_keep_segments(cut_segments: list[tuple[float, float]], total_duration: float) -> list[tuple[float, float]]:
    """Inverts cut segments to find segments to keep."""
    keep = []
    last_end = 0.0
    for start, end in sorted(cut_segments):
        if start > last_end:
            keep.append((last_end, start))
        last_end = max(last_end, end)
    if last_end < total_duration:
        keep.append((last_end, total_duration))
    return keep

def adjust_timestamps(segments: list[tuple[float, float]], keep_segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Adjusts timestamps from original timeline to the cut timeline."""
    adjusted = []
    for start, end in segments:
        new_start = None
        new_end = None
        current_new_time = 0.0
        for ks, ke in sorted(keep_segments):
            duration = ke - ks
            if new_start is None:
                if ks <= start < ke:
                    new_start = current_new_time + (start - ks)
                elif start < ks:
                    new_start = current_new_time
            if new_end is None:
                if ks <= end < ke:
                    new_end = current_new_time + (end - ks)
                elif end < ks and new_start is not None:
                    new_end = current_new_time
            current_new_time += duration
        if new_start is not None:
            if new_end is None: new_end = current_new_time
            adjusted.append((new_start, new_end))
    return adjusted
