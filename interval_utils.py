from typing import List, Tuple

def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
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

def calculate_keep_segments(cut_segments: List[Tuple[float, float]], total_duration: float) -> List[Tuple[float, float]]:
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

def adjust_timestamps(segments: List[Tuple[float, float]], keep_segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
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
