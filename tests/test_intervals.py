import pytest
from interval_utils import merge_intervals, calculate_keep_segments, adjust_timestamps, compress_global_silence

def test_merge_intervals():
    intervals = [(0, 5), (4, 8), (10, 12)]
    merged = merge_intervals(intervals)
    assert merged == [(0, 8), (10, 12)]

def test_calculate_keep_segments():
    cuts = [(5, 10)]
    total_dur = 15
    keep = calculate_keep_segments(cuts, total_dur)
    assert keep == [(0.0, 5.0), (10.0, 15.0)]

def test_compress_global_silence():
    # 5s silence should keep ~0.6s and cut ~4.4s
    silence = [(2, 7)]
    cuts = compress_global_silence(silence)
    assert len(cuts) == 1
    start, end = cuts[0]
    assert start > 2.0
    assert end == 7.0
    # Expected keep_dur = 0.2 + (1.0 - 0.2) / 2 = 0.6
    assert abs((start - 2.0) - 0.6) < 0.01

def test_adjust_timestamps():
    keep = [(0, 5), (10, 15)] # Cut 5-10
    segments = [(2, 12)] # Spans across the cut
    adjusted = adjust_timestamps(segments, keep)
    # 2 is in first segment -> new_start = 2
    # 12 is in second segment (at offset 2) -> new_end = 5 + 2 = 7
    assert adjusted == [(2.0, 7.0)]
