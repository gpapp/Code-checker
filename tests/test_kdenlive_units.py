import pytest
from kdenlive.kdenlive_lib import KdenliveProject
import os

def test_time_conversions_25fps():
    template = "kdenlive/empty.kdenlive"
    proj = KdenliveProject(template)
    proj.fps = 25.0

    # Seconds to frames (round to nearest)
    assert proj.seconds_to_frames(1.0) == 25
    assert proj.seconds_to_frames(1.02) == 26  # 25.5 -> 26
    assert proj.seconds_to_frames(1.01) == 25  # 25.25 -> 25

    # Frames to TC
    assert proj.frames_to_tc(0) == "00:00:00:00"
    assert proj.frames_to_tc(24) == "00:00:00:24"
    assert proj.frames_to_tc(25) == "00:00:01:00"
    assert proj.frames_to_tc(3600*25) == "01:00:00:00"

    # TC to frames
    assert proj.tc_to_frames("00:00:01:00") == 25
    assert proj.tc_to_frames("00:00:00:24") == 24
    assert proj.tc_to_frames("01:00:00:00") == 3600*25
    assert proj.tc_to_frames("00:00:01.0") == 25 # Clock format

def test_time_conversions_60fps():
    template = "kdenlive/empty.kdenlive"
    proj = KdenliveProject(template)
    proj.fps = 60.0

    assert proj.seconds_to_frames(1.0) == 60
    assert proj.frames_to_tc(59) == "00:00:00:59"
    assert proj.frames_to_tc(60) == "00:00:01:00"
    assert proj.tc_to_frames("00:00:01:00") == 60

def test_id_discovery():
    # Use the sample file which has many existing IDs
    sample = "kdenlive/samplekdenlive.kdenlive"
    proj = KdenliveProject(sample)

    # It should have discovered IDs higher than 100
    # Sample has filter15, chain5, transition3
    # Our default was 100, discovery should find nothing higher unless we change defaults
    # Wait, discovery sets counter to MAX(current, existing).
    # If existing are 1-15, and current is 100, it stays 100.
    assert proj.filter_counter >= 100

    fid = proj._get_next_filter_id()
    assert int(fid.replace("filter", "")) > 100

def test_add_transition():
    template = "kdenlive/empty.kdenlive"
    proj = KdenliveProject(template)

    trans = proj.addTransition(1, 2, "mix", {"sum": "1"})
    assert trans.tag == "transition"
    assert trans.find("property[@name='a_track']").text == "1"
    assert trans.find("property[@name='b_track']").text == "2"
    assert trans.find("property[@name='mlt_service']").text == "mix"
    assert trans.find("property[@name='sum']").text == "1"
