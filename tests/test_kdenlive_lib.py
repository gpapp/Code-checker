import os
import tempfile
from kdenlive.kdenlive_lib import KdenliveProject

def test_empty_template_loads():
    template = "kdenlive/empty.kdenlive"
    assert os.path.exists(template), "Template not found"
    
    proj = KdenliveProject(template)
    assert proj.seq_tractor_id is not None
    assert proj.main_bin is not None
    
def test_add_file_to_bin():
    template = "kdenlive/empty.kdenlive"
    proj = KdenliveProject(template)
    
    chain_id = proj.addFileToBin("dummy_video.mp4", duration_frames=500)
    
    # Check if chain is in root
    chain = proj.root.find(f".//chain[@id='{chain_id}']")
    assert chain is not None
    
    # Check if entry is in main_bin
    entry = proj.main_bin.find(f".//entry[@producer='{chain_id}']")
    assert entry is not None

def test_add_filter():
    template = "kdenlive/empty.kdenlive"
    proj = KdenliveProject(template)
    
    chain_id = proj.addFileToBin("dummy_video.mp4", duration_frames=500)
    proj.addFilterToChain(chain_id, "volume", {"level": "0.5"})
    
    chain = proj.root.find(f".//chain[@id='{chain_id}']")
    filt = chain.find(".//filter")
    assert filt is not None
    assert filt.find(".//property[@name='mlt_service']").text == "volume"
    assert filt.find(".//property[@name='level']").text == "0.5"

def test_add_clip_to_track():
    template = "kdenlive/empty.kdenlive"
    proj = KdenliveProject(template)
    
    chain_id = proj.addFileToBin("dummy_video.mp4", duration_frames=500)
    
    proj.addClipToTrack("V1", chain_id, in_frame=0, out_frame=100, timeline_start_frame=0)
    
    pl_id = proj.tracks["V1"]
    pl = proj.root.find(f".//playlist[@id='{pl_id}']")
    entry = pl.find(f".//entry[@producer='{chain_id}']")
    assert entry is not None

def test_case1_empty():
    template = "kdenlive/empty.kdenlive"
    proj = KdenliveProject(template)
    proj.save("tests/outputs/case1_empty.kdenlive")

def test_case2_add_files():
    template = "kdenlive/empty.kdenlive"
    proj = KdenliveProject(template)
    proj.addFileToBin("dummy_video.mp4", duration_frames=5000)
    proj.save("tests/outputs/case2_add_files.kdenlive")

def test_case3_add_filters():
    template = "kdenlive/empty.kdenlive"
    proj = KdenliveProject(template)
    chain_id = proj.addFileToBin("dummy_video.mp4", duration_frames=5000)
    proj.addFilterToChain(chain_id, "volume", {"level": "0.5"})
    proj.save("tests/outputs/case3_add_filters.kdenlive")

def test_case4_add_timeline():
    # In template approach, the timeline is always present. We just save it.    
    template = "kdenlive/empty.kdenlive"
    proj = KdenliveProject(template)
    chain_id = proj.addFileToBin("dummy_video.mp4", duration_frames=5000)
    proj.addFilterToChain(chain_id, "volume", {"level": "0.5"})
    proj.addTimeline()
    proj.save("tests/outputs/case4_add_timeline.kdenlive")

def test_case5_add_tracks_whole_files():
    template = "kdenlive/empty.kdenlive"
    proj = KdenliveProject(template)
    chain_id = proj.addFileToBin("dummy_video.mp4", duration_frames=5000)
    proj.addClipToTrack("V1", chain_id, in_frame=0, out_frame=5000, timeline_start_frame=0)
    proj.save("tests/outputs/case5_add_tracks_whole.kdenlive")

def test_case6_add_clips_individually():
    template = "kdenlive/empty.kdenlive"
    proj = KdenliveProject(template)
    chain_id = proj.addFileToBin("dummy_video.mp4", duration_frames=5000)
    # Add two split clips
    proj.addClipToTrack("V1", chain_id, in_frame=0, out_frame=1000, timeline_start_frame=0)
    # Add second clip starting at 1500 (meaning there is a 500 frame gap on the timeline relative to end of first clip)
    proj.addClipToTrack("V1", chain_id, in_frame=2000, out_frame=3000, timeline_start_frame=1500)
    proj.save("tests/outputs/case6_add_clips.kdenlive")
