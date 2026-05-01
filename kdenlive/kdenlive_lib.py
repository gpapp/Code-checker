import xml.etree.ElementTree as ET
import os
import uuid
import logging

logger = logging.getLogger(__name__)

class KdenliveProject:
    def __init__(self, template_path):
        self.tree = ET.parse(template_path)
        self.root = self.tree.getroot()
        
        # Read FPS from profile element
        profile = self.root.find(".//profile")
        if profile is not None:
            fps_num = int(profile.get("frame_rate_num", 25))
            fps_den = int(profile.get("frame_rate_den", 1))
            self.fps = fps_num / fps_den if fps_den > 0 else 25.0
        else:
            self.fps = 25.0
        
        # Discover main_bin playlist
        self.main_bin = self.root.find(".//playlist[@id='main_bin']")
        
        # Discover the sequence tractor ID from main_bin's activetimeline property
        active_timeline_prop = self.main_bin.find(".//property[@name='kdenlive:docproperties.activetimeline']")
        if active_timeline_prop is not None:
            self.seq_tractor_id = active_timeline_prop.text
        else:
            raise ValueError("Template is missing kdenlive:docproperties.activetimeline")
            
        self.seq_tractor = self.root.find(".//tractor[@id='{}']".format(self.seq_tractor_id))
        
        # Mapping for standard 4 tracks in the empty template: 
        self.tracks = {
            "A1": "playlist0",
            "A2": "playlist2",
            "V1": "playlist4",
            "V2": "playlist6"
        }
        
        self.chain_counter = 100
        self.filter_counter = 100
        
        # Clear out any existing user entries from main_bin
        self._reset_timelines()
        
    def _reset_timelines(self):
        for t_name, pl_id in self.tracks.items():
            pl = self.root.find(".//playlist[@id='{}']".format(pl_id))
            if pl is not None:
                for child in list(pl):
                    if child.tag in ["blank", "entry"]:
                        pl.remove(child)
    
    def _get_next_chain_id(self):
        self.chain_counter += 1
        return "chain{}".format(self.chain_counter)
        
    def _get_next_filter_id(self):
        self.filter_counter += 1
        return "filter{}".format(self.filter_counter)
    
    def _frames_to_tc(self, frames, fps=None):
        if fps is None:
            fps = self.fps
        seconds = frames / fps
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        f = int(frames % fps)
        return "{:02d}:{:02d}:{:02d}:{:02d}".format(h, m, s, f)
    
    def _tc_to_frames(self, tc):
        """Convert HH:MM:SS:FF timecode to frame count."""
        parts = tc.split(':')
        if len(parts) != 4:
            raise ValueError("Invalid timecode format: {}".format(tc))
        h, m, s, f = map(int, parts)
        total_seconds = h * 3600 + m * 60 + s
        return int(total_seconds * self.fps) + f
    
    def _get_playlist_duration_frames(self, playlist_id):
        """Calculate current duration of a playlist in frames."""
        pl = self.root.find(".//playlist[@id='{}']".format(playlist_id))
        if pl is None:
            return 0
        
        total_frames = 0
        for child in pl:
            if child.tag == "blank":
                length_str = child.get("length", "00:00:00:00")
                total_frames += self._tc_to_frames(length_str)
            elif child.tag == "entry":
                in_tc = child.get("in", "00:00:00:00")
                out_tc = child.get("out", "00:00:00:00")
                total_frames += self._tc_to_frames(out_tc) - self._tc_to_frames(in_tc) + 1
        return total_frames

    def addFileToBin(self, filepath, duration_frames=1000, clip_type="1"):
        chain_id = self._get_next_chain_id()
        tc = self._frames_to_tc(duration_frames - 1) if duration_frames > 0 else "00:00:00:00"
        
        # Create chain producer
        insert_idx = 0
        for i, child in enumerate(self.root):
            if child.tag == "tractor":
                insert_idx = i
                break
                
        chain = ET.Element("chain", id=chain_id, out=tc)
        ET.SubElement(chain, "property", name="resource").text = filepath
        ET.SubElement(chain, "property", name="mlt_service").text = "avformat-novalidate"
        ET.SubElement(chain, "property", name="kdenlive:clip_type").text = str(clip_type) 
        ET.SubElement(chain, "property", name="kdenlive:id").text = str(self.chain_counter)
        
        self.root.insert(insert_idx, chain)
        
        # Add to main_bin
        entry = ET.SubElement(self.main_bin, "entry", producer=chain_id, **{"in": "00:00:00:00", "out": tc})
        ET.SubElement(entry, "property", name="kdenlive:id").text = str(self.chain_counter)
        
        return chain_id

    def addFilterToChain(self, chain_id, service_name, properties):
        chain = self.root.find(".//chain[@id='{}']".format(chain_id))
        if chain is None:
            raise ValueError("Chain {} not found".format(chain_id))
            
        filt = ET.SubElement(chain, "filter", id=self._get_next_filter_id())
        ET.SubElement(filt, "property", name="mlt_service").text = service_name
        ET.SubElement(filt, "property", name="kdenlive_id").text = service_name
        for k, v in properties.items():
            ET.SubElement(filt, "property", name=k).text = str(v)
    
    def addFilterToTrack(self, track_name, service_name, properties):
        if track_name not in self.tracks:
            raise ValueError("Track {} not found.".format(track_name))
            
        pl_id = self.tracks[track_name]
        # Find the tractor that has a track with producer=pl_id
        tractor = None
        for tr in self.root.findall(".//tractor"):
            if tr.find("track[@producer='{}']".format(pl_id)) is not None:
                tractor = tr
                break
                
        if tractor is None:
            raise ValueError("Tractor for track {} not found".format(track_name))
            
        filt = ET.SubElement(tractor, "filter", id=self._get_next_filter_id())
        ET.SubElement(filt, "property", name="mlt_service").text = service_name
        ET.SubElement(filt, "property", name="kdenlive_id").text = service_name
        for k, v in properties.items():
            ET.SubElement(filt, "property", name=k).text = str(v)
    
    def addClipToTrack(self, track_name, chain_id, in_frame, out_frame, timeline_start_frame, extra_properties=None):
        if track_name not in self.tracks:
            raise ValueError("Track {} not found. Available: {}".format(track_name, list(self.tracks.keys())))
            
        pl_id = self.tracks[track_name]
        pl = self.root.find(".//playlist[@id='{}']".format(pl_id))
        
        tc_in = self._frames_to_tc(in_frame)
        tc_out = self._frames_to_tc(out_frame - 1) if out_frame > in_frame else tc_in
        
        # Calculate gap and add blank if needed
        current_duration = self._get_playlist_duration_frames(pl_id)
        gap_frames = timeline_start_frame - current_duration
        
        if gap_frames > 0:
            self.addBlankToTrack(track_name, gap_frames)
        elif gap_frames < 0:
            logger.warning("Overlapping clips detected: timeline_start={}, current_duration={}".format(timeline_start_frame, current_duration))
            
        entry = ET.SubElement(pl, "entry", producer=chain_id, **{"in": tc_in, "out": tc_out})
        ET.SubElement(entry, "property", name="kdenlive:id").text = chain_id.replace("chain", "")
        
        if extra_properties:
            for k, v in extra_properties.items():
                ET.SubElement(entry, "property", name=k).text = str(v)
        return entry

    def addFilterToEntry(self, entry, service_name, properties):
        filt = ET.SubElement(entry, "filter", id=self._get_next_filter_id())
        ET.SubElement(filt, "property", name="mlt_service").text = service_name
        ET.SubElement(filt, "property", name="kdenlive_id").text = service_name
        for k, v in properties.items():
            ET.SubElement(filt, "property", name=k).text = str(v)
        return filt

    def addBlankToTrack(self, track_name, blank_frames):
        """Add a blank element to a track's playlist."""
        if track_name not in self.tracks:
            raise ValueError("Track {} not found. Available: {}".format(track_name, list(self.tracks.keys())))
            
        pl_id = self.tracks[track_name]
        pl = self.root.find(".//playlist[@id='{}']".format(pl_id))
        
        blank_tc = self._frames_to_tc(blank_frames)
        ET.SubElement(pl, "blank", length=blank_tc)

    def addTimeline(self):
        """No-op for template-based approach - timeline already exists in template."""
        pass

    def save(self, output_path):
        # Fix formatting and save
        xmlstr = ET.tostring(self.root, encoding='utf-8', xml_declaration=True)
        with open(output_path, "wb") as f:
            f.write(xmlstr)
