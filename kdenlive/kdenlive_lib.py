import xml.etree.ElementTree as ET
import os
import uuid

class KdenliveProject:
    def __init__(self, template_path):
        self.tree = ET.parse(template_path)
        self.root = self.tree.getroot()
        
        # Discover main_bin playlist
        self.main_bin = self.root.find(".//playlist[@id='main_bin']")
        
        # Discover the sequence tractor ID from main_bin's activetimeline property
        active_timeline_prop = self.main_bin.find(".//property[@name='kdenlive:docproperties.activetimeline']")
        if active_timeline_prop is not None:
            self.seq_tractor_id = active_timeline_prop.text
        else:
            raise ValueError("Template is missing kdenlive:docproperties.activetimeline")
            
        self.seq_tractor = self.root.find(f".//tractor[@id='{self.seq_tractor_id}']")
        
        # Mapping for standard 4 tracks in the empty template: 
        # A1 (playlist0), A2 (playlist2), V1 (playlist4), V2 (playlist6)
        # We need to map which playlist is actually the user's timeline track
        # tractor0 = A1 (playlist0 + playlist1)
        # tractor1 = A2 (playlist2 + playlist3)
        # tractor2 = V1 (playlist4 + playlist5)
        # tractor3 = V2 (playlist6 + playlist7)
        self.tracks = {
            "A1": "playlist0",
            "A2": "playlist2",
            "V1": "playlist4",
            "V2": "playlist6"
        }
        
        self.chain_counter = 100
        self.filter_counter = 100
        
        # Clear out any existing user entries from main_bin (leave only sequence entry and properties)
        # Also clean out the track playlists
        self._reset_timelines()
        
    def _reset_timelines(self):
        for t_name, pl_id in self.tracks.items():
            pl = self.root.find(f".//playlist[@id='{pl_id}']")
            if pl is not None:
                # Remove all blanks and entries
                for child in list(pl):
                    if child.tag in ["blank", "entry"]:
                        pl.remove(child)

    def _get_next_chain_id(self):
        self.chain_counter += 1
        return f"chain{self.chain_counter}"
        
    def _get_next_filter_id(self):
        self.filter_counter += 1
        return f"filter{self.filter_counter}"

    def _frames_to_tc(self, frames: int, fps: float = 25.0) -> str:
        seconds = frames / fps
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        f = int(frames % fps)
        return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"

    def addFileToBin(self, filepath, duration_frames=1000, clip_type="1"):
        chain_id = self._get_next_chain_id()
        tc = self._frames_to_tc(duration_frames - 1) if duration_frames > 0 else "00:00:00:00"
        
        # Create chain producer
        # Insert before the first producer or chain to ensure correct parsing order
        # Actually, appending before the first tractor is safer
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
        chain = self.root.find(f".//chain[@id='{chain_id}']")
        if chain is None:
            raise ValueError(f"Chain {chain_id} not found")
            
        filt = ET.SubElement(chain, "filter", id=self._get_next_filter_id())
        ET.SubElement(filt, "property", name="mlt_service").text = service_name
        ET.SubElement(filt, "property", name="kdenlive_id").text = service_name
        for k, v in properties.items():
            ET.SubElement(filt, "property", name=k).text = str(v)

    def addFilterToTrack(self, track_name, service_name, properties):
        if track_name not in self.tracks:
            raise ValueError(f"Track {track_name} not found.")
            
        pl_id = self.tracks[track_name]
        # Find the tractor that has a track with producer=pl_id
        tractor = None
        for tr in self.root.findall(".//tractor"):
            if tr.find(f"track[@producer='{pl_id}']") is not None:
                tractor = tr
                break
                
        if tractor is None:
            raise ValueError(f"Tractor for track {track_name} not found")
            
        filt = ET.SubElement(tractor, "filter", id=self._get_next_filter_id())
        ET.SubElement(filt, "property", name="mlt_service").text = service_name
        ET.SubElement(filt, "property", name="kdenlive_id").text = service_name
        for k, v in properties.items():
            ET.SubElement(filt, "property", name=k).text = str(v)

    def addTimeline(self):
        # The template already has a timeline (sequence tractor + project tractor). 
        # Here we could just update the max duration or reset properties if needed.
        pass

    def addClipToTrack(self, track_name, chain_id, in_frame, out_frame, timeline_start_frame, extra_properties=None):
        if track_name not in self.tracks:
            raise ValueError(f"Track {track_name} not found. Available: {list(self.tracks.keys())}")
            
        pl_id = self.tracks[track_name]
        pl = self.root.find(f".//playlist[@id='{pl_id}']")
        
        tc_in = self._frames_to_tc(in_frame)
        tc_out = self._frames_to_tc(out_frame - 1) if out_frame > in_frame else tc_in
        
        if timeline_start_frame > 0:
            blank_tc = self._frames_to_tc(timeline_start_frame)
            ET.SubElement(pl, "blank", length=blank_tc)
            
        entry = ET.SubElement(pl, "entry", producer=chain_id, **{"in": tc_in, "out": tc_out})
        ET.SubElement(entry, "property", name="kdenlive:id").text = chain_id.replace("chain", "")
        
        if extra_properties:
            for k, v in extra_properties.items():
                ET.SubElement(entry, "property", name=k).text = str(v)

    def save(self, output_path):
        # Fix formatting and save
        xmlstr = ET.tostring(self.root, encoding='utf-8', xml_declaration=True)
        with open(output_path, "wb") as f:
            f.write(xmlstr)
