import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict
import json

class MLTInspector:
    """Utility to inspect and verify MLT XML files for Kdenlive compatibility."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        self.tree = ET.parse(file_path)
        self.root = self.tree.getroot()
        self.id_map = {child.get("id"): child for child in self.root if child.get("id")}

    def print_bin_summary(self):
        """Prints a summary of the project bin (Producers/Chains)."""
        print(f"\n--- Bin Summary: {self.file_path.name} ---")
        
        # Find producers and chains
        items = self.root.findall("producer") + self.root.findall("chain")
        print(f"Total Assets in Bin: {len(items)}")
        
        for item in items:
            item_id = item.get("id")
            resource = ""
            service = ""
            for prop in item.findall("property"):
                if prop.get("name") == "resource":
                    resource = prop.text
                if prop.get("name") == "mlt_service":
                    service = prop.text
            
            type_label = "Chain" if item.tag == "chain" else "Producer"
            print(f"  [{type_label}] ID: {item_id:15} Service: {service:12} Resource: {resource}")

    def _resolve_resource(self, producer_id: str) -> str:
        """Finds the physical resource path for a given producer ID."""
        elem = self.id_map.get(producer_id)
        if elem is None:
            return ""
        
        # If it's a playlist or tractor, look deeper
        if elem.tag == "playlist":
            entry = elem.find("entry")
            if entry is not None:
                return self._resolve_resource(entry.get("producer"))
        elif elem.tag == "tractor":
            track = elem.find("track")
            if track is not None:
                return self._resolve_resource(track.get("producer"))
        
        # It's a producer or chain, get resource property
        for prop in elem.findall("property"):
            if prop.get("name") == "resource":
                res = prop.text or ""
                # Normalize paths for comparison
                return res.replace("\\", "/").lower()
        
        return ""

    def _resolve_bin_item(self, producer_id: str) -> Tuple[str, str]:
        """Resolves a timeline producer to its bin item ID and resource."""
        elem = self.id_map.get(producer_id)
        if elem is None: return "", ""
        
        if elem.tag == "playlist":
            entry = elem.find("entry")
            if entry is not None: return self._resolve_bin_item(entry.get("producer"))
        elif elem.tag == "tractor":
            track = elem.find("track")
            if track is not None: return self._resolve_bin_item(track.get("producer"))
            
        # It's a producer/chain
        bin_id = elem.findtext("property[@name='kdenlive:id']", default="")
        res = self._resolve_resource(producer_id)
        return bin_id, res

    def print_timeline_overview(self):
        """Prints a high-level summary of the timeline tracks without individual clip details."""
        print(f"\n--- Timeline Overview ---")
        
        tractors = self.root.findall("tractor")
        seq_tractor = None
        for t in tractors:
            for prop in t.findall("property"):
                if prop.get("name") == "kdenlive:sequenceproperties.tracks":
                    seq_tractor = t
                    break
        
        if not seq_tractor:
            print("Could not identify main sequence tractor. File might not be in Kdenlive format.")
            return

        tracks = seq_tractor.findall("track")
        print(f"Total UI Tracks: {len(tracks) - 1}") # Subtract track 0 (background)

        for i, track in enumerate(tracks):
            prod_id = track.get("producer")
            if prod_id == "producer0":
                continue # Skip black background track
            
            track_tractor = self.root.find(f".//tractor[@id='{prod_id}']")
            track_name = "Unnamed"
            track_type = "Video"
            
            if track_tractor is not None:
                track_name = track_tractor.findtext("property[@name='kdenlive:track_name']", default="Unnamed")
                track_type = "Audio" if track_tractor.findtext("property[@name='kdenlive:audio_track']") == "1" else "Video"
                
                # Find the primary playlist (the one not hidden)
                inner_tracks = track_tractor.findall("track")
                if inner_tracks:
                    playlist_id = inner_tracks[0].get("producer")
                    playlist = self.root.find(f".//playlist[@id='{playlist_id}']")
                    
                    clips = playlist.findall("entry")
                    blanks = playlist.findall("blank")
                    print(f"  Track {i}: {track_name} ({track_type}) - {len(clips)} clips, {len(blanks)} blanks")

    def print_timeline_summary(self):
        """Prints a summary of the timeline tracks and clips."""
        print(f"\n--- Timeline Summary ---")
        
        # In Kdenlive format, tracks are represented as tractors containing playlists
        tractors = self.root.findall("tractor")
        # The sequence tractor usually has kdenlive:sequenceproperties.tracks
        seq_tractor = None
        for t in tractors:
            for prop in t.findall("property"):
                if prop.get("name") == "kdenlive:sequenceproperties.tracks":
                    seq_tractor = t
                    break
        
        if not seq_tractor:
            print("Could not identify main sequence tractor. File might not be in Kdenlive format.")
            return

        # Tracks in the sequence tractor point to other tractors or playlists
        tracks = seq_tractor.findall("track")
        print(f"Number of Tracks: {len(tracks) - 1}") # Subtract track 0 (background)

        for i, track in enumerate(tracks):
            prod_id = track.get("producer")
            if prod_id == "producer0":
                continue # Skip black background track
            
            # Look for the track tractor definition
            track_tractor = self.root.find(f".//tractor[@id='{prod_id}']")
            track_name = "Unnamed"
            track_type = "Video"
            
            if track_tractor is not None: # Ensure track_tractor was found
                track_name = track_tractor.findtext("property[@name='kdenlive:track_name']", default="Unnamed")
                track_type = "Audio" if track_tractor.findtext("property[@name='kdenlive:audio_track']") == "1" else "Video"

                # Find the primary playlist (the one not hidden)
                # In Kdenlive structure, track tractor has two tracks (playlist_v and playlist_a)
                inner_tracks = track_tractor.findall("track")
                if inner_tracks:
                    playlist_id = inner_tracks[0].get("producer")
                    playlist = self.root.find(f".//playlist[@id='{playlist_id}']")
                    
                    clips = playlist.findall("entry")
                    blanks = playlist.findall("blank")
                    print(f"  Track {i}: {track_name} ({track_type}) - {len(clips)} clips, {len(blanks)} blanks")
                    
                    for clip in clips:
                        c_prod = clip.get("producer")
                        c_in = clip.get("in")
                        c_out = clip.get("out")
                        print(f"    - Clip: {c_prod:15} In: {c_in:12} Out: {c_out:12}")

    def _parse_timecode(self, tc: str, fps: float) -> int:
        """Parses HH:MM:SS:FF or HH:MM:SS.mmm or frames to integer frames."""
        if tc is None: return 0
        if ":" not in tc:
            try: return int(tc)
            except: return 0
        parts = tc.split(":")
        try:
            if len(parts) == 4: # HH:MM:SS:FF
                h, m, s, f = map(int, parts)
                return int(round((h * 3600 + m * 60 + s) * fps + f))
            elif len(parts) == 3: # HH:MM:SS.mmm
                h, m, s_ms = parts
                total_seconds = int(h) * 3600 + int(m) * 60 + float(s_ms)
                return int(round(total_seconds * fps))
        except:
            pass
        return 0

    def _get_track_timeline(self, track_tractor_id: str, fps: float) -> List[Tuple[int, int]]:
        """Returns a list of (start_frame, end_frame) on the timeline for all clips in a track."""
        track_tractor = self.id_map.get(track_tractor_id)
        if track_tractor is None: return []
        inner_tracks = track_tractor.findall("track")
        if not inner_tracks: return []
        
        # Kdenlive track tractors contain playlists. The first one usually contains the clips.
        playlist_id = inner_tracks[0].get("producer")
        playlist = self.id_map.get(playlist_id)
        if playlist is None: return []
        
        segments = []
        current_frame = 0
        for child in playlist:
            if child.tag == "blank":
                length = int(child.get("length", 0))
                current_frame += length
            elif child.tag == "entry":
                in_f = self._parse_timecode(child.get("in"), fps)
                out_f = self._parse_timecode(child.get("out"), fps)
                duration = out_f - in_f + 1
                segments.append((current_frame, current_frame + duration))
                current_frame += duration
        return segments

    def validate_clip_alignment(self) -> List[str]:
        """Verifies video alignment across tracks and audio-video cut synchronization."""
        errors = []
        profile = self.root.find("profile")
        fps = 25.0
        if profile is not None:
            try:
                num = float(profile.get("frame_rate_num", 25))
                den = float(profile.get("frame_rate_den", 1))
                fps = num / den
            except: pass

        seq_tractor = next((t for t in self.root.findall("tractor") 
                           if t.find("property[@name='kdenlive:sequenceproperties.tracks']") is not None), None)
        if not seq_tractor: return []

        video_tracks = {}
        audio_starts, audio_ends = set(), set()

        for i, track in enumerate(seq_tractor.findall("track")):
            prod_id = track.get("producer")
            if prod_id == "producer0" or not prod_id: continue
            
            track_node = self.id_map.get(prod_id)
            if track_node is None: continue
            
            is_audio = track_node.findtext("property[@name='kdenlive:audio_track']") == "1"
            segments = self._get_track_timeline(prod_id, fps)
            
            if is_audio:
                for s, e in segments:
                    audio_starts.add(s)
                    audio_ends.add(e)
            else:
                video_tracks[i] = segments

        if not video_tracks: return []

        # 1. Video-to-Video Alignment Check
        ref_idx, ref_segs = next(iter(video_tracks.items()))
        for idx, segs in video_tracks.items():
            if segs != ref_segs:
                errors.append(f"Alignment Error: Video track {idx} does not match reference track {ref_idx}")

        # 2. Audio-Video Cut Sync Check
        if audio_starts or audio_ends:
            for s, e in ref_segs:
                if s not in audio_starts:
                    errors.append(f"Sync Error: Video clip start at frame {s} has no corresponding audio clip start")
                if e not in audio_ends:
                    errors.append(f"Sync Error: Video clip end at frame {e} has no corresponding audio clip end")
        return errors

    def validate_kdenlive_rules(self) -> List[str]:
        """Verifies if the XML follows mandatory Kdenlive 23.08+ formatting rules."""
        errors = []
        
        # 1. Root attributes
        if self.root.get("LC_NUMERIC") != "en_US.UTF-8":
            errors.append("Rule Violation: Root <mlt> missing LC_NUMERIC='en_US.UTF-8'")
        if self.root.get("producer") != "main_bin":
            errors.append("Rule Violation: Root <mlt> missing producer='main_bin'")

        # Get main sequence tractor and group definitions
        seq_tractor = None
        for t in self.root.findall("tractor"):
            if t.find("property[@name='kdenlive:sequenceproperties.tracks']") is not None:
                seq_tractor = t
                break
        
        groups_data = []
        if seq_tractor is not None:
            groups_json = seq_tractor.findtext("property[@name='kdenlive:sequenceproperties.groups']")
            if groups_json:
                try:
                    groups_data = json.loads(groups_json)
                except:
                    errors.append("Rule Violation: kdenlive:sequenceproperties.groups is not valid JSON")

        # 2. Track Stack Symmetry and Pairing Rules
        if seq_tractor is not None:
            tracks = seq_tractor.findall("track")
            # Track index -> (bin_id, resource)
            audio_track_info: Dict[int, Tuple[str, str]] = {}
            video_track_info: Dict[int, Tuple[str, str]] = {}
            
            current_mode = "audio" # Expect audio tracks first
            for i, track in enumerate(tracks):
                prod_id = track.get("producer")
                if prod_id == "producer0": continue
                
                track_tractor = self.id_map.get(prod_id)
                if track_tractor is not None:
                    is_audio = track_tractor.findtext("property[@name='kdenlive:audio_track']") == "1"
                    bin_id, res = self._resolve_bin_item(prod_id)
                    
                    if is_audio:
                        if current_mode == "video":
                            errors.append(f"Rule Violation: Audio track '{prod_id}' found after video tracks. Audios must be at the bottom.")
                        audio_track_info[i] = (bin_id, res)
                    else:
                        current_mode = "video"
                        video_track_info[i] = (bin_id, res)

                    # Check naming rule from filename __suffix
                    if res:
                        filename = Path(res).stem
                        if "__" in filename:
                            expected_name = filename.split("__", 1)[1]
                            track_name = track_tractor.findtext("property[@name='kdenlive:track_name']", default="")
                            if expected_name != track_name:
                                errors.append(f"Rule Violation: Track {i} name '{track_name}' does not match expected suffix '{expected_name}' derived from resource '{res}'")
            
            audio_res_list = [info[1] for info in audio_track_info.values()]
            video_res_list = [info[1] for info in video_track_info.values()]

            # Check Symmetry: Video resources should be the reverse of Audio resources (ignoring dedicated audio tracks like Effects)
            # We only compare tracks that have a corresponding pair
            paired_audios = [r for r in audio_res_list if r in video_res_list]
            paired_videos = [r for r in video_res_list if r in audio_res_list]
            
            if paired_audios != paired_videos[::-1]:
                errors.append("Rule Violation: Timeline track order is not mirrored. Expected Video stack to be reverse of Audio stack.")

            # 5. Verify AV Linking via properties and groups
            for a_idx, (a_bin, a_res) in audio_track_info.items():
                for v_idx, (v_bin, v_res) in video_track_info.items():
                    if a_res == v_res and a_res != "":
                        # Found a pair!
                        # A. Check bin item sharing
                        if a_bin != v_bin:
                            errors.append(f"Rule Violation: Audio track {a_idx} and Video track {v_idx} share resource '{a_res}' but have different bin IDs ({a_bin} vs {v_bin})")

                        # B. Check for group linking (AVSplit)
                        linked = False
                        for group in groups_data:
                            if group.get("type") == "AVSplit":
                                children_indices = []
                                for child in group.get("children", []):
                                    data = child.get("data", "")
                                    if ":" in data:
                                        try:
                                            children_indices.append(int(data.split(":")[0]))
                                        except: pass
                                if a_idx in children_indices and v_idx in children_indices:
                                    linked = True
                                    break
                        if not linked:
                            errors.append(f"Rule Violation: Paired tracks {a_idx} (Audio) and {v_idx} (Video) for resource '{a_res}' are not linked in kdenlive:sequenceproperties.groups")

            # Check sequence property counts
            expected_tracks_count = str(len(tracks) - 1)
            actual_tracks_count = seq_tractor.findtext("property[@name='kdenlive:sequenceproperties.tracksCount']")
            if actual_tracks_count != expected_tracks_count:
                errors.append(f"Metadata Mismatch: tracksCount is {actual_tracks_count}, expected {expected_tracks_count}")

            expected_video_tracks = str(len(video_res_list) + 1)
            actual_video_tracks = seq_tractor.findtext("property[@name='kdenlive:sequenceproperties.tracks']")
            if actual_video_tracks != expected_video_tracks:
                 # Kdenlive's 'tracks' property often counts video layers + 1
                errors.append(f"Metadata Mismatch: sequenceproperties.tracks is {actual_video_tracks}, expected {expected_video_tracks} (Videos + 1)")

        # 2. Main Bin Structure
        main_bin = self.root.find(".//playlist[@id='main_bin']")
        if main_bin is None:
            errors.append("Rule Violation: Missing <playlist id='main_bin'>")
        else:
            # CRITICAL: Properties must come before entries in main_bin
            found_entry = False
            for child in main_bin:
                if child.tag == "entry":
                    found_entry = True
                if child.tag == "property" and found_entry:
                    errors.append("Rule Violation: <property> found after <entry> in 'main_bin' playlist")
                    break

        # 4. Transition Rules (Star Model)
        if seq_tractor is not None:
            transitions = seq_tractor.findall("transition")
            for i, trans in enumerate(transitions):
                a_track = trans.findtext("property[@name='a_track']")
                b_track = trans.findtext("property[@name='b_track']")
                if a_track != "0":
                    errors.append(f"Rule Violation: Transition {i} has a_track={a_track}. Kdenlive requires blending against track 0.")
                if b_track != str(i + 1):
                    errors.append(f"Rule Violation: Transition {i} has b_track={b_track}, expected {i+1} to follow sequence order.")

        # 3. Project Tractor
        project_tractor = None
        for tractor in self.root.findall("tractor"):
            for prop in tractor.findall("property"):
                if prop.get("name") == "kdenlive:projectTractor" and prop.text == "1":
                    project_tractor = tractor
                    break
        
        if project_tractor is None:
            errors.append("Rule Violation: Missing tractor with kdenlive:projectTractor='1'")

        # 4. Track Blending
        seq_tractor = None
        for t in self.root.findall("tractor"):
            for prop in t.findall("property"):
                if prop.get("name") == "kdenlive:sequenceproperties.tracks":
                    seq_tractor = t
                    break
        
        if seq_tractor is not None:
            transitions = seq_tractor.findall("transition")
            # Every track (except 0) should have a transition blending it to track 0
            track_count = len(seq_tractor.findall("track"))
            if len(transitions) < track_count - 1:
                errors.append(f"Format Warning: Expected at least {track_count-1} transitions for track blending, found {len(transitions)}")

        return errors

    def run_full_report(self):
        """Runs all checks and prints a formatted report."""
        print(f"{'='*60}")
        print(f"MLT INSPECTION REPORT: {self.file_path.name}")
        print(f"{'='*60}")
        self.print_bin_summary()
        self.print_timeline_overview() # High-level summary
        self.print_timeline_summary()
        print("\n--- Validation Checks ---")
        errors = self.validate_kdenlive_rules()
        errors += self.validate_clip_alignment()
        if not errors:
            print("  [OK] All Kdenlive compatibility and alignment rules passed.")
        else:
            for err in errors:
                print(f"  [ERROR] {err}")
        print(f"{'='*60}\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Inspect MLT XML files for Kdenlive compatibility.")
    parser.add_argument("files", nargs="+", help="One or more MLT XML files to inspect.")
    args = parser.parse_args()

    for file in args.files:
        try:
            inspector = MLTInspector(file)
            inspector.run_full_report()
        except Exception as e:
            print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    main()