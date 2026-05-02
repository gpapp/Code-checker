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
        
        # Mapping for tracks: name -> playlist_id
        self.tracks = {}
        self.track_names = [] # ordered list of track names
        
        self.chain_counter = 100
        self.filter_counter = 100
        self.transition_counter = 100
        self.playlist_counter = 0
        self.tractor_counter = 0

        # Discover current counters to avoid collisions
        self._discover_counters()

        # Remove existing tracks from sequence and bin to start fresh
        self._clear_existing_tracks()

    def _discover_counters(self):
        """Scan XML for existing IDs to initialize counters correctly."""
        for elem in self.root.iter():
            eid = elem.get("id", "")
            if not eid: continue

            try:
                if eid.startswith("chain"):
                    self.chain_counter = max(self.chain_counter, int(eid.replace("chain", "")))
                elif eid.startswith("filter"):
                    self.filter_counter = max(self.filter_counter, int(eid.replace("filter", "")))
                elif eid.startswith("transition"):
                    self.transition_counter = max(self.transition_counter, int(eid.replace("transition", "")))
                elif eid.startswith("playlist"):
                    self.playlist_counter = max(self.playlist_counter, int(eid.replace("playlist", "")) + 1)
                elif eid.startswith("tractor"):
                    try:
                        self.tractor_counter = max(self.tractor_counter, int(eid.replace("tractor", "")) + 1)
                    except ValueError:
                        pass # Ignore UUID-style tractors
            except ValueError:
                pass

    def _clear_existing_tracks(self):
        """Removes all tracks except the background black track from the sequence tractor."""
        if self.seq_tractor is None: return

        # Keep track 0 (black track)
        tracks = self.seq_tractor.findall("track")
        for i, t in enumerate(tracks):
            if i > 0:
                # Remove playlist/tractor referenced by this track
                prod_id = t.get("producer")
                if prod_id:
                    # Find and remove the producer (playlist or tractor)
                    prod = self.root.find(".//*[@id='{}']".format(prod_id))
                    if prod is not None:
                        # If it's a tractor, it might have internal tracks/playlists
                        if prod.tag == "tractor":
                            for internal_t in prod.findall("track"):
                                internal_p_id = internal_t.get("producer")
                                internal_p = self.root.find(".//*[@id='{}']".format(internal_p_id))
                                if internal_p is not None:
                                    try: self.root.remove(internal_p)
                                    except ValueError: pass
                        try: self.root.remove(prod)
                        except ValueError: pass
                self.seq_tractor.remove(t)

        # Remove all transitions from sequence
        for trans in self.seq_tractor.findall("transition"):
            self.seq_tractor.remove(trans)

        # Clear main_bin entries
        if self.main_bin is not None:
            for entry in list(self.main_bin):
                if entry.tag == "entry":
                    self.main_bin.remove(entry)

        # Reset trackers
        self.tracks = {}
        self.track_names = []
    
    def _get_next_chain_id(self):
        self.chain_counter += 1
        return "chain{}".format(self.chain_counter)
        
    def _get_next_filter_id(self):
        self.filter_counter += 1
        return "filter{}".format(self.filter_counter)

    def _get_next_transition_id(self):
        self.transition_counter += 1
        return "transition{}".format(self.transition_counter)

    def _get_next_playlist_id(self):
        pid = "playlist{}".format(self.playlist_counter)
        self.playlist_counter += 1
        return pid

    def _get_next_tractor_id(self):
        tid = "tractor{}".format(self.tractor_counter)
        self.tractor_counter += 1
        return tid
    
    def frames_to_tc(self, frames, fps=None):
        """Convert frame count to HH:MM:SS:FF timecode."""
        if fps is None:
            fps = self.fps

        # MLT uses floor for frames in timecode
        total_seconds = int(frames // fps)
        remaining_frames = int(frames % fps)

        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60

        return "{:02d}:{:02d}:{:02d}:{:02d}".format(h, m, s, remaining_frames)

    def tc_to_frames(self, tc, fps=None):
        """Convert HH:MM:SS:FF or HH:MM:SS.ms or simple frame count to integer frames."""
        if fps is None:
            fps = self.fps

        if isinstance(tc, (int, float)):
            return int(round(tc))

        if ':' in tc:
            parts = tc.split(':')
            if len(parts) == 4:
                # HH:MM:SS:FF
                h, m, s, f = map(int, parts)
                return int((h * 3600 + m * 60 + s) * fps + f)
            elif len(parts) == 3:
                # HH:MM:SS.ms or HH:MM:SS
                h, m, seconds_str = parts
                h, m = int(h), int(m)
                s = float(seconds_str)
                return int(round((h * 3600 + m * 60 + s) * fps))

        try:
            return int(tc)
        except ValueError:
            return int(round(float(tc) * fps))

    def seconds_to_frames(self, seconds, fps=None):
        """Convert seconds to frames using MLT's lrint (round to nearest) logic."""
        if fps is None:
            fps = self.fps
        return int(round(seconds * fps))

    def _frames_to_tc(self, frames, fps=None):
        return self.frames_to_tc(frames, fps)
    
    def _tc_to_frames(self, tc):
        return self.tc_to_frames(tc)
    
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

    def addFileToBin(self, filepath, duration=40.0, clip_type="1"):
        chain_id = self._get_next_chain_id()
        duration_frames = self.seconds_to_frames(duration)
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
    
    def addClipToTrack(self, track_name, chain_id, in_time, out_time, timeline_start_time, extra_properties=None):
        if track_name not in self.tracks:
            raise ValueError("Track {} not found. Available: {}".format(track_name, list(self.tracks.keys())))
            
        pl_id = self.tracks[track_name]
        pl = self.root.find(".//playlist[@id='{}']".format(pl_id))
        
        in_frame = self.seconds_to_frames(in_time)
        out_frame = self.seconds_to_frames(out_time)
        timeline_start_frame = self.seconds_to_frames(timeline_start_time)

        tc_in = self._frames_to_tc(in_frame)
        tc_out = self._frames_to_tc(out_frame - 1) if out_frame > in_frame else tc_in
        
        # Calculate gap and add blank if needed
        current_duration = self._get_playlist_duration_frames(pl_id)
        gap_frames = timeline_start_frame - current_duration
        
        if gap_frames > 0:
            self.addBlankToTrackByFrames(track_name, gap_frames)
        elif gap_frames < 0:
            logger.warning("Overlapping clips detected: timeline_start={}, current_duration={}".format(timeline_start_time, current_duration / self.fps))
            
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

    def addTrack(self, track_type, name=None):
        """
        Dynamically adds a new track (Audio or Video) to the project.
        track_type: 'audio' or 'video'
        """
        is_audio = track_type.lower() == 'audio'

        # 1. Create Playlist(s)
        # Kdenlive usually uses two playlists per track (one for main clips, one for something else)
        # but we'll stick to one for simplicity, following the template structure.
        pl1_id = self._get_next_playlist_id()
        pl1 = ET.SubElement(self.root, "playlist", id=pl1_id)
        if is_audio:
            ET.SubElement(pl1, "property", name="kdenlive:audio_track").text = "1"

        pl2_id = self._get_next_playlist_id()
        pl2 = ET.SubElement(self.root, "playlist", id=pl2_id)
        if is_audio:
            ET.SubElement(pl2, "property", name="kdenlive:audio_track").text = "1"

        # 2. Create Track Tractor
        tr_id = self._get_next_tractor_id()
        tr = ET.SubElement(self.root, "tractor", id=tr_id, **{"in": "00:00:00:00"})
        if is_audio:
            ET.SubElement(tr, "property", name="kdenlive:audio_track").text = "1"
        ET.SubElement(tr, "property", name="kdenlive:trackheight").text = "64"
        ET.SubElement(tr, "property", name="kdenlive:timeline_active").text = "1"
        ET.SubElement(tr, "property", name="kdenlive:collapsed").text = "0"

        # Add tracks to tractor
        ET.SubElement(tr, "track", producer=pl1_id, hide="video" if is_audio else "audio")
        ET.SubElement(tr, "track", producer=pl2_id, hide="video" if is_audio else "audio")

        # Add default filters
        if is_audio:
            self.addFilterToTractor(tr, "volume", {
                "window": "75", "max_gain": "20dB", "channel_mask": "-1", "internal_added": "237", "disable": "1"
            })
            self.addFilterToTractor(tr, "panner", {
                "channel": "-1", "internal_added": "237", "start": "0.5", "disable": "1"
            })
            self.addFilterToTractor(tr, "audiolevel", {
                "iec_scale": "0", "internal_added": "237", "dbpeak": "1", "disable": "1"
            })

        # 3. Add to Sequence Tractor
        ET.SubElement(self.seq_tractor, "track", producer=tr_id)
        track_idx = len(self.seq_tractor.findall("track")) - 1 # Index 0 is black

        # 4. Add Transition in Sequence
        if is_audio:
            self.addTransition(0, track_idx, "mix", {
                "internal_added": "237", "always_active": "1", "accepts_blanks": "1", "sum": "1"
            })
        else:
            self.addTransition(0, track_idx, "qtblend", {
                "internal_added": "237", "always_active": "1", "compositing": "0", "distort": "0", "rotate_center": "0"
            })

        # Register track
        if not name:
            prefix = "A" if is_audio else "V"
            count = sum(1 for n in self.track_names if n.startswith(prefix)) + 1
            name = "{}{}".format(prefix, count)

        self.tracks[name] = pl1_id
        self.track_names.append(name)

        # Update sequence properties
        tracks_count_prop = self.seq_tractor.find("property[@name='kdenlive:sequenceproperties.tracksCount']")
        if tracks_count_prop is not None:
            tracks_count_prop.text = str(len(self.track_names))

        tracks_prop = self.seq_tractor.find("property[@name='kdenlive:sequenceproperties.tracks']")
        if tracks_prop is not None:
            tracks_prop.text = str(len(self.track_names))

        return name

    def addFilterToTractor(self, tractor, service_name, properties):
        filt = ET.SubElement(tractor, "filter", id=self._get_next_filter_id())
        ET.SubElement(filt, "property", name="mlt_service").text = service_name
        ET.SubElement(filt, "property", name="kdenlive_id").text = service_name
        for k, v in properties.items():
            ET.SubElement(filt, "property", name=k).text = str(v)
        return filt

    def addBlankToTrackByFrames(self, track_name, blank_frames):
        """Add a blank element to a track's playlist using frame count."""
        if track_name not in self.tracks:
            raise ValueError("Track {} not found. Available: {}".format(track_name, list(self.tracks.keys())))
            
        pl_id = self.tracks[track_name]
        pl = self.root.find(".//playlist[@id='{}']".format(pl_id))
        
        blank_tc = self._frames_to_tc(blank_frames)
        ET.SubElement(pl, "blank", length=blank_tc)

    def addBlankToTrack(self, track_name, duration):
        """Add a blank element to a track's playlist using seconds."""
        self.addBlankToTrackByFrames(track_name, self.seconds_to_frames(duration))

    def addVolumeMutes(self, entry, mutes, clip_start_time, clip_end_time):
        """
        Adds volume keyframes to an entry to mute specific intervals.
        mutes: list of (start_time, end_time) in global timeline or absolute file time
        clip_start_time, clip_end_time: the time range of the source file used in this clip
        """
        kfs = []
        clip_duration_frames = self.seconds_to_frames(clip_end_time - clip_start_time)

        # Convert mutes to clip-relative frames
        rel_mutes = []
        for ms, me in mutes:
            is_s = max(ms, clip_start_time)
            is_e = min(me, clip_end_time)
            if is_e > is_s:
                rs = self.seconds_to_frames(is_s - clip_start_time)
                re = self.seconds_to_frames(is_e - clip_start_time)
                rel_mutes.append((rs, re))

        if not rel_mutes: return

        rel_mutes.sort()
        # Simple interval merge for frames
        merged = []
        if rel_mutes:
            merged.append(list(rel_mutes[0]))
            for curr in rel_mutes[1:]:
                if curr[0] <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], curr[1])
                else:
                    merged.append(list(curr))

        if merged[0][0] > 0:
            kfs.append("0=1")

        for idx, (rs, re) in enumerate(merged):
            rs = max(0, rs)
            re = min(clip_duration_frames - 1, re)
            if re < rs: continue

            if rs > 0:
                # Ensure we were at level 1 before mute
                kfs.append(f"{max(0, rs-1)}=1")
            kfs.append(f"{rs}=0")
            kfs.append(f"{re}=0")

            if idx + 1 < len(merged):
                next_rs = merged[idx+1][0]
                if next_rs > re + 1:
                    kfs.append(f"{re+1}=1")
            else:
                if re < clip_duration_frames - 1:
                    kfs.append(f"{re+1}=1")

        if kfs:
            self.addFilterToEntry(entry, "volume", {"gain": ";".join(kfs)})

    def addTransition(self, a_track_idx, b_track_idx, service_name, properties):
        """Add a transition to the sequence tractor."""
        trans = ET.SubElement(self.seq_tractor, "transition", id=self._get_next_transition_id())
        ET.SubElement(trans, "property", name="a_track").text = str(a_track_idx)
        ET.SubElement(trans, "property", name="b_track").text = str(b_track_idx)
        ET.SubElement(trans, "property", name="mlt_service").text = service_name
        ET.SubElement(trans, "property", name="kdenlive_id").text = service_name
        for k, v in properties.items():
            ET.SubElement(trans, "property", name=k).text = str(v)
        return trans

    def addSubtitle(self, filepath, name=None):
        """
        Adds a subtitle file to the project and registers it in the sequence.
        """
        if not name:
            name = os.path.basename(filepath)

        # 1. Add as a project asset (chain)
        # Note: Kdenlive sometimes treats subtitles as distinct assets from media bin
        # but adding to bin ensures it is tracked.
        cid = self.addFileToBin(filepath, duration=0, clip_type="5") # 5 is typical for subtitles

        # 2. Add to sequence properties
        import json
        sub_prop = self.seq_tractor.find("property[@name='kdenlive:sequenceproperties.subtitles']")
        if sub_prop is None:
            sub_prop = ET.SubElement(self.seq_tractor, "property", name="kdenlive:sequenceproperties.subtitles")
            sub_data = {}
        else:
            try:
                sub_data = json.loads(sub_prop.text)
            except (ValueError, TypeError):
                sub_data = {}

        # Format for kdenlive subtitles property (based on general kdenlive JSON properties)
        sub_id = str(uuid.uuid4())
        sub_data[sub_id] = {
            "file": filepath,
            "name": name,
            "active": 1 if not sub_data else 0
        }

        sub_prop.text = json.dumps(sub_data, indent=4)
        return sub_id

    def setDuration(self, seconds):
        """Sets the sequence and project duration."""
        tc = self.frames_to_tc(self.seconds_to_frames(seconds) - 1)
        frames = self.seconds_to_frames(seconds)

        if self.seq_tractor is not None:
            self.seq_tractor.set("out", tc)

            for prop_name, val in [("kdenlive:duration", tc), ("kdenlive:maxduration", str(frames))]:
                prop = self.seq_tractor.find("property[@name='{}']".format(prop_name))
                if prop is not None: prop.text = val
                else: ET.SubElement(self.seq_tractor, "property", name=prop_name).text = val

        # Update black track producer
        bg = self.root.find(".//producer[@id='producer0']")
        if bg is not None:
            bg.set("out", tc)

    def addTimeline(self):
        """No-op for template-based approach - timeline already exists in template."""
        pass

    def save(self, output_path):
        # Fix formatting and save
        xmlstr = ET.tostring(self.root, encoding='utf-8', xml_declaration=True)
        with open(output_path, "wb") as f:
            f.write(xmlstr)
