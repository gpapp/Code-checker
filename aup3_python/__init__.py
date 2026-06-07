"""aup3_python - A lightweight Python wrapper around py-aup3 for common AUP3 tasks.

Audacity AUP3 files are SQLite databases with:
- `project.doc`: binary XML containing the project tree (tracks, clips, labels, etc.)
- `project.dict`: string dictionary mapping integer IDs to tag/attribute names
- `sampleblocks`: raw float32 PCM audio samples referenced by `blockid`

The binary XML tree structure:
  <project>
    <wavetrack name="..." rate="..." channel="..." sampleformat="...">
      <waveclip name="..." offset="...">
        <sequence sampleformat="...">
          <waveblock start="..." blockid="..." len="..." [trim_left="..."] [trim_right="..."] [n_channels="..."]/>
        </sequence>
        <envelope>
          <controlpoint/>
        </envelope>
      </waveclip>
    </wavetrack>
    <labeltrack name="...">
      <label t="..." t1="..." title="..."/>
    </labeltrack>

Audio is stored in sampleblocks as float32 PCM (4 bytes/sample, 1 channel per block).
WaveClip.offset gives the clip's position on the global timeline in seconds.
WaveBlock.start is the sample offset within the clip's sequence.
WaveBlock.trim_left/trim_right indicate trimmed samples at boundaries.
"""

from __future__ import annotations

import os
import dataclasses
from typing import List, Optional, Tuple

import numpy as np
from aup3 import AUP3, xml as aup3_xml


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Label:
    """A single label in a label track."""
    t: float       # start time in seconds on the global timeline
    t1: float      # end time in seconds
    title: str     # label text (e.g. "Bad", "False", "filler", etc.)


@dataclasses.dataclass
class WaveBlock:
    """A contiguous chunk of audio samples referenced by blockid.

    In the AUP3 schema each waveblock maps to one row in sampleblocks.
    """
    start: int            # sample offset *within* the parent clip's sequence
    blockid: int          # primary key into the sampleblocks table
    length: int           # number of audio frames in this block (after trim)
    n_channels: int       # 1 (mono) or 2 (stereo)
    trim_left: int        # samples trimmed from the *stored* block's left edge
    trim_right: int       # samples trimmed from the *stored* block's right edge


@dataclasses.dataclass
class WaveClip:
    """A clip on the timeline containing one or more waveblocks."""
    offset: float         # position on the global timeline (seconds)
    sample_rate: float
    num_samples: int      # total samples in this clip (from <sequence numsamples>)
    blocks: List[WaveBlock]


@dataclasses.dataclass
class WaveTrack:
    """An audio wave track with a list of non-overlapping clips."""
    name: str
    sample_rate: float
    channel: int          # 0 = mono, 1 = left, 2 = right, 3 = stereo
    clips: List[WaveClip]


@dataclasses.dataclass
class LabelTrack:
    """A label (annotation) track."""
    name: str
    labels: List[Label]


# ---------------------------------------------------------------------------
# Project wrapper
# ---------------------------------------------------------------------------

class Aup3Project:
    """High-level access to an Audacity AUP3 project file.

    Usage::

        proj = Aup3Project("recording.aup3")
        for track in proj.wave_tracks:
            print(track.name, track.sample_rate)
        for label in proj.label_tracks[0].labels:
            audio = proj.extract_audio(label.t, label.t1)
            # ... save as FLAC, etc.
    """

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)
        self.db = AUP3(self.path)

        self.wave_tracks: List[WaveTrack] = []
        self.label_tracks: List[LabelTrack] = []

        self._build_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self.db.conn.close()

    @property
    def sample_rate(self) -> float:
        """Return the project's primary sample rate (first wavetrack's rate).

        Raises RuntimeError if no wavetracks exist.
        """
        if not self.wave_tracks:
            raise RuntimeError("no wave tracks in project")
        return self.wave_tracks[0].sample_rate

    def find_labels(self, title_filter: Optional[str] = None) -> List[Tuple[str, Label]]:
        """Return all labels, optionally filtered by *title*.

        Returns (label_track_name, Label) pairs.
        """
        result: List[Tuple[str, Label]] = []
        for lt in self.label_tracks:
            for label in lt.labels:
                if title_filter is None or label.title == title_filter:
                    result.append((lt.name, label))
        return result

    def extract_audio_for_label(
        self, label: Label, track_name_hint: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Extract the raw float32 audio for *label* from the best-matching wave track.

        Parameters
        ----------
        label:
            The label interval to extract.
        track_name_hint:
            If given, only search this track.

        Returns
        -------
        A 1-D float32 NumPy array in the track's native sample rate,
        or ``None`` when no audio covers the interval.
        """
        tracks = self._tracks_for_timespan(label.t, label.t1, track_name_hint)
        if not tracks:
            return None
        # Prefer the track whose offset exactly matches label.t (within 10 ms)
        best = tracks[0]
        best_offset_diff = abs(best[0].clips[best[1]].offset - label.t) if len(best) >= 2 else float("inf")
        for candidate in tracks:
            clip_idx = candidate[1]
            clip = candidate[0].clips[clip_idx]
            diff = abs(clip.offset - label.t)
            if diff < best_offset_diff:
                best = candidate
                best_offset_diff = diff

        track = best[0]
        return self._extract_from_track(track, label.t, label.t1)

    def list_wave_tracks(self) -> List[str]:
        """Return names of all wave tracks."""
        return [t.name for t in self.wave_tracks]

    def list_label_tracks(self) -> List[str]:
        """Return names of all label tracks."""
        return [t.name for t in self.label_tracks]

    # ------------------------------------------------------------------
    # Internal – XML parsing
    # ------------------------------------------------------------------

    def _build_model(self) -> None:
        dict_raw, doc_raw = self.db.conn.execute(
            "SELECT dict, doc FROM project WHERE id = 1"
        ).fetchone()

        nodes = aup3_xml.decode(dict_raw, doc_raw)
        _doctype, root = aup3_xml.get_root(nodes)

        for child in root.children:
            if child.tag == "wavetrack":
                self.wave_tracks.append(self._parse_wavetrack(child))
            elif child.tag == "labeltrack":
                self.label_tracks.append(self._parse_labeltrack(child))

    def _parse_wavetrack(self, element) -> WaveTrack:
        name = element.attrs.get("name", ("String", ""))[1]
        rate = element.attrs.get("rate", ("Double", 44100.0))[1]
        channel = element.attrs.get("channel", ("Int", 0))[1]

        clips: List[WaveClip] = []
        for child in element.children:
            if child.tag == "waveclip":
                clip = self._parse_waveclip(child, rate)
                clips.append(clip)

        clips.sort(key=lambda c: c.offset)
        return WaveTrack(name=name, sample_rate=rate, channel=channel, clips=clips)

    def _parse_waveclip(self, element, track_rate: float) -> WaveClip:
        offset = element.attrs.get("offset", ("Double", 0.0))[1]
        num_samples = 0
        blocks: List[WaveBlock] = []

        for child in element.children:
            if child.tag == "sequence":
                for attr_key in ("numsamples", "numSamples", "NumSamples"):
                    if attr_key in child.attrs:
                        num_samples = child.attrs[attr_key][1]
                        break
                for wb_elem in child.children:
                    if wb_elem.tag == "waveblock":
                        block = self._parse_waveblock(wb_elem)
                        blocks.append(block)

        return WaveClip(offset=offset, sample_rate=track_rate, num_samples=num_samples, blocks=blocks)

    @staticmethod
    def _parse_waveblock(element) -> WaveBlock:
        start = element.attrs.get("start", ("LongLong", 0))[1]
        blockid = element.attrs.get("blockid", ("LongLong", 0))[1]
        n_channels = element.attrs.get("n_channels", ("Int", 1))[1]

        # len might be SizeT or LongLong depending on Audacity version
        len_attr = element.attrs.get("len", None)
        if len_attr and len_attr[0] in ("SizeT", "LongLong", "Int"):
            length = len_attr[1]
        elif len_attr:
            length = len_attr[1]
        else:
            length = 0

        trim_left_attr = element.attrs.get("trim_left", None)
        trim_right_attr = element.attrs.get("trim_right", None)
        trim_left = trim_left_attr[1] if trim_left_attr else 0
        trim_right = trim_right_attr[1] if trim_right_attr else 0

        return WaveBlock(
            start=start,
            blockid=blockid,
            length=length,
            n_channels=n_channels,
            trim_left=trim_left,
            trim_right=trim_right,
        )

    @staticmethod
    def _parse_labeltrack(element) -> LabelTrack:
        name = element.attrs.get("name", ("String", ""))[1]
        labels: List[Label] = []
        for child in element.children:
            if child.tag == "label":
                t = child.attrs.get("t", ("Double", 0.0))[1]
                t1 = child.attrs.get("t1", ("Double", 0.0))[1]
                title = child.attrs.get("title", ("String", ""))[1]
                labels.append(Label(t=t, t1=t1, title=title))
        return LabelTrack(name=name, labels=labels)

    # ------------------------------------------------------------------
    # Internal – audio extraction
    # ------------------------------------------------------------------

    def _tracks_for_timespan(
        self, start: float, end: float, hint: Optional[str] = None
    ) -> List[Tuple[WaveTrack, int]]:
        """Return (track, clip_index) pairs whose clips cover [start, end).

        If *hint* is given, only check the named track.
        """
        result: List[Tuple[WaveTrack, int]] = []
        tracks = (
            [t for t in self.wave_tracks if t.name == hint]
            if hint
            else self.wave_tracks
        )
        for track in tracks:
            for idx, clip in enumerate(track.clips):
                clip_end = clip.offset + self._clip_duration(clip)
                if clip.offset <= end and clip_end >= start:
                    result.append((track, idx))
        return result

    @staticmethod
    def _clip_duration(clip: WaveClip) -> float:
        """Total duration of a clip in seconds (from num_samples)."""
        if clip.num_samples > 0:
            return clip.num_samples / clip.sample_rate
        # Fallback: sum of block lengths
        total_samples = sum(b.length for b in clip.blocks)
        return total_samples / clip.sample_rate

    def _extract_from_track(self, track: WaveTrack, t_start: float, t_end: float) -> np.ndarray:
        """Read raw float32 audio from *track* for the range [t_start, t_end).

        Handles trim_left/trim_right and multi-channel blocks.
        """
        sample_rate = track.sample_rate
        start_sample = int(round(t_start * sample_rate))
        end_sample = int(round(t_end * sample_rate))
        total_needed = end_sample - start_sample

        gathered: List[np.ndarray] = []
        written = 0

        for clip in track.clips:
            clip_start_sample = int(round(clip.offset * sample_rate))
            clip_dur = self._clip_duration(clip)
            clip_end_sample = clip_start_sample + int(round(clip_dur * sample_rate))

            if clip_end_sample <= start_sample or clip_start_sample >= end_sample:
                continue

            # Intersection of [clip_start, clip_end) with [start_sample, end_sample)
            read_from = max(start_sample, clip_start_sample)
            read_to = min(end_sample, clip_end_sample)
            need = read_to - read_from
            if need <= 0:
                continue

            # Offset within clip
            clip_offset = read_from - clip_start_sample
            block_data = self._read_clip_range(clip, clip_offset, need)
            gathered.append(block_data)
            written += len(block_data)

        # Trim or pad to exact length
        if written < total_needed:
            gathered.append(np.zeros(total_needed - written, dtype=np.float32))
        elif written > total_needed:
            # Concatenate then trim
            pass

        result = np.concatenate(gathered)[:total_needed] if gathered else np.zeros(total_needed, dtype=np.float32)
        return result

    def _read_clip_range(self, clip: WaveClip, offset_samples: int, num_samples: int) -> np.ndarray:
        """Read *num_samples* from *clip* starting at *offset_samples* into the clip.

        The clip's blocks are concatenated in order (block.start defines the
        position within the clip's sample sequence).
        """
        sr = clip.sample_rate
        chunks: List[np.ndarray] = []
        remaining = num_samples
        clip_pos = 0  # sample position within the clip's reconstructed audio

        for block in clip.blocks:
            # Reconstruct this block's audio from raw samples
            block_audio = self._read_block(block)
            block_len = len(block_audio)

            block_start_in_clip = clip_pos
            block_end_in_clip = clip_pos + block_len

            if block_end_in_clip <= offset_samples or block_start_in_clip >= offset_samples + num_samples:
                clip_pos += block_len
                continue

            # Read window
            local_start = max(0, offset_samples - block_start_in_clip)
            local_end = min(block_len, offset_samples + num_samples - block_start_in_clip)
            chunk = block_audio[local_start:local_end]
            chunks.append(chunk)
            remaining -= len(chunk)

            clip_pos += block_len
            if remaining <= 0:
                break

        if remaining > 0:
            chunks.append(np.zeros(remaining, dtype=np.float32))

        return np.concatenate(chunks)

    def _read_block(self, block: WaveBlock) -> np.ndarray:
        """Fetch raw samples from DB for a single waveblock and apply trim."""
        raw = self.db.get_block(block.blockid)

        if raw is None:
            if block.length > 0:
                return np.zeros(block.length, dtype=np.float32)
            return np.array([], dtype=np.float32)

        # raw might be multi-channel (2D array) or 1D
        if raw.ndim > 1:
            # Sum stereo to mono
            raw = raw.mean(axis=1)

        # Ensure 1D float32
        raw = np.asarray(raw, dtype=np.float32).ravel()

        # Apply stored-block trimming (trim_left/trim_right refer to the stored
        # block data, NOT the final length; the attributes are only meaningful
        # when the stored block is longer than what the waveblock advertises)
        if block.trim_left or block.trim_right:
            raw = raw[block.trim_left:]
            if block.trim_right:
                raw = raw[:-block.trim_right] if block.trim_right < len(raw) else np.array([], dtype=np.float32)

        # If block.length is known (non-zero), pad/trim; otherwise use actual data length
        if block.length > 0:
            if len(raw) < block.length:
                raw = np.pad(raw, (0, block.length - len(raw)))
            elif len(raw) > block.length:
                raw = raw[:block.length]

        return raw


# ---------------------------------------------------------------------------
# Convenience — extract labeled clips from a project
# ---------------------------------------------------------------------------

def extract_label_clips(
    proj: Aup3Project,
    label_filter: Optional[str] = None,
    target_sr: int = 16000,
) -> List[Tuple[np.ndarray, int, str, Label]]:
    """Extract audio for every label in the project.

    Returns a list of ``(audio, sample_rate, track_name, label)`` tuples.
    *audio* is resampled to *target_sr* Hz, mono, float32.
    """
    import torch
    import torchaudio

    clips: List[Tuple[np.ndarray, int, str, Label]] = []

    for track_name, label in proj.find_labels(label_filter):
        audio = proj.extract_audio_for_label(label, track_name_hint=None)
        if audio is None or len(audio) == 0:
            continue

        # Determine source sample rate from the track that matched
        src_sr = proj.sample_rate  # fallback
        for wt in proj.wave_tracks:
            for clip in wt.clips:
                clip_end = clip.offset + sum(b.length for b in clip.blocks) / clip.sample_rate
                if clip.offset <= label.t < clip_end:
                    src_sr = clip.sample_rate
                    break

        # Resample to target_sr
        if src_sr != target_sr:
            tensor = torch.from_numpy(audio).float().unsqueeze(0)
            resampler = torchaudio.transforms.Resample(
                orig_freq=int(src_sr), new_freq=target_sr
            )
            tensor = resampler(tensor)
            audio = tensor.squeeze(0).numpy()

        clips.append((audio, target_sr, track_name, label))

    return clips
