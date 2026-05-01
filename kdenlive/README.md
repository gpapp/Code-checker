# Kdenlive / MLT XML Reference

The exporter (`exporter.py`) generates project files compatible with Kdenlive using the MLT framework. This document covers formatting rules and structural requirements.

## Timecodes

- Use `HH:MM:SS:FF` format (e.g., `00:00:01:12`) for frame-accurate timeline elements.
- The `out` point in MLT is **inclusive**. For a clip of 100 frames starting at 0: `in="00:00:00:00"` and `out="00:00:03:24"` (frame 99 at 25fps). Formula: `out = total_frames - 1`.

## XML Structure

### Chains vs Producers
- Use `<chain>` for physical media files (avformat)
- Use `<producer>` for generated content like `color` (black track)

### Tractors
Every track in Kdenlive is represented by a `<tractor>`.
- Audio tracks MUST have `kdenlive:audio_track=1`.
- Inside the track tractor, `<track>` elements should have `hide="video"` (audio tracks) or `hide="audio"` (video tracks).

### Sequence Properties
The sequence tractor should define:
- `kdenlive:sequenceproperties.tracks`
- `tracksCount` as integer values (total user tracks, excluding background black track)

## Filters (Non-destructive)

- **volume**: Use for muting. Keyframes follow `frame=level` format (e.g., `0=1.0;24=0.0`). Clip effects go on `<chain>`, track effects on `<tractor>`.
- **ladspa.1073**: Standard compressor.
- **dynamic_loudness**: Normalizes audio to target LUFS (default `-14`).

## Transitions (Timeline Stability)

- Transitions in the main sequence tractor MUST blend/mix tracks against track 0 (the `black_track`).
- Use `a_track="0"` and `b_track="<track_index>"` for all transitions. Do NOT cascade transitions (e.g., `1->2`, `2->3`).
- **qtblend**: Standard video track compositor.
- **mix**: Standard audio track mixer.

## Project Structure

```
source files → <chain> → <playlist> → track <tractor> → sequence <tractor> → project <tractor>
```
