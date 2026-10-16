"""Musical settings shared between the GUI and the melody engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MusicSettings:
    """Everything that shapes the melody apart from the equation and its parameters."""

    num_notes: int = 32
    n_start: float = 0.0
    n_step: float = 1.0
    root: int = 60  # MIDI note number, 60 is middle C (C4)
    scale: str = "Major"
    octaves: int = 2
    normalisation: str = "Percentile (robust)"
    clip_percent: float = 5.0
    tempo: float = 120.0  # beats per minute
    note_beats: float = 0.5  # length of each step in beats (0.5 is a quaver)
    gate: float = 0.85  # fraction of each step that sounds
    merge_repeats: bool = False
    timbre: str = "Soft square"
    progression: str = "None"  # "None" adds no chords and leaves the melody unchanged
    chord_beats: float = 4.0  # length of each chord in beats (4 is one bar of 4/4)
    chord_octave: int = -1  # chord register relative to the root note, in octaves
    chord_volume: float = 0.4  # chord level relative to the melody

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.tempo
