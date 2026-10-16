"""Fixed chord progressions played as sustained pads beneath the melody.

Triads are built by stacking thirds within the selected scale, so each mode
gives its own chord qualities (Dorian has a major IV, harmonic minor a major V).
The modulating circle of fifths instead plays major triads on roots rising by
fifths while the melody stays in the original key. Each chord is voiced as the
close-position inversion nearest the previous chord, within about an octave of
the chord register, so the harmony moves in small steps and cannot drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..state import MusicSettings
from .scales import SCALES

NO_PROGRESSION = "None"
MODULATING = "Circle of fifths modulating"

# Scale degrees counted from zero, so 0 is I and 4 is V.
PROGRESSIONS: dict[str, tuple[int, ...]] = {
    NO_PROGRESSION: (),
    "Drone (I)": (0,),
    "I–V–vi–IV": (0, 4, 5, 3),
    "I–IV–V–I": (0, 3, 4, 0),
    "ii–V–I": (1, 4, 0),
    "Circle of fifths in key": (0, 3, 6, 2, 5, 1, 4, 0),
    MODULATING: (),
}

BEATS_PER_BAR = 4.0
CHORD_LENGTHS = (("Half a bar", 2.0), ("One bar", 4.0), ("Two bars", 8.0))

_MAJOR_TRIAD = (0, 4, 7)


@dataclass(frozen=True)
class ChordEvent:
    pitches: tuple[int, ...]  # MIDI note numbers, lowest first
    start: float  # beats
    duration: float  # beats


def chord_centre(settings: MusicSettings) -> int:
    """The root note transposed into the chord register."""
    return settings.root + 12 * settings.chord_octave


def triad_classes(settings: MusicSettings, index: int) -> tuple[int, ...]:
    """Pitch classes of the chord at a given position in the progression."""
    if settings.progression == MODULATING:
        root = settings.root + 7 * index
        return tuple((root + step) % 12 for step in _MAJOR_TRIAD)
    degrees = PROGRESSIONS[settings.progression]
    degree = degrees[index % len(degrees)]
    intervals = SCALES[settings.scale]
    return tuple((settings.root + intervals[(degree + k) % 7]) % 12 for k in (0, 2, 4))


def _voicings(classes: tuple[int, ...], centre: int) -> list[tuple[int, ...]]:
    """Every close-position voicing whose lowest note lies within a tritone of the centre."""
    found = []
    for low in range(centre - 6, centre + 6):
        if low % 12 not in classes:
            continue
        chord = [low]
        for _ in range(len(classes) - 1):
            chord.append(next(p for p in range(chord[-1] + 1, chord[-1] + 13) if p % 12 in classes))
        found.append(tuple(chord))
    return found


def _nearest(candidates: list[tuple[int, ...]], previous: tuple[int, ...] | None,
             root_class: int, centre: int) -> tuple[int, ...]:
    if previous is None:
        # Begin in root position, as close to the centre as possible.
        rooted = [c for c in candidates if c[0] % 12 == root_class]
        return min(rooted or candidates, key=lambda c: abs(c[0] - centre))
    return min(candidates, key=lambda c: (sum(abs(a - b) for a, b in zip(c, previous)),
                                          abs(sum(c) / len(c) - centre - 4)))


def build_chords(settings: MusicSettings, total_beats: float) -> list[ChordEvent]:
    """Repeat the progression across the loop. A final partial chord is cut short."""
    if settings.progression == NO_PROGRESSION or total_beats <= 0:
        return []
    length = settings.chord_beats
    count = math.ceil(total_beats / length - 1e-9)
    centre = chord_centre(settings)
    chords: list[ChordEvent] = []
    previous: tuple[int, ...] | None = None
    for i in range(count):
        classes = triad_classes(settings, i)
        voicing = _nearest(_voicings(classes, centre), previous, classes[0], centre)
        previous = voicing
        pitches = tuple(p for p in voicing if 0 <= p <= 127)
        start = i * length
        chords.append(ChordEvent(pitches, start, min(length, total_beats - start)))
    return chords


def loop_fit_message(settings: MusicSettings) -> str | None:
    """Explain when the loop is not a whole number of chord lengths, or return None."""
    if settings.progression == NO_PROGRESSION:
        return None
    total = settings.num_notes * settings.note_beats
    if abs(total / settings.chord_beats - round(total / settings.chord_beats)) < 1e-9:
        return None
    per_chord = settings.chord_beats / settings.note_beats
    return (f"The loop does not hold a whole number of chords, so the last chord is cut short. "
            f"Choose a multiple of {per_chord:g} notes for whole chords.")
