"""Turn equation values into a melody.

The pipeline is sample, then normalise, then quantise, then build note events.
Undefined values (division by zero, logarithms of negatives, overflow) are
treated as rests rather than errors, so any equation remains playable.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

from ..state import MusicSettings
from .harmony import ChordEvent, build_chords
from .scales import SCALES

NORMALISATION_MODES = ("Percentile (robust)", "Median / MAD", "Rank (quantile)")

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

REST = -1
_TINY = 1e-12


def midi_to_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


@dataclass(frozen=True)
class NoteEvent:
    pitch: int  # MIDI note number
    start: float  # beats
    duration: float  # beats
    step: int  # index of the step that started the note


@dataclass
class Normalisation:
    z: np.ndarray  # values in [0, 1], NaN where undefined
    low: float  # value mapped to the bottom of the range (original units)
    high: float  # value mapped to the top of the range (original units)
    undefined: int
    flat: bool  # True when every defined value is effectively equal


@dataclass
class Melody:
    n: np.ndarray
    y: np.ndarray
    normalisation: Normalisation
    pitches: np.ndarray  # MIDI numbers, REST where undefined
    events: list[NoteEvent]
    total_beats: float
    settings: MusicSettings
    chords: list[ChordEvent] = field(default_factory=list)


def index_values(settings: MusicSettings) -> np.ndarray:
    return settings.n_start + settings.n_step * np.arange(settings.num_notes, dtype=float)


def sample(func: Callable[..., object], n: np.ndarray, values: Sequence[float]) -> np.ndarray:
    """Evaluate the compiled equation at every step. Never raises."""
    params = [np.float64(v) for v in values]  # NumPy floats give inf/NaN rather than exceptions
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            y = np.asarray(func(n, *params))
            if np.iscomplexobj(y):
                real_enough = np.abs(y.imag) <= 1e-9 * np.maximum(1.0, np.abs(y.real))
                y = np.where(real_enough, y.real, np.nan)
            y = np.broadcast_to(y.astype(float), n.shape).copy()
        except Exception:
            return np.full(n.shape, np.nan)
    y[~np.isfinite(y)] = np.nan
    return y


def normalise(y: np.ndarray, mode: str, clip_percent: float = 5.0) -> Normalisation:
    """Map values onto [0, 1] in a way that resists outliers and degenerate input.

    Values are first divided by their largest magnitude, so equations producing
    numbers near 1e300 or 1e-300 behave the same as ordinary ones. Three
    strategies are offered.

    Percentile clips to the chosen lower and upper percentiles, so a few extreme
    values cannot squash the rest of the melody into one note.
    Median / MAD centres on the median and squashes with tanh, scaled by the
    median absolute deviation. It falls back to the interquartile range, then the
    standard deviation, when most values are identical.
    Rank replaces values by their rank among distinct values, giving an even
    spread of pitches whatever the shape of the distribution.
    """
    z = np.full(y.shape, np.nan)
    finite = np.isfinite(y)
    undefined = int((~finite).sum())
    if not finite.any():
        return Normalisation(z, np.nan, np.nan, undefined, flat=True)

    x = y[finite]
    scale = float(np.max(np.abs(x)))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    xs = x / scale

    if mode.startswith("Rank"):
        distinct, inverse = np.unique(np.round(xs, 12), return_inverse=True)
        if len(distinct) == 1:
            z[finite] = 0.5
            return Normalisation(z, float(x.min()), float(x.max()), undefined, flat=True)
        z[finite] = inverse / (len(distinct) - 1)
        return Normalisation(z, float(x.min()), float(x.max()), undefined, flat=False)

    if mode.startswith("Median"):
        centre = float(np.median(xs))
        spread = 1.4826 * float(np.median(np.abs(xs - centre)))
        if spread <= _TINY:
            q1, q3 = np.percentile(xs, [25, 75])
            spread = float(q3 - q1) / 1.349
        if spread <= _TINY:
            spread = float(np.std(xs))
        if spread <= _TINY:
            z[finite] = 0.5
            return Normalisation(z, centre * scale, centre * scale, undefined, flat=True)
        z[finite] = 0.5 + 0.5 * np.tanh((xs - centre) / (2.0 * spread))
        return Normalisation(z, (centre - 2 * spread) * scale, (centre + 2 * spread) * scale,
                             undefined, flat=False)

    pct = float(np.clip(clip_percent, 0.0, 49.0))
    lo, hi = (float(v) for v in np.percentile(xs, [pct, 100.0 - pct]))
    if hi - lo <= _TINY:
        lo, hi = float(xs.min()), float(xs.max())  # widen to the full range before giving up
    if hi - lo <= _TINY:
        z[finite] = 0.5
        return Normalisation(z, lo * scale, hi * scale, undefined, flat=True)
    z[finite] = np.clip((xs - lo) / (hi - lo), 0.0, 1.0)
    return Normalisation(z, lo * scale, hi * scale, undefined, flat=False)


def scale_pitches(settings: MusicSettings) -> list[int]:
    """Every pitch available to the melody, lowest first."""
    intervals = SCALES[settings.scale]
    pitches = [settings.root + 12 * octave + step
               for octave in range(settings.octaves) for step in intervals]
    pitches.append(settings.root + 12 * settings.octaves)
    return [p for p in pitches if 0 <= p <= 127]


def quantise(z: np.ndarray, settings: MusicSettings) -> np.ndarray:
    """Snap normalised values to the nearest scale degree. Undefined values become rests."""
    available = np.array(scale_pitches(settings))
    pitches = np.full(z.shape, REST, dtype=int)
    defined = np.isfinite(z)
    positions = np.rint(z[defined] * (len(available) - 1)).astype(int)
    pitches[defined] = available[np.clip(positions, 0, len(available) - 1)]
    return pitches


def to_events(pitches: np.ndarray, settings: MusicSettings) -> list[NoteEvent]:
    beat = settings.note_beats
    events: list[NoteEvent] = []
    for step, pitch in enumerate(pitches):
        if pitch == REST:
            continue
        start = step * beat
        previous = events[-1] if events else None
        if (settings.merge_repeats and previous is not None and previous.pitch == pitch
                and abs(previous.start + previous.duration - start) < 1e-9):
            events[-1] = replace(previous, duration=previous.duration + beat)
        else:
            events.append(NoteEvent(int(pitch), start, beat, step))
    return events


def melody_pitches(func: Callable[..., object], values: Sequence[float],
                   settings: MusicSettings, n: np.ndarray | None = None) -> np.ndarray:
    n = index_values(settings) if n is None else n
    norm = normalise(sample(func, n, values), settings.normalisation, settings.clip_percent)
    return quantise(norm.z, settings)


def build_melody(func: Callable[..., object], values: Sequence[float],
                 settings: MusicSettings) -> Melody:
    n = index_values(settings)
    y = sample(func, n, values)
    norm = normalise(y, settings.normalisation, settings.clip_percent)
    pitches = quantise(norm.z, settings)
    total_beats = len(n) * settings.note_beats
    return Melody(n=n, y=y, normalisation=norm, pitches=pitches,
                  events=to_events(pitches, settings),
                  total_beats=total_beats, settings=settings,
                  chords=build_chords(settings, total_beats))


def probe_influence(func: Callable[..., object], values: Sequence[float],
                    ranges: Sequence[tuple[float, float]], settings: MusicSettings,
                    current: np.ndarray) -> list[bool]:
    """Report, for each parameter, whether moving it changes any note.

    Normalisation absorbs anything that only scales or shifts the whole equation,
    so such parameters are flagged as having no audible effect.
    """
    n = index_values(settings)
    flags: list[bool] = []
    for i, (lo, hi) in enumerate(ranges):
        changed = False
        for fraction in (0.0, 0.37, 0.71, 1.0):
            trial = list(values)
            trial[i] = lo + (hi - lo) * fraction
            if trial[i] == values[i]:
                continue
            if not np.array_equal(melody_pitches(func, trial, settings, n), current):
                changed = True
                break
        flags.append(changed)
    return flags
