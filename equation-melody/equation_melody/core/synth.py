"""Render note events to an audio buffer with simple additive oscillators."""

from __future__ import annotations

import math

import numpy as np

from ..state import MusicSettings
from .melody import Melody

# Harmonic number and relative amplitude for each timbre. Additive synthesis
# avoids aliasing because partials above the Nyquist limit are skipped.
TIMBRES: dict[str, tuple[tuple[int, float], ...]] = {
    "Sine": ((1, 1.0),),
    "Soft square": ((1, 1.0), (3, 1 / 3), (5, 1 / 5), (7, 1 / 7)),
    "Triangle": ((1, 1.0), (3, -1 / 9), (5, 1 / 25), (7, -1 / 49)),
    "Organ": ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.15)),
}

ATTACK_S = 0.005
DECAY_S = 0.08
SUSTAIN = 0.75
RELEASE_S = 0.06
PEAK = 0.8


def midi_to_hz(midi: int) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _oscillator(timbre: str, freq: float, t: np.ndarray, sample_rate: int) -> np.ndarray:
    wave = np.zeros_like(t)
    total = 0.0
    for harmonic, amplitude in TIMBRES.get(timbre, TIMBRES["Sine"]):
        if freq * harmonic >= 0.45 * sample_rate:
            continue
        wave += amplitude * np.sin(2 * math.pi * freq * harmonic * t)
        total += abs(amplitude)
    return wave / total if total else wave


def _envelope(hold: int, release: int, sample_rate: int) -> np.ndarray:
    attack = min(int(ATTACK_S * sample_rate), hold)
    decay = min(int(DECAY_S * sample_rate), hold - attack)
    sustain = hold - attack - decay
    body = np.concatenate([
        np.linspace(0.0, 1.0, attack, endpoint=False),
        np.linspace(1.0, SUSTAIN, decay, endpoint=False),
        np.full(sustain, SUSTAIN),
    ])
    level = body[-1] if len(body) else 0.0
    return np.concatenate([body, np.linspace(level, 0.0, release)])


def render_audio(melody: Melody, settings: MusicSettings, sample_rate: int = 44100) -> np.ndarray:
    """Return a mono float32 buffer exactly one loop long.

    Release tails that run past the end are folded back to the start so that
    looped playback is seamless.
    """
    spb = settings.seconds_per_beat
    length = max(int(math.ceil(melody.total_beats * spb * sample_rate)), 1)
    release = int(RELEASE_S * sample_rate)
    buffer = np.zeros(length + release, dtype=np.float64)
    cache: dict[tuple[int, int], np.ndarray] = {}  # melodies reuse few distinct notes

    for event in melody.events:
        start = int(round(event.start * spb * sample_rate))
        hold = max(int(event.duration * settings.gate * spb * sample_rate), 1)
        key = (event.pitch, hold)
        note = cache.get(key)
        if note is None:
            t = np.arange(hold + release) / sample_rate
            note = _oscillator(settings.timbre, midi_to_hz(event.pitch), t, sample_rate)
            note *= _envelope(hold, release, sample_rate)
            cache[key] = note
        end = min(start + len(note), len(buffer))
        buffer[start:end] += note[: end - start]

    buffer[:release] += buffer[length:length + release]
    buffer = buffer[:length]
    peak = float(np.max(np.abs(buffer))) if len(buffer) else 0.0
    if peak > 0:
        buffer *= PEAK / peak
    return buffer.astype(np.float32)
