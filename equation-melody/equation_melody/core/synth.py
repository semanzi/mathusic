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

# Chords use a fixed soft pad whatever the melody timbre, so they stay beneath it.
PAD_PARTIALS: tuple[tuple[int, float], ...] = ((1, 1.0), (2, 0.25))
PAD_ATTACK_S = 0.15
PAD_RELEASE_S = 0.3


def midi_to_hz(midi: int) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _oscillator(timbre: str, freq: float, t: np.ndarray, sample_rate: int) -> np.ndarray:
    return _additive(TIMBRES.get(timbre, TIMBRES["Sine"]), freq, t, sample_rate)


def _additive(partials: tuple[tuple[int, float], ...], freq: float, t: np.ndarray,
              sample_rate: int) -> np.ndarray:
    wave = np.zeros_like(t)
    total = 0.0
    for harmonic, amplitude in partials:
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


def _pad_envelope(hold: int, release: int, sample_rate: int) -> np.ndarray:
    attack = min(int(PAD_ATTACK_S * sample_rate), hold)
    body = np.concatenate([np.linspace(0.0, 1.0, attack, endpoint=False),
                           np.ones(hold - attack)])
    level = body[-1] if len(body) else 0.0
    return np.concatenate([body, np.linspace(level, 0.0, release)])


def render_audio(melody: Melody, settings: MusicSettings, sample_rate: int = 44100) -> np.ndarray:
    """Return a mono float32 buffer exactly one loop long.

    Chords are mixed beneath the melody at the chord volume. Release tails that
    run past the end are folded back to the start so that looped playback is
    seamless.
    """
    spb = settings.seconds_per_beat
    length = max(int(math.ceil(melody.total_beats * spb * sample_rate)), 1)
    release = int(RELEASE_S * sample_rate)
    pad_release = int(PAD_RELEASE_S * sample_rate)
    with_chords = bool(melody.chords) and settings.chord_volume > 0
    buffer = np.zeros(length + max(release, pad_release if with_chords else 0), dtype=np.float64)
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

    if with_chords:
        _mix_chords(buffer, melody, settings, sample_rate, pad_release)

    for offset in range(length, len(buffer), length):
        tail = buffer[offset:offset + length]
        buffer[:len(tail)] += tail
    buffer = buffer[:length]
    peak = float(np.max(np.abs(buffer))) if len(buffer) else 0.0
    if peak > 0:
        buffer *= PEAK / peak
    return buffer.astype(np.float32)


def _mix_chords(buffer: np.ndarray, melody: Melody, settings: MusicSettings,
                sample_rate: int, release: int) -> None:
    """Add sustained pad chords beneath the melody.

    Each chord in total is as loud as the melody's sustain level times the chord volume.
    """
    spb = settings.seconds_per_beat
    cache: dict[tuple[int, int], np.ndarray] = {}
    for chord in melody.chords:
        if not chord.pitches:
            continue
        start = int(round(chord.start * spb * sample_rate))
        hold = max(int(round(chord.duration * spb * sample_rate)), 1)
        gain = settings.chord_volume * SUSTAIN / len(chord.pitches)
        envelope = _pad_envelope(hold, release, sample_rate)
        for pitch in chord.pitches:
            key = (pitch, hold)
            voice = cache.get(key)
            if voice is None:
                t = np.arange(hold + release) / sample_rate
                voice = _additive(PAD_PARTIALS, midi_to_hz(pitch), t, sample_rate) * envelope
                cache[key] = voice
            end = min(start + len(voice), len(buffer))
            buffer[start:end] += gain * voice[: end - start]
