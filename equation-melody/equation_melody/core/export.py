"""Write the melody to WAV and MIDI using only the standard library."""

from __future__ import annotations

import struct
import wave

import numpy as np

from ..state import MusicSettings
from .melody import Melody

TICKS_PER_BEAT = 480


def write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _varlen(value: int) -> bytes:
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def midi_bytes(melody: Melody, settings: MusicSettings) -> bytes:
    """Build a single-track Standard MIDI File (format 0).

    The melody is on channel 1 and chords on channel 2.
    """
    timed: list[tuple[int, int, bytes]] = []  # (tick, order, message). Note-offs sort first.
    for event in melody.events:
        on = int(round(event.start * TICKS_PER_BEAT))
        off = on + max(int(round(event.duration * settings.gate * TICKS_PER_BEAT)), 1)
        timed.append((on, 1, bytes([0x90, event.pitch, 90])))
        timed.append((off, 0, bytes([0x80, event.pitch, 0])))
    velocity = int(round(90 * min(max(settings.chord_volume * 1.5, 0.1), 1.0)))
    for chord in melody.chords:  # chords go on channel 2 so they can be routed separately
        on = int(round(chord.start * TICKS_PER_BEAT))
        off = on + max(int(round(chord.duration * TICKS_PER_BEAT)), 1)
        for pitch in chord.pitches:
            timed.append((on, 1, bytes([0x91, pitch, velocity])))
            timed.append((off, 0, bytes([0x81, pitch, 0])))
    timed.sort(key=lambda item: (item[0], item[1]))

    tempo = int(round(60_000_000 / settings.tempo))
    track = bytearray(b"\x00\xff\x51\x03" + tempo.to_bytes(3, "big"))
    last = 0
    for tick, _, message in timed:
        track += _varlen(tick - last) + message
        last = tick
    end_tick = int(round(melody.total_beats * TICKS_PER_BEAT))
    track += _varlen(max(end_tick - last, 0)) + b"\xff\x2f\x00"

    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, TICKS_PER_BEAT)
    return header + b"MTrk" + struct.pack(">I", len(track)) + bytes(track)


def write_midi(path: str, melody: Melody, settings: MusicSettings) -> None:
    with open(path, "wb") as handle:
        handle.write(midi_bytes(melody, settings))
