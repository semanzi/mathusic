"""Looping audio playback whose buffer can be swapped while playing.

A persistent output stream reads from the current buffer in a callback. When
parameters change, the new buffer replaces the old one at the same position,
so the melody updates live without restarting.
"""

from __future__ import annotations

import threading

import numpy as np

try:
    import sounddevice as sd

    AUDIO_ERROR: str | None = None
except Exception as exc:  # PortAudio missing or package not installed
    sd = None
    AUDIO_ERROR = f"Audio playback is unavailable ({exc})."


class Player:
    def __init__(self, sample_rate: int = 44100) -> None:
        self.sample_rate = sample_rate
        self.loop = True
        self._buffer = np.zeros(0, dtype=np.float32)
        self._position = 0
        self._playing = False
        self._lock = threading.Lock()
        self._stream = None

    @property
    def available(self) -> bool:
        return sd is not None

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def position_seconds(self) -> float:
        return self._position / self.sample_rate

    def set_buffer(self, buffer: np.ndarray) -> None:
        data = np.ascontiguousarray(buffer, dtype=np.float32)
        with self._lock:
            self._buffer = data
            self._position = self._position % len(data) if len(data) else 0

    def play(self) -> None:
        if not self.available:
            raise RuntimeError(AUDIO_ERROR or "Audio playback is unavailable.")
        self._ensure_stream()
        with self._lock:
            self._position = 0
            self._playing = True

    def stop(self) -> None:
        with self._lock:
            self._playing = False
            self._position = 0

    def close(self) -> None:
        self.stop()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def _ensure_stream(self) -> None:
        if self._stream is None:
            self._stream = sd.OutputStream(samplerate=self.sample_rate, channels=1,
                                           dtype="float32", blocksize=1024,
                                           callback=self._callback)
            self._stream.start()

    def _callback(self, outdata, frames, time_info, status) -> None:  # audio thread
        out = outdata[:, 0]
        out.fill(0.0)
        with self._lock:
            buffer = self._buffer
            if not self._playing or len(buffer) == 0:
                return
            filled = 0
            while filled < frames:
                take = min(frames - filled, len(buffer) - self._position)
                out[filled:filled + take] = buffer[self._position:self._position + take]
                filled += take
                self._position += take
                if self._position >= len(buffer):
                    self._position = 0
                    if not self.loop:
                        self._playing = False
                        break
