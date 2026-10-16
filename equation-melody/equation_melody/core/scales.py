"""Seven-note scales shared by the melody and the harmony.

Every scale has seven degrees, so triads can be built by stacking thirds
within the scale and each mode gives its own chord qualities.
"""

from __future__ import annotations

SCALES: dict[str, tuple[int, ...]] = {
    "Major": (0, 2, 4, 5, 7, 9, 11),
    "Natural minor": (0, 2, 3, 5, 7, 8, 10),
    "Harmonic minor": (0, 2, 3, 5, 7, 8, 11),
    "Dorian": (0, 2, 3, 5, 7, 9, 10),
    "Mixolydian": (0, 2, 4, 5, 7, 9, 10),
}
