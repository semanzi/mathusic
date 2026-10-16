"""The three views: sampled equation, piano roll and rendered waveform."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..core.melody import Melody, midi_to_name, scale_pitches

_ACCENT = (52, 120, 200)
_BAND = (200, 120, 40)
_REST = (200, 60, 60)
_PLAYHEAD = (220, 50, 50)


class PlotArea(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.function_plot = pg.PlotWidget(title="Equation sampled at each step")
        self.function_plot.setLabel("bottom", "n")
        self.function_plot.setLabel("left", "f(n)")
        self.function_plot.showGrid(x=True, y=True, alpha=0.2)
        self.curve = self.function_plot.plot([], [], pen=pg.mkPen(_ACCENT, width=2))
        self.points = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(*_ACCENT), pen=None)
        self.rests = pg.ScatterPlotItem(size=9, symbol="x", brush=pg.mkBrush(*_REST),
                                        pen=pg.mkPen(_REST))
        self.function_plot.addItem(self.points)
        self.function_plot.addItem(self.rests)
        band_pen = pg.mkPen(_BAND, width=1, style=Qt.PenStyle.DashLine)
        self.low_line = pg.InfiniteLine(angle=0, pen=band_pen, movable=False)
        self.high_line = pg.InfiniteLine(angle=0, pen=band_pen, movable=False)
        self.function_plot.addItem(self.low_line)
        self.function_plot.addItem(self.high_line)

        self.roll_plot = pg.PlotWidget(title="Melody quantised to the scale")
        self.roll_plot.setLabel("bottom", "beats")
        self.roll_plot.showGrid(x=True, y=True, alpha=0.2)
        self._bars: pg.BarGraphItem | None = None
        self.roll_playhead = pg.InfiniteLine(angle=90, pen=pg.mkPen(_PLAYHEAD, width=2),
                                             movable=False)
        self.roll_plot.addItem(self.roll_playhead)

        self.wave_plot = pg.PlotWidget(title="Rendered waveform")
        self.wave_plot.setLabel("bottom", "seconds")
        self.wave_plot.setDownsampling(auto=True, mode="peak")
        self.wave_plot.setClipToView(True)
        self.wave_curve = self.wave_plot.plot([], [], pen=pg.mkPen(_ACCENT, width=1))
        self.wave_playhead = pg.InfiniteLine(angle=90, pen=pg.mkPen(_PLAYHEAD, width=2),
                                             movable=False)
        self.wave_plot.addItem(self.wave_playhead)

        layout.addWidget(self.function_plot, 3)
        layout.addWidget(self.roll_plot, 4)
        layout.addWidget(self.wave_plot, 2)

    def update_views(self, melody: Melody, audio: np.ndarray, sample_rate: int) -> None:
        self._update_function(melody)
        self._update_roll(melody)
        seconds = np.arange(len(audio)) / sample_rate
        self.wave_curve.setData(seconds, audio)
        self.wave_plot.setXRange(0, max(seconds[-1] if len(seconds) else 1.0, 1e-3), padding=0.01)
        self.wave_plot.setYRange(-1, 1, padding=0.05)

    def set_playhead(self, beats: float, seconds: float) -> None:
        self.roll_playhead.setPos(beats)
        self.wave_playhead.setPos(seconds)

    def _update_function(self, melody: Melody) -> None:
        n, y = melody.n, melody.y
        finite = np.isfinite(y)
        self.curve.setData(n, y, connect="finite")
        self.points.setData(n[finite], y[finite])
        norm = melody.normalisation
        baseline = norm.low if np.isfinite(norm.low) else 0.0
        self.rests.setData(n[~finite], np.full(int((~finite).sum()), baseline))
        bands_valid = np.isfinite(norm.low) and np.isfinite(norm.high) and not norm.flat
        for line, value in ((self.low_line, norm.low), (self.high_line, norm.high)):
            line.setVisible(bool(bands_valid))
            if bands_valid:
                line.setPos(value)
        self.function_plot.enableAutoRange()

    def _update_roll(self, melody: Melody) -> None:
        if self._bars is not None:
            self.roll_plot.removeItem(self._bars)
            self._bars = None
        events = melody.events
        if events:
            self._bars = pg.BarGraphItem(
                x0=[e.start for e in events],
                width=[e.duration * melody.settings.gate for e in events],
                y0=[e.pitch - 0.4 for e in events],
                height=[0.8] * len(events),
                brush=pg.mkBrush(*_ACCENT, 220),
                pen=pg.mkPen(None),
            )
            self.roll_plot.addItem(self._bars)
        pitches = scale_pitches(melody.settings)
        self.roll_plot.getAxis("left").setTicks([[(p, midi_to_name(p)) for p in pitches]])
        self.roll_plot.setYRange(pitches[0] - 1, pitches[-1] + 1, padding=0)
        self.roll_plot.setXRange(0, max(melody.total_beats, 1e-3), padding=0.01)
