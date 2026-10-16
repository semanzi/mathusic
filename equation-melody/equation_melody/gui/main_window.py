"""Main window. The equation is the primary input and everything else follows from it."""

from __future__ import annotations

import sys

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..audio.player import AUDIO_ERROR, Player
from ..core.export import write_midi, write_wav
from ..core.expression import SUPPORTED_SUMMARY, EquationError, ParsedEquation, parse_equation
from ..core.harmony import loop_fit_message
from ..core.melody import Melody, build_melody, probe_influence
from ..core.synth import render_audio
from .param_panel import ParamPanel
from .plots import PlotArea
from .settings_panel import HarmonyPanel, SettingsPanel

SAMPLE_RATE = 44100

EXAMPLES = (
    ("Two interfering waves", "sin(a*n) + b*sin(c*n)"),
    ("Slow and fast periods", "sin(n/p) + k*cos(n/q)"),
    ("Modular staircase", "(a*n) % m"),
    ("Square residues", "n^2 % m"),
    ("Decaying oscillation", "sin(w*n + phi) * exp(-r*n)"),
    ("Geometric wrap", "r^n % 1"),
    ("Folded tangent", "tan(w*n) / (1 + abs(tan(w*n)))"),
    ("Root envelope", "sqrt(a*n) * cos(w*n)"),
    ("Stepped wave", "floor(h*sin(w*n)) + sin(n/p)"),
)

_STATUS_COLOURS = {"ok": "", "warn": "color: #b26a00;", "error": "color: #c62828;"}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Equation Melody Explorer")
        self.player = Player(SAMPLE_RATE)
        self.parsed: ParsedEquation | None = None
        self.melody: Melody | None = None
        self.audio = np.zeros(0, dtype=np.float32)

        self._parse_timer = _single_shot(self, 350, self._parse)
        self._render_timer = _single_shot(self, 60, self._render)
        self._playhead_timer = QTimer(self)
        self._playhead_timer.setInterval(30)
        self._playhead_timer.timeout.connect(self._update_playhead)

        self._build_ui()
        self.equation_edit.setText(EXAMPLES[0][1])
        self._parse()

    # Layout ---------------------------------------------------------------

    def _build_ui(self) -> None:
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        mono.setPointSize(16)

        prefix = QLabel("f(n) =")
        prefix.setFont(mono)
        self.equation_edit = QLineEdit()
        self.equation_edit.setFont(mono)
        self.equation_edit.setPlaceholderText("Type an equation in n, for example sin(a*n) + b*cos(n/p)")
        self.equation_edit.setClearButtonEnabled(True)

        self.examples = QComboBox()
        self.examples.addItem("Examples")
        for name, equation in EXAMPLES:
            self.examples.addItem(f"{name}:  {equation}", equation)

        equation_row = QHBoxLayout()
        equation_row.addWidget(prefix)
        equation_row.addWidget(self.equation_edit, 1)
        equation_row.addWidget(self.examples)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        help_text = QLabel(
            "Use n as the step index. Every other letter becomes a parameter with its own slider. "
            f"Use ^ for powers and % for remainders. Functions available are {SUPPORTED_SUMMARY}."
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("color: gray;")

        self.param_panel = ParamPanel()
        self.settings_panel = SettingsPanel()
        self.harmony_panel = HarmonyPanel()
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.param_panel)
        left_layout.addWidget(self.settings_panel)
        left_layout.addWidget(self.harmony_panel)
        left_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setMinimumWidth(380)

        self.plots = PlotArea()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(self.plots)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 900])

        self.play_button = QPushButton("Play")
        self.play_button.setMinimumWidth(90)
        self.loop_box = QCheckBox("Loop")
        self.loop_box.setChecked(True)
        self.info = QLabel()
        export_midi = QPushButton("Export MIDI")
        export_wav = QPushButton("Export WAV")
        transport = QHBoxLayout()
        transport.addWidget(self.play_button)
        transport.addWidget(self.loop_box)
        transport.addSpacing(16)
        transport.addWidget(self.info, 1)
        transport.addWidget(export_midi)
        transport.addWidget(export_wav)

        central = QWidget()
        root = QVBoxLayout(central)
        root.addLayout(equation_row)
        root.addWidget(self.status)
        root.addWidget(help_text)
        root.addWidget(splitter, 1)
        root.addLayout(transport)
        self.setCentralWidget(central)

        self.equation_edit.textEdited.connect(lambda _text: self._parse_timer.start())
        self.equation_edit.returnPressed.connect(self._parse)
        self.examples.activated.connect(self._on_example)
        self.param_panel.valuesChanged.connect(lambda: self._render_timer.start())
        self.settings_panel.changed.connect(lambda: self._render_timer.start())
        self.harmony_panel.changed.connect(lambda: self._render_timer.start())
        self.play_button.clicked.connect(self._toggle_play)
        self.loop_box.toggled.connect(self._on_loop)
        export_midi.clicked.connect(self._export_midi)
        export_wav.clicked.connect(self._export_wav)

        if not self.player.available:
            self.play_button.setEnabled(False)
            self.play_button.setToolTip(AUDIO_ERROR or "")

    # Equation -------------------------------------------------------------

    def _on_example(self, index: int) -> None:
        if index > 0:
            self.equation_edit.setText(self.examples.itemData(index))
            self._parse()
        self.examples.setCurrentIndex(0)

    def _parse(self) -> None:
        self._parse_timer.stop()
        try:
            parsed = parse_equation(self.equation_edit.text())
        except EquationError as exc:
            # Keep the last valid melody playing so a typo does not interrupt exploration.
            self._set_status(str(exc), "error")
            return
        if self.parsed is None or parsed.signature != self.parsed.signature:
            self.param_panel.set_specs(parsed.params, self.param_panel.snapshot())
        self.parsed = parsed
        self._render()

    # Rendering ------------------------------------------------------------

    def _render(self) -> None:
        self._render_timer.stop()
        if self.parsed is None:
            return
        settings = self.harmony_panel.apply(self.settings_panel.settings())
        values = self.param_panel.values()
        melody = build_melody(self.parsed.func, values, settings)
        audio = render_audio(melody, settings, SAMPLE_RATE)
        self.melody, self.audio = melody, audio

        self.plots.update_views(melody, audio, SAMPLE_RATE)
        self.player.set_buffer(audio)
        ranges = [(p.minimum, p.maximum) for p in self.parsed.params]
        self.param_panel.set_influence(
            probe_influence(self.parsed.func, values, ranges, settings, melody.pitches))
        chords = f", {len(melody.chords)} chords" if melody.chords else ""
        seconds = len(audio) / SAMPLE_RATE
        self.info.setText(f"{len(melody.events)} notes{chords}, {seconds:.1f} seconds")
        self._report(melody)

    def _report(self, melody: Melody) -> None:
        assert self.parsed is not None
        names = ", ".join(p.name for p in self.parsed.params) or "none"
        messages = [f"Index {self.parsed.index.name}. Parameters {names}."]
        level = "ok"
        if self.parsed.warnings:
            messages.extend(self.parsed.warnings)
            level = "warn"
        norm = melody.normalisation
        total = len(melody.y)
        if norm.undefined == total:
            messages.append("The equation is undefined at every step, so there are no notes. "
                            "Try other parameter values or a different n range.")
            level = "warn"
        elif norm.undefined:
            messages.append(f"{norm.undefined} of {total} steps are undefined and play as rests.")
            level = "warn"
        elif norm.flat:
            messages.append("Every step gives the same value, so the melody is one repeated note.")
            level = "warn"
        fit = loop_fit_message(melody.settings)
        if fit:
            messages.append(fit)
            level = "warn"
        self._set_status(" ".join(messages), level)

    def _set_status(self, text: str, level: str) -> None:
        self.status.setStyleSheet(_STATUS_COLOURS[level])
        self.status.setText(text)

    # Transport ------------------------------------------------------------

    def _toggle_play(self) -> None:
        if self.player.playing:
            self.player.stop()
            self._playhead_timer.stop()
            self.play_button.setText("Play")
            self.plots.set_playhead(0.0, 0.0)
            return
        try:
            self.player.play()
        except Exception as exc:
            QMessageBox.warning(self, "Playback", str(exc))
            return
        self.play_button.setText("Stop")
        self._playhead_timer.start()

    def _on_loop(self, checked: bool) -> None:
        self.player.loop = checked

    def _update_playhead(self) -> None:
        if not self.player.playing:
            self._playhead_timer.stop()
            self.play_button.setText("Play")
            self.plots.set_playhead(0.0, 0.0)
            return
        if self.melody is None:
            return
        seconds = self.player.position_seconds
        self.plots.set_playhead(seconds / self.melody.settings.seconds_per_beat, seconds)

    def _export_midi(self) -> None:
        if not self.melody or not self.melody.events:
            QMessageBox.information(self, "Export MIDI", "There are no notes to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export MIDI", "melody.mid", "MIDI files (*.mid)")
        if path:
            write_midi(path, self.melody, self.melody.settings)

    def _export_wav(self) -> None:
        if len(self.audio) == 0:
            QMessageBox.information(self, "Export WAV", "There is no audio to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export WAV", "melody.wav", "WAV files (*.wav)")
        if path:
            write_wav(path, self.audio, SAMPLE_RATE)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self.player.close()
        super().closeEvent(event)


def _single_shot(parent: QWidget, interval_ms: int, slot) -> QTimer:
    timer = QTimer(parent)
    timer.setSingleShot(True)
    timer.setInterval(interval_ms)
    timer.timeout.connect(slot)
    return timer


def run() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    window = MainWindow()
    window.resize(1360, 880)
    window.show()
    sys.exit(app.exec())
