"""Musical settings that shape how the equation is heard, and the chords beneath it."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QWidget,
)

from ..core.harmony import CHORD_LENGTHS, NO_PROGRESSION, PROGRESSIONS
from ..core.melody import NORMALISATION_MODES, SCALES, midi_to_name
from ..core.synth import TIMBRES
from ..state import MusicSettings

NOTE_LENGTHS = (("Semiquaver (1/16)", 0.25), ("Quaver (1/8)", 0.5),
                ("Crotchet (1/4)", 1.0), ("Minim (1/2)", 2.0))


class SettingsPanel(QGroupBox):
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Musical settings", parent)
        d = MusicSettings()

        self.num_notes = _int_spin(4, 512, d.num_notes)
        self.n_start = _float_spin(-1e6, 1e6, d.n_start, 3, 1.0)
        self.n_step = _float_spin(0.001, 1000.0, d.n_step, 3, 0.1)

        self.root = QComboBox()
        for midi in range(36, 85):
            self.root.addItem(midi_to_name(midi), midi)
        self.root.setCurrentIndex(self.root.findData(d.root))

        self.scale = QComboBox()
        self.scale.addItems(list(SCALES))
        self.scale.setCurrentText(d.scale)
        self.octaves = _int_spin(1, 4, d.octaves)

        self.normalisation = QComboBox()
        self.normalisation.addItems(list(NORMALISATION_MODES))
        self.normalisation.setCurrentText(d.normalisation)
        self.normalisation.setToolTip(
            "Percentile clips extreme values. Median / MAD squashes them smoothly. "
            "Rank spreads pitches evenly whatever the distribution."
        )
        self.clip = _float_spin(0.0, 25.0, d.clip_percent, 1, 1.0)
        self.clip.setSuffix(" %")
        self.clip.setToolTip("Share of values at each extreme that are clipped.")

        self.tempo = _int_spin(30, 300, int(d.tempo))
        self.tempo.setSuffix(" bpm")
        self.note_length = QComboBox()
        for label, beats in NOTE_LENGTHS:
            self.note_length.addItem(label, beats)
        self.note_length.setCurrentIndex(self.note_length.findData(d.note_beats))
        self.gate = _float_spin(0.1, 1.0, d.gate, 2, 0.05)
        self.merge = QCheckBox("Sustain repeated notes")
        self.timbre = QComboBox()
        self.timbre.addItems(list(TIMBRES))
        self.timbre.setCurrentText(d.timbre)

        form = QFormLayout(self)
        form.addRow("Number of notes", self.num_notes)
        form.addRow("n starts at", self.n_start)
        form.addRow("n step", self.n_step)
        form.addRow("Root note", self.root)
        form.addRow("Scale", self.scale)
        form.addRow("Octave range", self.octaves)
        form.addRow("Normalisation", self.normalisation)
        form.addRow("Clip percentile", self.clip)
        form.addRow("Tempo", self.tempo)
        form.addRow("Note length", self.note_length)
        form.addRow("Gate", self.gate)
        form.addRow("", self.merge)
        form.addRow("Timbre", self.timbre)

        for spin in (self.num_notes, self.octaves, self.tempo):
            spin.valueChanged.connect(lambda *_: self.changed.emit())
        for spin in (self.n_start, self.n_step, self.clip, self.gate):
            spin.valueChanged.connect(lambda *_: self.changed.emit())
        for combo in (self.root, self.scale, self.normalisation, self.note_length, self.timbre):
            combo.currentIndexChanged.connect(lambda *_: self.changed.emit())
        self.merge.toggled.connect(lambda *_: self.changed.emit())
        self.normalisation.currentTextChanged.connect(self._update_clip_enabled)
        self._update_clip_enabled(self.normalisation.currentText())

    def settings(self) -> MusicSettings:
        return MusicSettings(
            num_notes=self.num_notes.value(),
            n_start=self.n_start.value(),
            n_step=self.n_step.value(),
            root=int(self.root.currentData()),
            scale=self.scale.currentText(),
            octaves=self.octaves.value(),
            normalisation=self.normalisation.currentText(),
            clip_percent=self.clip.value(),
            tempo=float(self.tempo.value()),
            note_beats=float(self.note_length.currentData()),
            gate=self.gate.value(),
            merge_repeats=self.merge.isChecked(),
            timbre=self.timbre.currentText(),
        )

    def _update_clip_enabled(self, mode: str) -> None:
        self.clip.setEnabled(mode.startswith("Percentile"))


class HarmonyPanel(QGroupBox):
    """Chord progression played as a sustained pad beneath the melody."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Harmony", parent)
        d = MusicSettings()

        self.progression = QComboBox()
        self.progression.addItems(list(PROGRESSIONS))
        self.progression.setCurrentText(d.progression)
        self.progression.setToolTip(
            "Chords are built from the selected scale, so each mode has its own chord qualities. "
            "The modulating circle of fifths changes key every chord while the melody stays put."
        )
        self.chord_length = QComboBox()
        for label, beats in CHORD_LENGTHS:
            self.chord_length.addItem(label, beats)
        self.chord_length.setCurrentIndex(self.chord_length.findData(d.chord_beats))
        self.chord_length.setToolTip("Bars are four beats long.")
        self.chord_octave = _int_spin(-3, 1, d.chord_octave)
        self.chord_octave.setToolTip("Chord register relative to the root note, in octaves.")
        self.chord_volume = _int_spin(0, 100, int(round(d.chord_volume * 100)))
        self.chord_volume.setSuffix(" %")
        self.chord_volume.setToolTip("Chord loudness relative to the melody.")

        form = QFormLayout(self)
        form.addRow("Progression", self.progression)
        form.addRow("Chord length", self.chord_length)
        form.addRow("Chord octave", self.chord_octave)
        form.addRow("Chord volume", self.chord_volume)

        for spin in (self.chord_octave, self.chord_volume):
            spin.valueChanged.connect(lambda *_: self.changed.emit())
        for combo in (self.progression, self.chord_length):
            combo.currentIndexChanged.connect(lambda *_: self.changed.emit())
        self.progression.currentTextChanged.connect(self._update_enabled)
        self._update_enabled(self.progression.currentText())

    def apply(self, settings: MusicSettings) -> MusicSettings:
        """Return a copy of the musical settings with the harmony settings filled in."""
        return replace(
            settings,
            progression=self.progression.currentText(),
            chord_beats=float(self.chord_length.currentData()),
            chord_octave=self.chord_octave.value(),
            chord_volume=self.chord_volume.value() / 100.0,
        )

    def _update_enabled(self, progression: str) -> None:
        active = progression != NO_PROGRESSION
        for widget in (self.chord_length, self.chord_octave, self.chord_volume):
            widget.setEnabled(active)


def _int_spin(lo: int, hi: int, value: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(lo, hi)
    spin.setValue(value)
    spin.setKeyboardTracking(False)
    return spin


def _float_spin(lo: float, hi: float, value: float, decimals: int, step: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setValue(value)
    spin.setKeyboardTracking(False)
    return spin
