"""Sliders generated from the equation's parameters.

The panel is rebuilt whenever the set of parameters or their inferred roles
changes. Values are carried across rebuilds when a parameter keeps its name
and role, so editing the equation does not lose your exploration.
"""

from __future__ import annotations

import random

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..core.expression import ParamSpec

_NO_EFFECT_TIP = (
    "Moving this parameter across its range does not change any note at present. "
    "Robust normalisation absorbs anything that only scales or shifts the whole "
    "equation, so the melody depends on shape rather than size."
)


class ParamRow(QWidget):
    valueChanged = Signal()
    RESOLUTION = 1000

    def __init__(self, spec: ParamSpec, value: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec

        name = QLabel(f"<b>{spec.name}</b>")
        role = QLabel(f"{spec.label}  [{_fmt(spec.minimum)} to {_fmt(spec.maximum)}]")
        role.setStyleSheet("color: gray;")
        role.setToolTip(spec.hint)
        name.setToolTip(spec.hint)
        self.effect = QLabel("")
        self.effect.setStyleSheet("color: #b26a00; font-style: italic;")
        self.effect.setToolTip(_NO_EFFECT_TIP)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        if spec.integer:
            self.slider.setRange(int(spec.minimum), int(spec.maximum))
        else:
            self.slider.setRange(0, self.RESOLUTION)

        span = spec.maximum - spec.minimum
        self.spin = QDoubleSpinBox()
        self.spin.setRange(-1e6, 1e6)  # typing may go beyond the slider range
        self.spin.setDecimals(0 if spec.integer else (4 if span < 1 else 3))
        self.spin.setSingleStep(1.0 if spec.integer else span / 100)
        self.spin.setKeyboardTracking(False)
        self.spin.setMinimumWidth(90)

        grid = QGridLayout(self)
        grid.setContentsMargins(0, 4, 0, 4)
        grid.addWidget(name, 0, 0)
        grid.addWidget(role, 0, 1)
        grid.addWidget(self.effect, 0, 2, Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self.slider, 1, 0, 1, 2)
        grid.addWidget(self.spin, 1, 2)
        grid.setColumnStretch(1, 1)

        self.set_value(value)
        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)

    def value(self) -> float:
        return float(self.spin.value())

    def set_value(self, value: float) -> None:
        self.spin.blockSignals(True)
        self.slider.blockSignals(True)
        self.spin.setValue(round(value) if self.spec.integer else value)
        self.slider.setValue(self._to_slider(value))
        self.spin.blockSignals(False)
        self.slider.blockSignals(False)

    def set_influence(self, has_effect: bool) -> None:
        self.effect.setText("" if has_effect else "no audible effect")

    def _to_slider(self, value: float) -> int:
        v = self.spec.clamp(value)
        if self.spec.integer:
            return int(round(v))
        span = self.spec.maximum - self.spec.minimum
        return int(round((v - self.spec.minimum) / span * self.RESOLUTION))

    def _from_slider(self, position: int) -> float:
        if self.spec.integer:
            return float(position)
        span = self.spec.maximum - self.spec.minimum
        return self.spec.minimum + span * position / self.RESOLUTION

    def _on_slider(self, position: int) -> None:
        self.spin.blockSignals(True)
        self.spin.setValue(self._from_slider(position))
        self.spin.blockSignals(False)
        self.valueChanged.emit()

    def _on_spin(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_slider(value))
        self.slider.blockSignals(False)
        self.valueChanged.emit()


class ParamPanel(QGroupBox):
    valuesChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Parameters", parent)
        self._rows: list[ParamRow] = []
        self._rng = random.Random()

        self.randomise_button = QPushButton("Randomise")
        self.reset_button = QPushButton("Reset")
        self.randomise_button.clicked.connect(self.randomise)
        self.reset_button.clicked.connect(self.reset)
        buttons = QHBoxLayout()
        buttons.addWidget(self.randomise_button)
        buttons.addWidget(self.reset_button)
        buttons.addStretch(1)

        self._empty = QLabel(
            "Parameters appear here once the equation contains letters other than n. "
            "Try sin(a*n) + b*cos(n/p)."
        )
        self._empty.setWordWrap(True)
        self._empty.setStyleSheet("color: gray;")

        self._rows_layout = QVBoxLayout()
        self._rows_layout.addWidget(self._empty)

        outer = QVBoxLayout(self)
        outer.addLayout(buttons)
        outer.addLayout(self._rows_layout)
        self._set_buttons_enabled(False)

    def snapshot(self) -> dict[str, tuple[float, str]]:
        return {row.spec.name: (row.value(), row.spec.role) for row in self._rows}

    def set_specs(self, specs: list[ParamSpec], previous: dict[str, tuple[float, str]]) -> None:
        for row in self._rows:
            self._rows_layout.removeWidget(row)
            row.deleteLater()
        self._rows = []
        for spec in specs:
            value = spec.default
            if spec.name in previous and previous[spec.name][1] == spec.role:
                value = previous[spec.name][0]
            row = ParamRow(spec, value)
            row.valueChanged.connect(self.valuesChanged.emit)
            self._rows_layout.addWidget(row)
            self._rows.append(row)
        self._empty.setVisible(not specs)
        self._set_buttons_enabled(bool(specs))

    def values(self) -> list[float]:
        return [row.value() for row in self._rows]

    def set_influence(self, flags: list[bool]) -> None:
        for row, flag in zip(self._rows, flags):
            row.set_influence(flag)

    def randomise(self) -> None:
        for row in self._rows:
            spec = row.spec
            if spec.integer:
                row.set_value(self._rng.randint(int(spec.minimum), int(spec.maximum)))
            else:
                row.set_value(self._rng.uniform(spec.minimum, spec.maximum))
        self.valuesChanged.emit()

    def reset(self) -> None:
        for row in self._rows:
            row.set_value(row.spec.default)
        self.valuesChanged.emit()

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self.randomise_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)


def _fmt(value: float) -> str:
    if abs(value - 6.283185307179586) < 1e-9:
        return "2π"
    return f"{value:g}"
