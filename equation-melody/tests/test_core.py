"""Tests for the engine. These need no GUI or audio device. Run with: pytest"""

import math
import struct
from dataclasses import replace

import numpy as np
import pytest

from equation_melody.core.expression import EquationError, parse_equation
from equation_melody.core.export import midi_bytes
from equation_melody.core.harmony import (
    MODULATING, PROGRESSIONS, build_chords, loop_fit_message, triad_classes,
)
from equation_melody.core.melody import (
    NORMALISATION_MODES, REST, SCALES, build_melody, normalise, probe_influence, scale_pitches,
)
from equation_melody.core.synth import render_audio
from equation_melody.state import MusicSettings


def roles(text):
    return {p.name: p.role for p in parse_equation(text).params}


# Parsing and adaptive sliders ------------------------------------------------

def test_parameters_detected_and_index_excluded():
    parsed = parse_equation("sin(a*n) + b*sin(c*n)")
    assert parsed.index.name == "n"
    assert [p.name for p in parsed.params] == ["a", "b", "c"]


def test_roles_follow_position_in_equation():
    assert roles("sin(a*n) + b*sin(c*n)") == {"a": "frequency", "b": "multiplier", "c": "frequency"}
    assert roles("sin(n/p) + k*cos(n/q)") == {"k": "multiplier", "p": "period", "q": "period"}
    assert roles("sin(w*n + phi) * exp(-r*n)") == {"phi": "phase", "r": "rate", "w": "frequency"}
    assert roles("(a*n) % m") == {"a": "multiplier", "m": "modulus"}
    assert roles("r^n % 1") == {"r": "growth"}
    assert roles("n^k + c") == {"c": "offset", "k": "exponent"}
    assert roles("log(a*n + 1)") == {"a": "positive"}
    assert roles("sqrt(a*n)") == {"a": "positive"}
    assert roles("n / d") == {"d": "divisor"}


def test_modulus_is_integer_slider():
    spec = parse_equation("n^2 % m").params[0]
    assert spec.integer and spec.minimum >= 2


def test_multi_letter_names_are_kept_whole():
    assert [p.name for p in parse_equation("amp*sin(freq*n)").params] == ["amp", "freq"]


def test_implicit_multiplication_and_caret():
    parsed = parse_equation("2n^2 + 3sin n")
    assert float(parsed.func(np.array([2.0]))[0]) == pytest.approx(8 + 3 * math.sin(2))


def test_lhs_is_ignored():
    assert parse_equation("f(n) = sin(a*n)").params[0].name == "a"


@pytest.mark.parametrize("bad", [
    "", "__import__('os')", "n.real", "sin(a*n", "import os", "sine(n)", "amp(n+1)", "n; 2",
])
def test_rejects_unsafe_or_invalid(bad):
    with pytest.raises(EquationError):
        parse_equation(bad)


def test_single_letter_before_bracket_is_multiplication():
    parsed = parse_equation("a(n+1)")
    assert float(parsed.func(np.array([2.0]), 3.0)[0]) == pytest.approx(9.0)


def test_missing_index_gives_warning():
    assert parse_equation("a + b").warnings


# Robust normalisation ----------------------------------------------------------

@pytest.mark.parametrize("mode", NORMALISATION_MODES)
def test_normalised_values_lie_in_unit_interval(mode):
    y = np.concatenate([np.sin(np.arange(50)), [1e9, -1e9]])
    z = normalise(y, mode, 5.0).z
    assert np.all((z >= 0) & (z <= 1))


def test_percentile_resists_outliers():
    y = np.sin(np.arange(64) * 0.4)
    y_spiked = y.copy()
    y_spiked[10] = 1e12
    z_clean = normalise(y, "Percentile (robust)", 5.0).z
    z_spiked = normalise(y_spiked, "Percentile (robust)", 5.0).z
    others = np.arange(64) != 10
    assert np.corrcoef(z_clean[others], z_spiked[others])[0, 1] > 0.99


@pytest.mark.parametrize("mode", NORMALISATION_MODES)
def test_nan_and_inf_become_undefined(mode):
    y = np.array([0.0, np.nan, 1.0, np.inf, -np.inf, 2.0])
    norm = normalise(y, mode, 5.0)
    assert norm.undefined == 3
    assert np.isnan(norm.z[[1, 3, 4]]).all()


@pytest.mark.parametrize("mode", NORMALISATION_MODES)
def test_constant_input_is_flat(mode):
    norm = normalise(np.full(20, 3.7), mode, 5.0)
    assert norm.flat and np.allclose(norm.z, 0.5)


@pytest.mark.parametrize("mode", NORMALISATION_MODES)
def test_extreme_magnitudes_do_not_overflow(mode):
    y = np.array([1e308, -1e308, 5e307, 0.0])
    z = normalise(y, mode, 0.0).z
    assert np.isfinite(z).all()


def test_mad_falls_back_when_most_values_equal():
    y = np.zeros(40)
    y[[5, 20]] = [1.0, -1.0]
    norm = normalise(y, "Median / MAD", 5.0)
    assert not norm.flat and norm.z[5] > 0.5 > norm.z[20]


def test_all_undefined_is_handled():
    norm = normalise(np.full(8, np.nan), "Percentile (robust)", 5.0)
    assert norm.undefined == 8 and norm.flat


# Melody -----------------------------------------------------------------------

def test_notes_stay_in_scale_and_range():
    settings = MusicSettings(num_notes=64, scale="Dorian", octaves=3)
    parsed = parse_equation("sin(a*n) + b*sin(c*n)")
    melody = build_melody(parsed.func, [0.7, 0.5, 2.1], settings)
    allowed = set(scale_pitches(settings))
    assert {e.pitch for e in melody.events} <= allowed


def test_undefined_values_become_rests():
    parsed = parse_equation("log(n - 5)")
    melody = build_melody(parsed.func, [], MusicSettings(num_notes=12))
    assert (melody.pitches[:6] == REST).all() and (melody.pitches[6:] != REST).all()


def test_division_by_zero_parameter_does_not_raise():
    parsed = parse_equation("1/d + sin(n)")
    build_melody(parsed.func, [0.0], MusicSettings())


def test_merge_repeats_sustains_notes():
    parsed = parse_equation("floor(n/4)")
    merged = build_melody(parsed.func, [], MusicSettings(num_notes=16, merge_repeats=True))
    assert len(merged.events) == 4 and merged.events[0].duration == pytest.approx(2.0)


def test_probe_flags_parameters_absorbed_by_normalisation():
    parsed = parse_equation("a*sin(w*n) + c")
    names = [p.name for p in parsed.params]  # a, c, w
    values = [2.0, 1.0, 0.5]
    settings = MusicSettings()
    melody = build_melody(parsed.func, values, settings)
    ranges = [(0.5, 5.0), (-10.0, 10.0), (0.01, 3.0)]  # positive scale only, for this check
    flags = dict(zip(names, probe_influence(parsed.func, values, ranges, settings, melody.pitches)))
    assert flags == {"a": False, "c": False, "w": True}


def test_only_seven_note_scales_remain():
    assert list(SCALES) == ["Major", "Natural minor", "Harmonic minor", "Dorian", "Mixolydian"]
    assert all(len(intervals) == 7 for intervals in SCALES.values())
    assert MusicSettings().scale == "Major"


# Harmony ----------------------------------------------------------------------

IN_KEY = [name for name in PROGRESSIONS if name not in ("None", MODULATING)]


def test_no_progression_adds_nothing():
    settings = MusicSettings()
    melody = build_melody(parse_equation("sin(n)").func, [], settings)
    assert settings.progression == "None" and melody.chords == []
    louder = MusicSettings(chord_volume=1.0, chord_octave=0, chord_beats=2.0)
    other = build_melody(parse_equation("sin(n)").func, [], louder)
    assert np.array_equal(render_audio(melody, settings), render_audio(other, louder))
    assert midi_bytes(melody, settings) == midi_bytes(other, louder)


@pytest.mark.parametrize("scale", list(SCALES))
@pytest.mark.parametrize("progression", IN_KEY)
def test_in_key_progressions_use_only_scale_notes(progression, scale):
    settings = MusicSettings(progression=progression, scale=scale, root=62, num_notes=64)
    classes = {(settings.root + i) % 12 for i in SCALES[scale]}
    chords = build_chords(settings, 32.0)
    assert chords and all(len(c.pitches) == 3 for c in chords)
    assert {p % 12 for c in chords for p in c.pitches} <= classes


def test_triads_follow_the_scale():
    def quality(scale, degree):
        settings = MusicSettings(progression="I–V–vi–IV", scale=scale, root=60)
        root, third, fifth = (settings.root + SCALES[scale][(degree + k) % 7] for k in (0, 2, 4))
        return ((third - root) % 12, (fifth - root) % 12)
    assert quality("Major", 4) == (4, 7)  # V major
    assert quality("Natural minor", 4) == (3, 7)  # v minor
    assert quality("Harmonic minor", 4) == (4, 7)  # V major
    assert quality("Harmonic minor", 2) == (4, 8)  # III augmented
    assert quality("Dorian", 3) == (4, 7)  # IV major
    settings = MusicSettings(progression="I–IV–V–I", scale="Harmonic minor", root=60)
    assert set(triad_classes(settings, 2)) == {7, 11, 2}  # G B D


def test_modulating_circle_roots_rise_by_fifths():
    settings = MusicSettings(progression=MODULATING, root=60)
    roots = [triad_classes(settings, i)[0] for i in range(13)]
    assert all((b - a) % 12 == 7 for a, b in zip(roots, roots[1:]))
    assert len(set(roots)) == 12
    for i in range(12):
        root, third, fifth = triad_classes(settings, i)
        assert ((third - root) % 12, (fifth - root) % 12) == (4, 7)


@pytest.mark.parametrize("progression", IN_KEY + [MODULATING])
def test_voice_leading_moves_in_small_steps_and_stays_in_register(progression):
    settings = MusicSettings(progression=progression, num_notes=192)
    chords = build_chords(settings, 96.0)
    centre = settings.root + 12 * settings.chord_octave
    for previous, chord in zip(chords, chords[1:]):  # no voice moves more than a major third
        assert max(abs(a - b) for a, b in zip(previous.pitches, chord.pitches)) <= 4
    assert all(centre - 6 <= c.pitches[0] < centre + 6 for c in chords)


def test_chords_fill_the_loop_and_the_last_is_cut_short():
    settings = MusicSettings(progression="ii–V–I", num_notes=30, note_beats=0.5,
                             chord_beats=4.0)
    chords = build_chords(settings, 15.0)
    assert [c.start for c in chords] == [0.0, 4.0, 8.0, 12.0]
    assert chords[-1].duration == pytest.approx(3.0)
    assert "multiple of 8 notes" in loop_fit_message(settings)
    assert loop_fit_message(MusicSettings(progression="ii–V–I", num_notes=32)) is None
    assert loop_fit_message(MusicSettings(num_notes=30)) is None


# Audio and export ---------------------------------------------------------------

def test_audio_length_matches_loop_and_is_bounded():
    settings = MusicSettings(num_notes=16, tempo=120, note_beats=0.5)
    parsed = parse_equation("sin(n)")
    melody = build_melody(parsed.func, [], settings)
    audio = render_audio(melody, settings, 44100)
    assert len(audio) == int(math.ceil(16 * 0.5 * 0.5 * 44100))
    assert audio.dtype == np.float32 and np.max(np.abs(audio)) <= 0.8 + 1e-6


def test_midi_file_structure():
    settings = MusicSettings(num_notes=8)
    melody = build_melody(parse_equation("n % 5").func, [], settings)
    data = midi_bytes(melody, settings)
    assert data[:4] == b"MThd" and data[14:18] == b"MTrk"
    assert struct.unpack(">I", data[18:22])[0] == len(data) - 22
    assert data.endswith(b"\xff\x2f\x00")


@pytest.mark.parametrize("volume", [0.0, 0.4, 1.0])
def test_audio_with_chords_is_bounded_and_melody_stays_on_top(volume):
    settings = MusicSettings(num_notes=16, progression="I–V–vi–IV", chord_volume=volume,
                             chord_beats=2.0)
    melody = build_melody(parse_equation("sin(n)").func, [], settings)
    audio = render_audio(melody, settings, 44100)
    assert len(audio) == int(math.ceil(16 * 0.5 * 0.5 * 44100))
    assert np.max(np.abs(audio)) <= 0.8 + 1e-6
    chords_only = build_melody(parse_equation("log(-1 - n)").func, [], settings)
    assert not chords_only.events
    pad = render_audio(chords_only, replace(settings, chord_volume=1.0), 44100)
    assert np.all(np.isfinite(pad)) and len(pad) == len(audio)


def test_short_loop_with_long_pad_release_wraps_cleanly():
    settings = MusicSettings(num_notes=4, note_beats=0.25, tempo=300, progression="Drone (I)",
                             chord_beats=2.0)
    melody = build_melody(parse_equation("sin(n)").func, [], settings)
    audio = render_audio(melody, settings, 44100)
    assert len(audio) == int(math.ceil(4 * 0.25 * 0.2 * 44100))
    assert np.max(np.abs(audio)) <= 0.8 + 1e-6


def _midi_messages(data):
    """Decode the note messages of the single track written by midi_bytes."""
    track, i, messages = data[22:], 7, []  # skip the tempo event
    while i < len(track):
        while track[i] & 0x80:  # delta time
            i += 1
        i += 1
        if track[i] == 0xFF:
            break
        messages.append(tuple(track[i:i + 3]))
        i += 3
    return messages


def test_midi_contains_chords_on_channel_two():
    settings = MusicSettings(num_notes=16, progression="I–IV–V–I")
    melody = build_melody(parse_equation("n % 5").func, [], settings)
    data = midi_bytes(melody, settings)
    assert struct.unpack(">I", data[18:22])[0] == len(data) - 22
    messages = _midi_messages(data)
    chord_ons = [m for m in messages if m[0] == 0x91]
    assert len(chord_ons) == 3 * len(melody.chords) == 6
    assert len([m for m in messages if m[0] == 0x81]) == len(chord_ons)
    assert len([m for m in messages if m[0] == 0x90]) == len(melody.events)
    assert sorted(m[1] for m in chord_ons) == sorted(p for c in melody.chords for p in c.pitches)
