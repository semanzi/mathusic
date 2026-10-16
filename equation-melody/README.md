# Equation Melody Explorer

A desktop application for discovering the melody hidden in a mathematical equation. You type an equation in the step index `n`, and the application samples it once per note, normalises the values robustly, snaps them to a musical scale and plays the result. Every other letter in the equation becomes a parameter with its own slider, and the slider ranges adapt to the role each letter plays in the equation.

## Requirements

The application needs Python 3.10 or later and Visual Studio Code with Microsoft's Python extension installed. On Windows and macOS the audio library ships with everything it needs. On Linux the PortAudio system library must be installed first.

```bash
sudo apt install libportaudio2      # Debian and Ubuntu
sudo dnf install portaudio          # Fedora
```

## Setting up in VS Code

1. Unzip the project and open the `equation-melody` folder in VS Code using File, then Open Folder.
2. Open the Command Palette (Ctrl+Shift+P, or Cmd+Shift+P on macOS) and choose **Python: Create Environment**, then **Venv**, then your Python interpreter. When VS Code offers to install from `requirements.txt`, accept.
3. If you prefer the terminal, the equivalent commands are below. Run them from the integrated terminal in the project folder.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS and Linux
source .venv/bin/activate
pip install -r requirements.txt
```

4. Confirm that the status bar at the bottom right of VS Code shows the `.venv` interpreter. If it does not, use **Python: Select Interpreter** from the Command Palette.

## Running the application

Press F5, or open the Run and Debug panel and choose **Run Equation Melody**. The launch configuration in `.vscode/launch.json` starts `app.py` in the integrated terminal. Opening `app.py` and pressing the Run button in the editor's top right corner also works. From a terminal, run `python app.py`.

To run the engine tests, open the Testing panel (the flask icon) or run `pytest` in the terminal. The tests cover parsing, slider role inference, every normalisation mode, rests, chord progressions and voice leading, synthesis and MIDI export, and need neither a display nor an audio device.

## Using the application

The equation box at the top is the main input. Type an equation and pause briefly, or press Enter, and the application parses it, builds the sliders and redraws everything. If you make a typing error, the last valid melody keeps playing and the status line explains the problem in red. The Examples menu offers nine starting points.

Write `n` for the step index. Implicit multiplication works, so `2n` and `3 sin n` are accepted. A single letter before a bracket is read as multiplication, so `a(n+1)` means `a*(n+1)`. Use `^` for powers and `%` for remainders. The supported functions are sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, exp, log, sqrt, abs, floor, ceil, sign and mod, together with the constants pi and e. Parameter names can be longer than one letter, so `amp*sin(freq*n)` gives sliders called `amp` and `freq`.

Press Play to hear the melody. With Loop ticked, changes to the equation, sliders or settings are heard immediately without restarting playback. Export MIDI writes the notes for use in a digital audio workstation, and Export WAV writes the audio exactly as heard.

### How the sliders adapt

Each parameter is classified by its position in the equation, and the classification sets the slider's range, default value and whether it moves in whole numbers. Hovering over a parameter's name shows a short explanation of its role.

| Role | Example | Range | Default |
|---|---|---|---|
| Modulus | `m` in `n^2 % m` | 2 to 24, whole numbers | 7 |
| Period | `p` in `sin(n/p)` | 1 to 64 | 8 |
| Frequency | `w` in `sin(w*n)` | 0.01 to 3 | 0.5 |
| Phase | `phi` in `sin(n + phi)` | 0 to 2π | 0 |
| Growth base | `r` in `r^n` | 0.8 to 1.25 | 1.05 |
| Rate | `r` in `exp(-r*n)` | −0.3 to 0.3 | 0.05 |
| Exponent | `k` in `n^k` | 0 to 4 | 2 |
| Divisor | `d` in `n/d` | 0.1 to 20 | 4 |
| Positive | `a` in `log(a*n)` or `sqrt(a*n)` | 0.1 to 10 | 1 |
| Multiplier | `b` in `b*sin(n)` | −5 to 5 | 1 |
| Offset | `c` in `sin(n) + c` | −10 to 10 | 0 |

When a letter appears in several places, the most specific role wins. Values typed into the number box may go beyond the slider's range. A parameter keeps its value when you edit the equation, provided its name and role are unchanged. Randomise picks fresh values within every range, which is a quick way to explore.

A parameter marked **no audible effect** does not change any note when moved across its range. This is a consequence of robust normalisation, which deliberately discards the overall size and position of the equation's output so that the melody reflects its shape. A multiplier on the whole equation, or a constant added to it, is absorbed in this way. A multiplier on one term among several is not, because it changes the balance between terms.

### Musical settings

These settings sit below the parameters and shape how the equation is heard. Number of notes, the starting value of `n` and the step size between notes together decide which stretch of the equation becomes the melody. A step size of 0.1 reveals fine detail in a slowly changing curve, while a large step can turn a smooth wave into something more angular through sampling effects. Root note, scale and octave range decide the available pitches. Tempo, note length and gate decide timing, and sustaining repeated notes lets rhythm emerge from plateaux in the equation.

Every scale has seven notes: Major, Natural minor, Harmonic minor, Dorian and Mixolydian.

### Harmony

The Harmony group adds a chord progression beneath the melody, played as a soft sustained pad. The default, None, adds nothing. The progressions on offer are a drone on the root, I–V–vi–IV, I–IV–V–I, ii–V–I, the circle of fifths within the key (I–IV–vii°–iii–vi–ii–V–I) and a modulating circle of fifths.

Chords are built by stacking thirds within the selected scale, so each scale gives its own chord qualities. In harmonic minor, for example, V is major and III is augmented, while in Dorian IV is major. The modulating circle instead plays major chords on roots rising by fifths (C, G, D, A and so on through all twelve keys). The melody stays in the original key, so clashes are part of the effect. Each chord is voiced as the inversion nearest the previous one, so no voice moves by more than a major third.

Everything is in 4/4 time. Chords last half a bar, one bar or two bars, and the progression repeats for the length of the loop. When the loop does not hold a whole number of chords, the last chord is cut short and the status line suggests a note count that fits. Chord octave sets the register relative to the root note, and chord volume sets the pad's level relative to the melody. The piano roll shows the chords as pale bars behind the melody, and MIDI export writes them on channel 2.

### Normalisation modes

All three modes first divide by the largest magnitude, so equations producing extremely large or small numbers behave like any other. Values that are undefined, such as a logarithm of a negative number or a division by zero, become rests and are marked with a red cross on the top plot.

**Percentile (robust)** is the default. It maps the chosen lower percentile to the lowest note and the upper percentile to the highest, clipping anything beyond, so that a few extreme spikes cannot squash the rest of the melody into a single note. The dashed lines on the top plot show these bounds.

**Median / MAD** centres the melody on the median and compresses extremes smoothly with a hyperbolic tangent scaled by the median absolute deviation. Where most values are identical, it falls back to the interquartile range and then the standard deviation.

**Rank (quantile)** ignores distances altogether and orders the distinct values, spreading pitches evenly. It suits equations whose values bunch tightly with occasional large departures.

## Project structure

```
equation-melody/
├── app.py                     entry point
├── requirements.txt
├── .vscode/                   launch and test configuration
├── equation_melody/
│   ├── state.py               musical settings
│   ├── core/
│   │   ├── expression.py      safe parsing and slider role inference
│   │   ├── scales.py          seven-note scales
│   │   ├── melody.py          sampling, normalisation, quantisation, notes
│   │   ├── harmony.py         chord progressions and voice leading
│   │   ├── synth.py           additive oscillators and envelopes
│   │   └── export.py          WAV and MIDI writers (melody on channel 1, chords on channel 2)
│   ├── audio/
│   │   └── player.py          looping playback with live buffer swapping
│   └── gui/
│       ├── main_window.py     layout and coordination
│       ├── param_panel.py     adaptive sliders
│       ├── settings_panel.py  musical and harmony settings
│       └── plots.py           equation, piano roll and waveform views
└── tests/
    └── test_core.py
```

The `core` package contains no Qt code, so it can be imported into a notebook or script for batch exploration. The equation is parsed with a restricted SymPy namespace rather than Python's `eval`, so typed input cannot run arbitrary code.

## Known limits

Rendering happens on the main thread, debounced by 60 milliseconds. Typical melodies render in well under 50 milliseconds, but the largest settings (512 long notes at a slow tempo, around 25 minutes of audio) take most of a second, during which the window pauses briefly. Equations are limited to 300 characters and 12 parameters. Recursive definitions, where each note depends on the previous one, are not supported in this version.

## Troubleshooting

If the Play button is greyed out, hover over it to see the audio error. On Linux this almost always means PortAudio is missing (see Requirements). If the window does not open and the terminal reports a Qt platform plugin error on Linux, install `libxcb-cursor0` with your package manager. If VS Code reports that modules cannot be found, check that the `.venv` interpreter is selected.
