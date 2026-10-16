"""Safe parsing of a user equation and inference of slider attributes.

The equation is parsed with SymPy using a restricted namespace, so arbitrary
Python cannot be executed. Every free symbol other than the index variable
becomes a parameter. Each parameter is then classified by where it sits in
the expression tree (inside a sine argument, as a modulus, as a divisor and
so on), and that role decides the slider range, default and integer stepping.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_application,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

INDEX_CANDIDATES = ("n", "t", "x")
MAX_PARAMETERS = 12
MAX_LENGTH = 300


class EquationError(ValueError):
    """Raised when the equation cannot be parsed into something playable."""


@dataclass(frozen=True)
class ParamSpec:
    """Slider description for one parameter, inferred from the equation."""

    name: str
    role: str
    label: str
    minimum: float
    maximum: float
    default: float
    integer: bool
    hint: str

    def clamp(self, value: float) -> float:
        return min(max(value, self.minimum), self.maximum)


@dataclass
class ParsedEquation:
    text: str
    expr: sp.Expr
    index: sp.Symbol
    params: list[ParamSpec]
    func: Callable[..., object]
    warnings: list[str] = field(default_factory=list)

    @property
    def signature(self) -> tuple[tuple[str, str], ...]:
        """Identifies the slider layout. The panel is rebuilt only when this changes."""
        return tuple((p.name, p.role) for p in self.params)


# role: (label, minimum, maximum, default, integer, hint)
ROLE_TABLE: dict[str, tuple[str, float, float, float, bool, str]] = {
    "modulus": ("Modulus", 2, 24, 7, True,
                "Wraps values into a repeating cycle. Small whole numbers give short repeating patterns."),
    "period": ("Period", 1, 64, 8, False,
               "Number of steps in one cycle of the oscillation."),
    "frequency": ("Frequency", 0.01, 3.0, 0.5, False,
                  "How far the oscillation advances per step, in radians."),
    "phase": ("Phase", 0.0, 2 * math.pi, 0.0, False,
              "Shifts the oscillation along the index, in radians."),
    "growth": ("Growth base", 0.8, 1.25, 1.05, False,
               "Raised to the power of the index. Above 1 grows, below 1 decays."),
    "rate": ("Rate", -0.3, 0.3, 0.05, False,
             "Exponential change per step."),
    "exponent": ("Exponent", 0.0, 4.0, 2.0, False,
                 "Power applied to a term."),
    "divisor": ("Divisor", 0.1, 20.0, 4.0, False,
                "Divides a term. Kept away from zero."),
    "positive": ("Positive", 0.1, 10.0, 1.0, False,
                 "Feeds a logarithm or root, so it should stay positive."),
    "multiplier": ("Multiplier", -5.0, 5.0, 1.0, False,
                   "Scales a term relative to the rest of the equation."),
    "offset": ("Offset", -10.0, 10.0, 0.0, False,
               "Shifts a term up or down."),
    "general": ("General", -5.0, 5.0, 1.0, False,
                "General parameter."),
}

# When a symbol appears several times, the most specific role wins.
ROLE_PRIORITY = [
    "modulus", "period", "frequency", "phase", "growth", "rate",
    "exponent", "divisor", "positive", "multiplier", "offset", "general",
]

_FUNCTIONS: dict[str, object] = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
    "abs": sp.Abs, "floor": sp.floor, "ceil": sp.ceiling, "sign": sp.sign,
    "mod": sp.Mod, "pi": sp.pi, "e": sp.E,
}

# Only the constructors that SymPy's tokeniser needs are made available.
_GLOBALS: dict[str, object] = {
    "__builtins__": {},
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Symbol": sp.Symbol,
    "Function": sp.Function,
}

_TRANSFORMS = standard_transformations + (
    implicit_multiplication,
    implicit_application,
    convert_xor,
)

_ALLOWED_CHARS = re.compile(r"^[A-Za-z0-9_\s.+\-*/^(),%=]*$")
_ATTRIBUTE_ACCESS = re.compile(r"[A-Za-z_]\w*\s*\.\s*[A-Za-z_]")
_CALL = re.compile(r"\b([A-Za-z_]\w+)\s*\(")

SUPPORTED_SUMMARY = (
    "sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, exp, log, sqrt, "
    "abs, floor, ceil, sign, mod, pi, e"
)


def parse_equation(text: str) -> ParsedEquation:
    """Parse the user's equation and describe its parameters."""
    raw = (text or "").strip()
    if not raw:
        raise EquationError("Type an equation in n, for example sin(a*n) + b*cos(n/p).")
    if len(raw) > MAX_LENGTH:
        raise EquationError(f"The equation is longer than {MAX_LENGTH} characters.")
    if "=" in raw:
        # Accept forms such as "y = ..." or "f(n) = ..." and keep the right-hand side.
        raw = raw.split("=")[-1].strip()
        if not raw:
            raise EquationError("Nothing follows the equals sign.")
    if not _ALLOWED_CHARS.match(raw):
        bad = sorted({c for c in raw if not _ALLOWED_CHARS.match(c)})
        raise EquationError(f"Unsupported character(s) {' '.join(bad)}.")
    if "__" in raw or _ATTRIBUTE_ACCESS.search(raw):
        raise EquationError("That syntax is not allowed in an equation.")
    # A single letter before a bracket, as in a(n+1), is read as multiplication.
    # A longer unknown name before a bracket is almost always a misspelt function.
    unknown = sorted({m for m in _CALL.findall(raw) if m not in _FUNCTIONS})
    if unknown:
        raise EquationError(
            f"Unknown function {', '.join(unknown)}. Supported functions are {SUPPORTED_SUMMARY}. "
            "For multiplication write name*(...)."
        )

    try:
        expr = parse_expr(
            raw,
            local_dict=dict(_FUNCTIONS),
            global_dict=dict(_GLOBALS),
            transformations=_TRANSFORMS,
            evaluate=True,
        )
    except Exception as exc:  # SymPy raises a wide variety of exception types
        raise EquationError(f"Could not read the equation ({_short(exc)}).") from None

    if not isinstance(expr, sp.Expr):
        raise EquationError("The input is not a mathematical expression.")

    undefined = sorted({f.func.__name__ for f in expr.atoms(AppliedUndef)})
    if undefined:
        names = ", ".join(undefined)
        raise EquationError(
            f"Unknown function {names}. Write a*(...) rather than a(...) for multiplication. "
            f"Supported functions are {SUPPORTED_SUMMARY}."
        )

    warnings: list[str] = []
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    index = next((s for s in symbols if s.name in INDEX_CANDIDATES), None)
    if index is None:
        index = sp.Symbol("n")
        warnings.append(
            "The equation does not use n, so every step gives the same value "
            "and the melody is a single repeated note."
        )
    elif index.name != "n":
        warnings.append(f"Using {index.name} as the step index.")

    param_symbols = [s for s in symbols if s != index]
    if len(param_symbols) > MAX_PARAMETERS:
        raise EquationError(
            f"The equation has {len(param_symbols)} parameters. The limit is {MAX_PARAMETERS}."
        )

    try:
        func = sp.lambdify([index, *param_symbols], expr, modules=["numpy"])
    except Exception as exc:
        raise EquationError(f"Could not compile the equation ({_short(exc)}).") from None

    params = [_make_spec(s, _classify(expr, s, index)) for s in param_symbols]
    return ParsedEquation(text=text, expr=expr, index=index, params=params,
                          func=func, warnings=warnings)


def _make_spec(symbol: sp.Symbol, role: str) -> ParamSpec:
    label, lo, hi, default, integer, hint = ROLE_TABLE[role]
    return ParamSpec(name=symbol.name, role=role, label=label, minimum=float(lo),
                     maximum=float(hi), default=float(default), integer=integer, hint=hint)


def _classify(expr: sp.Expr, symbol: sp.Symbol, index: sp.Symbol) -> str:
    """Return the most specific role across every occurrence of the symbol."""
    roles = [_role_for_path(path, index) for path in _occurrence_paths(expr, symbol)]
    if not roles:
        return "general"
    return min(roles, key=ROLE_PRIORITY.index)


def _occurrence_paths(expr: sp.Basic, target: sp.Symbol) -> list[list[tuple[sp.Basic, int]]]:
    """List the ancestor chain, as (node, child position) pairs, for each occurrence."""
    paths: list[list[tuple[sp.Basic, int]]] = []

    def walk(node: sp.Basic, path: list[tuple[sp.Basic, int]]) -> None:
        if node == target:
            paths.append(path)
            return
        for position, child in enumerate(node.args):
            if child.has(target):
                walk(child, path + [(node, position)])

    walk(expr, [])
    return paths


def _role_for_path(path: list[tuple[sp.Basic, int]], index: sp.Symbol) -> str:
    nodes = [node for node, _ in path]

    for node, position in path:
        if isinstance(node, sp.Mod) and position == 1:
            return "modulus"

    trig_at = _last(nodes, lambda n: isinstance(n, (TrigonometricFunction, HyperbolicFunction)))
    if trig_at is not None and nodes[trig_at].args[0].has(index):
        inside = path[trig_at + 1:]
        scales_index = any(isinstance(n, sp.Mul) and n.has(index) for n, _ in inside)
        divides = any(n.is_Pow and pos == 0 and _negative_number(n.exp) for n, pos in inside)
        if scales_index and divides:
            return "period"
        if scales_index:
            return "frequency"
        return "phase"

    exp_at = _last(nodes, lambda n: isinstance(n, sp.exp))
    if exp_at is not None:
        inside = path[exp_at + 1:]
        if any(isinstance(n, sp.Mul) and n.has(index) for n, _ in inside):
            return "rate"

    pow_at = _last(nodes, lambda n: n.is_Pow)
    if pow_at is not None:
        node, position = path[pow_at]
        if position == 1:
            return "exponent"
        if node.exp.has(index):
            return "growth"
        if _negative_number(node.exp):
            return "divisor"
        if node.exp.is_number and not node.exp.is_integer:
            return "positive"  # square roots and other fractional powers

    if any(isinstance(n, sp.log) for n in nodes):
        return "positive"

    if nodes:
        parent = nodes[-1]
        if isinstance(parent, sp.Mul):
            return "multiplier"
        if isinstance(parent, sp.Add):
            return "offset"
    return "general"


def _last(items: list, predicate: Callable[[object], bool]) -> int | None:
    for i in range(len(items) - 1, -1, -1):
        if predicate(items[i]):
            return i
    return None


def _negative_number(value: sp.Basic) -> bool:
    return bool(value.is_number) and bool(value.is_negative)


def _short(exc: Exception) -> str:
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return message[:120]
