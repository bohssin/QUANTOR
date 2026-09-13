"""Static validator for generated signal blocks. Plan §9.

**What this is, and what it is not.**

A restricted `exec` namespace is not a sandbox. Python cannot be sandboxed by
withholding names: `__import__` reaches anything, and numba's dispatchers need
it at call time anyway, so the namespace has to expose it. Withholding names
catches *mistakes* — a block calling an order function it was told not to use
gets a `NameError` instead of silently doing something — and that is worth
having, but it is a correctness boundary, not a security one.

This validator is the actual enforcement: an AST walk **before** execution that
rejects constructs the §9 contract forbids. It runs against an agent that is
following the contract imperfectly, not one that is hostile. Nothing here should
be described as a security guarantee, and a signal block should never be run on
input the owner did not initiate.

What it rejects, and why each matters:

- **imports** — the block gets bars, params and indicators; anything else is
  either a mistake or a dependency the engine cannot reproduce
- **file, network and process access** — a backtest that reads the filesystem is
  not a function of its inputs
- **`exec` / `eval` / `compile`** — defeats the point of validating statically
- **dunder attribute access** — `__globals__`, `__class__` and friends are the
  standard way out of a restricted namespace
- **forward indexing on bars** — `bars.close[i+1]`, `bars.close[5:]` and
  negative-start slices are look-ahead, which is the single most damaging bug a
  backtest can have because it looks like a brilliant strategy
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

__all__ = ["ValidationError", "ValidationResult", "validate_signal_block"]

FORBIDDEN_NAMES = frozenset({
    "exec", "eval", "compile", "open", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "__import__", "memoryview",
})

FORBIDDEN_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib",
    "requests", "urllib", "http", "pickle", "marshal", "ctypes", "builtins",
})

#: Series a block may read. Forward indexing into any of these is look-ahead.
SERIES_ATTRS = frozenset({"open", "high", "low", "close", "volume", "spread", "ms"})

REQUIRED_KEYS = ("long_entry", "short_entry", "stop_distance", "target_distance")


class ValidationError(Exception):
    pass


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValidationError(
                "signal block violates the §9 contract:\n  - "
                + "\n  - ".join(self.errors)
            )


def validate_signal_block(source: str) -> ValidationResult:
    """Check a generated signal block against the §9 contract."""
    errors: list[str] = []
    warnings: list[str] = []

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ValidationResult(False, [f"syntax error at line {exc.lineno}: {exc.msg}"])

    visitor = _Visitor()
    visitor.visit(tree)
    errors.extend(visitor.errors)
    warnings.extend(visitor.warnings)

    if not visitor.found_signal:
        errors.append("must define a function `signal(bars, p)`")
    elif visitor.signal_args is not None and len(visitor.signal_args) != 2:
        errors.append(
            f"signal() takes exactly (bars, p); got "
            f"({', '.join(visitor.signal_args) or 'nothing'})"
        )

    missing = [k for k in REQUIRED_KEYS if k not in visitor.returned_keys]
    if visitor.found_signal and visitor.returned_keys and missing:
        errors.append(
            f"the returned dict is missing {missing} — signal() must return "
            f"{list(REQUIRED_KEYS)}"
        )
    elif visitor.found_signal and not visitor.returned_keys:
        warnings.append(
            "could not statically read the returned dict keys; the engine will "
            "check them at run time"
        )

    return ValidationResult(not errors, errors, warnings)


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.found_signal = False
        self.signal_args: list[str] | None = None
        self.returned_keys: set[str] = set()

    def _err(self, node: ast.AST, msg: str) -> None:
        self.errors.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    # --- imports ---
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._err(node, f"import of {alias.name!r} — a signal block gets bars, "
                            "params and the indicator set, nothing else")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._err(node, f"import from {node.module!r} — not allowed in a signal block")

    # --- names and calls ---
    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id in FORBIDDEN_NAMES:
            self._err(node, f"use of {node.id!r} is not allowed in a signal block")
        if node.id in FORBIDDEN_MODULES:
            self._err(node, f"reference to module {node.id!r} is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self._err(node, f"dunder attribute {node.attr!r} — the standard route "
                            "out of a restricted namespace")
        self.generic_visit(node)

    # --- look-ahead ---
    def visit_Subscript(self, node: ast.Subscript) -> None:
        target = node.value
        if isinstance(target, ast.Attribute) and target.attr in SERIES_ATTRS:
            self._check_lookahead(node, f"bars.{target.attr}")
        self.generic_visit(node)

    def _check_lookahead(self, node: ast.Subscript, what: str) -> None:
        sl = node.slice
        if isinstance(sl, ast.Slice):
            lo = sl.lower
            if isinstance(lo, ast.Constant) and isinstance(lo.value, int) and lo.value > 0:
                self._err(node, f"{what}[{lo.value}:] shifts the series backwards in "
                                "time — this is look-ahead")
            if isinstance(lo, ast.UnaryOp) and isinstance(lo.op, ast.USub):
                self._err(node, f"{what}[-n:] takes the most recent values, which a "
                                "per-bar block must not do")
        elif isinstance(sl, ast.BinOp) and isinstance(sl.op, ast.Add):
            self._err(node, f"{what}[i + n] reads a future bar — look-ahead")

    # --- the signal function ---
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == "signal":
            self.found_signal = True
            self.signal_args = [a.arg for a in node.args.args]
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Dict):
                    for key in sub.value.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            self.returned_keys.add(key.value)
        self.generic_visit(node)
