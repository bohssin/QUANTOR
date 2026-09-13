"""Signal-block contract enforcement. Plan §9.

The validator is a correctness boundary, not a sandbox — see the module
docstring. These tests pin the constructs it must catch, with look-ahead first
because it is the one that produces a brilliant-looking strategy rather than an
error.
"""

from __future__ import annotations

import pytest

from engine.signal import ValidationError, validate_signal_block

VALID = '''
def signal(bars, p):
    fast = ema_fast(bars.close, p["fast"])
    slow = ema_fast(bars.close, p["slow"])
    up = fast > slow
    stop = atr_fast(bars.high, bars.low, bars.close, p["atr_len"]) * p["mult"]
    return {"long_entry": up, "short_entry": ~up,
            "stop_distance": stop, "target_distance": stop * 2.0}
'''


def test_a_conforming_block_passes():
    r = validate_signal_block(VALID)
    assert r.ok, r.errors


# --- look-ahead: the bug that looks like alpha -------------------------------

@pytest.mark.parametrize("src,why", [
    ('def signal(bars, p):\n    x = bars.close[5:]\n    return {"long_entry": x}',
     "forward slice shifts the series back in time"),
    ('def signal(bars, p):\n    x = bars.close[-10:]\n    return {"long_entry": x}',
     "negative slice takes the most recent values"),
    ('def signal(bars, p):\n    x = bars.high[i + 1]\n    return {"long_entry": x}',
     "indexing a future bar"),
])
def test_lookahead_is_rejected(src, why):
    r = validate_signal_block(src)
    assert not r.ok, why
    assert any("look-ahead" in e or "most recent" in e for e in r.errors), r.errors


def test_ordinary_history_access_still_passes():
    """The check must not be so blunt it forbids reading past bars."""
    r = validate_signal_block(
        'def signal(bars, p):\n'
        '    prev = bars.close[:-1]\n'
        '    return {"long_entry": prev, "short_entry": prev,\n'
        '            "stop_distance": prev, "target_distance": prev}'
    )
    assert r.ok, r.errors


# --- escapes -----------------------------------------------------------------

@pytest.mark.parametrize("src", [
    'def signal(bars, p):\n    import os\n    return {}',
    'import sys\ndef signal(bars, p):\n    return {}',
    'from subprocess import run\ndef signal(bars, p):\n    return {}',
])
def test_imports_are_rejected(src):
    assert not validate_signal_block(src).ok


@pytest.mark.parametrize("name", ["open", "eval", "exec", "compile", "__import__"])
def test_forbidden_builtins_are_rejected(name):
    r = validate_signal_block(f'def signal(bars, p):\n    {name}("x")\n    return {{}}')
    assert not r.ok
    assert any(name in e for e in r.errors)


def test_dunder_attribute_access_is_rejected():
    r = validate_signal_block(
        'def signal(bars, p):\n    g = signal.__globals__\n    return {}')
    assert not r.ok
    assert any("dunder" in e for e in r.errors)


# --- shape of the contract ---------------------------------------------------

def test_missing_signal_function_is_rejected():
    r = validate_signal_block('def helper(bars):\n    return 1')
    assert not r.ok
    assert any("signal(bars, p)" in e for e in r.errors)


def test_wrong_arity_is_rejected():
    r = validate_signal_block('def signal(bars):\n    return {}')
    assert not r.ok
    assert any("exactly (bars, p)" in e for e in r.errors)


def test_missing_return_keys_are_named():
    r = validate_signal_block(
        'def signal(bars, p):\n    return {"long_entry": 1}')
    assert not r.ok
    assert any("short_entry" in e and "stop_distance" in e for e in r.errors)


def test_syntax_error_reports_its_line():
    r = validate_signal_block('def signal(bars, p)\n    return {}')
    assert not r.ok
    assert "syntax error" in r.errors[0]


def test_unreadable_return_warns_rather_than_failing():
    """A dict built dynamically cannot be read statically — warn, and let the
    engine check at run time, rather than rejecting a legitimate block."""
    r = validate_signal_block(
        'def signal(bars, p):\n'
        '    out = {}\n'
        '    out["long_entry"] = bars.close > 0\n'
        '    return out'
    )
    assert r.ok
    assert r.warnings


def test_raise_if_invalid_lists_every_violation():
    r = validate_signal_block('import os\ndef signal(bars):\n    return {}')
    with pytest.raises(ValidationError) as exc:
        r.raise_if_invalid()
    assert "import" in str(exc.value)
    assert "exactly (bars, p)" in str(exc.value)
