# PineTS oracle (dev only)

Generates golden indicator fixtures for `tests/golden/`. See plan §3 and §8.2.

**PineTS is AGPL-3.0-only.** It is a *development* dependency used to produce
reference numbers and is never imported by `engine/`, never bundled, never
distributed. That keeps the shipped system copyleft-free. Do not add it to the
engine's dependencies.

## Usage

    npm install
    node generate_fixtures.mjs      # writes ../../tests/golden/fixtures/

Then verify the Python implementation against it:

    python3 -m pytest tests/golden/ -v

## Why keep an oracle at all

Leaving PineTS means we own indicator semantics, so we lose a free external
reference. Demoting it to a test oracle keeps the reference without the runtime
coupling, the transpiler bugs, or the licence.

Where the owner has an MT5 reference for the same indicator, add a second
fixture from MT5. **Where MT5 and Pine disagree, MT5 wins** (plan §8.2) — MT5 is
the platform the strategies actually run on.
