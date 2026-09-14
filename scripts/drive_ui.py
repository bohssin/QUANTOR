"""Drive the UI in a real browser and report what it does. Plan §19.

A page that returns HTTP 200 is not a page that works. This loads each route in
Chromium, clicks the things a person clicks, and fails on **any** console error
or unhandled rejection — the failure mode for a no-build ES-module frontend is a
silent syntax error in one module, which serves 200 and renders nothing.

    python3 scripts/drive_ui.py                    # check every page
    python3 scripts/drive_ui.py --shots out/       # and save screenshots

The environment's Chromium is older than the pinned Playwright, so the binary is
named explicitly rather than downloaded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CHROME_CANDIDATES = [
    "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
    "/opt/pw-browsers/chromium/chrome-linux/chrome",
]


def chrome_path() -> str | None:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    return None


class Driver:
    """A browser page that remembers every console error it saw."""

    def __init__(self, page, base: str) -> None:
        self.page = page
        self.base = base.rstrip("/")
        self.errors: list[str] = []
        page.on("console", self._console)
        page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))

    def _console(self, msg) -> None:
        if msg.type in ("error", "warning"):
            text = msg.text
            # A favicon 404 is not a bug in the application.
            if "favicon" in text.lower():
                return
            self.errors.append(f"console.{msg.type}: {text}")

    def go(self, hash_route: str, wait: str = ".card, .empty, .loading") -> None:
        self.page.goto(f"{self.base}/{hash_route}", wait_until="domcontentloaded")
        self.page.wait_for_timeout(150)
        self.page.evaluate("() => window.dispatchEvent(new HashChangeEvent('hashchange'))")
        self.page.wait_for_selector(wait, timeout=15_000)
        self.page.wait_for_timeout(700)

    def text(self, selector: str = "#main") -> str:
        return self.page.locator(selector).inner_text()

    def shot(self, out: Path | None, name: str) -> None:
        if out is None:
            return
        out.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(out / f"{name}.png"), full_page=False)

    def check(self, label: str) -> list[str]:
        found, self.errors = self.errors, []
        if found:
            print(f"  [FAIL] {label}")
            for e in found[:6]:
                print(f"         {e[:160]}")
        else:
            print(f"  [ok]   {label}")
        return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--shots", type=Path, default=None)
    ap.add_argument("--keep-open", action="store_true")
    ap.add_argument("--workflow", action="store_true",
                    help="also click through backtest / optimize / control / validate")
    ap.add_argument("--timeframe", default="H1")
    ap.add_argument("--strategy", default="",
                    help="pick this strategy by id instead of the first listed")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    exe = chrome_path()
    failures: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=exe,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": 1680, "height": 1000},
                                device_scale_factor=2)
        d = Driver(page, args.base)

        print(f"\ndriving {args.base}\n")

        d.go("#/data")
        print("  data page:", d.text().splitlines()[0] if d.text() else "(empty)")
        d.shot(args.shots, "01-data")
        failures += d.check("data page renders")

        d.go("#/strategies")
        d.shot(args.shots, "02-strategies")
        failures += d.check("strategies page renders")

        d.go("#/chart")
        # The chart is canvas, and it needs an API round trip before it draws —
        # counting immediately races the fetch and reports a false zero.
        try:
            page.wait_for_selector("#price-chart canvas", timeout=20_000)
        except Exception:                                   # noqa: BLE001
            pass
        canvases = page.locator("#price-chart canvas").count()
        print(f"  chart canvases: {canvases}")
        if canvases == 0 and "No data loaded" not in d.text():
            failures.append("chart rendered no canvas")
        d.shot(args.shots, "03-chart")
        failures += d.check("chart page renders")

        # --- the actual workflow, clicked the way a person clicks it ---
        if args.workflow:
            d.go("#/strategies")
            page.wait_for_selector(".list-item", timeout=10_000)
            if args.strategy:
                page.locator(".list-item", has_text=args.strategy).first.click()
            else:
                page.locator(".list-item").first.click()
            page.wait_for_selector("#run-data", timeout=10_000)
            page.select_option("#run-tf", args.timeframe)
            d.shot(args.shots, "05-strategy-detail")
            failures += d.check("strategy detail renders")

            for label, wait_for, name in [
                ("Backtest", ".metrics", "06-backtest"),
                ("Optimize", "table", "07-optimize"),
                ("Control test", ".verdict", "08-control"),
                ("Walk-forward validate", "table", "09-validate"),
            ]:
                if label == "Walk-forward validate":
                    page.fill("#run-train", "2000")
                    page.fill("#run-test", "500")
                    page.fill("#run-min-trades", "60")
                print(f"  clicking {label!r} …")
                page.get_by_role("button", name=label, exact=True).click()
                try:
                    page.wait_for_selector(f"#results {wait_for}", timeout=180_000)
                    page.wait_for_timeout(900)
                except Exception as exc:                    # noqa: BLE001
                    failures.append(f"{label} produced no result: {exc}")
                shown = d.text("#results")[:120].replace("\n", " | ")
                print(f"     -> {shown}")
                d.shot(args.shots, name)
                failures += d.check(f"{label} ran from the UI")

            # The chart, with that strategy's own trades drawn on it.
            d.go("#/chart")
            page.wait_for_selector("#price-chart canvas", timeout=20_000)
            page.wait_for_timeout(1200)
            canvases = page.locator("#price-chart canvas").count()
            trades = page.locator("#main table tbody tr").count()
            print(f"  chart with strategy: {canvases} canvases, {trades} trade rows")
            if canvases == 0:
                failures.append("chart drew no canvas with a strategy applied")
            d.shot(args.shots, "10-chart-with-trades")
            failures += d.check("chart renders the strategy's trades")

        d.go("#/history")
        d.shot(args.shots, "04-history")
        failures += d.check("history page renders")

        # The sidebar is on every page and is the piece most likely to throw.
        agent = page.locator("#agent-log")
        print(f"  agent panel present: {agent.count() == 1}")
        if agent.count() != 1:
            failures.append("agent sidebar missing")

        if args.keep_open:
            input("browser open — press enter to close ")
        browser.close()

    print()
    if failures:
        print(f"{len(failures)} problem(s) found.\n")
        return 1
    print("UI clean: every page rendered with no console errors.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
