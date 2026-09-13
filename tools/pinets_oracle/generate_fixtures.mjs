// Dev-only. Emits reference indicator values from PineTS 0.9.33.
// See plan §8.2. Deterministic: the bar series is a fixed pseudo-random walk.
import { PineTS } from 'pinets';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'tests', 'golden', 'fixtures');
const N = 5_000;   // ample for warmup + recursive drift; the 5e-11 headline was validated at 20k

function bars(n) {
  const out = new Array(n);
  let p = 2000, t = Date.UTC(2024, 0, 1);
  for (let i = 0; i < n; i++) {
    const r = Math.sin(i * 12.9898) * 43758.5453;
    const d = (r - Math.floor(r) - 0.5) * 2;
    const o = p;
    p = Math.max(1, p + d);
    out[i] = {
      open: o,
      high: Math.max(o, p) + Math.abs(d) * 0.5,
      low: Math.min(o, p) - Math.abs(d) * 0.5,
      close: p,
      volume: 100,
      openTime: t + i * 1000,
    };
  }
  return out;
}

const SRC = `//@version=6
indicator("golden")
plot(ta.sma(close, 20), "sma20")
plot(ta.ema(close, 20), "ema20")
plot(ta.rma(close, 20), "rma20")
plot(ta.rsi(close, 14), "rsi14")
plot(ta.atr(14),        "atr14")
`;

const KEYS = ['sma20', 'ema20', 'rma20', 'rsi14', 'atr14'];

const data = bars(N);
const { plots } = await new PineTS(data).run(SRC);
const series = KEYS.map((k) => plots[k].data.map((p) => p.value));

// One diffable CSV: inputs and reference outputs side by side. A change to this
// file is a change to indicator semantics and must be reviewed as one (plan §8.2).
// String(v) is JS's shortest round-trip float64 representation, so this is
// lossless AND compact — padded decimals tripled the file for no extra precision.
const f = (v) => (v === null || v === undefined || Number.isNaN(v) ? '' : String(v));
const lines = [['openTime', 'open', 'high', 'low', 'close', ...KEYS].join(',')];
for (let i = 0; i < N; i++) {
  const b = data[i];
  lines.push([
    b.openTime, f(b.open), f(b.high), f(b.low), f(b.close),
    ...series.map((s) => f(s[i])),
  ].join(','));
}

mkdirSync(OUT, { recursive: true });
writeFileSync(join(OUT, 'MANIFEST.json'), JSON.stringify({
  oracle: 'pinets@0.9.33',
  bars: N,
  series: KEYS,
  note: 'empty cell = na. Values are shortest round-trip float64 reprs (lossless).',
}, null, 2) + '\n');
writeFileSync(join(OUT, 'wilder_reference.csv'), lines.join('\n') + '\n');

console.log(`wrote ${N} bars + ${KEYS.length} reference series to tests/golden/fixtures/`);
