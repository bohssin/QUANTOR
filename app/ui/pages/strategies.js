// The strategies page: write it, backtest it, sweep it, walk it forward.
// Plan §15 — everything happens here, and every result is a library row.

import {
  api, el, num, int, money, pct, ago, signClass, toast, withBusy,
  metricTiles, dataTable, escapeHtml,
} from '../api.js';
import { state, remember, navigate, refreshRailStats } from '../app.js';

const TEMPLATE = `def signal(bars, p):
    """Return aligned arrays. No orders, no state, no forward indexing."""
    fast = ema_fast(bars.close, p["fast"])
    slow = ema_fast(bars.close, p["slow"])
    n = len(bars.close)

    above = fast > slow
    long_entry = np.zeros(n, bool)
    short_entry = np.zeros(n, bool)
    long_entry[1:] = above[1:] & ~above[:-1]
    short_entry[1:] = ~above[1:] & above[:-1]

    warm = np.isnan(fast) | np.isnan(slow)
    long_entry &= ~warm
    short_entry &= ~warm

    stop = atr_fast(bars.high, bars.low, bars.close, p["atr"]) * p["mult"]
    stop = np.where(np.isnan(stop) | (stop <= 0), 1.0, stop)

    return {
        "long_entry": long_entry,
        "short_entry": short_entry,
        "stop_distance": stop,
        "target_distance": stop * p["rr"],
    }
`;

export async function renderStrategies(root, { rest }) {
  const [strategies, sources] = await Promise.all([api.strategies(true), api.listData()]);
  const wanted = rest[0] || state.strategy;
  const active = strategies.find(s => s.strategy_id === wanted) || strategies[0] || null;

  const head = el('div', { class: 'page-head' },
    el('div', {},
      el('h1', {}, 'Strategies'),
      el('div', { class: 'sub' },
        `${strategies.filter(s => !s.archived).length} active, ` +
        `${strategies.filter(s => s.archived).length} archived — ` +
        'every version and run is kept')),
    el('button', { class: 'primary', onclick: () => openEditor(root, null, sources) },
      el('span', { html: '<svg viewBox="0 0 20 20"><path d="M10 4v12M4 10h12"/></svg>' }),
      'New strategy'));

  const left = el('div', { class: 'card' },
    el('h3', {}, 'Library'),
    strategies.length
      ? el('div', { class: 'list' }, strategies.map(s => el('div', {
          class: 'list-item' + (active && s.strategy_id === active.strategy_id ? ' active' : ''),
          onclick: () => { remember({ strategy: s.strategy_id, version: 0 }); navigate('#/strategies/' + encodeURIComponent(s.strategy_id)); },
        },
        el('div', { class: 'top' },
          el('span', { class: 'name' }, s.strategy_id),
          s.archived ? el('span', { class: 'pill warn' }, 'archived')
                     : el('span', { class: 'pill' }, `v${s.latest.version}`)),
        el('div', { class: 'meta' },
          `${s.runs} runs · ${int(s.comparisons)} comparisons · ${ago(s.last_run_at || s.created_at)}`),
        )))
      : el('div', { class: 'empty' },
          el('strong', {}, 'Nothing saved yet'),
          'Create a strategy, or ask the assistant to write one.'));

  const right = el('div', {});
  root.replaceChildren(head, el('div', { class: 'grid side' }, left, right));

  if (!active) {
    right.replaceChildren(el('div', { class: 'card' }, el('div', { class: 'empty' },
      el('strong', {}, 'No strategy selected'),
      'Create one on the right, or ask the assistant in the sidebar.')));
    return;
  }
  await renderDetail(right, active.strategy_id, sources, root);
}

// --- one strategy ------------------------------------------------------------

async function renderDetail(host, strategyId, sources, root) {
  const detail = await api.strategy(strategyId, state.version || 0);
  remember({ strategy: strategyId, version: detail.version });

  const results = el('div', { id: 'results' });

  host.replaceChildren(
    el('div', { class: 'card' },
      el('div', { class: 'card-head' },
        el('div', {},
          el('h2', {}, `${detail.strategy_id}  `,
            el('span', { class: 'pill info' }, `v${detail.version}`)),
          el('div', { class: 'sub dim' }, detail.description || 'No description')),
        el('div', { class: 'inline' },
          detail.source_url
            ? el('a', { class: 'btn', href: detail.source_url, target: '_blank', rel: 'noreferrer' }, 'Source')
            : null,
          el('button', {
            onclick: () => navigate(`#/chart?strategy=${encodeURIComponent(strategyId)}&version=${detail.version}`),
          }, 'Open in chart'),
          el('button', { onclick: () => openEditor(root, detail, sources) }, 'Edit / new version'),
          el('button', {
            class: 'danger',
            onclick: async (e) => {
              const archived = !(await api.strategies(true)).find(s => s.strategy_id === strategyId)?.archived;
              const reason = archived ? prompt('Why is this being archived? (kept, never deleted)') : '';
              if (archived && reason === null) return;
              await withBusy(e.currentTarget, async () => {
                await api.archive(strategyId, { archived, reason: reason || '' });
                toast(archived ? 'Archived — its runs are still queryable' : 'Restored', 'good');
                navigate('#/strategies/' + encodeURIComponent(strategyId));
              }, archived ? 'Archiving' : 'Restoring');
            },
          }, 'Archive')),
      ),
      el('div', { class: 'inline dim', style: 'font-size:11.5px;font-family:var(--mono)' },
        `family ${detail.family}`, ' · ',
        `${int(detail.comparisons)} cumulative comparisons`, ' · ',
        `sha ${detail.signal_sha}`, ' · ',
        `origin ${detail.origin}`),
      detail.warnings?.length
        ? el('div', { class: 'verdict warn', style: 'margin-top:10px' },
            el('b', {}, 'Validator warnings'), detail.warnings.join(' · '))
        : null,
    ),

    el('div', { class: 'grid two' },
      el('div', { class: 'card' },
        el('h3', {}, 'Versions'),
        versionTree(detail.tree, detail.version, async v => {
          remember({ version: v });
          await renderDetail(host, strategyId, sources, root);
        })),
      el('div', { class: 'card' },
        el('h3', {}, 'Signal block'),
        el('pre', {
          class: 'mono scroll-y',
          style: 'margin:0;font-size:12px;background:#090c11;padding:10px;border-radius:7px;overflow:auto',
        }, detail.signal_block)),
    ),

    runPanel(detail, sources, results),
    results,
  );

  loadRecentRuns(results, strategyId);
}

function versionTree(nodes, activeVersion, onPick) {
  function list(items) {
    return el('ul', {}, items.map(n => el('li', {},
      el('span', {
        class: 'node' + (n.version === activeVersion ? ' active' : ''),
        onclick: () => onPick(n.version),
      },
        `v${n.version}`,
        el('span', { class: 'd' }, n.description?.slice(0, 44) || '—'),
        n.origin === 'library' ? el('span', { class: 'pill' }, 'lux') : null),
      n.children.length ? list(n.children) : null)));
  }
  return el('div', { class: 'tree' }, list(nodes));
}

// --- the run panel -----------------------------------------------------------

function runPanel(detail, sources, results) {
  const params = { ...detail.params };
  const paramNames = Object.keys(params);

  const dataSelect = el('select', { id: 'run-data' },
    sources.map(s => el('option', { value: s.name, selected: s.name === state.data },
      `${s.name} — ${int(s.bars)} ${s.timeframe} bars`)));
  if (sources.length && !state.data) remember({ data: sources[0].name });

  const tfSelect = el('select', { id: 'run-tf' },
    el('option', { value: '' }, 'source default'),
    ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'].map(t =>
      el('option', { value: t, selected: t === state.timeframe }, t)));

  const paramInputs = el('div', { class: 'row' }, paramNames.map(k =>
    el('div', { class: 'field' },
      el('label', {}, k),
      el('input', {
        class: 'mono', id: 'p-' + k, value: params[k],
        oninput: e => { params[k] = coerce(e.target.value); },
      }))));

  // An integer default means an integer grid. Indicator periods are counts of
  // bars, and a bound like 38.4 is rejected by the engine — correctly, but it
  // is this generator's job not to write one in the first place.
  const gridDefault = paramNames
    .filter(k => typeof params[k] === 'number')
    .slice(0, 2)
    .map(k => {
      const v = Number(params[k]);
      if (Number.isInteger(v)) {
        const lo = Math.max(1, Math.round(v / 2));
        const hi = Math.max(lo + 1, Math.round(v * 1.6));
        const step = Math.max(1, Math.round((hi - lo) / 4));
        return `"${k}": [${lo}, ${hi}, ${step}]`;
      }
      const lo = +(v / 2).toFixed(3);
      const hi = +(v * 1.6).toFixed(3);
      return `"${k}": [${lo}, ${hi}, ${+((hi - lo) / 4).toFixed(3)}]`;
    }).join(', ');

  const gridInput = el('textarea', {
    class: 'code', id: 'run-grid', rows: 3,
    style: 'min-height:74px',
  }, `{${gridDefault}}`);

  const capital = el('input', { class: 'mono', id: 'run-capital', value: '10000', type: 'number' });
  const risk = el('input', { class: 'mono', id: 'run-risk', value: '1.0', type: 'number', step: '0.1' });
  const objective = el('select', { id: 'run-objective' },
    ['return_over_maxdd', 'sharpe', 'sortino', 'calmar', 'sqn', 'net_profit',
     'profit_factor', 'expectancy_r'].map(o => el('option', { value: o }, o)));
  const minTrades = el('input', { class: 'mono', id: 'run-min-trades', value: '30', type: 'number' });
  const train = el('input', { class: 'mono', id: 'run-train', value: '2000', type: 'number' });
  const test = el('input', { class: 'mono', id: 'run-test', value: '600', type: 'number' });

  function body() {
    return {
      strategy_id: detail.strategy_id,
      version: detail.version,
      data: dataSelect.value,
      timeframe: tfSelect.value,
    };
  }
  function grid() {
    try {
      const parsed = JSON.parse(gridInput.value);
      if (!Object.keys(parsed).length) throw new Error('the grid is empty');
      return parsed;
    } catch (err) {
      throw new Error(`parameter grid is not valid JSON: ${err.message}`);
    }
  }

  return el('div', { class: 'card' },
    el('div', { class: 'card-head' }, el('h2', {}, 'Run')),

    el('div', { class: 'row' },
      el('div', { class: 'field' }, el('label', {}, 'Data source'), dataSelect),
      el('div', { class: 'field' }, el('label', {}, 'Timeframe'), tfSelect),
      el('div', { class: 'field' }, el('label', {}, 'Capital'), capital),
      el('div', { class: 'field' }, el('label', {}, 'Risk % per trade'), risk)),

    paramNames.length ? el('div', {},
      el('h3', { style: 'margin-top:14px' }, 'Parameters'), paramInputs) : null,

    el('div', { class: 'hr' }),

    el('div', { class: 'row' },
      el('div', { class: 'field', style: 'flex:3;min-width:280px' },
        el('label', {}, 'Sweep grid — name: [low, high, step]'), gridInput),
      el('div', { class: 'field' }, el('label', {}, 'Objective'), objective),
      el('div', { class: 'field' }, el('label', {}, 'Min trades'), minTrades),
      el('div', { class: 'field' }, el('label', {}, 'Train bars'), train),
      el('div', { class: 'field' }, el('label', {}, 'Test bars'), test)),

    el('div', { class: 'inline', style: 'margin-top:14px' },
      el('button', {
        class: 'primary',
        onclick: e => withBusy(e.currentTarget, async () => {
          const out = await api.backtest({
            ...body(), params,
            initial_capital: Number(capital.value),
            risk_pct: Number(risk.value) / 100,
          });
          remember({ data: dataSelect.value, timeframe: tfSelect.value });
          showBacktest(results, out, detail);
          refreshRailStats();
        }, 'Backtesting'),
      }, 'Backtest'),

      el('button', {
        onclick: e => withBusy(e.currentTarget, async () => {
          const out = await api.optimize({
            ...body(), param_grid: grid(),
            objective: objective.value, min_trades: Number(minTrades.value),
            top_k: 25,
          });
          showOptimize(results, out, detail, params);
          refreshRailStats();
        }, 'Sweeping'),
      }, 'Optimize'),

      el('button', {
        onclick: e => withBusy(e.currentTarget, async () => {
          const out = await api.validate({
            ...body(), param_grid: grid(),
            train: Number(train.value), test: Number(test.value),
            min_trades: Math.max(2, Math.floor(Number(minTrades.value) / 3)),
          });
          showValidate(results, out);
          refreshRailStats();
        }, 'Walking forward'),
      }, 'Walk-forward validate'),

      el('button', {
        title: 'Re-run on shuffled returns. An edge that survives is not an edge.',
        onclick: e => withBusy(e.currentTarget, async () => {
          const out = await api.control({
            ...body(), params, trials: 5,
          });
          showControl(results, out);
        }, 'Shuffling'),
      }, 'Control test'),

      el('span', { class: 'spacer' }),
      el('span', { class: 'faint', style: 'font-size:11.5px' },
        'every run is recorded and survives a restart')));
}

// --- result views ------------------------------------------------------------

function showBacktest(host, out, detail) {
  host.replaceChildren(el('div', { class: 'card' },
    el('div', { class: 'card-head' },
      el('h2', {}, 'Backtest  ',
        el('span', { class: 'pill' }, `run ${out.run_id}`),
        ' ', el('span', { class: 'pill' }, out.timeframe)),
      el('div', { class: 'inline' },
        el('span', {
          class: 'pill ' + (out.ledger_reconciles ? 'ok' : 'bad'),
          title: 'The trade list and the equity curve must agree to the cent.',
        }, out.ledger_reconciles ? 'ledger reconciles' : 'LEDGER MISMATCH'),
        el('button', {
          onclick: () => window.location.hash =
            `#/chart?strategy=${encodeURIComponent(out.strategy_id)}&version=${out.version}&data=${encodeURIComponent(out.data)}`,
        }, 'Show on chart'))),

    metricTiles(out.metrics),

    el('div', { class: 'inline dim', style: 'margin-top:12px;font-size:11.5px;font-family:var(--mono)' },
      `${int(out.bars)} bars`, ' · ',
      `ambiguous exits ${out.ambiguous_exits} (${pct(out.ambiguity_rate, 2)})`, ' · ',
      `rejected for zero lots ${out.rejected_zero_lots}`),
    out.ambiguity_rate > 0.05
      ? el('div', { class: 'verdict warn', style: 'margin-top:10px' },
          el('b', {}, 'High ambiguity rate'),
          'Many bars touched both the stop and the target, so their order inside ' +
          'the bar decided the trade. Bar data cannot resolve that — the engine ' +
          'assumes the worse side. Treat the result as a lower bound.')
      : null));
}

function showOptimize(host, out, detail, params) {
  const p = out.plateau;
  const verdictClass = !p || p.robustness === null ? 'warn'
    : p.robustness >= 0.7 ? 'good' : p.robustness >= 0.4 ? 'warn' : 'bad';

  host.replaceChildren(el('div', { class: 'card' },
    el('div', { class: 'card-head' },
      el('h2', {}, 'Optimization  ',
        el('span', { class: 'pill' }, `run ${out.run_id}`),
        ' ', el('span', { class: 'pill' }, out.timeframe)),
      el('div', { class: 'inline faint', style: 'font-size:11.5px' },
        `${int(out.comparisons)} evaluated in ${out.seconds}s · `,
        `${int(out.family_comparisons)} cumulative for family "${out.family}"`)),

    p ? el('div', { class: 'verdict ' + verdictClass },
        el('b', {}, `Plateau: ${p.verdict}`),
        p.robustness !== null
          ? `robustness ${num(p.robustness, 3)} — ${p.neighbours} grid neighbours ` +
            `average ${num(p.neighbour_mean, 3)} against a peak of ${num(p.peak, 3)}. ` +
            'A sharp optimum surrounded by bad neighbours is a hole in the noise, not an edge.'
          : 'The best configuration is not profitable, so there is no peak to be robust around.')
      : null,

    el('div', { class: 'verdict' },
      el('b', {}, 'Deflated Sharpe needs this number'),
      `${int(out.family_comparisons)} configurations have now been scored against ` +
      `this family. That cumulative count — not this sweep's ${int(out.comparisons)} — ` +
      'is what a significance test must divide by.'),

    el('h3', { style: 'margin-top:14px' }, `Top ${out.top.length} of ${int(out.evaluated)}`),
    dataTable(out.top.map(r => ({ ...r.params, ...r })), [
      ...Object.keys(out.top[0]?.params || {}).map(k => ({
        key: k, label: k, num: true, fmt: r => String(r.params[k]),
      })),
      { key: 'score', label: 'score', num: true, fmt: r => num(r.score, 4), cls: r => signClass(r.score) },
      { key: 'net_profit', label: 'net', num: true, fmt: r => money(r.net_profit), cls: r => signClass(r.net_profit) },
      { key: 'max_drawdown', label: 'max dd', num: true, fmt: r => money(r.max_drawdown) },
      { key: 'sharpe', label: 'sharpe', num: true, fmt: r => num(r.sharpe) },
      { key: 'profit_factor', label: 'pf', num: true, fmt: r => num(r.profit_factor) },
      { key: 'win_rate', label: 'win', num: true, fmt: r => pct(r.win_rate) },
      { key: 'n_trades', label: 'trades', num: true, fmt: r => int(r.n_trades) },
    ], {
      sortKey: 'score',
      onRow: row => {
        for (const [k, v] of Object.entries(row.params)) {
          const input = document.getElementById('p-' + k);
          if (input) { input.value = v; params[k] = v; }
        }
        toast('Parameters loaded into the run panel — backtest to confirm', 'good');
      },
    }),
    el('div', { class: 'faint', style: 'margin-top:8px;font-size:11.5px' },
      'Click a row to load those parameters. Prefer a plateau over a peak.')));
}

function showValidate(host, out) {
  const eff = out.walk_forward_efficiency;
  const cls = out.verdict.startsWith('generalized') ? 'good'
    : out.verdict.startsWith('did not') ? 'bad' : 'warn';

  host.replaceChildren(el('div', { class: 'card' },
    el('div', { class: 'card-head' },
      el('h2', {}, 'Walk-forward  ',
        el('span', { class: 'pill' }, `run ${out.run_id}`),
        ' ', el('span', { class: 'pill' }, out.timeframe)),
      el('span', { class: 'faint', style: 'font-size:11.5px' },
        `${int(out.total_comparisons)} comparisons across ${out.folds} folds in ${out.seconds}s`)),

    el('div', { class: 'verdict ' + cls },
      el('b', {}, out.verdict),
      `Efficiency ${num(eff, 3)} — out-of-sample score as a fraction of in-sample. ` +
      `${out.positive_folds} of ${out.folds} test folds were profitable. ` +
      'A full search runs inside each train fold and its winner is scored once, ' +
      'out-of-sample, on data the search never saw.'),

    el('div', { class: 'metrics', style: 'margin-bottom:12px' },
      el('div', { class: 'metric' }, el('div', { class: 'k' }, 'Efficiency'),
        el('div', { class: 'v ' + signClass(eff) }, num(eff, 3))),
      el('div', { class: 'metric' }, el('div', { class: 'k' }, 'Folds'),
        el('div', { class: 'v' }, int(out.folds))),
      el('div', { class: 'metric' }, el('div', { class: 'k' }, 'Positive folds'),
        el('div', { class: 'v' }, `${out.positive_folds}/${out.folds}`)),
      el('div', { class: 'metric' }, el('div', { class: 'k' }, 'Mean OOS'),
        el('div', { class: 'v ' + signClass(out.mean_oos_score) }, num(out.mean_oos_score, 3)))),

    dataTable(out.per_fold, [
      { key: 'fold', label: 'fold', num: true },
      { key: 'params', label: 'chosen parameters', text: true,
        fmt: r => Object.entries(r.params).map(([k, v]) => `${k}=${v}`).join('  ') },
      { key: 'train', label: 'train', num: true, fmt: r => num(r.train, 3), cls: r => signClass(r.train) },
      { key: 'test', label: 'test (OOS)', num: true, fmt: r => num(r.test, 3), cls: r => signClass(r.test) },
      { key: 'net_profit', label: 'net', num: true, fmt: r => money(r.net_profit), cls: r => signClass(r.net_profit) },
      { key: 'trades', label: 'trades', num: true, fmt: r => int(r.trades) },
    ], { sortKey: 'fold' })));
}

function showControl(host, out) {
  host.replaceChildren(el('div', { class: 'card' },
    el('div', { class: 'card-head' },
      el('h2', {}, 'Control test  ',
        el('span', { class: 'pill' }, out.timeframe),
        ' ', el('span', { class: 'pill ' + (out.passed ? 'ok' : 'bad') },
          out.passed ? 'passed' : 'FAILED')),
      el('span', { class: 'faint', style: 'font-size:11.5px' },
        `${out.trials} shuffled trials`)),

    el('div', { class: 'verdict ' + (out.passed ? 'good' : 'bad') },
      el('b', {}, out.passed ? 'The edge lives in the ordering'
                             : 'This is not an edge'),
      out.verdict),

    el('div', { class: 'metrics', style: 'margin-bottom:10px' },
      el('div', { class: 'metric' },
        el('div', { class: 'k' }, 'Real expectancy R'),
        el('div', { class: 'v ' + signClass(out.real.expectancy_r) },
          num(out.real.expectancy_r, 4))),
      el('div', { class: 'metric' },
        el('div', { class: 'k' }, 'Shuffled mean'),
        el('div', { class: 'v ' + signClass(out.control_mean_expectancy_r) },
          num(out.control_mean_expectancy_r, 4))),
      el('div', { class: 'metric' },
        el('div', { class: 'k' }, 'Shuffled best'),
        el('div', { class: 'v ' + signClass(out.control_max_expectancy_r) },
          num(out.control_max_expectancy_r, 4))),
      el('div', { class: 'metric' },
        el('div', { class: 'k' }, 'Real trades'),
        el('div', { class: 'v' }, int(out.real.n_trades)))),

    dataTable(
      [{ what: 'real data', ...out.real },
       ...out.controls.map((c, i) => ({ what: `shuffled #${i + 1}`, ...c }))],
      [
        { key: 'what', label: '', text: true },
        { key: 'expectancy_r', label: 'expectancy R', num: true,
          fmt: r => num(r.expectancy_r, 4), cls: r => signClass(r.expectancy_r) },
        { key: 'profit_factor', label: 'profit factor', num: true, fmt: r => num(r.profit_factor) },
        { key: 'win_rate', label: 'win rate', num: true, fmt: r => pct(r.win_rate) },
        { key: 'n_trades', label: 'trades', num: true, fmt: r => int(r.n_trades) },
      ]),

    el('div', { class: 'faint', style: 'margin-top:8px;font-size:11.5px' },
      'Shuffling bar-to-bar returns keeps the distribution — same volatility, ' +
      'same costs, same bar count — and destroys the order. Expectancy per ' +
      'trade is shown rather than net profit because sizing is fixed-fractional, ' +
      'so profit compounds and its headline mostly measures how many trades there were.')));
}

async function loadRecentRuns(host, strategyId) {
  const runs = await api.runs({ strategy_id: strategyId, limit: 8 });
  if (!runs.length) return;
  host.replaceChildren(el('div', { class: 'card tight' },
    el('h3', {}, 'Recent runs'),
    dataTable(runs, [
      { key: 'run_id', label: 'id', num: true },
      { key: 'kind', label: 'kind', fmt: r => el('span', { class: 'pill' }, r.kind) },
      { key: 'version', label: 'ver', num: true, fmt: r => 'v' + r.version },
      { key: 'timeframe', label: 'tf' },
      { key: 'status', label: 'status',
        fmt: r => el('span', { class: 'pill ' + (r.status === 'ok' ? 'ok' : 'bad') }, r.status) },
      { key: 'summary', label: 'result', text: true, fmt: r => runSummary(r) },
      { key: 'started_at', label: 'when', fmt: r => ago(r.started_at) },
    ], { sortKey: 'run_id', onRow: r => navigate(`#/history/${r.run_id}`) })));
}

export function runSummary(r) {
  const m = r.metrics || {};
  if (r.status !== 'ok') return r.error?.slice(0, 80) || 'failed';
  if (r.kind === 'backtest') return `${money(m.net_profit)} · ${int(m.n_trades)} trades · sharpe ${num(m.sharpe)}`;
  if (r.kind === 'optimize') return `best ${num(m.best_score, 3)} · ${money(m.net_profit)}`;
  if (r.kind === 'validate') return `efficiency ${num(m.walk_forward_efficiency, 3)} · ${m.positive_folds}/${m.folds} folds positive`;
  return '—';
}

// --- the editor --------------------------------------------------------------

function openEditor(root, existing, sources) {
  const isNew = !existing;
  const id = el('input', { value: existing?.strategy_id || '', class: 'mono', placeholder: 'ema-atr-cross' });
  if (!isNew) id.disabled = true;
  const description = el('input', { value: '', placeholder: 'what changed, in one line' });
  const family = el('input', {
    class: 'mono', value: existing?.family || '',
    placeholder: 'shared across variants of one idea',
  });
  const sourceUrl = el('input', {
    class: 'mono', value: existing?.source_url || '',
    placeholder: 'https://www.luxalgo.com/library/concept/…',
  });
  const paramsInput = el('textarea', { class: 'code', rows: 2, style: 'min-height:56px' },
    JSON.stringify(existing?.params || { fast: 12, slow: 40, atr: 14, mult: 2.0, rr: 2.0 }, null, 0));
  const code = el('textarea', { class: 'code', spellcheck: 'false' },
    existing?.signal_block || TEMPLATE);

  const parentField = el('select', {},
    (existing?.versions || []).map(v =>
      el('option', { value: v.version, selected: v.version === existing.version },
        `v${v.version} — ${v.description?.slice(0, 40) || ''}`)));

  root.replaceChildren(
    el('div', { class: 'page-head' },
      el('div', {},
        el('h1', {}, isNew ? 'New strategy' : `New version of ${existing.strategy_id}`),
        el('div', { class: 'sub' },
          'Saved versions are never overwritten. Branch from any parent.')),
      el('button', { onclick: () => navigate('#/strategies' + (existing ? '/' + encodeURIComponent(existing.strategy_id) : '')) }, 'Cancel')),

    el('div', { class: 'card' },
      el('div', { class: 'row' },
        el('div', { class: 'field' }, el('label', {}, 'Strategy id'), id),
        el('div', { class: 'field', style: 'flex:2' }, el('label', {}, 'Description'), description),
        el('div', { class: 'field' }, el('label', {}, 'Family'), family),
        !isNew ? el('div', { class: 'field' }, el('label', {}, 'Branch from'), parentField) : null),
      el('div', { class: 'field' },
        el('label', {}, 'Source URL — attribution when the idea came from the LuxAlgo Library'),
        sourceUrl),
      el('div', { class: 'field' }, el('label', {}, 'Default parameters (JSON)'), paramsInput)),

    el('div', { class: 'card' },
      el('div', { class: 'card-head' },
        el('h2', {}, 'signal(bars, p)'),
        el('span', { class: 'faint', style: 'font-size:11.5px' },
          'np and sma/ema/rma/rsi/atr/true_range (+ _fast) are in scope · no imports, no orders, no forward indexing')),
      code,
      el('div', { class: 'inline', style: 'margin-top:12px' },
        el('button', {
          class: 'primary',
          onclick: e => withBusy(e.currentTarget, async () => {
            let params;
            try { params = JSON.parse(paramsInput.value || '{}'); }
            catch (err) { throw new Error('parameters are not valid JSON: ' + err.message); }
            if (!id.value.trim()) throw new Error('a strategy id is required');
            const out = await api.saveStrategy({
              strategy_id: id.value.trim(),
              description: description.value.trim() || 'saved from the UI',
              signal_block: code.value,
              params,
              family: family.value.trim(),
              source_url: sourceUrl.value.trim(),
              parent_version: isNew ? null : Number(parentField.value),
              origin: 'human',
            });
            toast(`Saved v${out.version}` + (out.warnings.length ? ` — ${out.warnings.length} warning(s)` : ''), 'good');
            remember({ strategy: out.strategy_id, version: out.version });
            refreshRailStats();
            navigate('#/strategies/' + encodeURIComponent(out.strategy_id));
          }, 'Validating'),
        }, 'Validate & save'),
        el('span', { class: 'faint', style: 'font-size:11.5px' },
          'The AST validator runs before anything is stored — it rejects imports, ' +
          'file access and forward indexing rather than letting them run.'))));
}

// --- small helpers -----------------------------------------------------------

function coerce(text) {
  const n = Number(text);
  return text.trim() !== '' && Number.isFinite(n) ? n : text;
}
