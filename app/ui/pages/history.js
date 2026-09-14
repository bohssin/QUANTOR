// Every run ever made, reopenable. Plan §16 — the record is the point.

import {
  api, el, num, int, money, pct, ago, signClass, dataTable, metricTiles,
} from '../api.js';
import { navigate } from '../app.js';
import { runSummary } from './strategies.js';

export async function renderHistory(root, { rest }) {
  const runs = await api.runs({ limit: 200 });
  const detail = el('div', {});

  const kindFilter = el('select', {},
    el('option', { value: '' }, 'all kinds'),
    ['backtest', 'optimize', 'validate'].map(k => el('option', { value: k }, k)));

  const table = el('div', {});

  function paint() {
    const rows = kindFilter.value ? runs.filter(r => r.kind === kindFilter.value) : runs;
    table.replaceChildren(rows.length ? dataTable(rows, [
      { key: 'run_id', label: 'id', num: true },
      { key: 'kind', label: 'kind', fmt: r => el('span', { class: 'pill' }, r.kind) },
      { key: 'strategy_id', label: 'strategy' },
      { key: 'version', label: 'ver', num: true, fmt: r => 'v' + r.version },
      { key: 'data', label: 'data' },
      { key: 'timeframe', label: 'tf' },
      { key: 'status', label: 'status',
        fmt: r => el('span', { class: 'pill ' + (r.status === 'ok' ? 'ok' : 'bad') }, r.status) },
      { key: 'summary', label: 'result', text: true, fmt: runSummary },
      { key: 'seconds', label: 'took', num: true, fmt: r => r.seconds ? r.seconds + 's' : '—' },
      { key: 'started_at', label: 'when', fmt: r => ago(r.started_at) },
    ], { sortKey: 'run_id', onRow: r => open(r.run_id) })
      : el('div', { class: 'empty' },
          el('strong', {}, 'Nothing recorded yet'),
          'Runs from the UI and from the assistant both land here.'));
  }

  async function open(runId) {
    detail.replaceChildren(el('div', { class: 'card' },
      el('div', { class: 'loading' }, el('span', { class: 'spin' }), ' Loading run…')));
    const run = await api.run(runId);
    detail.replaceChildren(runDetail(run));
    detail.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  kindFilter.addEventListener('change', paint);
  paint();

  root.replaceChildren(
    el('div', { class: 'page-head' },
      el('div', {},
        el('h1', {}, 'History'),
        el('div', { class: 'sub' },
          `${runs.length} runs — including the ones that failed, which is the point`)),
      kindFilter),
    detail,
    el('div', { class: 'card' }, table),
  );

  if (rest[0]) await open(Number(rest[0]));
}

function runDetail(run) {
  const parts = [
    el('div', { class: 'card-head' },
      el('div', {},
        el('h2', {}, `Run ${run.run_id} — ${run.kind}  `,
          el('span', { class: 'pill ' + (run.status === 'ok' ? 'ok' : 'bad') }, run.status)),
        el('div', { class: 'sub dim mono', style: 'font-size:11.5px' },
          `${run.strategy_id} v${run.version} · ${run.data} · ${run.timeframe} · ` +
          `${run.seconds ?? '—'}s · ${ago(run.started_at)}`)),
      el('button', {
        onclick: () => navigate(`#/strategies/${encodeURIComponent(run.strategy_id)}`),
      }, 'Open strategy')),
  ];

  if (run.status !== 'ok') {
    parts.push(el('div', { class: 'verdict bad' }, el('b', {}, 'This run failed'), run.error));
  }

  if (run.kind === 'backtest' && run.metrics) parts.push(metricTiles(run.metrics));

  if (run.kind === 'optimize') {
    const o = run.optimization || {};
    parts.push(el('div', { class: 'verdict' },
      el('b', {}, `${int(o.comparisons)} configurations, objective ${o.objective}`),
      o.verdict || 'no plateau verdict recorded'));
    if (run.results?.length) {
      const top = [...run.results]
        .filter(r => r.score !== null)
        .sort((a, b) => b.score - a.score).slice(0, 25);
      parts.push(el('h3', {}, `Top ${top.length} of ${run.results.length}`));
      parts.push(dataTable(top, [
        { key: 'params', label: 'parameters', text: true,
          fmt: r => Object.entries(r.params).map(([k, v]) => `${k}=${v}`).join('  ') },
        { key: 'score', label: 'score', num: true, fmt: r => num(r.score, 4), cls: r => signClass(r.score) },
        { key: 'net_profit', label: 'net', num: true, fmt: r => money(r.net_profit), cls: r => signClass(r.net_profit) },
        { key: 'max_drawdown', label: 'max dd', num: true, fmt: r => money(r.max_drawdown) },
        { key: 'sharpe', label: 'sharpe', num: true, fmt: r => num(r.sharpe) },
        { key: 'n_trades', label: 'trades', num: true, fmt: r => int(r.n_trades) },
      ], { sortKey: 'score' }));
    }
  }

  if (run.kind === 'validate') {
    const v = run.validation || {};
    const cls = (v.verdict || '').startsWith('generalized') ? 'good'
      : (v.verdict || '').startsWith('did not') ? 'bad' : 'warn';
    parts.push(el('div', { class: 'verdict ' + cls },
      el('b', {}, v.verdict || 'no verdict'),
      `efficiency ${num(v.efficiency, 3)} over ${v.folds} folds, ` +
      `${int(v.comparisons)} comparisons`));
    if (run.per_fold?.length) {
      parts.push(dataTable(run.per_fold, [
        { key: 'fold', label: 'fold', num: true },
        { key: 'params', label: 'chosen', text: true,
          fmt: r => Object.entries(r.params).map(([k, x]) => `${k}=${x}`).join('  ') },
        { key: 'train', label: 'train', num: true, fmt: r => num(r.train, 3), cls: r => signClass(r.train) },
        { key: 'test', label: 'test (OOS)', num: true, fmt: r => num(r.test, 3), cls: r => signClass(r.test) },
        { key: 'trades', label: 'trades', num: true, fmt: r => int(r.trades) },
      ], { sortKey: 'fold' }));
    }
  }

  parts.push(el('div', { class: 'hr' }));
  parts.push(el('h3', {}, 'Parameters'));
  parts.push(el('pre', {
    class: 'mono', style: 'margin:0;font-size:11.5px;background:#090c11;padding:10px;border-radius:7px;overflow:auto',
  }, JSON.stringify(run.params, null, 2)));

  return el('div', { class: 'card' }, parts);
}
