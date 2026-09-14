// Talking to the backend, and the small helpers every page needs.

const BASE = '';

async function request(method, path, body) {
  const res = await fetch(BASE + path, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { error: text }; }
  if (!res.ok) {
    // The server sends the real message (engine/service raises with the fix in
    // it); surfacing "500 Internal Server Error" instead would waste it.
    throw new Error(data?.error || data?.detail || `${res.status} ${res.statusText}`);
  }
  return data;
}

export const api = {
  health:       ()              => request('GET', '/api/health'),
  stats:        ()              => request('GET', '/api/stats'),

  listData:     ()              => request('GET', '/api/data'),
  loadData:     (body)          => request('POST', '/api/data', body),
  discover:     ()              => request('GET', '/api/data/discover'),
  startLoad:    (body)          => request('POST', '/api/data/start', body),
  job:          (id)            => request('GET', `/api/jobs/${id}`),
  jobs:         ()              => request('GET', '/api/jobs'),
  deleteData:   (name)          => request('DELETE', `/api/data/${encodeURIComponent(name)}`),

  strategies:   (archived)      => request('GET', `/api/strategies?include_archived=${archived ? 'true' : 'false'}`),
  strategy:     (id, v = 0)     => request('GET', `/api/strategies/${encodeURIComponent(id)}?version=${v}`),
  saveStrategy: (body)          => request('POST', '/api/strategies', body),
  archive:      (id, body)      => request('POST', `/api/strategies/${encodeURIComponent(id)}/archive`, body),

  backtest:     (body)          => request('POST', '/api/backtest', body),
  optimize:     (body)          => request('POST', '/api/optimize', body),
  validate:     (body)          => request('POST', '/api/validate', body),
  control:      (body)          => request('POST', '/api/control', body),

  chart:        (q)             => request('GET', '/api/chart?' + new URLSearchParams(q)),
  runs:         (q = {})        => request('GET', '/api/runs?' + new URLSearchParams(q)),
  run:          (id)            => request('GET', `/api/runs/${id}`),
};

// --- formatting --------------------------------------------------------------
//
// One rule throughout: a missing number renders as a dim "—", never as 0.
// "This strategy took no trades" and "this strategy made no profit" are
// different facts, and a table that prints 0.00 for both hides the first.

export function num(v, places = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  if (!Number.isFinite(v)) return '∞';
  return v.toLocaleString(undefined, {
    minimumFractionDigits: places, maximumFractionDigits: places,
  });
}

export function int(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Math.round(v).toLocaleString();
}

export function pct(v, places = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return (v * 100).toFixed(places) + '%';
}

export function money(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  const s = Math.abs(v).toLocaleString(undefined, {
    minimumFractionDigits: 2, maximumFractionDigits: 2,
  });
  return (v < 0 ? '-' : '') + '$' + s;
}

export function ago(seconds) {
  if (!seconds) return '—';
  const d = Math.floor(Date.now() / 1000 - seconds);
  if (d < 60) return 'just now';
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
}

export function signClass(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '';
  return v > 0 ? 'good' : v < 0 ? 'bad' : '';
}

// --- DOM ---------------------------------------------------------------------

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else node.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

export function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

export function toast(message, kind = '') {
  const box = document.getElementById('toasts');
  const node = el('div', { class: `toast ${kind}` }, message);
  box.append(node);
  setTimeout(() => {
    node.style.transition = 'opacity .25s';
    node.style.opacity = '0';
    setTimeout(() => node.remove(), 260);
  }, kind === 'bad' ? 8000 : 3600);
}

/** Run an async action with the button disabled and spinning. */
export async function withBusy(button, fn, label = 'Running') {
  const original = button.innerHTML;
  button.disabled = true;
  button.innerHTML = `<span class="spin"></span> ${escapeHtml(label)}…`;
  try {
    return await fn();
  } catch (err) {
    toast(err.message, 'bad');
    throw err;
  } finally {
    button.disabled = false;
    button.innerHTML = original;
  }
}

/** Metric tiles, in the order a person actually reads them. */
export function metricTiles(m) {
  if (!m) return el('div', { class: 'empty' }, 'No metrics');
  const tiles = [
    ['Net profit', money(m.net_profit), signClass(m.net_profit)],
    ['Trades', int(m.n_trades), ''],
    ['Win rate', pct(m.win_rate), ''],
    ['Profit factor', num(m.profit_factor), m.profit_factor > 1 ? 'good' : 'bad'],
    ['Sharpe', num(m.sharpe), signClass(m.sharpe)],
    ['Sortino', num(m.sortino), signClass(m.sortino)],
    ['Max drawdown', money(m.max_drawdown), m.max_drawdown > 0 ? 'bad' : ''],
    ['Return / MaxDD', num(m.max_drawdown > 0 ? m.net_profit / m.max_drawdown : null),
      signClass(m.max_drawdown > 0 ? m.net_profit / m.max_drawdown : null)],
    ['Expectancy R', num(m.expectancy_r, 3), signClass(m.expectancy_r)],
    ['SQN', num(m.sqn), signClass(m.sqn)],
    ['Calmar', num(m.calmar), signClass(m.calmar)],
    ['Max DD %', pct(m.max_drawdown_pct), m.max_drawdown_pct > 0 ? 'bad' : ''],
    ['Avg R', num(m.avg_r, 3), signClass(m.avg_r)],
    ['Std R', num(m.std_r, 3), ''],
    ['Largest loss', money(m.largest_loss), m.largest_loss ? 'bad' : ''],
    ['Worst streak', int(m.max_consecutive_losses), ''],
    ['Exposure', pct(m.exposure), ''],
  ];
  return el('div', { class: 'metrics' },
    tiles.map(([k, v, cls]) => el('div', { class: 'metric' },
      el('div', { class: 'k' }, k),
      el('div', { class: `v ${cls}` }, v))));
}

/** A sortable table. `columns` is [{key, label, fmt, cls, num}]. */
export function dataTable(rows, columns, { onRow, sortKey, selected } = {}) {
  let sort = { key: sortKey ?? null, asc: false };

  const wrap = el('div', { class: 'table-wrap' });
  const table = el('table');
  const thead = el('thead');
  const tbody = el('tbody');

  function render() {
    let data = [...rows];
    if (sort.key) {
      data.sort((a, b) => {
        const x = a[sort.key], y = b[sort.key];
        if (x === y) return 0;
        if (x === null || x === undefined) return 1;
        if (y === null || y === undefined) return -1;
        const cmp = typeof x === 'number' && typeof y === 'number'
          ? x - y : String(x).localeCompare(String(y));
        return sort.asc ? cmp : -cmp;
      });
    }
    tbody.replaceChildren(...data.map((row, i) => {
      const tr = el('tr', {
        class: [onRow ? 'clickable' : '', selected && selected(row) ? 'selected' : ''].join(' ').trim(),
      }, columns.map(c => el('td', {
        class: [c.num ? 'num' : '', c.text ? 'text' : '', c.cls ? c.cls(row) : ''].join(' ').trim(),
      }, c.fmt ? c.fmt(row, i) : row[c.key] ?? '—')));
      if (onRow) tr.addEventListener('click', () => onRow(row));
      return tr;
    }));

    thead.replaceChildren(el('tr', {}, columns.map(c => el('th', {
      class: ['sortable', sort.key === c.key ? 'sorted' : '', sort.key === c.key && sort.asc ? 'asc' : '']
        .join(' ').trim(),
      onclick: () => {
        sort = sort.key === c.key ? { key: c.key, asc: !sort.asc } : { key: c.key, asc: false };
        render();
      },
    }, c.label))));
  }

  render();
  table.append(thead, tbody);
  wrap.append(table);
  return wrap;
}
