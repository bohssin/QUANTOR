// Data sources: load a CSV, read its quality report, see what it can serve.

import { api, el, num, int, pct, ago, toast, withBusy, dataTable } from '../api.js';
import { remember, refreshRailStats } from '../app.js';

export async function renderData(root) {
  const sources = await api.listData();

  const name = el('input', { class: 'mono', placeholder: 'xau' });
  const path = el('input', { class: 'mono', placeholder: '/home/you/data/xau.csv' });
  const timeframe = el('select', {},
    ['M15', 'M1', 'M5', 'M30', 'H1', 'H4', 'D1'].map(t => el('option', { value: t }, t)));
  const offset = el('input', { class: 'mono', type: 'number', step: '0.25', value: '0' });
  const contract = el('input', { class: 'mono', type: 'number', value: '100' });
  const tickValue = el('input', { class: 'mono', type: 'number', step: '0.01', value: '0.10' });
  const commission = el('input', { class: 'mono', type: 'number', step: '0.1', value: '3.5' });
  const spread = el('input', { class: 'mono', type: 'number', step: '0.01', value: '0.30' });

  root.replaceChildren(
    el('div', { class: 'page-head' },
      el('div', {},
        el('h1', {}, 'Data'),
        el('div', { class: 'sub' },
          'Tick or bar CSVs. Columns and price precision are read from the file; ' +
          'the clock and the contract details are not in it and have to be told.'))),

    el('div', { class: 'card' },
      el('h2', {}, 'Load a CSV'),
      el('div', { class: 'row' },
        el('div', { class: 'field' }, el('label', {}, 'Name'), name),
        el('div', { class: 'field', style: 'flex:3' },
          el('label', {}, 'Path on this machine'), path),
        el('div', { class: 'field' }, el('label', {}, 'Timeframe'), timeframe),
        el('div', { class: 'field' },
          el('label', { title: 'Hours east of UTC. A GMT+3 export is 3.' }, 'GMT offset'), offset)),
      el('div', { class: 'row' },
        el('div', { class: 'field' }, el('label', {}, 'Contract size'), contract),
        el('div', { class: 'field' }, el('label', {}, 'Tick value'), tickValue),
        el('div', { class: 'field' }, el('label', {}, 'Commission / lot / side'), commission),
        el('div', { class: 'field' }, el('label', {}, 'Default spread'), spread),
        el('div', { class: 'field', style: 'flex:0 0 auto' },
          el('label', {}, ' '),
          el('button', {
            class: 'primary',
            onclick: e => withBusy(e.currentTarget, async () => {
              if (!name.value.trim() || !path.value.trim())
                throw new Error('a name and a path are required');
              const out = await api.loadData({
                name: name.value.trim(), path: path.value.trim(),
                timeframe: timeframe.value,
                utc_offset_hours: Number(offset.value),
                contract_size: Number(contract.value),
                tick_value: Number(tickValue.value),
                commission_per_lot_per_side: Number(commission.value),
                default_spread: Number(spread.value),
              });
              toast(`Loaded ${out.name}: ${int(out.rows)} rows → ${int(out.base_bars)} ` +
                    `${out.base_timeframe} bars in ${out.seconds}s`, 'good');
              remember({ data: out.name });
              refreshRailStats();
              renderData(root);
            }, 'Reading'),
          }, 'Load'))),
      el('div', { class: 'verdict' },
        el('b', {}, 'Large files are streamed'),
        'Above ~256 MB the file is read in bounded memory — peak RSS is set by the ' +
        'block size, not the file size, so an 11 GB tick archive costs the same as a ' +
        'small one. Bars are cached at M1, so every coarser timeframe afterwards is free. ' +
        'Paths are read on the machine running this server.')),

    el('div', { class: 'card' },
      el('h2', {}, 'Loaded sources'),
      sources.length
        ? el('div', { class: 'stack' }, sources.map(s => sourceCard(s, root)))
        : el('div', { class: 'empty' },
            el('strong', {}, 'No data loaded'),
            'Point the form above at a CSV on this machine.')),
  );
}

function sourceCard(s, root) {
  const q = s.quality || {};
  const warnings = [];
  if (q.out_of_order > 0) warnings.push(`${int(q.out_of_order)} rows out of order`);
  if (q.negative_spread > 0) warnings.push(`${int(q.negative_spread)} negative spreads`);
  if (q.bad_price > 0) warnings.push(`${int(q.bad_price)} non-positive prices`);
  if (q.duplicate_timestamps > 0)
    warnings.push(`${int(q.duplicate_timestamps)} duplicate timestamps (max run ${q.max_duplicate_run})`);

  return el('div', { class: 'card tight', style: 'background:var(--surface-2);margin:0' },
    el('div', { class: 'card-head' },
      el('div', {},
        el('h2', { style: 'margin:0' }, s.name, '  ',
          el('span', { class: 'pill info' }, s.kind),
          ' ', el('span', { class: 'pill' }, `base ${s.base_timeframe}`),
          ' ', el('span', { class: 'pill' }, `default ${s.timeframe}`),
          s.ingest === 'stream' ? el('span', { class: 'pill ok' }, ' streamed') : null),
        el('div', { class: 'meta faint mono', style: 'font-size:11px;margin-top:4px' }, s.path)),
      el('button', {
        class: 'danger',
        onclick: async e => {
          if (!confirm(`Forget "${s.name}"? Runs that used it are kept.`)) return;
          await withBusy(e.currentTarget, async () => {
            await api.deleteData(s.name);
            toast(`Forgot ${s.name}`, 'good');
            refreshRailStats();
            renderData(root);
          }, 'Removing');
        },
      }, 'Forget')),

    el('div', { class: 'metrics' },
      tile('Rows', int(s.rows)),
      tile('Bars', int(s.bars)),
      tile('Tick size', String(s.tick_size)),
      tile('GMT offset', (s.utc_offset_hours >= 0 ? '+' : '') + s.utc_offset_hours),
      tile('Span', `${s.span_days}d`),
      tile('Spread p50', num(q.spread_p50, 5)),
      tile('Spread p99', num(q.spread_p99, 5)),
      tile('Ingest', s.ingest)),

    el('div', { class: 'faint mono', style: 'font-size:11px;margin-top:9px' },
      `${s.span}  ·  serves ${s.serves.slice(0, 10).join(', ')}${s.serves.length > 10 ? ', …' : ''}`),

    warnings.length
      ? el('div', { class: 'verdict warn', style: 'margin:10px 0 0' },
          el('b', {}, 'Quality notes'),
          warnings.join(' · ') +
          '. A stream cannot sort, so out-of-order rows are reported rather than repaired.')
      : el('div', { class: 'verdict good', style: 'margin:10px 0 0' },
          el('b', {}, 'Clean'),
          'No out-of-order rows, duplicate timestamps, negative spreads or bad prices.'));
}

function tile(k, v) {
  return el('div', { class: 'metric' },
    el('div', { class: 'k' }, k), el('div', { class: 'v' }, v));
}
