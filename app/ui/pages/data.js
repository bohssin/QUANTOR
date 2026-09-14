// Data sources: load a CSV, read its quality report, see what it can serve.

import { api, el, num, int, pct, ago, toast, withBusy, dataTable } from '../api.js';
import { remember, refreshRailStats } from '../app.js';

export async function renderData(root) {
  const [sources, jobs] = await Promise.all([api.listData(), api.jobs()]);
  const running = jobs.find(j => !j.done);

  const name = el('input', { class: 'mono', placeholder: 'xau' });
  const path = el('input', { class: 'mono', placeholder: '/home/you/data/xau.csv' });
  const timeframe = el('select', {},
    ['M15', 'M1', 'M5', 'M30', 'H1', 'H4', 'D1'].map(t => el('option', { value: t }, t)));
  const baseTf = el('select', {},
    el('option', { value: '' }, 'M1 (default)'),
    ['S1', 'S5', 'S15', 'S30'].map(t => el('option', { value: t }, t)));
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
          el('label', { title: 'Hours east of UTC. A GMT+3 export is 3.' }, 'GMT offset'), offset),
        el('div', { class: 'field' },
          el('label', { title: 'How fine to cache the bars. S1 lets the engine settle stop-vs-target order inside each bar.' },
            'Base timeframe'), baseTf)),
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
                base_timeframe: baseTf.value,
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
        el('b', {}, 'Base timeframe: pick S1 for a tick archive'),
        'This engine fills on bars, so when one signal bar contains both your stop ' +
        'and your target it cannot say which came first — and assumes the stop. ' +
        'Caching S1 underneath lets the data decide instead. It costs roughly ' +
        'sixty times the bars, and it matters most when stops are tight: below ' +
        '0.4xATR on M1, roughly one trade in twelve is decided by an assumption.'),
      el('div', { class: 'verdict' },
        el('b', {}, 'Large files are streamed'),
        'Above ~256 MB the file is read in bounded memory — peak RSS is set by the ' +
        'block size, not the file size, so an 11 GB tick archive costs the same as a ' +
        'small one. Bars are cached at M1, so every coarser timeframe afterwards is free. ' +
        'Paths are read on the machine running this server.')),

    progressCard(running, root),
    discoveredCard(root),

    el('div', { class: 'card' },
      el('h2', {}, 'Loaded sources'),
      sources.length
        ? el('div', { class: 'stack' }, sources.map(s => sourceCard(s, root)))
        : el('div', { class: 'empty' },
            el('strong', {}, 'No data loaded'),
            'Point the form above at a CSV on this machine.')),
  );
}

// --- files found on this machine ---------------------------------------------

function discoveredCard(root) {
  const host = el('div', { class: 'card' },
    el('h2', {}, 'Found on this machine'),
    el('div', { class: 'loading' }, el('span', { class: 'spin' }), ' Looking…'));

  api.discover().then(found => {
    host.replaceChildren(
      el('div', { class: 'card-head' },
        el('h2', {}, 'Found on this machine'),
        el('span', { class: 'faint', style: 'font-size:11.5px' },
          `${found.length} candidate${found.length === 1 ? '' : 's'}, largest first`)),
      found.length
        ? el('div', { class: 'stack' }, found.map(c => candidateCard(c, root)))
        : el('div', { class: 'empty' },
            el('strong', {}, 'Nothing found automatically'),
            'Looked in Documents, Downloads, Desktop and MEGA folders for CSVs ' +
            'with a timestamp column. Use the form above if your file is elsewhere.'));
  }).catch(err => {
    host.replaceChildren(el('div', { class: 'verdict warn' },
      el('b', {}, 'Could not scan for files'), err.message));
  });
  return host;
}

function candidateCard(c, root) {
  const nameInput = el('input', { class: 'mono', value: c.suggested_name });
  const offsetInput = el('input', { class: 'mono', type: 'number', step: '0.25',
                                    value: '0', style: 'max-width:90px' });
  const tfSelect = el('select', { style: 'max-width:110px' },
    ['M1', 'M5', 'M15', 'M30', 'H1'].map(t =>
      el('option', { value: t, selected: t === (c.suggested_timeframe || 'M15') }, t)));
  const baseSelect = el('select', { style: 'max-width:130px' },
    el('option', { value: '' }, 'M1 (smaller)'),
    ['S1', 'S5', 'S15', 'S30'].map(t =>
      el('option', { value: t, selected: t === c.suggested_base }, t)));

  return el('div', { class: 'card tight', style: 'background:var(--surface-2);margin:0' },
    el('div', { class: 'card-head' },
      el('div', {},
        el('h2', { style: 'margin:0' }, c.name, '  ',
          el('span', { class: 'pill info' }, c.kind),
          ' ', el('span', { class: 'pill' }, c.size)),
        el('div', { class: 'faint mono', style: 'font-size:11px;margin-top:4px' }, c.path),
        el('div', { class: 'faint mono', style: 'font-size:11px' },
          `columns: ${c.columns.join(', ')}`),
        el('div', { class: 'faint mono', style: 'font-size:11px' },
          `first row: ${c.first_row}`)),
      el('button', {
        class: 'primary',
        onclick: e => withBusy(e.currentTarget, async () => {
          const job = await api.startLoad({
            name: nameInput.value.trim() || c.suggested_name,
            path: c.path,
            timeframe: tfSelect.value,
            base_timeframe: baseSelect.value,
            utc_offset_hours: Number(offsetInput.value),
          });
          toast(`Loading ${job.label} — this page shows progress`, 'good');
          renderData(root);
        }, 'Starting'),
      }, 'Load this')),

    el('div', { class: 'row', style: 'margin-top:4px' },
      el('div', { class: 'field' }, el('label', {}, 'Name'), nameInput),
      el('div', { class: 'field' }, el('label', {}, 'Timeframe'), tfSelect),
      el('div', { class: 'field' }, el('label', {}, 'Base'), baseSelect),
      el('div', { class: 'field' },
        el('label', { title: 'Hours east of UTC. A GMT+3 export is 3.' }, 'GMT offset'),
        offsetInput)),

    c.note ? el('div', { class: 'verdict', style: 'margin:10px 0 0' }, c.note) : null,
    el('div', { class: 'verdict warn', style: 'margin:10px 0 0' },
      el('b', {}, 'The GMT offset is the one thing the file cannot tell you'),
      'Vendors differ and the file rarely says. Getting it wrong shifts every ' +
      'daily boundary and every session filter. A GMT+3 export needs 3.'));
}

// --- a load in flight ---------------------------------------------------------

function progressCard(job, root) {
  if (!job) return null;
  const host = el('div', { class: 'card' });

  function paint(j) {
    const pctDone = j.total_bytes && j.rows
      ? null                    // rows read is known; bytes consumed is not
      : null;
    host.replaceChildren(
      el('div', { class: 'card-head' },
        el('h2', {}, 'Loading  ', el('span', { class: 'pill live' }, j.status)),
        el('span', { class: 'faint mono', style: 'font-size:11.5px' }, j.label)),
      el('div', { class: 'metrics' },
        el('div', { class: 'metric' }, el('div', { class: 'k' }, 'Rows read'),
          el('div', { class: 'v' }, int(j.rows))),
        el('div', { class: 'metric' }, el('div', { class: 'k' }, 'Bars built'),
          el('div', { class: 'v' }, int(j.bars))),
        el('div', { class: 'metric' }, el('div', { class: 'k' }, 'Rows / second'),
          el('div', { class: 'v' }, int(j.rows_per_second))),
        el('div', { class: 'metric' }, el('div', { class: 'k' }, 'Elapsed'),
          el('div', { class: 'v' }, `${Math.round(j.seconds)}s`)),
        el('div', { class: 'metric' }, el('div', { class: 'k' }, 'File'),
          el('div', { class: 'v' }, `${(j.total_bytes / (1 << 30)).toFixed(2)} GB`))),
      el('div', { class: 'faint', style: 'margin-top:9px;font-size:11.5px' },
        j.rows && j.message === 'reading'
          ? 'Reading and folding into bars. When the row count stops rising the ' +
            'read is done and the bars are being written to the library — that ' +
            'part reports no rows, and is the shorter half.'
          : 'Starting…'));
  }
  paint(job);

  const timer = setInterval(async () => {
    try {
      const j = await api.job(job.job_id);
      paint(j);
      if (j.done) {
        clearInterval(timer);
        if (j.status === 'ok') {
          toast(`Loaded ${j.result.name}: ${int(j.result.rows)} rows -> ` +
                `${int(j.result.base_bars)} ${j.result.base_timeframe} bars ` +
                `in ${j.result.seconds}s`, 'good');
        } else {
          toast(j.error || 'the load failed', 'bad');
        }
        refreshRailStats();
        renderData(root);
      }
    } catch (err) {
      clearInterval(timer);
    }
  }, 1500);

  return host;
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
