// The chart: candles, the run's own trades, and its equity below. Plan §14.4.
//
// The markers come from the engine's trade list, not from re-deriving entries in
// JavaScript. That distinction is the whole point — a chart that recomputes
// signals in the browser will eventually disagree with the backtest, and then
// the picture and the number are both untrustworthy with no way to tell which.

import {
  createChart, CandlestickSeries, LineSeries, HistogramSeries,
  createSeriesMarkers, CrosshairMode, LineStyle,
} from '../vendor/lightweight-charts.standalone.production.mjs';
import { api, el, num, int, money, pct, signClass, toast, withBusy, metricTiles } from '../api.js';
import { state, remember } from '../app.js';

const THEME = {
  layout: { background: { color: '#11151d' }, textColor: '#9aa5b8', fontSize: 11,
            fontFamily: 'ui-monospace, Menlo, Consolas, monospace' },
  grid: { vertLines: { color: '#1b2230' }, horzLines: { color: '#1b2230' } },
  crosshair: { mode: CrosshairMode.Normal,
               vertLine: { color: '#3a4658', labelBackgroundColor: '#242c3a' },
               horzLine: { color: '#3a4658', labelBackgroundColor: '#242c3a' } },
  rightPriceScale: { borderColor: '#242c3a' },
  timeScale: { borderColor: '#242c3a', timeVisible: true, secondsVisible: false },
};

let priceChart = null;
let equityChart = null;
let onResize = null;

export async function renderChart(root, { params }) {
  const sources = await api.listData();
  if (!sources.length) {
    root.replaceChildren(el('div', { class: 'card' }, el('div', { class: 'empty' },
      el('strong', {}, 'No data loaded'),
      'Load a CSV on the Data page first.')));
    return;
  }

  const strategies = await api.strategies(false);
  const pick = {
    data: params.data || state.data || sources[0].name,
    strategy: params.strategy ?? state.strategy ?? '',
    version: Number(params.version ?? state.version ?? 0),
    timeframe: params.timeframe || state.timeframe || '',
    limit: 1500,
    offset: 0,
  };
  if (!sources.some(s => s.name === pick.data)) pick.data = sources[0].name;
  if (pick.strategy && !strategies.some(s => s.strategy_id === pick.strategy)) pick.strategy = '';

  const dataSelect = el('select', {},
    sources.map(s => el('option', { value: s.name, selected: s.name === pick.data },
      `${s.name} — ${int(s.bars)} ${s.timeframe} bars`)));
  const stratSelect = el('select', {},
    el('option', { value: '' }, 'no strategy — candles only'),
    strategies.map(s => el('option', { value: s.strategy_id, selected: s.strategy_id === pick.strategy },
      `${s.strategy_id} (v${s.latest.version})`)));
  const tfSelect = el('select', {},
    el('option', { value: '' }, 'source default'),
    ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'].map(t =>
      el('option', { value: t, selected: t === pick.timeframe }, t)));
  const barsSelect = el('select', {},
    [500, 1000, 1500, 3000, 6000].map(n =>
      el('option', { value: n, selected: n === pick.limit }, `${int(n)} bars`)));

  const legend = el('div', { class: 'chart-legend' }, 'loading…');
  const priceBox = el('div', { class: 'chart-box' }, legend, el('div', { id: 'price-chart' }));
  const equityBox = el('div', { class: 'chart-box' }, el('div', { id: 'equity-chart' }));
  const summary = el('div', {});
  const tradesHost = el('div', {});

  const older = el('button', { title: 'Older bars' }, '← older');
  const newer = el('button', { title: 'Newer bars' }, 'newer →');

  root.replaceChildren(
    el('div', { class: 'page-head' },
      el('div', {}, el('h1', {}, 'Chart'),
        el('div', { class: 'sub' },
          "The run's own trade list, drawn on the bars it was measured on")),
      el('div', { class: 'inline' }, older, newer)),

    el('div', { class: 'card tight' },
      el('div', { class: 'row' },
        el('div', { class: 'field' }, el('label', {}, 'Data'), dataSelect),
        el('div', { class: 'field' }, el('label', {}, 'Strategy'), stratSelect),
        el('div', { class: 'field' }, el('label', {}, 'Timeframe'), tfSelect),
        el('div', { class: 'field' }, el('label', {}, 'Window'), barsSelect),
        el('div', { class: 'field', style: 'flex:0 0 auto' },
          el('label', {}, ' '),
          el('button', { class: 'primary', id: 'chart-apply', onclick: e => withBusy(e.currentTarget, load, 'Loading') }, 'Apply')))),

    summary,
    el('div', { class: 'chart-shell' }, priceBox, equityBox),
    tradesHost,
  );

  for (const control of [dataSelect, stratSelect, tfSelect, barsSelect]) {
    control.addEventListener('change', () => { pick.offset = 0; load(); });
  }
  older.addEventListener('click', () => { pick.offset += Math.floor(pick.limit * 0.8); load(); });
  newer.addEventListener('click', () => {
    pick.offset = Math.max(0, pick.offset - Math.floor(pick.limit * 0.8));
    load();
  });

  await load();

  async function load() {
    pick.data = dataSelect.value;
    pick.strategy = stratSelect.value;
    pick.timeframe = tfSelect.value;
    pick.limit = Number(barsSelect.value);
    remember({ data: pick.data, strategy: pick.strategy, timeframe: pick.timeframe });

    const payload = await api.chart({
      data: pick.data,
      strategy_id: pick.strategy,
      version: pick.version,
      timeframe: pick.timeframe,
      limit: pick.limit,
      offset: pick.offset,
      max_markers: 400,
    });
    draw(payload, { priceBox, equityBox, legend, summary, tradesHost });
    older.disabled = payload.start === 0;
    newer.disabled = pick.offset === 0;
  }
}

function draw(payload, { priceBox, equityBox, legend, summary, tradesHost }) {
  const priceEl = priceBox.querySelector('#price-chart');
  const equityEl = equityBox.querySelector('#equity-chart');

  if (priceChart) { priceChart.remove(); priceChart = null; }
  if (equityChart) { equityChart.remove(); equityChart = null; }
  if (onResize) window.removeEventListener('resize', onResize);

  priceChart = createChart(priceEl, { ...THEME, height: priceEl.clientHeight || 460 });
  const candles = priceChart.addSeries(CandlestickSeries, {
    upColor: '#26a69a', downColor: '#ef5350',
    borderUpColor: '#26a69a', borderDownColor: '#ef5350',
    wickUpColor: '#26a69a', wickDownColor: '#ef5350',
  });

  const b = payload.bars;
  const bars = b.time.map((t, i) => ({
    time: t, open: b.open[i], high: b.high[i], low: b.low[i], close: b.close[i],
  }));
  candles.setData(bars);

  if (payload.markers?.length) {
    // Shapes only, no labels. With 69 trades in view the per-marker text
    // overlapped into an unreadable wall that hid the candles it was
    // annotating; the trade table below carries the same numbers, legibly,
    // and hovering a row scrolls the chart to it.
    createSeriesMarkers(candles, payload.markers.map(m => ({
      time: m.time, position: m.position, shape: m.shape, color: m.color,
    })));
  }

  priceChart.timeScale().fitContent();

  // --- equity, in its own pane so its scale never distorts the price ---
  equityChart = createChart(equityEl, {
    ...THEME, height: equityEl.clientHeight || 150,
    rightPriceScale: { ...THEME.rightPriceScale, scaleMargins: { top: 0.15, bottom: 0.05 } },
  });
  const equityData = payload.equity || [];
  const line = equityChart.addSeries(LineSeries, {
    color: equityData.length && equityData[equityData.length - 1].value >= equityData[0].value
      ? '#4ade80' : '#f87171',
    lineWidth: 2, priceLineVisible: false,
  });
  line.setData(equityData);
  equityChart.timeScale().fitContent();

  // Pan the two together, so reading a drawdown against its bars is possible.
  let syncing = false;
  const sync = (from, to) => range => {
    if (syncing || !range) return;
    syncing = true;
    try { to.timeScale().setVisibleLogicalRange(range); } finally { syncing = false; }
  };
  priceChart.timeScale().subscribeVisibleLogicalRangeChange(sync(priceChart, equityChart));
  equityChart.timeScale().subscribeVisibleLogicalRangeChange(sync(equityChart, priceChart));

  onResize = () => {
    priceChart?.applyOptions({ width: priceEl.clientWidth });
    equityChart?.applyOptions({ width: equityEl.clientWidth });
  };
  window.addEventListener('resize', onResize);
  onResize();

  // --- legend ---
  const last = bars[bars.length - 1];
  function setLegend(bar, trade) {
    // `append` stringifies null into a literal "null" on the page, so the
    // optional rows are filtered out rather than passed through as blanks.
    const rows = [
      el('div', {}, el('b', {}, payload.data), ' · ', payload.timeframe, ' · ',
        `${int(payload.start)}–${int(payload.end)} of ${int(payload.total_bars)}`),
      bar && el('div', {},
        'O ', el('b', {}, num(bar.open, 3)), '  H ', el('b', {}, num(bar.high, 3)),
        '  L ', el('b', {}, num(bar.low, 3)), '  C ', el('b', {}, num(bar.close, 3))),
      payload.strategy && el('div', {}, payload.strategy, ' · ',
        `${int(payload.shown_trades)} of ${int(payload.total_trades)} trades shown`),
      trade && el('div', { class: signClass(trade.pnl) },
        `${trade.side} ${num(trade.entry, 3)} → ${num(trade.exit, 3)} ` +
        `${money(trade.pnl)} (${trade.reason})`),
    ].filter(Boolean);
    legend.replaceChildren(...rows);
  }
  setLegend(last, null);
  priceChart.subscribeCrosshairMove(param => {
    const bar = param.seriesData?.get(candles);
    setLegend(bar || last, null);
  });

  // One trade's levels at a time. Drawing every trade's entry line at once
  // turned the chart into a hatch pattern; drawing the selected one answers
  // the question you actually have, which is "where was this trade's stop".
  let levels = [];
  function showTrade(t) {
    for (const line of levels) candles.removePriceLine(line);
    const span = Math.max(t.exit_time - t.entry_time, 3600) * 3;
    levels = [
      candles.createPriceLine({
        price: t.entry, color: t.side === 'long' ? '#26a69a' : '#ef5350',
        lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true,
        title: `entry ${t.side}`,
      }),
      candles.createPriceLine({
        price: t.exit, color: t.pnl >= 0 ? '#22d3ee' : '#fb923c',
        lineWidth: 1, lineStyle: LineStyle.Dotted, axisLabelVisible: true,
        title: `exit ${t.reason}`,
      }),
    ];
    priceChart.timeScale().setVisibleRange({
      from: t.entry_time - span, to: t.exit_time + span,
    });
    setLegend(last, t);
  }

  // --- summary + trade table ---
  if (payload.metrics) {
    summary.replaceChildren(el('div', { class: 'card tight' },
      el('div', { class: 'card-head' },
        el('h2', {}, payload.strategy),
        el('span', { class: 'faint', style: 'font-size:11.5px' },
          'measured over the whole source, not just the window shown')),
      metricTiles(payload.metrics)));
  } else {
    summary.replaceChildren();
  }

  if (payload.trades?.length) {
    const wins = payload.trades.filter(t => t.pnl > 0).length;
    tradesHost.replaceChildren(el('div', { class: 'card tight' },
      el('div', { class: 'card-head' },
        el('h3', {}, `Trades in this window — ${payload.trades.length}`),
        el('span', { class: 'faint', style: 'font-size:11.5px' },
          `${wins} winners · ${payload.trades.length - wins} losers · ` +
          `net ${money(payload.trades.reduce((a, t) => a + t.pnl, 0))}`)),
      el('div', { class: 'scroll-y table-wrap' },
        el('table', {},
          el('thead', {}, el('tr', {},
            ['#', 'side', 'entry', 'exit', 'lots', 'P&L', 'exit reason', 'time']
              .map(h => el('th', {}, h)))),
          el('tbody', {}, payload.trades.map(t => el('tr', {
            class: 'clickable',
            onclick: () => showTrade(t),
            onmouseenter: () => setLegend(last, t),
          },
            el('td', {}, String(t.i)),
            el('td', { class: t.side === 'long' ? 'good' : 'bad' }, t.side),
            el('td', { class: 'num' }, num(t.entry, 3)),
            el('td', { class: 'num' }, num(t.exit, 3)),
            el('td', { class: 'num' }, num(t.lots, 3)),
            el('td', { class: 'num ' + signClass(t.pnl) }, money(t.pnl)),
            el('td', {}, el('span', { class: 'pill' }, t.reason)),
            el('td', { class: 'faint' },
              new Date(t.entry_time * 1000).toISOString().slice(0, 16).replace('T', ' '))))))) ));
  } else {
    tradesHost.replaceChildren(payload.strategy
      ? el('div', { class: 'card tight' }, el('div', { class: 'empty' },
          el('strong', {}, 'No trades in this window'),
          'Pan with ← older, or widen the window.'))
      : el('div', { class: 'card tight' }, el('div', { class: 'empty' },
          el('strong', {}, 'Candles only'),
          'Pick a strategy above to draw its trades.')));
  }
}
