// Router and shell. Plain ES modules, no build step — see the plan's §15 note:
// a bundler here would be one more thing to break for no benefit at this size.

import { api, el, int, toast } from './api.js';
import { renderStrategies } from './pages/strategies.js';
import { renderChart } from './pages/chart.js';
import { renderData } from './pages/data.js';
import { renderHistory } from './pages/history.js';
import { mountAgent } from './agent.js';

const ROUTES = {
  strategies: renderStrategies,
  chart: renderChart,
  data: renderData,
  history: renderHistory,
};

const main = document.getElementById('main');

/** Shared, so the chart page can open what the strategies page selected. */
export const state = {
  strategy: localStorage.getItem('q.strategy') || '',
  version: Number(localStorage.getItem('q.version') || 0),
  data: localStorage.getItem('q.data') || '',
  timeframe: localStorage.getItem('q.timeframe') || '',
};

export function remember(patch) {
  Object.assign(state, patch);
  for (const [k, v] of Object.entries(patch)) {
    if (v === null || v === undefined) localStorage.removeItem('q.' + k);
    else localStorage.setItem('q.' + k, String(v));
  }
}

function parseHash() {
  const raw = (location.hash || '#/strategies').slice(2);
  const [path, query] = raw.split('?');
  const parts = path.split('/').filter(Boolean);
  return {
    route: parts[0] || 'strategies',
    rest: parts.slice(1),
    params: Object.fromEntries(new URLSearchParams(query || '')),
  };
}

export function navigate(hash) {
  if (location.hash === hash) route();
  else location.hash = hash;
}

async function route() {
  const { route: name, rest, params } = parseHash();
  const render = ROUTES[name] || renderStrategies;

  document.querySelectorAll('.rail-link').forEach(a => {
    a.classList.toggle('active', a.dataset.route === name);
  });

  main.replaceChildren(el('div', { class: 'loading' },
    el('span', { class: 'spin' }), ' Loading…'));
  try {
    await render(main, { rest, params });
  } catch (err) {
    console.error(err);
    main.replaceChildren(el('div', { class: 'card' },
      el('div', { class: 'verdict bad' },
        el('b', {}, 'This page could not load'),
        err.message)));
  }
}

export async function refreshRailStats() {
  try {
    const s = await api.stats();
    document.getElementById('rail-stats').textContent =
      `${int(s.strategies)} strategies\n${int(s.runs)} runs\n` +
      `${int(s.data_sources)} sources\n${int(s.artifacts)} artifacts`;
  } catch { /* the rail is decoration; never let it break a page */ }
}

// --- boot --------------------------------------------------------------------

window.addEventListener('hashchange', route);

const toggle = document.getElementById('toggle-agent');
if (localStorage.getItem('q.agentHidden') === '1') document.body.classList.add('agent-hidden');
toggle.addEventListener('click', () => {
  const hidden = document.body.classList.toggle('agent-hidden');
  localStorage.setItem('q.agentHidden', hidden ? '1' : '0');
  window.dispatchEvent(new Event('resize'));   // let the charts re-measure
});

(async function boot() {
  try {
    const health = await api.health();
    if (!health.agent_available) {
      document.getElementById('agent-state').textContent = 'no CLI';
      document.getElementById('agent-state').className = 'pill warn';
    }
  } catch (err) {
    toast('The API is not responding: ' + err.message, 'bad');
  }
  mountAgent();
  refreshRailStats();
  route();
})();
