/* gallery.js — grid rendering, filtering, detail view */
import { navigate } from './app.js';
import { t } from './i18n.js';

const HF = 'https://huggingface.co/epigene/4cgt/resolve/main/showcase/';
const IS_TOUCH = matchMedia('(hover: none)').matches;
let _gallery = [];
let _filter = 'all';
let _search = '';

export async function init() {
  const res = await fetch('data/gallery.json');
  _gallery = await res.json();
  buildFilterBar();
  renderGrid();
}

export function route(path) {
  if (path.startsWith('/gallery/')) {
    const id = path.slice(9);
    showDetail(id);
    return true;
  }
  if (path === '/' || path === '/gallery') {
    document.getElementById('view-home').classList.add('active');
    renderGrid();
    return true;
  }
  return false;
}

/* Filter bar */
function buildFilterBar() {
  const bar = document.getElementById('filter-bar');
  bar.innerHTML = '';
  const series = [...new Set(_gallery.map(g => g.series).filter(Boolean))];

  addFilterBtn(bar, 'all', t('gallery.all'));
  series.forEach(s => addFilterBtn(bar, s, s));
  addFilterBtn(bar, '__originals', t('gallery.originals'));

  const input = document.createElement('input');
  input.className = 'filter-search';
  input.type = 'text';
  input.setAttribute('data-i18n', 'gallery.search');
  input.placeholder = t('gallery.search');
  input.addEventListener('input', e => { _search = e.target.value.toLowerCase(); renderGrid(); });
  bar.appendChild(input);
}

function addFilterBtn(bar, key, label) {
  const btn = document.createElement('button');
  btn.className = 'filter-btn' + (key === _filter ? ' active' : '');
  btn.textContent = label;
  btn.addEventListener('click', () => {
    _filter = key;
    bar.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    renderGrid();
  });
  bar.appendChild(btn);
}

/* Grid */
function renderGrid() {
  const grid = document.getElementById('gallery-grid');
  grid.innerHTML = '';
  const filtered = _gallery.filter(g => {
    if (_filter === '__originals') return !g.series;
    if (_filter !== 'all' && g.series !== _filter) return false;
    if (_search) {
      const hay = [g.character, g.series, ...g.tags, g.id].filter(Boolean).join(' ').toLowerCase();
      if (!hay.includes(_search)) return false;
    }
    return true;
  });

  filtered.forEach(g => {
    const fig = document.createElement('figure');
    fig.className = 'gallery-item';
    const img = document.createElement('img');
    img.src = HF + g.image;
    img.alt = (g.character || g.id) + (g.series ? ' — ' + g.series : '');
    img.loading = 'lazy';
    img.decoding = 'async';
    const cap = document.createElement('figcaption');
    cap.textContent = g.character || g.tags[0] || g.id;
    fig.appendChild(img);
    fig.appendChild(cap);
    fig.addEventListener('click', () => navigate('#/gallery/' + g.id));
    if (!IS_TOUCH) addTilt(fig);
    grid.appendChild(fig);
  });
}

/* 3D perspective tilt — desktop only */
function addTilt(el) {
  el.addEventListener('mousemove', e => {
    const r = el.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width - 0.5;
    const y = (e.clientY - r.top) / r.height - 0.5;
    el.style.transform = `perspective(800px) rotateY(${x * 10}deg) rotateX(${-y * 10}deg) scale(1.03)`;
  });
  el.addEventListener('mouseleave', () => { el.style.transform = ''; });
}

/* Detail view */
function showDetail(id) {
  const g = _gallery.find(e => e.id === id);
  if (!g) { navigate('#/'); return; }

  const view = document.getElementById('view-detail');
  view.classList.add('active');

  const idx = _gallery.indexOf(g);
  const prev = idx > 0 ? _gallery[idx - 1] : null;
  const next = idx < _gallery.length - 1 ? _gallery[idx + 1] : null;

  view.innerHTML = `
    <a class="detail-back" id="detail-back">&larr; ${t('nav.gallery')}</a>
    <div class="detail-frame">
      <img src="${HF}${g.image}" alt="${g.character || g.id}" />
    </div>
    <div class="detail-meta">
      <div class="detail-info">
        <h2>${g.character || g.tags[0] || g.id}</h2>
        ${g.series ? `<div class="series">${g.series}</div>` : ''}
        <div class="date">${g.date}</div>
        ${g.commentary ? `<div class="detail-commentary">${g.commentary}</div>` : ''}
        <a class="detail-commit" href="https://github.com/teenu/4cgt/tree/${g.commit}" target="_blank" rel="noopener">${t('gallery.commit')}: ${g.commit}</a>
        <div class="detail-reproduce" style="margin-top:16px">
          <a href="https://github.com/teenu/4cgt" class="btn btn-primary" style="padding:8px 20px;font-size:13px" target="_blank" rel="noopener">${t('gallery.reproduce')}</a>
        </div>
      </div>
      <div class="detail-settings">
        <table>
          <tr><td>${t('gallery.seed')}</td><td>${g.settings.seed}</td></tr>
          <tr><td>${t('gallery.steps')}</td><td>${g.settings.steps}</td></tr>
          <tr><td>${t('gallery.cfg')}</td><td>${g.settings.cfg_scale}</td></tr>
          <tr><td>${t('gallery.rescale')}</td><td>${g.settings.rescale_cfg}</td></tr>
          <tr><td>${t('gallery.resolution')}</td><td>${g.settings.width}x${g.settings.height}</td></tr>
          <tr><td>${t('gallery.model')}</td><td>${g.settings.model}</td></tr>
          <tr><td>${t('gallery.scheduler')}</td><td>${g.settings.scheduler}</td></tr>
          ${g.dora.enabled ? `
          <tr><td>${t('gallery.dora')}</td><td>${g.dora.adapter}</td></tr>
          <tr><td>${t('gallery.mode')}</td><td>${g.dora.mode}</td></tr>
          <tr><td>${t('gallery.strength')}</td><td>${g.dora.strength}</td></tr>
          <tr><td>${t('gallery.startStep')}</td><td>${g.dora.start_step}</td></tr>
          ` : ''}
        </table>
        <details class="detail-prompt">
          <summary>${t('gallery.prompt')}</summary>
          <pre>${g.prompt}</pre>
        </details>
        <details class="detail-prompt">
          <summary>${t('gallery.negPrompt')}</summary>
          <pre>${g.negative_prompt}</pre>
        </details>
      </div>
    </div>
    <div class="detail-nav">
      <button id="detail-prev" ${!prev ? 'disabled' : ''}>&larr; ${t('gallery.prev')}</button>
      <button id="detail-next" ${!next ? 'disabled' : ''}>${t('gallery.next')} &rarr;</button>
    </div>
    <div class="detail-support">
      <a href="#/support">${t('gallery.supportLine')}</a>
    </div>
  `;

  view.querySelector('#detail-back').addEventListener('click', e => { e.preventDefault(); navigate('#/'); });
  if (prev) view.querySelector('#detail-prev').addEventListener('click', () => navigate('#/gallery/' + prev.id));
  if (next) view.querySelector('#detail-next').addEventListener('click', () => navigate('#/gallery/' + next.id));
  window.scrollTo(0, 0);
}

/* Keyboard navigation for detail view */
document.addEventListener('keydown', e => {
  const view = document.getElementById('view-detail');
  if (!view.classList.contains('active')) return;
  if (e.key === 'ArrowLeft') { const btn = view.querySelector('#detail-prev'); if (btn && !btn.disabled) btn.click(); }
  if (e.key === 'ArrowRight') { const btn = view.querySelector('#detail-next'); if (btn && !btn.disabled) btn.click(); }
  if (e.key === 'Escape') navigate('#/');
});
