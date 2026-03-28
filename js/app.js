/* app.js — SPA router, age gate, hero, init */
import * as i18n from './i18n.js';

const HF = 'https://huggingface.co/epigene/4cgt/resolve/main/showcase/';

/* Router */
export function navigate(hash) {
  window.location.hash = hash;
}

function getRoute() {
  const h = window.location.hash.slice(1) || '/';
  return h.startsWith('/') ? h : '/' + h;
}

let _onRoute = null;
export function onRoute(cb) { _onRoute = cb; }

function handleRoute() {
  const route = getRoute();
  // Hide all views
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  // Update nav active state
  document.querySelectorAll('.nav-links a[data-route]').forEach(a => {
    a.classList.toggle('active', a.dataset.route === route || route.startsWith(a.dataset.route + '/'));
  });
  if (_onRoute) _onRoute(route);
}

/* Age gate */
function initAgeGate() {
  const gate = document.getElementById('age-gate');
  if (localStorage.getItem('4cgt_age_verified') === 'true') {
    gate.classList.remove('active');
    return;
  }
  document.getElementById('age-confirm').addEventListener('click', () => {
    localStorage.setItem('4cgt_age_verified', 'true');
    gate.classList.remove('active');
  });
  document.getElementById('age-deny').addEventListener('click', () => {
    window.location.href = 'https://www.google.com';
  });
}

/* Hero slideshow */
function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

async function initHero() {
  const bg = document.getElementById('hero-bg');
  let images;
  try {
    const res = await fetch('data/gallery.json');
    const gallery = await res.json();
    images = shuffle(gallery.map(e => e.image));
  } catch {
    images = ['frieren_beach.png'];
  }
  images.forEach((img, i) => {
    const div = document.createElement('div');
    div.className = 'hero-slide' + (i === 0 ? ' active' : '');
    div.style.backgroundImage = `url('${HF}${img}')`;
    bg.appendChild(div);
  });
  const slides = bg.querySelectorAll('.hero-slide');
  let idx = 0;
  if (slides.length > 1) {
    setInterval(() => {
      slides[idx].classList.remove('active');
      idx = (idx + 1) % slides.length;
      slides[idx].classList.add('active');
    }, 6000);
  }

  const hint = document.querySelector('.scroll-hint');
  const nav = document.querySelector('.nav');
  window.addEventListener('scroll', () => {
    hint.classList.toggle('fade', window.scrollY > 80);
    nav.classList.toggle('scrolled', window.scrollY > 100);
  }, { passive: true });
}

/* Nav burger */
function initNav() {
  const burger = document.querySelector('.nav-burger');
  const links = document.querySelector('.nav-links');
  burger.addEventListener('click', () => links.classList.toggle('open'));
  links.querySelectorAll('a').forEach(a => {
    a.addEventListener('click', () => links.classList.remove('open'));
  });

  // Language switcher
  const langBtn = document.querySelector('.nav-lang');
  const langs = ['en','ja','zh','fr'];
  const labels = { en:'EN', ja:'\u65E5\u672C\u8A9E', zh:'\u4E2D\u6587', fr:'FR' };
  langBtn.textContent = labels[i18n.currentLang()] || 'EN';
  langBtn.addEventListener('click', () => {
    const cur = langs.indexOf(i18n.currentLang());
    const next = langs[(cur + 1) % langs.length];
    i18n.setLang(next);
    langBtn.textContent = labels[next];
  });
}

/* Init */
export async function init() {
  await i18n.init();
  initAgeGate();
  await initHero();
  initNav();
  window.addEventListener('hashchange', handleRoute);
  handleRoute();
}
