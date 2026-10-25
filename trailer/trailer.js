/* IndicMT trailer: timeline + scene renderers (motion-quality pass).
   Vanilla JS, no build step, no libraries. Every on-screen number comes from
   window.TRAILER_DATA (trailer/html/data.js, generated from trailer/script/data.json).
   Every visual state is a pure function of t (seconds), so scrubbing and the
   prefers-reduced-motion end-state mode both work. Motion parameters follow
   trailer/research/animation-survey.md. */
(function () {
  'use strict';

  var DATA = window.TRAILER_DATA;
  // Fixed length: the video is final first, the music is generated to fit it afterwards
  // (author's decision). A shorter music.mp3 simply ends early; the timeline keeps running.
  var TOTAL = 70;
  var NS = 'http://www.w3.org/2000/svg';

  var REDUCED = false;
  try { REDUCED = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches); } catch (e) { REDUCED = false; }

  // ================================================================= easing library

  function clamp(v, a, b) { return v < a ? a : (v > b ? b : v); }
  function clamp01(x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }
  function lerp(a, b, p) { return a + (b - a) * p; }

  var Ease = {
    // 1 - 2^(-10x): fast start, long soft tail. Entrances, text reveals, count-ups.
    outExpo: function (x) { x = clamp01(x); return x === 1 ? 1 : 1 - Math.pow(2, -10 * x); },
    outQuart: function (x) { x = clamp01(x); return 1 - Math.pow(1 - x, 4); },
    // exits: leave fast
    inCubic: function (x) { x = clamp01(x); return x * x * x; },
    // camera moves, wipes, background interpolation, smooth steps
    inOutCubic: function (x) { x = clamp01(x); return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; },
    // 1 + (s+1)(x-1)^3 + s(x-1)^2: small pops (~10% overshoot at s = 1.70158)
    outBack: function (x, s) { x = clamp01(x); s = s === undefined ? 1.70158 : s; var y = x - 1; return 1 + (s + 1) * y * y * y + s * y * y; }
  };

  // Closed-form step response of a damped spring released at dt = 0 (0 before, -> 1 after).
  // Underdamped: 1 - e^(-zeta*w*dt) (cos(wd*dt) + (zeta*w/wd) sin(wd*dt)), wd = w*sqrt(1-zeta^2).
  // Critically damped (zeta >= 1): 1 - e^(-w*dt)(1 + w*dt). Deterministic in dt, so scrubbable.
  function spring(dt, w, zeta) {
    if (dt <= 0) return 0;
    if (zeta >= 1) return 1 - Math.exp(-w * dt) * (1 + w * dt);
    var wd = w * Math.sqrt(1 - zeta * zeta);
    return 1 - Math.exp(-zeta * w * dt) * (Math.cos(wd * dt) + (zeta * w / wd) * Math.sin(wd * dt));
  }
  // Damped sine impulse: e^(-zeta*w*dt) sin(wd*dt). A pop/pulse/kick that returns to rest.
  function impulse(dt, w, zeta) {
    if (dt <= 0 || REDUCED) return 0;
    var wd = w * Math.sqrt(1 - zeta * zeta);
    return Math.exp(-zeta * w * dt) * Math.sin(wd * dt);
  }
  // Spring presets (see survey sections 2 and 7): near-critical, Apple-calm, still visibly springy.
  var SPR = {
    bar: { w: 15, z: 0.72 },   // ~4% overshoot, peak ~0.3 s
    hero: { w: 12, z: 0.6 },   // ~9.5% overshoot: only the one hero bar per chart
    slam: { w: 20, z: 0.55 },  // title scale-down
    pair: { w: 14, z: 0.72 }   // scene 5 circle + bar, shared curve
  };
  // Monotone count-up: never overshoots, lands exactly on the data value.
  function countUp(finalValue, lt, at, dur) { return finalValue * Ease.outExpo((lt - at) / dur); }

  // ================================================================= formatting helpers

  // Absolute parameter count -> "154M"; only ever fed DATA values.
  function fmtParamsAbs(n) { return n < 1e9 ? Math.round(n / 1e6) + 'M' : (Math.round(n / 1e7) / 100) + 'B'; }
  function decimalsOf(v) { var s = String(v).split('.'); return s.length > 1 ? s[1].length : 0; }
  // A count-up value in billions, shown with the final value's own precision ("12.19B", "4B", "154M").
  function fmtSizeB(shown, finalB) {
    if (finalB < 1) return Math.round(shown * 1000) + 'M';
    return shown.toFixed(decimalsOf(finalB)) + 'B';
  }

  // ================================================================= DOM helpers

  function $(id) { return document.getElementById(id); }
  function setAttrs(el, attrs) { for (var k in attrs) { if (attrs.hasOwnProperty(k)) el.setAttribute(k, attrs[k]); } }
  function svgEl(tag, attrs, parent) {
    var el = document.createElementNS(NS, tag);
    if (attrs) setAttrs(el, attrs);
    if (parent) parent.appendChild(el);
    return el;
  }
  function divEl(cls, parent, text) {
    var el = document.createElement('div');
    el.className = cls;
    if (text !== undefined) el.textContent = text;
    if (parent) parent.appendChild(el);
    return el;
  }
  // A vertical magenta -> violet gradient for the hero bar / int8 marks (ids are per-SVG to stay unique).
  function addOursGradient(svg, id) {
    var defs = svgEl('defs', null, svg);
    var g = svgEl('linearGradient', { id: id, x1: 0, y1: 0, x2: 0, y2: 1 }, defs);
    svgEl('stop', { offset: '0%', 'stop-color': '#EC4899' }, g);
    svgEl('stop', { offset: '100%', 'stop-color': '#7C3AED' }, g);
    return 'url(#' + id + ')';
  }

  // Mask-reveal builder. tokens: strings (split on spaces into words) or {t, cls} (kept whole).
  // mode 'letters' splits every token into letters. Returns the inner spans that slide.
  function buildMasked(el, tokens, mode) {
    el.textContent = '';
    var inners = [], first = true;
    tokens.forEach(function (tok) {
      var isStr = typeof tok === 'string';
      var text = isStr ? tok : tok.t;
      var cls = isStr ? '' : (tok.cls || '');
      var pieces = mode === 'letters' ? text.split('') : (isStr ? text.split(' ') : [text]);
      pieces.forEach(function (p) {
        if (!p) return;
        if (!first && mode !== 'letters') el.appendChild(document.createTextNode(' '));
        first = false;
        var m = document.createElement('span'); m.className = 'mask';
        var s = document.createElement('span'); s.className = 'mask-in' + (cls ? ' ' + cls : '');
        s.textContent = p;
        m.appendChild(s); el.appendChild(m); inners.push(s);
      });
    });
    return inners;
  }
  // Staggered slide-up of masked spans: easeOutExpo from 110% below the mask edge to 0.
  function revealSpans(spans, lt, at, stagger, dur) {
    for (var i = 0; i < spans.length; i++) {
      var p = Ease.outExpo((lt - at - i * stagger) / dur);
      spans[i].style.transform = 'translate3d(0,' + ((1 - p) * 110).toFixed(2) + '%,0)';
    }
  }
  // A caption = a masked line. In: word reveal (80 ms stagger, 700 ms). Out: 220 ms lift + fade.
  function makeCaption(parent, tokens, cls) {
    var el = divEl(cls || 'caption', parent);
    return { el: el, spans: buildMasked(el, tokens, 'words') };
  }
  function renderCaption(c, lt, inAt, outAt) {
    var o = outAt === undefined ? 0 : Ease.inCubic((lt - outAt) / 0.22);
    if (lt < inAt || o >= 1) { c.el.style.visibility = 'hidden'; return; }
    c.el.style.visibility = 'visible';
    revealSpans(c.spans, lt, inAt, 0.08, 0.7);
    c.el.style.opacity = 1 - o;
    c.el.style.transform = 'translate3d(0,' + (-24 * o).toFixed(2) + 'px,0)';
  }
  // Simple fade + rise for secondary text blocks.
  function fadeRise(el, lt, at, base, dist) {
    var p = Ease.outExpo((lt - at) / 0.7);
    el.style.opacity = Ease.outQuart((lt - at) / 0.4);
    el.style.transform = (base || '') + ' translate3d(0,' + ((1 - p) * (dist === undefined ? 24 : dist)).toFixed(2) + 'px,0)';
  }

  // ================================================================= scene timing helpers

  // Local time within a scene. Under prefers-reduced-motion, a scene renders its end state
  // (its full duration) as soon as it starts.
  function localTime(t, start, end) {
    if (REDUCED) return t >= start ? end - start : -1;
    return t - start;
  }

  // Scene envelope (Apple-style depth): in = fade + 20 px rise + scale 1.03 -> 1 over 550 ms
  // straddling the bar line; out = 320 ms fade + 16 px lift + scale -> 0.98. cutIn/cutOut give a
  // hard switch (used for the intro -> directions match-cut). push = extra scale for slow push-ins.
  function sceneEnvelope(sec, t, start, end, opts) {
    opts = opts || {};
    var last = end >= TOTAL, visible = true, op = 1, dy = 0, sc = 1;
    if (REDUCED) {
      visible = t >= start && (t < end || last);
    } else {
      if (opts.cutIn) { if (t < start) visible = false; }
      else {
        var pi = (t - (start - 0.05)) / 0.55;
        if (pi <= 0) visible = false;
        else { var ei = Ease.outExpo(pi); op *= Ease.outQuart(pi); dy += 20 * (1 - ei); sc *= 1 + 0.03 * (1 - ei); }
      }
      if (!last) {
        if (opts.cutOut) { if (t >= end) visible = false; }
        else {
          var po = (t - (end - 0.3)) / 0.32;
          if (po >= 1) visible = false;
          else if (po > 0) { var eo = Ease.inCubic(po); op *= 1 - eo; dy -= 16 * eo; sc *= 1 - 0.02 * eo; }
        }
      }
    }
    if (!visible) {
      if (sec._vis !== false) { sec.style.visibility = 'hidden'; sec.style.opacity = '0'; sec._vis = false; }
      return false;
    }
    if (opts.push) sc *= opts.push;
    sec.style.visibility = 'visible';
    sec.style.opacity = op;
    sec.style.transform = 'translate3d(0,' + dy.toFixed(2) + 'px,0) scale(' + sc.toFixed(5) + ')';
    sec._vis = true;
    return true;
  }
  // Slow push-in during a hold: 1 -> 1 + amount, easeInOutCubic over dur.
  function pushIn(lt, at, dur, amount) { return 1 + amount * Ease.inOutCubic((lt - at) / dur); }

  // ================================================================= background (sunshine gradient)
  // Base gradient layer rotated per scene + four soft radial light blobs (transform/opacity only).
  // Each scene has a keyframe; across each scene boundary the state interpolates with
  // easeInOutCubic over 1.2 s centred on the cut; a slow sine drift keeps it alive.

  var BG_BOUNDS = [4, 8, 16, 24, 40, 44, 58]; // 44-58 is one "Built for deployment" keyframe
  // blob order: yellow, orange, amber, light pool. Each [x, y, scale, opacity] in stage px.
  var BG_KEYS = [
    { ang: 0,   b: [[960, 520, 1.35, 0.95], [1650, 950, 1.0, 0.6],  [260, 160, 0.9, 0.55], [960, 500, 0.9, 0.8]] },  // 0 intro: burst at centre
    { ang: 25,  b: [[1500, 300, 1.15, 0.85], [300, 900, 1.1, 0.55], [1700, 1000, 0.8, 0.45], [960, 560, 1.0, 0.8]] }, // 1 directions
    { ang: 45,  b: [[320, 260, 1.0, 0.65], [1750, 200, 0.9, 0.4],   [1500, 1050, 1.0, 0.4], [960, 620, 1.3, 0.9]] },   // 2 size
    { ang: 65,  b: [[1700, 700, 1.1, 0.65], [150, 150, 0.8, 0.4],   [400, 1050, 0.9, 0.45], [900, 560, 1.3, 0.9]] },   // 3 fertility
    { ang: 90,  b: [[960, 1180, 1.6, 0.75], [200, 300, 0.9, 0.45],  [1800, 250, 0.9, 0.45], [960, 520, 1.4, 0.85]] },  // 4 quality: sun rising
    { ang: 140, b: [[1400, 500, 1.1, 0.65], [200, 900, 0.9, 0.5],   [300, 100, 0.8, 0.4],   [900, 560, 1.2, 0.85]] },  // 5b judge
    { ang: 120, b: [[520, 600, 1.2, 0.65], [1800, 900, 0.9, 0.45],  [1500, 150, 0.8, 0.4],  [1000, 560, 1.3, 0.9]] },  // 5 quantization
    { ang: 180, b: [[960, 420, 1.7, 0.9], [300, 1000, 1.1, 0.55],   [1650, 1000, 1.1, 0.55], [960, 480, 1.0, 0.8]] }   // 6 end card
  ];
  var BG_BLOB_R = 550; // blobs are 1100 px squares; position by centre
  var bgBase = $('bg-base');
  var bgBlobs = [$('bg-blob-0'), $('bg-blob-1'), $('bg-blob-2'), $('bg-blob-3')];
  var bgFlash = $('bg-flash');
  var cameraEl = $('camera');
  // Impact cues: intro title slam, directions drop, quality lift, end card.
  var FLASHES = [{ at: 0.5, a: 0.6 }, { at: 4.0, a: 0.15 }, { at: 28.0, a: 0.4 }, { at: 58.0, a: 0.2 }, { at: 64.0, a: 0.22 }];
  var KICKS = [{ at: 0.5, a: 7 }, { at: 28.0, a: 4 }];

  function renderBackground(t) {
    var ang = BG_KEYS[0].ang;
    var st = BG_KEYS[0].b.map(function (v) { return v.slice(); });
    for (var k = 0; k < BG_BOUNDS.length; k++) {
      var p = REDUCED ? (t >= BG_BOUNDS[k] ? 1 : 0) : Ease.inOutCubic((t - (BG_BOUNDS[k] - 0.6)) / 1.2);
      if (p <= 0) break;
      var nk = BG_KEYS[k + 1];
      ang = lerp(ang, nk.ang, p);
      for (var i = 0; i < 4; i++) for (var j = 0; j < 4; j++) st[i][j] = lerp(st[i][j], nk.b[i][j], p);
    }
    var drift = REDUCED ? 0 : 1;
    bgBase.style.transform = 'rotate(' + (ang + drift * 4 * Math.sin(t * 2 * Math.PI / 29)).toFixed(3) + 'deg)';
    for (var b = 0; b < 4; b++) {
      var s = st[b];
      var x = s[0] + drift * 45 * Math.sin(t * 2 * Math.PI / 17 + b * 1.7);
      var y = s[1] + drift * 32 * Math.cos(t * 2 * Math.PI / 23 + b * 2.3);
      var sc = s[2] * (1 + drift * 0.04 * Math.sin(t * 2 * Math.PI / 13 + b));
      bgBlobs[b].style.transform = 'translate3d(' + (x - BG_BLOB_R).toFixed(1) + 'px,' + (y - BG_BLOB_R).toFixed(1) + 'px,0) scale(' + sc.toFixed(4) + ')';
      bgBlobs[b].style.opacity = s[3].toFixed(3);
    }
    var fl = 0, kick = 0;
    if (!REDUCED) {
      FLASHES.forEach(function (f) {
        var dt = t - f.at;
        if (dt >= -0.04 && dt < 1.5) fl += f.a * (dt < 0 ? (dt + 0.04) / 0.04 : Math.exp(-dt / 0.3));
      });
      KICKS.forEach(function (k2) { kick += k2.a * impulse(t - k2.at, 30, 0.35); });
    }
    bgFlash.style.opacity = clamp01(fl).toFixed(3);
    cameraEl.style.transform = 'translate3d(0,' + kick.toFixed(2) + 'px,0)';
  }

  // ================================================================= vertical bar-row engine
  // Used by scenes 2 (dynamic y-scale) and 4 (fixed y-scale). Fluid by construction:
  //  - each bar grows on a closed-form spring from its own start time (stagger 100-120 ms);
  //  - the y-max is a sum of smooth steps, max(t) = m0 + sum_k dm_k * S((t - t_k + lead)/D),
  //    starting 150 ms before each bar so the rescale anticipates it;
  //  - the row's slot count is n(t) = sum_k S(...), so re-centring pans continuously.
  // Values count up monotonically (never overshoot) while the geometry springs.

  function makeVRow(svg, cfg, items) {
    var root = svgEl('g', null, svg);
    svgEl('line', { class: 'bar-baseline', x1: cfg.chartX, y1: cfg.baselineY, x2: cfg.chartX + cfg.chartWidth, y2: cfg.baselineY }, root);
    var g = svgEl('g', null, root);
    items.forEach(function (it, i) {
      var sx = i * (cfg.bw + cfg.gap);
      it.slotX = sx;
      it.rectEl = svgEl('rect', { x: sx, width: cfg.bw, rx: 6, fill: it.fill, opacity: 0 }, g);
      var lx = sx + cfg.bw / 2 + 6, ly = cfg.baselineY + 24;
      it.labelRot = 'rotate(-35 ' + lx + ' ' + ly + ')';
      it.labelEl = svgEl('text', { class: 'bar-label' + (it.ours ? ' ours' : ''), x: lx, y: ly, 'text-anchor': 'end', transform: it.labelRot, opacity: 0 }, g);
      it.labelEl.textContent = it.label;
      it.valueEl = svgEl('text', { class: 'bar-value' + (it.ours ? ' ours' : ''), 'text-anchor': 'middle', opacity: 0 }, g);
    });
    var row = { root: root, g: g, cfg: cfg, items: items, tx: 0 };
    if (!cfg.fixedMax) {
      var m = Math.max(items[0].value, items.length > 1 ? items[1].value : items[0].value);
      row.m0 = m;
      row.steps = [];
      for (var k = 1; k < items.length; k++) {
        var nm = Math.max(m, items[k].value);
        if (nm > m) row.steps.push({ at: items[k].at, d: nm - m });
        m = nm;
      }
    }
    return row;
  }

  function renderVRow(row, t) {
    var cfg = row.cfg, items = row.items, i;
    var n = 0;
    for (i = 0; i < items.length; i++) n += Ease.inOutCubic((t - (items[i].at - 0.1)) / 0.5);
    var rowW = n * (cfg.bw + cfg.gap) - cfg.gap * Math.min(n, 1);
    var tx = cfg.chartX + (cfg.chartWidth - rowW) / 2;
    row.tx = tx;
    row.g.setAttribute('transform', 'translate(' + tx.toFixed(2) + ',0)');
    var max = cfg.fixedMax;
    if (!max) {
      max = row.m0;
      for (i = 0; i < row.steps.length; i++) max += row.steps[i].d * Ease.inOutCubic((t - (row.steps[i].at - 0.15)) / 0.45);
    }
    // Optional morph to a second data set (scene 4: en->xx -> xx->en). pm drives geometry
    // (heights, slot order, y-range), pc drives the monotone value count between the two exact values.
    var pm = cfg.morph ? cfg.morph(t) : 0, pc = cfg.morphCount ? cfg.morphCount(t) : 0;
    if (cfg.fixedMaxB) max = lerp(max, cfg.fixedMaxB, pm);
    var ppu = cfg.chartHeight / max;
    row.ppu = ppu;
    for (i = 0; i < items.length; i++) {
      var it = items[i], dt = t - it.at;
      if (dt <= 0) {
        it.rectEl.setAttribute('opacity', 0); it.valueEl.setAttribute('opacity', 0); it.labelEl.setAttribute('opacity', 0);
        it.h = 0; continue;
      }
      var sp = it.spring || SPR.bar;
      var gp = spring(dt, sp.w, sp.z);
      var hasB = it.valueB !== undefined;
      var v = hasB ? lerp(it.value, it.valueB, pm) : it.value;
      var sx = it.slotXB !== undefined ? lerp(it.slotX, it.slotXB, pm) : it.slotX;
      var h = Math.max(v * ppu * gp, 3);
      var y = cfg.baselineY - h;
      var dim = cfg.dim ? cfg.dim(it, t) : 1;
      setAttrs(it.rectEl, { x: sx.toFixed(2), y: y.toFixed(2), height: h.toFixed(2), opacity: (it.baseOp * dim * Ease.outQuart(dt / 0.15)).toFixed(3) });
      var shown = (hasB && pc > 0) ? lerp(it.value, it.valueB, pc) : countUp(it.value, dt, 0, cfg.countDur || 0.8);
      it.valueEl.textContent = it.fmt(shown, it);
      if (sx !== it.slotX) it.labelEl.setAttribute('transform', 'translate(' + (sx - it.slotX).toFixed(2) + ',0) ' + it.labelRot);
      else if (it._lastSx !== it.slotX) it.labelEl.setAttribute('transform', it.labelRot);
      it._lastSx = sx;
      it.vx = sx + cfg.bw / 2; it.vy = y - 12;
      it.valueEl.setAttribute('transform', 'translate(' + it.vx.toFixed(2) + ',' + it.vy.toFixed(2) + ')');
      var txtDim = dim < 1 ? lerp(0.45, 1, (dim - 0.45) / 0.55) : 1;
      it.valueEl.setAttribute('opacity', (Ease.outQuart((dt - 0.05) / 0.25) * txtDim).toFixed(3));
      it.labelEl.setAttribute('opacity', (Ease.outQuart(dt / 0.35) * txtDim).toFixed(3));
      it.h = h; it.top = y; it.cx = tx + sx + cfg.bw / 2;
    }
  }

  // ================================================================= timeline / HUD

  var stageEl = $('stage');
  var audio = $('music');
  var audioOk = false, audioChecked = false;
  audio.addEventListener('error', function () { audioOk = false; audioChecked = true; updateAudioStatus(); });
  audio.addEventListener('canplaythrough', function () { audioOk = true; audioChecked = true; updateAudioStatus(); });
  audio.addEventListener('loadedmetadata', function () { audioOk = true; audioChecked = true; updateAudioStatus(); applyAudioDuration(); });
  setTimeout(function () { if (!audioChecked) { audioChecked = true; updateAudioStatus(); } }, 1500);

  var hudPlay = $('hud-play'), hudRestart = $('hud-restart'), hudScrub = $('hud-scrubber'),
      hudTime = $('hud-time'), hudAudioStatus = $('hud-audio-status');
  hudScrub.max = String(TOTAL);
  // Adopt the audio's real length so the trailer ends exactly when the music does.
  function applyAudioDuration() {
    var d = audio.duration;
    // Length no longer follows the audio (see TOTAL); kept as a no-op hook for the next track.
    void d;
  }

  var state = { t: 0, playing: false, lastPerf: null, scrubbing: false };

  function updateAudioStatus() { hudAudioStatus.textContent = audioOk ? '(music.mp3)' : '(silent — no music.mp3)'; }
  updateAudioStatus();
  // metadata may already be loaded by the time this script runs
  if (audio.readyState >= 1) { audioOk = true; audioChecked = true; applyAudioDuration(); }

  function setPlaying(p) {
    state.playing = p;
    hudPlay.textContent = p ? 'Pause' : 'Play';
    if (p) {
      if (state.t >= TOTAL) state.t = 0;
      if (audioOk) { try { audio.currentTime = state.t; audio.play()['catch'](function () {}); } catch (e) {} }
      state.lastPerf = performance.now();
    } else if (audioOk) { try { audio.pause(); } catch (e) {} }
  }
  function restart() {
    state.t = 0;
    if (audioOk) { try { audio.currentTime = 0; } catch (e) {} }
    state.lastPerf = performance.now();
    render(state.t); updateHud();
  }
  function updateHud() {
    if (!state.scrubbing) hudScrub.value = state.t;
    hudTime.textContent = state.t.toFixed(2) + ' / ' + TOTAL.toFixed(2);
  }

  hudPlay.addEventListener('click', function () { setPlaying(!state.playing); });
  hudRestart.addEventListener('click', restart);
  hudScrub.addEventListener('input', function () {
    state.scrubbing = true;
    state.t = clamp(parseFloat(hudScrub.value), 0, TOTAL);
    if (audioOk) { try { audio.currentTime = state.t; } catch (e) {} }
    render(state.t); updateHud();
  });
  hudScrub.addEventListener('change', function () { state.scrubbing = false; });
  window.addEventListener('keydown', function (ev) {
    if (ev.target && ev.target.tagName === 'INPUT') return;
    if (ev.code === 'Space') { ev.preventDefault(); setPlaying(!state.playing); }
    else if (ev.key === 'r' || ev.key === 'R') restart();
    else if (ev.key === 'h' || ev.key === 'H') hudTime.classList.toggle('hidden');
  });

  function fitStage() {
    var s = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
    stageEl.style.transform = 'scale(' + s + ')';
  }
  window.addEventListener('resize', fitStage);
  fitStage();

  function tick(now) {
    if (state.playing) {
      if (audioOk && !audio.paused && !state.scrubbing) state.t = audio.currentTime;
      else state.t += (now - (state.lastPerf || now)) / 1000;
      state.lastPerf = now;
      if (state.t >= TOTAL) { state.t = TOTAL; setPlaying(false); }
    }
    render(state.t); updateHud();
    requestAnimationFrame(tick);
  }

  var SCENES = [];
  function render(t) {
    renderBackground(t);
    for (var i = 0; i < SCENES.length; i++) SCENES[i](t);
  }

  // Shared derived values (all from DATA).
  // "154M": 153,807,872 rounded to the nearest million, the same figure the charts show
  // (fmtParamsAbs), so one parameter count appears everywhere.
  var HEADLINE_M = Math.round(DATA.ours.params / 1e6);
  var LANG = DATA.languages; // { indic: 22, total_incl_english: 23 } from data/common/languages.py

  // ================================================================= scene 0: intro (0-4 s)

  (function sceneIntro() {
    var start = 0, end = 4;
    var sec = $('scene-0');
    var kicker = $('s0-kicker'), title = $('s0-title'), params = $('s0-params'), langs = $('s0-langs');
    var kSpans = buildMasked(kicker, ['Introducing'], 'words');
    var tSpans = buildMasked(title, [{ t: 'Indic' }, { t: 'MT', cls: 'accent' }], 'letters');
    var pSpans = buildMasked(params, [{ t: HEADLINE_M + 'M', cls: 'hl' }, 'parameters'], 'words');
    var lSpans = buildMasked(langs, [{ t: String(LANG.total_incl_english), cls: 'hl' }, 'languages', '·',
      { t: String(LANG.indic), cls: 'hl' }, 'Indic', '+', 'English'], 'words');
    // Match-cut target: the header position in scene 1 (.brand top 46% -> .brand-header top 9%).
    var HEADER_DY = (0.09 - 0.46) * 1080, HEADER_SCALE = 0.32;

    function exitOf(el, lt, at, base) {
      var o = Ease.inCubic((lt - at) / 0.3);
      el.style.opacity = 1 - o;
      el.style.transform = base + ' translate3d(0,' + (-30 * o).toFixed(2) + 'px,0)';
    }

    SCENES.push(function (t) {
      var lt = localTime(t, start, end);
      if (!sceneEnvelope(sec, t, start, end, { cutIn: true, cutOut: true })) return;

      // "Introducing": slide-up + short blur-in on the opening hit.
      revealSpans(kSpans, lt, 0, 0, 0.6);
      var kb = REDUCED ? 0 : 8 * (1 - Ease.outQuart(lt / 0.4));
      kicker.style.filter = kb > 0.05 ? 'blur(' + kb.toFixed(2) + 'px)' : 'none';
      exitOf(kicker, lt, 3.3, 'translate(-50%,-50%)');

      // "IndicMT": per-letter reveal (30 ms stagger) + a spring scale slam landing on the 0.5 s beat.
      revealSpans(tSpans, lt, 0.38, 0.03, 0.55);
      var slam = 1.18 - 0.18 * spring(lt - 0.38, SPR.slam.w, SPR.slam.z);
      var mc = Ease.inOutCubic((lt - 3.45) / 0.55);
      title.style.transform = 'translate(-50%,-50%) translate3d(0,' + (HEADER_DY * mc).toFixed(2) + 'px,0) scale(' + (slam * lerp(1, HEADER_SCALE, mc)).toFixed(4) + ')';

      // "154M parameters" at 1.5 s: count-up with the reveal.
      revealSpans(pSpans, lt, 1.5, 0.08, 0.7);
      pSpans[0].textContent = Math.round(countUp(HEADLINE_M, lt, 1.5, 0.8)) + 'M';
      exitOf(params, lt, 3.35, 'translate(-50%,0)');

      // "23 languages · 22 Indic + English" at 2.5 s.
      revealSpans(lSpans, lt, 2.5, 0.06, 0.7);
      lSpans[0].textContent = String(Math.round(countUp(LANG.total_incl_english, lt, 2.5, 0.6)));
      lSpans[3].textContent = String(Math.round(countUp(LANG.indic, lt, 2.6, 0.6)));
      exitOf(langs, lt, 3.4, 'translate(-50%,0)');
    });
  })();

  // ================================================================= scene 1: directions (4-8 s)

  (function sceneDirections() {
    var start = 4, end = 8;
    var sec = $('scene-1');
    var header = $('s1-title');
    buildMasked(header, [{ t: 'Indic' }, { t: 'MT', cls: 'accent' }], 'letters').forEach(function (s) { s.style.transform = 'none'; });
    var ticker = $('s1-ticker');
    // A fixed subset of the blog hero's own translation pairs (blog/assets/hero-pairs.js).
    var PAIRS = [
      { en: 'hello', lang: 'hi', name: 'Hindi', text: 'नमस्ते' },
      { en: 'thank you', lang: 'bn', name: 'Bengali', text: 'ধন্যবাদ' },
      { en: 'water', lang: 'ta', name: 'Tamil', text: 'தண்ணீர்' },
      { en: 'book', lang: 'te', name: 'Telugu', text: 'పుస్తకం' },
      { en: 'friend', lang: 'kn', name: 'Kannada', text: 'ಸ್ನೇಹಿತ' },
      { en: 'language', lang: 'ml', name: 'Malayalam', text: 'ഭാഷ' },
      { en: 'welcome', lang: 'gu', name: 'Gujarati', text: 'સ્વાગત છે' },
      { en: 'How are you?', lang: 'ur', name: 'Urdu', text: 'آپ کیسے ہیں؟' }
    ];
    // One pair per beat (0.5 s); even beats English -> Indic, odd beats Indic -> English.
    var items = PAIRS.map(function (p, k) {
      var toIndic = k % 2 === 0;
      var el = divEl('pair', ticker);
      var chip = divEl('chip', el, toIndic ? 'English → ' + p.name : p.name + ' → English');
      var line = divEl('line', el);
      function part(text, cls, lang) {
        var m = document.createElement('span'); m.className = 'mask';
        var s = document.createElement('span'); s.className = 'mask-in ' + cls; s.textContent = text;
        if (lang) { s.lang = lang; if (lang === 'ur') { s.dir = 'rtl'; s.style.unicodeBidi = 'isolate'; } }
        m.appendChild(s); line.appendChild(m); return s;
      }
      var src = toIndic ? part(p.en, 'src') : part(p.text, 'src', p.lang);
      var arrow = part('→', 'arrow');
      var tgt = toIndic ? part(p.text, 'tgt', p.lang) : part(p.en, 'tgt');
      return { el: el, chip: chip, spans: [src, arrow, tgt], at: 0.5 * k, next: k < PAIRS.length - 1 ? 0.5 * (k + 1) : Infinity };
    });
    var cap = makeCaption(sec, ['English', '↔', { t: String(LANG.indic), cls: 'hl' }, 'Indic', 'languages.'], 'dir-caption');
    var sub = $('s1-sub');

    SCENES.push(function (t) {
      var lt = localTime(t, start, end);
      if (!sceneEnvelope(sec, t, start, end, { cutIn: true })) return;
      items.forEach(function (it) {
        var o = Ease.inCubic((lt - (it.next - 0.05)) / 0.2);
        if (lt < it.at || o >= 1) { it.el.style.visibility = 'hidden'; return; }
        it.el.style.visibility = 'visible';
        revealSpans(it.spans, lt, it.at + 0.03, 0.06, 0.45);
        var cp = Ease.outExpo((lt - it.at) / 0.4);
        it.chip.style.opacity = cp;
        it.chip.style.transform = 'translate3d(0,' + ((1 - cp) * 14).toFixed(2) + 'px,0)';
        it.el.style.opacity = 1 - o;
        it.el.style.transform = 'translate(-50%,0) translate3d(0,' + (-60 * o).toFixed(2) + 'px,0)';
      });
      cap.el.style.transform = 'translate(-50%,0)';
      cap.el.style.visibility = lt >= 0.6 ? 'visible' : 'hidden';
      revealSpans(cap.spans, lt, 0.6, 0.08, 0.7);
      fadeRise(sub, lt, 1.1, 'translate(-50%,0)');
    });
  })();

  // ================================================================= scene 2: size (8-16 s)

  (function sceneSize() {
    var start = 8, end = 16;
    var sec = $('scene-2'), svg = $('s2-svg'), foot = $('s2-footnote');
    var oursFill = addOursGradient(svg, 'g-ours-2');
    // Chart top at y 380 so the centred heading and the centred spec block (ends ~y 305) clear it.
    var CHART = { chartX: 96, chartWidth: 1728, chartHeight: 440, baselineY: 820, bw: 90, gap: 30, countDur: 0.8 };
    // DATA.sizes_b is ours first, then ascending. Ours at 8.25 s; the other 11 from 9.0 s,
    // 110 ms apart (last starts 10.1 s, settled by ~10.7 s). Chart then holds.
    var items = DATA.sizes_b.map(function (s, i) {
      var ours = /ours/i.test(s.model);
      return {
        value: s.params_b, ours: ours, at: ours ? start + 0.25 : start + 1.0 + (i - 1) * 0.11,
        fill: ours ? oursFill : 'var(--chart-other)', baseOp: ours ? 1 : 0.55,
        label: ours ? 'IndicMT' : s.model,
        spring: ours ? SPR.hero : SPR.bar,
        fmt: function (v, it) { return fmtSizeB(v, it.value); }
      };
    });
    CHART.dim = function (it, t) { return it.ours ? 1 : 1 - 0.55 * Ease.inOutCubic((t - (start + 4.3)) / 0.5); };
    var row = makeVRow(svg, CHART, items);
    var ours = items.filter(function (it) { return it.ours; })[0];
    // Callout that rides our bar down as the scale grows: "IndicMT" + a small pointer.
    var callout = svgEl('g', { opacity: 0 }, svg);
    var calloutText = svgEl('text', { class: 'callout-label', 'text-anchor': 'middle', x: 0, y: -40 }, callout);
    calloutText.textContent = 'IndicMT';
    svgEl('path', { d: 'M -7 -32 L 7 -32 L 0 -24 Z', fill: 'var(--chart-ours)' }, callout);

    var capA = makeCaption(sec, ['Tiny, by design.']);
    // Apple-style spec reveal: "Up to" / huge "200×" / qualifier. Ratio = max of
    // DATA.size_ratios_vs_ours (199.6) rounded to the nearest 10, per the fact-check table.
    var ratioMax = Math.max.apply(null, Object.keys(DATA.size_ratios_vs_ours).map(function (k) { return DATA.size_ratios_vs_ours[k]; }));
    var roundedRatio = Math.round(ratioMax / 10) * 10;
    var spec = divEl('spec', sec);
    var specK = buildMasked(divEl('spec-kicker', spec), ['Up to'], 'words');
    var specN = buildMasked(divEl('spec-num', spec), [{ t: roundedRatio + '×', cls: 'num hl' }], 'words');
    var specQ = buildMasked(divEl('spec-qual', spec), ['smaller than the largest model here.'], 'words');

    SCENES.push(function (t) {
      var lt = localTime(t, start, end);
      var push = pushIn(lt, 4, 4, 0.025);
      if (!sceneEnvelope(sec, t, start, end)) return;
      row.root.setAttribute('transform', 'translate(960,' + CHART.baselineY + ') scale(' + push.toFixed(5) + ') translate(-960,' + (-CHART.baselineY) + ')');
      renderVRow(row, t);
      if (ours.h > 0) {
        callout.setAttribute('opacity', Ease.outQuart((t - ours.at - 0.1) / 0.3).toFixed(3));
        var pulse = 1 + 0.12 * impulse(t - (start + 4.3), 16, 0.35);
        callout.setAttribute('transform', 'translate(' + (row.tx + ours.slotX + CHART.bw / 2).toFixed(2) + ',' + (ours.top - 20).toFixed(2) + ') scale(' + pulse.toFixed(4) + ')');
      } else callout.setAttribute('opacity', 0);
      renderCaption(capA, lt, 0, 3.8);
      spec.style.visibility = lt >= 4 ? 'visible' : 'hidden';
      revealSpans(specK, lt, 4.0, 0, 0.7);
      revealSpans(specN, lt, 4.1, 0, 0.8);
      specN[0].textContent = Math.round(countUp(roundedRatio, lt, 4.1, 0.9)) + '×';
      revealSpans(specQ, lt, 4.35, 0.06, 0.7);
      foot.style.opacity = Ease.outQuart((lt - 0.3) / 0.5);
    });
  })();

  // ================================================================= scene 3: fertility (16-24 s)

  (function sceneFertility() {
    var start = 16, end = 24;
    var sec = $('scene-3'), svg = $('s3-svg'), foot = $('s3-footnote');
    var oursFill = addOursGradient(svg, 'g-ours-3');
    var fert = DATA.fertility;
    var oursEntry = fert.filter(function (f) { return /ours/i.test(f.tokenizer); })[0];
    // Merge tokenizers with identical fertility (Sarvam Translate / Indic Translate share the
    // Gemma vocabulary) into one bar, largest first; ours lands last.
    var byValue = {}, order = [];
    fert.forEach(function (f) {
      if (f === oursEntry) return;
      var key = f.macro_fertility.toFixed(4);
      if (!byValue[key]) { byValue[key] = { value: f.macro_fertility, names: [] }; order.push(key); }
      byValue[key].names.push(f.tokenizer.replace(/\s*\(.*\)$/, ''));
    });
    var rows = order.map(function (k) { return byValue[k]; }).sort(function (a, b) { return b.value - a.value; })
      .map(function (g) { return { label: g.names.join(' · '), value: g.value, ours: false }; });
    rows.push({ label: 'IndicMT', value: oursEntry.macro_fertility, ours: true });

    var C = { x0: 560, y0: 350, rowH: 120, barH: 60, maxBarPx: 1050 }; // first row below heading + sub
    var ppu = C.maxBarPx / Math.max.apply(null, rows.map(function (r) { return r.value; }));
    var g = svgEl('g', null, svg);
    var oursRow = rows[rows.length - 1];
    var washY = C.y0 + (rows.length - 1) * C.rowH - 18;
    var wash = svgEl('rect', { x: C.x0 - 14, y: washY, height: C.barH + 36, width: oursRow.value * ppu + 130, rx: 14, fill: 'var(--highlight-wash)', opacity: 0 }, g);
    rows.forEach(function (r, i) {
      r.y = C.y0 + i * C.rowH;
      r.at = r.ours ? 1.5 : 0.6 + i * 0.15; // local: others 150 ms apart, ours one beat later
      r.labelEl = svgEl('text', { class: 'row-label' + (r.ours ? ' ours' : ''), x: C.x0 - 22, y: r.y + C.barH / 2 + 8, 'text-anchor': 'end', opacity: 0 }, g);
      r.labelEl.textContent = r.label;
      r.rectEl = svgEl('rect', { x: C.x0, y: r.y, height: C.barH, width: 0, rx: 8, fill: r.ours ? oursFill : 'var(--chart-other)', opacity: 0 }, g);
      r.valueEl = svgEl('text', { class: 'bar-value' + (r.ours ? ' ours' : ''), y: r.y + C.barH / 2 + 8, opacity: 0 }, g);
    });
    var lowest = svgEl('text', { class: 'highlight-label', opacity: 0 }, g);
    lowest.textContent = 'Lowest';

    var capA = makeCaption(sec, ['Fewer tokens for every word.']);
    var capSub = makeCaption(sec, ['Fertility = subword tokens per word, averaged over ' + LANG.total_incl_english + ' languages. Lower is better.'], 'caption-sub');
    var capB = makeCaption(sec, ['Shorter sequences. Faster, cheaper inference.']);

    SCENES.push(function (t) {
      var lt = localTime(t, start, end);
      if (!sceneEnvelope(sec, t, start, end, { push: pushIn(lt, 4, 4, 0.02) })) return;
      rows.forEach(function (r) {
        var dt = lt - r.at;
        var sp = r.ours ? SPR.hero : SPR.bar;
        var w = r.value * ppu * spring(dt, sp.w, sp.z);
        r.rectEl.setAttribute('width', Math.max(w, 0).toFixed(2));
        r.rectEl.setAttribute('opacity', ((r.ours ? 1 : 0.55) * Ease.outQuart(dt / 0.15)).toFixed(3));
        var lp = Ease.outExpo(dt / 0.6);
        r.labelEl.setAttribute('opacity', Ease.outQuart(dt / 0.3).toFixed(3));
        r.labelEl.setAttribute('transform', 'translate(' + (-16 * (1 - lp)).toFixed(2) + ',0)');
        r.valueEl.setAttribute('x', (C.x0 + Math.max(w, 0) + 16).toFixed(2));
        r.valueEl.setAttribute('opacity', Ease.outQuart((dt - 0.05) / 0.25).toFixed(3));
        r.valueEl.textContent = countUp(r.value, dt, 0, 0.8).toFixed(2);
      });
      // Highlight wash sweeps in behind our bar (scaleX from the left), then "Lowest".
      var wp = Ease.inOutCubic((lt - 2.5) / 0.55);
      wash.setAttribute('opacity', wp > 0 ? 1 : 0);
      wash.setAttribute('transform', 'translate(' + (C.x0 - 14) + ',0) scale(' + wp.toFixed(4) + ',1) translate(' + (-(C.x0 - 14)) + ',0)');
      var lwp = Ease.outExpo((lt - 2.8) / 0.6);
      lowest.setAttribute('opacity', Ease.outQuart((lt - 2.8) / 0.3).toFixed(3));
      lowest.setAttribute('x', C.x0);
      lowest.setAttribute('y', (washY - 12 + 14 * (1 - lwp)).toFixed(2));
      renderCaption(capA, lt, 0, 3.8);
      renderCaption(capSub, lt, 0.3, 3.8);
      renderCaption(capB, lt, 4.0);
      foot.style.opacity = Ease.outQuart((lt - 0.3) / 0.5);
    });
  })();

  // ================================================================= scene 4: quality (24-40 s)
  // 24-36 English -> Indic (the payoff); 36-40 the same bars morph to Indic -> English.

  (function sceneQuality() {
    var start = 24, end = 40, LIFT = 28, SPEC = 30.5, XX = 36;
    var sec = $('scene-4'), svg = $('s4-svg'), foot = $('s4-footnote');
    var oursFill = addOursGradient(svg, 'g-ours-4');
    // Chart top at y 360: the centred two-line heading ends near y 216, the bracket sits at y 340.
    var CHART = { chartX: 96, chartWidth: 1728, chartHeight: 470, baselineY: 830, bw: 78, gap: 22, fixedMax: 55, countDur: 0.8 };

    var qual = DATA.quality_en_xx_in22gen; // first 9 = general models (ascending en->xx), last 3 = specialists
    var general = qual.slice(0, 9), specialists = qual.slice(9, 12);
    // Scene 4 scores the final model served through CTranslate2 float32 (DATA.ours_quality ct2_float32 values:
    // combined + human fine-tune without replay, step 1,800). Scenes 5 and 7 use the same model in CTranslate2.
    var OURS_Q = DATA.ours_quality.en_xx_chrf_ct2_float32, OURS_XX = DATA.ours_quality.xx_en_chrf_ct2_float32;
    // Specialists below our score enter with the general models (before IndicMT lands); the ones above
    // enter after it, and the "within N points" bracket spans only those.
    var specBelow = specialists.filter(function (m) { return m.en_xx_chrf < OURS_Q; });
    var specAbove = specialists.filter(function (m) { return m.en_xx_chrf >= OURS_Q; });
    var items = [];
    function row(m, tier, at, fill, baseOp) {
      return { value: m.en_xx_chrf, valueB: m.xx_en_chrf, model: m.model, at: at, tier: tier, fill: fill, baseOp: baseOp,
        label: m.model + (m.partial_16_of_22 ? '*' : '') + (m.rescored_by_us ? '†' : ''), fmt: function (v) { return v.toFixed(1); } };
    }
    general.forEach(function (m, i) { items.push(row(m, 'general', start + 0.8 + i * 0.1, 'var(--chart-other)', 0.55)); });
    specBelow.forEach(function (m, i) { items.push(row(m, 'specialist', start + 0.8 + (general.length + i) * 0.1, 'var(--chart-other-dark)', 0.85)); });
    var ours = { value: OURS_Q, valueB: OURS_XX, model: 'IndicMT', at: LIFT, tier: 'ours', ours: true, fill: oursFill, baseOp: 1,
      label: 'IndicMT', spring: SPR.hero, fmt: function (v) { return v.toFixed(2); } };
    items.push(ours);
    specAbove.forEach(function (m, i) { items.push(row(m, 'specialist', SPEC + i * 0.12, 'var(--chart-other-dark)', 0.85)); });

    // Indic -> English: re-sort ascending by xx->en, derive who we beat and who is just ahead.
    var COMPETITIVE_GAP = 2.5; // "competitive" = at least one general model beaten AND one within 2.5 chrF++ above
    items.slice().sort(function (a, b) { return a.valueB - b.valueB; }).forEach(function (it, rank) {
      it.slotXB = rank * (CHART.bw + CHART.gap);
    });
    var others = items.filter(function (it) { return !it.ours; });
    others.forEach(function (it) {
      it.xxBeaten = it.valueB < OURS_XX;
      it.xxGap = it.valueB - OURS_XX;
      it.xxNear = it.xxGap > 0 && it.xxGap <= COMPETITIVE_GAP;
    });
    var beatenGeneral = others.filter(function (it) { return it.tier === 'general' && it.xxBeaten; }).length;
    var nearItems = others.filter(function (it) { return it.xxNear; });
    var competitive = beatenGeneral > 0 && nearItems.length > 0;
    var sarvam = others.filter(function (it) { return it.model === 'Sarvam Translate'; })[0];
    var aheadSarvam = !!sarvam && sarvam.valueB < OURS_XX;
    CHART.fixedMaxB = Math.ceil(Math.max.apply(null, items.map(function (it) { return it.valueB; })) / 10) * 10; // 70
    // Author: no Indic -> English chart, only a short mention (capD). The bars stay on en->xx.
    CHART.morph = function () { return 0; };
    CHART.morphCount = function () { return 0; };
    // Dimming: general models dim before IndicMT lands (anticipation); in the xx->en beat the
    // models we beat and the nearest ones come back up, the rest stay dim.
    CHART.dim = function (it, t) {
      var en = it.tier === 'general' ? 1 - 0.5 * Ease.inOutCubic((t - (LIFT - 0.45)) / 0.4) : 1;
      return en;
    };

    // Halo + ring burst behind our bar (drawn under the row), "154M" tag inside it.
    var halo = svgEl('rect', { rx: 12, fill: 'var(--highlight-wash)', opacity: 0 }, svg);
    var ring = svgEl('rect', { rx: 10, fill: 'none', stroke: 'var(--accent-amber)', 'stroke-width': 3, opacity: 0 }, svg);
    var vrow = makeVRow(svg, CHART, items);
    var tag = svgEl('text', { class: 'bar-tag', 'text-anchor': 'middle', opacity: 0 }, svg);
    tag.textContent = fmtParamsAbs(DATA.ours.params);
    // xx->en: dashed line at our height, and the gap above each of the nearest models.
    var ourLine = svgEl('line', { stroke: 'var(--amber-text)', 'stroke-width': 2, 'stroke-dasharray': '8 8', opacity: 0 }, svg);
    nearItems.forEach(function (it) {
      it.gapEl = svgEl('text', { class: 'bracket-label', 'text-anchor': 'middle', opacity: 0 }, svg);
      it.gapEl.textContent = '+' + it.xxGap.toFixed(1);
    });

    // Bracket from our bar across the specialists still above us, labelled with the gap to each as a range,
    // rounded half-up to 0.1 (the epsilon keeps 1.95 from rounding down as 1.9499999).
    var r1 = function (v) { return (Math.round(v * 10 + 1e-6) / 10).toFixed(1); };
    var aboveVals = specAbove.map(function (m) { return m.en_xx_chrf - OURS_Q; });
    var gapPts = aboveVals.length ? r1(Math.min.apply(null, aboveVals)) + '–' + r1(Math.max.apply(null, aboveVals)) : '';
    var bracket = svgEl('path', { stroke: 'var(--amber-text)', 'stroke-width': 2.5, fill: 'none', opacity: 0, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, svg);
    var bracketLabel = svgEl('text', { class: 'bracket-label', 'text-anchor': 'middle', opacity: 0 }, svg);
    bracketLabel.textContent = gapPts + ' chrF++';
    // "up to 52x smaller": Indic Translate, the largest specialist, is 51.6x ours; round up.
    var upToRatio = Math.ceil(DATA.size_ratios_vs_ours['Indic Translate']);

    // Prominent metric label (was a small grey axis label): "chrF++" in the gradient highlight.
    var metricEn = makeCaption(sec, [{ t: 'chrF++', cls: 'hl' }, '·', 'English → Indic', '·', 'IN22-Gen', '·', { t: 'CPU inference', cls: 'hl hl-live hl-box' }], 'metric-label');

    var capA = makeCaption(sec, ['How well does it translate?']);
    // Only claim "beats every general model" if the data says so; otherwise fall back to neutral text.
    var generalMax = Math.max.apply(null, general.map(function (m) { return m.en_xx_chrf; }));
    // "and IndicTrans2" only when DATA.vs_indictrans2 records a significant paired-bootstrap win (same protocol).
    var vsIT2 = DATA.vs_indictrans2 || {};
    var beatsIT2 = specBelow.some(function (m) { return m.model === 'IndicTrans2'; }) && vsIT2.en_xx_significant === true;
    var capB = generalMax < OURS_Q
      ? makeCaption(sec, beatsIT2 ? ['Beats', { t: 'every general model.', cls: 'hl' }, 'Slightly ahead of', { t: 'IndicTrans2,', cls: 'hl' },
                                     { t: Math.floor(DATA.size_ratios_vs_ours['IndicTrans2']) + '×', cls: 'hl' }, 'smaller.']
                                  : ['Beats', { t: 'every general model', cls: 'hl' }, 'we compared.'])
      : makeCaption(sec, ['Compared with general models.']);
    var capC = makeCaption(sec, ['Within', { t: gapPts + ' points', cls: 'hl' }, 'of state-of-the-art Indic translation,', 'up to', { t: upToRatio + '×', cls: 'hl' }, 'smaller.']);
    // "Indic → English: competitive with general models, ahead of Sarvam Translate." Each part only
    // if the data supports it (see competitive / aheadSarvam above).
    // A one-line mention under the en->xx headline (no chart): our xx->en score, and
    // "ahead of Sarvam Translate" / "competitive with general models" only if the data says so.
    var capTail = aheadSarvam ? 'ahead of Sarvam Translate.' : (competitive ? 'competitive with general models.' : '');
    var capDTokens = ['Indic → English:', { t: OURS_XX.toFixed(1), cls: 'hl' }, capTail ? 'chrF++,' : 'chrF++.'];
    if (capTail) capDTokens.push(capTail);
    var capD = makeCaption(sec, capDTokens, 'caption-sub');
    // Sits directly under the "chrF++ · English → Indic" metric label (top-left, y 280).
    capD.el.style.top = '326px'; capD.el.style.left = '5%'; capD.el.style.right = 'auto';
    capD.el.style.textAlign = 'left'; capD.el.style.fontSize = '24px';

    SCENES.push(function (t) {
      var lt = localTime(t, start, end);
      if (!sceneEnvelope(sec, t, start, end)) return;
      var push = pushIn(lt, 8, 8, 0.025);
      var tr = 'translate(960,' + CHART.baselineY + ') scale(' + push.toFixed(5) + ') translate(-960,' + (-CHART.baselineY) + ')';
      vrow.root.setAttribute('transform', tr);
      [halo, ring, tag, bracket, bracketLabel, ourLine].forEach(function (el) { el.setAttribute('transform', tr); });
      nearItems.forEach(function (it) { it.gapEl.setAttribute('transform', tr); });
      renderVRow(vrow, t);

      if (ours.h > 0) {
        var x0 = ours.cx - CHART.bw / 2;
        var hp = Ease.outQuart((t - (LIFT + 0.1)) / 0.4);
        setAttrs(halo, { x: (x0 - 12).toFixed(2), y: (ours.top - 14).toFixed(2), width: CHART.bw + 24, height: (ours.h + 14).toFixed(2), opacity: (0.95 * hp).toFixed(3) });
        var q = Ease.outQuart((t - (LIFT + 0.12)) / 0.7);
        var rs = 1 + 0.18 * q, rcx = ours.cx, rcy = ours.top + ours.h / 2;
        setAttrs(ring, { x: (x0 - 6).toFixed(2), y: (ours.top - 6).toFixed(2), width: CHART.bw + 12, height: (ours.h + 6).toFixed(2),
          opacity: (q > 0 && q < 1 ? 0.8 * (1 - q) : 0).toFixed(3) });
        ring.setAttribute('transform', tr + ' translate(' + rcx + ',' + rcy + ') scale(' + rs.toFixed(4) + ') translate(' + (-rcx) + ',' + (-rcy) + ')');
        // Value pop as the count-up lands.
        var pop = 1 + 0.14 * impulse(t - (LIFT + 0.55), 18, 0.35);
        ours.valueEl.setAttribute('transform', 'translate(' + ours.vx.toFixed(2) + ',' + ours.vy.toFixed(2) + ') scale(' + pop.toFixed(4) + ')');
        setAttrs(tag, { x: ours.cx.toFixed(2), y: CHART.baselineY - 14, opacity: Ease.outQuart((t - (LIFT + 0.4)) / 0.3).toFixed(3) });
      } else { halo.setAttribute('opacity', 0); ring.setAttribute('opacity', 0); tag.setAttribute('opacity', 0); }

      // Bracket draws at SPEC + 1.0 (stroke-dashoffset), label follows; both leave before the morph.
      var specItems = items.slice(items.length - specAbove.length);
      var bOut = 1; // no morph any more, the bracket stays to the end of the scene
      var lastSpec = specItems[specItems.length - 1];
      if (lastSpec && lastSpec.h > 0 && bOut > 0) {
        // One bracket from our bar across the specialists above us, with a tick down to each of them.
        var ox = ours.cx, by = 340, ex = lastSpec.cx;
        var d = 'M' + ox.toFixed(2) + ' ' + (by + 18) + ' L' + ox.toFixed(2) + ' ' + by;
        specItems.forEach(function (s, i) {
          d += ' L' + s.cx.toFixed(2) + ' ' + by + ' L' + s.cx.toFixed(2) + ' ' + (by + 18);
          if (i < specItems.length - 1) d += ' L' + s.cx.toFixed(2) + ' ' + by;
        });
        bracket.setAttribute('d', d);
        var L = 18 + Math.abs(ex - ox) + 36 * (specItems.length - 1) + 18;
        var bp = Ease.inOutCubic((t - (SPEC + 1.0)) / 0.6);
        setAttrs(bracket, { 'stroke-dasharray': L.toFixed(2), 'stroke-dashoffset': (L * (1 - bp)).toFixed(2), opacity: bp > 0 ? bOut.toFixed(3) : 0 });
        var blp = Ease.outExpo((t - (SPEC + 1.3)) / 0.6);
        setAttrs(bracketLabel, { x: ((ox + ex) / 2).toFixed(2), y: (by - 14 + 12 * (1 - blp)).toFixed(2), opacity: (Ease.outQuart((t - (SPEC + 1.3)) / 0.3) * bOut).toFixed(3) });
      } else { bracket.setAttribute('opacity', 0); bracketLabel.setAttribute('opacity', 0); }

      // xx->en chart removed (author); these stay hidden.
      var lp = 0;
      var ly = CHART.baselineY - OURS_XX * vrow.ppu, lx1 = vrow.tx - 10, lx2 = vrow.tx + items.length * (CHART.bw + CHART.gap);
      setAttrs(ourLine, { x1: lx1.toFixed(2), x2: (lx1 + (lx2 - lx1) * lp).toFixed(2), y1: ly.toFixed(2), y2: ly.toFixed(2), opacity: lp > 0 ? 0.75 : 0 });
      nearItems.forEach(function (it) {
        if (!(it.h > 0)) { it.gapEl.setAttribute('opacity', 0); return; } // bar not on screen yet
        it.gapEl.setAttribute('opacity', 0);
      });

      renderCaption(metricEn, lt, 0.3);
      renderCaption(capA, lt, 0, LIFT - start - 0.1);
      renderCaption(capB, lt, LIFT - start + 0.2, 32 - start - 0.2);
      renderCaption(capC, lt, 32 - start);
      renderCaption(capD, lt, XX - start - 1.5);
      foot.style.opacity = Ease.outQuart((lt - 0.3) / 0.5); // on screen for the whole scene
    });
  })();

  // ================================================================= scene 5: "Built for deployment" part 1, quantization (44-52 s)
  // Section title, then: left half three nested circles, circle AREA encodes model size
  // (r = k * sqrt(size_mb)), all tangent to one bottom point (float32 largest, int8 innermost);
  // right half the three chrF++ bars. A variant's circle and its bar enter together on one spring
  // and share one colour on a light-to-dark violet ramp. Only English -> Indic is claimed.
  // At 51.35-52 everything but the int8 circle fades and the int8 circle shrinks into the
  // "171 MB" tile's disc in scene 7 (match-cut, same position and size at t = 52).

  var DISC = { x: 460, y: 330, r: 28 }; // scene 7 tile-0 disc, shared with the match-cut

  (function sceneQuant() {
    var start = 44, end = 52;
    var sec = $('scene-5'), svg = $('s5-svg'), small = $('s5-smalltext'), foot = $('s5-footnote');
    var int8Fill = addOursGradient(svg, 'g-ours-5');
    var qz = DATA.quantization_ct2;
    function variant(name) { return qz.filter(function (v) { return v.variant === name; })[0]; }
    var f32 = variant('float32'), i16 = variant('int16'), i8 = variant('int8');
    var ratio = (f32.size_mb / i8.size_mb).toFixed(1); // 3.7
    var drop = (f32.en_xx_chrf - i8.en_xx_chrf).toFixed(2);
    var enXxSig = DATA.significance_test.en_xx;
    var int8Same = enXxSig.int8_p >= 0.05; // paired bootstrap: no significant difference from float32

    // Circles share a bottom tangent point (CX, BASE): the free band at the top of each ring is
    // 2 * (r_outer - r_inner), so each ring's label fits in its own band without leader lines.
    var BASE = 800, CX = 500, R_MAX = 230, k = R_MAX / Math.sqrt(f32.size_mb); // area ∝ size_mb
    var barMax = Math.ceil(f32.en_xx_chrf / 10) * 10, BAR_H = 340, BW = 110, BX0 = 1150, BGAP = 70;
    // Violet ramp: float32 muted violet, int16 violet, int8 magenta -> violet gradient.
    var V = [
      { v: f32, fill: 'var(--q-f32)', at: 1.5, labelCls: 'ring' },
      { v: i16, fill: 'var(--q-i16)', at: 3.0, labelCls: 'ring' },
      { v: i8, fill: int8Fill, at: 4.5, int8: true, labelCls: 'on-dark' }
    ];
    V.forEach(function (d) { d.r = k * Math.sqrt(d.v.size_mb); });
    var mainG = svgEl('g', null, svg);   // everything that fades before the match-cut
    var discG = svgEl('g', null, svg);   // the int8 circle, which carries over into scene 7
    // (no baseline: it showed as a stray line under the "Built for deployment" title)
    V.forEach(function (d, c) {
      d.barX = BX0 + c * (BW + BGAP);
      d.fullH = d.v.en_xx_chrf / barMax * BAR_H;
      d.g = svgEl('g', null, mainG);
      d.circle = svgEl('circle', { cx: CX, cy: BASE - d.r, r: d.r, fill: d.fill, stroke: 'var(--chart-ours)', 'stroke-width': 3 }, d.int8 ? discG : d.g);
      var labelY = d.int8 ? BASE - d.r + 8 : BASE - 2 * d.r + (d.r - V[c + 1].r) + 8;
      d.label = svgEl('text', { class: 'q-label ' + d.labelCls, x: CX, y: labelY, 'text-anchor': 'middle' }, d.int8 ? discG : d.g);
      d.label.textContent = d.v.variant + ' · ' + Math.round(d.v.size_mb) + ' MB';
      d.bar = svgEl('rect', { x: d.barX, width: BW, rx: 6, fill: d.fill, stroke: 'var(--chart-ours)', 'stroke-width': 3 }, d.g);
      d.value = svgEl('text', { class: 'q-value', 'text-anchor': 'middle' }, d.g);
      d.barLabel = svgEl('text', { class: 'q-label' + (d.int8 ? ' int8' : ''), x: d.barX + BW / 2, y: BASE + 38, 'text-anchor': 'middle' }, d.g);
      d.barLabel.textContent = d.v.variant;
    });
    var refLine = svgEl('line', { stroke: 'var(--amber-text)', 'stroke-width': 2, 'stroke-dasharray': '8 8', opacity: 0 }, mainG);
    small.textContent = 'Circle area = model size. Bars = chrF++, IN22-Gen, English → Indic.';
    var enXx = enXxSig;
    foot.textContent = 'English → Indic, IN22-Gen, CTranslate2 on CPU, ' + DATA.deployment.checkpoint + '. ' +
      'sacrebleu paired bootstrap, 1,000 resamples, float32 baseline: int16 p = ' + enXx.int16_p.toFixed(2) +
      ', int8 p = ' + enXx.int8_p.toFixed(2) + (int8Same ? ' (no significant difference).' : ' (significant, −' + drop + ' chrF++).') +
      ' int8 is the served model.';

    var title = makeCaption(sec, ['Built for', { t: 'deployment.', cls: 'hl' }], 'caption section-title');
    var caps = [
      makeCaption(sec, ['Full precision:', { t: Math.round(f32.size_mb) + ' MB.', cls: 'hl' }]),
      makeCaption(sec, ['int16: half the size.']),
      makeCaption(sec, ['int8:', { t: Math.round(i8.size_mb) + ' MB.', cls: 'hl' }]),
      int8Same ? makeCaption(sec, [{ t: ratio + '×', cls: 'hl' }, 'smaller.', { t: 'Same quality.', cls: 'hl' }])
               : makeCaption(sec, [{ t: ratio + '×', cls: 'hl' }, 'smaller. Only', { t: drop + ' chrF++', cls: 'hl' }, 'lower.'])
    ];
    var d8 = V[2], c8 = { x: CX, y: BASE - d8.r };

    SCENES.push(function (t) {
      var lt = localTime(t, start, end);
      if (!sceneEnvelope(sec, t, start, end, { cutOut: true })) return;
      var fin = Ease.inOutCubic((lt - 6.0) / 0.5);
      // Match-cut progress (off under reduced motion, whose end state is the full chart).
      var m = REDUCED ? 0 : Ease.inOutCubic((lt - 7.35) / 0.65);
      var fade = REDUCED ? 0 : Ease.outQuart((lt - 7.3) / 0.35);
      mainG.setAttribute('opacity', (1 - fade).toFixed(3));
      V.forEach(function (d) {
        var dt = lt - d.at;
        if (dt <= 0) { d.g.setAttribute('opacity', 0); if (d.int8) discG.setAttribute('opacity', 0); return; }
        var gp = spring(dt, SPR.pair.w, SPR.pair.z); // one curve for circle and bar
        var pulse = d.int8 ? 1 + 0.07 * impulse(lt - 6.0, 16, 0.3) : 1;
        if (d.int8 && m > 0) {
          // shrink and travel into the scene 7 disc
          var X = lerp(c8.x, DISC.x, m), Y = lerp(c8.y, DISC.y, m), S = lerp(1, DISC.r / d8.r, m) * gp * pulse;
          d.circle.setAttribute('transform', 'translate(' + X.toFixed(2) + ',' + Y.toFixed(2) + ') scale(' + S.toFixed(4) + ') translate(' + (-c8.x) + ',' + (-c8.y).toFixed(2) + ')');
        } else {
          // circles grow from the shared bottom tangent point, so they stay nested while growing
          d.circle.setAttribute('transform', 'translate(' + CX + ',' + BASE + ') scale(' + (gp * pulse).toFixed(4) + ') translate(' + (-CX) + ',' + (-BASE) + ')');
        }
        var h = d.fullH * gp;
        setAttrs(d.bar, { y: (BASE - h).toFixed(2), height: Math.max(h, 0).toFixed(2) });
        setAttrs(d.value, { x: d.barX + BW / 2, y: (BASE - h - 14).toFixed(2), opacity: Ease.outQuart((dt - 0.05) / 0.25).toFixed(3) });
        d.value.textContent = countUp(d.v.en_xx_chrf, dt, 0, 0.8).toFixed(2);
        var lp = Ease.outExpo((dt - 0.15) / 0.6), lo = Ease.outQuart((dt - 0.15) / 0.3);
        var ltr = 'translate(0,' + (14 * (1 - lp)).toFixed(2) + ')';
        d.label.setAttribute('opacity', (d.int8 ? lo * (1 - fade) : lo).toFixed(3)); d.label.setAttribute('transform', ltr);
        d.barLabel.setAttribute('opacity', lo.toFixed(3)); d.barLabel.setAttribute('transform', ltr);
        d.g.setAttribute('opacity', (d.int8 ? 1 : 1 - 0.4 * fin).toFixed(3));
        if (d.int8) discG.setAttribute('opacity', 1);
      });
      // "Same height" reference: dashed line at float32's bar top, drawn across all three bars.
      var y = BASE - V[0].fullH, x1 = V[0].barX - 12, x2 = V[2].barX + BW + 12;
      var rp = Ease.inOutCubic((lt - 6.2) / 0.7);
      setAttrs(refLine, { x1: x1, y1: y, y2: y, x2: (x1 + (x2 - x1) * rp).toFixed(2), opacity: rp > 0 ? 0.8 : 0 });
      renderCaption(title, lt, 0, 1.3);
      renderCaption(caps[0], lt, 1.5, 2.85);
      renderCaption(caps[1], lt, 3.0, 4.35);
      renderCaption(caps[2], lt, 4.5, 5.85);
      renderCaption(caps[3], lt, 6.0, REDUCED ? undefined : 7.3);
      small.style.opacity = Ease.outQuart((lt - 1.8) / 0.5) * (1 - fade);
      foot.style.opacity = Ease.outQuart((lt - 1.5) / 0.5) * (1 - fade);
    });
  })();

  // ================================================================= scene 5b: LLM judge (40-44 s)
  // A 10 x 10 waffle: one cell per percent of the English -> Indic judgements of the final model
  // (DATA.llm_judge, V2 with the human reference; test sets and inference path named in the data). Cells fill row by row: first the "perfect
  // 5/5/5" share (gradient), then the rest of the "mean of the three >= 4" share (lighter).

  (function sceneJudge() {
    var start = 40, end = 44;
    var sec = $('scene-5b'), svg = $('s5b-svg');
    var big = $('s5b-big'), small = $('s5b-smalltext'), foot = $('s5b-footnote');
    var J = DATA.llm_judge, V2 = J.v2_with_reference, V1 = J.v1_reference_free;
    var ge4 = Math.round(V2.mean_ge4_pct);                            // mean of the three >= 4, V2
    var perfect = Math.round(V2.perfect_pct);                         // perfect 5/5/5, V2
    var fill = addOursGradient(svg, 'g-ours-5b');
    var GRID = { n: 10, x0: 650 - (10 * 52 - 8) / 2, y0: 300, cell: 44, gap: 8 };
    var cells = [];
    for (var i = 0; i < 100; i++) {
      var r = Math.floor(i / GRID.n), c = i % GRID.n;
      var x = GRID.x0 + c * (GRID.cell + GRID.gap), y = GRID.y0 + r * (GRID.cell + GRID.gap);
      var el = svgEl('rect', { x: x, y: y, width: GRID.cell, height: GRID.cell, rx: 8, fill: 'var(--chart-ours)', 'fill-opacity': 0.10, opacity: 0 }, svg);
      cells.push({ el: el, cx: x + GRID.cell / 2, cy: y + GRID.cell / 2, d: r + c, i: i });
    }

    big.innerHTML = '<span class="judge-num num" id="s5b-ge4"></span>' +
      '<span class="judge-sub">rated 4 or above out of 5</span>' +
      '<span class="judge-num judge-num-2 num" id="s5b-perfect"></span>' +
      '<span class="judge-sub">perfect on fluency, adequacy and overall</span>';
    var ge4El = $('s5b-ge4'), perfEl = $('s5b-perfect');
    small.innerHTML = '<span class="judge-key" style="background:linear-gradient(#EC4899,#7C3AED)"></span>perfect 5 / 5 / 5' +
      '<span class="judge-key" style="background:var(--chart-ours);opacity:0.45"></span>4 or above' +
      '<span class="judge-key" style="background:var(--chart-ours);opacity:0.12"></span>below 4';
    foot.textContent = 'Judge: ' + J.judge + ' with the human reference, ' + J.n_items.toLocaleString('en-US') +
      ' English → Indic sentences (' + J.test_sets + '), final model' + (J.runtime ? ', ' + J.runtime : '') +
      '. Without the reference: ' + Math.round(V1.mean_ge4_pct) + '% and ' + Math.round(V1.perfect_pct) +
      '%. Human references: about ' + J.human_reference_perfect_pct + '% perfect. The judge is also one of the compared systems.';

    var cap = makeCaption(sec, ['Judged', { t: 'sentence by sentence.', cls: 'hl' }]);
    var sub = makeCaption(sec, [J.judge + ' scores every translation against the human reference.'], 'caption-sub');

    SCENES.push(function (t) {
      var lt = localTime(t, start, end);
      if (!sceneEnvelope(sec, t, start, end)) return;
      renderCaption(cap, lt, 0);
      renderCaption(sub, lt, 0.4);
      // cells pop in on a diagonal, then fill to the measured shares
      var f = countUp(ge4, lt, 1.0, 1.3);                            // cells filled so far, 0..57
      cells.forEach(function (s) {
        var p = (lt - (0.3 + s.d * 0.02)) / 0.35;
        var sc = p <= 0 ? 0 : Ease.outBack(p);
        s.el.setAttribute('opacity', p > 0 ? 1 : 0);
        s.el.setAttribute('transform', 'translate(' + s.cx + ',' + s.cy + ') scale(' + sc.toFixed(4) + ') translate(' + (-s.cx) + ',' + (-s.cy) + ')');
        var on = Math.max(0, Math.min(1, f - s.i));                     // partial fill on the leading cell
        if (s.i < perfect) {
          s.el.setAttribute('fill', on > 0 ? fill : 'var(--chart-ours)');
          s.el.setAttribute('fill-opacity', on > 0 ? (0.10 + 0.90 * on).toFixed(3) : 0.10);
        } else {
          s.el.setAttribute('fill', 'var(--chart-ours)');
          s.el.setAttribute('fill-opacity', (0.10 + 0.35 * on).toFixed(3));
        }
      });
      fadeRise(big, lt, 1.0, 'translate(-50%, -50%)');
      ge4El.textContent = Math.round(countUp(ge4, lt, 1.0, 1.3)) + '%';
      perfEl.textContent = Math.round(countUp(perfect, lt, 1.0, 1.3)) + '%';
      small.style.opacity = Ease.outQuart((lt - 1.6) / 0.4);
      foot.style.opacity = Ease.outQuart((lt - 0.6) / 0.5);
    });
  })();

  // ================================================================= scene 7: "Built for deployment" part 2 (52-58 s)
  // Runs on a CPU: three spec tiles (int8 size, p50 latency, int8/float32 throughput), all from
  // DATA.deployment (quantization_report.md, final model, step 1,800). Then "One model. Every
  // direction.": 22 Indic languages around English, with pivot routes Indic -> English -> Indic.

  (function sceneDeploy() {
    var start = 52, end = 58, PHASE2 = 2.0;
    var sec = $('scene-7'), wrap = $('s7-tiles'), icons = $('s7-icons'), svg = $('s7-svg');
    var small = $('s7-smalltext'), foot = $('s7-footnote');
    var DEP = DATA.deployment, L = DATA.languages;
    var int8Fill = addOursGradient(icons, 'g-ours-7');
    var hubFill = addOursGradient(svg, 'g-ours-7b');

    var sizeMB = Math.round(DEP.int8_size_mb);                       // 171
    var latS = Math.round(DEP.int8_p50_ms / 100) / 10;               // 374.2 ms -> 0.4 s
    var speedup = DEP.int8_batch32_sent_per_s / DEP.float32_batch32_sent_per_s; // 8.4 / 5.9 = 1.42
    var TILES = [
      { x: 260, fmt: function (p) { return Math.round(sizeMB * p) + ' MB'; }, label: 'int8 model' },
      { x: 760, fmt: function (p) { return '~' + (latS * p).toFixed(1) + ' s'; }, label: 'per sentence, ' + DEP.intra_threads + ' CPU threads' },
      { x: 1260, fmt: function (p) { return (speedup * p).toFixed(1) + '×'; }, label: 'int8 vs float32 throughput' }
    ];
    TILES.forEach(function (d, i) {
      d.el = $('s7-tile-' + i);
      d.el.style.left = d.x + 'px';
      d.valueEl = d.el.querySelector('.dep-value');
      d.valueEl.className = 'dep-value num hl';
      d.el.querySelector('.dep-label').textContent = d.label;
      d.at = 0.05 + i * 0.2;
      d.cx = d.x + 200;
    });
    // Tile icons: the int8 disc (match-cut target), a clock ring, two throughput bars.
    var disc = svgEl('circle', { cx: DISC.x, cy: DISC.y, r: DISC.r, fill: int8Fill, stroke: 'var(--chart-ours)', 'stroke-width': (3 * DISC.r / 119.6).toFixed(2) }, icons);
    var clock = svgEl('g', { opacity: 0 }, icons);
    svgEl('circle', { cx: TILES[1].cx, cy: DISC.y, r: 26, fill: 'none', stroke: 'var(--chart-ours)', 'stroke-width': 4 }, clock);
    svgEl('path', { d: 'M' + TILES[1].cx + ' ' + DISC.y + ' L' + TILES[1].cx + ' ' + (DISC.y - 16) + ' M' + TILES[1].cx + ' ' + DISC.y + ' L' + (TILES[1].cx + 11) + ' ' + (DISC.y + 6), stroke: 'var(--chart-ours)', 'stroke-width': 4, 'stroke-linecap': 'round' }, clock);
    var bars = svgEl('g', { opacity: 0 }, icons);
    svgEl('rect', { x: TILES[2].cx - 24, y: DISC.y + 2, width: 18, height: 24, rx: 3, fill: 'var(--chart-other)', opacity: 0.55 }, bars);
    svgEl('rect', { x: TILES[2].cx + 6, y: DISC.y - 26, width: 18, height: 52, rx: 3, fill: 'var(--chart-ours)' }, bars);
    var iconGs = [null, clock, bars];

    // Hub-and-spoke: English at the centre, the 22 Indic language tags on an ellipse.
    var HUB = { x: 960, y: 715, rx: 600, ry: 165 };
    var spokesG = svgEl('g', null, svg), nodesG = svgEl('g', null, svg);
    var nodes = L.tags.map(function (tag, i) {
      var a = -Math.PI / 2 + (i + 0.5) * 2 * Math.PI / L.tags.length;
      var n = { tag: tag, x: HUB.x + HUB.rx * Math.cos(a), y: HUB.y + HUB.ry * Math.sin(a), a: a, at: PHASE2 + 0.15 + i * 0.03 };
      n.spoke = svgEl('line', { x1: HUB.x, y1: HUB.y, x2: n.x.toFixed(1), y2: n.y.toFixed(1), stroke: 'var(--accent-amber)', 'stroke-width': 2, opacity: 0.55 }, spokesG);
      n.len = Math.sqrt(Math.pow(n.x - HUB.x, 2) + Math.pow(n.y - HUB.y, 2));
      setAttrs(n.spoke, { 'stroke-dasharray': n.len.toFixed(1), 'stroke-dashoffset': n.len.toFixed(1) });
      n.dot = svgEl('circle', { cx: n.x.toFixed(1), cy: n.y.toFixed(1), r: 10, fill: 'var(--q-f32)', stroke: 'var(--chart-ours)', 'stroke-width': 2.5 }, nodesG);
      n.label = svgEl('text', { class: 'graph-label', x: (HUB.x + (HUB.rx + 42) * Math.cos(a)).toFixed(1), y: (HUB.y + (HUB.ry + 30) * Math.sin(a) + 7).toFixed(1), 'text-anchor': 'middle', opacity: 0 }, nodesG);
      n.label.textContent = (L.autonyms && L.autonyms[tag]) || tag; // each language in its own script
      return n;
    });
    var hub = svgEl('g', null, svg);
    svgEl('circle', { cx: HUB.x, cy: HUB.y, r: 58, fill: hubFill }, hub);
    var hubText = svgEl('text', { class: 'graph-hub', x: HUB.x, y: HUB.y + 9, 'text-anchor': 'middle' }, hub);
    hubText.textContent = 'English';
    // Pivot routes (Indic -> English -> Indic), one per beat; tags looked up, not positions.
    function node(tag) { return nodes.filter(function (n) { return n.tag === tag; })[0] || nodes[0]; }
    var ROUTES = [['ta', 'hi'], ['bn', 'ml'], ['gu', 'ur'], ['mni', 'kn'], ['sat', 'mr'], ['as', 'te']].map(function (r, i) {
      var dot = svgEl('circle', { r: 9, fill: 'var(--accent-orange)', opacity: 0 }, svg);
      return { a: node(r[0]), b: node(r[1]), at: PHASE2 + 0.9 + i * 0.5, dot: dot };
    });

    var capA = makeCaption(sec, [{ t: 'No GPUs needed.', cls: 'hl' }, 'Deploy on CPUs and edge devices.']);
    var capB = makeCaption(sec, ['One model.', { t: 'Every direction.', cls: 'hl' }]);
    var capSub = makeCaption(sec, [{ t: String(L.directions), cls: 'hl' }, 'English ↔ Indic directions, and any two of', { t: String(L.indic), cls: 'hl' }, 'Indic languages through English.'], 'caption-sub');
    small.textContent = 'One CTranslate2 model behind a FastAPI translation API and web frontend.';
    foot.textContent = 'CTranslate2 int8 on CPU (' + DEP.intra_threads + ' threads, beam 5), IN22-Gen, ' + DEP.checkpoint + '. Latency: single-sentence p50.';

    SCENES.push(function (t) {
      var lt = localTime(t, start, end);
      if (!sceneEnvelope(sec, t, start, end, { cutIn: true })) return;
      // Phase 1: tiles form around the arriving disc and count up.
      TILES.forEach(function (d, i) {
        var p = Ease.outExpo((lt - d.at) / 0.7);
        d.el.style.opacity = Ease.outQuart((lt - d.at) / 0.35);
        d.el.style.transform = 'translate3d(0,' + (i === 0 ? 0 : 30 * (1 - p)).toFixed(2) + 'px,0) scale(' + (0.97 + 0.03 * p).toFixed(4) + ')';
        d.valueEl.textContent = d.fmt(Ease.outExpo((lt - d.at - 0.1) / 0.8));
        if (iconGs[i]) {
          var ip = lt - d.at <= 0 ? 0 : Ease.outBack((lt - d.at) / 0.4);
          iconGs[i].setAttribute('opacity', lt > d.at ? 1 : 0);
          iconGs[i].setAttribute('transform', 'translate(' + d.cx + ',' + DISC.y + ') scale(' + ip.toFixed(4) + ') translate(' + (-d.cx) + ',' + (-DISC.y) + ')');
        }
      });
      disc.setAttribute('opacity', 1);
      // Phase 2: tiles glide up and shrink; the language graph grows below.
      var g2 = Ease.inOutCubic((lt - PHASE2) / 0.6);
      wrap.style.transform = 'translate3d(0,' + (-80 * g2).toFixed(2) + 'px,0) scale(' + (1 - 0.3 * g2).toFixed(4) + ')';
      var hp = lt <= PHASE2 ? 0 : spring(lt - PHASE2, SPR.hero.w, SPR.hero.z);
      hub.setAttribute('transform', 'translate(' + HUB.x + ',' + HUB.y + ') scale(' + hp.toFixed(4) + ') translate(' + (-HUB.x) + ',' + (-HUB.y) + ')');
      nodes.forEach(function (n) {
        var sp = Ease.outExpo((lt - n.at) / 0.5);
        n.spoke.setAttribute('stroke-dashoffset', (n.len * (1 - sp)).toFixed(1));
        var np = lt - n.at - 0.2 <= 0 ? 0 : Ease.outBack((lt - n.at - 0.2) / 0.35);
        var bump = 0;
        ROUTES.forEach(function (r) { if (r.a === n) bump += impulse(lt - r.at, 18, 0.4); if (r.b === n) bump += impulse(lt - r.at - 0.5, 18, 0.4); });
        n.dot.setAttribute('transform', 'translate(' + n.x.toFixed(1) + ',' + n.y.toFixed(1) + ') scale(' + (np * (1 + 0.5 * bump)).toFixed(4) + ') translate(' + (-n.x).toFixed(1) + ',' + (-n.y).toFixed(1) + ')');
        n.label.setAttribute('opacity', Ease.outQuart((lt - n.at - 0.25) / 0.3).toFixed(3));
      });
      // Pivot dots: source -> EN (0.25 s), EN -> target (0.25 s).
      ROUTES.forEach(function (r) {
        var q = (lt - r.at) / 0.5;
        if (REDUCED || q <= 0 || q >= 1.15) { r.dot.setAttribute('opacity', 0); return; }
        var x, y;
        if (q < 0.5) { var e1 = Ease.inOutCubic(q / 0.5); x = lerp(r.a.x, HUB.x, e1); y = lerp(r.a.y, HUB.y, e1); }
        else { var e2 = Ease.inOutCubic((q - 0.5) / 0.5); x = lerp(HUB.x, r.b.x, e2); y = lerp(HUB.y, r.b.y, e2); }
        setAttrs(r.dot, { cx: x.toFixed(1), cy: y.toFixed(1), opacity: q > 1 ? ((1.15 - q) / 0.15).toFixed(3) : 1 });
      });
      renderCaption(capA, lt, 0, PHASE2 - 0.15);
      renderCaption(capB, lt, PHASE2);
      renderCaption(capSub, lt, PHASE2 + 0.3);
      small.style.opacity = Ease.outQuart((lt - PHASE2 - 1.0) / 0.5);
      foot.style.opacity = Ease.outQuart((lt - 0.2) / 0.5);
    });
  })();

  // ================================================================= scene 6: end card (58 s to the end of the music)
  // Three beats on the bar grid, one continuous wordmark:
  //   58-62  beat 1: IndicMT, underline, stat tiles (154M, 171 MB, English <-> 22)
  //   62-66  beat 2: the wordmark shrinks to a header; blog, technical report and model-weights link cards
  //   66-70  beat 3: everything clears, IndicMT glides to the centre with a soft glow and
  //            "Built for India. To deploy at scale." lands under it; holds until the audio ends.

  (function sceneEnd() {
    var start = 58, end = 1e4; // runs to the end of the audio (TOTAL), no exit
    var BEAT2 = 4.0, BEAT3 = 8.0; // local: 62 s and 66 s (bar lines at 120 BPM); links hold 4 s
    var sec = $('scene-6');
    var title = $('s6-title'), underline = $('s6-underline'), closing = $('s6-closing'), glow = $('s6-glow');
    var tileWrap = $('s6-tiles'), tiles = [$('s6-tile-0'), $('s6-tile-1'), $('s6-tile-2')];
    var links = [$('s6-link-0'), $('s6-link-1'), $('s6-link-3'), $('s6-link-2')];
    var tSpans = buildMasked(title, [{ t: 'Indic' }, { t: 'MT', cls: 'accent' }], 'letters');
    var cSpans = buildMasked(closing, ['Built for India.', 'To deploy at scale.'], 'words');
    var i8 = DATA.quantization_ct2.filter(function (v) { return v.variant === 'int8'; })[0];
    var TILES = [
      { n: HEADLINE_M, pre: '', suf: 'M', label: 'parameters' },
      { n: Math.round(i8.size_mb), pre: '', suf: ' MB', label: 'int8 model size' },
      { n: LANG.indic, pre: 'English ↔ ', suf: '', label: 'Indic languages' }
    ];
    tiles.forEach(function (el, i) {
      TILES[i].valueEl = el.querySelector('.stat-value');
      el.querySelector('.stat-label').textContent = TILES[i].label;
    });
    // Link URLs come only from DATA.links; a null value shows a labelled placeholder.
    var shown = DATA.links.display || {};
    ['blog', 'report', 'code', 'weights'].forEach(function (key, i) {
      var url = DATA.links[key];
      var el = links[i].querySelector('.link-url');
      el.textContent = '';
      if (url) el.textContent = (shown[key] || url.replace(/^https?:\/\//, '')).replace(/\//g, '/\u200B'); // wrap only after a slash
      else { var ph = document.createElement('span'); ph.className = 'placeholder'; ph.textContent = 'link coming soon'; el.appendChild(ph); }
    });
    // Wordmark path: beat 1 at its own spot (top 26%), beat 2 header (top 9%, half size),
    // beat 3 exact stage centre, 1.3x. Offsets are relative to the .end-title anchor.
    var Y1 = 0.26 * 1080, Y2 = 0.09 * 1080, Y3 = 470; // beat 3: a little above centre, closing line below

    SCENES.push(function (t) {
      var lt = localTime(t, start, end);
      // Reduced motion: show each beat's settled state for its own time window.
      if (REDUCED && lt >= 0) { var r = t - start; lt = r < BEAT2 ? BEAT2 - 0.4 : (r < BEAT3 ? BEAT3 - 0.6 : 1e3); }
      if (!sceneEnvelope(sec, t, start, end, { push: pushIn(lt, 1, 2.6, 0.02) })) return;
      // Beat 1
      revealSpans(tSpans, lt, 0, 0.035, 0.7);
      var out1 = Ease.inCubic((lt - (BEAT2 - 0.3)) / 0.3);
      var o1 = (1 - out1).toFixed(3), lift1 = 'translate3d(0,' + (-20 * out1).toFixed(2) + 'px,0)';
      underline.style.transform = 'scaleX(' + Ease.inOutCubic((lt - 0.3) / 0.6).toFixed(4) + ')';
      underline.style.opacity = o1;
      tiles.forEach(function (el, i) {
        var at = 0.6 + i * 0.15;
        var p = Ease.outExpo((lt - at) / 0.8);
        el.style.opacity = Ease.outQuart((lt - at) / 0.5);
        el.style.transform = 'translate3d(0,' + (40 * (1 - p)).toFixed(2) + 'px,0) scale(' + (0.96 + 0.04 * p).toFixed(4) + ')';
        var d = TILES[i];
        d.valueEl.textContent = d.pre + Math.round(countUp(d.n, lt, at, 0.9)) + d.suf;
      });
      tileWrap.style.opacity = o1;
      tileWrap.style.transform = 'translate(-50%,0) ' + lift1;
      // Closing line belongs to beat 3: it reveals under the centred wordmark.
      revealSpans(cSpans, lt, BEAT3 + 0.35, 0.09, 0.7);
      closing.style.opacity = lt >= BEAT3 + 0.3 ? 1 : 0;
      closing.style.transform = 'translate(-50%,0)';
      // Beat 2: link cards rise in, then clear for beat 3.
      var out2 = Ease.inCubic((lt - (BEAT3 - 0.45)) / 0.3);
      links.forEach(function (el, i) {
        var at = BEAT2 + 0.05 + i * 0.15;
        var p = Ease.outExpo((lt - at) / 0.7);
        el.style.opacity = (Ease.outQuart((lt - at) / 0.4) * (1 - out2)).toFixed(3);
        el.style.transform = 'translate3d(0,' + (30 * (1 - p) - 20 * out2).toFixed(2) + 'px,0) scale(' + (0.97 + 0.03 * p).toFixed(4) + ')';
      });
      // Wordmark: beat 1 -> header (match-cut into beat 2) -> exact centre (beat 3), settle on 64 s.
      var m1 = Ease.inOutCubic((lt - (BEAT2 - 0.3)) / 0.6);
      var m2 = Ease.inOutCubic((lt - (BEAT3 - 0.5)) / 0.5);
      var y = lerp(lerp(Y1, Y2, m1), Y3, m2) - Y1;
      var sc = lerp(lerp(1, 0.5, m1), 1.3, m2) * (1 + 0.035 * impulse(lt - BEAT3, 14, 0.45));
      title.style.transform = 'translate(-50%,-50%) translate3d(0,' + y.toFixed(2) + 'px,0) scale(' + sc.toFixed(4) + ')';
      var gp = Ease.outQuart((lt - BEAT3) / 0.8);
      glow.style.opacity = (0.9 * gp).toFixed(3);
      glow.style.transform = 'scale(' + (0.85 + 0.15 * Ease.outExpo((lt - BEAT3) / 1.2)).toFixed(4) + ')';
    });
  })();

  // ================================================================= boot

  render(0);
  updateHud();
  requestAnimationFrame(tick);
})();
