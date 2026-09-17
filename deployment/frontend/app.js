// IndicMT frontend - talks to the API over real fetch() calls only (no
// in-process bypass of the API layer). Vanilla JS, no dependencies.
"use strict";

const $ = (id) => document.getElementById(id);
const MAX_CHARS = 2000;
const HISTORY_KEY = "indicmt.history";
const THEME_KEY = "indicmt.theme";
const LANGS_KEY = "indicmt.langs";
const HISTORY_MAX = 50, HISTORY_SHOWN = 8;
const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Native names for display next to the API's English names (display only).
const ENDONYMS = {
  as: "অসমীয়া", bn: "বাংলা", br: "बर'", doi: "डोगरी", gom: "कोंकणी", gu: "ગુજરાતી", hi: "हिन्दी",
  kn: "ಕನ್ನಡ", ks: "کٲشُر", mai: "मैथिली", ml: "മലയാളം", mr: "मराठी", mni: "মৈতৈলোন্", ne: "नेपाली",
  or: "ଓଡ଼ିଆ", pa: "ਪੰਜਾਬੀ", sa: "संस्कृतम्", sd: "سنڌي", sat: "ᱥᱟᱱᱛᱟᱲᱤ", ta: "தமிழ்", te: "తెలుగు", ur: "اردو",
};
const EXAMPLES = [
  { lang: "en", text: "Hello, how are you today?" },
  { lang: "hi", text: "नमस्ते, आप कैसे हैं?" },
];

// ---------- safe storage ----------
function store(key, value) {
  try {
    if (value === undefined) { const v = localStorage.getItem(key); return v === null ? null : v; }
    if (value === null) localStorage.removeItem(key); else localStorage.setItem(key, value);
  } catch (e) { /* storage blocked: app still works */ }
  return null;
}

const els = {
  tabs: [...document.querySelectorAll(".tab")],
  tabIndicator: document.querySelector(".tab-indicator"),
  panels: { translate: $("panel-translate"), benchmarks: $("panel-benchmarks") },
  disclaimer: $("disclaimer-banner"),
  swap: $("swap-btn"),
  src: $("source-text"), count: $("char-count"), countWrap: $("char-count-wrap"), clear: $("clear-btn"),
  paneIn: $("pane-in"), paneOut: $("pane-out"),
  placeholder: $("output-placeholder"), chips: $("example-chips"),
  out: $("output-text"), skeleton: $("skeleton"), slow: $("slow-note"),
  err: $("error-box"), errText: $("error-text"), retry: $("retry-btn"),
  badge: $("route-badge"), pivotNote: $("pivot-note"), pivotText: $("pivot-text"),
  latency: $("latency-chip"), copy: $("copy-btn"), copyLabel: $("copy-label"),
  translate: $("translate-btn"),
  histList: $("history-list"), histEmpty: $("history-empty"), histClear: $("history-clear"),
  toasts: $("toasts"), status: $("live-status"), alert: $("live-alert"),
  helpSheet: $("help-sheet"),
};

const state = {
  langs: [], byCode: {},
  source: "en", target: "hi",
  busy: false, lastReq: null, lastCopyText: "",
};

const nameOf = (c) => state.byCode[c]?.name ?? c;

// ---------- announcements & toasts ----------
function announce(msg, assertive = false) {
  const el = assertive ? els.alert : els.status;
  el.textContent = "";
  setTimeout(() => { el.textContent = msg; }, 30);
}

function toast(msg, kind = "ok", sticky = false) {
  const t = document.createElement("div");
  t.className = `toast glass ${kind}`;
  t.innerHTML = '<span class="msg"></span><button type="button" aria-label="Dismiss">×</button>';
  t.querySelector(".msg").textContent = msg;
  const close = () => {
    if (!t.isConnected) return;
    t.classList.add("out");
    setTimeout(() => t.remove(), reducedMotion() ? 0 : 140);
  };
  t.querySelector("button").addEventListener("click", close);
  while (els.toasts.children.length >= 3) els.toasts.firstElementChild.remove();
  els.toasts.appendChild(t);
  if (!sticky) {
    let timer = setTimeout(close, 5000);
    t.addEventListener("mouseenter", () => clearTimeout(timer));
    t.addEventListener("mouseleave", () => { timer = setTimeout(close, 2500); });
  }
  return close;
}

// ---------- theme ----------
function applyTheme(choice, persist) {
  const root = document.documentElement;
  if (choice === "light" || choice === "dark") root.dataset.theme = choice; else delete root.dataset.theme;
  document.querySelectorAll("[data-theme-choice]").forEach((b) =>
    b.setAttribute("aria-checked", String(b.dataset.themeChoice === choice)));
  if (persist) store(THEME_KEY, choice === "system" ? null : choice);
}
document.querySelectorAll("[data-theme-choice]").forEach((b) =>
  b.addEventListener("click", () => {
    applyTheme(b.dataset.themeChoice, true);
    announce(`Theme: ${b.dataset.themeChoice}`);
  }));
applyTheme(store(THEME_KEY) || "system", false);

// ---------- tabs ----------
function moveIndicator() {
  const active = els.tabs.find((t) => t.classList.contains("active"));
  if (!active) return;
  els.tabIndicator.style.width = active.offsetWidth + "px";
  els.tabIndicator.style.transform = `translateX(${active.offsetLeft}px)`;
}

function switchTab(name, focus = false) {
  const apply = () => {
    els.tabs.forEach((t) => {
      const on = t.dataset.tab === name;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
    });
    for (const [key, panel] of Object.entries(els.panels)) {
      const on = key === name;
      panel.classList.toggle("active", on);
      panel.hidden = !on;
      if (on && !reducedMotion()) { panel.classList.remove("enter"); void panel.offsetWidth; panel.classList.add("enter"); }
    }
    moveIndicator();
  };
  if (document.startViewTransition && !reducedMotion()) document.startViewTransition(apply); else apply();
  if (focus) els.tabs.find((t) => t.dataset.tab === name).focus();
  if (name === "benchmarks" && !bench.loaded) loadBenchmarks();
  try { history.replaceState(null, "", name === "translate" ? location.pathname + location.search : "#" + name); } catch (e) {}
}

els.tabs.forEach((t, i) => {
  t.addEventListener("click", () => switchTab(t.dataset.tab));
  t.addEventListener("keydown", (e) => {
    let j = null;
    if (e.key === "ArrowRight") j = (i + 1) % els.tabs.length;
    else if (e.key === "ArrowLeft") j = (i - 1 + els.tabs.length) % els.tabs.length;
    else if (e.key === "Home") j = 0; else if (e.key === "End") j = els.tabs.length - 1;
    if (j !== null) { e.preventDefault(); switchTab(els.tabs[j].dataset.tab, true); }
  });
});
window.addEventListener("resize", moveIndicator);

// ---------- language pickers ----------
const pop = $("picker-pop"), popSearch = $("picker-search"), popList = $("picker-list"), popEmpty = $("picker-empty"), scrim = $("picker-scrim");
const picker = { role: null, trigger: null, items: [], active: -1 };

function renderPickerValue(role) {
  const code = state[role];
  const el = $(`picker-${role}-val`);
  el.textContent = "";
  el.lang = code;
  el.append(nameOf(code));
  if (ENDONYMS[code]) { const s = document.createElement("small"); s.textContent = ENDONYMS[code]; el.append(s); }
}

function getRecent() {
  try { const r = JSON.parse(store(LANGS_KEY) || "[]"); return Array.isArray(r) ? r.filter((c) => state.byCode[c]) : []; } catch (e) { return []; }
}
function pushRecent(code) {
  const r = [code, ...getRecent().filter((c) => c !== code)].slice(0, 4);
  store(LANGS_KEY, JSON.stringify(r));
}

function highlight(text, q) {
  const frag = document.createDocumentFragment();
  const i = q ? text.toLowerCase().indexOf(q) : -1;
  if (i < 0) { frag.append(text); return frag; }
  frag.append(text.slice(0, i));
  const m = document.createElement("mark"); m.textContent = text.slice(i, i + q.length); frag.append(m);
  frag.append(text.slice(i + q.length));
  return frag;
}

function renderPickerList() {
  const q = popSearch.value.trim().toLowerCase();
  const other = picker.role === "source" ? state.target : state.source;
  const match = (l) => !q || l.name.toLowerCase().includes(q) || l.code.includes(q) || (ENDONYMS[l.code] || "").toLowerCase().includes(q);
  const all = state.langs.filter(match);
  const recent = q ? [] : getRecent().map((c) => state.byCode[c]).filter(Boolean);
  const groups = [];
  if (recent.length) groups.push(["Recent", recent]);
  groups.push([q ? "Results" : "All languages", all]);
  popList.textContent = "";
  picker.items = [];
  let n = 0;
  for (const [title, list] of groups) {
    if (!list.length) continue;
    const g = document.createElement("li"); g.className = "group"; g.setAttribute("role", "presentation"); g.textContent = title;
    popList.append(g);
    for (const l of list) {
      const li = document.createElement("li");
      li.setAttribute("role", "option");
      li.id = `opt-${n++}`;
      li.dataset.code = l.code;
      li.lang = l.code;
      li.setAttribute("aria-selected", String(l.code === state[picker.role]));
      const name = document.createElement("span");
      name.append(highlight(l.name, q));
      if (ENDONYMS[l.code]) { const s = document.createElement("small"); s.className = "muted"; s.style.marginLeft = "8px"; s.textContent = ENDONYMS[l.code]; name.append(s); }
      const code = document.createElement("span");
      code.className = "code";
      code.textContent = l.code === other ? `${l.code} · in use` : l.code;
      li.append(name, code);
      popList.append(li);
      picker.items.push(li);
    }
  }
  popEmpty.classList.toggle("hidden", picker.items.length > 0);
  const sel = picker.items.findIndex((li) => li.getAttribute("aria-selected") === "true");
  setActive(q ? 0 : Math.max(sel, 0), !q);
}

function setActive(i, scroll = true) {
  picker.items.forEach((li) => li.classList.remove("active"));
  picker.active = picker.items.length ? Math.min(Math.max(i, 0), picker.items.length - 1) : -1;
  const li = picker.items[picker.active];
  if (li) {
    li.classList.add("active");
    popSearch.setAttribute("aria-activedescendant", li.id);
    if (scroll) li.scrollIntoView({ block: "nearest" });
  } else popSearch.removeAttribute("aria-activedescendant");
}

function positionPicker() {
  if (window.matchMedia("(max-width: 767px)").matches) { pop.style.top = pop.style.left = ""; return; }
  const r = picker.trigger.getBoundingClientRect();
  pop.style.top = r.bottom + 6 + "px";
  pop.style.left = Math.max(8, Math.min(r.left, window.innerWidth - pop.offsetWidth - 8)) + "px";
}

function openPicker(role) {
  if (!state.langs.length) return;
  closePicker(false);
  picker.role = role;
  picker.trigger = document.querySelector(`#picker-${role} .picker-trigger`);
  picker.trigger.setAttribute("aria-expanded", "true");
  popSearch.value = "";
  pop.classList.remove("hidden");
  scrim.classList.toggle("hidden", !window.matchMedia("(max-width: 767px)").matches);
  renderPickerList();
  positionPicker();
  popSearch.focus();
}

function closePicker(restore = true) {
  if (!picker.role) return;
  picker.trigger.setAttribute("aria-expanded", "false");
  pop.classList.add("hidden"); scrim.classList.add("hidden");
  if (restore) picker.trigger.focus();
  picker.role = null;
}

function choose(code) {
  const role = picker.role;
  const other = role === "source" ? "target" : "source";
  if (code === state[other]) state[other] = state[role]; // picking the other side's language swaps them
  state[role] = code;
  pushRecent(code);
  renderPickerValue("source"); renderPickerValue("target");
  saveLangs();
  closePicker();
  announce(`Translating ${nameOf(state.source)} to ${nameOf(state.target)}`);
}

function saveLangs() { store("indicmt.pair", JSON.stringify([state.source, state.target])); }

document.querySelectorAll(".picker-trigger").forEach((b) => {
  const role = b.closest(".picker").dataset.role;
  b.addEventListener("click", () => (picker.role === role ? closePicker() : openPicker(role)));
  b.addEventListener("keydown", (e) => { if (e.key === "ArrowDown") { e.preventDefault(); openPicker(role); } });
});
popSearch.addEventListener("input", renderPickerList);
popSearch.addEventListener("keydown", (e) => {
  if (e.key === "ArrowDown") { e.preventDefault(); setActive(picker.active + 1); }
  else if (e.key === "ArrowUp") { e.preventDefault(); setActive(picker.active - 1); }
  else if (e.key === "Home") { e.preventDefault(); setActive(0); }
  else if (e.key === "End") { e.preventDefault(); setActive(picker.items.length - 1); }
  else if (e.key === "Enter") { e.preventDefault(); const li = picker.items[picker.active]; if (li) choose(li.dataset.code); }
  else if (e.key === "Escape") { e.preventDefault(); closePicker(); }
  else if (e.key === "Tab") closePicker(false);
});
popList.addEventListener("mousemove", (e) => { const li = e.target.closest('[role="option"]'); if (li) { const i = picker.items.indexOf(li); if (i !== picker.active) setActive(i, false); } });
popList.addEventListener("click", (e) => { const li = e.target.closest('[role="option"]'); if (li) choose(li.dataset.code); });
scrim.addEventListener("click", () => closePicker());
document.addEventListener("pointerdown", (e) => {
  if (picker.role && !pop.contains(e.target) && !picker.trigger.contains(e.target)) closePicker(false);
});
window.addEventListener("resize", () => picker.role && positionPicker());
window.addEventListener("scroll", () => picker.role && positionPicker(), { passive: true });

// ---------- swap ----------
function swapLanguages() {
  if (!state.langs.length) return;
  [state.source, state.target] = [state.target, state.source];
  renderPickerValue("source"); renderPickerValue("target");
  saveLangs();
  // Carry the current translation over as the new input.
  if (!els.out.classList.contains("hidden") && els.out.textContent) {
    els.src.value = els.out.textContent.slice(0, MAX_CHARS);
    onInput();
    resetOutput();
  }
  if (!reducedMotion()) {
    els.swap.classList.toggle("spun");
    for (const [pane, cls] of [[els.paneIn, "slide-l"], [els.paneOut, "slide-r"]]) {
      pane.classList.remove("slide-l", "slide-r"); void pane.offsetWidth; pane.classList.add(cls);
    }
  }
  announce(`Now translating ${nameOf(state.source)} to ${nameOf(state.target)}`);
}
els.swap.addEventListener("click", swapLanguages);

// ---------- input ----------
function autoGrow() {
  els.src.style.height = "auto";
  els.src.style.height = Math.max(els.src.scrollHeight, 0) + "px";
}
function onInput() {
  const n = els.src.value.length;
  els.count.textContent = n;
  els.countWrap.classList.toggle("near", n >= MAX_CHARS * 0.9 && n < MAX_CHARS);
  els.countWrap.classList.toggle("full", n >= MAX_CHARS);
  els.clear.classList.toggle("hidden", n === 0);
  els.src.lang = state.source;
  autoGrow();
  if (n >= MAX_CHARS && !onInput.warned) { onInput.warned = true; announce(`Character limit of ${MAX_CHARS} reached`); }
  if (n < MAX_CHARS) onInput.warned = false;
}
els.src.addEventListener("input", onInput);
els.clear.addEventListener("click", () => { els.src.value = ""; onInput(); resetOutput(); els.src.focus(); });

// ---------- output states ----------
function show(el, on) { el.classList.toggle("hidden", !on); }

function resetOutput() {
  show(els.out, false); show(els.skeleton, false); show(els.slow, false); show(els.err, false);
  show(els.badge, false); show(els.pivotNote, false); show(els.latency, false);
  show(els.placeholder, true);
  els.copy.disabled = true;
  els.paneOut.setAttribute("aria-busy", "false");
}

function showError(msg, retry) {
  show(els.skeleton, false); show(els.slow, false); show(els.out, false); show(els.placeholder, !retry);
  els.errText.textContent = msg;
  show(els.retry, !!retry);
  show(els.err, true);
  els.paneOut.setAttribute("aria-busy", "false");
}

function showResult(body, ms, fromHistory) {
  show(els.skeleton, false); show(els.slow, false); show(els.err, false); show(els.placeholder, false);
  els.out.lang = body.target_lang;
  els.out.textContent = body.translation;
  show(els.out, true);
  els.out.classList.remove("reveal"); void els.out.offsetWidth;
  if (!reducedMotion()) els.out.classList.add("reveal");

  els.badge.textContent = body.pivoted ? "Pivoted via English" : "Direct";
  els.badge.className = "route-badge" + (body.pivoted ? " pivot" : "");
  els.badge.title = body.pivoted ? "Translated source → English → target" : "Translated directly, no English pivot";
  show(els.badge, true);
  if (!reducedMotion()) { void els.badge.offsetWidth; els.badge.classList.add("pop"); }

  if (body.pivoted && body.pivot_intermediate != null) {
    els.pivotText.textContent = body.pivot_intermediate;
    show(els.pivotNote, true);
  } else show(els.pivotNote, false);

  if (ms != null) { els.latency.textContent = `${ms < 1000 ? Math.round(ms) + " ms" : (ms / 1000).toFixed(1) + " s"}`; show(els.latency, true); }
  else show(els.latency, false);

  els.copy.disabled = false;
  els.paneOut.setAttribute("aria-busy", "false");
}

// ---------- translate ----------
async function doTranslate(retryReq) {
  if (state.busy) return;
  const req = retryReq || { text: els.src.value.trim(), source_lang: state.source, target_lang: state.target };

  if (!req.text) { showError("Enter some text to translate."); els.src.focus(); return; }
  if (req.source_lang === req.target_lang) { showError("Source and target language must differ. Pick a different target."); return; }

  state.busy = true; state.lastReq = req;
  els.translate.disabled = true; els.translate.classList.add("loading");
  els.paneOut.setAttribute("aria-busy", "true");
  show(els.err, false); show(els.pivotNote, false); show(els.badge, false); show(els.latency, false); show(els.out, false);
  els.copy.disabled = true;

  // Skeleton only if the wait is long enough to matter (avoids flashes).
  show(els.placeholder, false);
  const skelTimer = setTimeout(() => show(els.skeleton, true), 350);
  const slowTimer = setTimeout(() => { show(els.slow, true); }, 10000);
  const t0 = performance.now();

  try {
    const res = await fetch("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    let body = null;
    try { body = await res.json(); } catch (e) { /* non-JSON error page */ }
    if (!res.ok) {
      const d = body && body.detail;
      const msg = typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg).join("; ") : `Translation failed (HTTP ${res.status}).`;
      showError(msg, res.status >= 500);
      if (res.status >= 500) toast(msg, "error", true);
      announce(msg, true);
      return;
    }
    const ms = performance.now() - t0;
    showResult(body, ms);
    addHistory({ text: req.text, translation: body.translation, s: body.source_lang, t: body.target_lang, pivoted: !!body.pivoted, time: Date.now() });
    announce(`Translated to ${nameOf(body.target_lang)}, ${body.pivoted ? "pivoted via English" : "direct"}, ${Math.round(ms)} milliseconds.`);
  } catch (err) {
    showError("Could not reach the translation API. Check your connection and retry.", true);
    toast("Could not reach the translation API.", "error", true);
    announce("Could not reach the translation API.", true);
  } finally {
    clearTimeout(skelTimer); clearTimeout(slowTimer);
    show(els.skeleton, false); show(els.slow, false);
    state.busy = false;
    els.translate.disabled = false; els.translate.classList.remove("loading");
  }
}
els.translate.addEventListener("click", () => doTranslate());
els.retry.addEventListener("click", () => doTranslate(state.lastReq));
els.src.addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); doTranslate(); } });

// ---------- copy ----------
async function copyOutput() {
  const text = els.out.textContent;
  if (!text || els.copy.disabled) return;
  let ok = false;
  try { await navigator.clipboard.writeText(text); ok = true; } catch (e) { /* fall through */ }
  if (!ok) {
    const ta = document.createElement("textarea");
    ta.value = text; ta.setAttribute("readonly", ""); ta.style.cssText = "position:fixed;top:0;left:0;opacity:0";
    document.body.appendChild(ta); ta.select();
    try { ok = document.execCommand("copy"); } catch (e) {}
    ta.remove();
  }
  if (!ok) { toast("Could not copy. Select the text and copy manually.", "error"); return; }
  els.copy.classList.add("copied"); els.copyLabel.textContent = "Copied";
  announce("Copied to clipboard");
  clearTimeout(copyOutput.t);
  copyOutput.t = setTimeout(() => { els.copy.classList.remove("copied"); els.copyLabel.textContent = "Copy"; }, 1600);
}
els.copy.addEventListener("click", copyOutput);

// ---------- examples ----------
function renderExamples() {
  els.chips.textContent = "";
  for (const ex of EXAMPLES) {
    if (!state.byCode[ex.lang]) continue;
    const b = document.createElement("button");
    b.type = "button"; b.className = "chip gbtn"; b.lang = ex.lang; b.textContent = ex.text;
    b.addEventListener("click", () => {
      state.source = ex.lang;
      if (state.target === ex.lang) state.target = ex.lang === "en" ? (state.langs.find((l) => l.code === "hi") || state.langs.find((l) => l.code !== "en")).code : "en";
      renderPickerValue("source"); renderPickerValue("target"); saveLangs();
      els.src.value = ex.text; onInput(); els.src.focus();
    });
    els.chips.append(b);
  }
}

// ---------- history (localStorage, try/catch everywhere) ----------
function loadHistory() {
  try { const h = JSON.parse(store(HISTORY_KEY) || "[]"); return Array.isArray(h) ? h : []; } catch (e) { return []; }
}
let historyItems = loadHistory();
function addHistory(item) {
  historyItems = [item, ...historyItems.filter((h) => !(h.text === item.text && h.s === item.s && h.t === item.t))].slice(0, HISTORY_MAX);
  store(HISTORY_KEY, JSON.stringify(historyItems));
  renderHistory();
}
function relTime(ts) {
  const s = Math.max(0, (Date.now() - ts) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.floor(s / 86400)} d ago`;
}
function renderHistory() {
  els.histList.textContent = "";
  const shown = historyItems.slice(0, HISTORY_SHOWN);
  show(els.histEmpty, shown.length === 0);
  show(els.histClear, shown.length > 0);
  shown.forEach((h, i) => {
    const li = document.createElement("li");
    li.className = "history-item"; li.style.setProperty("--i", i);
    const main = document.createElement("button");
    main.type = "button"; main.className = "h-main";
    main.setAttribute("aria-label", `Restore: ${nameOf(h.s)} to ${nameOf(h.t)}, ${h.text.slice(0, 60)}`);
    const pair = document.createElement("span"); pair.className = "h-pair";
    pair.textContent = `${nameOf(h.s)} → ${nameOf(h.t)}${h.pivoted ? " · via English" : ""} · ${relTime(h.time)}`;
    const a = document.createElement("span"); a.className = "h-src"; a.lang = h.s; a.textContent = h.text;
    const b = document.createElement("span"); b.className = "h-out"; b.lang = h.t; b.textContent = h.translation;
    main.append(pair, a, b);
    main.addEventListener("click", () => {
      if (!state.byCode[h.s] || !state.byCode[h.t]) return;
      state.source = h.s; state.target = h.t;
      renderPickerValue("source"); renderPickerValue("target"); saveLangs();
      els.src.value = h.text; onInput();
      showResult({ translation: h.translation, target_lang: h.t, pivoted: h.pivoted, pivot_intermediate: null }, null, true);
      show(els.pivotNote, false);
      announce("Restored from history");
      window.scrollTo({ top: 0, behavior: reducedMotion() ? "auto" : "smooth" });
    });
    const del = document.createElement("button");
    del.type = "button"; del.className = "ghost-btn gbtn"; del.textContent = "Remove"; del.setAttribute("aria-label", "Remove this entry");
    del.addEventListener("click", () => {
      historyItems = historyItems.filter((x) => x !== h && x.time !== h.time);
      store(HISTORY_KEY, JSON.stringify(historyItems)); renderHistory();
    });
    li.append(main, del);
    els.histList.append(li);
  });
}
els.histClear.addEventListener("click", () => {
  if (!confirm("Clear all recent translations from this browser?")) return;
  historyItems = []; store(HISTORY_KEY, null); renderHistory();
});

// ---------- help sheet & global shortcuts ----------
const helpBtn = $("help-btn");
let helpReturn = null;
function toggleHelp(open) {
  if (open) { helpReturn = document.activeElement; els.helpSheet.classList.remove("hidden"); $("help-close").focus(); }
  else { els.helpSheet.classList.add("hidden"); if (helpReturn) helpReturn.focus(); }
}
helpBtn.addEventListener("click", () => toggleHelp(true));
$("help-close").addEventListener("click", () => toggleHelp(false));
els.helpSheet.addEventListener("keydown", (e) => { if (e.key === "Tab") { e.preventDefault(); $("help-close").focus(); } });

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (!els.helpSheet.classList.contains("hidden")) { toggleHelp(false); return; }
    return;
  }
  const mod = e.ctrlKey || e.metaKey;
  if (mod && e.shiftKey && e.key.toLowerCase() === "s") { e.preventDefault(); swapLanguages(); return; }
  if (mod && e.shiftKey && e.key.toLowerCase() === "c") { e.preventDefault(); copyOutput(); return; }
  if (mod && e.key === "Enter") { e.preventDefault(); doTranslate(); return; }
  const tag = (e.target.tagName || "").toLowerCase();
  const typing = tag === "input" || tag === "textarea" || tag === "select" || e.target.isContentEditable;
  if (typing || mod || e.altKey) return;
  if (e.key === "?") { e.preventDefault(); toggleHelp(true); }
  else if (e.key === "1") switchTab("translate");
  else if (e.key === "2") switchTab("benchmarks");
});

// ---------- Benchmarks tab: static, pre-computed, "quick capability checking" only ----------
// One tiny shared store drives every renderer: hover (cross-highlight), pinned (Set of model
// names), hidden (Set of baseline names whose table markers are switched off).
const BENCH_KEY = "indicmt.bench";
const bench = {
  loaded: false, data: null, base: null, set: "in22-gen", dir: "all", q: "", sort: "dir", asc: true,
  hover: null, pinned: new Set(), hidden: new Set(),
  rankSort: "score", hidePartial: false, dense: false,
  models: new Map(),   // name -> {name, ours, partial, params, en_xx, xx_en}
  markIdx: new Map(),  // baseline name -> frozen marker index (colour + shape), built once at load
  revealed: {},
};

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("no " + url + " bundled");
  return res.json();
}

// ---------- small DOM + formatting helpers for the charts ----------
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}
function ordinal(n) {
  const t = n % 100, s = ["th", "st", "nd", "rd"];
  return n + (s[(t - 20) % 10] || s[t] || s[0]);
}
const slug = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
const oursMacro = (key) => bench.data?.["in22-gen"]?.[key]?.["chrf++"];
const oursBillions = () => bench.base?.ours_params?.total_b;
const setName = () => (bench.set === "in22-gen" ? "IN22-Gen" : "IN22-Conv");
const clamp01 = (v) => Math.max(0, Math.min(100, Number(v) || 0));

function saveBenchPrefs() {
  store(BENCH_KEY, JSON.stringify({
    pinned: [...bench.pinned], hidden: [...bench.hidden],
    rankSort: bench.rankSort, hidePartial: bench.hidePartial, dense: bench.dense,
  }));
}
function loadBenchPrefs() {
  try {
    const p = JSON.parse(store(BENCH_KEY) || "null");
    if (!p || typeof p !== "object") return;
    if (Array.isArray(p.pinned)) p.pinned.forEach((n) => { if (bench.models.has(n)) bench.pinned.add(n); });
    if (Array.isArray(p.hidden)) p.hidden.forEach((n) => { if (bench.markIdx.has(n)) bench.hidden.add(n); });
    if (["score", "size", "name"].includes(p.rankSort)) bench.rankSort = p.rankSort;
    bench.hidePartial = p.hidePartial === true;
    bench.dense = p.dense === true;
  } catch (e) { /* prefs are optional */ }
}

// Keep the old render visible (dimmed) while the new one is swapped in; never blank it.
function withUpdating(node, fn) {
  if (!node) { fn(); return; }
  node.classList.add("is-updating");
  try { fn(); } finally {
    requestAnimationFrame(() => requestAnimationFrame(() => node.classList.remove("is-updating")));
  }
}

// Model registry + frozen marker indices (fixes repaint-on-filter: keyed by NAME, built once).
function buildModels() {
  bench.models.clear(); bench.markIdx.clear();
  const base = bench.base;
  if (!base) return;
  Object.keys(base.per_language || {}).forEach((n, i) => bench.markIdx.set(n, i));
  for (const m of base.macro || []) {
    bench.models.set(m.model, { name: m.model, ours: false, partial: !!m.partial, params: m.params || null, en_xx: m.en_xx, xx_en: m.xx_en });
  }
  const a = oursMacro("MACRO_en->xx"), b = oursMacro("MACRO_xx->en"), mine = oursBillions();
  if (a != null || b != null) {
    bench.models.set("Ours", {
      name: "Ours", ours: true, partial: false, en_xx: a, xx_en: b,
      params: mine != null ? { total_b: mine, active_b: null, kind: "measured", note: base.ours_params.note } : null,
    });
  }
}

async function loadBenchmarks() {
  bench.loaded = true;
  const banner = $("benchmarks-banner");
  banner.textContent = "Baselines are taken from a published model card and were not re-run. Protocols differ, so comparisons are indicative.";
  banner.classList.remove("hidden");
  const [ours, base] = await Promise.allSettled([fetchJson("benchmarks.json"), fetchJson("baselines.json")]);
  bench.base = base.status === "fulfilled" ? base.value : null;
  if (ours.status === "fulfilled") bench.data = ours.value;
  else { banner.textContent = "No IndicMT score file (benchmarks.json) was added to this build; only the published baselines are shown."; bench.loaded = false; }
  try {
    buildModels();
    loadBenchPrefs();
    syncBenchControls();
    renderBenchLegend();
    if (bench.data) renderBenchmarks(true);
    renderRankCharts();
    renderSizeChart();
    applyEmphasis();
  } catch (err) {
    banner.textContent = "Could not draw the benchmark charts.";
  }
}

// ---------- shared emphasis: pin / hover / cross-highlight ----------
function setHover(name) {
  if (bench.hover === name) return;
  bench.hover = name;
  applyEmphasis();
}
function applyEmphasis() {
  const root = els.panels.benchmarks;
  if (!root) return;
  root.classList.toggle("has-pins", bench.pinned.size > 0);
  root.querySelectorAll(".rk-row[data-model]").forEach((r) => {
    const m = r.dataset.model, p = bench.pinned.has(m);
    r.classList.toggle("is-pinned", p);
    r.classList.toggle("is-hover", bench.hover === m);
    r.setAttribute("aria-pressed", String(p));
  });
  root.querySelectorAll(".bm[data-model]").forEach((mk) => {
    const m = mk.dataset.model;
    mk.classList.toggle("is-pinned", bench.pinned.has(m));
    mk.classList.toggle("is-hover", bench.hover === m);
  });
  renderPinChip();
}
function renderPinChip() {
  const chip = $("pin-chip"), txt = $("pin-text");
  if (!chip || !txt) return;
  const names = [...bench.pinned];
  chip.classList.toggle("hidden", names.length === 0);
  txt.textContent = "Pinned: " + (names.length > 2 ? `${names.slice(0, 2).join(", ")} +${names.length - 2}` : names.join(", "));
}
function togglePin(name) {
  if (!name) return;
  const was = bench.pinned.has(name);
  if (was) bench.pinned.delete(name); else bench.pinned.add(name);
  saveBenchPrefs();
  applyEmphasis();
  const n = bench.pinned.size;
  announce(`${name} ${was ? "unpinned" : "pinned"}; ${n === 0 ? "no models pinned" : plural(n, "model") + " pinned"}.`);
}
function clearPins() {
  if (!bench.pinned.size) return false;
  bench.pinned.clear();
  saveBenchPrefs();
  applyEmphasis();
  announce("All pins cleared.");
  return true;
}

// ---------- tooltip (one element on <body>, textContent only) ----------
const tipSpecs = new WeakMap();
const tip = { el: null, target: null, pt: null };
function ensureTip() {
  if (tip.el && tip.el.isConnected) return tip.el;
  const d = el("div", "bench-tip");
  d.id = "bench-tip"; d.setAttribute("role", "tooltip");
  document.body.appendChild(d);
  tip.el = d;
  return d;
}
function tipLine(parent, cls, text, key) {
  const d = el("div", cls);
  if (key) { const k = el("i", "tip-key"); k.style.background = key; d.appendChild(k); }
  d.appendChild(document.createTextNode(text));
  parent.appendChild(d);
}
function showTip(target) {
  const fn = tipSpecs.get(target);
  if (!fn) return;
  let spec;
  try { spec = fn(); } catch (e) { return; }
  if (!spec) return;
  hideTip(true);
  const t = ensureTip();
  t.textContent = "";
  const v = el("div", "tip-v", spec.value); t.appendChild(v);
  tipLine(t, "tip-n", spec.name, spec.key);
  (spec.lines || []).forEach((l) => tipLine(t, "tip-l", typeof l === "string" ? l : l.text, typeof l === "string" ? null : l.key));
  // Swap the native title for the custom tooltip so the two never stack.
  if (target.hasAttribute("title")) { target.dataset.title = target.title; target.removeAttribute("title"); }
  tip.target = target;
  t.classList.add("on");
  placeTip(target);
}
function placeTip(target) {
  const t = tip.el;
  if (!t) return;
  t.style.left = "0px"; t.style.top = "0px";
  const r = target.getBoundingClientRect();
  const w = t.offsetWidth, h = t.offsetHeight;
  const vw = window.innerWidth, vh = window.innerHeight;
  const cx = tip.pt && tip.pt.x != null ? tip.pt.x : r.left + Math.min(r.width / 2, 160);
  let x = cx - w / 2;
  if (r.right > vw - 8 && x + w > vw - 8) x = vw - 8 - w;   // right-align when the row runs off screen
  x = Math.max(8, Math.min(x, vw - w - 8));
  let y = r.bottom + 8;
  if (y + h > vh - 8) y = r.top - h - 8;                      // flip above
  if (y < 8) y = Math.max(8, Math.min(vh - h - 8, r.bottom + 8));
  t.style.left = Math.round(x) + "px"; t.style.top = Math.round(y) + "px";
}
function hideTip(keepTitle) {
  if (tip.el) tip.el.classList.remove("on");
  const tg = tip.target;
  if (tg && tg.dataset && tg.dataset.title != null) { tg.title = tg.dataset.title; delete tg.dataset.title; }
  tip.target = null;
}

// ---------- ranked score charts ----------
function chartRow(i, name, ours, label, vt) {
  const li = el("li", "rk-item");
  if (vt) li.style.viewTransitionName = vt;
  li.dataset.flip = vt || name;
  const row = el("div", "rk-row" + (ours ? " is-ours" : ""));
  row.setAttribute("role", "button");
  row.tabIndex = i === 0 ? 0 : -1;
  row.setAttribute("aria-pressed", String(bench.pinned.has(name)));
  row.setAttribute("aria-label", label);
  row.dataset.model = name;
  row.style.setProperty("--i", Math.min(i, 24));
  const nm = el("span", "rk-name"); nm.title = name;
  nm.appendChild(el("span", "nm", name));
  const track = el("span", "rk-track");
  row.append(nm, track);
  li.appendChild(row);
  return { li, row, nm, track };
}
function fillEl(v, partial) {
  const f = el("i", "rk-fill" + (partial ? " partial" : ""));
  f.setAttribute("aria-hidden", "true");
  f.style.setProperty("--v", clamp01(v).toFixed(2));
  return f;
}
function gridLines(track, xs) {
  xs.forEach((x) => { const g = el("i", "rk-grid"); g.style.setProperty("--x", x.toFixed(2)); g.setAttribute("aria-hidden", "true"); track.appendChild(g); });
}
function axisRow(label, ticks, third) {
  const axis = el("div", "rk-axis");
  axis.setAttribute("aria-hidden", "true");
  const tk = el("span", "ticks");
  ticks.forEach(([text, x]) => { const t = el("span", x >= 99 ? "end" : null, text); t.style.setProperty("--x", x.toFixed(2)); tk.appendChild(t); });
  axis.append(el("span", null, label), tk);
  if (third) axis.appendChild(el("span"));
  return axis;
}
function paramsLine(m) {
  if (!m.params) return null;
  const p = m.params;
  return p.total_b == null ? `Parameters: not disclosed (${p.note})` : `Parameters: ${sizeText(p)} (${p.kind}: ${p.note})`;
}

const RANK_PANELS = [
  ["out", "rank-out", "en_xx", "English to Indic", "panel-rk-out", "en->xx", "English → Indic"],
  ["in", "rank-in", "xx_en", "Indic to English", "panel-rk-in", "xx->en", "Indic → English"],
];

function rankData(key) {
  const rows = [];
  for (const m of bench.models.values()) {
    const v = m[key];
    if (v == null || !isFinite(v)) continue;
    if (bench.hidePartial && m.partial) continue;
    rows.push({ m, v });
  }
  const byScore = [...rows].sort((a, b) => b.v - a.v || (a.m.ours ? 1 : 0) - (b.m.ours ? 1 : 0));
  const rank = new Map(byScore.map((r, i) => [r.m.name, i + 1]));
  let order = byScore;
  if (bench.rankSort === "size") {
    order = [...rows].sort((a, b) => {
      const x = a.m.params?.total_b, y = b.m.params?.total_b;
      if (x == null && y == null) return b.v - a.v;
      if (x == null) return 1;
      if (y == null) return -1;
      return x - y;
    });
  } else if (bench.rankSort === "name") {
    order = [...rows].sort((a, b) => a.m.name.localeCompare(b.m.name));
  }
  return { rows, order, rank };
}

function renderRankCharts() {
  const base = bench.base;
  const ok = base && bench.data;
  const pair = document.querySelector(".bench-pair");
  const nMatch = /([\d,]+)\s+sentences per direction/.exec((base && base.protocol) || "");
  const meta = $("meta-rank");
  if (meta) meta.textContent = (nMatch ? `n = ${nMatch[1]} · ` : "") + "chrF++";
  if (pair) pair.classList.toggle("one", bench.dir !== "all");
  const sortDesc = bench.rankSort === "score" ? "highest first" : bench.rankSort === "size" ? "smallest model first" : "alphabetical";

  for (const [dk, hostId, key, label, panelId, bucket, title] of RANK_PANELS) {
    const host = $(hostId), panel = $(panelId);
    if (!host) continue;
    if (panel) panel.classList.toggle("hidden", bench.dir !== "all" && bench.dir !== bucket);
    host.textContent = "";
    const mine = bench.models.get("Ours");
    if (!ok || !mine || mine[key] == null) { host.appendChild(el("p", "bench-msg", "Baseline data is not available in this build.")); continue; }
    const { rows, order, rank } = rankData(key);
    const n = rows.length;

    // +Δ against the nearest baseline still shown (computed at runtime).
    let delta = null;
    const others = rows.filter((r) => !r.m.ours);
    if (others.length) {
      const me = mine[key];
      const near = others.reduce((b, r) => (Math.abs(r.v - me) < Math.abs(b.v - me) ? r : b), others[0]);
      const d = me - near.v;
      delta = { text: `${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(1)} vs ${near.m.name}` };
    }

    const chart = el("div", "rk-chart");
    const ticks = [0, 25, 50, 75, 100];
    chart.appendChild(axisRow("chrF++", ticks.map((t) => [String(t), t])));
    const list = el("ol", "rk");
    list.setAttribute("aria-label", `${label}, macro chrF++ on IN22-Gen, ${sortDesc}`);
    order.forEach((r, i) => {
      const m = r.m, rk = rank.get(m.name);
      const lbl = `${m.name}, ${r.v.toFixed(1)} chrF++, rank ${rk} of ${n}${m.partial ? ", partial language coverage" : ""}`;
      const { li, row, nm, track } = chartRow(i, m.name, m.ours, lbl, `rk-${dk}-${slug(m.name)}`);
      if (m.partial) nm.firstChild.textContent += " *";
      if (m.ours) {
        nm.append(el("span", "rk-chip", "Ours"), el("span", "rk-chip rank", `${ordinal(rk)} of ${n}`));
        if (delta) nm.appendChild(el("span", "rk-chip delta", delta.text));
      }
      gridLines(track, ticks.slice(1, -1));
      const val = el("span", "rk-val", r.v.toFixed(1));
      val.style.setProperty("--v", r.v);
      track.append(fillEl(r.v, m.partial), val);
      tipSpecs.set(row, () => {
        const lines = [`Rank ${rk} of ${n} · ${title}`];
        const pl = paramsLine(m); if (pl) lines.push(pl);
        if (m.partial) lines.push("Partial language coverage: not directly comparable (see note below the chart).");
        if (m.ours && delta) lines.push(delta.text);
        return { value: `${r.v.toFixed(1)} chrF++`, name: m.name, key: m.ours ? "var(--bench-fill-1)" : "var(--base-fill)", lines };
      });
      list.appendChild(li);
    });
    chart.appendChild(list);
    host.appendChild(chart);
    if (!bench.revealed[hostId]) { bench.revealed[hostId] = true; revealOnScroll(list); }
  }

  const notes = $("baseline-notes"); notes.textContent = "";
  if (!ok) { applyEmphasis(); return; }
  if (base.macro.some((m) => m.partial)) notes.appendChild(el("p", "bench-foot", "* " + base.partial_note));
  const p = el("p", "bench-foot", `Protocol: ${base.protocol}. Source: `);
  const a = el("a", null, base.source.name); a.href = base.source.url; a.rel = "noopener noreferrer"; a.target = "_blank";
  p.append(a, ` (fetched ${base.source.fetched}).`);
  notes.append(p, el("p", "bench-foot",
    "Baselines are the card's sentence-level chrF++ with greedy decoding, as published. Ours is corpus-level chrF++ with beam 5, so the comparison is indicative, not exact."));
  applyEmphasis();
}

// Bars start hidden and grow in (staggered) the first time the chart scrolls into view.
function revealOnScroll(list) {
  if (reducedMotion() || !("IntersectionObserver" in window)) return;
  list.classList.add("rk-wait");
  const io = new IntersectionObserver((entries) => {
    if (!entries.some((e) => e.isIntersecting)) return;
    list.classList.replace("rk-wait", "rk-go");
    io.disconnect();
  }, { threshold: 0.15 });
  io.observe(list);
}

// Reorder ranked rows: View Transitions where supported, else manual FLIP, else instant.
function reorderRankCharts() {
  const doRender = () => renderRankCharts();
  const rootEl = els.panels.benchmarks;
  if (reducedMotion() || !rootEl || rootEl.hidden) { doRender(); return; }
  if (typeof document.startViewTransition === "function") {
    try { document.startViewTransition(doRender); return; } catch (e) { /* fall through to FLIP */ }
  }
  const before = new Map();
  document.querySelectorAll(".rk-item[data-flip]").forEach((li) => before.set(li.dataset.flip, li.getBoundingClientRect()));
  doRender();
  document.querySelectorAll(".rk-item[data-flip]").forEach((li) => {
    const b = before.get(li.dataset.flip);
    if (!b) return;
    const dy = b.top - li.getBoundingClientRect().top;
    if (!dy) return;
    li.style.transition = "none";
    li.style.transform = `translateY(${dy}px)`;
    void li.offsetWidth;
    li.style.transition = "transform 320ms var(--ease-standard)";
    li.style.transform = "";
    li.addEventListener("transitionend", () => { li.style.transition = ""; }, { once: true });
  });
}

// ---------- model size (log scale, smallest first) ----------
const SIZE_MIN_B = 0.1, SIZE_MAX_B = 50;
const sizePos = (b) => 100 * Math.log10(b / SIZE_MIN_B) / Math.log10(SIZE_MAX_B / SIZE_MIN_B);
function fmtParams(b) {
  return b < 1 ? `${+(b * 1000).toPrecision(4)}M` : `${+b.toPrecision(3)}B`;
}
function sizeText(p) {
  return (p.kind === "nominal" || p.kind === "rounded" ? "≈" : "") + fmtParams(p.total_b) + (p.kind === "variant" ? " variant" : "");
}

function renderSizeChart() {
  const host = $("size-chart"), notes = $("size-notes");
  host.textContent = ""; notes.textContent = "";
  const base = bench.base, mine = oursBillions();
  const meta = $("meta-size"); if (meta) meta.textContent = "total parameters";
  if (!base || mine == null) { host.appendChild(el("p", "bench-msg", "Model size data is not available in this build.")); return; }
  // Ascending by total parameters: ours (smallest) first; undisclosed sizes last, with no bar.
  const rows = [...bench.models.values()].filter((m) => m.params).sort((a, b) => {
    const x = a.params.total_b, y = b.params.total_b;
    if (x == null && y == null) return a.name.localeCompare(b.name);
    if (x == null) return 1;
    if (y == null) return -1;
    return x - y;
  });
  const disclosed = rows.filter((m) => m.params.total_b != null).length;

  const chart = el("div", "rk-chart size");
  chart.appendChild(el("p", "sr-only", `Bar length is total parameters on a logarithmic scale, ${SIZE_MIN_B}B to ${SIZE_MAX_B}B; smaller is better. Models are listed smallest first.`));
  const cue = el("p", "rk-cue");
  cue.append(el("span", "arrow", "↑"), " Smaller is better · sorted smallest first");
  chart.appendChild(cue);
  const ticks = [0.1, 1, 10];
  const axis = axisRow("log scale", ticks.map((b) => [fmtParams(b), sizePos(b)]), true);
  const hint = el("div", "rk-axis-hint"); hint.setAttribute("aria-hidden", "true");
  hint.append(el("span"), el("span", null, "← smaller is better"), el("span"));
  chart.append(axis, hint);

  const list = el("ol", "rk");
  list.setAttribute("aria-label", "Total parameters per model on a logarithmic scale, smallest first (smaller is better)");
  rows.forEach((m, i) => {
    const p = m.params;
    const has = p.total_b != null;
    const lbl = has ? `${m.name}, ${sizeText(p)} parameters, size rank ${i + 1} of ${disclosed} (smallest first)` : `${m.name}, parameter count not disclosed`;
    const { li, row, nm, track } = chartRow(i, m.name, m.ours, lbl, null);
    const val = el("span", "rk-val col");
    row.title = p.note;
    if (m.ours) nm.appendChild(el("span", "rk-chip", "Ours"));
    if (p.kind === "variant") nm.appendChild(el("span", "rk-tag", "variant"));
    if (!has) {
      val.textContent = "not disclosed";
      val.classList.add("none");
    } else {
      gridLines(track, ticks.map(sizePos));
      track.appendChild(fillEl(sizePos(p.total_b), false));
      val.textContent = sizeText(p);
      if (p.active_b != null) {
        const mk = el("i", "rk-active"); mk.style.setProperty("--x", sizePos(p.active_b).toFixed(2));
        mk.title = `${fmtParams(p.active_b)} active per token`; mk.setAttribute("aria-hidden", "true");
        track.appendChild(mk);
        val.appendChild(el("small", null, `${fmtParams(p.active_b)} active`));
      }
    }
    val.appendChild(el("span", "sr-only", ` (${p.kind}: ${p.note})`));
    row.appendChild(val);
    tipSpecs.set(row, () => {
      const lines = [];
      lines.push(has ? `Size rank ${i + 1} of ${disclosed}, smallest first` : "Not disclosed: no bar drawn");
      lines.push(`Basis (${p.kind}): ${p.note}`);
      if (p.active_b != null) lines.push(`${fmtParams(p.active_b)} active per token`);
      const sc = [m.en_xx, m.xx_en].every((x) => x != null) ? `chrF++ ${m.en_xx.toFixed(1)} en→xx, ${m.xx_en.toFixed(1)} xx→en` : null;
      if (sc) lines.push(sc);
      return { value: has ? sizeText(p) : "not disclosed", name: m.name, key: m.ours ? "var(--bench-fill-1)" : "var(--base-fill)", lines };
    });
    list.appendChild(li);
  });
  chart.appendChild(list);
  host.appendChild(chart);
  if (!bench.revealed["size"]) { bench.revealed["size"] = true; revealOnScroll(list); }

  const rivals = Object.keys(base.per_language || {}).map((n) => base.macro.find((m) => m.model === n)).filter((m) => m && m.params && m.params.total_b)
    .sort((a, b) => a.params.total_b - b.params.total_b);
  if (rivals.length) {
    const ratio = (r) => (r >= 10 ? Math.round(r) : r.toFixed(1)) + "x";
    notes.appendChild(el("p", "bench-foot", "Ours is " + rivals.map((m) => `${ratio(m.params.total_b / mine)} smaller than ${m.model} (${sizeText(m.params)})`).join(", ") + "."));
  }
  notes.appendChild(el("p", "bench-foot", "Parameter counts are gathered from each model's public page or report, not measured here (our count is measured). ≈ marks a rounded or nominal size; the diamond marks active parameters of a mixture-of-experts model. Hover a row for its note."));
  applyEmphasis();
}

// ---------- language table ----------
function langLabel(code) { return state.byCode[code] ? `${code} (${state.byCode[code].name})` : code; }
const otherLang = (dir) => dir.split("-").find((c) => c !== "en") || dir;

function meterEl(v) {
  const m = document.createElement("span"); m.className = "meter"; m.setAttribute("aria-hidden", "true");
  const f = document.createElement("i"); f.className = "meter-fill";
  f.style.setProperty("--v", clamp01(v).toFixed(2));
  m.appendChild(f);
  return m;
}
// Baseline reference markers: one distinct shape + colour per model, frozen in bench.markIdx.
const MARKER_PATHS = ["M6 1.5a4.5 4.5 0 1 0 0 9a4.5 4.5 0 1 0 0-9z", "M6 .8L11.2 6L6 11.2L.8 6z", "M6 1L11.2 10.8H.8z"];
function markerEl(idx, name) {
  const i = idx % MARKER_PATHS.length;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 12 12"); svg.setAttribute("width", "12"); svg.setAttribute("height", "12");
  svg.setAttribute("class", "bm bm-" + i);
  svg.setAttribute("aria-hidden", "true");
  if (name) svg.dataset.model = name;
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", MARKER_PATHS[i]);
  svg.appendChild(path);
  return svg;
}
function baselineMarks(lang, bucket) {
  const per = bench.base?.per_language;
  if (!per || bench.set !== "in22-gen") return [];
  const out = [];
  for (const [name, idx] of bench.markIdx) {
    if (bench.hidden.has(name)) continue;
    const v = per[name]?.[lang]?.[bucket === "xx->en" ? 1 : 0];
    if (v != null) out.push({ name, idx, v });
  }
  return out;
}
function renderBenchLegend() {
  const legend = $("bench-legend");
  if (!legend) return;
  legend.textContent = "";
  const names = [...bench.markIdx.keys()];
  if (!names.length) return;
  legend.appendChild(el("span", "leg-title", "Published baseline chrF++ on the same scale:"));
  names.forEach((n) => {
    const b = el("button", "it leg-btn gbtn"); b.type = "button";
    b.setAttribute("aria-pressed", String(!bench.hidden.has(n)));
    b.title = `Show or hide ${n} markers in the table`;
    b.dataset.model = n;
    b.append(markerEl(bench.markIdx.get(n)), el("span", null, n));
    b.addEventListener("click", () => {
      const off = !bench.hidden.has(n);
      if (off) bench.hidden.add(n); else bench.hidden.delete(n);
      b.setAttribute("aria-pressed", String(!off));
      saveBenchPrefs();
      renderBenchmarks();
      announce(`${n} markers ${off ? "hidden" : "shown"}.`);
    });
    legend.appendChild(b);
  });
}
function syncLegend() {
  const legend = $("bench-legend");
  const show = bench.set === "in22-gen" && bench.markIdx.size > 0;
  if (legend) legend.classList.toggle("hidden", !show);
  const note = $("bench-conv-note"); if (note) note.classList.toggle("hidden", bench.set !== "in22-conv");
}

function scoreCell(v, cls, marks = []) {
  const td = document.createElement("td"); td.className = "num score " + cls;
  const n = document.createElement("span"); n.className = "score-num"; n.textContent = v.toFixed(2);
  const meter = meterEl(v);
  for (const m of marks) {
    const mk = markerEl(m.idx, m.name);
    mk.style.setProperty("--v", clamp01(m.v).toFixed(2));
    meter.appendChild(mk);
  }
  td.append(meter, n);
  if (marks.length) td.appendChild(el("span", "sr-only", " Baselines: " + marks.map((m) => `${m.name} ${m.v}`).join(", ")));
  return td;
}

function renderBenchmarks(animate = false) {
  withUpdating(document.querySelector(".table-wrap"), () => renderBenchmarksNow(animate));
}

function renderBenchmarksNow(animate) {
  const set = (bench.data && bench.data[bench.set]) || {};
  const tbody = document.querySelector("#benchmarks-table tbody");
  if (!tbody) return;
  $("bench-table-title").textContent = `Language-wise performance — ${setName()}`;
  syncLegend();

  const q = bench.q.trim().toLowerCase();
  const all = Object.entries(set).filter(([k]) => !k.startsWith("MACRO"));
  const rows = all.filter(([k, s]) => {
    if (bench.dir !== "all" && s.bucket !== bench.dir) return false;
    if (!q) return true;
    const other = otherLang(k);
    return k.includes(q) || (state.byCode[other]?.name || "").toLowerCase().includes(q);
  });
  const meta = $("meta-table");
  if (meta) meta.textContent = `${rows.length} of ${all.length} directions · chrF++ / BLEU`;

  // Macro tiles follow the same filter as the rows: direction filter hides a tile; a language
  // search switches the tile to the mean of the matching directions (labelled as filtered).
  const macro = $("benchmarks-macro");
  macro.textContent = "";
  for (const [bucket, key, lbl] of [["en->xx", "MACRO_en->xx", "en→xx"], ["xx->en", "MACRO_xx->en", "xx→en"]]) {
    if (bench.dir !== "all" && bench.dir !== bucket) continue;
    const inBucket = all.filter(([, s]) => s.bucket === bucket);
    let value, label, denom;
    if (!q) {
      if (!set[key]) continue;
      value = set[key]["chrf++"];
      label = `Macro ${lbl} chrF++`;
      denom = `${inBucket.length} directions · ${setName()}`;
    } else {
      const m = rows.filter(([, s]) => s.bucket === bucket);
      if (!m.length) continue;
      value = m.reduce((a, [, s]) => a + s["chrf++"], 0) / m.length;
      label = `Mean ${lbl} chrF++ (filtered)`;
      denom = `${m.length} of ${inBucket.length} directions · ${setName()}`;
    }
    const div = el("div", "macro-tile");
    div.append(el("div", "label", label), el("div", "value", value.toFixed(2)));
    const mt = meterEl(value); mt.classList.add("meter-sm"); div.appendChild(mt);
    div.appendChild(el("div", "denom", denom));
    macro.appendChild(div);
  }

  const val = ([k, s]) => bench.sort === "chrf" ? s["chrf++"] : bench.sort === "bleu" ? s.bleu : k;
  rows.sort((x, y) => {
    const a = val(x), b = val(y);
    const c = typeof a === "string" ? a.localeCompare(b) : a - b;
    return bench.asc ? c : -c;
  });
  tbody.textContent = "";
  const table = $("benchmarks-table");
  table.classList.toggle("bench-animate", !!animate);
  // "Best" is computed over the rows actually shown (empty result: no best).
  const maxC = rows.length ? Math.max(...rows.map(([, s]) => s["chrf++"])) : null;
  const maxB = rows.length ? Math.max(...rows.map(([, s]) => s.bleu)) : null;
  rows.forEach(([dir, s], i) => {
    const tr = document.createElement("tr");
    tr.style.setProperty("--i", Math.min(i, 24));
    const lang = otherLang(dir);
    const td0 = document.createElement("td");
    const chip = document.createElement("span");
    chip.className = "dir-chip " + (s.bucket === "xx->en" ? "dir-in" : "dir-out");
    chip.textContent = dir; chip.title = langLabel(lang);
    td0.appendChild(chip);
    const marks = baselineMarks(lang, s.bucket);
    const td1 = scoreCell(s["chrf++"], "chrf", marks);
    const td2 = scoreCell(s.bleu, "bleu");
    for (const [td, isBest] of [[td1, s["chrf++"] === maxC], [td2, s.bleu === maxB]]) {
      if (!isBest) continue;
      td.classList.add("is-best");
      td.insertBefore(el("span", "best-mark", "▲"), td.firstChild);
      td.appendChild(el("span", "sr-only", " (best in view)"));
    }
    tr.append(td0, td1, td2);
    tipSpecs.set(tr, () => {
      const lines = [`BLEU ${s.bleu.toFixed(2)}`];
      marks.forEach((m) => lines.push({ text: `${m.name} ${m.v.toFixed(1)} chrF++`, key: `var(--bm-${m.idx % 3})` }));
      if (!marks.length && bench.set === "in22-gen" && bench.markIdx.size && bench.hidden.size < bench.markIdx.size) lines.push("No per-language baseline values published for this direction.");
      return { value: `${s["chrf++"].toFixed(2)} chrF++`, name: `${dir} · ${langLabel(lang)}`, key: "var(--bench-fill-1)", lines };
    });
    tbody.append(tr);
  });
  $("bench-empty").classList.toggle("hidden", rows.length > 0);
  document.querySelectorAll("#benchmarks-table th").forEach((th) => {
    const b = th.querySelector("button");
    if (!b) return;
    th.setAttribute("aria-sort", b.dataset.sort === bench.sort ? (bench.asc ? "ascending" : "descending") : "none");
  });
  applyEmphasis();
}

// ---------- controls ----------
function syncRadios() {
  document.querySelectorAll('[role="radiogroup"]').forEach((g) => { if (typeof g._sync === "function") g._sync(); });
}
function syncBenchControls() {
  document.querySelectorAll("[data-rsort]").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.rsort === bench.rankSort)));
  document.querySelectorAll("[data-density]").forEach((b) => b.setAttribute("aria-checked", String((b.dataset.density === "compact") === bench.dense)));
  const hp = $("hide-partial"); if (hp) hp.setAttribute("aria-pressed", String(bench.hidePartial));
  const wrap = document.querySelector(".table-wrap"); if (wrap) wrap.classList.toggle("dense", bench.dense);
  syncRadios();
}

// Radiogroup keyboard model: arrows move selection + focus, roving tabindex.
function enhanceRadiogroup(group) {
  const btns = () => [...group.querySelectorAll('[role="radio"]')];
  group._sync = () => {
    const list = btns();
    const anyOn = list.some((b) => b.getAttribute("aria-checked") === "true");
    list.forEach((b, i) => { b.tabIndex = (anyOn ? b.getAttribute("aria-checked") === "true" : i === 0) ? 0 : -1; });
  };
  group._sync();
  group.addEventListener("keydown", (e) => {
    const list = btns(), n = list.length;
    const i = list.indexOf(document.activeElement);
    if (i < 0 || !n) return;
    let j = null;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") j = (i + 1) % n;
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") j = (i - 1 + n) % n;
    else if (e.key === "Home") j = 0; else if (e.key === "End") j = n - 1;
    if (j === null) return;
    e.preventDefault();
    list[j].click();
    list[j].focus();
  });
  group.addEventListener("click", () => group._sync());
}

function wireBenchControls() {
  document.querySelectorAll("[data-set]").forEach((b) => b.addEventListener("click", () => {
    bench.set = b.dataset.set;
    document.querySelectorAll("[data-set]").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    renderBenchmarks(true);
  }));
  document.querySelectorAll("[data-dir]").forEach((b) => b.addEventListener("click", () => {
    bench.dir = b.dataset.dir;
    document.querySelectorAll("[data-dir]").forEach((x) => x.setAttribute("aria-checked", String(x.dataset.dir === bench.dir)));
    syncRadios();
    reorderRankCharts();
    renderBenchmarks();
  }));
  document.querySelectorAll("[data-rsort]").forEach((b) => b.addEventListener("click", () => {
    if (bench.rankSort === b.dataset.rsort) return;
    bench.rankSort = b.dataset.rsort;
    saveBenchPrefs(); syncBenchControls();
    reorderRankCharts();
    announce(`Charts sorted by ${b.textContent.trim().toLowerCase()}.`);
  }));
  document.querySelectorAll("[data-density]").forEach((b) => b.addEventListener("click", () => {
    bench.dense = b.dataset.density === "compact";
    saveBenchPrefs(); syncBenchControls();
  }));
  const hp = $("hide-partial");
  if (hp) hp.addEventListener("click", () => {
    bench.hidePartial = !bench.hidePartial;
    saveBenchPrefs(); syncBenchControls();
    reorderRankCharts();
    announce(bench.hidePartial ? "Partial-coverage models hidden from the ranked charts." : "Partial-coverage models shown.");
  });
  const pc = $("pin-clear");
  if (pc) pc.addEventListener("click", () => clearPins());
  const search = $("bench-search");
  if (search) search.addEventListener("input", (e) => { bench.q = e.target.value; renderBenchmarks(); });
  document.querySelectorAll("#benchmarks-table th button").forEach((b) => b.addEventListener("click", () => {
    const k = b.dataset.sort;
    if (bench.sort === k) bench.asc = !bench.asc; else { bench.sort = k; bench.asc = k === "dir"; }
    renderBenchmarks();
    announce(`Table sorted by ${b.textContent.trim()}, ${bench.asc ? "ascending" : "descending"}.`);
  }));
}

// Pointer / focus / keyboard interactions for the charts and the table (delegated).
function wireBenchInteractions() {
  const root = els.panels.benchmarks;
  if (!root) return;
  const hit = (e) => (e.target && e.target.closest ? e.target.closest(".rk-row, #benchmarks-table tbody tr") : null);
  root.addEventListener("pointerover", (e) => {
    const t = hit(e); if (!t) return;
    if (t.classList.contains("rk-row")) setHover(t.dataset.model);
    tip.pt = { x: e.clientX, y: e.clientY };
    showTip(t);
  });
  root.addEventListener("pointerout", (e) => {
    if (e.pointerType && e.pointerType !== "mouse") return;      // touch: dismissed by tap-away / Esc
    const t = hit(e); if (!t) return;
    if (e.relatedTarget && t.contains(e.relatedTarget)) return;
    setHover(null); hideTip();
  });
  root.addEventListener("pointerdown", (e) => {
    const t = hit(e); if (!t) return;
    tip.pt = { x: e.clientX, y: e.clientY };
    if (t.classList.contains("rk-row")) setHover(t.dataset.model);
    showTip(t);
  });
  root.addEventListener("focusin", (e) => {
    const t = e.target && e.target.closest ? e.target.closest(".rk-row") : null;
    if (!t) return;
    const list = t.closest("ol");
    if (list) list.querySelectorAll(".rk-row").forEach((r) => { r.tabIndex = r === t ? 0 : -1; });
    tip.pt = null;
    setHover(t.dataset.model);
    showTip(t);
  });
  root.addEventListener("focusout", (e) => {
    const t = e.target && e.target.closest ? e.target.closest(".rk-row") : null;
    if (!t) return;
    setHover(null); hideTip();
  });
  root.addEventListener("keydown", (e) => {
    const row = e.target && e.target.closest ? e.target.closest(".rk-row") : null;
    if (!row) return;
    if (e.key === "Enter" || e.key === " " || e.key === "Spacebar") { e.preventDefault(); togglePin(row.dataset.model); return; }
    const list = row.closest("ol");
    if (!list) return;
    const rows = [...list.querySelectorAll(".rk-row")];
    const i = rows.indexOf(row);
    let j = null;
    if (e.key === "ArrowDown") j = Math.min(rows.length - 1, i + 1);
    else if (e.key === "ArrowUp") j = Math.max(0, i - 1);
    else if (e.key === "Home") j = 0; else if (e.key === "End") j = rows.length - 1;
    if (j === null) return;
    e.preventDefault();
    rows.forEach((r, k) => { r.tabIndex = k === j ? 0 : -1; });
    rows[j].focus();
  });
  root.addEventListener("click", (e) => {
    const row = e.target && e.target.closest ? e.target.closest(".rk-row") : null;
    if (row) { togglePin(row.dataset.model); return; }
    // Click on the card background clears pins.
    if (e.target && e.target.closest && !e.target.closest("button, input, a, label, .table-wrap, .seg, .bench-legend")) clearPins();
  });
  document.addEventListener("pointerdown", (e) => {
    if (!tip.target) return;
    if (!(e.target && e.target.closest && e.target.closest(".rk-row, #benchmarks-table tbody tr"))) { setHover(null); hideTip(); }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    hideTip();
    if (els.helpSheet.classList.contains("hidden") && !root.hidden) { setHover(null); clearPins(); }
  });
  window.addEventListener("scroll", () => { if (tip.target) hideTip(); }, { passive: true });
  window.addEventListener("resize", () => { if (tip.target) hideTip(); });
}

// ---------- init ----------
async function loadLanguages() {
  try {
    const res = await fetch("/api/languages");
    if (!res.ok) throw new Error(String(res.status));
    state.langs = await res.json();
  } catch (e) {
    toast("Could not load the language list. Reload to retry.", "error", true);
    showError("Could not load languages from the API.", false);
    return;
  }
  state.byCode = Object.fromEntries(state.langs.map((l) => [l.code, l]));
  state.source = state.byCode.en ? "en" : state.langs[0].code;
  state.target = state.langs.find((l) => l.code !== state.source)?.code ?? state.source;
  try {
    const p = JSON.parse(store("indicmt.pair") || "null");
    if (Array.isArray(p) && state.byCode[p[0]] && state.byCode[p[1]] && p[0] !== p[1]) [state.source, state.target] = p;
  } catch (e) {}
  renderPickerValue("source"); renderPickerValue("target");
  renderExamples();
  renderHistory();
}

async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    const health = await res.json();
    if (health.disclaimer) {
      els.disclaimer.textContent = `⚠ ${health.disclaimer}`;
      els.disclaimer.classList.remove("hidden");
    }
    if (health.status && health.status !== "ok") toast("The model is still loading; translations may fail for a moment.", "error");
  } catch (e) { /* language load reports connectivity problems */ }
}

(function init() {
  onInput();
  resetOutput();
  moveIndicator();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(moveIndicator);
  if (location.hash === "#benchmarks") switchTab("benchmarks");
  renderHistory();
  document.querySelectorAll('[role="radiogroup"]').forEach(enhanceRadiogroup);
  wireBenchControls();
  wireBenchInteractions();
  loadLanguages();
  loadHealth();
})();
