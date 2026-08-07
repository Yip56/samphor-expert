/* ═══════════════════════════════════════════════════════════════════
   drum-tutorial.js — Tutorial overlay for drum play mode.
═══════════════════════════════════════════════════════════════════ */
(function () {
  'use strict';

  /* ── Logging ────────────────────────────────────────────────────── */
  const P = '[DrumTutorial]';
  function log(m, ...a)  { console.log(`${P} ${m}`, ...a); }
  function warn(m, ...a) { console.warn(`${P} ${m}`, ...a); }
  function err(m, ...a)  { console.error(`${P} ${m}`, ...a); }

  /* ── Engine constants ───────────────────────────────────────────── */
  const NOTE_R     = 20;     // note circle radius, px
  const APPROACH   = 2.5;    // approach ring starts at NOTE_R × APPROACH radius
  const TRAVEL     = 1.5;    // seconds for a note to cross the full lane height
  const HIT_WINDOW = 0.100;  // ±100 ms hit window
  const HIT_LINE   = 34;     // px from lane bottom to note centre at hit_time

  /* ── Color palette ──────────────────────────────────────────────── */
  const COLOR = {
    a: '#D9A441', l: '#D9A441',
    s: '#B5552E', k: '#B5552E',
    d: '#4E8570', j: '#4E8570',
    f: '#6B3F5C', h: '#6B3F5C',
  };

  /* ═══════════════════════════════════════════════════════════════════
     CHART DEFINITIONS
     ─────────────────────────────────────────────────────────────────
     HOW TO CUSTOMISE A CHART
     ────────────────────────
     Each chart is an object  { notes: [ ...noteObjects ] }
     Each note object has two fields:

       time  — number (seconds after the tutorial starts) when the note
               should reach the hit line and be struck.
               • Keep the first note's time ≥ TRAVEL (1.5 s) so the
                 player sees an empty lane before the first note arrives.
               • Notes don't need to be in order here — they are sorted
                 automatically when the tutorial starts.

       key   — string, one of: 'a' 's' 'd' 'f'  (left drum, A=rim → F=mute)
                                'h' 'j' 'k' 'l'  (right drum, H=mute → L=rim)

     TIPS
     ────
       • Decrease the gap between consecutive times → faster / harder.
       • Add the same key twice close together for rapid-fire practice.
       • Mix both sides to train coordination (e.g. 'a' then 'l').
       • The helper buildNotes(keys, interval, startTime) below lets you
         describe a pattern as an array of key names without writing out
         every timestamp by hand.

     EXAMPLE — add a custom note at 5.0 s on key 'd':
       { time: 5.0, key: 'd' }
  ═══════════════════════════════════════════════════════════════════ */

  /* Utility: turn a key sequence into timed note objects. */
  function buildNotes(keys, interval, startTime) {
    return keys.map((k, i) => ({ time: +(startTime + i * interval).toFixed(3), key: k }));
  }

  /* ── CHART 1 — Easy ─────────────────────────────────────────────────
     Three passes through all 8 lanes in simple left-to-right order.
     0.75 s between each note — plenty of time to react.            */
  const CHART_1 = { notes: [
    // ── pass 1 ──
    ...buildNotes(['a','s','d','f','h','j','k','l'], 0.75, 2.0),
    // ── pass 2 ──
    ...buildNotes(['a','s','d','f','h','j','k','l'], 0.75, 2.0 + 8*0.75 + 0.5),
    // ── pass 3 ──
    ...buildNotes(['a','s','d','f','h','j','k','l'], 0.75, 2.0 + 8*0.75*2 + 1.0),
  ]};

  /* ── CHART 2 — Medium ───────────────────────────────────────────────
     Four varied patterns: outer→inner mirrored, inner→outer, left-to-
     right, right-to-left. 0.55 s between notes — requires more focus.
     ─────────────────────────────────────────────────────────────────
     TO CUSTOMISE: change the key arrays in buildNotes(), adjust the
     0.55 interval, or add/remove entire buildNotes() blocks.        */
  const CHART_2 = (() => {
    const GAP = 0.55;
    let t = 2.0;
    const notes = [];

    // outer → inner on both drums simultaneously (mirrored pairs)
    notes.push(...buildNotes(['a','l','s','k','d','j','f','h'], GAP, t));
    t += 8*GAP + 0.65;

    // inner → outer
    notes.push(...buildNotes(['h','f','j','d','k','s','l','a'], GAP, t));
    t += 8*GAP + 0.65;

    // left side then right side
    notes.push(...buildNotes(['a','s','d','f','h','j','k','l'], GAP, t));
    t += 8*GAP + 0.65;

    // right side then left side (reverse)
    notes.push(...buildNotes(['l','k','j','h','f','d','s','a'], GAP, t));

    return { notes };
  })();

  /* ── CHART 3 — Hard ─────────────────────────────────────────────────
     Six patterns with cross-hand sequences at 0.38 s per note.
     Tests both reaction time and hand alternation.
     ─────────────────────────────────────────────────────────────────
     TO CUSTOMISE: change key arrays or the GAP value.
     Reducing GAP below 0.30 is very challenging.                    */
  const CHART_3 = (() => {
    const GAP = 0.38;
    let t = 2.0;
    const notes = [];

    // mirrored pairs: each pair shares a colour
    notes.push(...buildNotes(['a','l','s','k','d','j','f','h'], GAP, t));
    t += 8*GAP + 0.5;

    // cross-pairs: left-rim then right-mute, etc.
    notes.push(...buildNotes(['a','h','s','j','d','k','f','l'], GAP, t));
    t += 8*GAP + 0.5;

    // same cross-pairs reversed
    notes.push(...buildNotes(['l','f','k','d','j','s','h','a'], GAP, t));
    t += 8*GAP + 0.5;

    // alternating left/right rims then inner
    notes.push(...buildNotes(['a','l','f','h','s','k','d','j'], GAP, t));
    t += 8*GAP + 0.5;

    // right-first sweep then left-first sweep
    notes.push(...buildNotes(['h','j','k','l','a','s','d','f'], GAP, t));
    t += 8*GAP + 0.5;

    // inner-to-outer both sides
    notes.push(...buildNotes(['f','d','s','a','h','j','k','l'], GAP, t));

    return { notes };
  })();

  const CHARTS = { 1: CHART_1, 2: CHART_2, 3: CHART_3 };

  log('Charts ready — notes per level:',
      Object.entries(CHARTS).map(([k, v]) => `Lv${k}:${v.notes.length}`).join(' '));

  /* ── Runtime state ──────────────────────────────────────────────── */
  let active       = false;
  let currentLevel = 1;
  let startPerf    = 0;
  let pending      = [];
  let live         = [];
  let totalNotes   = 0;
  let totalHit     = 0;
  let resultsShown = false;
  let rafId        = null;

  /* ── Helpers ────────────────────────────────────────────────────── */
  function laneEl(key) {
    const el = document.querySelector(`.note-lane[data-key="${key.toUpperCase()}"]`);
    if (!el) warn('Lane element not found for key:', key);
    return el;
  }

  function now() { return (performance.now() - startPerf) / 1000; }

  /* ── Feedback text ──────────────────────────────────────────────── */
  function showFeedback(isHit) {
    const el = document.getElementById('tutorial-feedback-text');
    if (!el) return;
    // Remove and re-add animation class to restart it even on rapid calls
    el.classList.remove('tf-show', 'tf-hit', 'tf-miss');
    void el.offsetWidth; // force reflow so removal is painted
    el.textContent = isHit ? 'Perfect!' : 'Miss';
    el.classList.add(isHit ? 'tf-hit' : 'tf-miss', 'tf-show');
  }

  /* ── Start / stop button + space hint visibility ───────────────── */
  function setRunningUI(running) {
    const startBtn  = document.getElementById('tutorial-start-btn');
    const stopBtn   = document.getElementById('tutorial-stop-btn');
    const levelSel  = document.getElementById('tutorial-level-select');
    const spaceHint = document.getElementById('tutorial-space-hint');
    if (startBtn)  startBtn.style.display = running ? 'none' : '';
    if (stopBtn)   stopBtn.style.display  = running ? ''     : 'none';
    if (spaceHint) spaceHint.textContent  = running ? 'Press Space to stop' : 'Press Space to start';
  }

  /* ── Note lifecycle ─────────────────────────────────────────────── */
  function spawnNote(note) {
    log('Spawn:', note.key, 'hitTime:', note.hitTime.toFixed(3));
    const lane = laneEl(note.key);
    if (!lane) { err('Lane missing for key:', note.key); return; }

    const color = COLOR[note.key];

    const wrap = document.createElement('div');
    wrap.className = 'tn-wrap';

    const ring = document.createElement('div');
    ring.className         = 'tn-ring';
    ring.style.borderColor = color;

    const dot = document.createElement('div');
    dot.className        = 'tn-dot';
    dot.style.background = color;
    dot.style.boxShadow  = `0 0 8px ${color}99`;

    wrap.appendChild(ring);
    wrap.appendChild(dot);
    lane.appendChild(wrap);

    note.wrapEl = wrap;
    note.ringEl = ring;
    note.dotEl  = dot;
    note.laneEl = lane;
  }

  function updateNote(note, t) {
    const laneH = note.laneEl.clientHeight;
    if (laneH === 0) warn('Lane clientHeight is 0 for key:', note.key);
    const hitY = laneH - HIT_LINE;
    const p    = (t - note.spawnTime) / TRAVEL;

    note.wrapEl.style.top = (hitY * p - NOTE_R) + 'px';

    const startR   = NOTE_R * APPROACH;
    const currentR = startR + (NOTE_R - startR) * Math.min(p, 1);
    const d = currentR * 2;
    note.ringEl.style.width  = d + 'px';
    note.ringEl.style.height = d + 'px';
  }

  function resolveNote(note, isHit) {
    if (note.scored) return;
    note.scored = true;
    live = live.filter(n => n !== note);

    totalNotes++;
    if (isHit) totalHit++;

    log('Resolved:', note.key, isHit ? 'HIT' : 'MISS',
        `| score: ${totalHit}/${totalNotes}`);

    showFeedback(isHit);

    note.ringEl.remove();
    if (isHit) {
      note.wrapEl.classList.add('tn-hit');
    } else {
      note.dotEl.style.background = '#c0392b';
      note.dotEl.style.boxShadow  = 'none';
      note.wrapEl.classList.add('tn-miss');
    }
    setTimeout(() => note.wrapEl && note.wrapEl.remove(), 280);
  }

  /* ── Game loop ──────────────────────────────────────────────────── */
  function gameLoop() {
    if (!active) return;
    const t = now();

    while (pending.length && t >= pending[0].spawnTime) {
      const note = pending.shift();
      spawnNote(note);
      live.push(note);
    }

    for (const note of [...live]) {
      updateNote(note, t);
      if (!note.scored && t > note.hitTime + HIT_WINDOW) {
        resolveNote(note, false);
      }
    }

    if (!resultsShown && pending.length === 0 && live.length === 0 && totalNotes > 0) {
      log('All notes resolved — showing results in 400 ms.');
      resultsShown = true;
      setTimeout(showResults, 400);
    }

    rafId = requestAnimationFrame(gameLoop);
  }

  /* ── Key handler ────────────────────────────────────────────────── */
  function onKeyDown(e) {
    if (!active || e.repeat) return;
    const key = e.key.toLowerCase();
    if (!COLOR[key]) return;

    const t = now();
    const inWindow = live.filter(n => n.key === key && !n.scored && Math.abs(t - n.hitTime) <= HIT_WINDOW);
    log('Key:', key, 't:', t.toFixed(3), '| notes in window:', inWindow.length);

    const target = inWindow.sort((a, b) => a.hitTime - b.hitTime)[0];
    if (target) {
      log('  → HIT delta:', (t - target.hitTime).toFixed(4) + 's');
      resolveNote(target, true);
    } else {
      log('  → free play (no note in window)');
    }
  }

  /* ── Results screen ─────────────────────────────────────────────── */
  function showResults() {
    const pct = totalNotes > 0 ? Math.round((totalHit / totalNotes) * 100) : 0;
    log('Results: Lv' + currentLevel, totalHit + '/' + totalNotes, '=', pct + '%');

    const pctEl    = document.getElementById('tr-pct');
    const detailEl = document.getElementById('tr-detail');
    const overlay  = document.getElementById('tutorial-results');

    if (!overlay) { err('Missing #tutorial-results'); return; }
    if (pctEl)    pctEl.textContent    = pct + '%';
    if (detailEl) detailEl.textContent = `${totalHit} / ${totalNotes} notes hit`;
    overlay.style.display = 'flex';
  }

  /* ── Start ──────────────────────────────────────────────────────── */
  function start(chart) {
    log('start() — active:', active, 'level:', currentLevel);
    if (active) { warn('Already running'); return; }

    chart = chart || CHARTS[currentLevel] || CHART_1;
    log('Starting Lv' + currentLevel + ' with', chart.notes.length, 'notes');

    const missingLanes = [...new Set(chart.notes.map(n => n.key.toLowerCase()))]
      .filter(k => !laneEl(k));
    if (missingLanes.length) {
      err('Lane elements missing for keys:', missingLanes.join(', '));
      err('Are the #note-lanes data-key divs in the DOM?');
      return;
    }

    active       = true;
    startPerf    = performance.now();
    totalNotes   = 0;
    totalHit     = 0;
    resultsShown = false;
    live         = [];

    pending = chart.notes
      .map((n, i) => ({
        id: i, key: n.key.toLowerCase(),
        hitTime: n.time, spawnTime: n.time - TRAVEL,
        scored: false,
        wrapEl: null, ringEl: null, dotEl: null, laneEl: null,
      }))
      .sort((a, b) => a.spawnTime - b.spawnTime);

    log('First spawn at t =', pending[0].spawnTime.toFixed(3) + 's',
        '| first hit at t =', pending[0].hitTime.toFixed(3) + 's');

    setRunningUI(true);
    document.addEventListener('keydown', onKeyDown);
    rafId = requestAnimationFrame(gameLoop);
    log('Loop started (rafId:', rafId + ')');
  }

  /* ── Stop ───────────────────────────────────────────────────────── */
  function stop() {
    log('stop() — active:', active);
    active = false;
    document.removeEventListener('keydown', onKeyDown);
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
    live.forEach(n => n.wrapEl && n.wrapEl.remove());
    live = []; pending = [];
    const r = document.getElementById('tutorial-results');
    if (r) r.style.display = 'none';
    setRunningUI(false);
    log('Stopped and cleaned up.');
  }

  /* ── Spacebar to start / stop ──────────────────────────────────── */
  // Space starts the tutorial when idle, and stops it when running.
  function onSpaceStart(e) {
    if (e.code !== 'Space') return;
    const drumUI = document.getElementById('drum-ui');
    if (!drumUI || drumUI.style.display !== 'flex') return;
    e.preventDefault();
    if (active) {
      log('Space pressed — stopping tutorial');
      stop();
    } else {
      log('Space pressed — starting tutorial (level', currentLevel + ')');
      start();
    }
  }

  /* ── DOMContentLoaded wiring ────────────────────────────────────── */
  document.addEventListener('DOMContentLoaded', () => {
    log('DOM ready — wiring buttons.');
    document.addEventListener('keydown', onSpaceStart);

    const startBtn = document.getElementById('tutorial-start-btn');
    if (startBtn) {
      log('Found #tutorial-start-btn');
      startBtn.addEventListener('click', () => { log('Start clicked'); start(); });
    } else { err('#tutorial-start-btn not found'); }

    const stopBtn = document.getElementById('tutorial-stop-btn');
    if (stopBtn) {
      log('Found #tutorial-stop-btn');
      stopBtn.addEventListener('click', () => { log('Stop clicked'); stop(); });
    } else { err('#tutorial-stop-btn not found'); }

    const retryBtn = document.getElementById('tr-retry');
    if (retryBtn) {
      retryBtn.addEventListener('click', () => {
        log('Retry clicked');
        document.getElementById('tutorial-results').style.display = 'none';
        stop();
        start();
      });
    } else { err('#tr-retry not found'); }

    const trStopBtn = document.getElementById('tr-stop');
    if (trStopBtn) {
      trStopBtn.addEventListener('click', () => {
        log('Results stop clicked');
        stop();
      });
    } else { err('#tr-stop not found'); }

    // Level picker
    const levelBtns = document.querySelectorAll('.tl-btn');
    if (levelBtns.length) {
      log('Found', levelBtns.length, 'level buttons');
      levelBtns.forEach(btn => {
        btn.addEventListener('click', () => {
          if (active) { warn('Cannot change level while tutorial is running'); return; }
          currentLevel = parseInt(btn.dataset.level, 10);
          levelBtns.forEach(b => b.classList.toggle('tl-active', b === btn));
          log('Level set to', currentLevel);
        });
      });
    } else { err('.tl-btn level buttons not found'); }
  });

  window.DRUM_TUTORIAL = { start, stop, CHARTS };
  log('Module loaded. DRUM_TUTORIAL.start() to begin.');
})();
