/* ═══════════════════════════════════════════════════════════════════
   drum-mode.js — Samphor entrance animation for drum play mode

   Each animation piece is its own function, callable in isolation.
   Master enterDrumMode() orchestrates the sequence.
   window.DRUM exposes every piece for testing.
═══════════════════════════════════════════════════════════════════ */
(function () {
  'use strict';

  /* ── Timing configuration ─────────────────────────────────────────
     All times in ms. Adjust any value here without touching the
     sequence logic below.
  ──────────────────────────────────────────────────────────────── */
  const TIMING = {
    topCurtain:    { delay: 0,    duration: 700, easing: 'cubic-bezier(0.22,1,0.36,1)' },
    sideCurtains:  { delay: 100,  duration: 700, easing: 'cubic-bezier(0.22,1,0.36,1)' },
    bottomTextile: { delay: 0,    duration: 600, easing: 'cubic-bezier(0.22,1,0.36,1)' },
    statues:       { delay: 0,    duration: 650, easing: 'cubic-bezier(0.22,1,0.36,1)' },
    botIcon:       { delay: 200,  duration: 500, easing: 'cubic-bezier(0.34,1.56,0.64,1)' },
    drumCircles:   { delay: 0,    duration: 500, easing: 'ease-out' },
    noteLanes:     { delay: 0,    duration: 400, easing: 'ease-out' },
    chatBox:       { delay: 0,    duration: 400, easing: 'ease-in'  },
  };

  /* ── Helpers ──────────────────────────────────────────────────── */
  function wait(ms) { return new Promise(r => setTimeout(r, ms)); }
  function el(id) { return document.getElementById(id); }
  function qs(sel) { return document.querySelector(sel); }

  /* ─────────────────────────────────────────────────────────────────
     SINGLE-PIECE ANIMATION FUNCTIONS
     Each returns a Promise resolving when the animation completes.
     Call individually for isolated testing, e.g.: DRUM.animateStatues()
  ───────────────────────────────────────────────────────────────── */

  function animateTopCurtain() {
    const t = TIMING.topCurtain;
    const wrap = el('curtain-top-wrap');        /* wrapper now owns the transition */
    wrap.style.transition = `transform ${t.duration}ms ${t.easing}`;
    return wait(t.delay).then(() => {
      wrap.classList.add('dm-in');
      return wait(t.duration);
    });
  }

  function animateLeftCurtain() {
    const t = TIMING.sideCurtains;
    const wrap = el('curtain-left-wrap');
    wrap.style.transition = `transform ${t.duration}ms ${t.easing}`;
    return wait(t.delay).then(() => {
      wrap.classList.add('dm-in');
      return wait(t.duration);
    });
  }

  function animateRightCurtain() {
    const t = TIMING.sideCurtains;
    const wrap = el('curtain-right-wrap');
    wrap.style.transition = `transform ${t.duration}ms ${t.easing}`;
    return wait(t.delay).then(() => {
      wrap.classList.add('dm-in');
      return wait(t.duration);
    });
  }

  function animateBottomTextile() {
    const t = TIMING.bottomTextile;
    const wrap = el('textile-bottom-wrap');     /* wrapper now owns the transition */
    wrap.style.transition = `transform ${t.duration}ms ${t.easing}`;
    return wait(t.delay).then(() => {
      wrap.classList.add('dm-in');
      return wait(t.duration);
    });
  }

  /* ── Fix 1: Statue rise from fully below stage floor + shake ──── */
  function animateStatues() {
    const t = TIMING.statues;
    const left  = el('statue-left-wrap');
    const right = el('statue-right-wrap');

    /* Only set `bottom` inline (dynamic per viewport); opacity and transform
       live in CSS so the .dm-in class can override them without specificity
       conflicts from inline styles (inline > class in the cascade).
       CSS initial state: opacity:0, transform:translateY(100%) — hidden below stage.
       CSS .dm-in state:  opacity:1, transform:translateY(0)    — at resting position. */
    const textileWrap = el('textile-bottom-wrap');
    const textileH = textileWrap ? textileWrap.offsetHeight : 64;
    left.style.bottom  = textileH + 'px';
    right.style.bottom = textileH + 'px';

    const trans = `opacity ${t.duration}ms ease-in, transform ${t.duration}ms ${t.easing}`;
    left.style.transition  = trans;
    right.style.transition = trans;

    return wait(t.delay).then(() => {
      /* .dm-in transitions: opacity → 1, transform → translateY(0).
         CSS shake animation also fires on the inner .statue-shake div. */
      left.classList.add('dm-in');
      right.classList.add('dm-in');
      return wait(t.duration);
    });
  }

  function animateBotIcon() {
    const t = TIMING.botIcon;
    const elem = el('drum-bot-icon');
    elem.style.transition = `opacity ${t.duration}ms ${t.easing}, transform ${t.duration}ms ${t.easing}`;
    return wait(t.delay).then(() => {
      elem.classList.add('dm-in');
      return wait(t.duration);
    });
  }

  function animateDrumCircles() {
    const t = TIMING.drumCircles;
    const elem = el('drum-circles');
    elem.style.transition = `opacity ${t.duration}ms ${t.easing}, transform ${t.duration}ms ${t.easing}`;
    return wait(t.delay).then(() => {
      elem.classList.add('dm-in');
      return wait(t.duration);
    });
  }

  function animateNoteLanes() {
    const t = TIMING.noteLanes;
    const elem = el('note-lanes');
    elem.style.transition = `opacity ${t.duration}ms ${t.easing}`;
    return wait(t.delay).then(() => {
      elem.classList.add('dm-in');
      return wait(t.duration);
    });
  }

  /* ── Fix 2: Chat UI retreat — chat box + input + action bar ────── */
  function minimizeChatBox() {
    const t        = TIMING.chatBox;
    const chatEl   = el('chat-box');
    const inputEl  = qs('.input-bar-wrapper');
    const actionEl = qs('.action-bar');
    const moveTrans = `transform ${t.duration}ms ${t.easing}`;
    const fadeTrans = `opacity ${t.duration}ms ease`;
    if (chatEl)   chatEl.style.transition   = `opacity 0.3s ease, ${moveTrans}`;
    if (inputEl)  inputEl.style.transition  = moveTrans;
    if (actionEl) actionEl.style.transition = fadeTrans;   /* Fix 2: fade, not snap */
    return wait(t.delay).then(() => {
      document.body.classList.add('drum-mode');
      return wait(t.duration);
    });
  }

  /* ── Show overlays ─────────────────────────────────────────────── */
  function showTheatre() { el('theatre-overlay').style.display = 'block'; }
  function showDrumUI()   { el('drum-ui').style.display        = 'flex';  }

  /* ─────────────────────────────────────────────────────────────────
     RESET — snaps all pieces back to pre-animation state.
     Useful for replaying the sequence or individual pieces.
  ───────────────────────────────────────────────────────────────── */
  function resetDrumMode() {
    /* Suppress transitions so pieces snap to off-screen instantly */
    const wrapperIds = [
      'curtain-top-wrap', 'curtain-left-wrap', 'curtain-right-wrap',
      'textile-bottom-wrap',
    ];
    wrapperIds.forEach(id => {
      const e = el(id);
      if (!e) return;
      e.style.transition = 'none';
      e.classList.remove('dm-in');
    });

    /* Statue wraps: only `bottom` was set inline (Fix 1); opacity/transform
       revert to CSS defaults automatically when .dm-in is removed. */
    ['statue-left-wrap', 'statue-right-wrap'].forEach(id => {
      const e = el(id);
      if (!e) return;
      e.style.transition = 'none';
      e.style.bottom     = '';   /* remove dynamic bottom → CSS fallback 68px */
      e.classList.remove('dm-in');
    });

    ['drum-bot-icon', 'drum-circles', 'note-lanes'].forEach(id => {
      const e = el(id);
      if (!e) return;
      e.style.transition = 'none';
      e.classList.remove('dm-in');
    });

    el('theatre-overlay').style.display = 'none';
    el('drum-ui').style.display         = 'none';

    /* Fix 2: restore chat UI */
    document.body.classList.remove('drum-mode');
    const chatEl   = el('chat-box');
    const inputEl  = qs('.input-bar-wrapper');
    const actionEl = qs('.action-bar');
    if (chatEl)   { chatEl.style.transition   = ''; chatEl.style.transform   = ''; }
    if (inputEl)  { inputEl.style.transition  = ''; inputEl.style.transform  = ''; }
    if (actionEl) { actionEl.style.transition = ''; actionEl.style.opacity   = ''; }
  }

  /* ─────────────────────────────────────────────────────────────────
     MASTER ENTRANCE SEQUENCE
     Sequence:
       t=0    chat UI retreats (concurrent, no await)
              theatre curtains + textile all start (concurrent)
              bot icon relocates (concurrent)
       after textile: statues rise with shake (chained)
       after bot icon: drum circles → note lanes (chained)
  ───────────────────────────────────────────────────────────────── */
  async function enterDrumMode() {
    showTheatre();
    showDrumUI();

    /* One rAF ensures display:block is painted before transitions fire */
    await wait(16);

    /* Chat UI retreats concurrently — no await */
    minimizeChatBox();

    /* Theatre — all start together; keep textile promise for statue chaining */
    const textileP = animateBottomTextile();
    animateTopCurtain();   /* concurrent */
    animateLeftCurtain();  /* concurrent, has own configured delay */
    animateRightCurtain(); /* concurrent */

    /* Statues only after textile lands */
    const statuesP = textileP.then(() => animateStatues());

    /* Bot icon starts concurrently with theatre */
    const botIconP = animateBotIcon();

    /* Drum circles after bot icon */
    await botIconP;
    await animateDrumCircles();

    /* Note lanes after drum circles */
    await animateNoteLanes();

    /* statuesP resolves in the background */
    void statuesP;
  }

  /* ─────────────────────────────────────────────────────────────────
     PUBLIC API
  ───────────────────────────────────────────────────────────────── */
  const DRUM = {
    TIMING,
    animateTopCurtain,
    animateLeftCurtain,
    animateRightCurtain,
    animateBottomTextile,
    animateStatues,
    animateBotIcon,
    animateDrumCircles,
    animateNoteLanes,
    minimizeChatBox,
    showTheatre,
    showDrumUI,
    resetDrumMode,
    enterDrumMode,
  };

  window.DRUM          = DRUM;
  window.enterDrumMode = enterDrumMode;

  document.addEventListener('DOMContentLoaded', () => {
    const exitBtn = el('drum-exit-btn');
    if (exitBtn) exitBtn.addEventListener('click', () => resetDrumMode());
  });
})();
