/* ═══════════════════════════════════════════════════════════════════
   drum-mode.js — Samphor entrance animation + drum play interaction

   Entrance animations : curtains, textile, statues, bot icon, drum UI.
   Drum interaction    : layered PNG frame animation, zone click
                         detection, hover highlight, keyboard play.
   window.DRUM exposes every piece for isolated testing.
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
  function el(id)   { return document.getElementById(id); }
  function qs(sel)  { return document.querySelector(sel); }

  /* ─────────────────────────────────────────────────────────────────
     SINGLE-PIECE ANIMATION FUNCTIONS
     Each returns a Promise resolving when the animation completes.
     Call individually for isolated testing, e.g.: DRUM.animateStatues()
  ───────────────────────────────────────────────────────────────── */

  function animateTopCurtain() {
    const t    = TIMING.topCurtain;
    const wrap = el('curtain-top-wrap');
    wrap.style.transition = `transform ${t.duration}ms ${t.easing}`;
    return wait(t.delay).then(() => { wrap.classList.add('dm-in'); return wait(t.duration); });
  }

  function animateLeftCurtain() {
    const t    = TIMING.sideCurtains;
    const wrap = el('curtain-left-wrap');
    wrap.style.transition = `transform ${t.duration}ms ${t.easing}`;
    return wait(t.delay).then(() => { wrap.classList.add('dm-in'); return wait(t.duration); });
  }

  function animateRightCurtain() {
    const t    = TIMING.sideCurtains;
    const wrap = el('curtain-right-wrap');
    wrap.style.transition = `transform ${t.duration}ms ${t.easing}`;
    return wait(t.delay).then(() => { wrap.classList.add('dm-in'); return wait(t.duration); });
  }

  function animateBottomTextile() {
    const t    = TIMING.bottomTextile;
    const wrap = el('textile-bottom-wrap');
    wrap.style.transition = `transform ${t.duration}ms ${t.easing}`;
    return wait(t.delay).then(() => { wrap.classList.add('dm-in'); return wait(t.duration); });
  }

  /* Fix 1: Statue rise from fully below stage floor + shake */
  function animateStatues() {
    const t     = TIMING.statues;
    const left  = el('statue-left-wrap');
    const right = el('statue-right-wrap');
    const textileWrap = el('textile-bottom-wrap');
    const textileH    = textileWrap ? textileWrap.offsetHeight : 64;
    left.style.bottom  = textileH + 'px';
    right.style.bottom = textileH + 'px';
    const trans = `opacity ${t.duration}ms ease-in, transform ${t.duration}ms ${t.easing}`;
    left.style.transition  = trans;
    right.style.transition = trans;
    return wait(t.delay).then(() => {
      left.classList.add('dm-in');
      right.classList.add('dm-in');
      return wait(t.duration);
    });
  }

  function animateBotIcon() {
    const t    = TIMING.botIcon;
    const elem = el('drum-bot-icon');
    elem.style.transition = `opacity ${t.duration}ms ${t.easing}, transform ${t.duration}ms ${t.easing}`;
    return wait(t.delay).then(() => { elem.classList.add('dm-in'); return wait(t.duration); });
  }

  function animateDrumCircles() {
    const t    = TIMING.drumCircles;
    const elem = el('drum-circles');
    elem.style.transition = `opacity ${t.duration}ms ${t.easing}, transform ${t.duration}ms ${t.easing}`;
    return wait(t.delay).then(() => { elem.classList.add('dm-in'); return wait(t.duration); });
  }

  function animateNoteLanes() {
    const t    = TIMING.noteLanes;
    const elem = el('note-lanes');
    elem.style.transition = `opacity ${t.duration}ms ${t.easing}`;
    return wait(t.delay).then(() => { elem.classList.add('dm-in'); return wait(t.duration); });
  }

  /* Fix 2: Chat UI retreat — chat box + input + action bar */
  function minimizeChatBox() {
    const t        = TIMING.chatBox;
    const chatEl   = el('chat-box');
    const inputEl  = qs('.input-bar-wrapper');
    const actionEl = qs('.action-bar');
    const moveTrans = `transform ${t.duration}ms ${t.easing}`;
    const fadeTrans = `opacity ${t.duration}ms ease`;
    if (chatEl)   chatEl.style.transition   = `opacity 0.3s ease, ${moveTrans}`;
    if (inputEl)  inputEl.style.transition  = moveTrans;
    if (actionEl) actionEl.style.transition = fadeTrans;
    return wait(t.delay).then(() => { document.body.classList.add('drum-mode'); return wait(t.duration); });
  }

  /* ── Show overlays ─────────────────────────────────────────────── */
  function showTheatre() { el('theatre-overlay').style.display = 'block'; }
  function showDrumUI()  { el('drum-ui').style.display         = 'flex';  }

  /* ─────────────────────────────────────────────────────────────────
     DRUM INTERACTION
     Frame animation: each drum zone has 4 PNG frames that play at
     50 ms intervals then snap back to the idle (first) frame.
     Zone detection: click distance from the container centre,
     thresholds expressed as fractions of the half-width so they
     scale automatically with the CSS container size.
     Keyboard: left hand (A-F) → BigHead; right hand (H-L) → SmallHead.
  ───────────────────────────────────────────────────────────────── */
  const RIM_FRAMES    = ['11','12','13','14'];
  const MIDDLE_FRAMES = ['7','8','9','10'];
  const CENTER_FRAMES = ['3','4','5','6'];
  const IMG_BASE      = '/static/Drum_image/SamphorDrumDesign';

  /* Keys closer to the centre gap map to inner zones.
     A/L = outermost (Rim), D-F / H-J = innermost (Center). */
  const KEY_MAP = {
    'a': { head: 'big',   zone: 'rim',    mute: false },
    's': { head: 'big',   zone: 'middle', mute: false },
    'd': { head: 'big',   zone: 'center', mute: false },
    'f': { head: 'big',   zone: 'center', mute: true  },
    'h': { head: 'small', zone: 'center', mute: true  },
    'j': { head: 'small', zone: 'center', mute: false },
    'k': { head: 'small', zone: 'middle', mute: false },
    'l': { head: 'small', zone: 'rim',    mute: false },
  };

  const HAND_IMG = {
    big:   { open: '/static/Open_hand_(left).png',  hit: '/static/Open_hand_Hit(left).png',  mute: '/static/Mute(left).png'  },
    small: { open: '/static/Open_hand_(right).png', hit: '/static/Open_hand_Hit(right).png', mute: '/static/Mute(right).png' },
  };

  const SOUNDS = {
    center_mute: new Audio('/static/Sounds/Center-mute.wav'),
    center_open: new Audio('/static/Sounds/Center-open.wav'),
    middle:      new Audio('/static/Sounds/Middle-Layer.wav'),
    rim:         new Audio('/static/Sounds/Rim.wav'),
  };

  function playDrumSound(zone, isMute) {
    const snd = zone === 'center'
      ? (isMute ? SOUNDS.center_mute : SOUNDS.center_open)
      : zone === 'middle' ? SOUNDS.middle : SOUNDS.rim;
    snd.currentTime = 0;
    snd.play().catch(() => {});
  }

  /* ── Hand zone offsets for keyboard hits ─────────────────────────────
     Adjust dx (right +) and dy (down +) so each hand lands visually
     inside the correct ring when keys are pressed.
     Click hits use the actual cursor position instead.
  ──────────────────────────────────────────────────────────────────── */
  const HAND_ZONE_OFFSETS = {
    big: {
      center: { dx: 0, dy:   0 },
      middle: { dx: 0, dy:  55 },
      rim:    { dx: 0, dy: 105 },
    },
    small: {
      center: { dx: 0, dy:  0 },
      middle: { dx: 0, dy: 36 },
      rim:    { dx: 0, dy: 68 },
    },
  };

  let drumModeActive = false;

  function initDrumImages() {
    el('Rim').src         = IMG_BASE + RIM_FRAMES[0]    + '.png';
    el('Middle').src      = IMG_BASE + MIDDLE_FRAMES[0] + '.png';
    el('Center').src      = IMG_BASE + CENTER_FRAMES[0] + '.png';
    el('SmallRim').src    = IMG_BASE + RIM_FRAMES[0]    + '.png';
    el('SmallMiddle').src = IMG_BASE + MIDDLE_FRAMES[0] + '.png';
    el('SmallCenter').src = IMG_BASE + CENTER_FRAMES[0] + '.png';
  }

  function playFrames(imgEl, frames) {
    frames.forEach((frame, i) => {
      setTimeout(() => { imgEl.src = IMG_BASE + frame + '.png'; }, 50 * i);
    });
    /* Snap back to idle frame after the last animation step */
    setTimeout(() => { imgEl.src = IMG_BASE + frames[0] + '.png'; }, 50 * frames.length);
  }

  function showHand(x, y, isMute, head) {
    const imgs = HAND_IMG[head];
    const hand = document.createElement('img');
    hand.className       = 'drum-hand';
    hand.src             = isMute ? imgs.mute : imgs.open;
    hand.style.left      = x + 'px';
    hand.style.top       = y + 'px';
    hand.style.transform = 'translate(-50%, -50%) scale(1.2)';
    document.body.appendChild(hand);

    /* Two rAFs ensure scale(1.2) is painted before the transition fires */
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        hand.style.transition = 'transform 150ms ease-out';
        hand.style.transform  = 'translate(-50%, -50%) scale(1.0)';
      });
    });

    /* Frame 2: swap to impact image after 150 ms, then remove */
    setTimeout(() => {
      hand.style.transition = '';
      hand.src = isMute ? imgs.mute : imgs.hit;
      setTimeout(() => hand.remove(), 150);
    }, 150);
  }

  function hitZone(head, zone, clientX, clientY, isMute) {
    const layerId = {
      big:   { rim: 'Rim',      middle: 'Middle',      center: 'Center'      },
      small: { rim: 'SmallRim', middle: 'SmallMiddle', center: 'SmallCenter' },
    }[head][zone];
    const frames = { rim: RIM_FRAMES, middle: MIDDLE_FRAMES, center: CENTER_FRAMES }[zone];
    if (layerId && frames) playFrames(el(layerId), frames);
    playDrumSound(zone, !!isMute);

    let x, y;
    if (clientX !== undefined) {
      x = clientX; y = clientY;
    } else {
      const headEl = el(head === 'big' ? 'BigHead' : 'SmallHead');
      const rect   = headEl.getBoundingClientRect();
      const off    = HAND_ZONE_OFFSETS[head][zone];
      x = rect.left + rect.width  / 2 + off.dx;
      y = rect.top  + rect.height / 2 + off.dy;
    }
    showHand(x, y, !!isMute, head);
  }

  function flashLane(key) {
    const lane = qs(`.note-lane[data-key="${key.toUpperCase()}"]`);
    if (!lane) return;
    lane.classList.add('lane-hit');
    setTimeout(() => lane.classList.remove('lane-hit'), 180);
  }

  /* Zone thresholds derived from drum.html original pixel values,
     normalised to the rendered container's half-width so they hold
     at any CSS size. BigHead: 100/177/260 out of 350 half-px.
                       SmallHead: 65/113/170 out of 225 half-px.  */
  function distToZone(dist, containerW, isBig) {
    const r = containerW / 2;
    if (isBig) {
      if (dist <= r * 0.286) return 'center';
      if (dist <= r * 0.506) return 'middle';
      if (dist <= r * 1.0) return 'rim';
    } else {
      if (dist <= r * 0.289) return 'center';
      if (dist <= r * 0.502) return 'middle';
      if (dist <= r * 1.0) return 'rim';
    }
    return null;
  }

  function setLayerBrightness(ids, value) {
    ids.forEach(id => { const e = el(id); if (e) e.style.filter = `brightness(${value})`; });
  }

  function resetDrumLayers() {
    setLayerBrightness(['Rim','Middle','Center','SmallRim','SmallMiddle','SmallCenter'], 1);
  }

  function onDrumClick(e) {
    if (!drumModeActive) return;
    const bigR   = el('BigHead').getBoundingClientRect();
    const smallR = el('SmallHead').getBoundingClientRect();

    function zoneFor(rect, isBig) {
      return distToZone(
        Math.hypot(e.clientX - (rect.left + rect.width / 2),
                   e.clientY - (rect.top  + rect.height / 2)),
        rect.width, isBig
      );
    }

    if (e.clientX >= bigR.left && e.clientX <= bigR.right &&
        e.clientY >= bigR.top  && e.clientY <= bigR.bottom) {
      const zone = zoneFor(bigR, true);
      if (zone) hitZone('big', zone, e.clientX, e.clientY, zone === 'center');
    } else if (e.clientX >= smallR.left && e.clientX <= smallR.right &&
               e.clientY >= smallR.top  && e.clientY <= smallR.bottom) {
      const zone = zoneFor(smallR, false);
      if (zone) hitZone('small', zone, e.clientX, e.clientY, zone === 'center');
    }
  }

  function onDrumMouseMove(e) {
    if (!drumModeActive) return;
    resetDrumLayers();
    const bigR   = el('BigHead').getBoundingClientRect();
    const smallR = el('SmallHead').getBoundingClientRect();

    function highlight(rect, isBig, prefix) {
      const zone = distToZone(
        Math.hypot(e.clientX - (rect.left + rect.width / 2),
                   e.clientY - (rect.top  + rect.height / 2)),
        rect.width, isBig
      );
      const layerId = { rim: prefix+'Rim', middle: prefix+'Middle', center: prefix+'Center' }[zone];
      if (layerId) setLayerBrightness([layerId], 1.25);
    }

    if (e.clientX >= bigR.left && e.clientX <= bigR.right &&
        e.clientY >= bigR.top  && e.clientY <= bigR.bottom) {
      highlight(bigR, true, '');
    } else if (e.clientX >= smallR.left && e.clientX <= smallR.right &&
               e.clientY >= smallR.top  && e.clientY <= smallR.bottom) {
      highlight(smallR, false, 'Small');
    }
  }

  function onDrumKeyDown(e) {
    if (!drumModeActive || e.repeat) return;
    const mapping = KEY_MAP[e.key.toLowerCase()];
    if (!mapping) return;
    hitZone(mapping.head, mapping.zone, undefined, undefined, mapping.mute);
    flashLane(e.key.toLowerCase());
  }

  /* ─────────────────────────────────────────────────────────────────
     RESET — snaps all pieces back to pre-animation state.
  ───────────────────────────────────────────────────────────────── */
  function resetDrumMode() {
    drumModeActive = false;

    /* Suppress transitions so pieces snap to off-screen instantly */
    ['curtain-top-wrap','curtain-left-wrap','curtain-right-wrap','textile-bottom-wrap']
      .forEach(id => {
        const e = el(id); if (!e) return;
        e.style.transition = 'none';
        e.classList.remove('dm-in');
      });

    ['statue-left-wrap','statue-right-wrap'].forEach(id => {
      const e = el(id); if (!e) return;
      e.style.transition = 'none';
      e.style.bottom     = '';
      e.classList.remove('dm-in');
    });

    ['drum-bot-icon','drum-circles','note-lanes'].forEach(id => {
      const e = el(id); if (!e) return;
      e.style.transition = 'none';
      e.classList.remove('dm-in');
    });

    resetDrumLayers();

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
     t=0    chat UI retreats (concurrent)
            theatre curtains + textile all start (concurrent)
            bot icon relocates (concurrent)
     after textile: statues rise (chained)
     after bot icon: drum circles → note lanes (chained)
  ───────────────────────────────────────────────────────────────── */
  async function enterDrumMode() {
    showTheatre();
    showDrumUI();
    initDrumImages();
    drumModeActive = true;

    /* One rAF ensures display:block/flex is painted before transitions fire */
    await wait(16);

    minimizeChatBox();

    const textileP = animateBottomTextile();
    animateTopCurtain();
    animateLeftCurtain();
    animateRightCurtain();

    const statuesP = textileP.then(() => animateStatues());

    const botIconP = animateBotIcon();
    await botIconP;
    await animateDrumCircles();
    await animateNoteLanes();

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
    hitZone,
    flashLane,
    initDrumImages,
    showHand,
    HAND_IMG,
    HAND_ZONE_OFFSETS,
    SOUNDS,
    playDrumSound,
  };

  window.DRUM          = DRUM;
  window.enterDrumMode = enterDrumMode;

  document.addEventListener('DOMContentLoaded', () => {
    const exitBtn = el('drum-exit-btn');
    if (exitBtn) exitBtn.addEventListener('click', () => resetDrumMode());

    /* Drum interaction listeners — guarded by drumModeActive flag,
       so they're silent while the chat UI is active. */
    document.addEventListener('click',     onDrumClick);
    document.addEventListener('mousemove', onDrumMouseMove);
    document.addEventListener('keydown',   onDrumKeyDown);
  });
})();
