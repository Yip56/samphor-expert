# Samphor Expert — Upgrade To-Do List

## Critical (Core Chatbot Capabilities)
- [x] **Fix 1 — Session Isolation**: one shared `MLEngine` instance means all browser tabs share history. Use Flask `session` (cookie-based) to store per-user context.
- [x] **Fix 2 — Conversation Context**: `_history` is recorded but never fed back into inference. Add follow-up detection ("tell me more") and pronoun resolution ("what is it made of?" after a history reply).
- [x] **Fix 3 — Multi-turn Clarification**: when confidence is in the 0.35–0.70 grey zone, ask the user "Did you mean X or Y?" instead of the generic fallback.
- [x] **Fix 4 — Wire Fallback Engine**: `RuleBasedEngine` exists in `chat_engine.py` but is never used. Auto-fallback to it if `MLEngine` throws or model is absent.

## High Impact (UX & Discoverability)
- [ ] **Fix 5 — Quick Reply Chips**: after every bot response show 3–4 clickable topic buttons so users know what they can ask.
- [ ] **Fix 6 — Typing Indicator**: show "..." animation while waiting for the server.
- [ ] **Fix 7 — Rich Media**: embed images, Wikipedia links, and audio samples in bot replies.
- [ ] **Fix 8 — Conversation Persistence**: save chat to `localStorage` so a page refresh doesn't wipe history.

## Medium Impact (Intelligence)
- [ ] **Fix 9 — Spell Correction**: handle "Samhor", "Sanphor" typos with `pyspellchecker` or fuzzy matching.
- [ ] **Fix 10 — Better NLP**: replace Bag-of-Words with sentence transformers for semantic understanding.
- [ ] **Fix 11 — Data Augmentation**: expand beyond 20 patterns per intent using synonym expansion.
- [ ] **Fix 12 — Feedback Buttons**: thumbs-up / thumbs-down on each reply to collect retraining data.

## Advanced (Power Features)
- [ ] **Fix 13 — Khmer Language Support**: add Khmer-script patterns and replies for native Cambodian users.
- [ ] **Fix 14 — Analytics Dashboard**: `/admin` page showing most-asked topics, fallback rate, confidence distribution.
- [ ] **Fix 15 — Active Learning**: save low-confidence turns to a review queue for human correction.
- [ ] **Fix 16 — Voice Input**: Web Speech API in the browser for hands-free querying.
- [ ] **Fix 17 — Rate Limiting**: throttle `/chat` to prevent abuse; add input length cap.
