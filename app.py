# =============================================================================
# app.py
# =============================================================================
# Flask web server — entry point for the Samphor Expert chatbot.
#
# UPGRADES APPLIED (see upgrade.md):
#   Fix 1 — Session isolation  : per-user context stored in Flask session
#   Fix 4 — Fallback engine    : auto-falls back to RuleBasedEngine if ML fails
#   Logging                    : every turn written to logs/chat.log
#   /logs endpoint             : returns recent log lines for inspection
# =============================================================================

import collections
import json
import logging
import os
import time
import uuid
from datetime import datetime

from flask import Flask, jsonify, render_template, request, session

from engine.chat_engine import RuleBasedEngine
from engine.ml_engine import MLEngine
from knowledge.samphor_kb import RICH_MEDIA

# =============================================================================
# ENGINE SETUP
# =============================================================================
_ml   = MLEngine()
_rule = RuleBasedEngine()

# Fix 5 — Quick reply chips: suggested follow-up questions per intent.
_CHIPS: dict[str, list[str]] = {
    "greeting":         ["What is the Samphor?", "When did it originate?", "What is it made of?", "How is it played?"],
    "ask_definition":   ["When did it originate?", "What is it made of?", "How is it played?", "Where is it used?"],
    "ask_history":      ["What is it made of?", "What does it look like?", "Where is it used?", "How is it preserved?"],
    "ask_material":     ["What does it look like?", "How is it played?", "How is it tuned?", "When did it originate?"],
    "ask_shape":        ["What is it made of?", "How is it played?", "How does it compare to other drums?", "When did it originate?"],
    "ask_playing":      ["How is it tuned?", "How do I learn to play it?", "What is it made of?", "Where is it used?"],
    "ask_tuning":       ["How is it played?", "What is it made of?", "How do I learn to play it?", "What does it look like?"],
    "ask_ceremonies":   ["What is the Pinpeat ensemble?", "When did it originate?", "How is it played?", "How is it preserved?"],
    "ask_pinpeat":      ["Where is it used?", "How is it played?", "How do I learn to play?", "When did it originate?"],
    "ask_compare":      ["What is it made of?", "When did it originate?", "How is it played?", "What does it look like?"],
    "ask_learning":     ["How is it played?", "How is it tuned?", "What is the Pinpeat ensemble?", "How is it preserved?"],
    "ask_preservation": ["When did it originate?", "How do I learn to play?", "Where is it used?", "What is the Pinpeat ensemble?"],
    "farewell":         [],
    "out_of_scope":     ["What is the Samphor?", "How is it played?", "When did it originate?", "What is it made of?"],
}


# Fix 14 — analytics counters (reset on server restart).
_stats: dict = {
    "total":          0,
    "intents":        collections.Counter(),
    "fallbacks":      0,
    "confidence_sum": 0.0,
}

# Fix 17 — in-memory rate limiter: maps IP → list of request timestamps.
_rate_limit: dict[str, list[float]] = {}
REVIEW_QUEUE_PATH = os.path.join("logs", "review_queue.jsonl")


def _check_rate(ip: str, max_per_minute: int = 30) -> bool:
    """Return True if within limit, False if the IP has exceeded max_per_minute."""
    now   = time.time()
    times = [t for t in _rate_limit.get(ip, []) if now - t < 60]
    if len(times) >= max_per_minute:
        return False
    times.append(now)
    _rate_limit[ip] = times
    return True


def _save_review(sid: str, message: str, confidence: float, intent: str | None) -> None:
    """Fix 15: Append a low-confidence turn to the active-learning review queue."""
    entry = {
        "ts":         datetime.now().isoformat(timespec="seconds"),
        "sid":        sid,
        "message":    message,
        "confidence": round(confidence, 4),
        "intent":     intent,
    }
    try:
        with open(REVIEW_QUEUE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def _get_reply(user_message: str, context: dict) -> tuple[str, dict]:
    """Route a user message through MLEngine, fall back to RuleBasedEngine."""
    if _ml._model is not None:
        try:
            return _ml.respond(user_message, context)
        except Exception as exc:
            _log.error("MLEngine.respond() raised %s — falling back to RuleBasedEngine", exc)
    else:
        _log.warning("MLEngine model not loaded — using RuleBasedEngine fallback")
    reply, ctx = _rule.respond(user_message, context)
    return reply, ctx


# =============================================================================
# LOGGING SETUP
# =============================================================================
# Every chat turn is appended to logs/chat.log (UTF-8) AND echoed to the
# console so you can watch live in the terminal.
# GET /logs returns the last 100 lines as JSON for quick inspection.
# =============================================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("logs/chat.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
_log = logging.getLogger("samphor")

# =============================================================================
# FLASK APP
# =============================================================================
app = Flask(__name__)

# secret_key is required for Flask session (cookie encryption).
# Override with the SECRET_KEY environment variable in production.
app.secret_key = os.environ.get("SECRET_KEY", "samphor-expert-dev-key-2024")


@app.before_request
def _ensure_session_id():
    """Assign a short session ID on first visit for log correlation."""
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex[:8]


# =============================================================================
# ROUTE: /
# =============================================================================
@app.route("/")
def index():
    return render_template("index.html")


# =============================================================================
# ROUTE: /chat
# =============================================================================
@app.route("/chat", methods=["POST"])
def chat():
    # Fix 17: Rate limiting.
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
    if not _check_rate(ip):
        return jsonify({"response": "Too many messages — please wait a moment before sending again."}), 429

    data         = request.get_json(force=True)
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Please enter a message."}), 400

    # Fix 17: Input length cap.
    if len(user_message) > 500:
        return jsonify({"response": "Your message is too long. Please keep it under 500 characters."}), 400

    context = session.get("chat_context", {})
    if "onboarding_step" not in context and "user_name" not in context:
        context = {"onboarding_step": "name"}

    # Clear one-shot flags written by the previous turn.
    context.pop("needs_review", None)
    context.pop("raw_confidence", None)

    reply, updated_context = _get_reply(user_message, context)

    session["chat_context"] = updated_context

    intent = updated_context.get("last_intent") or "fallback/clarify"
    conf   = updated_context.get("last_confidence", 0.0)
    sid    = session.get("sid", "?")

    # Fix 14: Update analytics counters.
    _stats["total"] += 1
    _stats["intents"][intent] += 1
    if intent in ("fallback/clarify", "out_of_scope"):
        _stats["fallbacks"] += 1
    if conf > 0:
        _stats["confidence_sum"] += conf

    # Fix 15: Persist low-confidence turns for human review.
    if updated_context.get("needs_review") and updated_context.get("raw_confidence", 1.0) < 0.45:
        _save_review(sid, user_message, updated_context["raw_confidence"], intent)

    _log.info(
        "[%s] USER: %r  |  INTENT: %s  |  CONF: %.0f%%  |  REPLY: %r",
        sid, user_message, intent, conf * 100, reply[:80],
    )

    chips = _CHIPS.get(intent, _CHIPS["out_of_scope"])
    media = RICH_MEDIA.get(intent, {})   # Fix 7: rich media links per intent
    return jsonify({"response": reply, "chips": chips, "media": media})


# =============================================================================
# ROUTE: /reset
# =============================================================================
@app.route("/reset", methods=["POST"])
def reset():
    _ml.reset()
    # Restart onboarding so the bot re-introduces itself on a fresh session.
    session["chat_context"] = {"onboarding_step": "name"}
    _log.info("[%s] Session reset", session.get("sid", "?"))
    return jsonify({
        "status": "Session reset.",
        "greeting": "Hello again! May I ask your name so we can start fresh?",
    })


# =============================================================================
# ROUTE: /feedback  — Fix 12: log thumbs-up / thumbs-down votes
# =============================================================================
@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json(force=True)
    vote = data.get("vote", "?")          # "up" or "down"
    text = data.get("text", "")
    sid  = session.get("sid", "?")
    _log.info("[%s] FEEDBACK: %s  |  TEXT: %r", sid, vote, text[:80])
    return jsonify({"status": "ok"})


# =============================================================================
# ROUTES: /admin  — Fix 14/15: analytics dashboard + review queue
# =============================================================================
@app.route("/admin")
def admin():
    return render_template("admin.html")


@app.route("/admin/stats")
def admin_stats():
    total    = _stats["total"]
    fallbacks = _stats["fallbacks"]
    avg_conf = (_stats["confidence_sum"] / total * 100) if total else 0
    top_intents = [
        {"intent": k, "count": v}
        for k, v in sorted(_stats["intents"].items(), key=lambda x: -x[1])
    ]
    return jsonify({
        "total":          total,
        "fallbacks":      fallbacks,
        "fallback_rate":  round(fallbacks / total * 100, 1) if total else 0,
        "avg_confidence": round(avg_conf, 1),
        "top_intents":    top_intents,
    })


@app.route("/admin/review")
def admin_review():
    """Fix 15: Return the last 50 low-confidence turns from the review queue."""
    try:
        with open(REVIEW_QUEUE_PATH, encoding="utf-8") as f:
            lines = f.readlines()
        entries = [json.loads(ln) for ln in lines if ln.strip()]
        return jsonify({"total": len(entries), "entries": entries[-50:]})
    except FileNotFoundError:
        return jsonify({"total": 0, "entries": []})


# =============================================================================
# ROUTE: /logs  — returns the last 100 log lines as JSON (for testing)
# =============================================================================
@app.route("/logs")
def get_logs():
    try:
        with open("logs/chat.log", encoding="utf-8") as f:
            lines = f.readlines()
        return jsonify({"total": len(lines), "lines": lines[-100:]})
    except FileNotFoundError:
        return jsonify({"total": 0, "lines": []})


# =============================================================================
# ENTRY POINT
# =============================================================================
# Railway injects the PORT environment variable automatically.
# host="0.0.0.0" is required so Railway's router can reach the process.
# debug must be False in production.
# =============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
