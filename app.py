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

import logging
import os
import uuid

from flask import Flask, jsonify, render_template, request, session

from engine.chat_engine import RuleBasedEngine
from engine.ml_engine import MLEngine

# =============================================================================
# ENGINE SETUP
# =============================================================================
# Both engines are created once at startup (model loading is expensive).
# _ml is the primary engine; _rule is the fallback used when the ML model
# is missing or raises an unexpected exception.
# =============================================================================
_ml   = MLEngine()
_rule = RuleBasedEngine()


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
    data         = request.get_json(force=True)
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Please enter a message."}), 400

    # Restore this user's context.
    # If onboarding_step is missing (brand-new session OR old session cookie
    # from before the onboarding feature was added) AND user_name is also absent,
    # restart from the beginning of the onboarding flow.
    context = session.get("chat_context", {})
    if "onboarding_step" not in context and "user_name" not in context:
        context = {"onboarding_step": "name"}

    reply, updated_context = _get_reply(user_message, context)

    # Persist updated context back into the session cookie.
    session["chat_context"] = updated_context

    # Log the turn.
    intent = updated_context.get("last_intent") or "fallback/clarify"
    conf   = updated_context.get("last_confidence", 0.0)
    sid    = session.get("sid", "?")
    _log.info(
        "[%s] USER: %r  |  INTENT: %s  |  CONF: %.0f%%  |  REPLY: %r",
        sid, user_message, intent, conf * 100, reply[:80],
    )

    return jsonify({"response": reply})


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
