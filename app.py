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
import urllib.parse
import urllib.request
import uuid
from datetime import datetime

from flask import Flask, jsonify, render_template, request, send_from_directory, session

from engine.chat_engine import RuleBasedEngine
from engine.ml_engine import MLEngine
from knowledge.samphor_kb import RICH_MEDIA

# =============================================================================
# ENGINE SETUP
# =============================================================================
_ml   = MLEngine()
_rule = RuleBasedEngine()

# Exact chip-text → intent map.  When the user message exactly matches a chip
# we served, bypass the ML model entirely to prevent misclassification
# (e.g. "How is it played?" was mapping to ask_ceremonies at 79% confidence).
_CHIP_INTENT_MAP: dict[str, str] = {
    "What is the Samphor?":              "ask_definition",
    "What is the Samphor drum?":         "ask_definition",
    "What does it look like?":           "ask_shape",
    "What is it made of?":               "ask_material",
    "How is it played?":                 "ask_playing",
    "How is the Samphor played?":        "ask_playing",
    "How is it tuned?":                  "ask_tuning",
    "Where is it used?":                 "ask_ceremonies",
    "What ceremonies use the Samphor?":  "ask_ceremonies",
    "When did it originate?":            "ask_history",
    "What is the history of the Samphor?": "ask_history",
    "How is it preserved?":              "ask_preservation",
    "What is the Pinpeat ensemble?":     "ask_pinpeat",
    "What is the Pinpeat?":              "ask_pinpeat",
    "How do I learn to play it?":        "ask_learning",
    "How do I learn to play?":           "ask_learning",
    "How does it compare to other drums?": "ask_compare",
    "Tell me about Angkor and the Samphor": "ask_angkor",
    "What is Angkor Wat?":               "ask_angkor",
    "What is Khmer?":                    "ask_khmer",
    "What is the Roneat?":               "ask_roneat",
    "What is the Sralai?":               "ask_sralai",
    "What is the Chhing?":               "ask_chhing",
    "What is Robam Kbach Boran?":        "ask_robam",
    "What is Sbek Thom?":                "ask_sbek_thom",
    "What is RUFA?":                     "ask_rufa",
}

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
    "ask_roneat":       ["What is the Pinpeat ensemble?", "What is the Sralai?", "What is the Chhing?", "How is the Samphor played?"],
    "ask_khmer":        ["What is Angkor Wat?", "What is the Samphor?", "What is the Pinpeat ensemble?", "When did it originate?"],
    "ask_angkor":       ["What is the history of the Samphor?", "What is Khmer?", "What is the Bayon temple?", "When did it originate?"],
    "ask_sralai":       ["What is the Pinpeat ensemble?", "What is the Roneat?", "What is the Chhing?", "How is the Samphor played?"],
    "ask_chhing":       ["What is the Pinpeat ensemble?", "What is the Roneat?", "What is the Sralai?", "How is the Samphor tuned?"],
    "ask_skor_thom":    ["What is the Pinpeat ensemble?", "How is the Samphor played?", "What is the Chhing?", "What is the Samphor?"],
    "ask_sbek_thom":    ["What is Robam Kbach Boran?", "What is the Pinpeat ensemble?", "What is the Samphor?", "How is it preserved?"],
    "ask_robam":        ["What is Sbek Thom?", "What is the Pinpeat ensemble?", "What ceremonies use the Samphor?", "How is it preserved?"],
    "ask_rufa":         ["How is the Samphor tradition being preserved?", "How do I learn to play the Samphor?", "What happened after the Khmer Rouge?", "What is RUFA?"],
    "ask_kroeung":      ["How is the Samphor tuned?", "What is the Samphor made of?", "What is the black paste on the Samphor?", "How is it played?"],
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


# Reverse map: chip text → intent (used for answered-intent deduplication).
_CHIP_TO_INTENT: dict[str, str] = {
    chip: intent
    for intent, chips in _CHIPS.items()
    for chip in chips
}
# Merge explicit chip-intent overrides into the reverse map.
_CHIP_TO_INTENT.update(_CHIP_INTENT_MAP)


def _get_reply(user_message: str, context: dict) -> tuple[str, dict]:
    """Route a user message through MLEngine, fall back to RuleBasedEngine.

    Exact chip-text matches bypass the ML model to prevent misclassification
    (e.g. "How is it played?" was mapping to ask_ceremonies at 79% conf).
    """
    from knowledge.samphor_kb import INTENT_RESPONSES
    import random as _random

    # Only apply chip shortcut after onboarding is done.
    if context.get("onboarding_step", "done") == "done":
        chip_intent = _CHIP_INTENT_MAP.get(user_message.strip())
        if chip_intent:
            responses = INTENT_RESPONSES.get(chip_intent, [])
            if responses:
                reply = _random.choice(responses)
                updated = {
                    **context,
                    "last_intent":      chip_intent,
                    "last_response":    reply,
                    "last_confidence":  1.0,
                }
                return reply, updated

    if _ml._st_model is not None:
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


@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory("images", filename)


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
    # Only trigger onboarding if the frontend intro hasn't already set user_name.
    if "onboarding_step" not in context and "user_name" not in context:
        context = {"onboarding_step": "name"}

    # Clear one-shot flags written by the previous turn.
    context.pop("needs_review", None)
    context.pop("raw_confidence", None)

    reply, updated_context = _get_reply(user_message, context)

    intent = updated_context.get("last_intent") or "fallback/clarify"
    conf   = updated_context.get("last_confidence", 0.0)
    sid    = session.get("sid", "?")

    # Deduplicate chips: filter out intents the user has already seen this session.
    answered = set(updated_context.get("answered_intents", []))
    answered.add(intent)
    updated_context["answered_intents"] = list(answered)

    session["chat_context"] = updated_context

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

    raw_chips = _CHIPS.get(intent, _CHIPS["out_of_scope"])
    # Remove chips whose intent has already been answered this session.
    fresh_chips = [c for c in raw_chips if _CHIP_TO_INTENT.get(c, "") not in answered]
    chips = fresh_chips if fresh_chips else raw_chips   # fallback to all if exhausted

    media = RICH_MEDIA.get(intent, {})

    return jsonify({
        "response":   reply,
        "chips":      chips,
        "media":      media,
        "intent":     intent,
        "confidence": round(conf * 100, 1),
    })


# =============================================================================
# ROUTE: /user-context  — called by the HTML intro when it completes,
#                         so the ML engine skips its own onboarding flow
#                         and the first real question is never hijacked.
# =============================================================================
@app.route("/user-context", methods=["POST"])
def user_context():
    data       = request.get_json(force=True)
    name       = data.get("name", "").strip()
    occupation = data.get("occupation", "").strip()
    ctx        = session.get("chat_context", {})
    ctx["onboarding_step"] = "done"
    if name:
        ctx["user_name"]       = name
    if occupation:
        ctx["user_occupation"] = occupation
    session["chat_context"] = ctx
    _log.info(
        "[%s] User context pre-set: name=%r, occupation=%r",
        session.get("sid", "?"), name, occupation,
    )
    return jsonify({"status": "ok"})


# =============================================================================
# ROUTE: /reset
# =============================================================================
@app.route("/reset", methods=["POST"])
def reset():
    _ml.reset()
    # Keep name/occupation from the intro but reset everything else.
    old_ctx  = session.get("chat_context", {})
    new_ctx  = {"onboarding_step": "done"}
    if old_ctx.get("user_name"):
        new_ctx["user_name"]       = old_ctx["user_name"]
    if old_ctx.get("user_occupation"):
        new_ctx["user_occupation"] = old_ctx["user_occupation"]
    session["chat_context"] = new_ctx
    _log.info("[%s] Session reset", session.get("sid", "?"))
    return jsonify({"status": "Session reset."})


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
# ROUTE: /wiki-image  — proxy to Wikipedia REST API to fetch page thumbnails
# =============================================================================
@app.route("/wiki-image")
def wiki_image():
    wiki_url = request.args.get("url", "")
    if "wikipedia.org/wiki/" not in wiki_url:
        return jsonify({"image_url": None})
    title = wiki_url.split("/wiki/")[-1]
    api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
    try:
        req = urllib.request.Request(
            api_url,
            headers={"User-Agent": "SamphorExpertBot/1.0 (educational project)"},
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read())
        image_url = data.get("thumbnail", {}).get("source")
        _log.info("[wiki-image] title=%r  image=%s", data.get("title", ""), image_url or "none")
        return jsonify({"image_url": image_url, "title": data.get("title", "")})
    except Exception:
        return jsonify({"image_url": None})


# =============================================================================
# ROUTE: /save-history  — saves chat history to logs/ directory
# =============================================================================
@app.route("/save-history", methods=["POST"])
def save_history():
    data  = request.get_json(force=True)
    lines = data.get("lines", [])
    ts    = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"chat_history_{ts}.txt"
    path     = os.path.join("logs", filename)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        sid = session.get("sid", "?")
        _log.info("[%s] Chat history saved to %s", sid, filename)
        return jsonify({"status": "ok", "filename": filename})
    except OSError as e:
        return jsonify({"status": "error", "message": str(e)}), 500


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
