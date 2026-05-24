# =============================================================================
# app.py
# =============================================================================
# This is the ENTRY POINT — the file you run to start the chatbot website.
# It uses Flask, a Python library for building web servers.
#
# WHAT IS A WEB SERVER?
#   When you open http://127.0.0.1:5000 in your browser, the browser is
#   asking this server for content.  Flask receives that request, figures out
#   what to send back (an HTML page, a JSON reply, etc.), and sends it.
#
# HOW THE CHAT WORKS END-TO-END:
#   1. User opens http://127.0.0.1:5000 in a browser
#   2. Flask serves templates/index.html  (the chat page)
#   3. User types a message and clicks Send
#   4. JavaScript in index.html sends a POST request to /chat
#   5. Flask receives it, passes the text to engine.respond()
#   6. MLEngine processes the text and returns a reply string
#   7. Flask wraps the reply in JSON and sends it back to the browser
#   8. JavaScript displays the reply in the chat box
#
# CONNECTIONS:
#   → Serves templates/index.html  (the visual chat UI)
#   → Calls engine.respond()       (engine/ml_engine.py)
#   → Calls engine.reset()         (engine/ml_engine.py)
# =============================================================================

# Flask    : the web framework
# jsonify  : converts a Python dict to a JSON response  {"key": "value"}
# render_template : loads an HTML file from the templates/ folder
# request  : lets us read what the browser sent us
from flask import Flask, jsonify, render_template, request

# ── ENGINE SELECTION ─────────────────────────────────────────────────────────
# The "engine" is the brain of the chatbot. You can swap between:
#   MLEngine        → smart neural-network engine (default)
#   RuleBasedEngine → simple keyword engine (backup)
#
# To switch, comment/uncomment the two pairs of lines below.
# ─────────────────────────────────────────────────────────────────────────────
from engine.ml_engine import MLEngine
engine = MLEngine()   # MLEngine auto-loads the saved model from model/

# To switch back to the rule-based engine, comment the two lines above
# and uncomment these two lines:
# from engine.chat_engine import RuleBasedEngine
# engine = RuleBasedEngine()
# ─────────────────────────────────────────────────────────────────────────────

# Create the Flask application object.
# __name__ tells Flask where to look for templates and static files.
# (Flask looks for a 'templates/' folder next to this file.)
app = Flask(__name__)


# =============================================================================
# ROUTE: /   (the homepage)
# =============================================================================
# A "route" is a URL path that Flask listens for.
# @app.route("/") means: "when the browser asks for the homepage, run index()".
# render_template("index.html") reads templates/index.html and sends it back.
# =============================================================================
@app.route("/")
def index():
    return render_template("index.html")


# =============================================================================
# ROUTE: /chat   (the chat API endpoint)
# =============================================================================
# methods=["POST"] means this route only accepts POST requests.
# GET requests (normal browser navigation) would return a 405 error.
# JavaScript in index.html sends a POST to this URL with JSON like:
#   { "message": "What is the Samphor" }
# Flask returns JSON like:
#   { "response": "The Samphor is a barrel-shaped drum..." }
# =============================================================================
@app.route("/chat", methods=["POST"])
def chat():
    # request.get_json() parses the JSON body the browser sent.
    # force=True means "try to parse as JSON even if the Content-Type header
    # is missing" — makes the API more forgiving.
    data = request.get_json(force=True)

    # .get("message", "") safely reads the "message" key.
    # If the key is missing, it returns "" instead of crashing.
    # .strip() removes any leading/trailing whitespace.
    user_message = data.get("message", "").strip()

    # If the user sent an empty message, return an error response.
    # HTTP status 400 = "Bad Request" (the client sent something invalid).
    if not user_message:
        return jsonify({"response": "Please enter a message."}), 400

    # Pass the message to the engine and get a reply string back.
    # This is where all the ML magic happens (inside ml_engine.py).
    reply = engine.respond(user_message)

    # Wrap the reply in a JSON object and send it back to the browser.
    # jsonify({"response": reply}) → HTTP 200 with body: {"response": "..."}
    return jsonify({"response": reply})


# =============================================================================
# ROUTE: /reset   (clear the conversation history)
# =============================================================================
# Called when the user clicks the ↺ reset button in the chat UI.
# engine.reset() clears the _history list in MLEngine.
# =============================================================================
@app.route("/reset", methods=["POST"])
def reset():
    engine.reset()
    return jsonify({"status": "Session reset."})


# =============================================================================
# ENTRY POINT
# =============================================================================
# This block only runs when you execute this file directly:
#   python app.py
# It does NOT run if another file imports app.py (e.g. in testing).
#
# debug=True enables:
#   - Auto-reload: Flask restarts whenever you save a .py file
#   - The Werkzeug debugger: shows a detailed error page in the browser
#     if the server crashes (never use debug=True in production!)
# =============================================================================
if __name__ == "__main__":
    app.run(debug=True)
