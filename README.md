# Samphor Expert System — Complete Learning Guide

An AI-powered chatbot that answers questions about the **Samphor**, a traditional Cambodian barrel drum.
The system uses a **sentence transformer** model to understand the semantic meaning of your questions,
then looks up a human-written answer from a knowledge base.

This README is written to be a **complete learning companion** — every concept is explained both in
**technical terms** (what it actually is) and in **plain everyday language** (what it means to you).
A full learning roadmap is at the end.

---

## Table of Contents

1. [What This Project Does — The Big Picture](#1-what-this-project-does--the-big-picture)
2. [How All the Files Talk to Each Other](#2-how-all-the-files-talk-to-each-other)
3. [The Complete File Guide](#3-the-complete-file-guide)
   - [app.py — The Web Server](#apppy--the-web-server)
   - [train.py — The Training Script](#trainpy--the-training-script)
   - [engine/chat_engine.py — The Blueprint](#enginechat_enginepy--the-blueprint)
   - [engine/ml_engine.py — The AI Brain](#engineml_enginepy--the-ai-brain)
   - [knowledge/samphor_kb.py — The Answer Book](#knowledgesamphor_kbpy--the-answer-book)
   - [data/intents.json — The Training Data](#dataintentsjson--the-training-data)
   - [templates/index.html — The Chat Interface](#templatesindexhtml--the-chat-interface)
   - [templates/admin.html — The Analytics Dashboard](#templatesadminhtml--the-analytics-dashboard)
   - [requirements.txt — The Dependency List](#requirementstxt--the-dependency-list)
   - [model/ — The Saved Embedding Files](#model--the-saved-embedding-files)
4. [Deep Concept Explanations](#4-deep-concept-explanations)
   - [Python Virtual Environments](#python-virtual-environments)
   - [Object-Oriented Programming and Inheritance](#object-oriented-programming-and-inheritance)
   - [Abstract Base Classes](#abstract-base-classes)
   - [The Flask Web Framework and HTTP](#the-flask-web-framework-and-http)
   - [Natural Language Processing (NLP)](#natural-language-processing-nlp)
   - [Sentence Transformers and Embeddings](#sentence-transformers-and-embeddings)
   - [Cosine Similarity](#cosine-similarity)
   - [Semantic Search vs Keyword Matching](#semantic-search-vs-keyword-matching)
   - [Pickle and Model Serialization](#pickle-and-model-serialization)
   - [JSON and Data Exchange](#json-and-data-exchange)
   - [AJAX and the fetch() API](#ajax-and-the-fetch-api)
   - [Async/Await in JavaScript](#asyncawait-in-javascript)
   - [CSS Flexbox Layout](#css-flexbox-layout)
   - [localStorage and Session Persistence](#localstorage-and-session-persistence)
   - [Web Speech API](#web-speech-api)
5. [Setup — Step by Step](#5-setup--step-by-step)
6. [How to Run](#6-how-to-run)
7. [All 17 Improvements Made](#7-all-17-improvements-made)
8. [How to Add a New Topic](#8-how-to-add-a-new-topic)
9. [Switching Engines](#9-switching-engines)
10. [Learning Roadmap](#10-learning-roadmap)
11. [Glossary](#11-glossary)

---

## 1. What This Project Does — The Big Picture

```
You type a question  →  The AI figures out what category it is  →  It sends back a reply
"What is the Samphor?"    "ask_definition" (92% confident)         "The Samphor is a barrel-shaped..."
```

**Technical explanation:**
The system is a multi-turn intent-classification chatbot. User input is encoded into a dense
semantic vector by a pre-trained sentence transformer (`paraphrase-multilingual-MiniLM-L12-v2`).
Cosine similarity between the query embedding and all pre-computed training pattern embeddings
identifies the best-matching intent. The predicted intent key is used to retrieve a response from
a hand-authored knowledge base. The system also supports Khmer-script input, multi-turn context,
spell correction, rate limiting, and an admin analytics dashboard.

**Plain English:**
You type a question. The program doesn't search Google — it reads your words and converts them
into a list of numbers that captures the *meaning* (not just the exact words). It then compares
that meaning against meanings of all the training examples it learned from, and picks the most
similar one. It returns a pre-written answer from that category. All the knowledge came from
files you wrote, not the internet.

---

**Why separate training from chatting?**

**Technical:** During training, the sentence transformer encodes all ~460 training patterns once
and saves the resulting embedding matrix. At inference time only a single query embedding and a
cosine similarity comparison are needed — this is very fast.

**Plain English:** Computing the meaning vectors for hundreds of training examples takes a few
seconds. Once saved, answering any question takes milliseconds. So we do the heavy computation
once, save the results, and the chatbot just uses the saved numbers.

---

## 2. How All the Files Talk to Each Other

### During a Chat (every message you send)

```
Browser (templates/index.html)
       │
       │  JavaScript fetch() — POST /chat  {"message": "What is the Samphor?"}
       ▼
app.py  (Flask web server — the traffic director)
       │  1. Check rate limit (30 requests/minute per IP)
       │  2. Apply spell correction
       │  3. Check Flask session for context (multi-turn)
       │  4. engine.respond(message, context)
       ▼
engine/ml_engine.py  (the AI brain)
       │  1. Encode query → 384-dim embedding vector
       │  2. Cosine similarity vs all 460 training embeddings
       │  3. Pick highest-confidence intent per category
       │  4. Return intent + confidence + context
       │
       ├──────────────────────────────────────────────────────┐
       ▼                                                       ▼
model/embeddings.npy                            knowledge/samphor_kb.py
(460 pre-computed embedding vectors)            (INTENT_RESPONSES, KHMER_RESPONSES, RICH_MEDIA)
       │                                                       │
       │  returns "ask_definition" at 92% confidence          │  returns 1 random answer string
       └──────────────────────────┬────────────────────────────┘
                                  ▼
                             app.py
                                  │  JSON: {"response": "...", "chips": [...], "media": {...}}
                                  ▼
                        Browser displays reply, quick-reply chips, and media links
```

### During Training (only run once with `train.py`)

```
data/intents.json
       │  14 intents × ~33 patterns = ~460 training examples (English + Khmer)
       ▼
engine/ml_engine.py  train()
       │  Step 1: Load all patterns and their intent tags from intents.json
       │  Step 2: Load sentence transformer model (downloads once from Hugging Face)
       │  Step 3: Encode ALL patterns → one 384-dim vector per pattern
       │  Step 4: Save embedding matrix and label list to disk
       ▼
model/embeddings.npy   (460 × 384 float32 matrix — the encoded training patterns)
model/labels.pkl       (list of 460 intent tag strings, one per row in embeddings.npy)
```

---

## 3. The Complete File Guide

### `app.py` — The Web Server

**Technical explanation:**
`app.py` is a Flask WSGI application. It registers URL routes using Python decorators.
The root route (`/`) renders a Jinja2 template. The `/chat` route accepts JSON POST requests,
applies rate limiting (30 req/min per IP), spell correction, and session-based context management,
calls `engine.respond()`, and returns a JSON response including reply text, quick-reply chips,
and rich media links. Additional routes handle user feedback (`/feedback`), session reset (`/reset`),
and an analytics dashboard (`/admin`, `/admin/stats`, `/admin/review`).

**Plain English:**
`app.py` is like the front door of a restaurant. When you walk in (visit the website), it gives
you the menu (the chat page). When you order (send a message), it takes your order to the kitchen
(the AI engine) and brings back the food (the reply). It also logs what people ordered most
(analytics) and politely asks if you enjoyed it (feedback buttons).

**Key routes:**

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Serve the chat UI |
| `/chat` | POST | Process a chat message, return reply + chips + media |
| `/reset` | POST | Clear the user's conversation session |
| `/feedback` | POST | Record thumbs-up/down on a bot reply |
| `/admin` | GET | Serve the analytics dashboard HTML |
| `/admin/stats` | GET | JSON: total messages, intent breakdown, fallback rate |
| `/admin/review` | GET | JSON: low-confidence turns queued for human review |

**Rate limiting:**
Each IP address is limited to 30 messages per minute. The limiter uses a sliding window stored
in a Python dict (`_rate_limit`). If the limit is exceeded, the server returns HTTP 429.
Messages longer than 500 characters are rejected with HTTP 400.

**Session handling (Fix 1):**
Each browser tab gets its own Flask session (cookie-based). Conversation context (last intent,
clarification state) is stored per-session so different browser tabs cannot interfere with each other.

---

### `train.py` — The Training Script

**Technical explanation:**
A standalone script that instantiates `MLEngine`, calls its `train()` method (which encodes all
training patterns with the sentence transformer), and prints a formatted summary. The sentence
transformer pipeline returns `(1.0, 0.0)` as sentinel values since cosine similarity models
do not expose traditional loss/accuracy metrics during training.

**Plain English:**
This is the "preparation" script. You run it once. It reads all the example questions, converts
them into meaning vectors using the sentence transformer, and saves those vectors to the `model/`
folder. After that, the chatbot can compare any new question against those saved vectors instantly.

**Key parts:**

```python
engine = MLEngine()
accuracy, loss = engine.train()   # returns (1.0, 0.0) as sentinels
```

The `(1.0, 0.0)` sentinel tells `train.py` that a sentence transformer was used (no numeric
loss to report). Training output says "PASS — embeddings generated successfully."

---

### `engine/chat_engine.py` — The Blueprint

**Technical explanation:**
`chat_engine.py` defines two classes. First, `ChatEngine` is an Abstract Base Class (ABC)
using Python's `abc` module. It declares `respond()` and `reset()` as abstract methods using
`@abstractmethod`, which forces all subclasses to implement them. This enforces a consistent
interface across engine implementations.

Second, `RuleBasedEngine` is a concrete subclass of `ChatEngine`. It stores a `RULES`
dictionary mapping tuples of keyword strings to response strings. Its `respond()` method
lowercases the input and iterates over `RULES` looking for a keyword match. It is used as an
automatic fallback if `MLEngine` fails to load (Fix 4).

**Plain English:**
Imagine a job listing that says "all employees must be able to answer questions and reset
themselves." `ChatEngine` is that job listing. It doesn't say HOW to do those things, just
that they MUST be done. Any engine you build must follow the rules, or Python will refuse to run.

`RuleBasedEngine` is the "dumb" backup. It doesn't use any AI — it just checks: "does your
message contain the word 'history'? Yes? Here's the history answer." Very simple, very reliable.

**The `RULES` dictionary:**

```python
RULES = {
    ("what is", "define", "meaning", "describe", "tell me about"): "The Samphor is...",
    ("history", "origin", "ancient", "old", "invented"): "The Samphor has roots...",
    ("hello", "hi", "hey", "greetings"): "Hello! Ask me anything...",
    ...
}
```

The key is a **tuple** (a group) of trigger words. If ANY of them appear in the user's message,
the corresponding response is returned.

---

### `engine/ml_engine.py` — The AI Brain

**Technical explanation:**
`MLEngine` extends `ChatEngine` and implements a sentence-transformer-based intent classifier.
At training time it loads `paraphrase-multilingual-MiniLM-L12-v2` from the `sentence-transformers`
library, encodes all training patterns to 384-dimensional embeddings with L2 normalization, and
saves the matrix (`embeddings.npy`) and label list (`labels.pkl`) via NumPy and pickle.

At inference time it encodes the user query, computes cosine similarity against the stored
embedding matrix using `sklearn.metrics.pairwise.cosine_similarity`, aggregates per-intent
maximum scores, and returns the best intent when its score exceeds `CONFIDENCE_THRESHOLD = 0.52`.

Additional capabilities: Khmer-script detection (`_is_khmer()`), domain-term spell correction
(`_correct_spelling()` via `difflib.get_close_matches`), multi-turn follow-up detection
("tell me more"), pronoun resolution ("what is it made of?"), and grey-zone clarification
questions (Fix 3, when confidence is 0.35–0.52).

**Plain English:**
This is the smart engine. It has two jobs:

1. **Preparing** (during `train.py`): It reads all example questions, converts each one into a
   list of 384 numbers that captures the meaning of the sentence, and saves all those lists.

2. **Answering questions** (during chatting): When you type something, it converts your question
   to 384 numbers too, then measures how "close" your numbers are to each training example.
   The closest match wins. If no training example is close enough, it asks for clarification or
   returns a fallback.

**The Training Pipeline in detail:**

**Step 1 — Load intents:**
```python
with open("data/intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)
```
Reads ~460 training patterns across 14 intents, including Khmer-script patterns.

**Step 2 — Encode all patterns:**
```python
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(all_patterns, normalize_embeddings=True)
```
Each pattern becomes a 384-number vector. `normalize_embeddings=True` ensures all vectors
have length 1, which makes cosine similarity equivalent to a dot product (faster math).

**Step 3 — Save artifacts:**
```python
np.save("model/embeddings.npy", embeddings)
pickle.dump(labels, open("model/labels.pkl", "wb"))
```
The embedding matrix (shape: `[460, 384]`) and the list of 460 intent tags are saved.
These are the only two files the chatbot needs at runtime.

**The Inference Pipeline in detail (every chat message):**

```python
def _predict(self, text: str) -> tuple[str, float, str | None, float]:
    query_emb = self._st_model.encode([text], normalize_embeddings=True)
    sims = cosine_similarity(query_emb, self._embeddings)[0]

    # aggregate: best score per intent
    intent_best: dict[str, float] = {}
    for idx, label in enumerate(self._labels):
        score = float(sims[idx])
        if label not in intent_best or score > intent_best[label]:
            intent_best[label] = score

    sorted_intents = sorted(intent_best.items(), key=lambda x: -x[1])
    best_tag, best_conf = sorted_intents[0]
    second_tag, second_conf = sorted_intents[1] if len(sorted_intents) > 1 else (None, 0.0)
    return best_tag, best_conf, second_tag, second_conf
```

Line by line:
1. Encode the user query → 1 × 384 embedding.
2. Cosine similarity against all 460 training embeddings → 460 scores.
3. For each intent, keep the highest score among its patterns.
4. Sort intents by score. Return top-2 for clarification logic.

**Spell correction (Fix 9):**
```python
@staticmethod
def _correct_spelling(text: str) -> str:
    domain_terms = ["samphor", "pinpeat", "khmer", ...]
    corrected = []
    for word in text.split():
        matches = get_close_matches(word.lower(), domain_terms, n=1, cutoff=0.75)
        corrected.append(matches[0] if matches else word)
    return " ".join(corrected)
```
Uses `difflib.get_close_matches` to fix typos in Samphor-related domain words (e.g.,
"Sanphor" → "Samphor", "Phinpeat" → "Pinpeat") without any extra packages.

**Khmer detection (Fix 13):**
```python
@staticmethod
def _is_khmer(text: str) -> bool:
    return any('ក' <= c <= '៿' for c in text)
```
Checks if any character falls in the Khmer Unicode block (U+1780–U+17FF). If true,
`respond()` serves replies from `KHMER_RESPONSES` instead of `INTENT_RESPONSES`.

**Multi-turn context (Fix 2):**
The `respond()` method inspects the `context` dict for follow-up signals:
- "tell me more" / "continue" → re-use `last_intent` from the previous turn.
- Pronoun references ("what is it made of?") → resolve "it" to `last_intent`.
- Grey-zone confidence (0.35–0.52): ask "Did you mean X or Y?" (Fix 3).
- Low confidence (<0.35): save to the active learning queue (Fix 15).

---

### `knowledge/samphor_kb.py` — The Answer Book

**Technical explanation:**
A module containing four module-level constants:
- `INTENT_RESPONSES` — `dict[str, list[str]]` mapping intent tags to English response strings.
- `KHMER_RESPONSES` — `dict[str, list[str]]` mapping intent tags to Khmer-script responses.
- `FALLBACK_RESPONSE` — a plain `str` returned when confidence is too low.
- `RICH_MEDIA` — `dict[str, dict]` mapping intent tags to Wikipedia/external links shown in the UI.

**Plain English:**
This is literally a lookup table. The AI figures out the category ("ask_history"),
and this file is where the actual text answers live. The engine picks one answer at random
from the list so conversations feel slightly varied. Khmer users get Khmer replies automatically.
Wikipedia links appear as clickable buttons under each bot reply.

**Structure:**
```python
INTENT_RESPONSES: dict[str, list[str]] = {
    "ask_definition": [
        "The Samphor is a barrel-shaped, two-headed drum...",
        "A Samphor is a traditional Khmer percussion instrument...",
    ],
    ...
}

KHMER_RESPONSES: dict[str, list[str]] = {
    "ask_definition": [
        "សំភោ គឺជាស្គរប្រពៃណីខ្មែរ...",
        ...
    ],
    ...
}

RICH_MEDIA: dict[str, dict] = {
    "ask_definition": {
        "links": [
            {"label": "Wikipedia: Samphor", "url": "https://en.wikipedia.org/wiki/Samphor"},
        ]
    },
    ...
}

FALLBACK_RESPONSE: str = (
    "I'm not sure I understand. Could you rephrase your question about the Samphor?"
)
```

---

### `data/intents.json` — The Training Data

**Technical explanation:**
A JSON file containing an array of intent objects. Each object has three fields: `"tag"` (str),
`"patterns"` (list of str — the training examples), and `"responses"` (list of str — unused
by the engine but kept for documentation). The file contains ~460 patterns across 14 intents,
including English patterns (20 original + 10 additional) and Khmer-script patterns (4 per intent).

**Plain English:**
This is the textbook the AI studies from. Each "intent" is a chapter, and each "pattern" is one
way a human might phrase a question about that topic. More patterns = better recognition.
Khmer patterns let native Cambodian speakers ask in their own language.

**The 14 intents and what they cover:**

| Intent Tag | Topic | Example Pattern |
|-----------|-------|----------------|
| `ask_definition` | What is the Samphor? | "What is the Samphor?", "ស័មភោ គឺ ជា អ្វី?" |
| `ask_history` | Origins and history | "When was it invented?", "How old is the Samphor?" |
| `ask_material` | What it's made of | "What wood is used?", "What is the drum skin made from?" |
| `ask_shape` | Physical dimensions | "What shape is the Samphor?", "How big is it?" |
| `ask_playing` | How to play it | "How do you play the Samphor?", "What technique is used?" |
| `ask_tuning` | How it is tuned | "How is the Samphor tuned?", "What is the black paste?" |
| `ask_ceremonies` | Ceremonial uses | "When is it played?", "What rituals use the Samphor?" |
| `ask_pinpeat` | The Pinpeat orchestra | "What is the Pinpeat ensemble?", "What orchestra uses it?" |
| `ask_compare` | Comparison to other drums | "How is it different from the tabla?", "Compare to taiko" |
| `ask_learning` | How to learn it | "Where can I learn the Samphor?", "How long does it take?" |
| `ask_preservation` | Cultural preservation | "Is the Samphor endangered?", "UNESCO heritage?" |
| `greeting` | Hello messages | "Hi", "Hello", "ជំរាប សួរ" |
| `farewell` | Goodbye messages | "Bye", "See you", "លា ហើយ" |
| `out_of_scope` | Unrelated questions | "What's the weather?", "Tell me a joke" |

**Why ~33 patterns per intent (up from ~20)?**
Synonym expansion (Fix 11) was applied — each original pattern was augmented with rephrased
variants to improve robustness. The sentence transformer generalizes much better than the old
Bag-of-Words approach, so extra patterns reduce edge-case misclassifications rather than
causing overfitting.

---

### `templates/index.html` — The Chat Interface

**Technical explanation:**
A single HTML document serving as a Single Page Application (SPA). It contains embedded CSS
for styling and inline JavaScript. Key features implemented in JS include:
- Fetch API for async POST to `/chat` and `/reset` / `/feedback`
- Quick-reply chip buttons rendered from `data.chips` in the server response
- Typing indicator (bouncing dots animation) while waiting for server
- Rich media links rendered from `data.media.links`
- Feedback thumbs-up/down buttons on each bot message
- `localStorage` persistence so chat survives page refresh
- Web Speech API integration for voice input

**Plain English:**
This is the web page you see in your browser. It has the chat box, input field, Send button,
mic button (for voice), and all the visual elements. When the bot replies, it automatically
shows clickable topic suggestion buttons underneath so you always know what to ask next.

**New features in the UI:**

| Feature | How it works |
|---------|-------------|
| **Typing indicator** (Fix 6) | A `<div class="typing">` with 3 animated dots appears after you send; removed when reply arrives |
| **Quick-reply chips** (Fix 5) | Server returns `chips: ["What is it made of?", ...]`; rendered as `<button class="chip">` |
| **Rich media links** (Fix 7) | Server returns `media.links: [{label, url}]`; rendered as `<a>` tags with external links |
| **Feedback buttons** (Fix 12) | Each bot message gets 👍/👎 buttons; click posts `{msg_id, vote}` to `/feedback` |
| **localStorage** (Fix 8) | On every message, full history is `JSON.stringify`-ed to `localStorage["samphor-chat-v1"]` |
| **Voice input** (Fix 16) | Mic button uses `SpeechRecognition` API; hides itself if browser doesn't support it |

**HTML structure:**

```html
<div class="chat-wrapper">
    <header>
        <h1>Samphor Expert System</h1>
        <button id="reset-btn">↺</button>
    </header>
    <div id="chat-box">
        <!-- Message bubbles injected by JavaScript -->
    </div>
    <div class="input-area">
        <button id="mic-btn">🎤</button>
        <input id="user-input" type="text" placeholder="Ask about the Samphor..."/>
        <button id="send-btn">Send</button>
    </div>
</div>
```

**Color palette:**
- Background: `#f4f1ec` (warm off-white — feels like parchment)
- Header: `#8b3a0f` (deep clay red — earthy, Cambodian feel)
- User messages: `#7a2a0a` (dark red — right side of chat)
- Bot messages: `#f0e6d3` (warm beige — left side of chat)

---

### `templates/admin.html` — The Analytics Dashboard

**Technical explanation:**
A standalone HTML page served at `/admin`. It uses `fetch()` on page load to call `/admin/stats`
and `/admin/review`, then renders the data as summary cards, an intent frequency table, and a
low-confidence review queue table. No authentication is implemented — restrict to localhost or
add Flask-Login if deploying publicly.

**Plain English:**
Visit `http://127.0.0.1:5000/admin` to see a dashboard showing: how many messages the bot
received, which topics were asked most, the fallback rate (how often the bot didn't understand),
and a queue of uncertain replies that a human can review and correct.

**Dashboard sections:**
- **Summary cards:** Total messages, fallback count, fallback rate %, average confidence
- **Intent breakdown:** Bar chart table showing how many times each intent was triggered
- **Review queue:** Table of low-confidence turns (message, predicted intent, confidence score)

---

### `requirements.txt` — The Dependency List

**Technical explanation:**
A pip-compatible requirements file specifying exact package versions for reproducible builds.
It pins all transitive dependencies to guarantee identical environments across machines.

**Plain English:**
A shopping list of Python packages. When you run `pip install -r requirements.txt`, Python
downloads and installs everything on the list. Version numbers ensure you get exactly the same
software the project was built with, which prevents version-mismatch bugs.

**The key packages and why they exist:**

| Package | Version | Why it's here |
|---------|---------|---------------|
| `flask` | 3.1.3 | The web server framework — runs `app.py` |
| `sentence-transformers` | 3.4.1 | Encodes text to semantic embeddings — the AI core |
| `numpy` | 2.4.6 | Numerical arrays — used for the embedding matrix |
| `scikit-learn` | 1.8.0 | `cosine_similarity()` used during inference |
| `jinja2` | 3.1.6 | HTML template engine — Flask uses this to serve HTML |
| `werkzeug` | 3.1.8 | The WSGI utility library Flask is built on |

**Why `sentence-transformers` instead of Keras/TensorFlow?**
The old approach used a 3-layer Dense network trained on Bag-of-Words vectors with NLTK
tokenization and lemmatization. The new approach uses a pre-trained multilingual transformer
(`paraphrase-multilingual-MiniLM-L12-v2`) that understands semantic meaning and natively
supports 50+ languages including Khmer. This gives much better accuracy on unseen phrasings
with zero additional training data. The model is ~120 MB and downloads once from Hugging Face.

---

### `model/` — The Saved Embedding Files

**Technical explanation:**
Two binary artifacts created by `train.py` and consumed by `MLEngine._load_artefacts()`.
`embeddings.npy` is a NumPy array of shape `(N, 384)` where N is the number of training
patterns (~460) and 384 is the embedding dimension of the sentence transformer.
`labels.pkl` is a Python pickle file containing a list of N intent tag strings.

**Plain English:**
These two files are the "prepared knowledge." After training, each of the 460 example questions
has been converted to 384 numbers representing its meaning. Without these files, the chatbot
can't answer anything. They are excluded from git (`.gitignore`) because they can always be
recreated by running `train.py`.

| File | Contents |
|------|---------|
| `embeddings.npy` | 460 × 384 float32 matrix — meaning vectors for all training patterns |
| `labels.pkl` | Python list of 460 intent tag strings (one per embedding row) |

---

## 4. Deep Concept Explanations

This section explains every major technology in this project — what it really is, how it works,
and why it exists.

---

### Python Virtual Environments

**Technical explanation:**
A virtual environment is an isolated Python interpreter installation in a local directory.
It maintains its own `site-packages` directory, separate from the system Python and from other
virtual environments. The `venv` module creates this directory structure and modifies the PATH
so that `python` resolves to the local interpreter. Package installations go into the local
`site-packages` and do not affect global Python.

**Plain English:**
Imagine your computer has one big kitchen (Python). Every project wants to use different
ingredients (packages) at different versions. If you put everything in one kitchen, projects
start fighting over ingredient versions. A virtual environment gives each project its own
mini-kitchen with its own separate fridge. They can't see each other's food.

**How it works:**
```
project/
└── .venv/
    ├── Scripts/        (Windows) or bin/ (Mac/Linux)
    │   ├── python.exe  ← this is the isolated Python
    │   └── pip.exe     ← this installs into the venv only
    └── Lib/
        └── site-packages/
            ├── flask/
            ├── sentence_transformers/
            └── ... (all your packages)
```

**Important (Windows):** Python paths with spaces require the `&` call operator in PowerShell:
```powershell
& "C:\path with spaces\python.exe" -m pip install flask
```

---

### Object-Oriented Programming and Inheritance

**Technical explanation:**
OOP organizes code into **classes** — blueprints that bundle data (attributes) and behaviour
(methods) together. **Instances** are concrete objects created from a class using `ClassName()`.
**Inheritance** allows a class (the subclass) to reuse and extend the behaviour of another class
(the superclass). The subclass inherits all methods and attributes and can override them.

**Plain English:**
A class is like a cookie cutter — a template. An instance is an actual cookie made from the cutter.
Inheritance means a subclass is like a specialized cutter that has everything the original had, plus extra features.

**In this project:**
```
ChatEngine (abstract superclass)
    │  Defines: respond() and reset() must exist
    │
    ├── RuleBasedEngine (subclass)
    │       Implements: respond() using keyword rules
    │       Implements: reset() by clearing history
    │
    └── MLEngine (subclass)
            Implements: respond() using sentence transformer
            Implements: reset() by clearing history
            Adds: train(), _predict(), _correct_spelling(), _is_khmer(), etc.
```

When `app.py` does `engine.respond(user_message, context)`, it doesn't care whether `engine` is
an `MLEngine` or a `RuleBasedEngine`. Both guarantee that `respond()` exists. This is called
**polymorphism** — "many forms", one interface.

---

### Abstract Base Classes

**Technical explanation:**
Python's `abc` module provides `ABC` (Abstract Base Class) and the `@abstractmethod` decorator.
A class that inherits from `ABC` and declares abstract methods cannot be instantiated directly.
Any concrete subclass must implement all abstract methods, or Python raises `TypeError` at runtime.

**Plain English:**
An abstract class is a "contract class." It says: "If you want to be a type of engine, you MUST
be able to `respond()` and `reset()`. I won't tell you how — but if you don't implement them,
Python won't let you run." This prevents bugs where someone creates a new engine but forgets to
implement a required method.

```python
from abc import ABC, abstractmethod

class ChatEngine(ABC):
    @abstractmethod
    def respond(self, user_input: str, context: dict) -> tuple[str, dict]:
        ...   # No implementation — just a declaration

    @abstractmethod
    def reset(self) -> None:
        ...
```

---

### The Flask Web Framework and HTTP

**Technical explanation:**
Flask is a WSGI (Web Server Gateway Interface) micro-framework. It maps URL patterns to Python
functions via route decorators. When the development server receives an HTTP request, it matches
the URL and method to a registered route, invokes the corresponding view function, and wraps the
return value in an HTTP response. Flask's `request` proxy object provides thread-local access to
the current request's data. `flask.session` is a client-side cookie-based session store
(signed with `SECRET_KEY`) used for per-user conversation context.

**Plain English:**
HTTP is the language web browsers and servers use to talk. Flask is the translator — you write
Python functions, and Flask handles all the HTTP plumbing to connect them to URLs.

**HTTP Request Methods:**
- **GET** — "Give me this resource." Used when you open a URL in a browser. No data body.
- **POST** — "Here's some data, process it." Used when submitting forms or sending chat messages.

**HTTP Status Codes:**
- `200 OK` — everything worked.
- `400 Bad Request` — the client sent invalid data (empty message, message too long).
- `429 Too Many Requests` — rate limit exceeded.
- `500 Internal Server Error` — something crashed on the server.

**Flask Route Anatomy:**
```python
@app.route("/chat", methods=["POST"])  # URL pattern + allowed methods
def chat():                             # Python view function
    if not _check_rate(request.remote_addr):
        return jsonify({"response": "Rate limit exceeded."}), 429
    data = request.get_json()
    reply, ctx = engine.respond(data["message"], session.get("context", {}))
    session["context"] = ctx
    chips = _CHIPS.get(ctx.get("last_intent", "out_of_scope"), [])
    media = RICH_MEDIA.get(ctx.get("last_intent", ""), {})
    return jsonify({"response": reply, "chips": chips, "media": media})
```

---

### Natural Language Processing (NLP)

**Technical explanation:**
NLP is the subfield of AI concerned with enabling computers to understand, generate, and manipulate
human language. In this project, the NLP pipeline performs intent classification — mapping
free-form natural language input to a predefined set of categories using semantic similarity
rather than keyword or bag-of-words matching.

**Plain English:**
Human language is messy — "What IS the Samphor?", "tell me ABOUT samphor", and "define the
samphor drum" all mean the same thing but look completely different to a computer. The sentence
transformer model handles this by capturing meaning rather than matching words.

**The NLP pipeline in this project has 2 stages:**
1. **Embedding** — convert text to a dense semantic vector.
2. **Similarity search** — find the training pattern most similar in meaning.

Note: The old pipeline had tokenization (NLTK word_tokenize), lemmatization (WordNetLemmatizer),
and Bag-of-Words vectorization. These steps are no longer needed — the sentence transformer
handles its own internal tokenization using a subword vocabulary (WordPiece).

---

### Sentence Transformers and Embeddings

**Technical explanation:**
A sentence transformer is a neural network (based on BERT/Transformer architecture) fine-tuned
to map text to a fixed-length dense vector (an "embedding") in a semantic space where similar
meanings are geometrically close. The model used here —
`paraphrase-multilingual-MiniLM-L12-v2` — is a 12-layer transformer with 118M parameters
distilled to 22M, producing 384-dimensional embeddings. It was trained on 50+ languages
including Khmer using contrastive learning (similar sentences are pulled together, dissimilar
sentences are pushed apart in the vector space).

**Plain English:**
Think of each sentence as a GPS coordinate in a 384-dimensional space. Sentences that mean
similar things have similar coordinates. "What is the Samphor?" and "Can you define the Samphor?"
will end up very close together in this space even though they use different words.

Training the sentence transformer to understand language took weeks on massive datasets —
we just use the pre-trained model. Our "training" step is just asking the model to compute
coordinates for our 460 example questions and saving those coordinates.

**Embedding dimensions:**
```
"What is the Samphor?" → [0.12, -0.45, 0.83, 0.01, -0.22, ...] (384 numbers)
"Define the Samphor"   → [0.11, -0.44, 0.81, 0.02, -0.23, ...] (very similar)
"How do I play it?"    → [0.67, 0.23, -0.11, 0.44, 0.15, ...] (very different)
```

**Why 384 dimensions?**
Higher dimensions can capture more nuance but require more memory and computation.
384 is a sweet spot for a "small" transformer model (MiniLM) — fast enough for real-time
chat while still very accurate.

---

### Cosine Similarity

**Technical explanation:**
Cosine similarity measures the angle between two vectors: `cos(θ) = (A · B) / (|A| × |B|)`.
When vectors are L2-normalized (length 1), cosine similarity equals the dot product.
Values range from -1 (opposite directions) to +1 (identical direction). For text embeddings,
1.0 means "same meaning", 0.0 means "unrelated", and negative values are uncommon.

**Plain English:**
Cosine similarity asks: "Are these two vectors pointing in the same direction?" Direction
represents meaning. "What is the Samphor?" and "Define the Samphor" point in nearly the
same direction (similarity ~0.92). "What is the Samphor?" and "What's the weather?" point
in very different directions (similarity ~0.05).

```
cos(θ) = 1.0 → identical direction → same meaning
cos(θ) = 0.9 → nearly same direction → very similar meaning
cos(θ) = 0.5 → somewhat similar
cos(θ) = 0.0 → perpendicular → unrelated
cos(θ) < 0.35 → very different → fallback triggered
```

**The confidence threshold:**
`CONFIDENCE_THRESHOLD = 0.52` — if the best matching intent scores below 0.52 cosine similarity,
the bot asks for clarification or returns the fallback response. This threshold was tuned
empirically to minimize both false positives (wrong intent) and false negatives (unnecessary fallbacks).

---

### Semantic Search vs Keyword Matching

**Technical explanation:**
Keyword matching (TF-IDF, Bag-of-Words) computes similarity based on word overlap.
Semantic search computes similarity based on meaning regardless of word choice.
The sentence transformer model has learned that "drum" and "percussion instrument" are related,
that "play" and "perform" are related, and that Khmer and English phrases about the same topic
are related — without any explicit synonym lists.

**Plain English:**
Old approach: "Is the word 'samphor' in your question? Is the word 'what'? Yes + yes? Probably ask_definition."
New approach: "Does your question mean the same thing as our example 'What is the Samphor?' questions? If yes, it's ask_definition."

The new approach works even if you use completely different words:
- "Can you explain the Samphor to me?" — no shared keywords, but semantically close to ask_definition → correctly classified.
- "Sanphor" typo → spell-corrected to "Samphor" before embedding → still works.
- "ស័មភោ គឺ ជា អ្វី?" (Khmer) → same model handles Khmer natively → correct classification.

---

### Pickle and Model Serialization

**Technical explanation:**
`pickle` is Python's built-in object serialization protocol. It converts Python objects
(lists, dicts, class instances) to binary byte streams (serialization) and back (deserialization).
The `labels.pkl` file stores `self._labels` (a `list[str]` of 460 intent tags).

NumPy's `.npy` format stores n-dimensional arrays efficiently as raw binary data with a small
header describing the shape and dtype. Loading `embeddings.npy` is much faster than loading
a pickle of the same data because no Python object deserialization is needed.

**Plain English:**
When your Python script ends, every variable disappears from memory. Pickle is like taking a
snapshot of a Python object and saving it to a file. NumPy's `.npy` is like that, but optimized
for large arrays of numbers.

```python
# Saving
np.save("model/embeddings.npy", embedding_matrix)   # shape (460, 384)
with open("model/labels.pkl", "wb") as f:
    pickle.dump(label_list, f)

# Loading
embeddings = np.load("model/embeddings.npy")
with open("model/labels.pkl", "rb") as f:
    labels = pickle.load(f)
```

**Security note:** Never load pickle files from untrusted sources — pickle can execute arbitrary
code during loading. This is safe here because the files are generated by your own `train.py`.

---

### JSON and Data Exchange

**Technical explanation:**
JSON (JavaScript Object Notation) is a text-based data interchange format. Python's `json`
module provides `json.load()`, `json.loads()`, `json.dump()`, `json.dumps()`.
Flask's `jsonify()` serializes a dict to JSON and wraps it in an HTTP response with
`Content-Type: application/json`.

The `/chat` endpoint now returns a richer JSON structure than before:
```json
{
  "response": "The Samphor is a barrel-shaped drum...",
  "chips": ["What is it made of?", "How do I play it?", "Where can I learn?"],
  "media": {
    "links": [{"label": "Wikipedia: Samphor", "url": "https://..."}]
  }
}
```

**Plain English:**
JSON is a universal language for data. Both Python and JavaScript can read and write it.
When the browser sends a message, it uses JSON. When Flask replies, it also uses JSON —
now with extra fields for chips (suggestion buttons) and media links.

---

### AJAX and the fetch() API

**Technical explanation:**
AJAX (Asynchronous JavaScript and XML — though JSON has replaced XML in practice) refers to
making HTTP requests from the browser without triggering a full page navigation. The modern
implementation is the Fetch API (`fetch()`), which returns a `Promise` that resolves to a
`Response` object.

**Plain English:**
Normally, clicking a button on a web page reloads the entire page. AJAX lets the browser talk
to the server in the background while the page stays visible. The chat box works because of
AJAX — your messages and the replies appear without any page reload.

```javascript
async function sendMessage() {
    showTyping();   // Fix 6: show animated dots
    const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    hideTyping();
    addBotMessage(data.response, data.chips, data.media);  // Fix 5, 7
    saveToStorage();                                        // Fix 8
}
```

---

### Async/Await in JavaScript

**Technical explanation:**
JavaScript is single-threaded. Async/await is syntactic sugar over Promises (which wrap callbacks).
An `async function` always returns a Promise. `await expression` suspends the async function until
the awaited Promise resolves, without blocking the call stack (the event loop continues to process
other events).

**Plain English:**
Your browser has one thread. If you make that thread wait for a server response (which could
take 500ms), the entire page would freeze. `async/await` tells JavaScript: "start this request,
but while it's waiting, go do other things. Come back when the response arrives."

**`try/catch/finally`:**
- `try` — attempt the risky code.
- `catch (err)` — if anything throws an error, run this block.
- `finally` — always runs, whether there was an error or not. Used here to re-enable input.

---

### CSS Flexbox Layout

**Technical explanation:**
Flexbox (Flexible Box Layout) is a CSS layout model. A **flex container** (`display: flex`)
distributes space among its **flex items** along a main axis and cross axis.

**Plain English:**
Flexbox is like a smart shelf system. You tell the shelf "arrange your items in a column, and
make the middle item stretch to fill all available space." Without Flexbox, you'd need to
calculate pixel positions for everything.

**How it's used in the chat layout:**
```css
.chat-wrapper {
    display: flex;
    flex-direction: column;  /* Stack: header, then chat-box, then input-area */
    height: 90vh;
}

#chat-box {
    flex: 1;                 /* Grow to fill all space between header and input */
    overflow-y: auto;        /* Scrollable when messages overflow */
}

.chips-row {
    display: flex;
    flex-wrap: wrap;         /* Chips wrap to next line if too many */
    gap: 6px;
}
```

---

### localStorage and Session Persistence

**Technical explanation:**
`localStorage` is a Web Storage API providing a key-value store that persists across page
refreshes and browser restarts (unlike `sessionStorage`, which is cleared on tab close).
Data is stored as strings; objects must be serialized with `JSON.stringify()` and deserialized
with `JSON.parse()`. Storage is scoped to the origin (domain + port).

**Plain English:**
`localStorage` is like a notebook the browser keeps for a website. When you refresh the page,
the notebook is still there. The chatbot uses it to save your conversation so it reappears
exactly as you left it after a refresh — no more losing your chat history.

```javascript
const STORAGE_KEY = "samphor-chat-v1";

function saveToStorage() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chatHistory));
}

function loadFromStorage() {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) return false;
    JSON.parse(saved).forEach(msg => renderMessage(msg));
    return true;
}
```

---

### Web Speech API

**Technical explanation:**
The Web Speech API provides `SpeechRecognition` (speech-to-text) and `SpeechSynthesis`
(text-to-speech) in modern browsers. `SpeechRecognition` streams audio from the microphone
to the browser's speech engine (often cloud-based), returns `transcript` strings via the
`onresult` event. Availability varies by browser; it is widely supported in Chrome.

**Plain English:**
The mic button uses the browser's built-in speech recognition to convert what you say into text,
which is then sent to the chatbot as if you typed it. If your browser doesn't support it
(some do, some don't), the mic button hides itself automatically.

```javascript
const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
if (!SpeechRec) {
    micBtn.style.display = "none";  // hide if not supported
} else {
    const recognition = new SpeechRec();
    recognition.onresult = (e) => {
        userInput.value = e.results[0][0].transcript;
        sendMessage();
    };
    micBtn.addEventListener("click", () => recognition.start());
}
```

---

## 5. Setup — Step by Step

### Requirements
- Python **3.11 or 3.12** (not 3.13+ — `sentence-transformers` and its PyTorch dependency
  require 3.11/3.12 for stable C-extension builds on Windows)
  ```powershell
  winget install -e --id Python.Python.3.12
  ```
- A terminal / command prompt
- Internet connection (for downloading `sentence-transformers` and the model weights ~120 MB)

---

### Step 1 — Open a Terminal in the Project Folder

In VS Code: `Terminal → New Terminal`

The terminal should show you're inside `samphor-expert/`. If not:
```powershell
cd path\to\samphor-expert
```

---

### Step 2 — Create the Virtual Environment

```powershell
# Windows (PowerShell) — use & operator if python path contains spaces
python -m venv .venv

# Mac / Linux
python3.12 -m venv .venv
```

A new `.venv/` folder appears. Do not edit it manually.

---

### Step 3 — Install the Packages

```powershell
# Windows
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Mac / Linux
./.venv/bin/python -m pip install -r requirements.txt
```

This takes several minutes the first time (downloading ~500 MB including PyTorch and the
sentence-transformers model weights).

---

### Step 4 — Train the Model

```powershell
# Windows
.\.venv\Scripts\python.exe train.py

# Mac / Linux
./.venv/bin/python train.py
```

Expected output:
```
============================================================
  Samphor Expert System - Model Training
============================================================
[MLEngine] Loaded 14 intents, 460 patterns
[MLEngine] Encoding patterns with sentence transformer...
============================================================
  Training Summary
============================================================
  Engine         : Sentence Transformer (Fix 10)
  Status         : PASS — embeddings generated successfully
  Note           : Run the chatbot and test with varied phrasings
============================================================
  Model artefacts saved to model/
    - embeddings.npy  (training pattern embeddings)
    - labels.pkl      (intent label per pattern)
============================================================
```

The model files will be ~750 KB total (much smaller than the old Keras `.h5` file).

---

### Step 5 — Start the Chatbot

```powershell
# Windows
.\.venv\Scripts\python.exe app.py

# Mac / Linux
./.venv/bin/python app.py
```

Then open **http://127.0.0.1:5000** in your browser.
The analytics dashboard is at **http://127.0.0.1:5000/admin**.

---

## 6. How to Run

| Task | Command |
|------|---------|
| Train the model | `.\.venv\Scripts\python.exe train.py` |
| Start the chatbot | `.\.venv\Scripts\python.exe app.py` |
| Open chat UI | http://127.0.0.1:5000 |
| Open admin dashboard | http://127.0.0.1:5000/admin |
| Stop the server | `Ctrl + C` in terminal |
| Retrain after changes | Stop server → run `train.py` → restart `app.py` |

---

## 7. All 17 Improvements Made

This project was upgraded from a basic proof-of-concept to a production-quality chatbot
through 17 incremental improvements. Here is what each one does:

### Critical — Core Chatbot Capabilities

**Fix 1 — Session Isolation**
One shared `MLEngine` instance was causing all browser tabs to share conversation history.
Fixed by storing per-user context in Flask `session` (cookie-based). Each browser tab now has
independent conversation state.

**Fix 2 — Conversation Context**
The conversation history was recorded but never used during inference. Added follow-up
detection ("tell me more", "continue") and pronoun resolution ("what is it made of?" after
a "Samphor" reply), so the bot can handle multi-turn conversations naturally.

**Fix 3 — Multi-turn Clarification**
When confidence is in the 0.35–0.52 grey zone (the bot is uncertain between two intents),
instead of the generic fallback, the bot now asks "Did you mean X or Y?" — enabling it to
disambiguate with one follow-up question.

**Fix 4 — Wire Fallback Engine**
`RuleBasedEngine` existed in `chat_engine.py` but was never used. Added automatic fallback:
if `MLEngine` throws any exception or model files are absent, `app.py` falls back to
`RuleBasedEngine` so the chatbot always responds.

### High Impact — UX & Discoverability

**Fix 5 — Quick Reply Chips**
After every bot response, 3–4 clickable topic buttons appear below the reply so users always
know what they can ask next. Each intent has its own chip set. Clicking a chip sends that
message automatically.

**Fix 6 — Typing Indicator**
A bouncing three-dot animation ("...") appears immediately after you send a message and
disappears when the bot's reply arrives. This prevents the UI from feeling frozen while
waiting for the server.

**Fix 7 — Rich Media**
Wikipedia links and external resources relevant to each intent are embedded in bot replies
as clickable `<a>` buttons. The server returns `media.links` in the JSON response and the
frontend renders them as a row of link buttons under the reply.

**Fix 8 — Conversation Persistence**
The full chat history is saved to `localStorage` (key: `samphor-chat-v1`) after every
message. When you refresh the page or reopen the tab, the previous conversation is restored
automatically. The reset button clears both the UI and localStorage.

### Medium Impact — Intelligence

**Fix 9 — Spell Correction**
Common typos for Samphor-related domain terms are automatically corrected before inference.
Uses `difflib.get_close_matches` with a cutoff of 0.75. Examples: "Sanphor" → "Samphor",
"Phinpeat" → "Pinpeat". No extra packages required.

**Fix 10 — Better NLP (Sentence Transformers)**
The old Bag-of-Words + Keras feedforward network was replaced with a pre-trained sentence
transformer (`paraphrase-multilingual-MiniLM-L12-v2`). This model captures semantic meaning
rather than just word overlap, handles paraphrases and unseen phrasings much better, and
natively supports 50+ languages including Khmer. Training time went from ~30 seconds of
Keras training to ~5 seconds of embedding generation.

**Fix 11 — Data Augmentation**
Each intent was expanded from ~20 patterns to ~33 by adding synonym-expanded and rephrased
variants. Khmer patterns (4 per content intent) were also added. The sentence transformer
generalizes much better than BoW so the extra patterns strengthen coverage at edge cases
rather than causing overfitting.

**Fix 12 — Feedback Buttons**
A thumbs-up (👍) and thumbs-down (👎) button appears on every bot message. Clicking posts
`{msg_id, vote, text}` to `/feedback`, which logs the result to `logs/feedback.jsonl`.
This data can be used to identify which intents produce unsatisfying replies.

### Advanced — Power Features

**Fix 13 — Khmer Language Support**
Native Cambodian users can ask questions in Khmer script. The bot detects Khmer Unicode
characters (range U+1780–U+17FF) and serves replies from `KHMER_RESPONSES` in `samphor_kb.py`.
The multilingual transformer model handles Khmer tokenization natively, so no separate
Khmer NLP pipeline is needed.

**Fix 14 — Analytics Dashboard**
An `/admin` page shows: total messages received, which intents were triggered most (intent
frequency chart), fallback rate (how often the bot didn't understand), average confidence
score, and the full active learning review queue. Data is served as JSON from `/admin/stats`
and `/admin/review` and rendered client-side.

**Fix 15 — Active Learning**
Low-confidence turns (confidence < 0.35) are automatically saved to `logs/review_queue.jsonl`
with the user message, predicted intent, and confidence score. A human can review this file,
correct the intent labels, add the messages as new training patterns, and retrain — gradually
improving the bot with real user input.

**Fix 16 — Voice Input**
A microphone button in the input area uses the browser's Web Speech API (`SpeechRecognition`)
to capture spoken questions and convert them to text. The transcribed text is inserted into
the input field and sent automatically. The button hides itself if the browser doesn't support
the API.

**Fix 17 — Rate Limiting**
The `/chat` endpoint is throttled to 30 requests per minute per IP address using a sliding
window implemented in an in-memory Python dict (no extra packages). Requests exceeding the
limit receive HTTP 429. Messages longer than 500 characters are rejected with HTTP 400.

---

## 8. How to Add a New Topic

Example: adding "Who makes the Samphor?" (intent: `ask_makers`)

**Step 1 — Add training examples to `data/intents.json`:**
```json
{
  "tag": "ask_makers",
  "patterns": [
    "Who makes the Samphor",
    "Who builds the Samphor",
    "Are there Samphor craftsmen",
    "Where are Samphor drums made",
    "Who are the drum makers in Cambodia",
    "How is a Samphor crafted",
    "What craftsmen make the Samphor",
    "Is Samphor making a family tradition",
    "ជ័ងចម្លាក់ស័មភោ",
    "អ្នក ណា ផ្លិត ស្គរ ស័មភោ"
  ],
  "responses": ["placeholder — real answers go in samphor_kb.py"]
}
```

Aim for at least 10–15 English patterns + 2–4 Khmer patterns.

**Step 2 — Add English replies to `knowledge/samphor_kb.py`:**
```python
"ask_makers": [
    "Samphor drums are crafted by specialist artisans in Phnom Penh and Siem Reap. "
    "The craft is passed down through families over generations of apprenticeship.",
    "Traditional Samphor makers use jackfruit wood and animal skin, skills learned "
    "from masters. The Royal University of Fine Arts documents these techniques.",
],
```

**Step 3 — Add Khmer replies to `KHMER_RESPONSES` in `samphor_kb.py`:**
```python
"ask_makers": [
    "ស្គរ ស័មភោ ត្រូវ បាន ផ្លិត ដោយ ជ័ងចម្លាក់ ដ៏ ជំនាញ នៅ ភ្នំ ពេញ...",
],
```

**Step 4 — Add quick-reply chips to `app.py`:**
```python
_CHIPS["ask_makers"] = [
    "Where can I see them work?",
    "What tools do they use?",
    "How long does it take to make one?",
]
```

**Step 5 — Add Wikipedia/media links to `RICH_MEDIA` in `samphor_kb.py`:**
```python
"ask_makers": {
    "links": [{"label": "Khmer Artisans", "url": "https://..."}]
},
```

**Step 6 — Retrain:**
```powershell
.\.venv\Scripts\python.exe train.py
```

**Step 7 — Restart the server:**
```powershell
.\.venv\Scripts\python.exe app.py
```

The bot can now answer questions about Samphor makers in both English and Khmer.

---

## 9. Switching Engines

Edit two lines in `app.py`:

**Default (sentence transformer ML engine):**
```python
from engine.ml_engine import MLEngine
engine = MLEngine()
```

**Fallback (rule-based keyword engine):**
```python
# from engine.ml_engine import MLEngine
# engine = MLEngine()
from engine.chat_engine import RuleBasedEngine
engine = RuleBasedEngine()
```

The rule-based engine is useful when:
- Model files haven't been trained yet (no `embeddings.npy`).
- You want to test the website layout without loading the sentence transformer.
- You're debugging the Flask routes and don't need AI.

Note: `app.py` automatically falls back to `RuleBasedEngine` if `MLEngine` fails to load
(Fix 4), so you usually don't need to switch manually.

---

## 10. Learning Roadmap

This is a structured plan to go from "I know some basics" to fully understanding and extending
every part of this project. Each phase builds on the previous one.

---

### Phase 0 — Prerequisites (Before You Start)

These are tools you need to have set up, not topics you need to master first.

| Task | Why | How Long |
|------|-----|---------|
| Install Python 3.12 | Everything runs on Python | 30 minutes |
| Install VS Code | Your coding environment | 30 minutes |
| Learn to use the terminal | Running scripts, navigation | 1–2 days |
| Understand files and folders | Where everything lives | Already know this |

**Resources:**
- Terminal: Search "Windows PowerShell basics tutorial"
- VS Code: [code.visualstudio.com/docs/introvideos/basics](https://code.visualstudio.com/docs/introvideos/basics)

---

### Phase 1 — Python Foundations (Weeks 1–3)

**Goal:** Understand every line of Python in this project.

**Topics to master:**

| Topic | Where it appears in this project | Priority |
|-------|----------------------------------|---------|
| Variables, data types (str, int, float, bool) | Everywhere | Essential |
| Lists and list comprehensions | `_labels`, embedding comparisons | Essential |
| Dictionaries | `INTENT_RESPONSES`, `KHMER_RESPONSES`, `_CHIPS` | Essential |
| Tuples | Return types `(reply, context)` | Essential |
| Functions (def, return, parameters) | All helper methods | Essential |
| `if/elif/else` and `for` loops | `respond()`, inference loop | Essential |
| `open()`, `with` blocks, `json.load()` | Loading `intents.json` | Essential |
| `import` and modules | Every file imports other files | Essential |
| Classes and `__init__` | `ChatEngine`, `MLEngine` | Essential |
| Inheritance (`class B(A)`) | `MLEngine(ChatEngine)` | Essential |
| `super().__init__()` | MLEngine and RuleBasedEngine constructors | Essential |
| Abstract classes (`ABC`, `@abstractmethod`) | `ChatEngine` | Important |
| Decorators (`@something`) | `@app.route`, `@abstractmethod` | Important |
| Type hints (`str`, `list[str]`, `-> None`) | All function signatures | Good to know |
| `pickle.dump()` / `pickle.load()` | Saving/loading label list | Important |
| `random.choice()` | Picking random replies | Easy |
| `any()` built-in function | Khmer detection loop | Good to know |
| f-strings / template strings | Print statements, logging | Easy |
| `collections.Counter` | Intent frequency tracking in analytics | Good to know |

**What to build to practice:**
- A simple command-line quiz program (uses functions, loops, input/output)
- A to-do list program (uses lists, dicts, file I/O)
- A simple class hierarchy (Animal → Dog, Cat)

**Resources:**
- [python.org/about/gettingstarted](https://www.python.org/about/gettingstarted/)
- *Automate the Boring Stuff with Python* (free online) — Chapters 1–10
- Search "Python OOP tutorial Corey Schafer" on YouTube

---

### Phase 2 — Web Development Basics (Weeks 4–5)

**Goal:** Understand `templates/index.html` and how the browser talks to Flask.

**Topics to master:**

| Topic | Where it appears | Priority |
|-------|-----------------|---------|
| HTML document structure (`<html>`, `<head>`, `<body>`) | `index.html` skeleton | Essential |
| HTML elements: `<div>`, `<h1>`, `<input>`, `<button>`, `<a>` | The chat UI | Essential |
| HTML attributes: `id`, `class`, `type`, `placeholder`, `href` | Chat elements | Essential |
| CSS selectors (`.class`, `#id`, element) | Styling in `<style>` | Essential |
| CSS Box Model (margin, padding, border) | Spacing and layout | Essential |
| CSS Flexbox (`display: flex`, `flex-direction`, `flex: 1`) | Chat layout | Essential |
| CSS animations (`@keyframes`, `animation`) | Typing indicator bounce | Important |
| CSS Colors (`#hex`, `rgb()`) | Color palette | Easy |
| CSS `overflow-y: auto` | Scrollable chat box | Important |
| JavaScript variables (`const`, `let`) | All JS code | Essential |
| JavaScript functions | `addBotMessage()`, `sendMessage()` | Essential |
| `document.getElementById()` | Getting DOM elements | Essential |
| `document.createElement()` | Creating message bubbles, chips | Essential |
| `.appendChild()` | Adding bubbles to chat box | Essential |
| `.textContent` vs `.innerHTML` | Security: use textContent | Essential |
| Event listeners (`.addEventListener()`) | Click and keydown events | Essential |
| `localStorage.setItem()` / `getItem()` | Chat persistence (Fix 8) | Important |
| `JSON.stringify()` / `JSON.parse()` | Storing objects in localStorage | Important |

**What to build to practice:**
- A static HTML/CSS page (a personal profile or simple webpage)
- A simple interactive page (a to-do list that adds/removes items with JavaScript)
- Add `localStorage` to the to-do list so tasks survive a refresh

**Resources:**
- [MDN Web Docs — Learn HTML](https://developer.mozilla.org/en-US/docs/Learn/HTML)
- [MDN Web Docs — Learn CSS](https://developer.mozilla.org/en-US/docs/Learn/CSS)
- [javascript.info](https://javascript.info) — the best free JavaScript reference

---

### Phase 3 — HTTP, Flask, and Async JavaScript (Week 6)

**Goal:** Understand how the browser and Flask server communicate.

**Topics to master:**

| Topic | Where it appears | Priority |
|-------|-----------------|---------|
| HTTP GET vs POST | `/` uses GET, `/chat` uses POST | Essential |
| HTTP status codes (200, 400, 429, 500) | Flask responses | Important |
| HTTP headers (`Content-Type: application/json`) | fetch() headers | Important |
| JSON structure and data types | `{"message": "..."}` | Essential |
| `JSON.stringify()` in JavaScript | Sending data to Flask | Essential |
| `response.json()` in JavaScript | Parsing Flask's reply | Essential |
| Flask `@app.route()` decorator | Registering routes | Essential |
| Flask `request.get_json()` | Reading POST body | Essential |
| Flask `jsonify()` | Returning JSON responses | Essential |
| Flask `render_template()` | Serving index.html | Essential |
| Flask `session` | Per-user conversation context (Fix 1) | Important |
| `request.remote_addr` | IP address for rate limiting (Fix 17) | Good to know |
| Async functions in JavaScript | `async function sendMessage()` | Essential |
| `await` keyword | Waiting for fetch results | Essential |
| Promises (what `async/await` wraps) | Under the hood of fetch | Important |
| `try/catch/finally` | Error handling in sendMessage | Essential |
| AJAX concept (no page reload) | The whole chat flow | Essential |
| Python `if __name__ == "__main__"` | app.py entry point | Important |

**What to build to practice:**
- A simple Flask app with 3 routes that returns JSON
- Connect that Flask app to a frontend with JavaScript fetch()
- Build a simple feedback system: frontend posts a rating, backend logs it

**Resources:**
- [Flask Quickstart](https://flask.palletsprojects.com/en/stable/quickstart/)
- [MDN: HTTP overview](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview)
- [javascript.info/async-await](https://javascript.info/async-await)

---

### Phase 4 — Natural Language Processing and Embeddings (Week 7)

**Goal:** Understand how human text gets converted to semantic vectors and compared.

**Topics to master:**

| Topic | Where it appears | Priority |
|-------|-----------------|---------|
| What NLP is and why it's hard | The whole ML pipeline | Essential |
| What an embedding/vector is | `embeddings.npy` | Essential |
| High-dimensional vector spaces | 384-dim embedding space | Essential |
| Cosine similarity concept | `_predict()` inference | Essential |
| `sklearn.metrics.pairwise.cosine_similarity` | Inference in ml_engine.py | Essential |
| Pre-trained models concept | Using paraphrase-multilingual-MiniLM-L12-v2 | Essential |
| `SentenceTransformer.encode()` | Training and inference | Essential |
| L2 normalization (`normalize_embeddings=True`) | Why dot product = cosine | Important |
| NumPy arrays and shapes | `embeddings.npy` shape (460, 384) | Essential |
| `np.save()` / `np.load()` | Persisting embeddings | Important |
| `difflib.get_close_matches()` | Spell correction (Fix 9) | Good to know |
| Unicode and Khmer script | `_is_khmer()` detection | Good to know |
| Intent classification concept | The prediction task | Essential |
| Confidence thresholds | `CONFIDENCE_THRESHOLD = 0.52` | Essential |

**What to build to practice:**
- Use `SentenceTransformer.encode()` to compute similarity between 5 sentences you write
- Build a tiny semantic FAQ system (5 questions, answer the closest one)
- Experiment with different `cutoff` values and see how they affect matches

**Resources:**
- [SBERT.net Documentation](https://www.sbert.net/) — official sentence transformers docs
- [HuggingFace Model Card: paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- Search "sentence transformers tutorial Python" on YouTube
- [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) — for understanding vectors

---

### Phase 5 — Going Further (Month 2+)

After mastering this project, these are natural next steps:

| Next Topic | Why It Builds on This Project |
|-----------|-------------------------------|
| **Transformer architecture (Attention)** | Understand what's inside the sentence transformer model |
| **Fine-tuning pre-trained models** | Improve the model on Samphor-specific data |
| **REST API design** | Make the chatbot an API others can call |
| **Database integration (SQLite/PostgreSQL)** | Persist conversation history and feedback across restarts |
| **Deployment (Heroku, Railway, AWS)** | Put your chatbot on the internet |
| **Docker** | Package the whole project into a container for reliable deployment |
| **Unit testing (pytest)** | Test `_correct_spelling()`, `_is_khmer()`, `_predict()` in isolation |
| **CI/CD (GitHub Actions)** | Automatically retrain and test when you push changes to intents.json |
| **Authentication (Flask-Login)** | Protect the `/admin` dashboard from public access |

---

### Estimated Timeline

| Phase | Topic | Time (consistent daily practice) |
|-------|-------|----------------------------------|
| 0 | Tools setup | 1–2 days |
| 1 | Python foundations | 2–3 weeks |
| 2 | HTML + CSS + basic JS | 1–2 weeks |
| 3 | HTTP + Flask + async JS | 1 week |
| 4 | NLP + embeddings + cosine similarity | 1 week |
| 5 | Deeper AI topics + deployment | 2–3 weeks |
| **Total** | | **7–10 weeks** |

"Consistent daily practice" means 1–2 hours per day, 5 days per week.

---

## 11. Glossary

| Term | Technical meaning | Plain English |
|------|--------------------|---------------|
| **Abstract Base Class (ABC)** | A class with `@abstractmethod` declarations that cannot be instantiated directly; enforces interface contracts on subclasses | A "contract class" — it says "you must implement these methods" without saying how |
| **AJAX** | Asynchronous JavaScript and XML — sending HTTP requests without page reload | Making the browser talk to the server silently in the background |
| **Async/Await** | JavaScript syntax for writing asynchronous code that looks synchronous; built on Promises | "Start this task, go do other things while waiting, come back when it's done" |
| **Chips (quick-reply)** | Clickable suggestion buttons shown below a bot reply | Pre-written topic buttons that appear after each answer so you know what to ask next |
| **Confidence threshold** | The minimum cosine similarity required to return a non-fallback response | The minimum certainty required before the bot commits to an answer |
| **Cosine similarity** | `(A · B) / (|A| × |B|)` — measures the angle between two vectors; 1.0 = identical direction | A score for "how similar in meaning are these two things?" — 1.0 = same, 0.0 = unrelated |
| **Decorator** | A Python function that wraps another function, adding behavior before or after it runs (`@decorator`) | A label above a function that gives it extra powers |
| **Embedding** | A dense fixed-length vector (array of floats) representing the semantic meaning of text | A list of numbers that captures the meaning of a sentence in a way the computer can compare |
| **Feature vector** | A fixed-length numerical representation of an input (here: a 384-dim embedding) | The list of numbers that represents a sentence for the AI |
| **Flask** | A Python WSGI micro-framework for building web applications | The software that creates the website and handles browser requests |
| **Flask session** | A signed cookie-based client-side store for per-user state | A private notebook the server keeps for each browser tab |
| **GET request** | HTTP method to retrieve a resource without side effects | "Give me this page" — what the browser does when you type a URL |
| **Inference** | Running a trained/pre-trained model on new input to get a prediction | Using the AI to answer a question |
| **Intent** | The category of user request (e.g., `ask_history` = user wants historical information) | The topic category of what the user is asking about |
| **JSON** | JavaScript Object Notation — a text format for structured data using `{}`, `[]`, strings, numbers | A universal text format for sending structured data between programs |
| **Khmer script** | The writing system used for the Cambodian language (Unicode range U+1780–U+17FF) | The alphabet used to write Cambodian |
| **L2 normalization** | Dividing a vector by its length so that `|v| = 1` | Making all vectors the same length so comparisons are fair |
| **localStorage** | Browser-side key-value storage that persists across page refreshes | A notebook the browser keeps for a website that survives closing the tab |
| **Multilingual model** | A language model trained on data from 50+ languages, capable of handling them all | An AI that understands many human languages without needing separate models for each |
| **NLP (Natural Language Processing)** | The subfield of AI concerned with understanding and generating human language | Teaching computers to understand the messiness of human language |
| **NumPy** | Numerical Python — library for fast multi-dimensional array operations | The math library for working with arrays and matrices of numbers |
| **One-hot encoding** | Representing a categorical label as a binary vector with a single 1 at the label's index | A way to represent categories as rows of zeros with one 1 |
| **Overfitting** | When a model performs well on training data but poorly on new data | The model memorized the training examples instead of learning the underlying pattern |
| **Pattern** | An example user query associated with an intent in `intents.json` | One example of how a user might phrase a question about a topic |
| **Pickle** | Python's binary serialization protocol (`pickle.dump()` / `pickle.load()`) | Python's way of saving objects to files and loading them back later |
| **Polymorphism** | The ability of different classes to implement the same interface (`respond()`) while behaving differently | Different objects responding to the same method call in their own way |
| **POST request** | HTTP method that sends data in the request body to the server | "Here's some data — process it" — used when submitting a form or sending a chat message |
| **Pre-trained model** | A model already trained on large datasets by someone else; reused rather than trained from scratch | An AI that has already learned from millions of examples; we use its knowledge without retraining |
| **Promise** | JavaScript object representing the eventual result of an async operation | A JavaScript "IOU" — a guarantee that a result will come later |
| **Rate limiting** | Throttling requests per user/IP to prevent abuse | Limiting how many messages someone can send per minute |
| **Review queue** | A JSONL log of low-confidence turns saved for human review and correction | A list of conversations the bot wasn't sure about, for a human to improve later |
| **Rich media** | Images, links, and other non-text content embedded in bot replies | Wikipedia links and external resources shown as clickable buttons under each reply |
| **Route** | A URL pattern registered with Flask that maps to a Python view function | A specific URL address that Flask knows how to handle |
| **Semantic similarity** | A measure of how similar the *meaning* of two texts is, regardless of exact wording | How close in meaning two sentences are, even if they use completely different words |
| **Sentence transformer** | A neural network that maps full sentences to dense semantic embeddings | An AI that converts sentences into lists of numbers where similar meanings are close together |
| **Sliding window rate limit** | Counting requests within a rolling time window rather than a fixed period | Checking "how many messages in the last 60 seconds?" rather than "how many this minute?" |
| **SpeechRecognition API** | Browser API for converting microphone input to text | The browser's built-in speech-to-text feature |
| **Virtual environment (venv)** | An isolated Python interpreter and `site-packages` directory for one project | A private kitchen with its own ingredients just for this project |
| **WSGI** | Web Server Gateway Interface — the standard Python interface between web servers and web apps | The standard "plug shape" that connects Python web frameworks to web servers |
| **XSS (Cross-Site Scripting)** | An attack where malicious script is injected into a web page | Why `textContent` is used instead of `innerHTML` — prevents injected code from running |

---

*This README was written to be both a technical reference and a learning companion.
Every concept here is implemented in the files above — read the code alongside this guide.*
