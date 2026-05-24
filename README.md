# Samphor Expert System

An AI chatbot that answers questions about the **Samphor** — a traditional Cambodian barrel drum.
It uses a small neural network trained on hand-written example questions to understand what you are asking,
then looks up a human-written answer from a knowledge base.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [How the Project is Wired Together](#2-how-the-project-is-wired-together)
3. [File-by-File Guide](#3-file-by-file-guide)
4. [Setup — Step by Step](#4-setup--step-by-step)
5. [How to Run](#5-how-to-run)
6. [How to Add a New Topic](#6-how-to-add-a-new-topic)
7. [Switching Engines](#7-switching-engines)
8. [What You Need to Learn to Build This](#8-what-you-need-to-learn-to-build-this)
9. [Glossary](#9-glossary)

---

## 1. What This Project Does

```
You type a question  →  The AI figures out what category it is  →  It sends back a reply
"What is the Samphor?"    "ask_definition" (85% confident)         "The Samphor is a barrel-shaped..."
```

The "figuring out" part is done by a **neural network** — a program that learned from 280 example questions.
It does not search the internet; everything it knows was written by hand in two files:
`data/intents.json` (training examples) and `knowledge/samphor_kb.py` (the actual answers).

---

## 2. How the Project is Wired Together

This diagram shows how the files talk to each other when you type a message:

```
Browser (index.html)
       │
       │  POST /chat  {"message": "What is the Samphor?"}
       ▼
app.py  (Flask web server)
       │
       │  engine.respond("What is the Samphor?")
       ▼
engine/ml_engine.py  (the AI brain)
       │                        │
       │  predict intent        │  look up reply text
       ▼                        ▼
model/samphor_model.h5    knowledge/samphor_kb.py
(trained neural network)  (the answer book)
       │
       │  returns "The Samphor is a barrel-shaped drum..."
       ▼
app.py  → {"response": "The Samphor is..."}
       │
       ▼
Browser displays the reply in the chat box
```

**Training** (a separate process, only done once):

```
data/intents.json  →  engine/ml_engine.py (train())  →  model/samphor_model.h5
                                                          model/words.pkl
                                                          model/classes.pkl
```

---

## 3. File-by-File Guide

### `app.py` — The Web Server

**What it does:** Starts the website and listens for messages from the browser.
Think of it like a restaurant waiter — it takes your order (message) to the kitchen (ML engine)
and brings back the food (reply).

**Key parts:**
| Part | What it does |
|------|-------------|
| `Flask(__name__)` | Creates the web application |
| `@app.route("/")` | Serves the chat page when you open the URL |
| `@app.route("/chat")` | Receives your message and returns the AI's reply |
| `@app.route("/reset")` | Clears the conversation when you click ↺ |

**Talks to:** `engine/ml_engine.py` (calls `respond()` and `reset()`),
`templates/index.html` (serves it to the browser)

---

### `engine/chat_engine.py` — The Blueprint

**What it does:** Defines the rules that every engine must follow.
Think of it like a job description — it says "you must be able to `respond()` and `reset()`"
without saying how.

Also contains **RuleBasedEngine** — a simple backup engine that matches keywords:
"if the message contains 'history', reply with this text."

**Key concept:** This is called an **Abstract Base Class (ABC)**. Any engine that extends it
must implement `respond()` and `reset()`, otherwise Python refuses to run.

**Talks to:** Nothing directly — other files extend or import it.

---

### `engine/ml_engine.py` — The AI Brain

**What it does:** The smart engine that uses a neural network.
This is the most complex file in the project.

**Two modes:**

| Mode | When | What happens |
|------|------|-------------|
| Training | You run `train.py` | Reads intents.json, builds a neural network, saves it to model/ |
| Chatting | Every user message | Converts text → numbers → feeds to network → picks a reply |

**Key steps in training:**
1. Load example questions from `data/intents.json`
2. Split each question into words (tokenize)
3. Reduce words to their root form: "playing" → "play" (lemmatize)
4. Convert each question to a list of 0s and 1s (bag-of-words)
5. Train a neural network for 200 passes through the data
6. Save the network and vocabulary to `model/`

**Key steps when chatting:**
1. Take the user's message
2. Apply the same word-cleaning steps as training
3. Convert to 0s and 1s
4. Ask the neural network: "which category does this belong to?"
5. If confidence ≥ 70%, look up the reply in `samphor_kb.py`
6. If confidence < 70%, return the fallback "I didn't understand" message

**Talks to:**
- Extends `engine/chat_engine.py` (follows its blueprint)
- Reads `data/intents.json` (training data)
- Saves/loads files in `model/` (trained network)
- Imports replies from `knowledge/samphor_kb.py`

---

### `knowledge/samphor_kb.py` — The Answer Book

**What it does:** Stores all the text replies the chatbot can give.
It is just a Python dictionary — no AI, no math.

```
"ask_history" → ["The Samphor traces its origins...", "Historical evidence places..."]
```

The engine picks one reply at random so the bot doesn't always say the exact same sentence.

**Talks to:** `engine/ml_engine.py` imports `INTENT_RESPONSES` and `FALLBACK_RESPONSE` from here.

**To add new answers:** Add an entry here with the same key name as the intent tag in `intents.json`.

---

### `data/intents.json` — The Training Data

**What it does:** Contains every example question used to teach the neural network.
Each "intent" has:
- `"tag"` — the category name (e.g. `"ask_history"`)
- `"patterns"` — 15–20 different ways a user might ask that question
- `"responses"` — not used for training; the real answers are in `samphor_kb.py`

**Example:**
```json
{
  "tag": "ask_history",
  "patterns": [
    "What is the history of the Samphor",
    "When was the Samphor invented",
    "How old is the Samphor",
    ...
  ],
  "responses": ["..."]
}
```

**Talks to:** `engine/ml_engine.py` reads this file during `train()`.

**If you change this file:** You must re-run `python train.py` to update the model.

---

### `model/` — Saved Model Files

These files are created by `train.py` and loaded automatically when the chatbot starts.

| File | What's inside |
|------|--------------|
| `samphor_model.h5` | The trained neural network (all its learned weights) |
| `words.pkl` | The vocabulary — 389 unique root words from all training patterns |
| `classes.pkl` | The 14 intent category names |

These files are in `.gitignore` — they are not committed to git because they are large binary
files that can always be regenerated by running `train.py`.

---

### `train.py` — The Training Script

**What it does:** A simple script that kicks off the training process.
You only need to run this once (or again if you change `intents.json`).

```
python train.py
```

**What it prints:**
```
============================================================
  Samphor Expert System - Model Training
============================================================
[MLEngine] Loaded 14 intents ...
[MLEngine] Vocabulary size : 389 words
[MLEngine] Training ...
Epoch 1/200 ...
...
Final accuracy : 99.29%
Status         : PASS (>= 70% accuracy threshold)
============================================================
```

---

### `templates/index.html` — The Chat Interface

**What it does:** The web page the user sees.
It has three parts:

| Part | What it is |
|------|-----------|
| CSS (inside `<style>`) | Controls colours, layout, sizes |
| HTML (inside `<body>`) | The actual elements: header, chat box, input, buttons |
| JavaScript (inside `<script>`) | Handles typing, clicking, and talking to the server |

**How the JavaScript works:**
1. You type a message and click Send (or press Enter)
2. JavaScript shows your message immediately in the chat box
3. It sends the message to `POST /chat` using `fetch()` (an HTTP request)
4. It waits for Flask to reply with JSON
5. It shows the bot's reply in the chat box

No page reload happens — this is called an **AJAX** request.

---

## 4. Setup — Step by Step

### Requirements
- Python **3.13** (not 3.14 — TensorFlow does not support it yet)
- VS Code (optional but recommended)
- Internet connection (for downloading packages the first time)

---

### Step 1 — Open a Terminal in VS Code

In VS Code: `Terminal → New Terminal`
Make sure the terminal is inside the project folder:
```
cd path/to/samphor-expert
```

---

### Step 2 — Create the Virtual Environment

A virtual environment is an isolated box for this project's Python packages.
It stops packages from different projects colliding with each other.

```bash
# Windows
"C:/Users/YourName/AppData/Local/Programs/Python/Python313/python.exe" -m venv venv

# Mac / Linux
python3.13 -m venv venv
```

You will see a new `venv/` folder appear in the project.

---

### Step 3 — Install the Packages

```bash
# Windows
./venv/Scripts/python.exe -m pip install flask nltk numpy scikit-learn pandas tensorflow

# Mac / Linux
./venv/bin/python -m pip install flask nltk numpy scikit-learn pandas tensorflow
```

This installs everything listed in `requirements.txt`. It may take a few minutes the first time.

---

### Step 4 — Train the Model

```bash
# Windows
./venv/Scripts/python.exe train.py

# Mac / Linux
./venv/bin/python train.py
```

This reads `data/intents.json`, trains the neural network for 200 epochs,
and saves the result to `model/`. You should see accuracy above 90%.

---

### Step 5 — Start the Chatbot

```bash
# Windows
./venv/Scripts/python.exe app.py

# Mac / Linux
./venv/bin/python app.py
```

Then open your browser and go to: **http://127.0.0.1:5000**

---

## 5. How to Run

| Task | Command |
|------|---------|
| Train the model (first time or after changing intents.json) | `./venv/Scripts/python.exe train.py` |
| Start the chatbot | `./venv/Scripts/python.exe app.py` |
| Open the chatbot | Go to http://127.0.0.1:5000 in your browser |
| Stop the server | Press `Ctrl + C` in the terminal |

---

## 6. How to Add a New Topic

Let's say you want the bot to answer "Who makes the Samphor?" (intent: `ask_makers`).

**Step 1 — Add training examples to `data/intents.json`:**
```json
{
  "tag": "ask_makers",
  "patterns": [
    "Who makes the Samphor",
    "Who builds the Samphor",
    "Are there Samphor craftsmen",
    "Where are Samphor drums made",
    "Who are the drum makers in Cambodia"
  ],
  "responses": ["placeholder"]
}
```

**Step 2 — Add the real reply to `knowledge/samphor_kb.py`:**
```python
"ask_makers": [
    "Samphor drums are made by specialist craftsmen in Phnom Penh and Siem Reap. "
    "The craft is passed down through families and learned over many years of apprenticeship.",
],
```

**Step 3 — Retrain:**
```bash
./venv/Scripts/python.exe train.py
```

**Step 4 — Restart the chatbot** (it needs to reload the new model):
```bash
./venv/Scripts/python.exe app.py
```

---

## 7. Switching Engines

The project has two engines. To swap between them, edit two lines in `app.py`:

**Use the AI engine (default):**
```python
from engine.ml_engine import MLEngine
engine = MLEngine()
```

**Use the simple keyword engine:**
```python
# from engine.ml_engine import MLEngine
# engine = MLEngine()
from engine.chat_engine import RuleBasedEngine
engine = RuleBasedEngine()
```

The simple engine is useful if the model files are missing or you just want to test the website.

---

## 8. What You Need to Learn to Build This

Here are the subjects this project covers, from simplest to most advanced.
Each item links to a free resource.

### Foundation (learn these first)

| Topic | Why you need it | Where to learn |
|-------|----------------|----------------|
| **Python basics** | Every file is Python. Variables, lists, dictionaries, loops, functions, classes. | [python.org/about/gettingstarted](https://www.python.org/about/gettingstarted/) |
| **Python classes & OOP** | `ChatEngine`, `MLEngine`, and `RuleBasedEngine` are all classes. Understanding inheritance explains why MLEngine can "be" a ChatEngine. | Search "Python OOP tutorial for beginners" on YouTube |
| **Python file I/O** | `open()`, `json.load()`, `pickle.dump()` — reading and writing files. | Python docs: `open()` |
| **Virtual environments** | `venv` isolates your packages. You need to understand why before you can debug install problems. | [docs.python.org/3/library/venv.html](https://docs.python.org/3/library/venv.html) |

---

### Web Development

| Topic | Why you need it | Where to learn |
|-------|----------------|----------------|
| **HTML** | The structure of `index.html` — headings, divs, inputs, buttons. | [MDN: Learn HTML](https://developer.mozilla.org/en-US/docs/Learn/HTML) |
| **CSS** | Styling: colours, layout (flexbox), border-radius, fonts. | [MDN: Learn CSS](https://developer.mozilla.org/en-US/docs/Learn/CSS) — especially Flexbox |
| **JavaScript (basics)** | Variables, functions, `document.getElementById()`, event listeners. | [javascript.info](https://javascript.info) (free, excellent) |
| **async/await & fetch()** | How the browser sends a message to the server without reloading the page. | [javascript.info/async-await](https://javascript.info/async-await) |
| **JSON** | The format used to pass data between the browser and the server: `{"message": "..."}` | [json.org](https://www.json.org/json-en.html) |
| **Flask** | The Python web framework used in `app.py`. Routes, `render_template`, `jsonify`, `request`. | [flask.palletsprojects.com/quickstart](https://flask.palletsprojects.com/en/stable/quickstart/) |
| **HTTP basics** | GET vs POST requests, status codes (200, 400), JSON responses. | [MDN: HTTP overview](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview) |

---

### Machine Learning & NLP

| Topic | Why you need it | Where to learn |
|-------|----------------|----------------|
| **What is machine learning** | The big picture: training, inference, accuracy. | [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) |
| **Natural Language Processing (NLP)** | Tokenization (splitting text into words), lemmatization (finding root words). | Search "NLTK tutorial beginner Python" |
| **Bag of Words (BoW)** | How text is converted to numbers for the neural network. | Search "bag of words NLP explained simply" |
| **Neural networks basics** | What neurons, layers, weights, and activation functions are. | [3Blue1Brown Neural Networks playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — very visual, free |
| **TensorFlow / Keras** | The library used to build and train the neural network in `ml_engine.py`. | [keras.io/getting_started](https://keras.io/getting_started/) |
| **Dense layers** | The type of layer used in this model. | Covered in the Keras getting-started guide above |
| **Dropout** | Technique to prevent overfitting (memorising training data). | Search "dropout regularization explained" |
| **Softmax & classification** | How the network produces a percentage confidence score for each category. | Search "softmax function explained simply" |
| **SGD optimizer** | How the network learns by adjusting its weights. | Google ML Crash Course covers this |

---

### Tools & Workflow

| Topic | Why you need it | Where to learn |
|-------|----------------|----------------|
| **VS Code** | The code editor used in this project (settings.json configures it). | [code.visualstudio.com/docs](https://code.visualstudio.com/docs) |
| **Git & GitHub** | Version control — saving snapshots of your code and collaborating. | [git-scm.com/book](https://git-scm.com/book/en/v2) Chapter 1–3 |
| **pip & requirements.txt** | Installing and managing Python packages. | [pip documentation](https://pip.pypa.io/en/stable/getting-started/) |
| **Terminal / Command Line** | Running Python scripts, navigating folders. | Search "command line basics tutorial Windows" |

---

### Suggested Learning Order

If you are starting from scratch, follow this order:

1. Python basics (2–3 weeks)
2. HTML + CSS (1 week)
3. JavaScript basics (1 week)
4. Flask (a few days — very easy once you know Python)
5. HTTP + async/await + fetch() (a few days)
6. What is ML / neural networks (videos — 1 week)
7. NLTK and bag-of-words (a few days)
8. TensorFlow/Keras (1–2 weeks)

Total estimated time from zero: **2–3 months** of consistent practice.

---

## 9. Glossary

| Word | Plain-English meaning |
|------|----------------------|
| **Intent** | The category of what the user is asking. e.g. "ask_history" means the user wants to know about the Samphor's history. |
| **Pattern** | An example question used to train the model. e.g. "When was the Samphor invented?" |
| **Bag of Words (BoW)** | Converting a sentence into a list of 0s and 1s — one slot per word in the vocabulary. 1 if the word appears, 0 if not. |
| **Tokenize** | Split a sentence into individual words. "What is this?" → ["What", "is", "this", "?"] |
| **Lemmatize** | Reduce a word to its root form. "playing" → "play", "drums" → "drum". Helps the model recognise the same concept written differently. |
| **Neural network** | A program inspired by the brain. It has layers of "neurons" that each do simple math. Together they can learn complex patterns. |
| **Dense layer** | A layer where every neuron connects to every neuron in the next layer. The most basic neural network building block. |
| **Dropout** | During training, randomly ignore some neurons. Forces the network to not rely on any one path — makes it more general. |
| **Softmax** | A math function on the output layer that converts raw numbers into percentages that add up to 100%. Used for classification. |
| **Confidence score** | The percentage the model assigns to its best guess. If it says 85%, it is 85% sure this is the right category. |
| **Confidence threshold** | The minimum confidence required to trust the model. Set to 70% here — below that, the bot says "I didn't understand." |
| **Epoch** | One full pass through all training examples. This project trains for 200 epochs. |
| **Loss** | A number measuring how wrong the model's predictions are. Lower is better. The SGD optimizer tries to reduce this. |
| **Accuracy** | What percentage of training examples the model predicted correctly. 99% means it got 277 out of 280 right. |
| **Overfitting** | When the model memorises the training data instead of learning general patterns. It scores well on training data but fails on new questions. Dropout helps prevent this. |
| **Pickle** | Python's way to save any object to a binary file and load it back later. Used to save the vocabulary and class lists. |
| **Virtual environment (venv)** | An isolated Python installation just for this project. Packages installed here don't affect other projects on your computer. |
| **Flask route** | A URL pattern that Flask listens for. `@app.route("/chat")` means "when the browser visits /chat, run this function." |
| **JSON** | JavaScript Object Notation — a text format for structured data. `{"message": "hello"}` is JSON. Flask sends and receives data this way. |
| **AJAX / fetch()** | A way for JavaScript to send HTTP requests and receive responses without reloading the whole page. |
| **Abstract Base Class (ABC)** | A Python class that acts as a blueprint. It defines which methods sub-classes must implement, but doesn't implement them itself. |
