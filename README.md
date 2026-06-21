# Samphor Expert System — Complete Learning Guide

An AI-powered chatbot that answers questions about the **Samphor**, a traditional Cambodian barrel drum.
The system uses a small neural network trained on hand-written example questions to understand what you are asking,
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
   - [requirements.txt — The Dependency List](#requirementstxt--the-dependency-list)
   - [model/ — The Saved Brain Files](#model--the-saved-brain-files)
4. [Deep Concept Explanations](#4-deep-concept-explanations)
   - [Python Virtual Environments](#python-virtual-environments)
   - [Object-Oriented Programming and Inheritance](#object-oriented-programming-and-inheritance)
   - [Abstract Base Classes](#abstract-base-classes)
   - [The Flask Web Framework and HTTP](#the-flask-web-framework-and-http)
   - [Natural Language Processing (NLP)](#natural-language-processing-nlp)
   - [Tokenization](#tokenization)
   - [Lemmatization](#lemmatization)
   - [Bag of Words](#bag-of-words)
   - [Neural Networks from Scratch](#neural-networks-from-scratch)
   - [Activation Functions: ReLU and Softmax](#activation-functions-relu-and-softmax)
   - [Dropout Regularization](#dropout-regularization)
   - [The SGD Optimizer and Backpropagation](#the-sgd-optimizer-and-backpropagation)
   - [Loss, Accuracy, and Epochs](#loss-accuracy-and-epochs)
   - [Pickle and Model Serialization](#pickle-and-model-serialization)
   - [JSON and Data Exchange](#json-and-data-exchange)
   - [AJAX and the fetch() API](#ajax-and-the-fetch-api)
   - [Async/Await in JavaScript](#asyncawait-in-javascript)
   - [CSS Flexbox Layout](#css-flexbox-layout)
5. [Setup — Step by Step](#5-setup--step-by-step)
6. [How to Run](#6-how-to-run)
7. [How to Add a New Topic](#7-how-to-add-a-new-topic)
8. [Switching Engines](#8-switching-engines)
9. [Learning Roadmap](#9-learning-roadmap)
10. [Glossary](#10-glossary)

---

## 1. What This Project Does — The Big Picture

```
You type a question  →  The AI figures out what category it is  →  It sends back a reply
"What is the Samphor?"    "ask_definition" (85% confident)         "The Samphor is a barrel-shaped..."
```

**Technical explanation:**
The system is a multi-layer intent-classification chatbot. User input is preprocessed with NLP
techniques (tokenization, lemmatization), converted to a bag-of-words feature vector, passed through
a feedforward neural network (Sequential Keras model with Dense and Dropout layers), and the
predicted intent class is used as a key to retrieve a response from a hand-authored knowledge base.

**Plain English:**
You type a question. The program doesn't search Google — it reads your words, turns them into a list
of numbers, feeds those numbers to a small AI brain it was trained on beforehand, and the brain says
"I think they're asking about the history" (or whatever category it is). The program then picks a
pre-written answer from that category and sends it back. All the knowledge came from files you wrote,
not the internet.

---

**Why separate training from chatting?**

**Technical:** Training is computationally expensive (forward pass + backpropagation × 200 epochs × 280 samples).
Inference is cheap (one forward pass). Separating them means the chatbot starts instantly.

**Plain English:** Teaching the AI takes a few minutes and lots of math. Once it's learned, answering
a question takes milliseconds. So we teach it once, save what it learned, and let the chatbot just use
the saved result.

---

## 2. How All the Files Talk to Each Other

### During a Chat (every message you send)

```
Browser (templates/index.html)
       │
       │  JavaScript fetch() — POST /chat  {"message": "What is the Samphor?"}
       ▼
app.py  (Flask web server — the traffic director)
       │
       │  engine.respond("What is the Samphor?")
       ▼
engine/ml_engine.py  (the AI brain)
       │  1. tokenize + lemmatize user text
       │  2. convert to bag-of-words vector
       │  3. run neural network forward pass
       │  4. pick highest confidence intent
       │
       ├──────────────────────────────────────────────────────┐
       ▼                                                       ▼
model/samphor_model.h5                            knowledge/samphor_kb.py
(the saved neural network weights)                (the dictionary of pre-written answers)
       │                                                       │
       │  returns "ask_definition" at 85% confidence          │  returns 1 random answer string
       └──────────────────────────┬────────────────────────────┘
                                  ▼
                             app.py
                                  │  JSON: {"response": "The Samphor is a barrel-shaped drum..."}
                                  ▼
                        Browser displays reply in chat box
```

### During Training (only run once with `train.py`)

```
data/intents.json
       │  14 intents × ~20 patterns = ~280 training examples
       ▼
engine/ml_engine.py  train()
       │  Step 1: Load all patterns and their intent tags
       │  Step 2: Tokenize and lemmatize every pattern
       │  Step 3: Build vocabulary (list of all unique root words)
       │  Step 4: Convert each pattern to a bag-of-words vector
       │  Step 5: Convert each intent tag to a one-hot vector
       │  Step 6: Build Keras neural network architecture
       │  Step 7: Train for 200 epochs with SGD optimizer
       │  Step 8: Save trained network and vocabulary to disk
       ▼
model/samphor_model.h5   (the trained network — ~1 MB binary file)
model/words.pkl          (389 root words, the vocabulary)
model/classes.pkl        (14 intent names, the output categories)
```

---

## 3. The Complete File Guide

### `app.py` — The Web Server

**Technical explanation:**
`app.py` is a Flask WSGI application. It registers three URL routes using Python decorators.
The root route (`/`) renders a Jinja2 template. The `/chat` route accepts JSON POST requests,
calls `engine.respond()`, and returns a JSON response. The `/reset` route clears conversation state.
The engine is instantiated once at module level so it persists across requests.

**Plain English:**
`app.py` is like the front door of a restaurant. When you walk in (visit the website), it gives
you the menu (the chat page). When you order (send a message), it takes your order to the kitchen
(the AI engine) and brings back the food (the reply). It runs continuously until you press Ctrl+C.

**Full breakdown of every line that matters:**

```python
from flask import Flask, jsonify, render_template, request
```
Imports four tools from Flask:
- `Flask` — the class that creates the web application
- `jsonify` — converts a Python dictionary to a JSON HTTP response
- `render_template` — finds an HTML file in `templates/` and sends it to the browser
- `request` — lets you read the data the browser sent (like the message text)

```python
app = Flask(__name__)
```
Creates the Flask application. `__name__` is a Python built-in that equals the file's name
(`"__main__"` when run directly). Flask uses it to know where to look for templates and static files.

```python
engine = MLEngine()
```
Creates one instance of the AI engine when the server starts. It is created once and reused
for every message. This means the model is loaded into memory once, not on every chat request.
**This is important** — loading the model on every request would be very slow.

```python
@app.route("/")
def index():
    return render_template("index.html")
```
The `@app.route("/")` is a **decorator**. It tells Flask: "when the browser visits the root URL
(`http://127.0.0.1:5000/`), call the `index()` function." The function returns the rendered
`templates/index.html` file.

```python
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "Please type a message."}), 400
    reply = engine.respond(user_message)
    return jsonify({"response": reply})
```
- `methods=["POST"]` — only accepts POST requests (not GET). A POST request carries data in its body.
- `request.get_json()` — parses the JSON body the browser sent.
- `data.get("message", "")` — safely extracts the "message" field; defaults to `""` if missing.
- `.strip()` — removes leading/trailing whitespace.
- The `400` in the return is an HTTP status code meaning "Bad Request" — tells the browser something was wrong.
- `jsonify({"response": reply})` — wraps the reply in JSON and sends it back.

```python
@app.route("/reset", methods=["POST"])
def reset():
    engine.reset()
    return jsonify({"status": "ok"})
```
Clears conversation history when the user clicks the ↺ button. The browser sends a POST to `/reset`,
Flask calls `engine.reset()`, and returns `{"status": "ok"}` so the browser knows it succeeded.

```python
if __name__ == "__main__":
    app.run(debug=True)
```
`if __name__ == "__main__"` is a Python idiom. It means: "only run this block if this file was
started directly (not imported by another file)." `debug=True` enables automatic server restart
when you save a file, and shows detailed error messages in the browser.

---

### `train.py` — The Training Script

**Technical explanation:**
A standalone script that instantiates `MLEngine`, calls its `train()` method (which performs
the full NLP preprocessing and model fitting pipeline), and prints a formatted summary of the
final training metrics.

**Plain English:**
This is the "teacher" script. You run it once. It reads all the example questions, runs the
math to teach the neural network, and saves everything it learned to the `model/` folder.
After that, the chatbot can use the saved result without needing to learn again.

**Key parts:**

```python
engine = MLEngine()
accuracy, loss = engine.train()
```
Creates the engine and starts training. `train()` returns the final accuracy and loss after
all 200 epochs.

```python
status = "PASS" if accuracy >= 0.70 else "FAIL"
```
A simple quality check. If accuracy is below 70%, something went wrong (maybe not enough
training data, or the data is inconsistent). The system expects >99% on this small dataset.

---

### `engine/chat_engine.py` — The Blueprint

**Technical explanation:**
`chat_engine.py` defines two classes. First, `ChatEngine` is an Abstract Base Class (ABC)
using Python's `abc` module. It declares `respond()` and `reset()` as abstract methods using
`@abstractmethod`, which forces all subclasses to implement them. This enforces a consistent
interface across engine implementations.

Second, `RuleBasedEngine` is a concrete subclass of `ChatEngine`. It stores a `RULES`
dictionary mapping tuples of keyword strings to response strings. Its `respond()` method
lowercases the input and iterates over `RULES` looking for a keyword match.

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
the corresponding response is returned. The iteration checks keywords in order, so more specific
intents should be listed first to avoid false matches.

**The respond() method logic:**

```python
def respond(self, user_input: str) -> str:
    lowered = user_input.lower()
    for keywords, response in self.RULES.items():
        if any(kw in lowered for kw in keywords):
            self._history.append({"role": "user", "content": user_input})
            self._history.append({"role": "assistant", "content": response})
            return response
    return "I can only answer questions about the Samphor drum."
```
- `.lower()` — converts to lowercase so "Hello" and "hello" both match.
- `any(kw in lowered for kw in keywords)` — a generator expression that checks if at least one keyword is a substring of the message. This is called a **short-circuit evaluation** — it stops checking as soon as one match is found.
- If no rule matches, a default fallback is returned.

---

### `engine/ml_engine.py` — The AI Brain

**Technical explanation:**
`MLEngine` extends `ChatEngine` and implements the full ML pipeline. It uses NLTK's
`WordNetLemmatizer` for morphological reduction, a bag-of-words vectorization scheme
for feature extraction, and a Keras Sequential model with Dense/Dropout layers trained
with stochastic gradient descent (SGD with Nesterov momentum) and categorical crossentropy loss.
Model artifacts are serialized to disk via `h5py` (`.h5` format for Keras weights) and
`pickle` (for vocabulary and class label lists).

**Plain English:**
This is the real AI — the smart engine. It has two jobs:

1. **Teaching itself** (during `train.py`): It reads all the example questions, cleans them up,
   converts them to numbers, and uses those numbers to train a tiny neural network. It runs through
   all the data 200 times, getting a little better each time.

2. **Answering questions** (during chatting): When you type something, it cleans up your words
   the same way it did during training, converts them to numbers, and asks the saved neural network:
   "which category does this belong to?" The network gives a confidence score for each of the 14
   categories, and the engine picks the highest one.

**The Training Pipeline in detail:**

**Step 1 — Loading intents:**
```python
with open("data/intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)
intents = data["intents"]
```
Opens and parses the JSON file. `encoding="utf-8"` ensures Khmer script and special characters
are read correctly. `json.load()` converts the JSON text into a Python dictionary.

**Step 2 — Tokenizing and lemmatizing every pattern:**
```python
for intent in intents:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        lemmatized = self._lemmatize_tokens(tokens)
        all_words.extend(lemmatized)
        documents.append((lemmatized, intent["tag"]))
```
For each of the 280 example questions:
- `nltk.word_tokenize()` splits it into individual words and punctuation tokens.
- `_lemmatize_tokens()` reduces each word to its root form and discards punctuation.
- The result is stored as a pair: `(cleaned_words, intent_tag)`.
- All unique root words are collected to form the vocabulary.

**Step 3 — Building bag-of-words training data:**
```python
X = []
y = []
for doc_words, tag in documents:
    bow = self._make_bow(doc_words, self._words)
    X.append(bow)
    y.append(class_index)
X = np.array(X)
y = to_categorical(y, num_classes=len(self._classes))
```
- Each of the 280 patterns becomes one **row** in `X` (a 389-length vector of 0s and 1s).
- Each intent tag becomes one **row** in `y` (a 14-length one-hot vector, e.g., `[0,0,1,0,...]`).
- `to_categorical()` converts class indices to one-hot vectors. This is called **one-hot encoding**.
- `np.array()` wraps the lists in NumPy arrays, which Keras requires.

**Step 4 — Building the neural network:**
```python
model = Sequential([
    Dense(128, input_shape=(len(self._words),), activation="relu"),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(len(self._classes), activation="softmax"),
])
```
This is the architecture — the "shape" of the brain:
- **Layer 1:** Takes 389 numbers in, outputs 128 numbers. Uses ReLU activation.
- **Dropout:** Randomly ignores 50% of layer-1 neurons during each training step.
- **Layer 2:** Takes 128 numbers in, outputs 64 numbers. Uses ReLU activation.
- **Dropout:** Randomly ignores 50% of layer-2 neurons.
- **Output Layer:** Takes 64 numbers in, outputs 14 confidence scores. Uses Softmax activation.

**Step 5 — Compiling and training:**
```python
model.compile(
    loss="categorical_crossentropy",
    optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
    metrics=["accuracy"],
)
model.fit(X, y, epochs=200, batch_size=5, verbose=1)
```
- `loss="categorical_crossentropy"` — the math formula used to measure how wrong the predictions are. Correct for multi-class classification.
- `SGD` — Stochastic Gradient Descent, the optimizer that updates the network weights to reduce loss.
- `momentum=0.9` — makes the optimizer "remember" the direction it's been moving, so it doesn't reverse sharply.
- `nesterov=True` — a refinement of momentum that looks ahead before computing the gradient.
- `epochs=200` — passes through all 280 examples 200 times.
- `batch_size=5` — updates weights after every 5 examples (not all 280 at once).

**Step 6 — Saving:**
```python
model.save("model/samphor_model.h5")
with open("model/words.pkl", "wb") as f:
    pickle.dump(self._words, f)
with open("model/classes.pkl", "wb") as f:
    pickle.dump(self._classes, f)
```
Saves the neural network in HDF5 format (`.h5`), and pickles the vocabulary and class list.
Without saving, everything would be lost when the script ends.

**The Inference Pipeline in detail (every chat message):**

```python
def respond(self, user_input: str) -> str:
    tokens = nltk.word_tokenize(user_input)
    lemmatized = self._lemmatize_tokens(tokens)
    bow = self._make_bow(lemmatized, self._words)
    prediction = self._model.predict(np.array([bow]), verbose=0)[0]
    idx = int(np.argmax(prediction))
    confidence = float(prediction[idx])
    if confidence >= 0.70:
        tag = self._classes[idx]
        replies = INTENT_RESPONSES.get(tag, [FALLBACK_RESPONSE])
        reply = random.choice(replies)
    else:
        reply = FALLBACK_RESPONSE
    self._history.append({"role": "user", "content": user_input})
    self._history.append({"role": "assistant", "content": reply})
    return reply
```

Line by line:
1. Tokenize and lemmatize the user's input — **exact same steps as during training**.
2. Convert to bag-of-words vector — must use the same vocabulary as training.
3. `np.array([bow])` — wraps the vector in an extra dimension. Keras expects shape `(batch_size, features)` so this becomes shape `(1, 389)`.
4. `self._model.predict(...)` — runs the neural network forward pass. Returns an array of shape `(1, 14)`.
5. `[0]` — takes the first (only) result, giving shape `(14,)`.
6. `np.argmax(prediction)` — finds the index of the highest value (most confident intent).
7. `confidence = float(prediction[idx])` — the confidence score (0.0 to 1.0).
8. If confidence ≥ 70%, look up the intent name, get its possible replies, pick one at random.
9. Otherwise, return the fallback message.

**Helper methods:**

```python
def _lemmatize_tokens(self, tokens: list[str]) -> list[str]:
    return [
        self._lemmatizer.lemmatize(t.lower())
        for t in tokens
        if t.isalpha()
    ]
```
- `t.isalpha()` — filters out punctuation and numbers (keeps only pure letters).
- `self._lemmatizer.lemmatize(t.lower())` — lowercases and reduces to root form.

```python
def _make_bow(self, tokens: list[str], vocab: list[str]) -> list[int]:
    return [1 if word in tokens else 0 for word in vocab]
```
For each word in the 389-word vocabulary, checks if it appears in the user's tokens.
Returns a list of 389 zeros and ones. This is the feature vector.

---

### `knowledge/samphor_kb.py` — The Answer Book

**Technical explanation:**
A module containing two module-level constants: `INTENT_RESPONSES` (a `dict[str, list[str]]`
mapping intent tags to lists of response strings) and `FALLBACK_RESPONSE` (a plain `str` returned
when classifier confidence falls below the threshold). No ML logic exists here.

**Plain English:**
This is literally a dictionary — a lookup table. The AI figures out the category ("ask_history"),
and then this file is where the actual text answers are stored. The engine picks one answer at
random from the list so conversations feel slightly varied rather than robotic and repetitive.

**Why keep answers separate from training data?**
`data/intents.json` has `"responses"` fields too, but they're not used. The reason is
**separation of concerns**: intents.json is about teaching the AI to classify; samphor_kb.py
is about what to say. You can improve answers without retraining. You can retrain without
worrying about answer wording.

**Structure:**
```python
INTENT_RESPONSES: dict[str, list[str]] = {
    "ask_definition": [
        "The Samphor is a barrel-shaped, two-headed drum...",
        "A Samphor is a traditional Khmer percussion instrument...",
    ],
    "ask_history": [
        "The Samphor traces its origins to the Angkor Empire...",
        "Historical evidence from temple bas-reliefs at Angkor Wat...",
    ],
    # ... 12 more intents
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
by the engine but kept for documentation). The file encodes the complete labeled training dataset
for the intent classifier.

**Plain English:**
This is the textbook the AI studies from. Each "intent" is a chapter, and each "pattern" is one
way a human might phrase a question about that topic. The more patterns you write, the better the
AI gets at recognizing that topic.

**The 14 intents and what they cover:**

| Intent Tag | Topic | Example Pattern |
|-----------|-------|----------------|
| `ask_definition` | What is the Samphor? | "What is the Samphor?", "Define the Samphor" |
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
| `greeting` | Hello messages | "Hi", "Hello", "Good morning" |
| `farewell` | Goodbye messages | "Bye", "See you", "Thank you, goodbye" |
| `out_of_scope` | Unrelated questions | "What's the weather?", "Tell me a joke" |

**Why ~20 patterns per intent?**
Neural networks need enough examples to find patterns. Too few (fewer than 5) and the model
memorizes exact sentences rather than learning concepts. ~20 gives enough variety while keeping
the dataset small enough to train in seconds.

**One-hot encoding of intent tags:**
During training, `"ask_history"` becomes `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
(a 1 in position 2, zeros everywhere else). This is **one-hot encoding** — a way to represent
categorical labels as numbers without implying any mathematical relationship between categories.

---

### `templates/index.html` — The Chat Interface

**Technical explanation:**
A single HTML document that serves as a Single Page Application (SPA). It contains embedded CSS
for styling and inline JavaScript. The JS uses the Fetch API to make asynchronous POST requests to
the `/chat` and `/reset` Flask endpoints. The chat log is rendered by dynamically creating DOM
elements via `document.createElement()` and appending them to a scrollable container div.

**Plain English:**
This is the web page you see in your browser. It has three parts baked into one file:

- **HTML** — the skeleton: the header, the chat box area, the text input, the Send button, the Reset button.
- **CSS** — the skin: colors, fonts, rounded corners, shadows, responsive sizing.
- **JavaScript** — the muscles: what happens when you click Send, how your message appears, how the
  reply comes back without the page reloading.

**HTML structure:**

```html
<div class="chat-wrapper">
    <header>
        <h1>Samphor Expert System</h1>
    </header>
    <div id="chat-box">
        <!-- Message bubbles are added here by JavaScript -->
    </div>
    <div class="input-area">
        <input id="user-input" type="text" placeholder="Ask about the Samphor..."/>
        <button id="send-btn">Send</button>
        <button id="reset-btn">↺</button>
    </div>
</div>
```

**CSS key concepts used:**

| CSS Property/Concept | What it does |
|---------------------|-------------|
| `display: flex` | Enables Flexbox layout — items can be aligned and distributed |
| `flex-direction: column` | Stack children top-to-bottom |
| `overflow-y: auto` | Scrollbar appears when content is taller than the container |
| `border-radius: 24px` | Rounded corners on inputs and buttons |
| `box-shadow` | Adds a subtle shadow behind the chat card for depth |
| `max-width: 700px` | Chat box never gets wider than 700px |
| `90vh` | Height is 90% of the browser viewport height |
| `margin-left: auto` | Together with `margin-right: auto`, centers the card horizontally |

**Color palette:**
- Background: `#f4f1ec` (warm off-white — feels like parchment)
- Header: `#8b3a0f` (deep clay red — earthy, Cambodian feel)
- User messages: `#7a2a0a` (dark red — right side of chat)
- Bot messages: `#f0e6d3` (warm beige — left side of chat)

**JavaScript functions:**

```javascript
function addMessage(text, role) {
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    div.textContent = text;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}
```
- `document.createElement("div")` — creates a new HTML `<div>` element in memory (not yet on the page).
- `div.className = \`msg ${role}\`` — sets its CSS class. Role is either `"user"` or `"bot"`.
  Template literals (backticks) allow embedding variables directly into strings.
- `div.textContent = text` — sets the text inside the div. `textContent` (not `innerHTML`)
  is used to prevent XSS (cross-site scripting) attacks — it treats the text as plain text, not HTML.
- `chatBox.appendChild(div)` — adds the div to the chat box (now it appears on screen).
- `chatBox.scrollTop = chatBox.scrollHeight` — auto-scrolls to the bottom so the latest message is visible.

```javascript
async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    addMessage(text, "user");
    userInput.value = "";
    userInput.disabled = true;
    sendBtn.disabled = true;

    try {
        const res = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text }),
        });
        const data = await res.json();
        addMessage(data.response, "bot");
    } catch (err) {
        addMessage("Error: could not reach the server.", "bot");
    } finally {
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
    }
}
```
Step by step:
1. Read input and trim whitespace. If empty, do nothing (`return`).
2. Show the user's message immediately (before the server replies).
3. Clear the input field and disable both input and button while waiting.
4. `await fetch("/chat", {...})` — sends an async HTTP POST to `/chat` with JSON body.
5. `await res.json()` — waits for the response and parses it as JSON.
6. Show the bot's reply.
7. `catch` — if the server is down or the network fails, show an error message.
8. `finally` — always re-enables input and refocuses, even if there was an error.

**Event listeners:**

```javascript
sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
});
```
The first line connects a click on the Send button to `sendMessage()`.
The second listens to every keypress in the text box — if the key pressed was "Enter", it calls
`sendMessage()`. This is why pressing Enter works the same as clicking Send.

---

### `requirements.txt` — The Dependency List

**Technical explanation:**
A pip-compatible requirements file specifying exact package versions for reproducible builds.
It pins all transitive dependencies (not just direct ones) to guarantee identical environments
across machines.

**Plain English:**
A shopping list of Python packages. When you run `pip install -r requirements.txt`, Python goes
and downloads and installs everything on the list. The version numbers (`==2.21.0`) ensure you
get the exact same software versions as the project was built with, which prevents bugs from
version mismatches.

**The key packages and why they exist:**

| Package | Version | Why it's here |
|---------|---------|---------------|
| `flask` | 3.1.3 | The web server framework — runs `app.py` |
| `tensorflow` | 2.21.0 | The deep learning library — builds and runs the neural network |
| `keras` | 3.14.1 | The high-level neural network API built on top of TensorFlow |
| `numpy` | 2.4.6 | Numerical arrays — used for the bag-of-words vectors and matrix math |
| `nltk` | 3.9.4 | Natural Language Toolkit — tokenization and lemmatization |
| `scikit-learn` | 1.8.0 | ML utilities (used indirectly by training pipeline) |
| `h5py` | 3.14.0 | Reads and writes HDF5 files — used to load `.h5` model files |
| `jinja2` | 3.1.6 | HTML template engine — Flask uses this to serve `index.html` |
| `werkzeug` | 3.1.8 | The WSGI utility library Flask is built on |

**Why TensorFlow 2.x and not 3.x?**
TensorFlow 2 is stable and widely used. TensorFlow 3 (if released) would require code changes.
Pinning to `2.21.0` ensures stability.

---

### `model/` — The Saved Brain Files

**Technical explanation:**
Three binary artifacts created by `train.py` and consumed by `MLEngine._load_artefacts()`.
`samphor_model.h5` is an HDF5 file containing the Keras model architecture (as JSON) and all
learned weight tensors. `words.pkl` and `classes.pkl` are Python pickle files containing
the vocabulary list and intent class list respectively.

**Plain English:**
These three files are the "saved brain." After training, the AI's learned knowledge is stored here.
Without them, the chatbot can't answer anything. They are excluded from git (`.gitignore`) because:
- They are large binary files (`.h5` can be megabytes).
- They can always be recreated by running `train.py`.
- Binary files don't version-control well (git can't show a human-readable diff).

| File | Size | Contents |
|------|------|---------|
| `samphor_model.h5` | ~1 MB | Neural network architecture + 389×128 + 128×64 + 64×14 weight matrices |
| `words.pkl` | ~5 KB | Python list of 389 root words (the vocabulary) |
| `classes.pkl` | ~1 KB | Python list of 14 intent tag strings |

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
ingredients (packages) at different versions. If you put everything in one kitchen, the projects
start fighting over which version of the ingredient to use. A virtual environment is like giving
each project its own mini-kitchen with its own separate fridge. They can't see each other's food.

**How it works:**
```
project/
└── venv/
    ├── Scripts/        (Windows) or bin/ (Mac/Linux)
    │   ├── python.exe  ← this is the isolated Python
    │   └── pip.exe     ← this installs into the venv only
    └── Lib/
        └── site-packages/
            ├── flask/
            ├── tensorflow/
            └── ... (all your packages)
```

When you run `./venv/Scripts/python.exe`, Python uses only the packages in `venv/Lib/site-packages/`.

---

### Object-Oriented Programming and Inheritance

**Technical explanation:**
OOP organizes code into **classes** — blueprints that bundle data (attributes) and behaviour
(methods) together. **Instances** are concrete objects created from a class using `ClassName()`.
**Inheritance** allows a class (the subclass) to reuse and extend the behaviour of another class
(the superclass). The subclass inherits all methods and attributes and can override them.

**Plain English:**
A class is like a cookie cutter — a template. An instance is an actual cookie made from the cutter.
Inheritance means a subclass is like a specialized cookie cutter that has everything the original
had, plus extra features.

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
            Implements: respond() using neural network
            Implements: reset() by clearing history
            Adds: train(), _make_bow(), _lemmatize_tokens(), etc.
```

When `app.py` does `engine.respond(user_message)`, it doesn't care whether `engine` is an
`MLEngine` or a `RuleBasedEngine`. Both guarantee that `respond()` exists. This is called
**polymorphism** — "many forms", one interface.

**Superclass method calling (`super().__init__()`):**
When a subclass defines `__init__`, it should call the parent's `__init__` too:
```python
class MLEngine(ChatEngine):
    def __init__(self):
        super().__init__()  # calls ChatEngine.__init__()
        self._model = None
        self._lemmatizer = WordNetLemmatizer()
        ...
```
Without `super().__init__()`, the parent class's initialization (like setting up `_history`)
would never run.

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
    def respond(self, user_input: str) -> str:
        ...   # No implementation — just a declaration

    @abstractmethod
    def reset(self) -> None:
        ...
```

If you tried to do `engine = ChatEngine()`, Python would immediately raise:
`TypeError: Can't instantiate abstract class ChatEngine with abstract method respond`

---

### The Flask Web Framework and HTTP

**Technical explanation:**
Flask is a WSGI (Web Server Gateway Interface) micro-framework. It maps URL patterns to Python
functions via route decorators. When the development server receives an HTTP request, it matches
the URL and method to a registered route, invokes the corresponding view function, and wraps the
return value in an HTTP response. Flask's `request` proxy object provides thread-local access to
the current request's data (headers, body, query parameters).

**Plain English:**
HTTP is the language web browsers and servers use to talk. A browser says "GET /chat-page" (a request),
and the server responds "200 OK, here's the HTML" (a response). Flask is the translator — you write
Python functions, and Flask handles all the HTTP plumbing to connect them to URLs.

**HTTP Request Methods:**
- **GET** — "Give me this resource." Used when you open a URL in a browser. No data body.
- **POST** — "Here's some data, process it." Used when submitting forms or sending chat messages.
  Carries a data body (in this case, JSON with the message text).

**HTTP Status Codes:**
- `200 OK` — everything worked.
- `400 Bad Request` — the client sent invalid data (used here when message is empty).
- `404 Not Found` — the URL doesn't exist.
- `500 Internal Server Error` — something crashed on the server.

**Flask Route Anatomy:**
```python
@app.route("/chat", methods=["POST"])  # URL pattern + allowed methods
def chat():                             # Python function (called a "view function")
    data = request.get_json()          # read the POST body as parsed JSON
    reply = engine.respond(data["message"])
    return jsonify({"response": reply}) # return JSON with 200 status
```

**`jsonify()` vs `json.dumps()`:**
`jsonify()` is Flask's helper — it sets the `Content-Type: application/json` header automatically.
`json.dumps()` is Python's raw JSON serializer — it just gives you a string.

---

### Natural Language Processing (NLP)

**Technical explanation:**
NLP is the subfield of AI concerned with enabling computers to understand, generate, and manipulate
human language. In this project, the NLP pipeline performs intent classification — mapping
free-form natural language input to a predefined set of categories using statistical methods.

**Plain English:**
Human language is messy — "What IS the Samphor?", "tell me ABOUT samphor", and "define the samphor
drum" all mean the same thing but look completely different to a computer. NLP is the set of
techniques that help computers deal with that messiness.

**The NLP pipeline in this project has 3 stages:**
1. **Tokenization** — split text into pieces.
2. **Lemmatization** — reduce each piece to its root form.
3. **Bag-of-Words** — convert the pieces to numbers a neural network can process.

---

### Tokenization

**Technical explanation:**
Tokenization is the segmentation of a character sequence into a list of tokens (discrete meaningful
units — typically words and punctuation). NLTK's `word_tokenize()` uses a pre-trained Punkt
sentence tokenizer followed by a rule-based word tokenizer (TreebankWordTokenizer) that handles
contractions, punctuation, and special characters correctly.

**Plain English:**
Tokenization is like cutting a sentence into individual word-chips. "What is the Samphor?" becomes
`["What", "is", "the", "Samphor", "?"]`. Each word and punctuation mark is a separate token.
Computers can't understand "What is the Samphor?" as a whole — they need to look at each piece.

**Examples:**
```python
import nltk
nltk.word_tokenize("What is the Samphor?")
# → ["What", "is", "the", "Samphor", "?"]

nltk.word_tokenize("I'm asking about it.")
# → ["I", "'m", "asking", "about", "it", "."]
# Note: contractions are split correctly
```

**Why not just use `.split(" ")`?**
`"What is the Samphor?".split(" ")` gives `["What", "is", "the", "Samphor?"]` — "Samphor?" has the
question mark glued to it. NLTK handles edge cases like contractions, hyphens, and punctuation properly.

---

### Lemmatization

**Technical explanation:**
Lemmatization is morphological reduction — mapping inflected word forms to their canonical base
form (the lemma) using vocabulary and morphological analysis. NLTK's `WordNetLemmatizer` looks up
words in WordNet (a lexical database) to find their root forms. Unlike stemming (rule-based suffix
removal), lemmatization always produces a real dictionary word.

**Plain English:**
"Playing", "plays", "played" all mean the same root concept: "play." Lemmatization reduces all of
them to "play" so the AI treats them as the same word. Without it, "plays the drum" and "playing
the drum" would look different to the model because "plays" and "playing" are different strings.

**Examples:**
```python
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()

lem.lemmatize("playing")   # → "playing" (without specifying POS)
lem.lemmatize("playing", pos="v")  # → "play" (verb form)
lem.lemmatize("drums")     # → "drum"
lem.lemmatize("histories") # → "history"
lem.lemmatize("running", pos="v")  # → "run"
```

**Stemming vs Lemmatization:**
- Stemming (e.g., Porter Stemmer): fast, rule-based, often produces non-words. "running" → "run", but "studies" → "studi".
- Lemmatization: uses a dictionary, always produces real words. Slower but more accurate.

---

### Bag of Words

**Technical explanation:**
The Bag-of-Words (BoW) model represents a document as a multiset (or binary vector) over a fixed
vocabulary. Each position in the vector corresponds to one vocabulary word; the value is 1 if the
word appears (binary BoW) or the count of occurrences (count BoW). Word order and grammar are
discarded — hence "bag." The result is a fixed-length numerical vector that a neural network can
consume.

**Plain English:**
Imagine you have a list of 389 words (the vocabulary). For each message, you create a row of
389 zeros. For each word in the message, you put a 1 in the matching position. That's it.
"What is the Samphor?" might produce something like:
```
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, ...]
   ^       ^                   ^
  "a"    "drum"              "samphor"
```

The neural network then reads these numbers (not the words) to make its prediction.

**Why does order not matter?**
For intent classification (figuring out the TOPIC), word order is less important than WHICH words
appear. "What is the Samphor?" and "The Samphor — what is it?" contain the same words and thus
produce the same BoW vector. The model doesn't lose meaningful information for this task.

**Why binary (0 or 1) instead of counts?**
In short training patterns (10–20 words each), a word almost never appears more than once. Using 1
vs. count doesn't change anything, and binary vectors are simpler and faster.

---

### Neural Networks from Scratch

**Technical explanation:**
A feedforward neural network is a directed acyclic graph of computational nodes (neurons) organized
into layers. Each neuron computes a weighted sum of its inputs plus a bias term, then applies an
activation function. During training, weights are updated via backpropagation (computing gradients
with the chain rule) guided by an optimizer (SGD) to minimize a loss function.

**Plain English:**
Think of a neural network as a voting machine with three rooms:

**Room 1 (128 voters):** Each of the 389 input numbers (the BoW vector) gets sent to all 128 voters.
Each voter has its own weights — how much they trust each input. They add up their weighted inputs
and decide a number. That's one "neuron."

**Room 2 (64 voters):** The 128 numbers from Room 1 come in, same process, produces 64 numbers.

**Output Room (14 voters):** Takes the 64 numbers, produces 14 numbers — one for each intent.
The highest number wins. That intent is the predicted category.

**How the weights get set:**
At the start, weights are random. During training:
1. Feed in a pattern → get a prediction (probably wrong at first).
2. Compare to the correct answer → compute "how wrong" (the loss).
3. Work backward through the network, adjusting weights to reduce the error.
4. Repeat 56,000 times (280 patterns × 200 epochs).
After all that, the weights encode the patterns the network has learned.

**Why "deep" learning?**
This network has multiple hidden layers (two of them: 128 and 64). "Deep" just means more than one
hidden layer. More layers allow the network to learn more abstract patterns.

**The network in numbers:**
```
Layer 1: 389 inputs × 128 neurons = 49,792 weights + 128 biases = 49,920 parameters
Layer 2: 128 inputs × 64 neurons  =  8,192 weights +  64 biases =  8,256 parameters
Output:   64 inputs × 14 neurons  =    896 weights +  14 biases =    910 parameters
Total trainable parameters: ~59,086
```

---

### Activation Functions: ReLU and Softmax

**Technical explanation:**
Activation functions introduce non-linearity into neural networks. Without them, stacking multiple
linear layers would be mathematically equivalent to a single linear layer (and couldn't learn
non-linear patterns).

**ReLU (Rectified Linear Unit):** `f(x) = max(0, x)`.
Outputs x if positive, 0 if negative. Computationally cheap, reduces the vanishing gradient problem,
and empirically performs well on most tasks.

**Softmax:** `f(z_i) = e^{z_i} / Σ e^{z_j}`.
Converts a vector of raw scores (logits) into a probability distribution — values between 0 and 1
that sum to exactly 1.0. Used in the output layer for multi-class classification.

**Plain English:**

**ReLU** is like a light switch that can dim. Negative signal? Output zero (neuron is "off").
Positive signal? Pass it through unchanged. This simple rule lets the network decide which neurons
should respond to which patterns.

**Softmax** is like a confidence voting system. After all the math, you have 14 raw scores (one per
intent). Softmax converts them to percentages that add to 100%. So instead of raw numbers like
`[2.3, 0.1, 5.6, ...]`, you get `[0.08, 0.01, 0.85, ...]` — the model is 85% confident it's
`ask_definition`.

```
Before softmax:  [2.1, 0.5, 1.8, 4.2, 0.1, ...]   ← raw scores, hard to interpret
After softmax:   [0.07, 0.01, 0.05, 0.82, 0.002, ...] ← probabilities, sum = 1.0
                                         ↑ 82% confident = most likely category
```

---

### Dropout Regularization

**Technical explanation:**
Dropout is a regularization technique that randomly sets a fraction `p` of neuron activations to
zero during each training step (but not during inference). This prevents co-adaptation of neurons
— individual neurons cannot rely on specific other neurons always being present, forcing each to
learn more independently useful representations. The result is reduced overfitting.

At inference time, all neurons are active, and their outputs are implicitly scaled by `(1-p)` to
account for the fact that only `(1-p)` fraction were active during training.

**Plain English:**
Overfitting is when a student memorizes the exact test questions instead of understanding the
subject. Dropout is like randomly covering some students' notes during each practice session.
They can't rely on any specific piece of information always being there, so they're forced to
learn the overall patterns instead of memorizing specific examples.

In this project, `Dropout(0.5)` means 50% of neurons in the previous layer are randomly turned off
during each training batch. The network has to learn patterns using only half its neurons at a time,
making it more robust.

**During training vs. inference:**
```
Training:   [n1=on, n2=off, n3=on, n4=off, ...]  ← 50% randomly zeroed
Inference:  [n1=on, n2=on,  n3=on, n4=on,  ...]  ← all active (Keras handles scaling automatically)
```

---

### The SGD Optimizer and Backpropagation

**Technical explanation:**
Stochastic Gradient Descent (SGD) is an iterative first-order optimization algorithm. At each step
it computes the gradient of the loss function with respect to the model parameters on a mini-batch
of training examples (here `batch_size=5`), then updates parameters in the direction that reduces
loss: `w ← w - lr × ∇L(w)`.

Nesterov momentum computes the gradient at a "look-ahead" position:
`v_t = γv_{t-1} + lr × ∇L(w - γv_{t-1})`, `w ← w - v_t`.
This gives faster convergence than standard momentum.

Backpropagation is the algorithm to compute `∇L(w)` — the gradient of the loss with respect to
every weight. It applies the chain rule of calculus layer-by-layer backward from the output.

**Plain English:**

**Gradient Descent** — imagine you're blindfolded on a hilly landscape, trying to reach the lowest
valley. You feel which direction is downhill under your feet and take a step that way. The "valley"
is the minimum loss (fewest errors). Each step is one weight update.

**Stochastic** (SGD) — instead of looking at all 280 training examples before taking a step
(too slow), you look at just 5 (the mini-batch), take a step, then look at 5 more. It's noisier
but much faster.

**Momentum** — imagine rolling a ball downhill instead of walking. The ball picks up speed in the
direction it's rolling. If the gradient keeps pointing the same way (consistent downhill), the
optimizer takes bigger and bigger steps. If the gradient reverses, it slows down. This helps
escape "flat" areas in the loss landscape.

**Nesterov** — before computing the gradient, the optimizer "peeks" ahead in the direction momentum
is taking it. It makes a smarter guess and converges faster.

**Backpropagation** — after the network makes a prediction, we know how wrong it was. To fix it,
we need to know how each weight contributed to the error. Backpropagation traces the error backward
through the network, computing each weight's share of the blame (the gradient). Then SGD uses those
gradients to update the weights.

---

### Loss, Accuracy, and Epochs

**Technical explanation:**
**Loss (Categorical Crossentropy):** `-Σ y_i log(ŷ_i)` where `y_i` are the true one-hot labels
and `ŷ_i` are the predicted probabilities. Measures the divergence between the predicted
distribution and the true distribution. Lower is better; zero means perfect prediction.

**Accuracy:** The fraction of samples for which `argmax(ŷ) == argmax(y)`. Ranges from 0.0 to 1.0.
For this project, anything below 0.70 triggers a FAIL.

**Epoch:** One complete pass through the entire training dataset. With 280 samples and batch_size=5,
one epoch = 56 gradient updates. After 200 epochs = 11,200 total updates.

**Plain English:**

**Loss** — a number measuring how wrong the model is. If the model is 85% confident the answer is
"ask_history" but the correct answer was "ask_material", the loss is high. If it's 99% confident
in the right answer, loss is near zero. The goal of training is to minimize loss.

**Accuracy** — what percentage of the 280 training examples the model gets right. 99% accuracy
means it gets 277 out of 280 correct. This is the easier-to-understand metric.

**Epoch** — one complete trip through all training data. Like re-reading the same textbook chapter.
More epochs = more practice. But too many epochs can lead to overfitting (memorizing instead of learning).
200 epochs was chosen because the model consistently reaches >99% accuracy by then on this dataset.

---

### Pickle and Model Serialization

**Technical explanation:**
`pickle` is Python's built-in object serialization protocol. It converts Python objects (lists,
dicts, class instances) to binary byte streams (serialization) and back (deserialization). The
`.pkl` files store `self._words` (a `list[str]`) and `self._classes` (a `list[str]`).

The `.h5` format (HDF5 — Hierarchical Data Format version 5) is used by Keras to store model
architecture (as JSON) and weight tensors (as numerical arrays). It supports large numerical
datasets efficiently.

**Plain English:**
When your Python script ends, every variable disappears from memory. Pickle is like taking a
snapshot of a Python object and saving it to a file. When you load it back, the object is
exactly as you left it.

```python
# Saving
with open("model/words.pkl", "wb") as f:   # "wb" = write binary
    pickle.dump(vocabulary_list, f)

# Loading
with open("model/words.pkl", "rb") as f:   # "rb" = read binary
    vocabulary_list = pickle.load(f)
```

The `.h5` file is more complex — it's essentially a compressed folder structure inside one file,
designed for large numerical datasets (like neural network weight matrices with thousands of numbers).

**Security note:** Never load pickle files from untrusted sources — pickle can execute arbitrary
code during loading. This is safe here because the files are generated by your own code.

---

### JSON and Data Exchange

**Technical explanation:**
JSON (JavaScript Object Notation) is a text-based data interchange format. It is a strict subset
of JavaScript's object literal syntax. Python's `json` module provides `json.load()` (file to dict),
`json.loads()` (string to dict), `json.dump()` (dict to file), `json.dumps()` (dict to string).
Flask's `jsonify()` serializes a dict to a JSON string and wraps it in an HTTP response with
`Content-Type: application/json`.

**Plain English:**
JSON is a universal language for data. Both Python and JavaScript can read and write it. When the
browser sends a message to Flask, it wraps it in JSON:
```json
{"message": "What is the Samphor?"}
```
When Flask replies, it also uses JSON:
```json
{"response": "The Samphor is a barrel-shaped drum..."}
```

JSON has only 6 data types: string, number, boolean, null, array, and object. That simplicity is
why almost every web API in the world uses it.

**Python dict vs JSON:**
```python
# Python dict
{"key": "value", "list": [1, 2, 3], "nested": {"a": True}}

# JSON (same data, text format)
'{"key": "value", "list": [1, 2, 3], "nested": {"a": true}}'
#                                                       ↑ Python True → JSON true (lowercase)
```

---

### AJAX and the fetch() API

**Technical explanation:**
AJAX (Asynchronous JavaScript and XML — though JSON has replaced XML in practice) refers to
making HTTP requests from the browser without triggering a full page navigation. The modern
implementation is the Fetch API (`fetch()`), which returns a `Promise` that resolves to a
`Response` object. The response body can be consumed as text, JSON, Blob, etc.

**Plain English:**
Normally, clicking a button on a web page reloads the entire page (you see it go blank and reload).
AJAX lets the browser talk to the server in the background while the page stays visible. The chat
box works because of AJAX — your messages and the replies appear without any page reload.

```javascript
// "I promise this will eventually return a result" — that's what fetch() does
const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: "What is the Samphor?" })
});
const data = await response.json();
console.log(data.response); // "The Samphor is..."
```

`fetch()` returns a Promise — think of it as an IOU. `await` pauses the function until the IOU
is paid (the response arrives), then continues.

---

### Async/Await in JavaScript

**Technical explanation:**
JavaScript is single-threaded. Async/await is syntactic sugar over Promises (which wrap callbacks).
An `async function` always returns a Promise. `await expression` suspends the async function until
the awaited Promise resolves, without blocking the call stack (the event loop continues to process
other events). This is cooperative multi-tasking via the microtask queue.

**Plain English:**
Your browser has one thread (one worker). If you make that worker wait for a server response
(which could take 500ms), the entire page would freeze — you couldn't type, scroll, or click
anything. `async/await` tells JavaScript: "start this request, but while it's waiting, go do
other things. Come back when the response arrives." The page stays responsive.

```javascript
// WITHOUT async/await — the page would freeze
function badSendMessage() {
    // This blocks the entire page for 500ms
    const response = slowFetchRequest(); // can't actually do this — JavaScript doesn't work this way
}

// WITH async/await — correct and non-blocking
async function sendMessage() {
    const response = await fetch("/chat", {...});  // "go do other things while waiting"
    const data = await response.json();            // "wait for JSON parsing too"
    addMessage(data.response, "bot");              // now update the UI
}
```

**`try/catch/finally`:**
- `try` — attempt the risky code.
- `catch (err)` — if anything throws an error, run this block. Prevents the whole function from crashing.
- `finally` — always runs, whether there was an error or not. Used here to re-enable the input button.

---

### CSS Flexbox Layout

**Technical explanation:**
Flexbox (Flexible Box Layout) is a CSS layout model. A **flex container** (element with `display: flex`)
distributes space among its **flex items** (direct children) along a main axis and cross axis.
Key properties: `flex-direction` (axis direction), `justify-content` (main axis alignment),
`align-items` (cross axis alignment), `flex: 1` on an item (grow to fill remaining space),
`overflow-y: auto` on the container (scroll when content overflows).

**Plain English:**
Flexbox is like a smart shelf system. You tell the shelf "arrange your items in a column, and
make the middle item stretch to fill all available space." Without Flexbox, you'd need to calculate
pixel positions for everything. With it, the layout adjusts automatically for different screen sizes.

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

.input-area {
    display: flex;           /* Put input and buttons side by side */
    gap: 8px;
}

#user-input {
    flex: 1;                 /* Input stretches to fill remaining width */
}
```

Without `flex: 1` on `#chat-box`, the chat box would only be as tall as its content, and the
input area wouldn't be pinned to the bottom.

---

## 5. Setup — Step by Step

### Requirements
- Python **3.13** (not 3.14 — TensorFlow 2.x does not support 3.14 yet)
- A terminal / command prompt
- Internet connection for the first install

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
# Windows (PowerShell)
"C:/Users/YourName/AppData/Local/Programs/Python/Python313/python.exe" -m venv venv

# Mac / Linux
python3.13 -m venv venv
```

A new `venv/` folder appears. Do not edit it manually.

---

### Step 3 — Install the Packages

```powershell
# Windows
./venv/Scripts/python.exe -m pip install flask nltk numpy scikit-learn pandas tensorflow

# Mac / Linux
./venv/bin/python -m pip install flask nltk numpy scikit-learn pandas tensorflow
```

This takes a few minutes the first time (downloading ~500 MB including TensorFlow).

---

### Step 4 — Train the Model

```powershell
# Windows
./venv/Scripts/python.exe train.py

# Mac / Linux
./venv/bin/python train.py
```

Expected output:
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

### Step 5 — Start the Chatbot

```powershell
# Windows
./venv/Scripts/python.exe app.py

# Mac / Linux
./venv/bin/python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## 6. How to Run

| Task | Command |
|------|---------|
| Train the model | `./venv/Scripts/python.exe train.py` |
| Start the chatbot | `./venv/Scripts/python.exe app.py` |
| Open in browser | http://127.0.0.1:5000 |
| Stop the server | `Ctrl + C` in terminal |
| Retrain after changes | Stop server → run `train.py` → restart `app.py` |

---

## 7. How to Add a New Topic

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
    "Is Samphor making a family tradition"
  ],
  "responses": ["placeholder — real answers go in samphor_kb.py"]
}
```

More patterns = better classification. Aim for at least 10–15.

**Step 2 — Add replies to `knowledge/samphor_kb.py`:**
```python
"ask_makers": [
    "Samphor drums are crafted by specialist artisans in Phnom Penh and Siem Reap. "
    "The craft is passed down through families over generations of apprenticeship.",
    "Traditional Samphor makers use jackfruit wood and animal skin, skills learned "
    "from masters. The Royal University of Fine Arts documents these techniques.",
],
```

Add at least 2 replies so the bot has variety.

**Step 3 — Retrain:**
```powershell
./venv/Scripts/python.exe train.py
```

**Step 4 — Restart the server:**
```powershell
./venv/Scripts/python.exe app.py
```

The bot can now answer questions about Samphor makers.

---

## 8. Switching Engines

Edit two lines in `app.py`:

**Default (ML engine):**
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
- Model files haven't been trained yet.
- You want to test the website layout without the ML overhead.
- You're debugging the Flask routes and don't need AI.

---

## 9. Learning Roadmap

This is a structured plan to go from "I know some basics" to fully understanding and extending
every part of this project. Each phase builds on the previous one.

---

### Phase 0 — Prerequisites (Before You Start)

These are tools you need to have set up, not topics you need to master first.

| Task | Why | How Long |
|------|-----|---------|
| Install Python 3.13 | Everything runs on Python | 30 minutes |
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
| Lists and list comprehensions | `_words`, `_classes`, BoW vectors | Essential |
| Dictionaries | `INTENT_RESPONSES`, `RULES` | Essential |
| Tuples | `RULES` keys, `(tokens, tag)` pairs | Essential |
| Functions (def, return, parameters) | All helper methods | Essential |
| `if/elif/else` and `for` loops | `respond()`, training loop | Essential |
| `open()`, `with` blocks, `json.load()` | Loading `intents.json` | Essential |
| `import` and modules | Every file imports other files | Essential |
| Classes and `__init__` | `ChatEngine`, `MLEngine` | Essential |
| Inheritance (`class B(A)`) | `MLEngine(ChatEngine)` | Essential |
| `super().__init__()` | MLEngine and RuleBasedEngine constructors | Essential |
| Abstract classes (`ABC`, `@abstractmethod`) | `ChatEngine` | Important |
| Decorators (`@something`) | `@app.route`, `@abstractmethod` | Important |
| Type hints (`str`, `list[str]`, `-> None`) | All function signatures | Good to know |
| `pickle.dump()` / `pickle.load()` | Saving/loading vocabulary | Important |
| List comprehensions (`[x for x in y if z]`) | `_lemmatize_tokens`, `_make_bow` | Important |
| `random.choice()` | Picking random replies | Easy |
| `any()`, `all()` built-in functions | RuleBasedEngine keyword matching | Good to know |
| f-strings / template strings | Print statements, logging | Easy |

**What to build to practice:**
- A simple command-line quiz program (uses functions, loops, input/output)
- A to-do list program (uses lists, dicts, file I/O)
- A simple class hierarchy (Animal → Dog, Cat)

**Resources:**
- [python.org/about/gettingstarted](https://www.python.org/about/gettingstarted/)
- *Automate the Boring Stuff with Python* (free online) — Chapters 1–10
- Search "Python OOP tutorial Corey Schafer" on YouTube (very good series)

---

### Phase 2 — Web Development Basics (Weeks 4–5)

**Goal:** Understand `templates/index.html` and how the browser talks to Flask.

**Topics to master:**

| Topic | Where it appears | Priority |
|-------|-----------------|---------|
| HTML document structure (`<html>`, `<head>`, `<body>`) | `index.html` skeleton | Essential |
| HTML elements: `<div>`, `<h1>`, `<input>`, `<button>` | The chat UI | Essential |
| HTML attributes: `id`, `class`, `type`, `placeholder` | Chat elements | Essential |
| CSS selectors (`.class`, `#id`, element) | Styling in `<style>` | Essential |
| CSS Box Model (margin, padding, border) | Spacing and layout | Essential |
| CSS Flexbox (`display: flex`, `flex-direction`, `flex: 1`) | Chat layout | Essential |
| CSS Colors (`#hex`, `rgb()`) | Color palette | Easy |
| CSS `overflow-y: auto` | Scrollable chat box | Important |
| JavaScript variables (`const`, `let`) | All JS code | Essential |
| JavaScript functions | `addMessage()`, `sendMessage()` | Essential |
| `document.getElementById()` | Getting DOM elements | Essential |
| `document.createElement()` | Creating message bubbles | Essential |
| `.appendChild()` | Adding bubbles to chat box | Essential |
| `.textContent` vs `.innerHTML` | Security: use textContent | Essential |
| Event listeners (`.addEventListener()`) | Click and keydown events | Essential |
| `e.key === "Enter"` | Enter to send | Easy |

**What to build to practice:**
- A static HTML/CSS page (a personal profile or simple webpage)
- A simple interactive page (a to-do list that adds/removes items with JavaScript)

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
| HTTP status codes (200, 400, 404, 500) | Flask responses | Important |
| HTTP headers (`Content-Type: application/json`) | fetch() headers | Important |
| JSON structure and data types | `{"message": "..."}` | Essential |
| `JSON.stringify()` in JavaScript | Sending data to Flask | Essential |
| `response.json()` in JavaScript | Parsing Flask's reply | Essential |
| Flask `@app.route()` decorator | Registering routes | Essential |
| Flask `request.get_json()` | Reading POST body | Essential |
| Flask `jsonify()` | Returning JSON responses | Essential |
| Flask `render_template()` | Serving index.html | Essential |
| Async functions in JavaScript | `async function sendMessage()` | Essential |
| `await` keyword | Waiting for fetch results | Essential |
| Promises (what `async/await` wraps) | Under the hood of fetch | Important |
| `try/catch/finally` | Error handling in sendMessage | Essential |
| AJAX concept (no page reload) | The whole chat flow | Essential |
| Python `if __name__ == "__main__"` | app.py entry point | Important |
| Flask `debug=True` | Auto-reload during development | Easy |

**What to build to practice:**
- A simple Flask app with 2 routes that returns JSON
- Connect that Flask app to a frontend with JavaScript fetch()
- Build a simple number guessing game: frontend sends a guess, backend validates it

**Resources:**
- [Flask Quickstart](https://flask.palletsprojects.com/en/stable/quickstart/)
- [MDN: HTTP overview](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview)
- [javascript.info/async-await](https://javascript.info/async-await)

---

### Phase 4 — Natural Language Processing (Week 7)

**Goal:** Understand how human text gets converted to numbers the neural network can read.

**Topics to master:**

| Topic | Where it appears | Priority |
|-------|-----------------|---------|
| What NLP is | The whole ML pipeline | Essential |
| Tokenization concept | `nltk.word_tokenize()` | Essential |
| `nltk.word_tokenize()` in practice | Training and inference | Essential |
| Lemmatization vs stemming | `WordNetLemmatizer` | Essential |
| `WordNetLemmatizer.lemmatize()` | `_lemmatize_tokens()` | Essential |
| Vocabulary / corpus concept | `self._words` (389 words) | Essential |
| Bag-of-Words model | `_make_bow()` | Essential |
| Why word order doesn't matter for intent | BoW design decision | Important |
| One-hot encoding for class labels | `to_categorical()` | Essential |
| `nltk.download()` (required for NLTK data) | First-time setup | Important |
| Stop words (and why we don't remove them here) | NLP preprocessing choice | Good to know |

**What to build to practice:**
- Write a script that tokenizes and lemmatizes a paragraph of text
- Write a script that builds a vocabulary from a set of sentences
- Write a script that converts sentences to bag-of-words vectors manually

**Resources:**
- [NLTK Book — Chapter 1 & 2](https://www.nltk.org/book/) (free online)
- Search "bag of words NLP explained" on YouTube
- Search "tokenization lemmatization Python tutorial"

---

### Phase 5 — Neural Networks and Deep Learning (Weeks 8–10)

**Goal:** Understand what the neural network is doing, why it works, and how to tweak it.

**Topics to master:**

| Topic | Where it appears | Priority |
|-------|-----------------|---------|
| What a neuron is (weighted sum + activation) | Every Dense layer | Essential |
| What a layer is (collection of neurons) | `Dense(128, ...)` | Essential |
| Forward pass (input → prediction) | Inference in `respond()` | Essential |
| Loss functions — categorical crossentropy | `model.compile(loss=...)` | Essential |
| What accuracy measures | `metrics=["accuracy"]` | Essential |
| Backpropagation concept | How weights are updated | Essential |
| Gradient descent intuition | The optimizer's job | Essential |
| Stochastic vs batch gradient descent | `batch_size=5` | Important |
| Momentum in SGD | `momentum=0.9` | Important |
| Nesterov momentum | `nesterov=True` | Good to know |
| Learning rate | `learning_rate=0.01` | Essential |
| Overfitting — when the model memorizes | Why Dropout is needed | Essential |
| Dropout regularization | `Dropout(0.5)` | Essential |
| ReLU activation function | Hidden layers | Essential |
| Softmax activation function | Output layer | Essential |
| `np.argmax()` — finding the highest value | Intent prediction | Essential |
| Epochs — multiple passes through data | `epochs=200` | Essential |
| `model.fit()` | Training call | Essential |
| `model.predict()` | Inference call | Essential |
| Keras Sequential API | Model building | Essential |
| `Dense()` layer | Each layer in the model | Essential |
| `model.compile()` | Setting up training | Essential |
| `model.save()` / `load_model()` | Persisting the model | Essential |

**What to build to practice:**
- Build a simple Keras model that classifies handwritten digits (MNIST — the "hello world" of ML)
- Modify this project: change the number of hidden neurons and observe the effect on accuracy
- Try training with different learning rates and compare results

**Resources:**
- [3Blue1Brown — Neural Networks playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — WATCH FIRST, best visual explanation
- [keras.io — Getting Started](https://keras.io/getting_started/)
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)
- *Deep Learning with Python* by François Chollet (the Keras creator) — the definitive book

---

### Phase 6 — Putting It All Together (Week 11)

**Goal:** Be able to extend, modify, and debug every part of this project.

**Exercises:**
1. **Add 3 new intents** to `intents.json` and `samphor_kb.py`. Retrain and verify they work.
2. **Change the neural network architecture**: Add a third hidden layer. Observe the effect.
3. **Change the confidence threshold** from 70% to 85%. Test edge cases.
4. **Add a new API endpoint** to Flask, e.g. `GET /history` that returns the conversation log as JSON.
5. **Style the frontend**: Change the color scheme, add the Cambodian flag emoji, change the font.
6. **Add typing indicators**: Show "Bot is typing..." while waiting for the server response.
7. **Implement conversation context**: Modify `respond()` to include the last 3 messages when generating a reply.

---

### Phase 7 — Going Further (Month 3+)

After mastering this project, these are natural next steps:

| Next Topic | Why It Builds on This Project |
|-----------|-------------------------------|
| **REST API design** | Make the chatbot an API others can call |
| **Database integration (SQLite/PostgreSQL)** | Persist conversation history across restarts |
| **Transformer models (BERT, GPT)** | Replace the bag-of-words + Dense model with state-of-the-art NLP |
| **Hugging Face Transformers library** | Pre-trained models for intent classification |
| **Deployment (Heroku, Railway, AWS)** | Put your chatbot on the internet |
| **Docker** | Package the whole project into a container for reliable deployment |
| **Unit testing (pytest)** | Test `_make_bow()`, `_lemmatize_tokens()` in isolation |
| **CI/CD (GitHub Actions)** | Automatically test and retrain when you push changes |

---

### Estimated Timeline

| Phase | Topic | Time (consistent daily practice) |
|-------|-------|----------------------------------|
| 0 | Tools setup | 1–2 days |
| 1 | Python foundations | 2–3 weeks |
| 2 | HTML + CSS + basic JS | 1–2 weeks |
| 3 | HTTP + Flask + async JS | 1 week |
| 4 | NLP (tokenization, lemmatization, BoW) | 1 week |
| 5 | Neural networks + Keras | 2–3 weeks |
| 6 | Integration + extensions | 1 week |
| **Total** | | **8–11 weeks** |

"Consistent daily practice" means 1–2 hours per day, 5 days per week.
If you do more, you'll finish faster. The ML phase (Phase 5) is the hardest and takes the longest.

---

## 10. Glossary

| Term | Technical meaning | Plain English |
|------|--------------------|---------------|
| **Abstract Base Class (ABC)** | A class with `@abstractmethod` declarations that cannot be instantiated directly; enforces interface contracts on subclasses | A "contract class" — it says "you must implement these methods" without saying how |
| **Accuracy** | The fraction of predictions that match the true labels. `correct / total` | What percentage of the training questions the model answered correctly |
| **Activation function** | A non-linear function applied to a neuron's output, enabling the network to learn non-linear patterns | The rule a neuron uses to decide how loud its output is |
| **AJAX** | Asynchronous JavaScript and XML — sending HTTP requests without page reload | Making the browser talk to the server silently in the background |
| **Async/Await** | JavaScript syntax for writing asynchronous code that looks synchronous; built on Promises | "Start this task, go do other things while waiting, come back when it's done" |
| **Backpropagation** | Algorithm to compute the gradient of the loss with respect to every weight, using the chain rule of calculus | The math that figures out how much each weight contributed to the error |
| **Bag of Words (BoW)** | Representing a document as a binary or count vector over a fixed vocabulary, discarding word order | Converting a sentence to a row of 0s and 1s — 1 for each word that appears |
| **Batch size** | The number of training examples used to compute one gradient update | How many questions the network studies before adjusting its weights |
| **Categorical crossentropy** | Loss function for multi-class classification: `-Σ y log(ŷ)`. Measures divergence between true and predicted distributions | A number measuring how wrong the predictions are for classification tasks |
| **Confidence threshold** | The minimum prediction probability required to return a non-fallback response | The minimum certainty required before the bot commits to an answer |
| **Dense layer** | A fully connected neural network layer where every input connects to every output neuron | A layer where every neuron talks to every neuron in the next layer |
| **Decorator** | A Python function that wraps another function, adding behavior before or after it runs (`@decorator`) | A label above a function that gives it extra powers |
| **Dropout** | Regularization technique that randomly zeroes some neuron outputs during training to prevent overfitting | Randomly turning off neurons during practice so the network learns to be robust |
| **Epoch** | One complete pass through all training data | One full study session through all training examples |
| **Feature vector** | A fixed-length numerical representation of an input sample (here: the BoW vector) | The list of numbers that represents a piece of text for the neural network |
| **Flask** | A Python WSGI micro-framework for building web applications | The software that creates the website and handles browser requests |
| **GET request** | HTTP method to retrieve a resource without side effects | "Give me this page" — what the browser does when you type a URL |
| **Gradient descent** | Iterative optimization: update parameters in the direction that reduces the loss | Rolling downhill in the error landscape until you reach the bottom |
| **Inference** | Running a trained model on new input to get a prediction | Using the trained AI to answer a question |
| **Intent** | The category of user request (e.g., `ask_history` = user wants historical information) | The topic category of what the user is asking about |
| **JSON** | JavaScript Object Notation — a text format for structured data using `{}`, `[]`, strings, numbers | A universal text format for sending structured data between programs |
| **Keras** | High-level deep learning API (runs on top of TensorFlow) | The simpler, friendlier way to build neural networks |
| **Lemmatization** | Morphological reduction of an inflected word to its canonical base form (lemma) using a vocabulary | Reducing words like "playing" to their root "play" |
| **Lemma** | The base dictionary form of a word (e.g., "play" is the lemma of "playing", "plays", "played") | The root/base form of a word |
| **Loss** | A scalar measuring how wrong the model's predictions are on a batch; minimized during training | A score for "how badly is the model doing" — lower is better |
| **Momentum** | SGD enhancement that accumulates a velocity vector in the gradient direction across updates | The optimizer remembers which direction it was heading and keeps going that way |
| **Neural network** | A parameterized function composed of layers of neurons, trained to map inputs to outputs | A multi-layer voting machine that learned patterns from examples |
| **Neuron** | A computational unit: weighted sum of inputs + bias → activation function | One voter in the network that takes numbers in and outputs a single number |
| **Nesterov momentum** | A variant of momentum that computes the gradient at a look-ahead position | An improved version of momentum that peeks ahead before taking a step |
| **NLTK** | Natural Language Toolkit — Python library for NLP tasks (tokenization, lemmatization, parsing, etc.) | Python's Swiss army knife for working with human language |
| **NumPy** | Numerical Python — library for fast multi-dimensional array operations | The math library for working with arrays and matrices of numbers |
| **One-hot encoding** | Representing a categorical label as a binary vector with a single 1 at the label's index | A way to represent categories as rows of zeros with one 1 |
| **Overfitting** | When a model performs well on training data but poorly on new data — memorized instead of learned | The model memorized the training examples instead of learning the underlying pattern |
| **Pattern** | An example user query associated with an intent in `intents.json` | One example of how a user might phrase a question about a topic |
| **Pickle** | Python's binary serialization protocol (`pickle.dump()` / `pickle.load()`) | Python's way of saving objects to files and loading them back later |
| **Polymorphism** | The ability of different classes to implement the same interface (`respond()`) while behaving differently | Different objects responding to the same method call in their own way |
| **POST request** | HTTP method that sends data in the request body to the server | "Here's some data — process it" — used when submitting a form or sending a chat message |
| **Promise** | JavaScript object representing the eventual result of an async operation | A JavaScript "IOU" — a guarantee that a result will come later |
| **ReLU** | Rectified Linear Unit: `f(x) = max(0, x)`. Introduces non-linearity into hidden layers | "If positive, pass it through; if negative, output zero" |
| **Route** | A URL pattern registered with Flask that maps to a Python view function | A specific URL address that Flask knows how to handle |
| **Softmax** | `f(z_i) = e^{z_i} / Σ e^{z_j}` — converts logits to a probability distribution summing to 1 | Converts raw numbers to confidence percentages (0–100%) that add up to 100 |
| **Stochastic Gradient Descent (SGD)** | Gradient descent with mini-batches rather than the full dataset per update | Updating weights using a small random sample of training data each step |
| **Tokenization** | Segmenting a text string into discrete tokens (words, punctuation, subwords) | Cutting a sentence into individual word pieces |
| **Training** | Iteratively updating model weights to minimize loss on labeled training data | Teaching the neural network by showing it examples and correcting its mistakes |
| **Virtual environment (venv)** | An isolated Python interpreter and `site-packages` directory for one project | A private kitchen with its own ingredients just for this project |
| **Vocabulary** | The set of unique words (after lemmatization) seen in all training patterns | The master word list the model knows about (389 words) |
| **Weight** | A learnable parameter in a neural network that scales an input | A number the network adjusts during training to get better at predictions |
| **WSGI** | Web Server Gateway Interface — the standard Python interface between web servers and web apps | The standard "plug shape" that connects Python web frameworks to web servers |
| **XSS (Cross-Site Scripting)** | An attack where malicious script is injected into a web page | Why `textContent` is used instead of `innerHTML` — prevents injected code from running |

---

*This README was written to be both a technical reference and a learning companion.
Every concept here is implemented in the files above — read the code alongside this guide.*
