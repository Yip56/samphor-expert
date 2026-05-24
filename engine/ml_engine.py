# =============================================================================
# engine/ml_engine.py
# =============================================================================
# This is the "smart brain" of the chatbot.
# Instead of matching keywords like RuleBasedEngine, it uses a trained
# neural network to UNDERSTAND what the user is asking, even if they use
# words that were never in the training data.
#
# HOW IT WORKS (the big picture):
#   TRAINING TIME  (run train.py once):
#     1. Read every example question from data/intents.json
#     2. Clean up the text (lowercase, remove punctuation, find root words)
#     3. Convert each question into a list of 0s and 1s (bag-of-words)
#     4. Feed those numbers into a neural network and let it learn patterns
#     5. Save the trained network to model/samphor_model.h5
#
#   CHAT TIME (every user message):
#     1. Clean up the user's message the same way as training
#     2. Convert it to 0s and 1s
#     3. Ask the network: "which category does this belong to?"
#     4. If the network is ≥ 70% confident, look up the reply in samphor_kb.py
#     5. Otherwise, send the fallback "I didn't understand" message
#
# CONNECTIONS:
#   ← Extends ChatEngine blueprint (engine/chat_engine.py)
#   ← Reads training data from data/intents.json
#   ← Saves/loads model files in model/
#   ← Looks up reply text in knowledge/samphor_kb.py
#   → Used by app.py to answer every user message
# =============================================================================

import json       # for reading the intents.json training data file
import os         # for building file paths that work on any operating system
import pickle     # for saving/loading Python objects (words list, classes list)
import random     # for picking a random reply from the list of possible replies

import nltk                            # Natural Language Toolkit — text processing library
import numpy as np                     # numpy — fast math library, used for arrays
from nltk.stem import WordNetLemmatizer  # turns words into their root form (e.g. "playing" → "play")

# Import the abstract blueprint this engine must follow.
from engine.chat_engine import ChatEngine

# Import the reply texts and the fallback message from the knowledge base.
from knowledge.samphor_kb import FALLBACK_RESPONSE, INTENT_RESPONSES

# -----------------------------------------------------------------------------
# NLTK DATA AUTO-DOWNLOAD
# -----------------------------------------------------------------------------
# NLTK needs extra data files (dictionaries, tokeniser rules) to work.
# This block checks if each package is already downloaded; if not, it
# downloads it automatically and quietly.
# 'punkt' / 'punkt_tab' : rules for splitting sentences into words
# 'wordnet'             : dictionary of word root forms (for lemmatization)
# 'omw-1.4'            : multilingual wordnet data
# 'stopwords'           : common words like "the", "is", "a" (not used here
#                         but useful if you extend the engine later)
# -----------------------------------------------------------------------------
for pkg in ("punkt", "wordnet", "omw-1.4", "stopwords", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg.startswith("punkt") else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# -----------------------------------------------------------------------------
# FILE PATH CONSTANTS
# -----------------------------------------------------------------------------
# Using os.path.join() instead of hardcoding "/" or "\" means the paths work
# on both Windows and Mac/Linux without changes.
# -----------------------------------------------------------------------------
INTENTS_PATH = os.path.join("data", "intents.json")   # training data
MODEL_DIR    = "model"                                  # folder for all saved files
MODEL_PATH   = os.path.join(MODEL_DIR, "samphor_model.h5")  # the trained neural network
WORDS_PATH   = os.path.join(MODEL_DIR, "words.pkl")         # vocabulary list (389 words)
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.pkl")        # list of 14 intent names

# The neural network returns a confidence score between 0.0 and 1.0.
# If it's below this value, we say "I don't know" instead of guessing.
CONFIDENCE_THRESHOLD = 0.7

# Characters we want to strip out before processing (they carry no meaning).
IGNORE_CHARS = set("?!.,;:'\"")


# =============================================================================
# CLASS: MLEngine
# =============================================================================
class MLEngine(ChatEngine):
    """
    ML-based chat engine.
    Extends ChatEngine, so it MUST implement respond() and reset().
    Also adds train() which is only called from train.py, not during chat.
    """

    # -------------------------------------------------------------------------
    # __init__ : runs automatically when you do engine = MLEngine()
    # -------------------------------------------------------------------------
    def __init__(self):
        # WordNetLemmatizer converts words to their base/root form.
        # Examples: "playing" → "play", "drums" → "drum", "asked" → "ask"
        # This helps the model recognise the same concept written differently.
        self._lemmatizer = WordNetLemmatizer()

        # _words   : the full vocabulary list built during training (389 unique words)
        # _classes : the list of 14 intent category names (e.g. "ask_history")
        # _model   : the Keras neural network object (None until loaded or trained)
        # _history : list of all messages in the current conversation session
        self._words: list[str] = []
        self._classes: list[str] = []
        self._model = None
        self._history: list[dict] = []

        # If the model files already exist in model/ (i.e. you already ran train.py),
        # load them now so the chatbot is ready immediately without retraining.
        if self._artefacts_exist():
            self._load_artefacts()

    # =========================================================================
    # PUBLIC METHOD: train()
    # =========================================================================
    # Called by: train.py
    # This is the full ML pipeline — it reads data, builds a neural network,
    # trains it, and saves the results to disk.
    # =========================================================================
    def train(self) -> None:

        # TensorFlow/Keras is only imported here (inside the function) rather than
        # at the top of the file. This is called a "lazy import".
        # Reason: TensorFlow is very large and takes several seconds to load.
        # If you only want to chat (not train), you don't want to wait for it.
        import tensorflow as tf
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import SGD

        print("\n[MLEngine] -- Starting training pipeline ----------------------")

        # ------------------------------------------------------------------
        # STEP 1: Load the training data from data/intents.json
        # ------------------------------------------------------------------
        # intents.json contains a list of "intents" — each one has:
        #   "tag"      : the category name, e.g. "ask_history"
        #   "patterns" : example questions a user might ask
        #   "responses": (not used for training — responses live in samphor_kb.py)
        # ------------------------------------------------------------------
        with open(INTENTS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        intents = data["intents"]
        print(f"[MLEngine] Loaded {len(intents)} intents from {INTENTS_PATH}")

        # ------------------------------------------------------------------
        # STEP 2: Tokenise and lemmatise every example pattern
        # ------------------------------------------------------------------
        # 'words'     : a flat list of EVERY word found in ALL patterns
        # 'classes'   : a list of intent tag names (each added only once)
        # 'documents' : pairs of (word_list, tag) — one entry per pattern
        # ------------------------------------------------------------------
        words: list[str] = []
        classes: list[str] = []
        documents: list[tuple[list[str], str]] = []

        for intent in intents:
            tag = intent["tag"]
            # Add this tag to our list of categories (if not already there).
            if tag not in classes:
                classes.append(tag)

            for pattern in intent["patterns"]:
                # nltk.word_tokenize splits a sentence into individual words.
                # "What is the Samphor" → ["What", "is", "the", "Samphor"]
                tokens = nltk.word_tokenize(pattern)
                words.extend(tokens)                      # add them to the master list
                documents.append((tokens, tag))           # remember which tag they belong to

        # Now clean up the master word list:
        #   - lemmatize: "playing" → "play", "drums" → "drum"
        #   - lowercase: "Samphor" → "samphor"
        #   - remove punctuation characters
        #   - sorted(set(...)) removes duplicates and sorts alphabetically
        # The result is the VOCABULARY — 389 unique root words.
        words = sorted(
            set(
                self._lemmatizer.lemmatize(w.lower())
                for w in words
                if w not in IGNORE_CHARS
            )
        )
        classes = sorted(classes)   # sort the 14 intent names alphabetically

        print(f"[MLEngine] Vocabulary size : {len(words)} words")
        print(f"[MLEngine] Classes         : {len(classes)}")

        # ------------------------------------------------------------------
        # STEP 3: Convert text into numbers — "Bag of Words" (BoW)
        # ------------------------------------------------------------------
        # Neural networks can't read text — they only understand numbers.
        # Bag-of-Words turns each pattern into a list of 0s and 1s:
        #   - The list is as long as the vocabulary (389 items)
        #   - 1 means "this word from the vocabulary appears in this pattern"
        #   - 0 means "this word does NOT appear"
        #
        # Example with a tiny vocabulary ["drum", "history", "play"]:
        #   "Tell me about Samphor history" → [0, 1, 0]
        #   (only "history" is present)
        #
        # training_X : list of BoW vectors (one per pattern) — the INPUT
        # training_y : list of label vectors (one per pattern) — the ANSWER
        #
        # A label vector is also 0s and 1s, one slot per class.
        #   14 classes → label has 14 items; only the correct class is 1.
        #   e.g. "ask_history" is index 2 → [0, 0, 1, 0, 0, 0, ...]
        # ------------------------------------------------------------------
        training_X: list[list[int]] = []
        training_y: list[list[int]] = []

        for token_list, tag in documents:
            bow = self._make_bow(token_list, words)       # convert pattern to BoW vector
            label = [0] * len(classes)                    # start with all zeros
            label[classes.index(tag)] = 1                 # flip the correct class to 1
            training_X.append(bow)
            training_y.append(label)

        # np.array() wraps Python lists into NumPy arrays.
        # NumPy arrays are like super-powered lists optimised for math.
        # Keras (the neural network library) requires NumPy arrays as input.
        X = np.array(training_X)   # shape: (280 samples, 389 features)
        y = np.array(training_y)   # shape: (280 samples, 14 classes)
        print(f"[MLEngine] Training samples: {len(X)}")

        # ------------------------------------------------------------------
        # STEP 4: Build the Neural Network
        # ------------------------------------------------------------------
        # A Sequential model is a stack of layers — data flows through them
        # one after another, like a production line.
        #
        # Dense(128, relu) — "thinking layer" with 128 neurons.
        #   relu activation: any negative number becomes 0, positives stay.
        #   This lets the network learn complex patterns.
        #
        # Dropout(0.5) — randomly switches off 50% of neurons during training.
        #   This prevents "memorising" the training data (called overfitting).
        #   Think of it as forcing the network to not rely on any single neuron.
        #
        # Dense(64, relu) — second, smaller thinking layer (64 neurons).
        #
        # Dense(14, softmax) — the OUTPUT layer. One neuron per intent class.
        #   softmax turns the raw numbers into percentages that add up to 100%.
        #   e.g. [0.02, 0.85, 0.01, ...] → "85% sure this is intent #2"
        # ------------------------------------------------------------------
        model = Sequential([
            Dense(128, input_shape=(len(X[0]),), activation="relu"),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(len(classes), activation="softmax"),
        ])

        # SGD = Stochastic Gradient Descent — the algorithm that adjusts the
        # network's internal numbers (weights) to improve predictions.
        # learning_rate : how big each adjustment step is (0.01 = small steps)
        # momentum      : carry some speed from the previous step (like rolling downhill)
        # nesterov      : a smarter variant of momentum (slightly more accurate)
        sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

        # compile() sets up the learning process:
        # loss="categorical_crossentropy" : measures how wrong the predictions are
        #   (used for problems with multiple categories)
        # metrics=["accuracy"] : also track what % of predictions are correct
        model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

        print("\n[MLEngine] Model architecture:")
        model.summary()   # prints a table showing every layer and its parameter count

        # ------------------------------------------------------------------
        # STEP 5: Train the model
        # ------------------------------------------------------------------
        # model.fit() is where the actual learning happens.
        # epochs=200    : go through ALL 280 training examples 200 times
        # batch_size=5  : update the weights after every 5 examples (not all at once)
        # verbose=1     : print a progress bar for each epoch
        #
        # 'history' records the loss and accuracy after every epoch,
        # so we can report the final values below.
        # ------------------------------------------------------------------
        print("\n[MLEngine] Training ...")
        history = model.fit(
            X, y,
            epochs=200,
            batch_size=5,
            verbose=1,
        )

        # Grab the LAST value from the recorded history (= final epoch result).
        final_loss = history.history["loss"][-1]
        final_acc  = history.history["accuracy"][-1]
        print(f"\n[MLEngine] Final loss     : {final_loss:.4f}")
        print(f"[MLEngine] Final accuracy : {final_acc * 100:.2f}%")

        # ------------------------------------------------------------------
        # STEP 6: Save everything to disk
        # ------------------------------------------------------------------
        # We save three things so the chatbot can answer questions without
        # retraining every time it starts:
        #
        #   samphor_model.h5 — the trained neural network (all its weights)
        #   words.pkl        — the vocabulary list (389 words), needed to
        #                      convert new user messages to BoW vectors
        #   classes.pkl      — the 14 intent names, needed to interpret the
        #                      network's output back into a category name
        #
        # pickle is Python's standard way to save any object to a binary file.
        # 'wb' = write binary mode
        # ------------------------------------------------------------------
        os.makedirs(MODEL_DIR, exist_ok=True)   # create model/ folder if it doesn't exist
        model.save(MODEL_PATH)

        with open(WORDS_PATH, "wb") as f:
            pickle.dump(words, f)
        with open(CLASSES_PATH, "wb") as f:
            pickle.dump(classes, f)

        # Also update the live instance so you can immediately call respond()
        # after training without restarting the program.
        self._words   = words
        self._classes = classes
        self._model   = model

        print(f"[MLEngine] Model saved   -> {MODEL_PATH}")
        print(f"[MLEngine] Words saved   -> {WORDS_PATH}")
        print(f"[MLEngine] Classes saved -> {CLASSES_PATH}")
        print("[MLEngine] -- Training complete -------------------------------------\n")

        # Return both metrics so train.py can print the summary.
        return final_acc, final_loss

    # =========================================================================
    # PUBLIC METHOD: respond()
    # =========================================================================
    # Called by: app.py → /chat route, on every user message
    # This is the real-time inference path — no training happens here.
    # =========================================================================
    def respond(self, user_input: str) -> str:

        # Guard: if train() was never run and no saved model exists, bail out.
        if self._model is None:
            return "Model not loaded. Please run train() or load a saved model first."

        # Save the user's message to conversation history.
        self._history.append({"role": "user", "content": user_input})

        # ------------------------------------------------------------------
        # Convert the user's text into a BoW vector — the same transformation
        # used during training so the numbers are compatible with the model.
        # ------------------------------------------------------------------
        tokens = nltk.word_tokenize(user_input)              # split into words
        bow    = np.array([self._make_bow(tokens, self._words)])  # shape: (1, 389)

        # ------------------------------------------------------------------
        # Run the neural network forward (inference / prediction).
        # predictions is an array of 14 confidence scores (one per intent),
        # e.g. [0.01, 0.03, 0.85, 0.02, ...]  — they sum to ~1.0
        # ------------------------------------------------------------------
        predictions = self._model.predict(bow, verbose=0)[0]

        # np.argmax finds the INDEX of the highest value.
        # That index corresponds to the intent the network is most confident about.
        best_idx   = int(np.argmax(predictions))
        confidence = float(predictions[best_idx])   # the actual confidence score

        # ------------------------------------------------------------------
        # Confidence check: only trust the prediction if ≥ 70%.
        # If the network isn't sure, return the fallback "I didn't understand" message.
        # ------------------------------------------------------------------
        if confidence < CONFIDENCE_THRESHOLD:
            reply = FALLBACK_RESPONSE   # imported from knowledge/samphor_kb.py
        else:
            # Map the winning index back to an intent name (e.g. "ask_history").
            tag = self._classes[best_idx]

            # Look up the list of possible replies for this intent in samphor_kb.py.
            responses = INTENT_RESPONSES.get(tag)

            if responses:
                # Pick one reply at random so the bot doesn't always say the same thing.
                reply = random.choice(responses)
            else:
                # The tag exists in classes but has no entry in INTENT_RESPONSES — shouldn't
                # happen in normal operation, but fall back gracefully just in case.
                reply = FALLBACK_RESPONSE

        self._history.append({"role": "assistant", "content": reply})
        return reply

    # =========================================================================
    # PUBLIC METHOD: reset()
    # =========================================================================
    # Called by: app.py → /reset route, when user clicks the ↺ button
    # =========================================================================
    def reset(self) -> None:
        # Wipe the conversation history for this session.
        self._history.clear()

    # =========================================================================
    # PRIVATE HELPERS  (only used inside this class, not from outside)
    # =========================================================================

    def _lemmatize_tokens(self, tokens: list[str]) -> list[str]:
        """
        Take a list of raw words and return their lemmatized (root) forms.
        Punctuation characters are removed at this step.
        Example: ["playing", "the", "Samphor", "?"] → ["play", "the", "samphor"]
        """
        return [
            self._lemmatizer.lemmatize(t.lower())
            for t in tokens
            if t not in IGNORE_CHARS
        ]

    def _make_bow(self, tokens: list[str], vocab: list[str]) -> list[int]:
        """
        Convert a list of tokens into a Bag-of-Words vector.
        The vector has one slot for every word in the vocabulary.
        Slot = 1 if that vocabulary word appears in 'tokens', else 0.

        Used during BOTH training (step 3) and inference (respond()).
        """
        lemmatized = set(self._lemmatize_tokens(tokens))   # use a set for fast lookup
        return [1 if w in lemmatized else 0 for w in vocab]

    def _artefacts_exist(self) -> bool:
        """
        Return True only if ALL three saved files exist.
        If even one is missing we can't load a partial model, so we return False
        and wait for the user to run train.py.
        """
        return all(os.path.exists(p) for p in (MODEL_PATH, WORDS_PATH, CLASSES_PATH))

    def _load_artefacts(self) -> None:
        """
        Load the pre-trained model and vocabulary from disk.
        Called automatically in __init__ if the files exist.
        'rb' = read binary mode (opposite of 'wb' used when saving).
        """
        from tensorflow.keras.models import load_model  # type: ignore

        self._model = load_model(MODEL_PATH)
        with open(WORDS_PATH, "rb") as f:
            self._words = pickle.load(f)
        with open(CLASSES_PATH, "rb") as f:
            self._classes = pickle.load(f)
        print(f"[MLEngine] Loaded pre-trained model from {MODEL_PATH}")
