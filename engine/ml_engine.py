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
from knowledge.samphor_kb import FALLBACK_RESPONSE, INTENT_LABELS, INTENT_RESPONSES

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
    # Called by: app.py → /chat route, on every user message.
    #
    # context : per-user session dict passed in from Flask session storage.
    #           Keys used: onboarding_step, user_name, user_occupation,
    #                      last_intent, last_response, last_confidence
    # Returns : (reply_string, updated_context_dict)
    #
    # Flow:
    #   0. Onboarding gate — name → occupation → done (no NN needed)
    #   1. Follow-up detection  — "tell me more" reuses the last intent
    #   2. Pronoun context boost — "what is it made of?" resolves to last topic
    #   3. Clarification query  — mid-confidence asks "did you mean X or Y?"
    # =========================================================================
    def respond(self, user_input: str, context: dict | None = None) -> tuple[str, dict]:

        ctx  = context or {}
        step = ctx.get("onboarding_step", "done")

        # Onboarding runs before any ML inference — no model needed.
        if step != "done":
            return self._onboarding_respond(user_input, ctx)

        if self._model is None:
            return "Model not loaded. Please run train.py first.", {}

        self._history.append({"role": "user", "content": user_input})

        last_intent   = ctx.get("last_intent")
        last_response = ctx.get("last_response")
        user_name     = ctx.get("user_name", "")
        lower         = user_input.lower().strip()

        # ------------------------------------------------------------------
        # LAYER 1: Follow-up detection
        # If the user sends a short "tell me more"-style message and we
        # already know what they were asking about, return an alternate reply
        # for that same intent rather than re-running the neural network.
        # ------------------------------------------------------------------
        FOLLOW_UP_WORDS    = {"more", "elaborate", "continue", "expand", "further", "details"}
        FOLLOW_UP_PHRASES  = {"tell me more", "what else", "go on", "more detail", "more details",
                              "anything else", "keep going", "say more"}
        short_message      = len(user_input.split()) <= 5
        word_match         = bool(set(lower.split()) & FOLLOW_UP_WORDS)
        phrase_match       = any(p in lower for p in FOLLOW_UP_PHRASES)
        is_follow_up       = last_intent and short_message and (word_match or phrase_match)

        if is_follow_up:
            responses = INTENT_RESPONSES.get(last_intent, [])
            # Prefer a reply the user hasn't seen yet.
            fresh = [r for r in responses if r != last_response]
            if fresh:
                reply = random.choice(fresh)
            else:
                reply = (
                    "I've shared everything I know on that topic. "
                    "Try asking about another aspect — history, materials, "
                    "playing technique, tuning, or preservation."
                )
            self._history.append({"role": "assistant", "content": reply})
            return reply, {**ctx, "last_intent": last_intent, "last_response": reply}

        # ------------------------------------------------------------------
        # Run the neural network to get confidence scores for all 14 intents.
        # ------------------------------------------------------------------
        tokens      = nltk.word_tokenize(user_input)
        bow         = np.array([self._make_bow(tokens, self._words)])
        predictions = self._model.predict(bow, verbose=0)[0]

        sorted_idx      = np.argsort(predictions)[::-1]   # highest first
        best_idx        = int(sorted_idx[0])
        second_idx      = int(sorted_idx[1])
        confidence      = float(predictions[best_idx])
        second_conf     = float(predictions[second_idx])

        # ------------------------------------------------------------------
        # LAYER 2: Pronoun / context boost
        # If confidence is low but the message contains a context pronoun
        # ("it", "its", "this", "that") and the message is short, assume
        # the user is asking a follow-up about the last intent.
        # ------------------------------------------------------------------
        CONTEXT_PRONOUNS = {"it", "its", "this", "that"}
        has_pronoun      = bool(set(lower.split()) & CONTEXT_PRONOUNS)

        if (has_pronoun
                and last_intent
                and confidence < CONFIDENCE_THRESHOLD
                and len(user_input.split()) <= 10):
            tag       = last_intent
            responses = INTENT_RESPONSES.get(tag, [FALLBACK_RESPONSE])
            fresh     = [r for r in responses if r != last_response]
            reply     = random.choice(fresh if fresh else responses)
            self._history.append({"role": "assistant", "content": reply})
            updated = {**ctx, "last_intent": tag, "last_response": reply,
                       "last_confidence": round(confidence, 4)}
            return reply, updated

        # ------------------------------------------------------------------
        # LAYER 3: Confidence-based clarification
        # Below threshold but above a noise floor → ask which of the two
        # most likely topics the user meant instead of a dead-end fallback.
        # ------------------------------------------------------------------
        name_prefix = f"{user_name}, " if user_name else ""
        if confidence < CONFIDENCE_THRESHOLD:
            if confidence > 0.35 and second_conf > 0.15:
                label1 = INTENT_LABELS.get(self._classes[best_idx],  self._classes[best_idx])
                label2 = INTENT_LABELS.get(self._classes[second_idx], self._classes[second_idx])
                reply  = (
                    f"{name_prefix}I'm not quite sure what you'd like to know. "
                    f"Were you asking about {label1}, or perhaps {label2}? "
                    f"Try rephrasing and I'll do my best!"
                )
            else:
                if user_name:
                    reply = (
                        f"I'm not confident I understood that, {user_name}. Could you rephrase? "
                        "You can ask about the Samphor's definition, history, materials, shape, "
                        "playing technique, tuning, ceremonies, the Pinpeat ensemble, comparisons, "
                        "learning, or preservation."
                    )
                else:
                    reply = FALLBACK_RESPONSE
            self._history.append({"role": "assistant", "content": reply})
            return reply, {**ctx, "last_intent": None}

        # ------------------------------------------------------------------
        # Normal path: confident prediction → look up reply in knowledge base.
        # ------------------------------------------------------------------
        tag       = self._classes[best_idx]
        responses = INTENT_RESPONSES.get(tag, [FALLBACK_RESPONSE])
        reply     = random.choice(responses)

        self._history.append({"role": "assistant", "content": reply})
        updated = {
            **ctx,
            "last_intent":    tag,
            "last_response":  reply,
            "last_confidence": round(confidence, 4),
        }
        return reply, updated

    # =========================================================================
    # ONBOARDING HELPERS
    # =========================================================================
    # These four methods handle the small "get to know you" conversation that
    # runs before the main Q&A.  No neural network is involved — it is pure
    # state-machine logic driven by ctx["onboarding_step"].
    #
    # State flow:  "name" → "occupation" → "done"
    # =========================================================================

    def _onboarding_respond(self, user_input: str, ctx: dict) -> tuple[str, dict]:
        """Route to the correct onboarding step handler."""
        step = ctx.get("onboarding_step", "name")
        if step == "name":
            name = self._extract_name(user_input)
            reply = (
                f"Lovely to meet you, {name}! "
                f"What do you do for a living, if you don't mind me asking?"
            )
            updated = {**ctx, "onboarding_step": "occupation", "user_name": name}
        elif step == "occupation":
            name       = ctx.get("user_name", "friend")
            occupation = self._extract_occupation(user_input)
            reply      = self._occupation_greeting(name, occupation)
            updated    = {**ctx, "onboarding_step": "done", "user_occupation": occupation}
        else:
            # Shouldn't reach here, but be safe.
            reply   = "What would you like to know about the Samphor?"
            updated = {**ctx, "onboarding_step": "done"}
        self._history.append({"role": "assistant", "content": reply})
        return reply, updated

    @staticmethod
    def _extract_name(text: str) -> str:
        """Strip common lead-in phrases and return a title-cased name."""
        text = text.strip().strip(".,!?")
        SKIP_WORDS = {"skip", "pass", "private", "anonymous", "secret", "nothing", "nope", "no"}
        if any(w in text.lower().split() for w in SKIP_WORDS):
            return "friend"
        for prefix in [
            "my name is", "i'm called", "i am called", "you can call me",
            "call me", "the name is", "name's", "it's", "i'm", "i am",
        ]:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip().strip(".,!?")
                break
        words = text.split()
        # Clamp to two words so "John Smith who loves drums" → "John Smith"
        name = " ".join(words[:2]) if len(words) > 2 else " ".join(words)
        return name.title() if name else "friend"

    @staticmethod
    def _extract_occupation(text: str) -> str:
        """Strip lead-in phrases and return a lowercase occupation string."""
        text = text.strip().strip(".,!?")
        SKIP_WORDS = {"skip", "pass", "private", "nothing", "nope", "no", "secret"}
        if any(w in text.lower().split() for w in SKIP_WORDS):
            return "professional"
        # Sort longest-first so "i work as a" is tried before "i work"
        PREFIXES = sorted([
            "i work as a", "i work as an", "i am a", "i am an",
            "i'm a", "i'm an", "my job is", "i work in", "i work as",
            "i am", "i'm",
        ], key=len, reverse=True)
        for prefix in PREFIXES:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip().strip(".,!?")
                break
        return text.lower() if text else "professional"

    @staticmethod
    def _occupation_greeting(name: str, occupation: str) -> str:
        """Return a personalised transition message based on the user's occupation."""
        occ = occupation.lower()
        words = set(occ.split())

        MUSICIAN_WORDS    = {"musician", "music", "singer", "guitarist", "pianist",
                              "drummer", "performer", "artist", "band", "composer"}
        TEACHER_WORDS     = {"teacher", "professor", "educator", "lecturer",
                              "instructor", "tutor", "academic"}
        STUDENT_WORDS     = {"student", "pupil", "learner", "undergraduate",
                              "graduate", "studying", "study"}
        RESEARCHER_WORDS  = {"researcher", "historian", "anthropologist",
                              "ethnomusicologist", "scholar", "archivist"}

        if words & MUSICIAN_WORDS:
            return (
                f"How fitting, {name}! As a fellow music person, "
                f"you will have a natural appreciation for the Samphor's playing technique "
                f"and its role as the rhythmic anchor of the Pinpeat ensemble. "
                f"What would you like to explore first?"
            )
        if words & TEACHER_WORDS:
            return (
                f"Wonderful, {name}! Educators like yourself are vital for keeping traditions alive. "
                f"The Samphor's rich history and cultural significance make for fascinating teaching material. "
                f"What shall we start with?"
            )
        if words & STUDENT_WORDS:
            return (
                f"Great to have you here, {name}! Students are always my favourite visitors. "
                f"You have come to the right place to learn about Cambodia's iconic drum. "
                f"What would you like to discover first?"
            )
        if words & RESEARCHER_WORDS:
            return (
                f"Excellent, {name}! Scholars will find the Samphor particularly compelling — "
                f"its history spans over a millennium and its preservation story is remarkable. "
                f"What aspect would you like to explore first?"
            )
        return (
            f"How interesting, {name}! I am delighted to have you here. "
            f"Feel free to ask me anything about the Samphor — its history, how it is played, "
            f"its cultural significance, and much more. Where shall we begin?"
        )

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
