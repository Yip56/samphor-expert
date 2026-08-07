# =============================================================================
# engine/ml_engine.py
# =============================================================================
# Fix 10 — Sentence Transformer upgrade:
#   The Bag-of-Words + Keras neural network has been replaced with
#   paraphrase-multilingual-MiniLM-L12-v2, a sentence transformer that
#   understands semantic meaning and supports 50+ languages (including Khmer).
#
# HOW IT WORKS NOW:
#   TRAINING TIME  (run train.py once):
#     1. Load all patterns from data/intents.json
#     2. Encode each pattern → 384-dimensional embedding vector
#     3. Save embeddings + intent labels to model/
#
#   CHAT TIME (every user message):
#     1. Encode user message into the same embedding space
#     2. Cosine similarity against all training embeddings
#     3. Pick intent with highest max-similarity score
#     4. Confidence >= 0.52 → return reply; else fallback / clarify
#
# CONNECTIONS:
#   ← Extends ChatEngine blueprint (engine/chat_engine.py)
#   ← Reads training data from data/intents.json
#   ← Saves/loads model files in model/
#   ← Looks up reply text in knowledge/samphor_kb.py
#   → Used by app.py to answer every user message
# =============================================================================

import json
import os
import pickle
import random
from difflib import get_close_matches

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from engine.chat_engine import ChatEngine
from knowledge.samphor_kb import FALLBACK_RESPONSE, INTENT_LABELS, INTENT_RESPONSES

# =============================================================================
# FILE PATH CONSTANTS
# =============================================================================
INTENTS_PATH    = os.path.join("data", "intents.json")
MODEL_DIR       = "model"
EMBEDDINGS_PATH = os.path.join(MODEL_DIR, "embeddings.npy")
LABELS_PATH     = os.path.join(MODEL_DIR, "labels.pkl")

# Multilingual sentence transformer — supports English, Khmer, and 50+ other
# languages out of the box. Downloaded from HuggingFace Hub on first use
# (~118 MB) and cached locally by the transformers library.
ST_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Cosine similarity threshold for a "confident" match.
# Cosine similarity ranges ~0.3–0.9 for this model on related text;
# this replaces the old BOW probability threshold of 0.7.
CONFIDENCE_THRESHOLD = 0.52

# Fix 9 — domain vocabulary for fuzzy spell correction.
_DOMAIN_TERMS = [
    "samphor", "pinpeat", "roneat", "khmer", "angkor", "cambodia",
    "cambodian", "sralai", "chhing", "skor", "ensemble",
    "percussion", "ceremony", "preservation", "xylophones",
    # topic words — so typos like "signnificance" correct to "significance"
    "significance", "cultural", "history", "origin", "material",
    "technique", "tradition", "heritage", "instrument", "kroeung",
    "robam", "sbek", "rufa", "reamker",
]

IGNORE_CHARS = set("?!.,;:'\"")


# =============================================================================
# CLASS: MLEngine
# =============================================================================
class MLEngine(ChatEngine):
    """
    Sentence-transformer-based chat engine.
    Extends ChatEngine, so it MUST implement respond() and reset().
    Also adds train() which is only called from train.py, not during chat.
    """

    def __init__(self):
        self._st_model            = None   # SentenceTransformer instance
        self._embeddings          = None   # np.ndarray shape (N, 384)
        self._labels: list[str]   = []     # intent tag per training pattern
        self._history: list[dict] = []

        if self._artefacts_exist():
            self._load_artefacts()

    # =========================================================================
    # PUBLIC METHOD: train()
    # =========================================================================
    def train(self) -> tuple[float, float]:
        from sentence_transformers import SentenceTransformer

        print("\n[MLEngine] -- Sentence Transformer training pipeline --------")
        print(f"[MLEngine] Model : {ST_MODEL_NAME}")
        print("[MLEngine] (downloading ~118 MB on first run — cached afterwards)\n")

        model = SentenceTransformer(ST_MODEL_NAME)

        with open(INTENTS_PATH, encoding="utf-8") as f:
            data = json.load(f)

        patterns: list[str] = []
        labels:   list[str] = []
        for intent in data["intents"]:
            tag = intent["tag"]
            for pattern in intent["patterns"]:
                patterns.append(pattern)
                labels.append(tag)

        n_intents = len(set(labels))
        print(f"[MLEngine] {len(patterns)} patterns across {n_intents} intents")
        print("[MLEngine] Encoding patterns ...")

        # normalize_embeddings=True makes dot-product == cosine similarity,
        # which speeds up inference slightly.
        embeddings = model.encode(
            patterns,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        os.makedirs(MODEL_DIR, exist_ok=True)
        np.save(EMBEDDINGS_PATH, embeddings)
        with open(LABELS_PATH, "wb") as f:
            pickle.dump(labels, f)

        self._st_model   = model
        self._embeddings = embeddings
        self._labels     = labels

        print(f"\n[MLEngine] Embeddings saved  -> {EMBEDDINGS_PATH}  {embeddings.shape}")
        print(f"[MLEngine] Labels saved      -> {LABELS_PATH}")
        print("[MLEngine] -- Training complete ------------------------------------\n")

        # Sentence transformers don't expose a loss / accuracy metric after
        # embedding — return sentinel values so train.py can detect this.
        return 1.0, 0.0

    # =========================================================================
    # PUBLIC METHOD: respond()
    # =========================================================================
    def respond(self, user_input: str, context: dict | None = None) -> tuple[str, dict]:

        ctx        = context or {}
        # Expand single-letter abbreviations before spell correction
        _lower_stripped = user_input.lower().lstrip()
        for abbr, full in (("y ", "why "), ("r ", "are "), ("u ", "you ")):
            if _lower_stripped.startswith(abbr):
                user_input = full + user_input[len(abbr):]
                break
        user_input = self._normalise_informal(user_input)
        user_input = self._correct_spelling(user_input)  # Fix 9
        step       = ctx.get("onboarding_step", "done")

        if step != "done":
            return self._onboarding_respond(user_input, ctx)

        if self._st_model is None:
            return "Model not loaded. Please run train.py first.", {}

        self._history.append({"role": "user", "content": user_input})

        last_intent   = ctx.get("last_intent")
        last_response = ctx.get("last_response")
        user_name     = ctx.get("user_name", "")
        lower         = user_input.lower().strip()

        # ------------------------------------------------------------------
        # LAYER 1: Follow-up detection
        # ------------------------------------------------------------------
        FOLLOW_UP_WORDS   = {"more", "elaborate", "continue", "expand", "further", "details"}
        FOLLOW_UP_PHRASES = {"tell me more", "what else", "go on", "more detail", "more details",
                             "anything else", "keep going", "say more"}
        short_message = len(user_input.split()) <= 5
        word_match    = bool(set(lower.split()) & FOLLOW_UP_WORDS)
        phrase_match  = any(p in lower for p in FOLLOW_UP_PHRASES)
        is_follow_up  = last_intent and short_message and (word_match or phrase_match)

        if is_follow_up:
            responses = INTENT_RESPONSES.get(last_intent, [])
            fresh     = [r for r in responses if r != last_response]
            reply     = random.choice(fresh) if fresh else (
                "I've shared everything I know on that topic. "
                "Try asking about another aspect — history, materials, "
                "playing technique, tuning, or preservation."
            )
            self._history.append({"role": "assistant", "content": reply})
            return reply, {**ctx, "last_intent": last_intent, "last_response": reply}

        # ------------------------------------------------------------------
        # Sentence-transformer inference — encode + cosine similarity.
        # ------------------------------------------------------------------
        tag, confidence, second_tag, second_conf = self._predict(user_input)

        # ------------------------------------------------------------------
        # LAYER 2: Pronoun / context boost
        # ------------------------------------------------------------------
        CONTEXT_PRONOUNS = {"it", "its", "this", "that"}
        has_pronoun      = bool(set(lower.split()) & CONTEXT_PRONOUNS)

        if (has_pronoun
                and last_intent
                and confidence < CONFIDENCE_THRESHOLD
                and len(user_input.split()) <= 10):
            responses = INTENT_RESPONSES.get(last_intent, [FALLBACK_RESPONSE])
            fresh     = [r for r in responses if r != last_response]
            reply     = random.choice(fresh if fresh else responses)
            self._history.append({"role": "assistant", "content": reply})
            updated = {**ctx, "last_intent": last_intent, "last_response": reply,
                       "last_confidence": round(confidence, 4)}
            return reply, updated

        # ------------------------------------------------------------------
        # LAYER 3: Confidence-based clarification
        # ------------------------------------------------------------------
        name_prefix = f"{user_name}, " if user_name else ""
        if confidence < CONFIDENCE_THRESHOLD:
            if confidence > 0.38 and second_conf > 0.22 and second_tag:
                label1 = INTENT_LABELS.get(tag, tag)
                label2 = INTENT_LABELS.get(second_tag, second_tag)
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
            return reply, {**ctx, "last_intent": None,
                           "needs_review": True, "raw_confidence": round(confidence, 4)}

        # ------------------------------------------------------------------
        # Normal path: confident prediction.
        # Fix 13: serve Khmer reply when user writes in Khmer script.
        # ------------------------------------------------------------------
        from knowledge.samphor_kb import KHMER_RESPONSES
        if self._is_khmer(user_input) and tag in KHMER_RESPONSES:
            responses = KHMER_RESPONSES[tag]
        else:
            responses = INTENT_RESPONSES.get(tag, [FALLBACK_RESPONSE])

        reply = random.choice(responses)
        # Hint when the message looks like a compound question ("X and Y?")
        QUESTION_WORDS = {"what", "when", "where", "who", "why", "how"}
        lower_words    = set(lower.split())
        if " and " in lower and bool(lower_words & QUESTION_WORDS):
            reply += " (You asked multiple questions — I answered the main one. Feel free to ask the second part separately!)"
        self._history.append({"role": "assistant", "content": reply})
        updated = {
            **ctx,
            "last_intent":     tag,
            "last_response":   reply,
            "last_confidence": round(confidence, 4),
        }
        return reply, updated

    # =========================================================================
    # ONBOARDING HELPERS
    # =========================================================================

    @staticmethod
    def _normalise_informal(text: str) -> str:
        """Expand common no-apostrophe shorthands before name/occupation extraction."""
        import re
        _INFORMAL = [
            # contraction shorthands (word-boundary so "him" isn't touched)
            (r"\bim\b",       "i'm"),
            (r"\bdont\b",     "don't"),
            (r"\bcant\b",     "can't"),
            (r"\bwont\b",     "won't"),
            (r"\bisnt\b",     "isn't"),
            (r"\bwasnt\b",    "wasn't"),
            (r"\bwerent\b",   "weren't"),
            (r"\bdidnt\b",    "didn't"),
            (r"\bdoesnt\b",   "doesn't"),
            (r"\bhavent\b",   "haven't"),
            (r"\bwouldnt\b",  "wouldn't"),
            (r"\bshouldnt\b", "shouldn't"),
            (r"\bcouldnt\b",  "couldn't"),
            (r"\bwhos\b",     "who's"),
            (r"\bwhats\b",    "what's"),
            (r"\bhows\b",     "how's"),
            (r"\bthats\b",    "that's"),
            (r"\bits\b",      "it's"),
            (r"\bive\b",      "i've"),
            (r"\byoure\b",    "you're"),
            (r"\btheyre\b",   "they're"),
            (r"\bwere\b",     "we're"),
            # common chat abbreviations
            (r"\bidk\b",      "i don't know"),
            (r"\bpls\b",      "please"),
            (r"\bplz\b",      "please"),
            (r"\bthx\b",      "thanks"),
            (r"\bbtw\b",      "by the way"),
            (r"\bngl\b",      "not gonna lie"),
            (r"\btbh\b",      "to be honest"),
            (r"\bimo\b",      "in my opinion"),
            (r"\bgonna\b",    "going to"),
            (r"\bwanna\b",    "want to"),
            (r"\bgotta\b",    "got to"),
            (r"\blemme\b",    "let me"),
            (r"\bdunno\b",    "don't know"),
            # common keyboard typos / phonetic misspellings
            (r"\bwat\b",      "what"),
            (r"\bwut\b",      "what"),
            (r"\bwot\b",      "what"),
            (r"\biz\b",       "is"),
            (r"\bteh\b",      "the"),
            (r"\bda\b",       "the"),
            (r"\bwaz\b",      "was"),
            (r"\bdat\b",      "that"),
            (r"\bdis\b",      "this"),
            (r"\btel\b",      "tell"),
            (r"\bsomthing\b", "something"),
            (r"\bsomething\b","something"),
            (r"\bsignificance\b","significance"),
        ]
        for pattern, replacement in _INFORMAL:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _onboarding_respond(self, user_input: str, ctx: dict) -> tuple[str, dict]:
        step = ctx.get("onboarding_step", "name")
        if step == "name":
            name  = self._extract_name(user_input)
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
            reply   = "What would you like to know about the Samphor?"
            updated = {**ctx, "onboarding_step": "done"}
        self._history.append({"role": "assistant", "content": reply})
        return reply, updated

    @staticmethod
    def _extract_name(text: str) -> str:
        text = text.strip().strip(".,!?")
        SKIP_WORDS = {"skip", "pass", "private", "anonymous", "secret", "nothing", "nope", "no"}
        if any(w in text.lower().split() for w in SKIP_WORDS):
            return "friend"
        # If user typed a greeting instead of their name, don't use it as a name
        GREETING_WORDS = {"hello", "hi", "hey", "greetings", "howdy", "hiya", "sup", "yo", "good morning",
                          "good afternoon", "good evening", "good day"}
        if text.lower().strip() in GREETING_WORDS:
            return "friend"
        for prefix in sorted([
            # formal
            "my name is", "i'm called", "i am called", "you can call me",
            "call me", "the name is", "it's", "i'm", "i am",
            "my name's", "they call me", "just call me", "people call me",
            # informal / shorthand (what real users type)
            "im called", "im known as", "im",   # "im yip" → "yip"
            "name is", "name's",
        ], key=len, reverse=True):              # longest prefix first to avoid partial matches
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip().strip(".,!?")
                break
        words = text.split()
        name  = " ".join(words[:2]) if len(words) > 2 else " ".join(words)
        return name.title() if name else "friend"

    @staticmethod
    def _extract_occupation(text: str) -> str:
        text = text.strip().strip(".,!?")
        SKIP_WORDS = {"skip", "pass", "private", "nothing", "nope", "no", "secret"}
        if any(w in text.lower().split() for w in SKIP_WORDS):
            return "professional"
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
        occ   = occupation.lower()
        words = set(occ.split())
        MUSICIAN_WORDS   = {"musician", "music", "singer", "guitarist", "pianist",
                             "drummer", "performer", "artist", "band", "composer"}
        TEACHER_WORDS    = {"teacher", "professor", "educator", "lecturer",
                             "instructor", "tutor", "academic"}
        STUDENT_WORDS    = {"student", "pupil", "learner", "undergraduate",
                             "graduate", "studying", "study"}
        RESEARCHER_WORDS = {"researcher", "historian", "anthropologist",
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
    def reset(self) -> None:
        self._history.clear()

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _predict(self, text: str) -> tuple[str, float, str | None, float]:
        """
        Encode text with the sentence transformer, compute cosine similarity
        against all training embeddings, and return the top-2 intents.
        """
        query_emb = self._st_model.encode([text], normalize_embeddings=True)
        sims      = cosine_similarity(query_emb, self._embeddings)[0]

        # Take the max similarity score per intent (not the average),
        # so a single very-close training pattern can strongly activate an intent.
        intent_best: dict[str, float] = {}
        for idx, label in enumerate(self._labels):
            score = float(sims[idx])
            if label not in intent_best or score > intent_best[label]:
                intent_best[label] = score

        sorted_intents = sorted(intent_best.items(), key=lambda x: -x[1])
        best_tag,    best_conf   = sorted_intents[0]
        second_tag,  second_conf = sorted_intents[1] if len(sorted_intents) > 1 else (None, 0.0)
        return best_tag, best_conf, second_tag, second_conf

    @staticmethod
    def _correct_spelling(text: str) -> str:
        """Fix 9: Fuzzy-correct domain-specific typos (e.g. 'Samhor' → 'samphor')."""
        tokens = text.split()
        result = []
        for tok in tokens:
            clean = tok.lower().strip("?!.,;:'\"")
            if len(clean) < 4:
                result.append(tok)
                continue
            matches = get_close_matches(clean, _DOMAIN_TERMS, n=1, cutoff=0.72)
            if matches and matches[0] != clean:
                lead = len(tok) - len(tok.lstrip("?!.,;:'\""))
                tail = len(tok) - len(tok.rstrip("?!.,;:'\""))
                result.append(tok[:lead] + matches[0] + (tok[len(tok) - tail:] if tail else ""))
            else:
                result.append(tok)
        return " ".join(result)

    @staticmethod
    def _is_khmer(text: str) -> bool:
        """Return True if text contains Khmer Unicode characters (Fix 13)."""
        return any('ក' <= c <= '៿' for c in text)

    def _artefacts_exist(self) -> bool:
        """Return True only if both saved artefact files exist."""
        return all(os.path.exists(p) for p in (EMBEDDINGS_PATH, LABELS_PATH))

    def _load_artefacts(self) -> None:
        """Load the pre-trained embeddings and sentence transformer from disk/cache."""
        from sentence_transformers import SentenceTransformer
        print(f"[MLEngine] Loading sentence transformer: {ST_MODEL_NAME}")
        self._st_model   = SentenceTransformer(ST_MODEL_NAME)
        self._embeddings = np.load(EMBEDDINGS_PATH)
        with open(LABELS_PATH, "rb") as f:
            self._labels = pickle.load(f)
        print(f"[MLEngine] Loaded {len(self._labels)} embeddings from {EMBEDDINGS_PATH}")
