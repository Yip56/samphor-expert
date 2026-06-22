# =============================================================================
# engine/chat_engine.py
# =============================================================================
# This file defines the BLUEPRINT (called an abstract base class) that every
# chat engine in this project must follow.
#
# Think of it like a job description:
#   "Any engine you build MUST be able to respond() and reset()."
# It doesn't say HOW to do those things — that's up to each engine.
#
# Two engines currently exist:
#   1. ChatEngine      → the blueprint (this file)
#   2. RuleBasedEngine → a simple keyword-matching engine (also this file)
#   3. MLEngine        → the smart AI engine  (engine/ml_engine.py)
#
# app.py imports one of these engines and uses it to answer user questions.
# =============================================================================

# 'ABC' stands for Abstract Base Class — it's Python's way of writing blueprints.
# 'abstractmethod' is a tag you put on methods that MUST be written by sub-classes.
from abc import ABC, abstractmethod


# -----------------------------------------------------------------------------
# BLUEPRINT: ChatEngine
# -----------------------------------------------------------------------------
# Any class that "extends" ChatEngine (writes ': ChatEngine' after its name)
# is promising to provide a respond() method and a reset() method.
# If it forgets, Python will throw an error when you try to create one.
# This keeps every engine consistent — app.py can swap engines without
# changing any other code.
# -----------------------------------------------------------------------------
class ChatEngine(ABC):

    # @abstractmethod means: "subclasses MUST override this — no skipping."
    @abstractmethod
    def respond(self, user_input: str, context: dict | None = None) -> tuple[str, dict]:
        # user_input : the text the user typed in the chat box
        # context    : per-session state dict (last_intent, last_response, …)
        # returns    : (reply_text, updated_context)
        pass

    @abstractmethod
    def reset(self) -> None:
        # Called when the user clicks the reset (↺) button.
        # The engine should forget the current conversation.
        pass


# -----------------------------------------------------------------------------
# SIMPLE ENGINE: RuleBasedEngine
# -----------------------------------------------------------------------------
# This engine works like a basic IF/ELSE chain:
#   "If the user's message contains the word 'history', reply with this text."
#
# It is fast and always predictable, but it can only handle keywords it
# already knows about.  The smarter MLEngine (ml_engine.py) replaced this
# as the default, but it's kept here as a backup — you can switch back to it
# in app.py with just two line changes.
# -----------------------------------------------------------------------------
class RuleBasedEngine(ChatEngine):

    # RULES is a dictionary (a lookup table).
    # Each KEY is a group of trigger words (a tuple).
    # Each VALUE is the reply text to send when any trigger word is found.
    RULES = {
        ("what is", "define", "tell me about", "explain"): (
            "The Samphor (ស័មភោ) is a barrel-shaped, double-headed drum central to Cambodian classical music."
        ),
        ("history", "origin", "ancient", "old"): (
            "The Samphor has roots stretching back to the Angkor period (9th–15th century CE), "
            "depicted in bas-reliefs at Angkor Wat."
        ),
        ("material", "made of", "wood", "skin"): (
            "The body is carved from a single log of jackfruit wood; the heads are covered with cow or goat skin."
        ),
        ("play", "technique", "how to", "hit", "strike"): (
            "The Samphor is played with both hands — the right hand uses fingertips and palm strikes "
            "while the left hand controls tension on the head."
        ),
        ("pinpeat", "ensemble", "orchestra"): (
            "The Samphor is the rhythmic anchor of the Pinpeat ensemble, "
            "cueing other instruments and marking ceremonial sections."
        ),
        ("ceremony", "ritual", "wedding", "festival"): (
            "It is used in royal ceremonies, religious festivals, shadow-puppet theatre (Sbek Thom), "
            "and classical ballet (Robam Kbach Boran)."
        ),
        ("hello", "hi", "hey", "greet"): "Hello! Ask me anything about the Samphor drum.",
        ("bye", "goodbye", "farewell", "see you"): "Goodbye! Keep the rhythm of tradition alive.",
    }

    def __init__(self):
        # _history stores every message exchanged so far.
        # Each item is a small dictionary like: {"role": "user", "content": "Hi"}
        # The leading underscore (_) means "don't touch this from outside the class."
        self._history = []

    def respond(self, user_input: str, context: dict | None = None) -> tuple[str, dict]:
        lower = user_input.lower()
        self._history.append({"role": "user", "content": user_input})

        for keywords, reply in self.RULES.items():
            if any(kw in lower for kw in keywords):
                self._history.append({"role": "assistant", "content": reply})
                return reply, {}

        fallback = (
            "I'm not sure about that. Try asking about the Samphor's history, "
            "materials, playing technique, or its role in Cambodian ceremonies."
        )
        self._history.append({"role": "assistant", "content": fallback})
        return fallback, {}

    def reset(self) -> None:
        # list.clear() empties the list completely.
        # After this, _history == []
        self._history.clear()
