# =============================================================================
# train.py
# =============================================================================
# This script trains the neural network model from scratch.
# Run it once before starting the chatbot, or whenever you update intents.json.
#
#   HOW TO RUN:
#     python train.py
#
#   WHAT IT DOES:
#     1. Creates an MLEngine object
#     2. Calls engine.train() — this does all the heavy lifting in ml_engine.py
#     3. Prints a summary of the results (accuracy, loss, pass/fail)
#
#   CONNECTIONS:
#     → Calls engine.train() in engine/ml_engine.py
#     ← engine.train() reads data/intents.json
#     ← engine.train() saves files to model/
#
#   YOU DO NOT NEED TO RUN THIS FILE AGAIN UNLESS:
#     - You add new intents to data/intents.json
#     - You change the model architecture in ml_engine.py
#     - The model/ files get deleted
# =============================================================================

import sys   # sys.exit() lets us exit the program with an error code

from engine.ml_engine import MLEngine


def main() -> None:
    print("=" * 60)
    print("  Samphor Expert System - Model Training")
    print("=" * 60)

    # Create an MLEngine instance.
    # If model files already exist, MLEngine will load them in __init__.
    # But train() will overwrite them with newly trained weights, which is fine.
    engine = MLEngine()

    # Run the full training pipeline (see engine/ml_engine.py for details).
    # Returns a tuple: (final_accuracy, final_loss)
    result = engine.train()

    # Sanity check: if train() somehow returned nothing, something went wrong.
    if result is None:
        print("[train.py] ERROR: train() returned None. Check ml_engine.py.", file=sys.stderr)
        sys.exit(1)   # exit with code 1 = failure (code 0 = success)

    # Unpack the two values from the tuple.
    final_acc, final_loss = result

    # ------------------------------------------------------------------
    # Print a readable training summary.
    # f-strings let you put variables inside strings:
    #   f"accuracy: {final_acc * 100:.2f}%"
    #   → final_acc is e.g. 0.9929
    #   → * 100 converts to percentage: 99.29
    #   → :.2f formats to 2 decimal places: "99.29%"
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Training Summary")
    print("=" * 60)

    # Sentence transformer training returns (1.0, 0.0) as sentinels
    # because cosine-similarity models don't expose a loss/accuracy metric.
    if final_loss == 0.0 and final_acc == 1.0:
        print("  Engine         : Sentence Transformer (Fix 10)")
        print("  Status         : PASS — embeddings generated successfully")
        print("  Note           : Run the chatbot and test with varied phrasings")
    else:
        print(f"  Final accuracy : {final_acc * 100:.2f}%")
        print(f"  Final loss     : {final_loss:.4f}")
        if final_acc >= 0.70:
            print("  Status         : PASS (>= 70% accuracy threshold)")
        else:
            print("  Status         : WARN - accuracy below 70% target")
            print("  Consider adding more training patterns to data/intents.json.")

    print("=" * 60)
    print("  Model artefacts saved to model/")
    print("    - embeddings.npy  (training pattern embeddings)")
    print("    - labels.pkl      (intent label per pattern)")
    print("=" * 60)


# This guard ensures main() only runs when you execute THIS file directly.
# If another file does "import train", main() will NOT run automatically.
if __name__ == "__main__":
    main()
