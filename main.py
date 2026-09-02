# main.py

import sys

from scripts.ml_preprocessing import preprocess
from scripts.model import run_models


def main():
    try:
        print("🚀 STARTING FOOTBALL MATCH ANALYSIS PIPELINE\n")

        # ===============================
        # STEP 1: DATA + FEATURES (INSIDE preprocess)
        # ===============================
        print("⚙️ Creating dataset + features...")
        X_train, X_test, y_train, y_test = preprocess()
        print("✅ Dataset ready and split\n")

        # ===============================
        # STEP 2: MODEL TRAINING
        # ===============================
        print("🤖 Training models...\n")
        run_models(X_train, X_test, y_train, y_test)

        # ===============================
        # DONE
        # ===============================
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY")
        print("📊 Results generated. Check terminal & outputs folder.\n")

    except Exception as e:
        print("\n❌ ERROR OCCURRED:")
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()