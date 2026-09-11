"""Runs the full pipeline and saves the resulting scoring bundle to
model/bundle.joblib for the live API (app.py) to load.

Usage:
    python train.py
"""

import os

from src.inference import build_bundle, save_bundle
from src.pipeline import run

MODEL_DIR = "model"


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    results = run()
    bundle = build_bundle(results)
    save_bundle(bundle, os.path.join(MODEL_DIR, "bundle.joblib"))
    print(f"Saved scoring bundle to {MODEL_DIR}/bundle.joblib")


if __name__ == "__main__":
    main()
