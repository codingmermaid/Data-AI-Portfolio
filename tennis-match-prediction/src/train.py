"""Train and evaluate match-winner models on a chronological split.

Split (no shuffling — the past predicts the future):
  train      2000-2021
  validation 2022-2023   (model selection)
  test       2024-2026   (reported once, untouched during selection)

Usage: python src/train.py
Writes outputs/metrics.json, outputs/model.joblib, outputs/book.joblib,
outputs/calibration.png.
"""

from __future__ import annotations

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, PlayerBook, build_dataset

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def evaluate(name, y_true, proba):
    return {
        "model": name,
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, proba >= 0.5)), 4),
        "log_loss": round(float(log_loss(y_true, proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, y, meta, book = build_dataset()
    year = meta["date"].dt.year
    tr, va, te = year <= 2021, (year >= 2022) & (year <= 2023), year >= 2024
    print(f"rows: train={tr.sum()}  val={va.sum()}  test={te.sum()}")

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.06, max_iter=400,
            l2_regularization=1.0, random_state=7,
        ),
    }

    val_scores, fitted = {}, {}
    for name, model in models.items():
        model.fit(X[tr], y[tr])
        proba_va = model.predict_proba(X[va])[:, 1]
        val_scores[name] = log_loss(y[va], proba_va)
        fitted[name] = model
        print(f"{name}: val log-loss {val_scores[name]:.4f}")

    best_name = min(val_scores, key=val_scores.get)
    best = fitted[best_name]

    # Baselines on the test years for context.
    elo_proba = 1 / (1 + 10 ** (-X.loc[te, "elo_diff"] / 400))
    rank_pick = (X.loc[te, "rank_diff"] > 0).astype(int)

    metrics = {
        "split": {"train": "2000-2021", "val": "2022-2023", "test": "2024-2026"},
        "validation_log_loss": {k: round(v, 4) for k, v in val_scores.items()},
        "selected_model": best_name,
        "test": [
            evaluate("baseline_higher_rank",
                     y[te], rank_pick.to_numpy().astype(float)),
            evaluate("baseline_elo_only", y[te], elo_proba.to_numpy()),
            evaluate(best_name, y[te], best.predict_proba(X[te])[:, 1]),
        ],
    }

    # Calibration on test years.
    proba_te = best.predict_proba(X[te])[:, 1]
    frac_pos, mean_pred = calibration_curve(y[te], proba_te, n_bins=10)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.plot(mean_pred, frac_pos, "o-")
    ax.set_xlabel("Predicted P(p1 wins)")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration, test 2024-2026 ({best_name})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "calibration.png"), dpi=120)

    # Refit the selected model on ALL rows so the final predictor uses
    # every match played, then persist model + ratings snapshot.
    best.fit(X, y)
    joblib.dump(best, os.path.join(OUT_DIR, "model.joblib"))
    joblib.dump(book, os.path.join(OUT_DIR, "book.joblib"))

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics["test"], indent=2))


if __name__ == "__main__":
    main()
