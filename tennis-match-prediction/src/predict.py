"""Predict a single matchup with the trained model.

Usage:
  python src/predict.py "Novak Djokovic" "Felix Auger-Aliassime" \
      --surface Grass --best-of 5 --p1-rank 7 --p2-rank 3

Ranks can be overridden because the dataset may lag the live rankings;
everything else (Elo, form, H2H, age, height) comes from the data.
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd

from features import FEATURES, load_matches

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def last_known(df: pd.DataFrame, player: str, col_w: str, col_l: str):
    """Most recent recorded value (rank/age/height) for a player."""
    as_w = df.loc[df["winner_name"] == player, ["tourney_date", col_w]]
    as_l = df.loc[df["loser_name"] == player, ["tourney_date", col_l]]
    both = pd.concat([as_w.rename(columns={col_w: "v"}),
                      as_l.rename(columns={col_l: "v"})]).dropna()
    if both.empty:
        return np.nan
    return both.sort_values("tourney_date")["v"].iloc[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("p1")
    ap.add_argument("p2")
    ap.add_argument("--surface", default="Grass")
    ap.add_argument("--best-of", type=int, default=5)
    ap.add_argument("--p1-rank", type=float, default=None)
    ap.add_argument("--p2-rank", type=float, default=None)
    args = ap.parse_args()

    model = joblib.load(os.path.join(OUT_DIR, "model.joblib"))
    book = joblib.load(os.path.join(OUT_DIR, "book.joblib"))
    df = load_matches()

    for p in (args.p1, args.p2):
        if book.n_matches[p] == 0:
            raise SystemExit(f"No matches on record for {p!r} - check spelling.")

    s1 = book.snapshot(args.p1, args.surface)
    s2 = book.snapshot(args.p2, args.surface)
    rank1 = args.p1_rank or last_known(df, args.p1, "winner_rank", "loser_rank")
    rank2 = args.p2_rank or last_known(df, args.p2, "winner_rank", "loser_rank")
    age1 = last_known(df, args.p1, "winner_age", "loser_age")
    age2 = last_known(df, args.p2, "winner_age", "loser_age")
    ht1 = last_known(df, args.p1, "winner_ht", "loser_ht")
    ht2 = last_known(df, args.p2, "winner_ht", "loser_ht")

    def log_rank(r):
        return np.log2(r) if pd.notna(r) and r > 0 else np.log2(500)

    row = pd.DataFrame([[
        s1["elo"] - s2["elo"],
        s1["surface_elo"] - s2["surface_elo"],
        s1["form"] - s2["form"],
        s1["surface_wr"] - s2["surface_wr"],
        book.h2h[(args.p1, args.p2)] - book.h2h[(args.p2, args.p1)],
        book.h2h_surface[(args.p1, args.p2, args.surface)]
        - book.h2h_surface[(args.p2, args.p1, args.surface)],
        log_rank(rank2) - log_rank(rank1),
        age1 - age2,
        ht1 - ht2,
        1 if args.best_of == 5 else 0,
    ]], columns=FEATURES)

    p1_win = float(model.predict_proba(row)[0, 1])

    print(f"\n{args.p1}  vs  {args.p2}  ({args.surface}, best of {args.best_of})")
    print("-" * 60)
    for label, s, rank in ((args.p1, s1, rank1), (args.p2, s2, rank2)):
        print(f"{label:26s} elo {s['elo']:7.1f}  {args.surface.lower()}-elo "
              f"{s['surface_elo']:7.1f}  form(L25) {s['form']:.2f}  rank {rank:.0f}")
    h2h = (book.h2h[(args.p1, args.p2)], book.h2h[(args.p2, args.p1)])
    print(f"head-to-head {args.p1} {h2h[0]} - {h2h[1]} {args.p2}")
    print("-" * 60)
    print(f"P({args.p1} wins) = {p1_win:.1%}")
    print(f"P({args.p2} wins) = {1 - p1_win:.1%}")
    fav, prob = (args.p1, p1_win) if p1_win >= 0.5 else (args.p2, 1 - p1_win)
    print(f">>> model favours: {fav.upper()} ({prob:.1%})\n")


if __name__ == "__main__":
    main()
