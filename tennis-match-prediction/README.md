# Wimbledon 2026 QF: Djokovic vs Auger-Aliassime — Match Prediction

End-to-end ML project that predicts the winner of an ATP tennis match from
historical results, applied to the **Novak Djokovic vs Felix Auger-Aliassime**
Wimbledon quarter-final (7 July 2026).

> **Result: the model gives Djokovic a 61.6% win probability**, driven by his
> overall Elo (2215 vs 2012) and a large grass-Elo gap (2197 vs 1674).
> Test-set performance: **64.1% accuracy / 0.621 log-loss / 0.707 ROC-AUC**
> on ~4.7k unseen 2024–26 matches — better than Elo-only (63.7% / 0.638) and
> higher-rank (63.7%) baselines.

## Workflow

```
data/download_data.sh          1. DATA: ~78k ATP tour-level matches, 2000-2026,
        │                         one CSV per season (TML-Database, Sackmann schema)
        ▼
src/features.py                2. FEATURES: one chronological pass builds pre-match
        │                         state per player — overall Elo, surface Elo,
        │                         last-25 form, surface win rate, head-to-head.
        │                         Strictly leak-free: a row only uses matches
        │                         played before it; player order is randomised.
        ▼
src/train.py                   3. MODEL: chronological split (train 2000-21,
        │                         val 2022-23, test 2024-26). Logistic regression
        │                         vs gradient boosting, selected on validation
        │                         log-loss; test metrics + calibration reported,
        │                         then refit on all data.
        ▼
src/predict.py                 4. PREDICT: snapshot current ratings, score the
                                  matchup (surface, format, ranks as inputs).
```

## Reproduce

```bash
pip install -r requirements.txt
bash data/download_data.sh        # ~16 MB of CSVs (not committed)
python src/train.py               # writes outputs/{metrics.json,model.joblib,...}
python src/predict.py "Novak Djokovic" "Felix Auger-Aliassime" \
    --surface Grass --best-of 5 --p1-rank 7 --p2-rank 3
```

Or read the full story with EDA, evaluation, calibration and feature-importance
plots in [`notebooks/wimbledon_qf_prediction.ipynb`](notebooks/wimbledon_qf_prediction.ipynb).

## Data

- **Source**: [TML-Database](https://github.com/Tennismylife/TML-Database) —
  community-maintained, live-updated ATP results in the same schema as Jeff
  Sackmann's well-known `tennis_atp` dataset (which was unreachable at build time).
- **Scope**: completed tour-level matches (Slams, Masters, 500s, 250s, Finals);
  Davis Cup and walkovers excluded.
- **Known gap**: the feed lags the live season — 2026 coverage stops in January,
  so both players' 2026 form (including their four wins at this Wimbledon) is not
  in the features. That most likely overstates Djokovic's edge, since
  Auger-Aliassime's strong 2026 season is what made him the 3-seed.

## Features (all as player1 − player2 differences)

| feature | idea |
|---|---|
| `elo_diff` | overall Elo; K decays with career matches (FiveThirtyEight-style), ×1.1 at Slams |
| `surface_elo_diff` | separate Elo book per surface — grass specialists exist |
| `form_diff` | win rate over the last 25 matches |
| `grass_wr_diff` | career win rate on the match surface |
| `h2h_diff` | head-to-head balance |
| `rank_diff` | log₂ ATP rank difference |
| `age_diff`, `ht_diff` | demographics |
| `best_of_5` | Slam format favours favourites |

## Honest limitations / next steps

- No within-tournament signal: fatigue (Auger-Aliassime came through a 4h26
  five-setter in R16), injuries, or live odds. Adding bookmaker odds is the
  best-documented accuracy upgrade (~70% ceiling in the literature).
- Elo lags sharp late-career decline — relevant for a 39-year-old Djokovic.
- Next: automated weekly data refresh, set/game-level modelling, serving the
  predictor behind a small API.
