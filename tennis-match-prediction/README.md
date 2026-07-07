# Wimbledon 2026 QF: Djokovic vs Auger-Aliassime — Match Prediction

End-to-end ML project that predicts the winner of an ATP tennis match from
historical results, applied to the **Novak Djokovic vs Felix Auger-Aliassime**
Wimbledon quarter-final (7 July 2026).

> **Result: the model gives Djokovic a 66.8% win probability** (with live data
> through 6 July 2026), driven by his overall Elo (2184 vs 2007) and a large
> grass-Elo gap (2203 vs 1779). Test-set performance: **63.7% accuracy /
> 0.624 log-loss / 0.701 ROC-AUC** on ~6.2k unseen 2024–26 matches — better
> than Elo-only (63.2% / 0.641) and higher-rank (60.4%) baselines.
> Interestingly, backfilling the live 2026 season *strengthened* Djokovic's
> case (the stale-data version said 61.6%): Auger-Aliassime's raw 2026 form is
> weaker than his 3-seed suggests, while Djokovic's grass rating held.

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

## Run locally (VS Code)

```bash
cd tennis-match-prediction
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash data/download_data.sh        # skip if the data/ CSVs came with your download
python src/train.py               # writes outputs/{metrics.json,model.joblib,...}
python src/predict.py "Novak Djokovic" "Felix Auger-Aliassime" \
    --surface Grass --best-of 5 --p1-rank 7 --p2-rank 3
```

In VS Code: open this folder, pick the `.venv` interpreter
(`Ctrl/Cmd+Shift+P` → *Python: Select Interpreter*), and the notebooks run
with the built-in Jupyter support.

### Refresh with live data (api-tennis.com)

The CSV feed lags the live season. With an [api-tennis.com](https://api-tennis.com)
key you can backfill everything since the cutoff:

```bash
export TENNIS_API_KEY=your_key_here     # never commit the key
python src/ingest_api.py                # writes data/api_supplement.csv
python src/train.py                     # ratings + model now include it
```

`load_matches()` picks the supplement up automatically. Run it from your own
machine — sandboxed cloud environments may block the API host.

## Notebooks

| notebook | contents |
|---|---|
| [`notebooks/wimbledon_qf_prediction.ipynb`](notebooks/wimbledon_qf_prediction.ipynb) | the story end-to-end: workflow, modeling, the QF prediction |
| [`notebooks/eda_and_results.ipynb`](notebooks/eda_and_results.ipynb) | deeper preprocessing/EDA: missing data, coverage, upset rates, career Elo trajectories, feature separation/correlation, calibration, importance, final prediction |

## Data

- **Source**: [TML-Database](https://github.com/Tennismylife/TML-Database) —
  community-maintained, live-updated ATP results in the same schema as Jeff
  Sackmann's well-known `tennis_atp` dataset (which was unreachable at build time).
- **Scope**: completed tour-level matches (Slams, Masters, 500s, 250s, Finals);
  Davis Cup and walkovers excluded.
- **Live supplement**: the yearly CSVs lag the live season (2026 stops in
  January), so `src/ingest_api.py` backfills Feb–Jul 2026 from api-tennis.com
  into `data/api_supplement.csv`, with abbreviated API names ("N. Djokovic")
  normalized onto the historical full names at load time.
- **Known gap**: the Jan 19–31 2026 Australian Open window falls between the
  two sources (rerun ingest with `--start 2026-01-15` to close it).

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
