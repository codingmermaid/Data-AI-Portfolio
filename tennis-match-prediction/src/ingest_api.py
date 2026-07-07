"""Backfill recent ATP results from api-tennis.com into the dataset.

The CSV feed lags the live season (2026 stops in January). This script pulls
completed ATP singles matches from the API and writes them to
data/api_supplement.csv in the same schema as the yearly CSVs, where
features.load_matches() picks them up automatically (deduplicated).

Setup:
  export TENNIS_API_KEY=...        # your api-tennis.com key (or pass --key)
  python src/ingest_api.py --start 2026-01-01

Note: run this from your own machine - some sandboxed/cloud environments
block api-tennis.com.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta

import pandas as pd
import requests

BASE = "https://api.api-tennis.com/tennis/"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_CSV = os.path.join(DATA_DIR, "api_supplement.csv")

GRAND_SLAMS = ("australian open", "roland garros", "french open",
               "wimbledon", "us open")
MASTERS = ("indian wells", "miami", "monte carlo", "madrid", "rome",
           "canada", "toronto", "montreal", "cincinnati", "shanghai", "paris")

# Columns load_matches() actually uses; the rest of the schema is left NaN.
COLUMNS = ["tourney_id", "tourney_name", "surface", "tourney_level", "indoor",
           "tourney_date", "match_num", "winner_name", "winner_rank",
           "winner_age", "winner_ht", "loser_name", "loser_rank", "loser_age",
           "loser_ht", "score", "best_of", "round"]


def call(method: str, key: str, **params) -> list[dict]:
    params = {"method": method, "APIkey": key, **params}
    r = requests.get(BASE, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if int(payload.get("success", 0)) != 1:
        raise RuntimeError(f"{method} failed: {payload}")
    return payload.get("result", [])


def tournament_surfaces(key: str) -> dict[str, str]:
    """tournament_key -> surface, where the API provides it."""
    surfaces = {}
    for t in call("get_tournaments", key):
        s = (t.get("tournament_surface") or "").strip().capitalize()
        if s in ("Hard", "Clay", "Grass", "Carpet"):
            surfaces[str(t.get("tournament_key"))] = s
    return surfaces


def infer_level(name: str) -> tuple[str, int]:
    low = name.lower()
    if any(g in low for g in GRAND_SLAMS):
        return "G", 5
    if any(m in low for m in MASTERS):
        return "M", 3
    return "250", 3  # conservative default for other tour events


def infer_surface(name: str, when: date) -> str:
    low = name.lower()
    if "wimbledon" in low or when.month in (6,) or "grass" in low:
        return "Grass"
    if when.month in (4, 5) or any(k in low for k in ("roland", "clay", "monte")):
        return "Clay"
    return "Hard"


def fetch(key: str, start: date, stop: date) -> pd.DataFrame:
    surf_map = tournament_surfaces(key)
    rows, chunk = [], timedelta(days=13)
    d = start
    while d <= stop:
        d2 = min(d + chunk, stop)
        events = call("get_fixtures", key,
                      date_start=d.isoformat(), date_stop=d2.isoformat(),
                      timezone="UTC")
        for e in events:
            if "atp" not in str(e.get("event_type_type", "")).lower():
                continue
            if "doubles" in str(e.get("event_type_type", "")).lower():
                continue
            if str(e.get("event_status", "")).lower() != "finished":
                continue
            winner_flag = str(e.get("event_winner", ""))
            p1, p2 = e.get("event_first_player"), e.get("event_second_player")
            if not p1 or not p2 or "player" not in winner_flag.lower():
                continue
            winner, loser = (p1, p2) if "first" in winner_flag.lower() else (p2, p1)
            when = pd.to_datetime(e.get("event_date")).date()
            tname = e.get("tournament_name", "")
            level, best_of = infer_level(tname)
            surface = surf_map.get(str(e.get("tournament_key")),
                                   infer_surface(tname, when))
            # set score like "3 - 1" -> keep API string; not parsed downstream
            rows.append({
                "tourney_id": f"api-{e.get('tournament_key')}",
                "tourney_name": tname, "surface": surface,
                "tourney_level": level, "indoor": "O",
                "tourney_date": int(when.strftime("%Y%m%d")),
                "match_num": int(e.get("event_key", 0)),
                "winner_name": winner, "loser_name": loser,
                "winner_rank": None, "winner_age": None, "winner_ht": None,
                "loser_rank": None, "loser_age": None, "loser_ht": None,
                "score": e.get("event_final_result", ""),
                "best_of": best_of, "round": e.get("tournament_round", ""),
            })
        print(f"  {d} .. {d2}: {len(rows)} matches so far")
        d = d2 + timedelta(days=1)
        time.sleep(0.4)  # stay well under the rate limit
    return pd.DataFrame(rows, columns=COLUMNS)


def default_start() -> str:
    """Day after the newest record in the yearly CSVs (avoids overlap:
    CSVs stamp tournament start dates, the API stamps match dates)."""
    import glob
    newest = 0
    for f in glob.glob(os.path.join(DATA_DIR, "[12][0-9][0-9][0-9].csv")):
        d = pd.read_csv(f, usecols=["tourney_date"])["tourney_date"].max()
        newest = max(newest, int(d) if pd.notna(d) else 0)
    if not newest:
        return "2026-01-01"
    last = pd.to_datetime(str(newest), format="%Y%m%d").date()
    return (last + timedelta(days=14)).isoformat()  # skip that event's span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=os.environ.get("TENNIS_API_KEY"))
    ap.add_argument("--start", default=None,
                    help="YYYY-MM-DD; default: after the newest CSV record")
    ap.add_argument("--stop", default=date.today().isoformat())
    args = ap.parse_args()
    if not args.key:
        sys.exit("Set TENNIS_API_KEY or pass --key (never commit the key).")
    if args.start is None:
        args.start = default_start()
        print(f"start date (auto): {args.start}")

    df = fetch(args.key, date.fromisoformat(args.start),
               date.fromisoformat(args.stop))
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {len(df)} matches -> {OUT_CSV}")
    print("Re-run `python src/train.py` to refresh ratings and the model.")


if __name__ == "__main__":
    main()
