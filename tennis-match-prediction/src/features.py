"""Feature engineering for ATP match prediction.

Builds a leak-free training set by iterating through matches in
chronological order: every feature attached to a match is computed only
from matches played strictly before it (pre-match Elo, form, H2H, ...).
"""

from __future__ import annotations

import glob
import os
import re
from collections import defaultdict, deque

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Tour levels kept for training (Grand Slam, Masters, 500, 250, Finals).
# Davis Cup ("D") is excluded: uneven stakes and lineups add noise.
LEVELS = {"G", "M", "F", "250", "500"}

INITIAL_ELO = 1500.0
SURFACES = ("Hard", "Clay", "Grass", "Carpet")

FEATURES = [
    "elo_diff",          # overall Elo, p1 - p2
    "surface_elo_diff",  # surface-specific Elo, p1 - p2
    "form_diff",         # win rate over last 25 matches, p1 - p2
    "grass_wr_diff",     # career win rate on today's surface, p1 - p2
    "h2h_diff",          # head-to-head wins, p1 - p2
    "rank_diff",         # log2(p2_rank) - log2(p1_rank)  (positive = p1 better)
    "age_diff",          # p1 - p2, years
    "ht_diff",           # p1 - p2, cm
    "best_of_5",         # match context: 1 if best-of-five
]


_ABBREV = re.compile(r"^(?:[A-Z]\.\s*)+")  # "N. Djokovic", "J.L. Struff", ...


def _norm(s: str) -> str:
    return s.strip().lower().replace("-", " ")


def _normalize_api_names(supp: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Map the API's abbreviated names ('N. Djokovic') onto the historical
    full names ('Novak Djokovic') so the supplement updates the same players'
    ratings. Keys are (first initial, surname), with first/last-token
    fallbacks for middle-name mismatches ('T. A. Tirante' vs 'Thiago Agustin
    Tirante'). Collisions resolve to the player with more matches; unmatched
    names (true newcomers) keep the API form."""
    counts = pd.concat([hist["winner_name"], hist["loser_name"]]).value_counts()
    lookup: dict = {}

    def index(key, full, n):
        if key[1] and (key not in lookup or n > lookup[key][1]):
            lookup[key] = (full, n)

    for full, n in counts.items():
        parts = str(full).split(" ", 1)
        if len(parts) < 2 or not parts[0]:
            continue
        initial, surname = parts[0][0].upper(), _norm(parts[1])
        tokens = surname.split()
        index((initial, surname), full, n)
        index((initial, tokens[-1]), full, n)
        index((initial, tokens[0]), full, n)

    def fix(name):
        if not isinstance(name, str):
            return name
        m = _ABBREV.match(name)
        if not m:
            return name
        surname = _norm(name[m.end():])
        tokens = surname.split()
        for key in ((name[0], surname), (name[0], tokens[-1]), (name[0], tokens[0])):
            hit = lookup.get((key[0].upper(), key[1]))
            if hit:
                return hit[0]
        return name

    supp = supp.copy()
    supp["winner_name"] = supp["winner_name"].map(fix)
    supp["loser_name"] = supp["loser_name"].map(fix)
    return supp


def load_matches() -> pd.DataFrame:
    """Load all yearly CSVs, keep completed tour-level matches, sort by date."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "[12][0-9][0-9][0-9].csv")))
    if not files:
        raise FileNotFoundError(
            f"No data in {DATA_DIR}. Run data/download_data.sh first."
        )
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    supplement = os.path.join(DATA_DIR, "api_supplement.csv")
    if os.path.exists(supplement):
        supp = _normalize_api_names(pd.read_csv(supplement), df)
        df = pd.concat([df, supp], ignore_index=True)
    # The API supplement can overlap the yearly CSVs (e.g. January events).
    df = df.drop_duplicates(subset=["tourney_date", "winner_name", "loser_name"])
    df = df[df["tourney_level"].astype(str).isin(LEVELS)]
    df = df[~df["score"].astype(str).str.contains("W/O|DEF|ABN", na=False)]
    df = df.dropna(subset=["winner_name", "loser_name", "surface"])
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
    # Order within a tournament follows match_num (round order).
    df = df.sort_values(["tourney_date", "tourney_id", "match_num"])
    return df.reset_index(drop=True)


# Module-level factories keep PlayerBook picklable (lambdas are not).
def _initial_elo() -> float:
    return INITIAL_ELO


def _recent_window() -> deque:
    return deque(maxlen=25)


class PlayerBook:
    """Running state per player, updated one match at a time."""

    def __init__(self):
        self.elo = defaultdict(_initial_elo)
        self.surface_elo = {s: defaultdict(_initial_elo) for s in SURFACES}
        self.n_matches = defaultdict(int)
        self.n_surface = {s: defaultdict(int) for s in SURFACES}
        self.recent = defaultdict(_recent_window)  # 1 = win
        self.surface_w = {s: defaultdict(int) for s in SURFACES}
        self.h2h = defaultdict(int)  # (a, b) -> wins of a over b

    @staticmethod
    def _k(n_played: int, slam: bool) -> float:
        """FiveThirtyEight-style decaying K, boosted a touch for Slams."""
        k = 250.0 / (n_played + 5) ** 0.4
        return k * 1.1 if slam else k

    @staticmethod
    def expect(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def snapshot(self, p: str, surface: str) -> dict:
        recent = self.recent[p]
        surf_n = self.n_surface[surface][p]
        return {
            "elo": self.elo[p],
            "surface_elo": self.surface_elo[surface][p],
            "form": (sum(recent) / len(recent)) if recent else 0.5,
            "surface_wr": (self.surface_w[surface][p] / surf_n) if surf_n >= 5 else 0.5,
            "n_matches": self.n_matches[p],
        }

    def update(self, winner: str, loser: str, surface: str, slam: bool) -> None:
        exp_w = self.expect(self.elo[winner], self.elo[loser])
        self.elo[winner] += self._k(self.n_matches[winner], slam) * (1 - exp_w)
        self.elo[loser] -= self._k(self.n_matches[loser], slam) * (1 - exp_w)

        s_elo = self.surface_elo[surface]
        exp_ws = self.expect(s_elo[winner], s_elo[loser])
        s_elo[winner] += self._k(self.n_surface[surface][winner], slam) * (1 - exp_ws)
        s_elo[loser] -= self._k(self.n_surface[surface][loser], slam) * (1 - exp_ws)

        for p, won in ((winner, 1), (loser, 0)):
            self.n_matches[p] += 1
            self.n_surface[surface][p] += 1
            self.recent[p].append(won)
        self.surface_w[surface][winner] += 1
        self.h2h[(winner, loser)] += 1


def _feature_row(a: dict, b: dict, h2h_ab: int, h2h_ba: int,
                 rank_a, rank_b, age_a, age_b, ht_a, ht_b,
                 best_of_5: int) -> list:
    def log_rank(r):
        return np.log2(r) if pd.notna(r) and r > 0 else np.log2(500)

    return [
        a["elo"] - b["elo"],
        a["surface_elo"] - b["surface_elo"],
        a["form"] - b["form"],
        a["surface_wr"] - b["surface_wr"],
        h2h_ab - h2h_ba,
        log_rank(rank_b) - log_rank(rank_a),
        (age_a - age_b) if pd.notna(age_a) and pd.notna(age_b) else 0.0,
        (ht_a - ht_b) if pd.notna(ht_a) and pd.notna(ht_b) else 0.0,
        best_of_5,
    ]


def build_dataset(min_prior_matches: int = 10, seed: int = 7):
    """Single chronological pass -> (X, y, meta) plus the final PlayerBook.

    Each match becomes one row. Player order is randomised (p1 vs p2) so the
    label carries no positional information; y = 1 iff p1 won.
    Matches where either player has fewer than `min_prior_matches` on record
    are skipped as training rows (cold-start Elo is meaningless) but still
    update the ratings.
    """
    df = load_matches()
    rng = np.random.default_rng(seed)
    book = PlayerBook()
    rows, labels, meta = [], [], []

    for m in df.itertuples(index=False):
        surface = m.surface if m.surface in SURFACES else "Hard"
        slam = m.tourney_level == "G"
        w, l = m.winner_name, m.loser_name

        if (book.n_matches[w] >= min_prior_matches
                and book.n_matches[l] >= min_prior_matches):
            sw, sl = book.snapshot(w, surface), book.snapshot(l, surface)
            bo5 = 1 if m.best_of == 5 else 0
            winner_first = bool(rng.integers(0, 2))
            if winner_first:
                row = _feature_row(sw, sl, book.h2h[(w, l)], book.h2h[(l, w)],
                                   m.winner_rank, m.loser_rank,
                                   m.winner_age, m.loser_age,
                                   m.winner_ht, m.loser_ht, bo5)
                labels.append(1)
            else:
                row = _feature_row(sl, sw, book.h2h[(l, w)], book.h2h[(w, l)],
                                   m.loser_rank, m.winner_rank,
                                   m.loser_age, m.winner_age,
                                   m.loser_ht, m.winner_ht, bo5)
                labels.append(0)
            rows.append(row)
            meta.append((m.tourney_date, m.tourney_name, surface))

        book.update(w, l, surface, slam)

    X = pd.DataFrame(rows, columns=FEATURES)
    y = np.array(labels)
    meta = pd.DataFrame(meta, columns=["date", "tourney", "surface"])
    return X, y, meta, book
