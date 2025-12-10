#!/usr/bin/env python3
"""Create team-round features from the Kaggle CS:GO economy dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import pandas as pd

KAGGLE_DIR = Path("data/kaggle")
ECONOMY_CSV = KAGGLE_DIR / "economy.csv"
RESULTS_CSV = KAGGLE_DIR / "results.csv"
OUTPUT_CSV = Path("data/processed/team_round_features_kaggle.csv")

ROUND_RANGE = range(1, 31)


def _side_for_round(start_side: str, round_num: int) -> str:
    start_side = (start_side or "").strip().lower()
    if round_num <= 15:
        return "CT" if start_side == "ct" else "T"
    return "T" if start_side == "ct" else "CT"


def _iter_round_records(row: pd.Series, match_meta: dict) -> Iterator[dict]:
    score_team1 = 0
    score_team2 = 0
    for rnd in ROUND_RANGE:
        eq_t1 = row.get(f"{rnd}_t1")
        eq_t2 = row.get(f"{rnd}_t2")
        winner = row.get(f"{rnd}_winner")
        if pd.isna(eq_t1) and pd.isna(eq_t2) and pd.isna(winner):
            continue
        # Build per-team records before updating the running score
        for team_idx, team_name in enumerate((row["team_1"], row["team_2"]), start=1):
            opp_name = row["team_2"] if team_idx == 1 else row["team_1"]
            eq_val = row.get(f"{rnd}_t{team_idx}") or 0.0
            team_score_before = score_team1 if team_idx == 1 else score_team2
            opp_score_before = score_team2 if team_idx == 1 else score_team1
            record = {
                "dataset": "kaggle",
                "match_id": f"{row['match_id']}_{row['_map']}",
                "match_name": f"{row['team_1']} vs {row['team_2']}",
                "match_date": match_meta.get("date"),
                "map_name": row["_map"],
                "round_num": rnd,
                "round_duration_seconds": 0.0,
                "team_name": team_name,
                "opponent_name": opp_name,
                "team_side": _side_for_round(
                    row["t1_start"] if team_idx == 1 else row["t2_start"], rnd
                ),
                "team_score_before": team_score_before,
                "opp_score_before": opp_score_before,
                "score_diff": team_score_before - opp_score_before,
                "team_eq_start": eq_val,
                "team_eq_end": eq_val,
                "team_eq_spend": 0.0,
                "team_alive_end": 0,
                "team_total_utility": 0.0,
                "team_kills": 0,
                "team_deaths": 0,
                "team_damage": 0,
                "team_round_result": int(winner == team_idx),
                "team_cumulative_kills": 0,
                "team_cumulative_deaths": 0,
                "team_cumulative_damage": 0,
                "team_cumulative_eq_spend": 0.0,
                "team_is_match_winner": int(match_meta.get("match_winner") == team_idx),
                "match_winner": match_meta.get("match_winner_name"),
            }
            yield record

        # Update running score after creating the records
        if winner == 1:
            score_team1 += 1
        elif winner == 2:
            score_team2 += 1


def _load_match_metadata() -> dict[str, dict]:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError("Kaggle results.csv not found. Please place it under data/kaggle/")
    results = pd.read_csv(RESULTS_CSV)
    results["match_key"] = results["match_id"].astype(str) + "_" + results["_map"].astype(str)
    indexed = results.set_index("match_key")
    return indexed.to_dict(orient="index")


def preprocess(limit: int | None = None) -> None:
    if not ECONOMY_CSV.exists():
        raise FileNotFoundError("Kaggle economy.csv not found. Please place it under data/kaggle/")

    metadata = _load_match_metadata()
    economy = pd.read_csv(ECONOMY_CSV)
    if limit:
        economy = economy.head(limit)

    records: list[dict] = []
    for _, row in economy.iterrows():
        match_key = f"{row['match_id']}_{row['_map']}"
        meta_row = metadata.get(match_key)
        if meta_row is None:
            continue
        winner_idx = meta_row.get("match_winner")
        if pd.isna(winner_idx):
            winner_idx = None
        meta = {
            "date": meta_row.get("date"),
            "match_winner": winner_idx,
            "match_winner_name": row["team_1"] if winner_idx == 1 else row["team_2"] if winner_idx == 2 else None,
        }
        for record in _iter_round_records(row, meta):
            records.append(record)

    if not records:
        raise SystemExit("No Kaggle round records generated. Check source files.")

    df = pd.DataFrame(records)
    df.sort_values(["match_id", "round_num", "team_side"], inplace=True)
    df = df.fillna(0)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} Kaggle team-round samples to {OUTPUT_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Kaggle CS:GO economy data")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of matches to process")
    args = parser.parse_args()
    preprocess(limit=args.limit or None)


if __name__ == "__main__":
    main()
