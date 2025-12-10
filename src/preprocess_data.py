#!/usr/bin/env python3
"""Create tabular features for team-round samples from the parsed ESTA data."""

from collections import defaultdict
from pathlib import Path
import json

import pandas as pd

PARSED_DIR = Path("data/esta/parsed")
PROCESSED_DIR = Path("data/processed")
OUTPUT_CSV = PROCESSED_DIR / "team_round_features.csv"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SIDE_FIELDS = {
    "CT": {
        "team": "ct_team",
        "score": "ct_score",
        "eq_start": "ct_eq_start",
        "eq_end": "ct_eq_end",
        "eq_spend": "ct_eq_spend",
        "alive": "ct_alive",
        "utility": "ct_total_utility",
        "kills": "ct_kills",
        "deaths": "ct_deaths",
        "damage": "ct_damage",
    },
    "T": {
        "team": "t_team",
        "score": "t_score",
        "eq_start": "t_eq_start",
        "eq_end": "t_eq_end",
        "eq_spend": "t_eq_spend",
        "alive": "t_alive",
        "utility": "t_total_utility",
        "kills": "t_kills",
        "deaths": "t_deaths",
        "damage": "t_damage",
    },
}


def _build_record(round_info: dict, match_id: str, match_winner: str, map_name: str, match_name: str, match_date: int, cumulative: dict, side: str) -> dict:
    mapping = SIDE_FIELDS[side]
    opp_side = "T" if side == "CT" else "CT"
    opp_mapping = SIDE_FIELDS[opp_side]

    team_name = round_info.get(mapping["team"])
    opp_name = round_info.get(opp_mapping["team"])
    if not team_name or not opp_name:
        return {}

    team_score = round_info.get(mapping["score"], 0)
    opp_score = round_info.get(opp_mapping["score"], 0)
    eq_start = round_info.get(mapping["eq_start"], 0)
    eq_end = round_info.get(mapping["eq_end"], 0)
    eq_spend = round_info.get(mapping["eq_spend"], 0)
    alive_end = round_info.get(mapping["alive"], 0)
    utility = round_info.get(mapping["utility"], 0)
    kills = round_info.get(mapping["kills"], 0)
    deaths = round_info.get(mapping["deaths"], 0)
    damage = round_info.get(mapping["damage"], 0)

    record = {
        "match_id": match_id,
        "match_name": match_name,
        "match_date": match_date,
        "map_name": map_name,
        "round_num": round_info.get("round_num"),
        "round_duration_seconds": round_info.get("round_duration_seconds") or 0,
        "team_name": team_name,
        "opponent_name": opp_name,
        "team_side": side,
        "team_score_before": team_score,
        "opp_score_before": opp_score,
        "score_diff": team_score - opp_score,
        "team_eq_start": eq_start,
        "team_eq_end": eq_end,
        "team_eq_spend": eq_spend,
        "team_alive_end": alive_end,
        "team_total_utility": utility,
        "team_kills": kills,
        "team_deaths": deaths,
        "team_damage": damage,
        "team_round_result": int(team_name == round_info.get("winning_team")),
        "team_cumulative_kills": cumulative[team_name]["kills"],
        "team_cumulative_deaths": cumulative[team_name]["deaths"],
        "team_cumulative_damage": cumulative[team_name]["damage"],
        "team_cumulative_eq_spend": cumulative[team_name]["eq_spend"],
        "team_is_match_winner": int(team_name == match_winner),
        "match_winner": match_winner,
    }

    cumulative[team_name]["kills"] += kills
    cumulative[team_name]["deaths"] += deaths
    cumulative[team_name]["damage"] += damage
    cumulative[team_name]["eq_spend"] += eq_spend

    return record


def _load_match(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def main() -> None:
    parsed_files = sorted(PARSED_DIR.glob("*.json"))
    if not parsed_files:
        raise SystemExit("No parsed ESTA files found. Run scripts/parse_esta.py first.")

    records = []
    for match_path in parsed_files:
        match_data = _load_match(match_path)
        match_id = match_data.get("match_id")
        match_winner = match_data.get("match_winner")
        map_name = match_data.get("map_name")
        match_name = match_data.get("match_name")
        match_date = match_data.get("match_date")
        cumulative = defaultdict(lambda: {"kills": 0, "deaths": 0, "damage": 0, "eq_spend": 0})
        for round_info in match_data.get("rounds", []):
            for side in ("CT", "T"):
                record = _build_record(
                    round_info,
                    match_id,
                    match_winner,
                    map_name,
                    match_name,
                    match_date,
                    cumulative,
                    side,
                )
                if record:
                    records.append(record)

    if not records:
        raise SystemExit("No training records generated.")

    df = pd.DataFrame(records)
    df = df.sort_values(["match_id", "round_num", "team_side"]).reset_index(drop=True)
    df = df.fillna(0)
    df["team_round_result"] = df["team_round_result"].astype(int)
    df["team_is_match_winner"] = df["team_is_match_winner"].astype(int)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} team-round samples to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
