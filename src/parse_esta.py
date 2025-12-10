#!/usr/bin/env python3
"""Summarize ESTA matches for fast downstream preprocessing."""

import argparse
import json
import lzma
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
from typing import Dict

from tqdm import tqdm

RAW_DIR = Path("data/esta/raw")
PARSED_DIR = Path("data/esta/parsed")
PARSED_DIR.mkdir(parents=True, exist_ok=True)


def _load_raw_match(path: Path) -> Dict:
    try:
        if path.name.endswith(".json.xz"):
            with lzma.open(path, "rt", encoding="utf-8") as fp:
                return json.load(fp)
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except json.JSONDecodeError:
        return {}


def _aggregate_round(round_data: Dict, tick_rate: int) -> Dict:
    frames = round_data.get("frames") or []
    final_frame = frames[-1] if frames else {}

    ct_frame = final_frame.get("ct", {})
    t_frame = final_frame.get("t", {})

    kills = round_data.get("kills", [])
    damages = round_data.get("damages", [])

    ct_kills = sum(1 for kill in kills if kill.get("attackerSide") == "CT")
    t_kills = sum(1 for kill in kills if kill.get("attackerSide") == "T")
    ct_deaths = sum(1 for kill in kills if kill.get("victimSide") == "CT")
    t_deaths = sum(1 for kill in kills if kill.get("victimSide") == "T")

    ct_damage = sum(entry.get("hpDamage", 0) for entry in damages if entry.get("attackerSide") == "CT")
    t_damage = sum(entry.get("hpDamage", 0) for entry in damages if entry.get("attackerSide") == "T")

    start_tick = round_data.get("startTick", 0)
    end_tick = round_data.get("endOfficialTick") or round_data.get("endTick") or 0
    round_duration_ticks = max(0, end_tick - start_tick)
    round_duration_seconds = round_duration_ticks / tick_rate if tick_rate else None

    return {
        "round_num": round_data.get("roundNum"),
        "winning_team": round_data.get("winningTeam"),
        "winning_side": round_data.get("winningSide"),
        "ct_score": round_data.get("ctScore", 0),
        "t_score": round_data.get("tScore", 0),
        "ct_team": round_data.get("ctTeam"),
        "t_team": round_data.get("tTeam"),
        "ct_eq_start": round_data.get("ctRoundStartEqVal", 0),
        "t_eq_start": round_data.get("tRoundStartEqVal", 0),
        "ct_eq_spend": round_data.get("ctRoundSpendMoney", 0),
        "t_eq_spend": round_data.get("tRoundSpendMoney", 0),
        "ct_eq_end": ct_frame.get("teamEqVal", 0),
        "t_eq_end": t_frame.get("teamEqVal", 0),
        "ct_alive": ct_frame.get("alivePlayers", 0),
        "t_alive": t_frame.get("alivePlayers", 0),
        "ct_total_utility": ct_frame.get("totalUtility", 0),
        "t_total_utility": t_frame.get("totalUtility", 0),
        "ct_kills": ct_kills,
        "t_kills": t_kills,
        "ct_deaths": ct_deaths,
        "t_deaths": t_deaths,
        "ct_damage": ct_damage,
        "t_damage": t_damage,
        "round_duration_ticks": round_duration_ticks,
        "round_duration_seconds": round_duration_seconds,
    }


def parse_match(raw_path: Path) -> Dict:
    match_data = _load_raw_match(raw_path)
    tick_rate = match_data.get("tickRate") or 64
    raw_rounds = [rnd for rnd in match_data.get("gameRounds", []) if not rnd.get("isWarmup")]
    if not raw_rounds:
        return {}

    rounds = [_aggregate_round(rnd, tick_rate) for rnd in raw_rounds]
    base_match_id = match_data.get("matchId") or match_data.get("demoId") or raw_path.stem
    map_name = match_data.get("mapName") or "unknown_map"
    match_id = f"{base_match_id}_{map_name}"
    match_winner = next((rnd.get("winning_team") for rnd in reversed(rounds) if rnd.get("winning_team")), None)
    if not match_winner:
        return {}

    return {
        "match_id": match_id,
        "map_name": map_name,
        "match_name": match_data.get("matchName"),
        "match_date": match_data.get("matchDate"),
        "tick_rate": tick_rate,
        "match_winner": match_winner,
        "rounds": rounds,
        "source_file": raw_path.name,
    }


def _parse_worker(path_str: str) -> Dict:
    path = Path(path_str)
    summary = parse_match(path)
    return {
        "raw_path": path_str,
        "summary": summary,
    }


def _find_raw_files() -> list[Path]:
    matches = [
        path
        for pattern in ("*.json", "*.json.xz")
        for path in RAW_DIR.glob(pattern)
        if path.is_file()
    ]
    if matches:
        return sorted(matches)
    return sorted(
        [
            path
            for pattern in ("*.json", "*.json.xz")
            for path in RAW_DIR.rglob(pattern)
            if path.is_file()
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse ESTA raw JSON into summaries.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of matches to parse (0 for no limit).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of matches to skip from the start of the sorted raw file list.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Number of parallel workers to use.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw_files = _find_raw_files()
    if not raw_files:
        raise SystemExit("No raw ESTA JSON files found in data/esta/raw/")
    if args.offset > 0:
        raw_files = raw_files[args.offset :]
    limit = args.limit if args.limit > 0 else None
    
    todo = raw_files
    if limit:
        todo = todo[:limit]

    parsed = 0
    skipped_missing = 0
    skipped_existing = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_parse_worker, str(path)) for path in todo]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing ESTA matches"):
            result = future.result()
            summary = result["summary"]
            raw_path = Path(result["raw_path"])
            if not summary:
                skipped_missing += 1
                continue
            dest_path = PARSED_DIR / f"{summary['match_id']}.json"
            if dest_path.exists():
                skipped_existing += 1
                continue
            with dest_path.open("w", encoding="utf-8") as fp:
                json.dump(summary, fp)
            parsed += 1

    print(f"Parsed {parsed} matches. Saved summaries to {PARSED_DIR.resolve()}")
    if limit and parsed >= limit:
        print(f"Stopped early because --limit={args.limit}. Use --limit 0 to parse everything.")
    if skipped_existing or skipped_missing:
        print(
            f"Skipped {skipped_existing} existing outputs and {skipped_missing} files without valid rounds/winners."
        )


if __name__ == "__main__":
    main()
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse ESTA raw JSON into summaries.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of matches to parse (0 for no limit).",
    )
    return parser.parse_args()
