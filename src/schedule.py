# src/schedule.py
from typing import Dict, List, Tuple, Iterable, Union
import pandas as pd

TeamId = int
Game = Tuple[TeamId, TeamId]       # (home, away)
Round = List[Game]
Schedule = List[Round]

def round_robin_pairs(team_ids: List[int]) -> List[List[Tuple[int, int]]]:
    """
    Standard Berger algorithm. Produces n-1 rounds; each round has n/2 pairs.
    """
    teams = list(team_ids)
    n = len(teams)
    if n % 2 != 0:
        raise ValueError("Need even number of teams")

    # Fix the first team; rotate the others
    fixed = teams[0]
    others = teams[1:]

    rounds = []
    for _ in range(n - 1):
        pairs = []

        # First pairing involving the fixed team
        # Fixed team (fixed) plays against the last rotating team (others[-1])
        pairs.append((fixed, others[-1]))

        # Remaining pairings (mirroring around the center)
        for i in range(n // 2 - 1):
            pairs.append((others[i], others[-i - 2]))

        rounds.append(pairs)

        # Rotate others (one step clockwise)
        others = [others[-1]] + others[:-1]

    return rounds


def build_double_round_robin(team_ids: List[TeamId]) -> Schedule:
    first = round_robin_pairs(team_ids)
    second = [[(b, a) for (a, b) in rnd] for rnd in first]
    return first + second

def schedule_to_dataframe(schedule: Schedule, teams: Dict[TeamId, str]) -> pd.DataFrame:
    rows = []
    for slot, games in enumerate(schedule):
        for h, a in games:
            rows.append({"slot": slot, "home_id": h, "away_id": a,
                         "home": teams[h], "away": teams[a]})
    return pd.DataFrame(rows).sort_values(["slot", "home"]).reset_index(drop=True)

def _dist(D: Union[pd.DataFrame, Dict[Tuple[int,int], int]], i: int, j: int) -> int:
    if isinstance(D, dict):
        return int(D[(i, j)])
    return int(D.loc[i, j])

def team_travel(team_id: TeamId, schedule: Schedule,
                D: Union[pd.DataFrame, Dict[Tuple[int,int], int]]) -> int:
    seq = []
    for games in schedule:
        found = False
        for h, a in games:
            if h == team_id:
                seq.append(("H", a)); found = True; break
            if a == team_id:
                seq.append(("A", h)); found = True; break
        if not found:
            seq.append(("B", None))
    travel = 0
    at_home = True
    last_away = None
    for ha, opp in seq:
        if ha == "H":
            if not at_home and last_away is not None:
                travel += _dist(D, last_away, team_id)
                at_home, last_away = True, None
        elif ha == "A":
            if at_home:
                travel += _dist(D, team_id, opp)  # home -> opp
                at_home, last_away = False, opp
            else:
                travel += _dist(D, last_away, opp)  # opp_i -> opp_j
                last_away = opp
    if not at_home and last_away is not None:
        travel += _dist(D, last_away, team_id)
    return int(travel)

def total_travel(team_ids: Iterable[TeamId], schedule: Schedule,
                 D: Union[pd.DataFrame, Dict[Tuple[int,int], int]]) -> int:
    return sum(team_travel(t, schedule, D) for t in team_ids)


# --- DRR (Double Round-Robin) checks ---

from collections import defaultdict

def check_double_round_robin(
    schedule: Schedule,
    team_ids: List[TeamId],
    enforce_round_count: bool = True,
) -> bool:
    """
    Returns True iff:
      - Each round has each team exactly once (no team appears twice / no byes)
      - Each unordered pair appears exactly twice overall
      - The two games for a pair have opposite venues (home/away)
      - (Optional) number of rounds == 2*(n-1)
    """
    n = len(team_ids)
    team_set = set(team_ids)

    # Optional sanity: 2*(n-1) rounds for a double round-robin with even n
    if enforce_round_count and len(schedule) != 2 * (n - 1):
        return False

    # 1) Per-round integrity: each team plays exactly once
    for rnd in schedule:
        seen = set()
        for (h, a) in rnd:
            if h == a:
                return False  # cannot play itself
            if h in seen or a in seen:
                return False  # a team appears twice this round
            seen.add(h); seen.add(a)
        if seen != team_set:
            return False  # missing or extra teams in this round

    # 2) Pair integrity across whole season
    #    Each unordered pair should appear exactly twice, with opposite venues
    pair_map = defaultdict(list)  # key: (min, max) -> list of (home, away)
    for rnd in schedule:
        for (h, a) in rnd:
            key = (h, a) if h < a else (a, h)
            pair_map[key].append((h, a))

    # Expect exactly C(n,2) unordered pairs
    if len(pair_map) != n * (n - 1) // 2:
        return False

    for key, legs in pair_map.items():
        if len(legs) != 2:
            return False
        # Opposite venues: home teams must differ across the two legs
        if legs[0][0] == legs[1][0]:
            return False

    return True


def list_drr_issues(
    schedule: Schedule,
    team_ids: List[TeamId],
) -> list[str]:
    """
    Same checks as above but returns a list of human-readable issues (for debugging).
    """
    issues = []
    n = len(team_ids)
    team_set = set(team_ids)

    # Round integrity
    for r, rnd in enumerate(schedule):
        seen = set()
        for (h, a) in rnd:
            if h == a:
                issues.append(f"Round {r}: team {h} plays itself")
            if h in seen or a in seen:
                issues.append(f"Round {r}: a team appears twice")
            seen.add(h); seen.add(a)
        if seen != team_set:
            issues.append(f"Round {r}: team set mismatch")

    # Pair integrity
    pair_map = defaultdict(list)
    for r, rnd in enumerate(schedule):
        for (h, a) in rnd:
            key = (h, a) if h < a else (a, h)
            pair_map[key].append((h, a, r))

    expected_pairs = n * (n - 1) // 2
    if len(pair_map) != expected_pairs:
        issues.append(f"Pair count mismatch: got {len(pair_map)} vs {expected_pairs}")

    for key, legs in pair_map.items():
        if len(legs) != 2:
            issues.append(f"Pair {key} appears {len(legs)} times (expected 2)")
            continue
        (h1, a1, r1), (h2, a2, r2) = legs
        if h1 == h2:
            issues.append(f"Pair {key} has same home twice (rounds {r1} and {r2})")

    return issues


def build_balanced_double_round_robin(
    team_ids: List[TeamId],
    max_streak,
) -> Schedule:
    """
    Construct a double round-robin schedule that:
      - Always satisfies DRR (each pair twice, opposite venues).
      - For NL4 / NL10 automatically satisfies SE1 (gap = n-1, which is within [1, max_sep]).
      - Heuristically tries to keep home/away streaks <= max_streak for every team.

    It works in two phases:
      1) Build a first-round-robin with carefully chosen home/away assignments
         to keep streaks short.
      2) Add a mirrored second-round-robin (opposite venues) with a fixed offset
         so that the separation between the two games of any pair is exactly n-1.
    """
    n = len(team_ids)
    assert n % 2 == 0, "Need an even number of teams"
    rounds_single = round_robin_pairs(team_ids)  # pairings, orientation not yet final

    total_rounds = 2 * (n - 1)
    full_schedule: Schedule = [[] for _ in range(total_rounds)]

    # Track streak state for each team
    streak_type: Dict[TeamId, str] = {t: None for t in team_ids}  # 'H' or 'A'
    streak_len: Dict[TeamId, int] = {t: 0 for t in team_ids}

    # To record where each unordered pair was played in the first half
    # key: (min_id, max_id) -> (round_index, home, away)
    pair_map: Dict[Tuple[TeamId, TeamId], Tuple[int, TeamId, TeamId]] = {}

    def simulate_new_streak(t: TeamId, new_type: str) -> int:
        """Return streak length if team t plays new_type ('H' or 'A') this round."""
        prev_type = streak_type[t]
        prev_len = streak_len[t]
        if prev_type == new_type:
            return prev_len + 1
        return 1

    def apply_game(round_index: int, home: TeamId, away: TeamId) -> None:
        """Add game to schedule and update streaks."""
        full_schedule[round_index].append((home, away))
        for t, t_type in ((home, "H"), (away, "A")):
            prev_type = streak_type[t]
            if prev_type == t_type:
                streak_len[t] += 1
            else:
                streak_type[t] = t_type
                streak_len[t] = 1

    # 1) First half: choose H/A orientation greedily to keep streaks <= max_streak
    for r_idx, rnd in enumerate(rounds_single):
        for a, b in rnd:
            # Try both orientations for this pair
            candidates = [(a, b), (b, a)]  # (home, away)
            chosen = None

            # First, see if either orientation keeps both teams within max_streak
            for home, away in candidates:
                h_new = simulate_new_streak(home, "H")
                a_new = simulate_new_streak(away, "A")
                if h_new <= max_streak and a_new <= max_streak:
                    chosen = (home, away)
                    break

            # If neither orientation is “perfect”, pick the one that minimizes the worst streak
            if chosen is None:
                best_score = None
                best_pair = None
                for home, away in candidates:
                    h_new = simulate_new_streak(home, "H")
                    a_new = simulate_new_streak(away, "A")
                    score = max(h_new, a_new)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pair = (home, away)
                chosen = best_pair

            home, away = chosen
            apply_game(r_idx, home, away)

            key = (home, away) if home < away else (away, home)
            pair_map[key] = (r_idx, home, away)

    # 2) Second half: mirror each pair with opposite venue, offset by (n-1) rounds.
    #    This yields a separation gap of exactly (n - 1), which satisfies:
    #      - NL4: min=1, max=6  -> gap=3 (OK)
    #      - NL10: min=1, max=18 -> gap=9 (OK)
    offset = n - 1
    for (t_min, t_max), (r1, home1, away1) in pair_map.items():
        r2 = r1 + offset
        assert 0 <= r2 < total_rounds
        # Opposite venue in second leg
        home2, away2 = away1, home1
        apply_game(r2, home2, away2)

    # Sanity check: must be a valid double round robin
    assert check_double_round_robin(full_schedule, team_ids), "Balanced DRR builder broke DRR!"

    return full_schedule


