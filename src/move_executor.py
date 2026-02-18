# src/move_executor.py

from typing import List, Tuple, Dict, Any
from schedule import TeamId, Game, Round, Schedule
import copy


# =====================================================================
# A. SIMPLE LOCAL STRUCTURAL MOVES
# =====================================================================

def swap_games_between_rounds(schedule: Schedule,
                              round_a: int, game_idx_a: int,
                              round_b: int, game_idx_b: int) -> Schedule:
    new_sched = copy.deepcopy(schedule)

    if round_a < 0 or round_b < 0 or round_a >= len(new_sched) or round_b >= len(new_sched):
        return new_sched

    rndA = new_sched[round_a]
    rndB = new_sched[round_b]

    if game_idx_a < 0 or game_idx_b < 0 or game_idx_a >= len(rndA) or game_idx_b >= len(rndB):
        return new_sched

    rndA[game_idx_a], rndB[game_idx_b] = rndB[game_idx_b], rndA[game_idx_a]
    return new_sched


def flip_home_away(schedule: Schedule, round_idx: int, game_idx: int) -> Schedule:
    new_sched = copy.deepcopy(schedule)

    if not (0 <= round_idx < len(new_sched)):
        return new_sched

    rnd = new_sched[round_idx]
    if not (0 <= game_idx < len(rnd)):
        return new_sched

    h, a = rnd[game_idx]
    rnd[game_idx] = (a, h)

    # Flip the corresponding other leg
    for r in range(len(new_sched)):
        if r == round_idx:
            continue
        for idx, (hh, aa) in enumerate(new_sched[r]):
            if (hh == a and aa == h) or (hh == h and aa == a):
                new_sched[r][idx] = (aa, hh)
                return new_sched

    return new_sched


def rotate_rounds(schedule: Schedule, k: int) -> Schedule:
    new_sched = copy.deepcopy(schedule)
    n = len(new_sched)
    if n == 0:
        return new_sched

    k = k % n
    return new_sched[-k:] + new_sched[:-k]


# =====================================================================
# NEW MOVE 1 — rotate_team_labels
# =====================================================================

def rotate_team_labels(schedule: Schedule, team_ids: List[int], k: int) -> Schedule:
    """
    Reassign team IDs by rotating them by k positions.

    IMPORTANT:
    - Must accept (schedule, team_ids, k) so apply_move(schedule, **params) works.
    """
    new_sched = copy.deepcopy(schedule)
    n = len(team_ids)

    # Build rotation mapping
    rotation_map = {}
    for idx, old in enumerate(team_ids):
        rotation_map[old] = team_ids[(idx + k) % n]

    # Apply mapping to every (h, a)
    for r_idx, rnd in enumerate(new_sched):
        for g_idx, (h, a) in enumerate(rnd):
            new_sched[r_idx][g_idx] = (
                rotation_map[h],
                rotation_map[a]
            )

    return new_sched


# =====================================================================
# NEW MOVE 2 — transpose_home_away_for_team
# =====================================================================

def transpose_home_away_for_team(schedule: Schedule, team_id: int) -> Schedule:
    """
    Flip home/away ONLY for games involving the given team.
    """
    new_sched = copy.deepcopy(schedule)

    for r_idx, rnd in enumerate(new_sched):
        for g_idx, (h, a) in enumerate(rnd):

            # Team played at home → flip
            if h == team_id:
                new_sched[r_idx][g_idx] = (a, h)

            # Team played away → flip
            elif a == team_id:
                new_sched[r_idx][g_idx] = (a, h)

    return new_sched


# =====================================================================
# MOVE REGISTRY (updated)
# =====================================================================

MOVE_REGISTRY: Dict[str, Any] = {
    "swap_games_between_rounds": swap_games_between_rounds,
    "flip_home_away": flip_home_away,
    "rotate_rounds": rotate_rounds,
    "rotate_team_labels": rotate_team_labels,                 # NEW
    "transpose_home_away_for_team": transpose_home_away_for_team,  # NEW
}


def apply_move(schedule: Schedule, move_name: str, params: Dict[str, Any]) -> Schedule:
    fn = MOVE_REGISTRY.get(move_name)
    if fn is None:
        print(f"[move_executor] ERROR: Unknown move {move_name}")
        return schedule

    try:
        return fn(schedule, **params)
    except Exception as e:
        print(f"[move_executor] ERROR executing move {move_name}: {e}")
        return schedule
