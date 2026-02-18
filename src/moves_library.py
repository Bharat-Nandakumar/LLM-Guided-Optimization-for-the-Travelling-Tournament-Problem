# src/moves_library.py

"""
This module defines the MOVE LIBRARY for the LLM-guided TTP optimizer.

Each move entry describes:
    - name:      Move identifier (used by LLM)
    - category:  High-level grouping
    - params:    Parameters required for the move
    - desc:      What the move does conceptually

The actual implementation of each move will live in move_executor.py.
"""

MOVE_LIBRARY = [
    # ---------------------------------------------------------
    # BASIC LOCAL MOVES
    # ---------------------------------------------------------
    {
        "name": "swap_rounds",
        "category": "basic",
        "params": ["round_a", "round_b"],
        "desc": "Swap the entire set of games in round A with round B."
    },
    {
        "name": "swap_games_within_round",
        "category": "basic",
        "params": ["round", "game_i", "game_j"],
        "desc": "Swap two games inside the same round."
    },
    {
        "name": "swap_games_across_rounds",
        "category": "basic",
        "params": ["round_a", "game_i", "round_b", "game_j"],
        "desc": "Swap a single game from round A with a game from round B."
    },
    {
        "name": "flip_home_away",
        "category": "basic",
        "params": ["round", "game"],
        "desc": "Flip home/away teams for one game."
    },
    {
        "name": "shift_game",
        "category": "basic",
        "params": ["round_from", "game", "round_to"],
        "desc": "Move a game from one round to another."
    },
    {
        "name": "rotate_round_trip",
        "category": "basic",
        "params": ["team_id", "start_round", "end_round"],
        "desc": "Rotate the sequence of games for one team within a range of rounds."
    },

    # ---------------------------------------------------------
    # TEAM-FOCUSED BALANCING MOVES
    # ---------------------------------------------------------
    {
        "name": "reduce_home_streak",
        "category": "team_balance",
        "params": ["team_id"],
        "desc": "Break a long home-game streak for a team by swapping or shifting games."
    },
    {
        "name": "reduce_away_streak",
        "category": "team_balance",
        "params": ["team_id"],
        "desc": "Break a long away-game streak for a team."
    },
    {
        "name": "improve_team_trip_cost",
        "category": "team_balance",
        "params": ["team_id"],
        "desc": "Try to reduce total travel for a team by adjusting trip ordering."
    },
    {
        "name": "fix_team_conflicts",
        "category": "team_balance",
        "params": ["team_id"],
        "desc": "Repair DRR or separation issues for one team."
    },

    # ---------------------------------------------------------
    # MATCH-FOCUSED (PAIRWISE)
    # ---------------------------------------------------------
    {
        "name": "adjust_separation_constraint",
        "category": "pairwise",
        "params": ["team_a", "team_b"],
        "desc": "Ensure (A,B) games are separated by required SE1 distance."
    },
    {
        "name": "adjust_home_away_balance_for_pair",
        "category": "pairwise",
        "params": ["team_a", "team_b"],
        "desc": "Ensure the two games between A & B are properly spaced and ordered."
    },

    # ---------------------------------------------------------
    # STRUCTURAL ROUND TRANSFORMATIONS
    # ---------------------------------------------------------
    {
        "name": "reverse_round_subsequence",
        "category": "structural",
        "params": ["start_round", "end_round"],
        "desc": "Reverse the order of rounds in a specified range."
    },
    {
        "name": "cyclic_shift_rounds",
        "category": "structural",
        "params": ["start_round", "end_round", "shift"],
        "desc": "Cyclically shift a block of rounds forward or backward."
    },
    {
        "name": "shuffle_games_in_round",
        "category": "structural",
        "params": ["round"],
        "desc": "Randomly reshuffle games within a round while preserving home/away."
    },

    # ---------------------------------------------------------
    # LARGE NEIGHBORHOOD SEARCH (LNS)
    # ---------------------------------------------------------
    {
        "name": "destroy_and_repair_team",
        "category": "lns",
        "params": ["team_id"],
        "desc": "Remove all games of a team and rebuild them using constraints."
    },
    {
        "name": "destroy_and_repair_round_block",
        "category": "lns",
        "params": ["start_round", "end_round"],
        "desc": "Remove all games in a round block and regenerate them."
    },
    {
        "name": "destroy_and_repair_conflict_zone",
        "category": "lns",
        "params": ["start_round", "end_round"],
        "desc": "Target a region containing CA3/SE1 conflicts and rebuild it."
    },
    {
        "name": "fix_multiple_long_streaks",
        "category": "lns",
        "params": [],
        "desc": "Identify all teams with long streaks and repair them simultaneously."
    },
    {
        "name": "neighborhood_rebuild",
        "category": "lns",
        "params": ["teams", "start_round", "end_round"],
        "desc": "Rebuild a sub-schedule for a selected set of teams and rounds."
    },

    # ---------------------------------------------------------
    # TRAVEL OPTIMIZATION MOVES
    # ---------------------------------------------------------
    {
        "name": "swap_adjacent_trips",
        "category": "travel",
        "params": ["team_id"],
        "desc": "Swap order of adjacent away trips for travel-cost improvement."
    },
    {
        "name": "merge_trips",
        "category": "travel",
        "params": ["team_id"],
        "desc": "Combine two separate away trips into one if possible."
    },
    {
        "name": "rearrange_trip_order",
        "category": "travel",
        "params": ["team_id"],
        "desc": "Change ordering of away games to reduce travel distance."
    },
    {
        "name": "compress_trip",
        "category": "travel",
        "params": ["team_id"],
        "desc": "Bring separated away games closer together to reduce travel."
    },

    # ---------------------------------------------------------
    # HEURISTIC-LEVEL GUIDANCE MOVES
    # ---------------------------------------------------------
    {
        "name": "heuristic_relocate_match",
        "category": "heuristic",
        "params": ["team_a", "team_b"],
        "desc": "LLM proposes relocating one of the meetings between A and B."
    },
    {
        "name": "heuristic_improve_team_cost",
        "category": "heuristic",
        "params": ["team_id"],
        "desc": "LLM suggests a high-level strategy to reduce travel for a specific team."
    },
    {
        "name": "heuristic_balance_global_home_away",
        "category": "heuristic",
        "params": [],
        "desc": "LLM suggests global home/away balancing logic."
    },

    # ---------------------------------------------------------
    # BLOCK TRANSFORMATIONS
    # ---------------------------------------------------------
    {
        "name": "mirror_schedule",
        "category": "structural_block",
        "params": [],
        "desc": "Flip all home/away assignments in entire schedule or half season."
    },
    {
        "name": "regenerate_second_half_from_first_half",
        "category": "structural_block",
        "params": [],
        "desc": "Rebuild rounds N..2N-1 from the structure of rounds 0..N-1."
    },
    {
        "name": "symmetric_pair_adjustment",
        "category": "structural_block",
        "params": ["team_a", "team_b"],
        "desc": "Enforce symmetric placement of the two games for team pair A,B."
    },

    # ---------------------------------------------------------
    # CONSTRAINT REPAIR MOVES
    # ---------------------------------------------------------
    {
        "name": "CA3_repair",
        "category": "repair",
        "params": ["team_id"],
        "desc": "Fix capacity violations (>3 home or away)."
    },
    {
        "name": "SE1_fix_pair",
        "category": "repair",
        "params": ["team_a", "team_b"],
        "desc": "Fix SE1 violations for pair A,B."
    },
    {
        "name": "DRR_repair",
        "category": "repair",
        "params": [],
        "desc": "Repair missing/extra home or away occurrences to restore DRR validity."
    },

    # ---------------------------------------------------------
    # META MOVES (SEQUENCES)
    # ---------------------------------------------------------
    {
        "name": "multi_move",
        "category": "meta",
        "params": ["moves"],
        "desc": "Execute multiple moves in a sequence."
    },
    {
        "name": "conditional_move",
        "category": "meta",
        "params": ["move_if_true", "move_if_false", "condition"],
        "desc": "Perform move A if a condition holds, else move B."
    },
]

def list_move_names():
    return [m["name"] for m in MOVE_LIBRARY]

def get_move_spec(name):
    for m in MOVE_LIBRARY:
        if m["name"] == name:
            return m
    return None
