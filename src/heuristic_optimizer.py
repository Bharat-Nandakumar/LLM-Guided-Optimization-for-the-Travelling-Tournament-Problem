# src/heuristic_optimizer.py

from typing import Dict, List, Tuple, Any
import pandas as pd

from parser import parse_instance
from schedule import (
    build_balanced_double_round_robin,
    total_travel,
    check_double_round_robin,
    round_robin_pairs
)
from evaluator import (
    compute_travel_metrics,
    check_capacity_constraints,
    check_separation_constraints,
)

from heuristic_agent import HeuristicLLMAgent
from move_executor import apply_move

from schedule_diff import diff_schedules


# canonical types
TeamId = int
Game = Tuple[TeamId, TeamId]
Round = List[Game]
Schedule = List[Round]

def apply_exhaustive_global_move(schedule, move_name, params, team_ids, D, teams, max_streak, max_sep):
    """
    For moves with many valid parameter choices, evaluate ALL options
    and return the best schedule + metrics.
    """

    # Default: just apply the move normally
    from move_executor import apply_move, rotate_rounds, rotate_team_labels, transpose_home_away_for_team

    # ============================
    # Case 1 — rotate_rounds(k)
    # ============================
    if move_name == "rotate_rounds":
        best_sched = None
        best_metrics = None

        n = len(schedule)

        for k in range(1, n):  # skip k = 0
            candidate_sched = rotate_rounds(schedule, k)
            candidate_metrics = evaluate_schedule(candidate_sched, teams, team_ids, D, max_streak, max_sep)

            if (best_metrics is None) or (candidate_metrics["raw_total_travel"] < best_metrics["raw_total_travel"]):
                best_sched = candidate_sched
                best_metrics = candidate_metrics

        return best_sched, best_metrics, {"k": k}

    # ============================
    # Case 2 — rotate_team_labels(k)
    # ============================
    if move_name == "rotate_team_labels":
        best_sched = None
        best_metrics = None

        n = len(team_ids)

        for k in range(1, n):  # skip k = 0
            candidate_sched = rotate_team_labels(schedule, team_ids, k)
            candidate_metrics = evaluate_schedule(candidate_sched, teams, team_ids, D, max_streak, max_sep)

            if (best_metrics is None) or (candidate_metrics["raw_total_travel"] < best_metrics["raw_total_travel"]):
                best_sched = candidate_sched
                best_metrics = candidate_metrics

        return best_sched, best_metrics, {"k": k}

    # ============================
    # Case 3 — transpose_home_away_for_team(team_id)
    # ============================
    if move_name == "transpose_home_away_for_team":
        best_sched = None
        best_metrics = None

        for tid in team_ids:
            candidate_sched = transpose_home_away_for_team(schedule, tid)
            candidate_metrics = evaluate_schedule(candidate_sched, teams, team_ids, D, max_streak, max_sep)

            if (best_metrics is None) or (candidate_metrics["raw_total_travel"] < best_metrics["raw_total_travel"]):
                best_sched = candidate_sched
                best_metrics = candidate_metrics

        return best_sched, best_metrics, {"team_id": tid}

    # ============================
    # Fallback for normal moves
    # ============================
    new_sched = apply_move(schedule, move_name, params)
    new_metrics = evaluate_schedule(new_sched, teams, team_ids, D, max_streak, max_sep)
    return new_sched, new_metrics, params

def repair_streak_violations(schedule, team_ids, max_streak):
    """
    Try to fix H/A streak violations by flipping home/away orientation
    for second-half games, because flipping preserves DRR validity.
    """

    def streaks_ok(games):
        streak_type = {t: None for t in team_ids}
        streak_len  = {t: 0 for t in team_ids}

        for rnd in games:
            for home, away in rnd:
                # update home team
                if streak_type[home] == "H":
                    streak_len[home] += 1
                else:
                    streak_type[home] = "H"
                    streak_len[home] = 1

                if streak_len[home] > max_streak:
                    return False

                # update away team
                if streak_type[away] == "A":
                    streak_len[away] += 1
                else:
                    streak_type[away] = "A"
                    streak_len[away] = 1

                if streak_len[away] > max_streak:
                    return False

        return True

    # first, try naive flipping of second-half games
    n = len(team_ids)
    offset = n - 1

    for r in range(offset, len(schedule)):
        new_round = []
        for home, away in schedule[r]:
            # flip
            new_round.append((away, home))

        old = schedule[r]
        schedule[r] = new_round
        if streaks_ok(schedule):
            return schedule  # fixed

        # undo flip if not good
        schedule[r] = old

    # fallback: return original (may still violate)
    return schedule


def build_drr_with_streak_control(team_ids: List[int], max_streak: int = 3) -> List[List[Tuple[int, int]] | None]:
    """
    Builds a DRR schedule using backtracking to satisfy the separation (n-1) 
    and max_streak constraints.
    
    Returns: A valid DRR schedule, or None if no solution is found.
    """
    n = len(team_ids)
    if n % 2 != 0:
        raise ValueError("Number of teams must be even.")
        
    total_rounds = 2 * (n - 1)
    rounds_single = round_robin_pairs(team_ids)
    
    # Flatten the games for sequential assignment
    games_flat = [game for rnd in rounds_single for game in rnd]
    
    # Track the home/away assignment for each team for easy checking.
    # team_venues[team][round_idx] = 'H', 'A', or None
    # This is the primary state we mutate and check.
    team_venues: Dict[int, List[str | None]] = {
        t: [None] * total_rounds for t in team_ids
    }
    
    def check_streak(team: int, round_idx: int, venue: str) -> bool:
        """
        Check if assigning 'venue' to 'team' at 'round_idx' would 
        create a streak violation (> max_streak).
        """
        current_streak = 0
        
        # Check backwards from the previous round
        for i in range(round_idx - 1, -1, -1):
            if team_venues[team][i] == venue:
                current_streak += 1
            elif team_venues[team][i] is not None:
                # Streak broken by an opposite venue
                break
            # If team_venues[team][i] is None, this is a gap, and the streak is broken
            
        return (current_streak + 1) <= max_streak

    def backtrack(game_idx: int) -> bool:
        """
        Tries to assign venues to the game at games_flat[game_idx].
        """
        if game_idx == len(games_flat):
            # Base case: All games from the first half assigned successfully.
            return True

        t1, t2 = games_flat[game_idx]
        r1 = game_idx // (n // 2)

        # ----------------------------------------
        # Candidate 1: t1=Home, t2=Away in r1
        # ----------------------------------------
        h1, a1 = t1, t2
        r2 = r1 + n - 1  # Fixed second-half round
        h2, a2 = a1, h1   # Venues in r2 must be reversed

        # 1. Check streak validity for r1 (t1=H, t2=A)
        valid_r1_h = check_streak(h1, r1, "H")
        valid_r1_a = check_streak(a1, r1, "A")
        
        # 2. Check streak validity for r2 (t2=H, t1=A)
        valid_r2_h = check_streak(h2, r2, "H")
        valid_r2_a = check_streak(a2, r2, "A")

        if valid_r1_h and valid_r1_a and valid_r2_h and valid_r2_a:
            # ASSIGN STATE (r1 and r2 games)
            team_venues[h1][r1] = "H"
            team_venues[a1][r1] = "A"
            team_venues[h2][r2] = "H"
            team_venues[a2][r2] = "A"
            
            # RECURSE
            if backtrack(game_idx + 1):
                return True
            
            # UNASSIGN STATE (Backtrack)
            team_venues[h1][r1] = None
            team_venues[a1][r1] = None
            team_venues[h2][r2] = None
            team_venues[a2][r2] = None

        # ----------------------------------------
        # Candidate 2: t1=Away, t2=Home in r1
        # ----------------------------------------
        h1, a1 = t2, t1
        r2 = r1 + n - 1
        h2, a2 = a1, h1

        # 1. Check streak validity for r1 (t2=H, t1=A)
        valid_r1_h = check_streak(h1, r1, "H")
        valid_r1_a = check_streak(a1, r1, "A")
        
        # 2. Check streak validity for r2 (t1=H, t2=A)
        valid_r2_h = check_streak(h2, r2, "H")
        valid_r2_a = check_streak(a2, r2, "A")

        if valid_r1_h and valid_r1_a and valid_r2_h and valid_r2_a:
            # ASSIGN STATE
            team_venues[h1][r1] = "H"
            team_venues[a1][r1] = "A"
            team_venues[h2][r2] = "H"
            team_venues[a2][r2] = "A"
            
            # RECURSE
            if backtrack(game_idx + 1):
                return True
            
            # UNASSIGN STATE (Backtrack)
            team_venues[h1][r1] = None
            team_venues[a1][r1] = None
            team_venues[h2][r2] = None
            team_venues[a2][r2] = None

        # If neither candidate worked, return False and backtrack one level up
        return False
        
    # Start the backtracking process
    if not backtrack(0):
        print(f"No solution found for N={n} with max_streak={max_streak}.")
        return None 

    # --- Reconstruct Final Schedule ---
    final_schedule: List[List[Tuple[int, int]]] = [[] for _ in range(total_rounds)]
    
    # First half (Rounds 0 to n-2)
    for r1 in range(n - 1):
        for t1, t2 in rounds_single[r1]:
            # Use the determined venue from team_venues
            if team_venues[t1][r1] == "H":
                home, away = t1, t2
            else:
                home, away = t2, t1
            final_schedule[r1].append((home, away))
            
    # Second half (Rounds n-1 to 2n-3)
    offset = n - 1
    for r1 in range(n - 1):
        r2 = r1 + offset
        for h1, a1 in final_schedule[r1]:
            # Mirror the game
            final_schedule[r2].append((a1, h1))
            
    # Final validation check
    assert check_double_round_robin(final_schedule, team_ids), "Internal Schedule Error!"
    
    return final_schedule


# =====================================================================
# Evaluation wrapper
# =====================================================================

def evaluate_schedule(
    schedule: Schedule,
    teams: Dict[int, str],
    team_ids: List[int],
    D: pd.DataFrame,
    max_streak: int,
    max_sep: int,
) -> Dict[str, Any]:

    feasible_drr = check_double_round_robin(schedule, team_ids)
    raw_total = float(total_travel(team_ids, schedule, D))

    ca_list = check_capacity_constraints(schedule, teams, max_streak)
    se_list = check_separation_constraints(schedule, teams , 1, max_sep)

    ca_v = len(ca_list)
    se_v = len(se_list)

    feasible = feasible_drr and ca_v == 0 and se_v == 0
    final_total = raw_total if feasible else float("inf")

    return {
        "feasible_drr": feasible_drr,
        "feasible": feasible,
        "raw_total_travel": raw_total,
        "total_travel": final_total,
        "ca3_violations": ca_v,
        "se1_violations": se_v,
    }


def print_schedule(schedule: Schedule, teams: Dict[int, str]):
    print("\n=== FULL SCHEDULE ===")
    for r_idx, rnd in enumerate(schedule):
        print(f"\nRound {r_idx}:")
        for (h, a) in rnd:
            print(f"  {teams[h]} vs {teams[a]}")
    print("\n======================")

def penalized_score(metrics: Dict[str, Any]) -> float:
    """
    Convert feasibility + violations + travel into a single score.
    Lower is better.
    """
    base = metrics["raw_total_travel"]

    # Penalties
    PENALTY_DRR = 1_000_000
    PENALTY_CA3 = 100_000
    PENALTY_SE1 = 100_000

    score = base

    if not metrics["feasible_drr"]:
        score += PENALTY_DRR

    score += metrics["ca3_violations"] * PENALTY_CA3
    score += metrics["se1_violations"] * PENALTY_SE1

    return score


# =====================================================================
# Main heuristic optimizer
# =====================================================================

def main():
    # -----------------------------
    # 1. Load instance
    # -----------------------------
    xml_path = "daata/NL4.xml"  # modify if needed
    teams, slots, D ,max_streak, max_sep = parse_instance(xml_path)
    team_ids = sorted(teams.keys())
    num_rounds = 2 * (len(team_ids) - 1)

    print(f"\nLoaded instance with {len(teams)} teams, {num_rounds} rounds.")
    print("Distance matrix shape:", D.shape)
    print(f"Max streak: {max_streak}")
    print(f"Max separation: {max_sep}")

    # -----------------------------
    # 2. Build baseline schedule
    # -----------------------------
    schedule = build_drr_with_streak_control(team_ids, max_streak=max_streak)
    metrics = evaluate_schedule(schedule, teams, team_ids, D, max_streak, max_sep)

    print("\n=== Baseline Schedule ===")
    print(f"Raw travel: {metrics['raw_total_travel']:.1f}")
    print(f"Feasible (DRR+CA3+SE1)? {metrics['feasible']}")
    print(f"CA3 violations: {metrics['ca3_violations']}")
    print(f"SE1 violations: {metrics['se1_violations']}")
    print_schedule(schedule, teams)

    # tracked best
    best_schedule = schedule
    best_metrics = metrics

    # -----------------------------
    # 3. Initialize heuristic agent
    # -----------------------------
    agent = HeuristicLLMAgent()
    history: List[Dict[str, Any]] = []

    iterations = 30
    print(f"\n🔍 Starting LLM-guided heuristic search for {iterations} iterations...\n")

    # -----------------------------
    # 4. Iterative improvement loop
    # -----------------------------
    for it in range(1, iterations + 1):
        print(f"\n--- Iteration {it} ---")

        best_score_disp = (
            "inf" if best_metrics["total_travel"] == float("inf")
            else f"{best_metrics['total_travel']:.1f}"
        )
        print(f"Current best feasible = {best_metrics['feasible']}, travel = {best_score_disp}")

        # Get LLM's move choice
        decision = agent.choose_move(
            schedule=best_schedule,
            teams=teams,
            metrics=best_metrics,    # FIXED
            history=history,
            iteration=it,
            team_ids=team_ids,
            D=D,
        )

        if decision is None:
            print("⚠️  LLM returned invalid decision; skipping iteration.")
            history.append({
                "iter": it,
                "accepted": False,
                "reason": "invalid decision",
            })
            continue

        move_name = decision["move_name"]
        params = decision.get("params", {})
        rationale = decision.get("rationale", "")

        print(f"LLM suggests: {move_name} {params}")
        print(f"Reason: {rationale}")

        # Execute move
        new_schedule, new_metrics, real_params = apply_exhaustive_global_move(
        best_schedule, move_name, params, team_ids, D, teams, max_streak, max_sep
        )


        print(f"→ New schedule raw travel = {new_metrics['raw_total_travel']:.1f}")
        print(f"→ feasible={new_metrics['feasible']}, CA3={new_metrics['ca3_violations']}, SE1={new_metrics['se1_violations']}")

        print("\n=== Differences from previous schedule ===")
        print(diff_schedules(best_schedule, new_schedule, teams))

        # ---------------------------
        # Acceptance rule (improved)
        # ---------------------------
        accepted = False

        old_score = penalized_score(best_metrics)
        new_score = penalized_score(new_metrics)

        accepted = new_score < old_score

        if accepted:
            print("✅ Move accepted. (score improved)")
            best_schedule = new_schedule
            best_metrics = new_metrics
        else:
            print("❌ Move rejected. (score worse)")


        # track history
        history.append({
            "iter": it,
            "move": move_name,
            "params": params,
            "accepted": accepted,
            "raw_total": new_metrics["raw_total_travel"],
            "feasible": new_metrics["feasible"],
            "ca3": new_metrics["ca3_violations"],
            "se1": new_metrics["se1_violations"],
            "rationale": rationale,
        })

    # -----------------------------
    # 5. Final summary
    # -----------------------------
    print("\n=== FINAL SUMMARY ===")
    best_score = (
        "inf" if best_metrics["total_travel"] == float("inf")
        else f"{best_metrics['total_travel']:.1f}"
    )
    print(f"Best feasible schedule: {best_metrics['feasible']}")
    print(f"Total travel: {best_score}")
    print(f"CA3 violations: {best_metrics['ca3_violations']}")
    print(f"SE1 violations: {best_metrics['se1_violations']}")
    print_schedule(best_schedule, teams)


if __name__ == "__main__":
    main()
