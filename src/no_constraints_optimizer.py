# src/heuristic_optimizer_no_constraints.py

from typing import Dict, List, Tuple, Any
import pandas as pd

from parser import parse_instance
from schedule import (
    build_balanced_double_round_robin,
    total_travel,
    check_double_round_robin,
    round_robin_pairs
)

from heuristic_agent import HeuristicLLMAgent
from move_executor import apply_move, transpose_home_away_for_team, rotate_team_labels, rotate_rounds
from schedule_diff import diff_schedules

from heuristic_optimizer import build_drr_with_streak_control


# canonical types
TeamId = int
Game = Tuple[TeamId, TeamId]
Round = List[Game]
Schedule = List[Round]


# =====================================================================
# Evaluation WITHOUT CA3 or SE1
# =====================================================================

def evaluate_schedule_no_constraints(
    schedule: Schedule,
    teams: Dict[int, str],
    team_ids: List[int],
    D: pd.DataFrame,
) -> Dict[str, Any]:

    # Only DRR matters. Ignore CA3 and SE1 completely.
    feasible_drr = check_double_round_robin(schedule, team_ids)

    raw_total = float(total_travel(team_ids, schedule, D))

    # If DRR is valid, we treat schedule as fully feasible.
    feasible = feasible_drr
    final_total = raw_total if feasible else float("inf")

    return {
        "feasible_drr": feasible_drr,
        "feasible": feasible,
        "raw_total_travel": raw_total,
        "total_travel": final_total,
        "ca3_violations": 0,   # Ignored
        "se1_violations": 0,   # Ignored
    }


def penalized_score(metrics: Dict[str, Any]) -> float:
    """
    Pure travel-based score.
    Lower is better.
    """
    if not metrics["feasible_drr"]:
        return float("inf")
    return metrics["raw_total_travel"]


def print_schedule(schedule: Schedule, teams: Dict[int, str]):
    print("\n=== FULL SCHEDULE ===")
    for r_idx, rnd in enumerate(schedule):
        print(f"\nRound {r_idx}:")
        for (h, a) in rnd:
            print(f"  {teams[h]} vs {teams[a]}")
    print("\n======================")


def apply_exhaustive_global_move(
    schedule, move_name, params, team_ids, D, teams
):
    """
    For global moves (rotations, transpose), try ALL valid parameters
    and return the best result.
    """

    # ---------------------------
    # rotate_rounds(k)
    # ---------------------------
    if move_name == "rotate_rounds":
        n = len(schedule)
        best_sched, best_metrics = None, None
        best_param = None

        for k in range(1, n):
            cand = rotate_rounds(schedule, k)
            cand_metrics = evaluate_schedule_no_constraints(cand, teams, team_ids, D)

            if (best_metrics is None) or cand_metrics["raw_total_travel"] < best_metrics["raw_total_travel"]:
                best_sched = cand
                best_metrics = cand_metrics
                best_param = {"k": k}

        return best_sched, best_metrics, best_param

    # ---------------------------
    # rotate_team_labels(k)
    # ---------------------------
    if move_name == "rotate_team_labels":
        n = len(team_ids)
        best_sched, best_metrics = None, None
        best_param = None

        for k in range(1, n):
            cand = rotate_team_labels(schedule, team_ids, k)
            cand_metrics = evaluate_schedule_no_constraints(cand, teams, team_ids, D)

            if (best_metrics is None) or cand_metrics["raw_total_travel"] < best_metrics["raw_total_travel"]:
                best_sched = cand
                best_metrics = cand_metrics
                best_param = {"k": k}

        return best_sched, best_metrics, best_param

    # ---------------------------
    # transpose_home_away_for_team(team_id)
    # ---------------------------
    if move_name == "transpose_home_away_for_team":
        best_sched, best_metrics = None, None
        best_param = None

        for tid in team_ids:
            cand = transpose_home_away_for_team(schedule, tid)
            cand_metrics = evaluate_schedule_no_constraints(cand, teams, team_ids, D)

            if (best_metrics is None) or cand_metrics["raw_total_travel"] < best_metrics["raw_total_travel"]:
                best_sched = cand
                best_metrics = cand_metrics
                best_param = {"team_id": tid}

        return best_sched, best_metrics, best_param

    # ---------------------------
    # DEFAULT: normal move
    # ---------------------------
    new_sched = apply_move(schedule, move_name, params)
    new_metrics = evaluate_schedule_no_constraints(new_sched, teams, team_ids, D)
    return new_sched, new_metrics, params

# =====================================================================
# MAIN
# =====================================================================

def main():
    # -----------------------------------
    # 1. Load instance
    # -----------------------------------
    xml_path = "daata/NL6.xml"
    teams, slots, D, max_streak, max_sep = parse_instance(xml_path)
    team_ids = sorted(teams.keys())
    num_rounds = 2 * (len(team_ids) - 1)

    print(f"\nLoaded instance with {len(teams)} teams, {num_rounds} rounds.")
    print("Distance matrix shape:", D.shape)

    # -----------------------------------
    # 2. Build baseline schedule
    # -----------------------------------
    schedule = build_drr_with_streak_control(team_ids, max_streak=max_streak)
    metrics = evaluate_schedule_no_constraints(schedule, teams, team_ids, D)

    print("\n=== Baseline Schedule ===")
    print(f"Raw travel: {metrics['raw_total_travel']:.1f}")
    print(f"Feasible DRR? {metrics['feasible_drr']}")
    print_schedule(schedule, teams)

    best_schedule = schedule
    best_metrics = metrics

    # -----------------------------------
    # 3. Initialize agent
    # -----------------------------------
    agent = HeuristicLLMAgent()
    print(f"Using LLM model: {agent.llm.model}")
    history = []

    iterations = 30
    print(f"\n🔍 Starting unconstrained LLM-guided optimization ({iterations} iterations)...\n")

    # -----------------------------------
    # 4. Iterative search
    # -----------------------------------
    for it in range(1, iterations + 1):
        print(f"\n--- Iteration {it} ---")
        print(f"Current best travel = {best_metrics['raw_total_travel']:.1f}")

        decision = agent.choose_move(
            schedule=best_schedule,
            teams=teams,
            metrics=best_metrics,
            history=history,
            iteration=it,
            team_ids=team_ids,
            D=D,
        )

        if decision is None:
            print("⚠️  LLM returned invalid decision; skipping.")
            history.append({"iter": it, "accepted": False, "reason": "invalid"})
            continue

        move_name = decision["move_name"]
        params = decision.get("params", {})
        rationale = decision.get("rationale", "")

        print(f"LLM suggests: {move_name} {params}")
        print(f"Reason: {rationale}")

        new_schedule, new_metrics, real_params = apply_exhaustive_global_move(
            best_schedule, move_name, params, team_ids, D, teams
        )

        print(f"→ New raw travel = {new_metrics['raw_total_travel']:.1f}")
        print(diff_schedules(best_schedule, new_schedule, teams))

        # pure travel comparison
        old_score = penalized_score(best_metrics)
        new_score = penalized_score(new_metrics)

        accepted = new_score < old_score

        if accepted:
            print("✅ Move accepted.")
            best_schedule = new_schedule
            best_metrics = new_metrics
        else:
            print("❌ Move rejected.")

        history.append({
            "iter": it,
            "move": move_name,
            "params": params,
            "accepted": accepted,
            "raw_total": new_metrics["raw_total_travel"],
            "rationale": rationale,
        })

    # -----------------------------------
    # 5. Final summary
    # -----------------------------------
    print("\n=== FINAL SUMMARY ===")
    print(f"Final travel = {best_metrics['raw_total_travel']:.1f}")
    print_schedule(best_schedule, teams)


if __name__ == "__main__":
    main()
