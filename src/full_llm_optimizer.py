# src/full_llm_optimizer.py

from typing import Dict, List, Any, Tuple

import pandas as pd

from parser import parse_instance
from schedule import (
    build_double_round_robin,
    total_travel,
    check_double_round_robin,
    build_balanced_double_round_robin
)
from evaluator import (
    compute_travel_metrics,
    check_capacity_constraints,
    check_separation_constraints,
)
from full_llm_agent import FullScheduleLLMAgent

from schedule_diff import diff_schedules

# Type alias: schedule is list of rounds, each is list of (home_id, away_id)
Schedule = List[List[Tuple[int, int]]]


def evaluate_schedule_full(
    schedule: Schedule,
    teams: Dict[int, str],
    team_ids: List[int],
    D: pd.DataFrame,
    max_streak: int,
    max_sep: int,
) -> Dict[str, Any]:
    """
    Evaluate a schedule for the full-LLM experiment.

    Returns a dict with:
      - feasible_drr: bool
      - feasible: bool  (drr + CA3 + SE1)
      - raw_total_travel: float (even if infeasible)
      - total_travel: float (inf if infeasible)
      - ca3_violations: int
      - se1_violations: int
    """
    # DRR integrity
    feasible_drr = check_double_round_robin(schedule, team_ids)

    # Always compute raw travel (even if DRR broken) for history feedback.
    raw_total = float(total_travel(team_ids, schedule, D))

    # Constraint checks (only meaningful if DRR is roughly okay, but we can still count)
    ca3_list = check_capacity_constraints(schedule, teams, max_streak)
    se1_list = check_separation_constraints(schedule, teams, max_sep)
    print( "Max_streak:", max_streak)
    print( "Max Sep:", max_sep)
    ca3_count = len(ca3_list)
    se1_count = len(se1_list)

    feasible = feasible_drr and ca3_count == 0 and se1_count == 0
    final_total = raw_total if feasible else float("inf")

    return {
        "feasible_drr": feasible_drr,
        "feasible": feasible,
        "raw_total_travel": raw_total,
        "total_travel": final_total,
        "ca3_violations": ca3_count,
        "se1_violations": se1_count,
    }


def print_schedule_head(schedule, teams, num_rounds_to_show=None):
    """
    Pretty-print the full schedule.
    If num_rounds_to_show is None → show ALL rounds.
    Otherwise, show the first N rounds.
    """
    total_rounds = len(schedule)
    limit = total_rounds if num_rounds_to_show is None else min(num_rounds_to_show, total_rounds)

    print("\n=== FULL SCHEDULE ===")
    for r_idx in range(limit):
        rnd = schedule[r_idx]
        print(f"\nRound {r_idx}:")
        for (h, a) in rnd:
            print(f"  {teams[h]} vs {teams[a]}")
    print("\n======================")



def main():
    # --- 1. Load data instance ---
    xml_path = "daata/NL4.xml"   # adjust if your path is different
    teams, slots, D, max_streak, max_sep = parse_instance(xml_path)
    team_ids = sorted(teams.keys())
    num_rounds = 2 * (len(team_ids) - 1)

    print(f"Loaded  with {len(teams)} teams and {num_rounds} rounds expected.")
    print("Distance matrix shape:", D.shape)

    # --- 2. Baseline DRR schedule ---
    baseline: Schedule = build_balanced_double_round_robin(team_ids, max_streak=3)
    base_eval = evaluate_schedule_full(baseline, teams, team_ids, D, max_streak, max_sep)

    print("\n=== Baseline Double Round-Robin (from generator) ===")
    print(f"Baseline raw total travel: {base_eval['raw_total_travel']:.1f}")
    print(f"Baseline feasible_drr    : {base_eval['feasible_drr']}")
    print(f"Baseline CA3 violations  : {base_eval['ca3_violations']}")
    print(f"Baseline SE1 violations  : {base_eval['se1_violations']}")
    print(f"Baseline fully feasible  : {base_eval['feasible']}")
    print_schedule_head(baseline, teams)

    # --- 3. Full LLM agent setup ---
    agent = FullScheduleLLMAgent()
    history: List[Dict[str, Any]] = []

    best_schedule: Schedule = baseline
    best_metrics = base_eval

    iterations = 10  # start small; you can increase after confirming it works
    print(f"\n🤖 Starting full-LLM optimization for {iterations} iterations on dataset...\n")

    for it in range(1, iterations + 1):
        print(f"\n--- Iteration {it} ---")
        best_score_display = (
            "inf" if best_metrics["total_travel"] == float("inf")
            else f"{best_metrics['total_travel']:.1f}"
        )
        print(f"Current best (feasible={best_metrics['feasible']}): total_travel={best_score_display}")

        # Ask LLM for a new full schedule
        new_sched = agent.propose_schedule(
            current_best=best_schedule,
            teams=teams,
            D=D,
            num_rounds=num_rounds,
            history=history,
            best_total_travel=best_metrics["total_travel"],
            iteration=it,
        )

        if new_sched is None:
            print("[main] LLM proposal invalid; skipping this iteration.")
            # still record a "failed" iteration for history, if desired
            history.append({
                "iter": it,
                "raw_total_travel": float("inf"),
                "feasible": False,
                "ca3_violations": 0,
                "se1_violations": 0,
            })
            continue

        # Evaluate LLM-proposed schedule
        metrics = evaluate_schedule_full(new_sched, teams, team_ids, D)
        print(
            f"LLM schedule: raw_total={metrics['raw_total_travel']:.1f}, "
            f"feasible={metrics['feasible']}, "
            f"CA3={metrics['ca3_violations']}, SE1={metrics['se1_violations']}"
        )
        print("\n=== Differences from previous schedule ===")
        print(diff_schedules(best_schedule, new_sched, teams))

        # Add to history (this is what gets fed back into the prompt)
        history.append({
            "iter": it,
            "raw_total_travel": metrics["raw_total_travel"],
            "feasible": metrics["feasible"],
            "ca3_violations": metrics["ca3_violations"],
            "se1_violations": metrics["se1_violations"],
        })

        # Update best if improved and feasible
        if metrics["feasible"] and metrics["total_travel"] < best_metrics["total_travel"]:
            print(
                f"✅ New best feasible schedule found! "
                f"{best_metrics['total_travel']:.1f} → {metrics['total_travel']:.1f}"
            )
            best_schedule = new_sched
            best_metrics = metrics
        else:
            print("No improvement on best feasible schedule.")

    # --- 4. Final summary ---
    print("\n=== FINAL SUMMARY (Full LLM Mode on dataset) ===")

# Fix: safer display string for total_travel
    if best_metrics["total_travel"] == float("inf"):
        best_travel_display = "inf"
    else:
        best_travel_display = f"{best_metrics['total_travel']:.1f}"

    print(f"Best feasible: {best_metrics['feasible']} | total_travel={best_travel_display}")
    print(f"CA3 violations: {best_metrics['ca3_violations']}")
    print(f"SE1 violations: {best_metrics['se1_violations']}")

    print_schedule_head(best_schedule, teams)


if __name__ == "__main__":
    main()
