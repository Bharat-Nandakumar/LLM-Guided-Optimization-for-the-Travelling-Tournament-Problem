# src/evaluator.py

import pandas as pd
import itertools
from schedule import total_travel

# ---------------- TRAVEL METRICS ---------------- #

def compute_travel_metrics(teams, team_ids, schedule, D):
    """
    Compute per-team and total travel distances.
    """
    results = []
    for tid in team_ids:
        travel = total_travel([tid], schedule, D)
        results.append({
            "team": teams[tid],
            "travel_distance": travel
        })

    df = pd.DataFrame(results)
    total = df["travel_distance"].sum()
    avg = df["travel_distance"].mean()
    var = df["travel_distance"].var()

    summary = {
        "total_travel": total,
        "average_travel": avg,
        "variance_travel": var
    }

    return df, summary


# ---------------- CONSTRAINT CHECKS ---------------- #

def check_capacity_constraints(schedule, teams, max_streak):
    """
    Implements CA3: No more than 3 consecutive home or away games per team.
    Returns list of teams violating the rule.
    """
    violations = []

    for tid in teams.keys():
        streak_home, streak_away = 0, 0
        max_home, max_away = 0, 0

        for rnd in schedule:
            found = False
            for home, away in rnd:
                if home == tid:
                    streak_home += 1
                    streak_away = 0
                    found = True
                    break
                elif away == tid:
                    streak_away += 1
                    streak_home = 0
                    found = True
                    break

            if not found:
                # bye week
                streak_home = 0
                streak_away = 0

            max_home = max(max_home, streak_home)
            max_away = max(max_away, streak_away)

        if max_home > max_streak or max_away > max_streak:
            violations.append(teams[tid])

    return violations


def check_separation_constraints(schedule, teams, min_sep, max_sep):
    """
    Implements SE1: For each pair of teams, their two games (home/away)
    must be separated by between min_sep and max_sep rounds.
    """
    violations = []

    team_ids = list(teams.keys())

    for t1, t2 in itertools.combinations(team_ids, 2):
        rounds = []
        for r, rnd in enumerate(schedule):
            for h, a in rnd:
                if (h == t1 and a == t2) or (h == t2 and a == t1):
                    rounds.append(r)

        if len(rounds) != 2:
            continue  # should always be 2 in a valid double round robin

        gap = rounds[1] - rounds[0]
        if gap < min_sep or gap > max_sep:
            violations.append((teams[t1], teams[t2], gap))

    return violations


# ---------------- MAIN EVALUATOR ---------------- #

def evaluate_schedule(teams, team_ids, schedule, D, save_path="results/NL4_evaluation.txt"):
    travel_df, travel_summary = compute_travel_metrics(teams, team_ids, schedule, D)

    cap_violations = check_capacity_constraints(schedule, teams)
    sep_violations = check_separation_constraints(schedule, teams)

    feasible = len(cap_violations) == 0 and len(sep_violations) == 0
    total_travel = travel_summary["total_travel"]

    if not feasible:
        total_travel = float("inf")

    with open(save_path, "w") as f:
        f.write("===== TRAVEL SUMMARY =====\n")
        f.write(travel_df.to_string(index=False))
        f.write("\n\nTOTAL TRAVEL: {:.0f}".format(travel_summary["total_travel"]))
        f.write("\nAVERAGE TRAVEL: {:.1f}".format(travel_summary["average_travel"]))
        f.write("\nVARIANCE: {:.1f}".format(travel_summary["variance_travel"]))

        f.write("\n\n===== CONSTRAINT CHECKS =====\n")
        f.write(f"Capacity Violations (CA3): {len(cap_violations)}\n")
        for t in cap_violations:
            f.write(f"  - {t}\n")

        f.write(f"Separation Violations (SE1): {len(sep_violations)}\n")
        for t1, t2, gap in sep_violations:
            f.write(f"  - {t1} vs {t2} gap={gap}\n")

        f.write(f"\nFeasible: {feasible}\n")

    print("✅ Evaluation complete. Results saved to", save_path)
    return total_travel
