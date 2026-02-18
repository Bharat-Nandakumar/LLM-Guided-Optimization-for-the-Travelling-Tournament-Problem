import copy
import random
from typing import Dict, List, Tuple
from llm_agent import LLMHeuristicAgent
import json
import os

import pandas as pd

from parser import parse_instance
from schedule import (  
    build_double_round_robin,
    schedule_to_dataframe,
    check_double_round_robin
)
from evaluator import (
    evaluate_schedule,
    compute_travel_metrics,
)

# ---------------------------------------
# Helpers: DataFrame <-> rounds conversion
# ---------------------------------------

def df_to_rounds(df: pd.DataFrame, teams: Dict[int, str]) -> List[List[Tuple[int, int]]]:
    """
    Convert schedule DataFrame (slot, home, away with names) into
    a list of rounds, each a list of (home_id, away_id).
    """
    # name -> id map
    name_to_id = {name: tid for tid, name in teams.items()}
    rounds = []
    for slot in sorted(df["slot"].unique()):
        rnd = []
        sub = df[df["slot"] == slot]
        for _, row in sub.iterrows():
            h_id = name_to_id[row["home"]]
            a_id = name_to_id[row["away"]]
            rnd.append((h_id, a_id))
        rounds.append(rnd)
    return rounds


def evaluator_wrapper(df_sched: pd.DataFrame, teams: Dict[int, str], team_ids, D) -> Dict[str, float]:
    """
    Wrap your evaluator so the optimizer can call it on a DataFrame schedule.
    - Converts DF -> rounds
    - Calls evaluate_schedule (writes the .txt summary)
    - Also computes total travel from compute_travel_metrics for scoring
    Returns: {"feasible": bool, "total_travel": float}
    """
    rounds = df_to_rounds(df_sched, teams)

    # Reject early if schedule breaks DRR (pair counts or venue balance)
    if not check_double_round_robin(rounds, team_ids):
        return {"feasible": False, "total_travel": float("inf")}
    
    feasible = evaluate_schedule(teams, team_ids, rounds, D, save_path="results/NL4_evaluation.txt")
    # For scoring, compute total travel directly (no extra file writes)
    travel_df, travel_summary = compute_travel_metrics(teams, team_ids, rounds, D)
    return {"feasible": feasible, "total_travel": float(travel_summary["total_travel"])}


# -----------------------
# Schedule Pool
# -----------------------

class SchedulePool:
    def __init__(self, keep_top: int = 20):
        self.keep_top = keep_top
        self.pool = []  # each: {"schedule": df, "score": float, "feasible": bool}

    def add(self, sched_df: pd.DataFrame, score: float, feasible: bool):
        self.pool.append({"schedule": sched_df, "score": score, "feasible": feasible})
        # keep best few
        self.pool = sorted(self.pool, key=lambda x: x["score"])[: self.keep_top]

    def best(self):
        return min(self.pool, key=lambda x: x["score"]) if self.pool else None


# -----------------------
# Modification Agent
# -----------------------

class ModificationAgent:
    def mutate(self, sched_df: pd.DataFrame, move: str = None) -> pd.DataFrame:
        s = copy.deepcopy(sched_df)

        # If no specific move passed (for backward compatibility), pick randomly
        if move is None:
            move = random.choice(["swap_rounds", "swap_two_matches", "flip_home_away"])

        if move == "swap_rounds":
            rounds = sorted(s["slot"].unique().tolist())
            r1, r2 = random.sample(rounds, 2)
            # swap slot labels r1 <-> r2
            s.loc[s["slot"] == r1, "slot"] = -1  # temp
            s.loc[s["slot"] == r2, "slot"] = r1
            s.loc[s["slot"] == -1, "slot"] = r2
            s = s.sort_values(["slot", "home"]).reset_index(drop=True)

        elif move == "swap_two_matches":
            # pick a round with at least 2 matches
            r = random.choice(s["slot"].unique().tolist())
            sub_idx = s.index[s["slot"] == r].tolist()
            if len(sub_idx) >= 2:
                i1, i2 = random.sample(sub_idx, 2)
                # swap entire (home, away) pairs
                h1, a1 = s.at[i1, "home"], s.at[i1, "away"]
                h2, a2 = s.at[i2, "home"], s.at[i2, "away"]
                s.at[i1, "home"], s.at[i1, "away"] = h2, a2
                s.at[i2, "home"], s.at[i2, "away"] = h1, a1

        elif move == "flip_home_away":
            # flip a single match's venue
            i = random.choice(s.index.tolist())
            h, a = s.at[i, "home"], s.at[i, "away"]
            s.at[i, "home"], s.at[i, "away"] = a, h

        else:
            # fallback: no-op mutation
            pass

        return s



# -----------------------
# Optimizer
# -----------------------

class Optimizer:
    def __init__(self, xml_path: str):
        
        os.makedirs("results", exist_ok=True)
        self.log_path = "results/optimization_trace.jsonl"
        with open(self.log_path, "w") as f:
            f.write("")  # clear previous log

        # Parsing    
        teams, slots, D = parse_instance(xml_path)
        self.teams: Dict[int, str] = teams
        self.slots: Dict[int, str] = slots
        self.D = D
        self.team_ids = sorted(self.teams.keys())
        self.llm_agent = LLMHeuristicAgent()

        # Initializing baseline schedule
        base_sched = build_double_round_robin(self.team_ids)
        base_df = schedule_to_dataframe(base_sched, self.teams)

        # Evaluator on baseline
        base_eval = evaluator_wrapper(base_df, self.teams, self.team_ids, self.D)

        # Pool + agent
        self.pool = SchedulePool(keep_top=20)
        if base_eval["feasible"]:
            self.pool.add(base_df, base_eval["total_travel"], True)
        else:
            
            raise RuntimeError("Baseline schedule is infeasible; check generator.")

        self.agent = ModificationAgent()

    def run(self, iterations: int = 30):
        print("Starting optimization...")
        for it in range(1, iterations + 1):
            best = self.pool.best()
            move_type = self.llm_agent.suggest_next_move()
            cand_df = self.agent.mutate(best["schedule"], move_type)
            cand_eval = evaluator_wrapper(cand_df, self.teams, self.team_ids, self.D)

            prev_score = best["score"]
            new_score = cand_eval["total_travel"]
            delta = prev_score - new_score


            # Record feedback
            self.llm_agent.record_feedback(it, move_type, best["score"], cand_eval["total_travel"])
            rationale = self.llm_agent.generate_explanation(move_type)

            # Mild acceptance tolerance
            if cand_eval["feasible"]:
                if cand_eval["total_travel"] <= best["score"] * 1.01:
                    self.pool.add(cand_df, cand_eval["total_travel"], True)

            # Prepare iteration log entry
            log_entry = {
                "iteration": it,
                "move": move_type,
                "rationale": rationale,
                "prev_score": prev_score,
                "new_score": new_score,
                "delta": delta,
                "feasible": cand_eval["feasible"],
                "weights": dict(self.llm_agent.weights)
            }

            # Append to results log
            with open(self.log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            curr_best = self.pool.best()
            print(f"Iteration {it:03d} | move={move_type:17s} | Δ={delta:+.1f} | best={curr_best['score']:.1f}")

        final = self.pool.best()
        self.llm_agent.summarize_performance()
        print("\n==== FINAL BEST SCHEDULE (top rows) ====")
        print(final["schedule"].head(12).to_string(index=False))
        print(f"\nTotal Travel: {final['score']:.1f}")




# if __name__ == "__main__":
#     opt = Optimizer(xml_path="daata/NL4.xml")
#     opt.run(iterations=50)

if __name__ == "__main__":
    from parser import parse_instance
    from schedule import build_double_round_robin, schedule_to_dataframe
    from agent_coordinator import CoordinatorAgent

    teams, slots, D = parse_instance("daata/NL10.xml")
    print(D.shape, D.isna().sum().sum())
    team_ids = sorted(teams.keys())
    base_sched = build_double_round_robin(team_ids)
    base_df = schedule_to_dataframe(base_sched, teams)
    # print("baseline sched:", base_sched)
    # print("baseline df:", base_df)
    # print("Distance matrix shape:", D.shape)
    # print("Any NaN:", D.isna().sum().sum())
    # print(D)
    # print("Feasible check:", check_double_round_robin(base_df, list(range(10))))
    # print("Manual travel:", sum(D.loc[h, a] for h, a in zip(base_df.home_id, base_df.away_id)))
    # print("Evaluator output:", evaluate_schedule(teams, team_ids, base_sched, D))

    coordinator = CoordinatorAgent(teams, slots, D, base_df)
    coordinator.run(iterations=25)


# always remmeber to change max_sep when cahnging dataset

