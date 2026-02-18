# src/agent_coordinator.py
import json
import os
import pandas as pd
from typing import Dict

from llm_agent import LLMHeuristicAgent
from evaluator import evaluate_schedule, compute_travel_metrics
from optimizer import df_to_rounds
from schedule import check_double_round_robin
from optimizer import ModificationAgent, evaluator_wrapper
from explainer_agent import ExplainerAgent

from dotenv import load_dotenv
load_dotenv()



# -----------------------------
# Evaluator Agent
# -----------------------------

class EvaluatorAgent:
    """
    Handles all evaluation logic — feasibility, constraint checks,
    and total travel computation. This isolates evaluation
    from the optimization logic for cleaner agent interaction.
    """

    def __init__(self, teams: Dict[int, str], team_ids, D):
        self.teams = teams
        self.team_ids = team_ids
        self.D = D

    def evaluate(self, sched_df: pd.DataFrame):
        """
        Evaluate a schedule (in DataFrame form) and return performance metrics.
        """
        rounds = df_to_rounds(sched_df, self.teams)
        feasible_drr = check_double_round_robin(rounds, self.team_ids)
        if not feasible_drr:
            return {"feasible": False, "total_travel": float("inf")}

        feasible = evaluate_schedule(self.teams, self.team_ids, rounds, self.D)
        travel_df, travel_summary = compute_travel_metrics(self.teams, self.team_ids, rounds, self.D)

        return {
            "feasible": feasible,
            "total_travel": float(travel_summary["total_travel"]),
            "avg_travel": float(travel_summary["average_travel"]),
            "var_travel": float(travel_summary["variance_travel"]),
        }


# -----------------------------
# Coordinator Agent
# -----------------------------

class CoordinatorAgent:
    """
    The main orchestrator — coordinates between:
      - HeuristicAgent (decides what move to make)
      - ModificationAgent (executes the move)
      - EvaluatorAgent (evaluates new schedules)
    """

    def __init__(self, teams, slots, D, base_df):
        self.teams = teams
        self.team_ids = sorted(teams.keys())
        self.D = D
        self.heuristic_agent = LLMHeuristicAgent()
        self.mod_agent = ModificationAgent()
        self.eval_agent = EvaluatorAgent(teams, sorted(teams.keys()), D)
        self.explainer = ExplainerAgent(log_path="results/logs/agent_dialogue.txt")
        self.current_best = {"schedule": base_df, "score": float("inf")}
        self.history = []

        os.makedirs("results", exist_ok=True)
        self.log_path = "results/agent_trace.jsonl"
        with open(self.log_path, "w") as f:
            f.write("")  # clear previous log

    def step(self, iteration: int):
        # === STEP 1: Get heuristic suggestion (now with context) ===
        prev_best_score = self.current_best.get("score", float("inf"))
        move = self.heuristic_agent.suggest_next_move(
            iteration=iteration,
            best_score=prev_best_score,
        )
        rationale = self.heuristic_agent.generate_explanation(move)

        # === STEP 2: Apply mutation ===
        candidate = self.mod_agent.mutate(self.current_best["schedule"], move)

        # === STEP 3: Evaluate candidate ===
        eval_result = evaluator_wrapper(candidate, self.teams, self.team_ids, self.D)
        feasible = eval_result["feasible"]
        new_score = eval_result["total_travel"]

        # === STEP 4: Compute Δ ===
        prev_score = prev_best_score
        delta = prev_score - new_score if feasible else float("-inf")

        # === STEP 5: Record feedback (update move weights) ===
        self.heuristic_agent.record_feedback(
            iteration=iteration,
            move_type=move,
            prev_score=prev_score,
            new_score=new_score
        )

        # === STEP 6: If feasible & better, update best schedule ===
        if feasible and new_score < prev_score:
            self.current_best = {"schedule": candidate, "score": new_score}

        # === STEP 7: Log reasoning and outcome ===
        _ = self.explainer.narrate_step(
            iteration=iteration,
            move=move,
            rationale=rationale,
            prev_score=prev_score,
            new_score=new_score,
            delta=delta,
            feasible=feasible,
            weights_after=self.heuristic_agent.weights
        )

        # === STEP 8: Print progress ===
        print(f"Iter {iteration:03d} | move={move:<15} | Δ={delta:+.1f} | best={self.current_best['score']:.1f}")

    def run(self, iterations: int = 50):
        """
        Runs a full optimization cycle.
        """
        print("🤖 Starting multi-agent optimization...")
        for it in range(1, iterations + 1):
            self.step(it)

        # Final summary
        print("\n=== Final Best Schedule ===")
        print(self.current_best["schedule"].head(12).to_string(index=False))
        print(f"\nTotal Travel: {self.current_best['score']:.1f}")
        print(f"Optimization trace saved to {self.log_path}\n")
        self.heuristic_agent.summarize_performance()
