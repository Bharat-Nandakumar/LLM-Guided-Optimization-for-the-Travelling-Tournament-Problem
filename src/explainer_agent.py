# src/explainer_agent.py
import os
from typing import Dict, Optional

class ExplainerAgent:
    """
    Produces short, human-readable explanations for:
      - why a move was chosen (from HeuristicAgent)
      - what outcome occurred (Δ, feasibility)
      - what we learned (weight shift summary)
    Logs to logs/agent_dialogue.txt and returns a one-line summary for console.
    """

    def __init__(self, log_path: str = "results/logs/agent_dialogue.txt"):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        # wipe file on init
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("=== Agent Dialogue Trace ===\n")

    def _fmt_weights(self, weights: Dict[str, float]) -> str:
        parts = [f"{k}={v:.3f}" for k, v in sorted(weights.items())]
        return ", ".join(parts)

    def narrate_step(
        self,
        iteration: int,
        move: str,
        rationale: str,
        prev_score: float,
        new_score: float,
        delta: float,
        feasible: bool,
        weights_after: Dict[str, float],
        note: Optional[str] = None,
    ) -> str:
        """
        Write a compact multi-line explanation for one iteration.
        Returns a single-line summary for console printing if needed.
        """
        status = (
            "improved" if delta > 0 else
            "unchanged" if delta == 0 else
            "worsened"
        )
        feas_str = "feasible ✅" if feasible else "infeasible ❌"

        lines = [
            f"\n[Iteration {iteration:03d}]",
            f" HeuristicAgent → move: {move}",
            f" Rationale      → {rationale}",
            f" EvaluatorAgent → prev={prev_score:.1f}, new={new_score:.1f}, Δ={delta:+.1f}, {feas_str} ({status})",
            f" Weights (post) → {self._fmt_weights(weights_after)}",
        ]
        if note:
            lines.append(f" Note           → {note}")

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        # one-line console summary (optional)
        return f"[Explain] it={iteration:03d} | {move} | Δ={delta:+.1f} | {feas_str}"
