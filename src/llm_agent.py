# src/llm_agent.py
import random
import numpy as np
from typing import Dict, Optional, List
from llm_client import LLMClient
import os

_ALLOWED_MOVES = ["swap_rounds", "swap_two_matches", "flip_home_away"]

_SYSTEM_PROMPT = """You are a scheduling optimization assistant.
You choose exactly ONE move type per turn to improve a double round-robin league schedule.

Allowed moves (choose one):
- swap_rounds: swap two entire rounds
- swap_two_matches: swap two matches within the same round
- flip_home_away: flip the venue for a single match

Goals, in priority order:
1) keep the schedule feasible (double round robin integrity and fairness rules),
2) reduce total travel distance,
3) avoid long home/away streaks.

You MUST reply with a strict JSON object on one line:
{"move": "<one of: swap_rounds|swap_two_matches|flip_home_away>", "reason": "<short reason>"}.
No extra text. No markdown. No explanations outside JSON.
"""

_USER_TEMPLATE = """Context:
- iteration: {iteration}
- current_best_total_travel: {best_score}
- recent_moves: {recent_moves}
- current_move_weights: {weights}

Constraints snapshot:
- CA3 (max home/away streak): 3
- SE1 (min/max separation between pair matches): 1..6

Pick the next move to try now.
"""

def _safe_json_pick_move(text: str) -> Optional[Dict[str, str]]:
    # very defensive parser
    try:
        import json
        # grab the first {...} block if model added chatter
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        obj = json.loads(text[start:end+1])
        mv = obj.get("move", "").strip()
        rs = obj.get("reason", "").strip()
        if mv in _ALLOWED_MOVES:
            return {"move": mv, "reason": rs}
    except Exception:
        pass
    return None

class LLMHeuristicAgent:
    def __init__(self, llm: Optional[LLMClient] = None):
        # Available mutation types
        self.move_types = list(_ALLOWED_MOVES)
        # Initialize equal weights (will adapt dynamically)
        self.weights = {m: 1.0 for m in self.move_types}
        # Keep record of past performance
        self.history = []
        # optional real LLM client
        self.llm = llm or self._maybe_make_default_client()

    def _maybe_make_default_client(self) -> Optional[LLMClient]:
        try:
            return LLMClient()  # will raise at call-time if unconfigured
        except Exception:
            return None

    def record_feedback(self, iteration, move_type, prev_score, new_score):
        # Handle infinities safely
        if not (isinstance(prev_score, (int, float)) and isinstance(new_score, (int, float))):
            return
        if prev_score == float("inf") or np.isnan(prev_score):
            prev_score = new_score
        if np.isnan(new_score):
            new_score = prev_score

        delta = prev_score - new_score
        improved = delta > 0
        self.history.append({
            "iter": iteration,
            "move": move_type,
            "prev_score": prev_score,
            "new_score": new_score,
            "delta": delta,
            "improved": improved
        })

        # Adaptive learning rate
        alpha = 0.2
        if improved:
            self.weights[move_type] += alpha * abs(delta) / max(prev_score, 1e-9)
        else:
            self.weights[move_type] *= (1 - alpha / 2)

        # Normalize weights to avoid explosion
        total = sum(self.weights.values())
        if total <= 0 or np.isnan(total):
            total = 1.0

        for m in self.move_types:
            self.weights[m] = max(self.weights[m] / total, 0.01)

        # Final renormalization to sum to 1.0
        total = sum(self.weights.values())
        for m in self.move_types:
            self.weights[m] = self.weights[m] / total

    def _fallback_weighted_sample(self) -> str:
        moves = list(self.weights.keys())
        probs = np.array(list(self.weights.values()), dtype=float)
        if np.isnan(probs).any() or probs.sum() <= 0:
            probs = np.ones(len(moves)) / len(moves)
        else:
            probs = probs / probs.sum()
        return np.random.choice(moves, p=probs)

    def suggest_next_move(
        self,
        iteration: int,
        best_score: float,
        recent_k: int = 6,
    ) -> str:
        """
        Ask the LLM for the next move using current context.
        On any failure or invalid reply, falls back to weighted sampling.
        """
        used_fallback = False

        if self.llm is None:
            used_fallback = True
            move = self._fallback_weighted_sample()
        else:
            # Build short context
            recent = self.history[-recent_k:]
            recent_moves = [h["move"] for h in recent] if recent else []
            weights_sorted = {k: round(float(v), 3) for k, v in sorted(self.weights.items())}

            user = _USER_TEMPLATE.format(
                iteration=iteration,
                best_score=round(float(best_score), 1) if best_score != float("inf") else "inf",
                recent_moves=recent_moves,
                weights=weights_sorted,
            )
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]

            try:
                text = self.llm.chat(messages)
                os.makedirs("results/logs", exist_ok=True)
                with open("results/logs/llm_raw_responses.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n=== Iteration {iteration} ===\n")
                    f.write("Prompt:\n")
                    for msg in messages:
                        f.write(f"{msg['role'].upper()}: {msg['content']}\n")
                    f.write("\nResponse:\n")
                    f.write(text.strip() + "\n")
                    f.write("=" * 60 + "\n")
                parsed = _safe_json_pick_move(text)
                if parsed:
                    self._last_reason = parsed.get("reason", "")
                    move = parsed["move"]
                else:
                    used_fallback = True
                    move = self._fallback_weighted_sample()
            except Exception as e:
                used_fallback = True
                move = self._fallback_weighted_sample()

        # 👇 Logging statement here
        if used_fallback:
            print(f"[LLMHeuristicAgent] ⚙️ Used fallback sampler (no or invalid LLM response).")
        else:
            print(f"[LLMHeuristicAgent] 🤖 Used real LLM for move selection: {move}")

        return move

    def generate_explanation(self, move_type):
        # Prefer model's concise reason if present
        if getattr(self, "_last_reason", ""):
            return self._last_reason.strip()[:280]
        explanations = {
            "swap_rounds": "Swapping rounds may group distant away trips to reduce total travel.",
            "swap_two_matches": "Swapping two matches within the same round can fix streak or venue balance issues.",
            "flip_home_away": "Flipping venues can address long home/away streaks without changing round structure."
        }
        return explanations.get(move_type, "Exploring a new move for diversity.")

    def summarize_performance(self):
        print("\n=== Heuristic Agent Summary ===")
        for m, w in self.weights.items():
            print(f"{m:20s}: weight={w:.3f}")
