# src/heuristic_agent.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import json
import textwrap
import pandas as pd

from heuristic_prompts import (
    HEURISTIC_SYSTEM_PROMPT,
    HEURISTIC_USER_TEMPLATE,
    MOVE_CATALOG,
)

from llm_client import LLMClient

# ================================================================
# Type Aliases
# ================================================================
TeamId = int
Game = Tuple[TeamId, TeamId]
Round = List[Game]
Schedule = List[Round]


# ================================================================
# Heuristic LLM Agent
# ================================================================
class HeuristicLLMAgent:
    """
    LLL-based local-search heuristic agent.
    It:
      - observes schedule + travel metrics
      - asks the LLM to propose exactly one move
      - returns structured move dictionary
    """

    def __init__(
        self,
        model: Optional[str] = None,
        max_history_items: int = 5,
        max_rounds_in_prompt: int = 8,
    ):
        self.llm = LLMClient(model=model, temperature=0.3)
        self.max_history_items = max_history_items
        self.max_rounds_in_prompt = max_rounds_in_prompt

    
    def _count_recent_failures(self, history, move_name):
        """
        Count how many times this move_name failed consecutively,
        starting from the most recent iteration backward.
        """
        count = 0
        for h in reversed(history):
            if h.get("move") == move_name:
                if not h.get("accepted", True):
                    count += 1
                else:
                    break  # chain breaks if an accepted move appears
            else:
                break  # different move breaks the chain
        return count

    # ------------------------------------------------------------
    # Main decision function
    # ------------------------------------------------------------
    def choose_move(
        self,
        schedule: Schedule,
        teams: Dict[TeamId, str],
        metrics: Dict[str, Any],
        history: List[Dict[str, Any]],
        iteration: int,
        team_ids: List[int],
        D: Any,
    ) -> Optional[Dict[str, Any]]:

        messages = self._build_messages(schedule, teams, metrics, history, iteration, team_ids, D)

        try:
            raw = self.llm.chat(messages)
        except Exception as e:
            print(f"[HeuristicLLMAgent] LLM call failed: {e}")
            return None

        if not raw or raw.strip() == "":
            print("[HeuristicLLMAgent] Empty response from LLM.")
            return None

        # Parse JSON
        try:
            parsed = json.loads(raw)
        except Exception:
            print("[HeuristicLLMAgent] JSON parsing failed.")
            print("Raw LLM output:\n", raw)
            return None

        if "move_name" not in parsed or "params" not in parsed:
            print("[HeuristicLLMAgent] Missing required fields.")
            return None

        move_name = parsed["move_name"]
        params = parsed["params"]

        # ----------------------------------------------------------
        # SAFETY RULE — Move must exist in MOVE_CATALOG
        # ----------------------------------------------------------
        if move_name not in MOVE_CATALOG:
            print(f"[HeuristicLLMAgent] Invalid move: {move_name}")
            return None

        # ----------------------------------------------------------
        # BLOCKING RULE A — Never repeat identical rejected move
        # ----------------------------------------------------------
        for h in history:
            if (h.get("move") == move_name and
                h.get("params") == params and
                not h.get("accepted", True)):
                print("[Agent] Rejecting: Same rejected move & parameters used before.")
                return None

        # ----------------------------------------------------------
        # BLOCKING RULE B — If move_name failed ≥3 times in a row,
        # require a different move_name.
        # ----------------------------------------------------------
        fail_count = self._count_recent_failures(history, move_name)
        if fail_count >= 3:
            print(f"[Agent] Rejecting: move '{move_name}' failed {fail_count} times in a row. Use a different move.")
            return None
        
        # ----------------------------------------------------------
        # RETURN MOVE
        # ----------------------------------------------------------
        return {
            "move_name": move_name,
            "params": params,
            "rationale": parsed.get("rationale", "")
        }
    
    def _format_rejected_moves(self, history: List[Dict[str, Any]]) -> str:
        rejected = []
        for h in history:
            if h.get("accepted") is False:
                rejected.append({
                    "move_name": h.get("move"),
                    "params": h.get("params", {})
                })
        if not rejected:
            return "None"
        return json.dumps(rejected, indent=2)

    # ------------------------------------------------------------
    # Prompt-building functions
    # ------------------------------------------------------------
    def _build_messages(
        self,
        schedule: Schedule,
        teams: Dict[TeamId, str],
        metrics: Dict[str, Any],
        history: List[Dict[str, Any]],
        iteration: int,
        team_ids: List[int],
        D: Any,
    ) -> List[Dict[str, str]]:

        user = HEURISTIC_USER_TEMPLATE.format(
            iteration=iteration,
            feasible_drr=metrics.get("feasible_drr", True),
            move_catalog=self._format_move_catalog(),
            current_metrics=self._format_metrics(metrics),
            ca3=metrics.get("ca3_violations", 0),
            history_summary=self._format_history(history),
            rejected_moves=self._format_rejected_moves(history),
            schedule_summary=self._format_schedule(schedule, teams),
            travel_insights=self._format_travel_insights(schedule, teams, team_ids, D),
        )

        return [
            {"role": "system", "content": HEURISTIC_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    # ------------------------------------------------------------
    def _format_move_catalog(self) -> str:
        lines = []
        for name, info in MOVE_CATALOG.items():
            params = info.get("params", {})
            param_str = ", ".join(f"{p}: {t}" for p, t in params.items())
            desc = info.get("description", "")
            lines.append(f"- {name}({param_str}): {desc}")
        return "\n".join(lines)

    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        keys = ["raw_total_travel", "total_travel"]
        return "\n".join(f"{k}: {metrics.get(k, '?')}" for k in keys)

    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """
        Show recent iterations including:
        - move name
        - parameters
        - acceptance status
        - raw travel value

        This gives the LLM enough context to avoid repeating bad moves.
        """
        if not history:
            return "No previous iterations."

        last = history[-self.max_history_items:]
        lines = []

        for h in last:
            move = h.get("move", "?")
            params = h.get("params", {})
            accepted = h.get("accepted", "?")
            raw_total = h.get("raw_total", "?")

            lines.append(
                f"- Iter {h.get('iter')}: move={move}, params={params}, "
                f"raw_total={raw_total}, accepted={accepted}"
            )

        return "\n".join(lines)

    def _format_schedule(self, schedule: Schedule, teams: Dict[TeamId, str]) -> str:
        limit = min(self.max_rounds_in_prompt, len(schedule))
        lines = [f"Showing first {limit} rounds:"]
        for r in range(limit):
            rnd = schedule[r]
            games = [f"{teams[h]} vs {teams[a]}" for (h, a) in rnd]
            wrapped = textwrap.wrap(", ".join(games), width=100)
            lines.append(f"Round {r}:")
            for w in wrapped:
                lines.append(f"  {w}")
        return "\n".join(lines)

    def _format_travel_insights(
        self, schedule: Schedule, teams: Dict[int, str], team_ids: List[int], D: Any
    ) -> str:

        totals = {tid: 0 for tid in team_ids}

        for rnd in schedule:
            for h, a in rnd:
                totals[a] += float(D.iloc[h, a])

        ordered = sorted(totals.items(), key=lambda x: -x[1])

        lines = ["Per-team travel:"]
        for tid, t in ordered:
            lines.append(f"{teams[tid]}: {t:.1f}")
        lines.append("")

        legs = []
        for rnd in schedule:
            for h, a in rnd:
                legs.append((float(D.iloc[h, a]), teams[h], teams[a]))
        legs.sort(reverse=True)

        lines.append("Top 5 expensive legs:")
        for dist, h, a in legs[:5]:
            lines.append(f"{h} → {a} = {dist:.1f}")

        return "\n".join(lines)
