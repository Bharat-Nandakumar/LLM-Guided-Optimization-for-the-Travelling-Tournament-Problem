# src/full_llm_agent.py
import json
import os
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd  # only for typing clarity; you already depend on pandas

from llm_client import LLMClient
from full_llm_prompts import SYSTEM_PROMPT, USER_TEMPLATE

Schedule = List[List[Tuple[int, int]]]  # list of rounds, each is list of (home_id, away_id)


class FullScheduleLLMAgent:
    """
    Agent that asks an LLM to propose a full schedule (all rounds),
    given the current best schedule, distances, and a short eval history.
    """

    def __init__(self, log_path: str = "results/logs/full_llm_raw.txt"):
        self.llm = LLMClient()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path

    # ---------- helpers: JSON conversions ----------

    @staticmethod
    def schedule_to_json(schedule: Schedule, teams: Dict[int, str]) -> str:
        """
        Convert internal schedule (list of rounds with (home_id, away_id)) to
        a JSON string with team NAMES, in the format described in USER_TEMPLATE.
        """
        rounds_json = []
        for r_idx, rnd in enumerate(schedule):
            matches = []
            for h, a in rnd:
                matches.append({
                    "home": teams[h],
                    "away": teams[a],
                })
            rounds_json.append({
                "round": r_idx,
                "matches": matches,
            })
        return json.dumps(rounds_json, indent=2)

    @staticmethod
    def distances_to_json(D: pd.DataFrame, teams: Dict[int, str]) -> str:
        """
        Convert distance matrix (indexed by team ids) to a nested dict keyed by team names.
        {
          "ATL": {"ATL": 0, "NYM": 200, ...},
          "NYM": {...},
          ...
        }
        """
        matrix: Dict[str, Dict[str, int]] = {}
        for i, name_i in teams.items():
            row = {}
            for j, name_j in teams.items():
                row[name_j] = int(D.loc[i, j])
            matrix[name_i] = row
        return json.dumps(matrix, indent=2)

    @staticmethod
    def format_history_for_prompt(history: List[Dict[str, Any]], k: int = 5) -> str:
        """
        Turn last k evaluation entries into a textual block for the prompt.
        Each entry in history is expected to have:
          - iter
          - raw_total_travel
          - feasible
          - ca3_violations
          - se1_violations
        """
        if not history:
            return "No previous iterations. This is the first attempt."

        recent = history[-k:]
        lines = []
        for h in recent:
            lines.append(
                f"Iter {h['iter']}: "
                f"raw_total_travel={h['raw_total_travel']}, "
                f"feasible={h['feasible']}, "
                f"CA3_violations={h['ca3_violations']}, "
                f"SE1_violations={h['se1_violations']}"
            )
        return "\n".join(lines)

    @staticmethod
    def _extract_json_array(text: str) -> Optional[Any]:
        """
        Very defensive: try to locate the first [...] JSON array in the response.
        Returns parsed Python object or None.
        """
        try:
            start = text.find("[")
            end = text.rfind("]")
            if start == -1 or end == -1 or end <= start:
                return None
            snippet = text[start:end + 1]
            return json.loads(snippet)
        except Exception:
            return None

    @staticmethod
    def json_to_schedule(obj: Any, teams: Dict[int, str], num_rounds: int) -> Optional[Schedule]:
        """
        Convert parsed JSON array of rounds with team NAMES back into
        list of rounds with team IDs: [[(home_id, away_id), ...], ...].

        Expects structure:
        [
          {"round": 0, "matches": [{"home":"ATL","away":"NYM"}, ...]},
          ...
        ]
        """
        if not isinstance(obj, list):
            return None

        name_to_id = {name: tid for tid, name in teams.items()}
        schedule: Schedule = []

        # We ignore the "round" field and just use the array order,
        # but we can roughly check the number of rounds.
        if len(obj) != num_rounds:
            # still attempt, but caller may reject during DRR check
            pass

        for round_obj in obj:
            if not isinstance(round_obj, dict):
                return None
            matches = []
            for m in round_obj.get("matches", []):
                if not isinstance(m, dict):
                    return None
                h_name = m.get("home")
                a_name = m.get("away")
                if h_name not in name_to_id or a_name not in name_to_id:
                    return None
                h_id = name_to_id[h_name]
                a_id = name_to_id[a_name]
                matches.append((h_id, a_id))
            schedule.append(matches)

        return schedule

    # ---------- main API ----------

    def propose_schedule(
        self,
        current_best: Schedule,
        teams: Dict[int, str],
        D: pd.DataFrame,
        num_rounds: int,
        history: List[Dict[str, Any]],
        best_total_travel: float,
        iteration: int,
    ) -> Optional[Schedule]:
        """
        Ask the LLM to propose a new full schedule.
        Returns a new Schedule (list-of-rounds) or None on failure.
        """

        schedule_json = self.schedule_to_json(current_best, teams)
        distances_json = self.distances_to_json(D, teams)
        history_block = self.format_history_for_prompt(history)

        user_prompt = USER_TEMPLATE.format(
            num_teams=len(teams),
            num_rounds=num_rounds,
            distances_json=distances_json,
            schedule_json=schedule_json,
            history_block=history_block,
            best_total_travel=(
                "inf" if best_total_travel == float("inf") else round(float(best_total_travel), 1)
            ),
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]

        # Call LLM and log raw prompt/response
        text = self.llm.chat(messages)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== Iteration {iteration} ===\n")
            f.write("SYSTEM:\n" + SYSTEM_PROMPT.strip() + "\n\n")
            f.write("USER PROMPT:\n" + user_prompt.strip() + "\n\n")
            f.write("RESPONSE:\n" + text.strip() + "\n")
            f.write("=" * 60 + "\n")

        parsed = self._extract_json_array(text)
        if parsed is None:
            print(f"[FullScheduleLLMAgent] Failed to parse JSON array from LLM response.")
            return None

        schedule = self.json_to_schedule(parsed, teams, num_rounds)
        if schedule is None:
            print(f"[FullScheduleLLMAgent] Failed to convert JSON to internal schedule.")
            return None

        return schedule
