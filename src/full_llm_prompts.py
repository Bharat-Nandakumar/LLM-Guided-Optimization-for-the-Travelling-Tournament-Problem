# src/full_llm_prompts.py
SYSTEM_PROMPT = """
You are an expert in sports league scheduling and combinatorial optimization.

You are solving a Travelling Tournament Problem (TTP) instance:
- Double round-robin league.
- Objective: minimize total travel distance.
- Constraints:
  - DRR: each pair of teams plays exactly two games, one home and one away.
  - CA3: no team may have more than 3 consecutive home or away games.
  - SE1: for each pair of teams, their two games must be separated by between 1 and 6 rounds.

You must be careful, precise, and constraint-aware.
"""

USER_TEMPLATE = """
You are given a CURRENT schedule, a distance matrix, and a summary of recent evaluations.

Number of teams: {num_teams}
Number of rounds: {num_rounds}

Teams and distance matrix (travel cost when the row team travels to the column team), in JSON:
{distances_json}

CURRENT schedule, in JSON (list of rounds, each with matches using team NAMES):
{schedule_json}

Recent evaluation history (best is lower total travel):
{history_block}

Best total travel seen so far: {best_total_travel}

TASK:
Propose a NEW schedule for this league that:

1) Is a valid double round-robin:
   - Each round: every team plays at most one game.
   - Across all rounds: each unordered pair of teams plays exactly TWO games, one at each venue.

2) Respects constraints:
   - CA3: no team should have more than 3 consecutive home or away games.
   - SE1: the two games between any pair of teams should be separated by at least 1 and at most 6 rounds.

3) Attempts to reduce total travel compared to the best total travel above.
   (Even if you cannot guarantee it, try to improve or at least not worsen it.)

OUTPUT FORMAT (STRICT):
Return ONLY a JSON array on a single line, with this exact structure:

[
  {{
    "round": 0,
    "matches": [
      {{"home": "TEAM_NAME", "away": "TEAM_NAME"}},
      ...
    ]
  }},
  {{
    "round": 1,
    "matches": [
      ...
    ]
  }},
  ...
]

Requirements:
- Use ONLY team names that appear in the current schedule.
- Do NOT include any extra keys, comments, or text outside the JSON array.
- Ensure the number of rounds is exactly {num_rounds}.
"""
