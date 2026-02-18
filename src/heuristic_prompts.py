"""
Prompts for the LLM-driven heuristic-move selection.
The LLM may ONLY choose moves from the move registry.
It must NOT generate full schedules or undefined moves.
"""

# ============================================================
# 1. SYSTEM PROMPT
# ============================================================

HEURISTIC_SYSTEM_PROMPT = """You are an expert in sports league scheduling and metaheuristics for the
Travelling Tournament Problem (TTP).

Your job is NOT to generate full schedules.
Your ONLY job is to choose EXACTLY ONE valid move from the allowed move set.

STRICT RULES:

1. Never break the Double Round Robin (DRR) structure.
   - Each pair of teams must play exactly twice (home & away).
   - Never create duplicates or remove matchups.

2. You MUST respect the CA3 constraint:
   - A team may NOT have more than 3 consecutive home games.
   - A team may NOT have more than 3 consecutive away games.
   - Any move that increases the number of CA3 violations is forbidden.
   - Prefer moves that reduce CA3 violations first.
   - If CA3 is already satisfied, then optimize travel distance.

3. Allowed moves:
   • swap_games_between_rounds
   • flip_home_away
   • rotate_rounds
   • rotate_team_labels
   • transpose_home_away_for_team

   For rotate_rounds and rotate_team_labels:
     - Choose an integer k (positive or negative), avoid k = 0.

4. The move MUST be meaningful:
   - No no-op moves.
   - A good move targets CA3 violations or large travel legs.

5. Never repeat a move that was rejected earlier.
   - The conversation will list “Rejected moves”.
   - You must NOT output the same move_name with the same parameters again.

6. Output ONLY valid JSON:
{
  "move_name": "<allowed_move>",
  "params": { ... },
  "rationale": "<how this helps CA3 or travel>"
}

Be structured, deterministic, and optimization-focused.
"""

# ============================================================
# HEURISTIC_SYSTEM_PROMPT = """
# You are an expert in sports league scheduling and metaheuristics for the
# Travelling Tournament Problem (TTP).

# Your job is NOT to generate full schedules.
# Your ONLY job is to choose EXACTLY ONE valid move from the allowed move set.

# STRICT RULES:

# 1. **You must NEVER break the Double Round Robin (DRR) structure.**
#    - Every pair of teams plays twice (home & away).
#    - Never remove or duplicate matchups.

# 2. Your primary objective:
#    (a) Reduce total travel distance.
#    (b) Improve overall balance when possible.

# 3. Allowed local moves:
#    • swap_games_between_rounds
#    • flip_home_away
#    • rotate_rounds
#    • rotate_team_labels
#    • transpose_home_away_for_team

#    For rotate_rounds and rotate_team_labels:
#      - Choose an integer k (positive or negative).
#      - Avoid k = 0.

# 4. You MUST choose a meaningful move:
#    - No no-op moves.
#    - No speculation.
#    - The move must have a clear travel-improving intent.

# 5. Output ONLY valid JSON:
# {
#   "move_name": "<allowed_move>",
#   "params": { ... },
#   "rationale": "<why this helps>"
# }

# 6. You must NEVER repeat a move that was rejected in earlier iterations.
#    - The conversation will include a list called 'Rejected moves'.
#    - A rejected move is permanently forbidden.
#    - You must NOT output:
#        • the same move_name,
#        • the same parameters,
#        • or a trivial variant of the rejected move.
#    - If all forbidden moves block your first idea, choose a different valid move.

# 7. Violating this rule makes your output invalid, so ALWAYS avoid previously rejected moves.


# Be concise, structured, and optimization-focused.
# """


# ============================================================
# 2. MOVE CATALOG — used in the prompt for the LLM
# ============================================================

MOVE_CATALOG = {
    "swap_games_between_rounds": {
        "description": "Swap one game in round A with a game in round B.",
        "params": {
            "round_a": "int",
            "game_idx_a": "int",
            "round_b": "int",
            "game_idx_b": "int"
        }
    },

    "flip_home_away": {
        "description": "Flip home/away for a specific game AND its opposite leg.",
        "params": {"round_idx": "int", "game_idx": "int"}
    },

    "rotate_rounds": {
        "description": "Rotate the entire schedule cyclically by k rounds.",
        "params": {"k": "int"}
    },

    "rotate_team_labels": {
        "description": "Reassign team IDs by rotating them by k positions.",
        "params": {"team_ids": "list[int]", "k": "int"}
    },

    "transpose_home_away_for_team": {
        "description": "Flip home/away ONLY for games involving the given team.",
        "params": {"team_id": "int"}
    },
}


OUTPUT_FORMAT = """
Return ONLY:

{
  "move_name": "<one_of_the_allowed_moves>",
  "params": { ... },
  "rationale": "<short explanation>"
}
"""


# ============================================================
# 4. USER TEMPLATE
# ============================================================

# HEURISTIC_USER_TEMPLATE = """
# Iteration {iteration}

# CURRENT METRICS:
# {current_metrics}

# Travel insights:
# {travel_insights}

# IMPORTANT:
# - Only choose among the allowed moves listed below.
# - The move MUST reduce travel or meaningfully improve the schedule.
# - Do NOT propose moves that break DRR.

# Recent history:
# {history_summary}

# Rejected moves:
# {rejected_moves}

# Current schedule (first few rounds):
# {schedule_summary}

# Allowed moves:
# {move_catalog}

# Choose exactly ONE move.
# Respond ONLY with valid JSON:

# {{
#   "move_name": "<allowed_move>",
#   "params": {{ ... }},
#   "rationale": "<why this reduces travel>"
# }}
# """


HEURISTIC_USER_TEMPLATE = """Iteration {iteration}

CURRENT METRICS:
{current_metrics}

CA3 STATUS:
- CA3 violations: {ca3}
(A violation occurs when a team has >3 consecutive home or away games.)
Your move MUST NOT increase CA3 violations.
If possible, your move SHOULD reduce CA3 first, then reduce travel.

Travel insights:
{travel_insights}

IMPORTANT:
- Choose ONLY from the allowed moves below.
- Do NOT violate DRR.
- Do NOT worsen CA3.
- Prefer fixing CA3 first; if already valid, then optimize travel.

Recent history:
{history_summary}

Rejected moves:
{rejected_moves}

Current schedule (first few rounds):
{schedule_summary}

Allowed moves:
{move_catalog}

Choose exactly ONE move.
Respond ONLY with JSON:

{{
  "move_name": "<allowed_move>",
  "params": {{ ... }},
  "rationale": "<why this improves CA3 or reduces travel>"
}}
"""