# src/schedule_diff.py

from typing import List, Tuple, Dict

Schedule = List[List[Tuple[int, int]]]

def diff_schedules(
    old: Schedule,
    new: Schedule,
    teams: Dict[int, str]
) -> str:
    """
    Returns a human-readable diff between two schedules.
    Shows:
      - if number of rounds changed
      - for each round: unchanged, modified, or added/removed
      - for each modified round: list of changed matches
    """

    lines = []
    rounds_old = len(old)
    rounds_new = len(new)

    # Round count changes
    if rounds_old != rounds_new:
        lines.append(f"⚠️ Round count changed: {rounds_old} → {rounds_new}")

    r_max = min(rounds_old, rounds_new)

    for r in range(r_max):
        old_r = old[r]
        new_r = new[r]

        if old_r == new_r:
            continue  # no difference

        lines.append(f"\n🔄 Round {r} changed:")

        # Build per-match comparison
        for i, (o, n) in enumerate(zip(old_r, new_r)):
            if o != n:
                o_h, o_a = o
                n_h, n_a = n
                lines.append(
                    f"   • Match {i}: "
                    f"{teams[o_h]} vs {teams[o_a]}  →  {teams[n_h]} vs {teams[n_a]}"
                )

        # Extra matches (unequal number in round)
        if len(old_r) != len(new_r):
            lines.append(
                f"   ⚠️ Different number of matches: {len(old_r)} → {len(new_r)}"
            )

    if rounds_new > rounds_old:
        lines.append(f"\n➕ New rounds added: {list(range(rounds_old, rounds_new))}")
    elif rounds_new < rounds_old:
        lines.append(f"\n➖ Rounds removed: {list(range(rounds_new, rounds_old))}")

    return "\n".join(lines) if lines else "No changes in schedule."
