import xml.etree.ElementTree as ET
import pandas as pd
from typing import Dict, Tuple, Optional

def parse_instance(xml_path: str):
    """
    Returns:
      teams:       {team_id -> name}
      slots:       {slot_id -> name}
      D:           distance matrix DataFrame
      max_streak:  max allowed consecutive home OR away games (from CA3)
      max_sep:     max allowed separation between two games (from SE1)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # ----------------------------
    # TEAMS
    # ----------------------------
    teams = {int(t.get("id")): t.get("name")
             for t in root.find("Resources").find("Teams")}

    # ----------------------------
    # SLOTS
    # ----------------------------
    slots = {int(s.get("id")): s.get("name")
             for s in root.find("Resources").find("Slots")}

    # ----------------------------
    # DISTANCE MATRIX
    # ----------------------------
    rows = []
    for d in root.find("Data").find("Distances"):
        rows.append((int(d.get("team1")), int(d.get("team2")), int(d.get("dist"))))
    df = pd.DataFrame(rows, columns=["i", "j", "dist"])
    D = df.pivot(index="i", columns="j", values="dist").sort_index().sort_index(axis=1)

    # ----------------------------
    # CAPACITY CONSTRAINTS (CA3)
    # ----------------------------
    max_streak = None
    cap_root = root.find("Constraints").find("CapacityConstraints")
    if cap_root is not None:
        for cap in cap_root:
            if cap.tag == "CA3":
                max_streak = int(cap.get("max"))  # e.g., 3
                break

    # Default fallback (safe)
    if max_streak is None:
        max_streak = 3

    # ----------------------------
    # SEPARATION CONSTRAINTS (SE1)
    # ----------------------------
    max_sep = None
    sep_root = root.find("Constraints").find("SeparationConstraints")
    if sep_root is not None:
        for sep in sep_root:
            if sep.tag == "SE1":
                max_sep = int(sep.get("max"))  # e.g., 6
                break

    # Default fallback (safe)
    if max_sep is None:
        max_sep = 6

    return teams, slots, D, max_streak, max_sep


if __name__ == "__main__":
    teams, slots, D, max_streak, max_sep = parse_instance("data/NL4.xml")

    print("Teams:", teams)
    print("Slots:", slots)
    print("Distance matrix:\n", D)
    print("max_streak (CA3):", max_streak)
    print("max_sep (SE1):", max_sep)
