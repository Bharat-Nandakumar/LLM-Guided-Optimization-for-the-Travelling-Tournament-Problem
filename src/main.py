
from parser import parse_instance
from schedule import build_double_round_robin, schedule_to_dataframe, total_travel
from evaluator import evaluate_schedule

def main():
    teams, slots, D = parse_instance("daata/NL4.xml")
    team_ids = sorted(teams.keys())

    schedule = build_double_round_robin(team_ids)
    df = schedule_to_dataframe(schedule, teams)
    print("Total travel:", total_travel(team_ids, schedule, D))
    # Save to results
    df.to_csv("results/NL4_schedule.csv", index=False)

    #evaluate
    evaluate_schedule(teams, team_ids, schedule, D, save_path="results/NL4_evaluation.txt")

if __name__ == "__main__":
    main()
