# LLM-Guided Optimization for the Travelling Tournament Problem (TTP)

LLM-guided heuristic optimization framework for the Travelling Tournament Problem (TTP) with constraint-aware move selection and benchmark evaluation.

---

## 📌 Overview

The **Travelling Tournament Problem (TTP)** is a combinatorial optimization problem where:

- Each team plays every other team twice (Double Round Robin).
- Total travel distance must be minimized.
- Fairness constraints such as CA3 (capacity) and SE1 (separation) must be satisfied.

This project proposes a **structured, LLM-driven heuristic optimization framework** that integrates a Large Language Model (LLM) into a controlled local-search loop for explainable and adaptive optimization.

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```


### 2️⃣ Set API Key

Create a .env file in the root directory and give OPENAI_API_KEY = "the key you have for openai".

### 3️⃣ Run the Optimizer
```bash
python src/heuristic_optimizer.py
```

We also have an optimizer which runs which does the optimization without respecting any of the constraints and can be run by:
```bash
python src/no_constraints_optimizer.py
```

## 🧠 Key Idea

Instead of allowing the LLM to freely generate entire schedules (which often breaks structural constraints), we:

1. Build a baseline Double Round Robin schedule.
2. Provide a summarized state to the LLM.
3. Allow the LLM to select exactly **one move** from a predefined move set.
4. Apply the move safely.
5. Accept the move only if it improves the objective.
6. Repeat iteratively.

This ensures feasibility while leveraging LLM reasoning.

---

## 🎯 Objective Function

We minimize total travel distance:

\[
f(S) = \sum_{i=1}^{n} \sum_{(h,a) \in S_i} D_{ha}
\]

Where:

- \( S \) is a feasible tournament schedule.
- \( S_i \) represents games played by team \( i \).
- \( D_{ha} \) denotes the travel distance between home team \( h \) and away team \( a \).

Constraint violations are penalized in the scoring function.

---

## 🏗 Architecture





Core modules:

- `heuristic_agent.py` – LLM decision logic
- `move_executor.py` – Safe move implementations
- `heuristic_optimizer.py` – Iterative improvement loop
- `parser.py` – XML instance reader
- `evaluator.py` – Constraint and travel evaluation

---

## 🔧 Move Catalogue

The LLM is restricted to the following moves:

- `swap_games_between_rounds`
- `flip_home_away`
- `rotate_rounds`
- `rotate_team_labels`
- `transpose_home_away_for_team`

All moves preserve Double Round Robin structure.

---

## 📊 Benchmark Evaluation

Evaluated on **RobinX benchmark datasets**:

- NL4
- NL6
- NL8
- NL10

RobinX provides well-established benchmark solutions representing near-optimal travel distances.  
We compare our optimized schedules directly against these ideal values.

---


### Methodology

## 1️⃣ Initial Fully-Generative LLM Approach

In the initial phase of this work, we experimented with a fully generative LLM approach.

The LLM was provided with:

- The complete Double Round Robin (DRR) schedule

- Travel objective description

- CA3 (max 3 consecutive home/away games) constraint

- SE1 separation constraints

- Natural language instructions to reduce travel while maintaining feasibility

The model was asked to directly modify the schedule to improve feasibility and reduce total travel.

# Issues Observed

- Frequent violation of DRR structure

- Broken home/away balance

- Duplicated or missing matchups

- Ambiguous modifications difficult to parse programmatically

- Poor travel improvement

This approach lacked structural guarantees and resulted in unstable optimization behavior.

## 2️⃣ Final Move-Based LLM-Guided Optimization Framework

Instead of allowing the LLM to modify schedules freely:

- The LLM selects exactly one move from a predefined move catalog

- All moves are programmatically validated

- Feasibility is enforced at every iteration

- Only improvements are accepted




## Results

### 📌 Results of Initial Fully Generative Approach
1. Travel distances were significantly higher than RobinX benchmarks.

2. Frequent CA3 violations occurred.

3. Schedules often became infeasible.

4. No consistent convergence behavior was observed.

This confirmed that unconstrained LLM schedule modification is unreliable for structured combinatorial optimization.


### 📌 Results of Final Move-Based LLM Framework

1. Significant improvement over initial approach.

2. CA3 violations were reduced in most datasets.

3. Feasible zero-violation schedules achieved for smaller instances (NL4).

4. Performance improves with stronger models (GPT-5.1 vs GPT-4o-mini).

5. Still above RobinX ideal scores, but substantially closer than initial approach.


## Key Takeaways
1. Significant improvement over initial approach.

2. CA3 violations were reduced in most datasets.

3. Feasible zero-violation schedules achieved for smaller instances (NL4).

4. Performance improves with stronger models (GPT-5.1 vs GPT-4o-mini).

5. Still above RobinX ideal scores, but substantially closer than initial approach.
