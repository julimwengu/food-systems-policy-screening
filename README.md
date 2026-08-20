# Replication package: dynamic food-systems policy screening

This package reproduces every numerical table and figure in the revised manuscript and supplementary information. The analysis is deterministic: the master script uses random seed `26030713`.

## Run the full analysis

From the package root:

```bash
python -m pip install -r code/requirements.txt
python code/run_all.py
```

The master script deletes and recreates `outputs/`, then writes all manuscript figures, supplementary figures, parameter tables, country-level coverage tables, sensitivity draws, diagnostics, and an output manifest.

## Contents

- `code/run_all.py`: executable master script.
- `code/requirements.txt`: Python dependencies.
- `data/raw/`: source FSCI export, interaction matrix, and metadata supplied for the analysis.
- `outputs/figures/`: Figures 1–5 and Supplementary Figure S1.
- `outputs/tables/`: main and supplementary tables, raw sensitivity draws, and calibration registry.
- `outputs/model_diagnostics.json`: seed, horizon, discount factor, initial conditions, solver diagnostics, uncertainty design, and baseline summary.
- `outputs/output_manifest.txt`: complete list of generated outputs.

## Identification and calibration

The script estimates outcome persistence with within-country demeaned AR(1) regressions where repeated observations are available. The cost-of-healthy-diet estimate exceeds one and is capped at 0.95 for stability. MDD-W has no usable within-country lagged observations in the supplied export, so its baseline persistence is a documented fallback calibration of 0.75.

The interaction topology comes from the supplied FSCI directed-link matrix. Link magnitudes, the interaction scale, the policy-impact matrix, planner weights, policy costs, initial gaps, and the ten-period horizon are calibrated or normative assumptions rather than causally identified quantities. `outputs/tables/calibration_and_identification_registry.csv` records the status and sensitivity rule for each parameter block.



## Interaction-matrix orientation

The supplied FSCI interaction matrix is stored in **source-oriented form**: rows are transmitting outcomes and columns are receiving outcomes. The model state is a column vector, so the transition operator used in the dynamics is

`A = D + lambda * W.T`

and the state update is `x(t+1) = A @ x(t) + B @ u(t)`. `table_S2_unscaled_interaction_matrix.csv` preserves the source-oriented convention (rows transmit, columns receive), while `table_S3_combined_system_matrix.csv` reports the actual transition operator (rows receive, columns transmit).

## Target treatment

Signed states are measured relative to transformation targets. Negative values are shortfalls. The reported objective penalizes only remaining shortfalls, `max(-x, 0)^2`, and constrains policy effort to be non-negative. Meeting or exceeding a target is therefore not penalized. This finite-horizon convex formulation replaces the symmetric target penalty that would arise in an unconstrained linear-quadratic regulator.

## Sensitivity and uncertainty

The script performs:

- 100 one-at-a-time draws for each of six parameter blocks: policy impacts, interactions, persistence, planner weights, policy costs, and initial conditions.
- 1,000 joint Monte Carlo draws varying all six blocks simultaneously.
- A supplementary tornado plot based on the 5th–95th percentile range of the affordability-oriented share of cumulative policy effort.
- Structural diagnostics that impose equal policy costs, vary the initial gaps, weaken social-protection direct effects, remove social-protection coverage from the objective and direct loading, weaken affordability-to-nutrition links, add environmental-to-affordability feedbacks, and impose a sparser alternative network topology.

These ranges are structured uncertainty intervals for calibrated assumptions; they are not statistical confidence intervals.

## Reproducibility check

A successful run prints the number of generated files and creates `outputs/output_manifest.txt`. All reported solvers should converge. The reference run used Python 3.13.5; the exact interpreter and package versions used for the run are recorded in `outputs/model_diagnostics.json`. The package relies only on the dependencies listed above.
