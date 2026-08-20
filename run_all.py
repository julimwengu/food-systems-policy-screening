
from __future__ import annotations

import json
import math
import shutil
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import openpyxl
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs"
OUT_TABLES = OUT_DIR / "tables"
OUT_FIGS = OUT_DIR / "figures"

SEED = 26030713
T = 10
DISCOUNT = 0.95
N_GROUP_DRAWS = 100
N_GLOBAL_DRAWS = 1000

OUTCOMES = [
    "Cost of healthy diet",
    "Cannot afford healthy diet",
    "Prevalence of undernourishment",
    "Minimum dietary diversity, women",
    "Food system emissions",
    "Emissions intensity",
    "Social protection coverage",
]
POLICIES = [
    "Social protection policy",
    "Infrastructure investment",
    "Environmental and research-oriented policy",
]

DIRECT_INDICATORS = {
    "Cost of healthy diet": "Cost of a healthy diet",
    "Cannot afford healthy diet": "Percent of the population who cannot afford a healthy diet",
    "Prevalence of undernourishment": "Prevalence of undernourishment (SDG 2.1.1)",
    "Minimum dietary diversity, women": "MDD-W: Minimum Dietary Diversity for Women (SDG 2.2.4)",
    "Food system emissions": "Agri-food systems greenhouse gas emissions",
    "Social protection coverage": "Social protection coverage",
}
INTENSITY_INDICATORS = [
    "Greenhouse gas emissions intensity for cereals (excluding rice)",
    "Greenhouse gas emissions intensity for rice",
    "Greenhouse gas emissions intensity for beef",
    "Greenhouse gas emissions intensity for cow's milk",
]


@dataclass
class ModelInputs:
    A: np.ndarray
    D: np.ndarray
    W: np.ndarray
    B: np.ndarray
    q: np.ndarray
    r: np.ndarray
    x0: np.ndarray
    lambda_interaction: float


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_emissions_intensity_composite(fsci: pd.DataFrame) -> pd.DataFrame:
    d = fsci.loc[fsci["Indicator"].isin(INTENSITY_INDICATORS)].copy()
    d["Value"] = safe_numeric(d["Value"])
    d = d.dropna(subset=["ISO3", "Start Year", "Value"])
    d["z_indicator"] = d.groupby("Indicator")["Value"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else np.nan
    )
    group_cols = ["Country", "ISO3", "Region", "Start Year", "End Year"]
    comp = (
        d.groupby(group_cols, as_index=False)["z_indicator"]
        .mean()
        .rename(columns={"z_indicator": "Value"})
    )
    comp["Indicator"] = "Emissions intensity composite"
    return comp


def within_ar1_estimate(
    data: pd.DataFrame,
    value_col: str = "Value",
    entity: str = "ISO3",
    time: str = "Start Year",
) -> Tuple[float, float, int, int]:
    """Within-country AR(1) coefficient and conventional OLS standard error."""
    d = data[[entity, time, value_col]].copy()
    d[value_col] = safe_numeric(d[value_col])
    d = d.dropna().sort_values([entity, time])
    d["lag"] = d.groupby(entity)[value_col].shift(1)
    d = d.dropna()
    if d.empty:
        return np.nan, np.nan, 0, 0
    d["y_dm"] = d[value_col] - d.groupby(entity)[value_col].transform("mean")
    d["lag_dm"] = d["lag"] - d.groupby(entity)["lag"].transform("mean")
    d = d.dropna()
    x = d["lag_dm"].to_numpy(float)
    y = d["y_dm"].to_numpy(float)
    denom = float(x @ x)
    if denom <= 0:
        return np.nan, np.nan, int(len(d)), int(d[entity].nunique())
    beta = float((x @ y) / denom)
    resid = y - beta * x
    dof = max(len(y) - 1, 1)
    sigma2 = float((resid @ resid) / dof)
    se = math.sqrt(sigma2 / denom)
    return beta, se, int(len(d)), int(d[entity].nunique())


def prepare_panel_and_coverage(fsci: pd.DataFrame, intensity: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for outcome, indicator in DIRECT_INDICATORS.items():
        d = fsci.loc[fsci["Indicator"] == indicator, ["Country", "ISO3", "Region", "Start Year", "Value"]].copy()
        d["Value"] = safe_numeric(d["Value"])
        d = d.dropna(subset=["ISO3", "Start Year", "Value"])
        d["Outcome"] = outcome
        frames.append(d)
    d = intensity[["Country", "ISO3", "Region", "Start Year", "Value"]].copy()
    d = d.dropna(subset=["ISO3", "Start Year", "Value"])
    d["Outcome"] = "Emissions intensity"
    frames.append(d)
    panel = pd.concat(frames, ignore_index=True)
    panel["Start Year"] = panel["Start Year"].astype(int)
    panel = panel.drop_duplicates(["ISO3", "Start Year", "Outcome"])

    rows = []
    for iso3, g in panel.groupby("ISO3", sort=True):
        first_year = int(g["Start Year"].min())
        last_year = int(g["Start Year"].max())
        span_years = last_year - first_year + 1
        counts = g.groupby("Outcome").size().reindex(OUTCOMES, fill_value=0)
        total_obs = int(len(g))
        rows.append(
            {
                "Country": str(g["Country"].dropna().iloc[0]) if g["Country"].notna().any() else "",
                "ISO3": iso3,
                "Region": str(g["Region"].dropna().iloc[0]) if g["Region"].notna().any() else "",
                "First year": first_year,
                "Last year": last_year,
                "Distinct years observed": int(g["Start Year"].nunique()),
                "Outcomes represented (of 7)": int((counts > 0).sum()),
                "Total outcome-year observations": total_obs,
                "Coverage rate across country-year-outcome cells (%)": round(
                    100 * total_obs / (span_years * len(OUTCOMES)), 1
                ),
                **{f"{o} observations": int(counts[o]) for o in OUTCOMES},
            }
        )
    coverage = pd.DataFrame(rows).sort_values(["Country", "ISO3"]).reset_index(drop=True)

    summary_rows = []
    for outcome in OUTCOMES:
        g = panel.loc[panel["Outcome"] == outcome]
        summary_rows.append(
            {
                "Outcome": outcome,
                "Countries": int(g["ISO3"].nunique()),
                "Country-year observations": int(len(g)),
                "First year": int(g["Start Year"].min()),
                "Last year": int(g["Start Year"].max()),
            }
        )
    indicator_summary = pd.DataFrame(summary_rows)
    return coverage, indicator_summary


def load_baseline_inputs(fsci: pd.DataFrame, interaction: pd.DataFrame, intensity: pd.DataFrame) -> Tuple[ModelInputs, pd.DataFrame]:
    estimates = []
    for outcome, indicator in DIRECT_INDICATORS.items():
        sub = fsci.loc[fsci["Indicator"] == indicator]
        beta, se, nobs, ncountries = within_ar1_estimate(sub)
        estimates.append([outcome, indicator, beta, se, nobs, ncountries])
    beta, se, nobs, ncountries = within_ar1_estimate(intensity)
    estimates.append(
        ["Emissions intensity", "Composite of four standardized GHG-intensity series", beta, se, nobs, ncountries]
    )
    est = pd.DataFrame(
        estimates,
        columns=[
            "Outcome",
            "Series used",
            "Within-country AR(1) estimate",
            "Conventional standard error",
            "Lagged observations",
            "Countries contributing lagged observations",
        ],
    ).set_index("Outcome").reindex(OUTCOMES).reset_index()

    persistence = np.array(
        [
            0.95,  # raw estimate > 1; capped ex ante
            float(est.loc[est["Outcome"] == "Cannot afford healthy diet", "Within-country AR(1) estimate"].iloc[0]),
            float(est.loc[est["Outcome"] == "Prevalence of undernourishment", "Within-country AR(1) estimate"].iloc[0]),
            0.75,  # no lagged MDD-W observations
            float(est.loc[est["Outcome"] == "Food system emissions", "Within-country AR(1) estimate"].iloc[0]),
            float(est.loc[est["Outcome"] == "Emissions intensity", "Within-country AR(1) estimate"].iloc[0]),
            float(est.loc[est["Outcome"] == "Social protection coverage", "Within-country AR(1) estimate"].iloc[0]),
        ],
        dtype=float,
    )
    persistence = np.clip(persistence, 0.0, 0.98)

    interaction = interaction.rename(columns=lambda c: "Indicator" if "Unnamed" in str(c) else c)
    reduced = interaction.set_index("Indicator").loc[OUTCOMES, OUTCOMES].astype(float)
    lambda_interaction = 0.60
    W = reduced.to_numpy(float)
    D = np.diag(persistence)
    # W is stored in source-oriented form: rows transmit and columns receive.
    # States are column vectors, so the transition operator uses W.T.
    A = D + lambda_interaction * W.T

    B = np.array(
        [
            [0.00, 0.18, 0.00],
            [0.22, 0.12, 0.00],
            [0.20, 0.00, 0.00],
            [0.08, 0.00, 0.00],
            [0.00, 0.00, 0.15],
            [0.00, 0.00, 0.24],
            [0.18, 0.00, 0.00],
        ],
        dtype=float,
    )
    q = np.array([2.4, 2.8, 2.0, 1.1, 1.8, 1.0, 1.0], dtype=float)
    r = np.array([0.55, 1.00, 0.80], dtype=float)
    x0 = np.array([-2.2, -2.0, -1.6, -1.1, -2.4, -1.2, -0.9], dtype=float)

    return ModelInputs(A=A, D=D, W=W, B=B, q=q, r=r, x0=x0, lambda_interaction=lambda_interaction), est


def solve_shortfall_problem(
    A: np.ndarray,
    B: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    x0: np.ndarray,
    T: int = T,
    discount: float = DISCOUNT,
    start: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """
    Solve the finite-horizon convex policy-screening problem.

    State x is signed performance relative to target. Negative values are
    shortfalls. The loss uses max(-x, 0)^2, so meeting or exceeding a target
    is not penalized. Policy effort is constrained to be non-negative.
    """
    n, m = A.shape[0], B.shape[1]
    xbase = np.zeros((T + 1, n))
    xbase[0] = x0
    for t in range(T):
        xbase[t + 1] = A @ xbase[t]

    G = np.zeros((T + 1, n, T * m))
    for t in range(T):
        G[t + 1] = A @ G[t]
        G[t + 1, :, t * m : (t + 1) * m] += B

    disc = discount ** np.arange(T + 1)

    def objective_and_gradient(flat_u: np.ndarray) -> Tuple[float, np.ndarray]:
        x = xbase + np.einsum("tnk,k->tn", G, flat_u)
        neg = np.minimum(x, 0.0)
        value = float(np.sum(disc[:, None] * neg**2 * q[None, :]))
        grad = np.einsum("tnk,tn->k", G, 2.0 * disc[:, None] * neg * q[None, :])

        u = flat_u.reshape(T, m)
        value += float(np.sum(disc[:T, None] * u**2 * r[None, :]))
        grad = grad.reshape(T, m)
        grad += 2.0 * disc[:T, None] * u * r[None, :]
        return value, grad.ravel()

    if start is None:
        start_flat = np.zeros(T * m)
    else:
        start_flat = np.asarray(start, dtype=float).reshape(-1)
    result = minimize(
        objective_and_gradient,
        start_flat,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * (T * m),
        options={"maxiter": 250, "ftol": 1e-10, "gtol": 1e-7},
    )
    u = result.x.reshape(T, m)
    x = xbase + np.einsum("tnk,k->tn", G, result.x)
    return x, u, float(result.fun), bool(result.success)


def remaining_shortfall(x: np.ndarray) -> np.ndarray:
    return np.maximum(-x, 0.0)


def policy_metrics(x: np.ndarray, u: np.ndarray) -> Dict[str, float | str | bool]:
    cumulative = u.sum(axis=0)
    total = float(cumulative.sum())
    affordability_share = float((cumulative[0] + cumulative[1]) / total) if total > 0 else np.nan
    initial_top = POLICIES[int(np.argmax(u[0]))]
    cumulative_top = POLICIES[int(np.argmax(cumulative))]
    final = remaining_shortfall(x[-1])
    initial = remaining_shortfall(x[0])
    return {
        "Initial top policy": initial_top,
        "Cumulative top policy": cumulative_top,
        "Affordability-oriented cumulative effort share": affordability_share,
        "All seven final shortfalls below initial": bool(np.all(final < initial)),
        "Final mean remaining shortfall": float(final.mean()),
        "Final affordability mean shortfall": float(final[[0, 1]].mean()),
        "Final nutrition mean shortfall": float(final[[2, 3]].mean()),
        "Final environmental mean shortfall": float(final[[4, 5]].mean()),
        "Environmental policy cumulative effort": float(cumulative[2]),
    }


def random_log_multiplier(rng: np.random.Generator, low: float, high: float, size) -> np.ndarray:
    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


def draw_inputs(
    base: ModelInputs,
    rng: np.random.Generator,
    groups: Iterable[str],
    persistence_estimates: pd.DataFrame,
) -> ModelInputs:
    groups = set(groups)
    B = base.B.copy()
    W = base.W.copy()
    lam = float(base.lambda_interaction)
    D = base.D.copy()
    q = base.q.copy()
    r = base.r.copy()
    x0 = base.x0.copy()

    if "Policy-impact matrix" in groups:
        mask = B > 0
        mult = rng.triangular(0.75, 1.0, 1.25, size=B.shape)
        B[mask] *= mult[mask]

    if "Interaction matrix" in groups:
        mask = W != 0
        mult = rng.triangular(0.50, 1.0, 1.50, size=W.shape)
        W[mask] *= mult[mask]
        lam = float(rng.uniform(0.40, 0.80))

    if "Persistence parameters" in groups:
        diag = np.diag(D).copy()
        se_map = persistence_estimates.set_index("Outcome")["Conventional standard error"].to_dict()
        for i, outcome in enumerate(OUTCOMES):
            if outcome == "Cost of healthy diet":
                diag[i] = rng.triangular(0.88, 0.95, 0.98)
            elif outcome == "Minimum dietary diversity, women":
                diag[i] = rng.uniform(0.60, 0.90)
            else:
                sigma = max(float(se_map.get(outcome, np.nan) or 0.0) * 2.0, 0.03)
                diag[i] = np.clip(rng.normal(diag[i], sigma), 0.05, 0.98)
        D = np.diag(diag)

    if "Planner weights" in groups:
        q *= random_log_multiplier(rng, 0.50, 1.50, q.shape)

    if "Policy costs" in groups:
        r *= random_log_multiplier(rng, 0.50, 1.50, r.shape)

    if "Initial conditions" in groups:
        # Preserve the direction of each shortfall while varying its magnitude.
        x0 *= rng.uniform(0.70, 1.30, size=x0.shape)

    # W rows transmit and columns receive; column-vector dynamics require W.T.
    A = D + lam * W.T
    return ModelInputs(A=A, D=D, W=W, B=B, q=q, r=r, x0=x0, lambda_interaction=lam)


def run_sensitivity(
    base: ModelInputs,
    estimates: pd.DataFrame,
    baseline_u: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    groups = [
        "Policy-impact matrix",
        "Interaction matrix",
        "Persistence parameters",
        "Planner weights",
        "Policy costs",
        "Initial conditions",
    ]
    rows = []
    for group in groups:
        for draw in range(N_GROUP_DRAWS):
            inp = draw_inputs(base, rng, [group], estimates)
            x, u, loss, ok = solve_shortfall_problem(
                inp.A, inp.B, inp.q, inp.r, inp.x0, start=baseline_u
            )
            row = {
                "Sensitivity block": group,
                "Draw": draw + 1,
                "Objective value": loss,
                "Solver converged": ok,
                **policy_metrics(x, u),
            }
            rows.append(row)
    oat = pd.DataFrame(rows)

    global_rows = []
    for draw in range(N_GLOBAL_DRAWS):
        inp = draw_inputs(base, rng, groups, estimates)
        x, u, loss, ok = solve_shortfall_problem(
            inp.A, inp.B, inp.q, inp.r, inp.x0, start=baseline_u
        )
        global_rows.append(
            {
                "Draw": draw + 1,
                "Objective value": loss,
                "Solver converged": ok,
                **policy_metrics(x, u),
            }
        )
    global_df = pd.DataFrame(global_rows)

    summary_rows = []
    for block, d in list(oat.groupby("Sensitivity block")) + [("All parameter groups jointly", global_df)]:
        affordability = d["Affordability-oriented cumulative effort share"]
        summary_rows.append(
            {
                "Sensitivity block": block,
                "Draws": len(d),
                "Solver convergence (%)": round(100 * d["Solver converged"].mean(), 1),
                "Affordability-oriented effort share, median": float(affordability.median()),
                "Affordability-oriented effort share, 5th percentile": float(affordability.quantile(0.05)),
                "Affordability-oriented effort share, 95th percentile": float(affordability.quantile(0.95)),
                "Social protection or infrastructure ranks first (%)": round(
                    100
                    * d["Cumulative top policy"].isin(
                        ["Social protection policy", "Infrastructure investment"]
                    ).mean(),
                    1,
                ),
                "All seven shortfalls improve (%)": round(
                    100 * d["All seven final shortfalls below initial"].mean(), 1
                ),
                "Median final mean shortfall": float(d["Final mean remaining shortfall"].median()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return oat, global_df, summary


def run_structural_robustness(base: ModelInputs, baseline_u: np.ndarray) -> pd.DataFrame:
    """Reviewer-requested diagnostic scenarios that change maintained model structure."""
    scenarios = []

    def add(name: str, description: str, A=None, B=None, q=None, r=None, x0=None):
        xs, us, loss, ok = solve_shortfall_problem(
            base.A if A is None else A,
            base.B if B is None else B,
            base.q if q is None else q,
            base.r if r is None else r,
            base.x0 if x0 is None else x0,
            start=baseline_u,
        )
        cumulative = us.sum(axis=0)
        metrics = policy_metrics(xs, us)
        scenarios.append({
            "Scenario": name,
            "Change from baseline": description,
            "Solver converged": ok,
            "Objective value": loss,
            "Social protection cumulative effort": float(cumulative[0]),
            "Infrastructure cumulative effort": float(cumulative[1]),
            "Environmental policy cumulative effort": float(cumulative[2]),
            **metrics,
        })

    add("Baseline", "Maintained baseline calibration")
    add("Initial gaps 30% smaller", "All initial shortfall magnitudes multiplied by 0.70", x0=0.70 * base.x0)
    add("Initial gaps 30% larger", "All initial shortfall magnitudes multiplied by 1.30", x0=1.30 * base.x0)
    add("Equal policy costs", "R diagonal set to 1.00 for all three policy blocs", r=np.ones_like(base.r))

    B_weak = base.B.copy()
    B_weak[[1, 2, 3, 6], 0] *= 0.25
    add("Weak social-protection direct effects", "Social-protection loadings on affordability, nutrition, and coverage reduced by 75%", B=B_weak)

    B_no_coverage = base.B.copy()
    B_no_coverage[6, 0] = 0.0
    q_no_coverage = base.q.copy()
    q_no_coverage[6] = 0.0
    add("Exclude social-protection coverage", "Coverage removed from the objective and its direct policy loading set to zero", B=B_no_coverage, q=q_no_coverage)

    W_weak_aff = base.W.copy()
    W_weak_aff[0:2, 2:4] *= 0.25
    add("Weak affordability-to-nutrition links", "Affordability-to-nutrition interaction links reduced by 75%", A=base.D + base.lambda_interaction * W_weak_aff.T)

    W_env = base.W.copy()
    W_env[4, 0] = 0.08
    W_env[5, 1] = 0.08
    add("Environmental-to-affordability feedbacks", "Adds positive emissions and emissions-intensity feedbacks to diet cost and affordability", A=base.D + base.lambda_interaction * W_env.T)

    W_sparse = base.W.copy()
    W_sparse[np.abs(W_sparse) < 0.10] = 0.0
    add("Sparse alternative topology", "Removes directed links below 0.10 in the unscaled interaction matrix", A=base.D + base.lambda_interaction * W_sparse.T)

    return pd.DataFrame(scenarios)


def write_baseline_outputs(base: ModelInputs, estimates: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, float | str | bool]]:
    x, u, loss, ok = solve_shortfall_problem(base.A, base.B, base.q, base.r, base.x0)
    if not ok:
        raise RuntimeError("Baseline optimizer did not converge.")

    shortfall = remaining_shortfall(x)
    traj = pd.DataFrame(shortfall, columns=OUTCOMES)
    traj.insert(0, "Period", np.arange(T + 1))
    traj.to_csv(OUT_TABLES / "baseline_remaining_shortfalls.csv", index=False)

    signed = pd.DataFrame(x, columns=OUTCOMES)
    signed.insert(0, "Period", np.arange(T + 1))
    signed.to_csv(OUT_TABLES / "baseline_signed_states.csv", index=False)

    policy = pd.DataFrame(u, columns=POLICIES)
    policy.insert(0, "Period", np.arange(T))
    policy.to_csv(OUT_TABLES / "baseline_policy_paths.csv", index=False)

    # Figure 1
    plt.figure(figsize=(9.0, 5.6))
    for outcome in OUTCOMES:
        plt.plot(traj["Period"], traj[outcome], label=outcome)
    plt.xlabel("Period")
    plt.ylabel("Remaining shortfall (standardized units)")
    plt.title("Baseline model-implied remaining shortfalls")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "figure_1_baseline_outcome_trajectories.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 2
    plt.figure(figsize=(8.4, 5.2))
    for p in POLICIES:
        plt.plot(policy["Period"], policy[p], label=p)
    plt.xlabel("Period")
    plt.ylabel("Model-implied policy effort")
    plt.title("Baseline model-implied policy paths")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "figure_2_policy_paths.png", dpi=300)
    plt.close()

    # Figure 3: all outcomes across weight scenarios
    scenarios = {
        "Baseline": base.q,
        "Affordability priority": np.array([3.6, 4.0, 1.8, 1.0, 1.5, 0.9, 0.9]),
        "Nutrition priority": np.array([2.0, 2.2, 3.8, 1.6, 1.5, 0.9, 1.0]),
        "Environment priority": np.array([1.8, 2.0, 1.7, 1.0, 4.0, 2.2, 0.9]),
    }
    scenario_rows = []
    matrix = []
    for name, q in scenarios.items():
        xs, us, obj, conv = solve_shortfall_problem(base.A, base.B, q, base.r, base.x0, start=u)
        final = remaining_shortfall(xs[-1])
        matrix.append(final)
        for outcome, value in zip(OUTCOMES, final):
            scenario_rows.append({"Scenario": name, "Outcome": outcome, "Final remaining shortfall": value})
    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(OUT_TABLES / "figure_3_final_shortfalls_by_scenario.csv", index=False)
    mat = np.array(matrix)

    plt.figure(figsize=(9.2, 4.8))
    im = plt.imshow(mat, aspect="auto")
    plt.colorbar(im, label="Final remaining shortfall")
    plt.yticks(np.arange(len(scenarios)), list(scenarios.keys()))
    plt.xticks(np.arange(len(OUTCOMES)), OUTCOMES, rotation=45, ha="right")
    plt.title("Model-implied final shortfalls under alternative planner weights")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "figure_3_priority_scenarios.png", dpi=300)
    plt.close()

    # Figure 4: all seven outcomes, with and without interactions
    no_interaction_A = base.D.copy()
    x_no, u_no, _, _ = solve_shortfall_problem(
        no_interaction_A, base.B, base.q, base.r, base.x0, start=u
    )
    s_no = remaining_shortfall(x_no)
    comp = pd.DataFrame({"Period": np.arange(T + 1)})
    for i, outcome in enumerate(OUTCOMES):
        comp[f"{outcome} - full model"] = shortfall[:, i]
        comp[f"{outcome} - no interactions"] = s_no[:, i]
    comp.to_csv(OUT_TABLES / "figure_4_interaction_comparison.csv", index=False)

    fig, axes = plt.subplots(4, 2, figsize=(9.0, 10.5), sharex=True)
    axes = axes.ravel()
    for i, outcome in enumerate(OUTCOMES):
        axes[i].plot(np.arange(T + 1), shortfall[:, i], label="Full model")
        axes[i].plot(np.arange(T + 1), s_no[:, i], linestyle="--", label="No interactions")
        axes[i].set_title(outcome, fontsize=9)
        axes[i].set_ylabel("Shortfall")
    axes[-1].axis("off")
    axes[0].legend(frameon=False)
    for ax in axes[:-1]:
        ax.set_xlabel("Period")
    fig.suptitle("Model-implied shortfalls with and without cross-outcome interactions")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_FIGS / "figure_4_all_outcomes_interactions.png", dpi=300)
    plt.close(fig)

    # Figure 5: objective decomposition
    disc = DISCOUNT ** np.arange(T + 1)
    gap_loss = float(np.sum(disc[:, None] * shortfall**2 * base.q[None, :]))
    policy_loss = float(np.sum(disc[:T, None] * u**2 * base.r[None, :]))
    welfare = pd.DataFrame(
        {
            "Component": ["Remaining outcome shortfalls", "Policy effort"],
            "Discounted loss": [gap_loss, policy_loss],
        }
    )
    welfare.to_csv(OUT_TABLES / "figure_5_objective_decomposition.csv", index=False)
    plt.figure(figsize=(6.4, 4.8))
    plt.bar(welfare["Component"], welfare["Discounted loss"])
    plt.ylabel("Discounted loss")
    plt.title("Baseline objective decomposition")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "figure_5_welfare_decomposition.png", dpi=300)
    plt.close()

    # Tables
    pd.DataFrame({"Outcome": OUTCOMES, "Persistence coefficient": np.diag(base.D)}).to_csv(
        OUT_TABLES / "table_S1_persistence_parameters.csv", index=False
    )
    table_s2 = pd.DataFrame(base.W, index=OUTCOMES, columns=OUTCOMES)
    table_s2.index.name = "Transmitting outcome"
    table_s2.columns.name = "Receiving outcome"
    table_s2.to_csv(OUT_TABLES / "table_S2_unscaled_interaction_matrix.csv")

    table_s3 = pd.DataFrame(base.A, index=OUTCOMES, columns=OUTCOMES)
    table_s3.index.name = "Receiving outcome"
    table_s3.columns.name = "Transmitting outcome"
    table_s3.to_csv(OUT_TABLES / "table_S3_combined_system_matrix.csv")
    pd.DataFrame(base.B, index=OUTCOMES, columns=POLICIES).to_csv(
        OUT_TABLES / "table_S4_policy_impact_matrix.csv"
    )
    pd.DataFrame({"Outcome": OUTCOMES, "Planner weight": base.q}).to_csv(
        OUT_TABLES / "table_S5_planner_weights.csv", index=False
    )
    pd.DataFrame({"Policy bloc": POLICIES, "Policy cost": base.r}).to_csv(
        OUT_TABLES / "table_S6_policy_costs.csv", index=False
    )
    estimates.to_csv(OUT_TABLES / "persistence_estimation_details.csv", index=False)

    metrics = policy_metrics(x, u)
    metrics["Objective value"] = loss
    metrics["Solver converged"] = ok
    return x, u, metrics


def write_parameter_registry(base: ModelInputs, estimates: pd.DataFrame) -> None:
    rows = []
    est_map = estimates.set_index("Outcome")
    for i, outcome in enumerate(OUTCOMES):
        status = "Estimated then capped" if outcome == "Cost of healthy diet" else (
            "Calibrated fallback" if outcome == "Minimum dietary diversity, women" else "Estimated"
        )
        basis = (
            f"Within-country AR(1) raw estimate {est_map.loc[outcome, 'Within-country AR(1) estimate']:.3f}; capped at 0.95"
            if outcome == "Cost of healthy diet"
            else (
                "No usable lagged observations; baseline set to 0.75"
                if outcome == "Minimum dietary diversity, women"
                else f"Within-country AR(1), SE {est_map.loc[outcome, 'Conventional standard error']:.3f}"
            )
        )
        rows.append(
            {
                "Parameter block": "Persistence",
                "Element": outcome,
                "Baseline value": float(base.D[i, i]),
                "Identification status": status,
                "Basis": basis,
                "Sensitivity range or rule": (
                    "Triangular 0.88-0.98"
                    if outcome == "Cost of healthy diet"
                    else ("Uniform 0.60-0.90" if outcome == "Minimum dietary diversity, women" else "Truncated normal; SD=max(2×SE,0.03)")
                ),
            }
        )
    rows += [
        {
            "Parameter block": "Interaction matrix",
            "Element": "Non-zero directed links",
            "Baseline value": "See Table S2",
            "Identification status": "Calibrated from qualitative network topology",
            "Basis": "FSCI directed-link structure; row-normalized source matrix. W rows transmit and columns receive; column-vector dynamics use W.T.",
            "Sensitivity range or rule": "Each non-zero link multiplied by triangular 0.50-1.50",
        },
        {
            "Parameter block": "Interaction scale",
            "Element": "lambda",
            "Baseline value": base.lambda_interaction,
            "Identification status": "Calibrated",
            "Basis": "Chosen within stable range to preserve moderate spillovers",
            "Sensitivity range or rule": "Uniform 0.40-0.80",
        },
        {
            "Parameter block": "Policy-impact matrix",
            "Element": "Non-zero direct effects",
            "Baseline value": "See Table S4",
            "Identification status": "Calibrated assumption",
            "Basis": "Direction and relative magnitude aligned with policy-outcome mapping; not causal estimates",
            "Sensitivity range or rule": "Each non-zero effect multiplied by triangular 0.75-1.25",
        },
        {
            "Parameter block": "Planner weights",
            "Element": "Diagonal entries",
            "Baseline value": "See Table S5",
            "Identification status": "Normative assumption",
            "Basis": "Illustrative baseline priorities",
            "Sensitivity range or rule": "Independent log-uniform multipliers 0.50-1.50",
        },
        {
            "Parameter block": "Policy costs",
            "Element": "Diagonal entries",
            "Baseline value": "See Table S6",
            "Identification status": "Calibrated assumption",
            "Basis": "Relative implementation-cost index in normalized units",
            "Sensitivity range or rule": "Independent log-uniform multipliers 0.50-1.50",
        },
        {
            "Parameter block": "Initial conditions",
            "Element": "Seven signed target gaps",
            "Baseline value": "See model_diagnostics.json",
            "Identification status": "Scenario assumption",
            "Basis": "Illustrative below-target starting configuration",
            "Sensitivity range or rule": "Independent uniform multipliers 0.70-1.30; structural scenarios at 0.70 and 1.30",
        },
    ]
    pd.DataFrame(rows).to_csv(OUT_TABLES / "calibration_and_identification_registry.csv", index=False)


def plot_sensitivity_tornado(summary: pd.DataFrame, baseline_share: float) -> None:
    d = summary.loc[summary["Sensitivity block"] != "All parameter groups jointly"].copy()
    d["low_delta"] = d["Affordability-oriented effort share, 5th percentile"] - baseline_share
    d["high_delta"] = d["Affordability-oriented effort share, 95th percentile"] - baseline_share
    d = d.sort_values("high_delta" - d["low_delta"] if False else "Sensitivity block")
    y = np.arange(len(d))
    plt.figure(figsize=(8.2, 5.0))
    for i, row in d.reset_index(drop=True).iterrows():
        plt.plot([100 * row["low_delta"], 100 * row["high_delta"]], [i, i], linewidth=8)
        plt.plot(0, i, marker="|", markersize=13)
    plt.axvline(0, linewidth=1)
    plt.yticks(y, d.reset_index(drop=True)["Sensitivity block"])
    plt.xlabel("Change from baseline affordability-oriented effort share (percentage points)")
    plt.title("One-at-a-time sensitivity of the model-implied policy mix")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "supplementary_figure_S1_sensitivity_tornado.png", dpi=300)
    plt.close()


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    fsci = pd.read_csv(RAW_DIR / "fsd-fsci-full-export-2026-01-09.csv", low_memory=False)
    interaction = pd.read_excel(RAW_DIR / "Food systems interactions matrix_row_normalized.xlsx")
    intensity = build_emissions_intensity_composite(fsci)

    selected = pd.DataFrame(
        [
            ["Cost of healthy diet", "Cost of a healthy diet", "Affordability", "Higher normalized values indicate lower cost."],
            ["Cannot afford healthy diet", "Percent of the population who cannot afford a healthy diet", "Food access", "Higher normalized values indicate a smaller unaffordable share."],
            ["Prevalence of undernourishment", "Prevalence of undernourishment (SDG 2.1.1)", "Food security", "Higher normalized values indicate less undernourishment."],
            ["Minimum dietary diversity, women", "MDD-W: Minimum Dietary Diversity for Women (SDG 2.2.4)", "Nutrition", "Higher normalized values indicate greater dietary diversity."],
            ["Food system emissions", "Agri-food systems greenhouse gas emissions", "Environmental sustainability", "Higher normalized values indicate lower emissions."],
            ["Emissions intensity", "Composite of four GHG-intensity indicators", "Environmental efficiency", "Higher normalized values indicate lower emissions intensity."],
            ["Social protection coverage", "Social protection coverage", "Resilience and equity", "Higher normalized values indicate broader coverage."],
        ],
        columns=["Outcome", "Source series", "Domain", "Interpretation after sign correction and standardization"],
    )
    selected.to_csv(OUT_TABLES / "table_1_selected_indicators.csv", index=False)

    coverage, indicator_summary = prepare_panel_and_coverage(fsci, intensity)
    coverage.to_csv(OUT_TABLES / "table_S7_country_data_coverage.csv", index=False)
    indicator_summary.to_csv(OUT_TABLES / "table_S8_indicator_coverage_summary.csv", index=False)

    base, estimates = load_baseline_inputs(fsci, interaction, intensity)
    x, u, baseline_metrics = write_baseline_outputs(base, estimates)
    write_parameter_registry(base, estimates)

    oat, global_df, summary = run_sensitivity(base, estimates, u)
    oat.to_csv(OUT_TABLES / "one_at_a_time_sensitivity_draws.csv", index=False)
    global_df.to_csv(OUT_TABLES / "global_monte_carlo_draws.csv", index=False)
    summary.to_csv(OUT_TABLES / "table_S9_sensitivity_summary.csv", index=False)
    plot_sensitivity_tornado(summary, float(baseline_metrics["Affordability-oriented cumulative effort share"]))
    structural = run_structural_robustness(base, u)
    structural.to_csv(OUT_TABLES / "table_S11_structural_robustness.csv", index=False)

    diagnostics = {
        "seed": SEED,
        "horizon": T,
        "discount_factor": DISCOUNT,
        "interaction_scale": base.lambda_interaction,
        "interaction_matrix_convention": "W rows transmit and columns receive; with column-vector states the transition operator is A = D + lambda * W.T.",
        "spectral_radius_open_loop": float(max(abs(np.linalg.eigvals(base.A)))),
        "initial_signed_gap_vector": dict(zip(OUTCOMES, base.x0.tolist())),
        "baseline_metrics": baseline_metrics,
        "uncertainty_design": {
            "group_draws": N_GROUP_DRAWS,
            "global_draws": N_GLOBAL_DRAWS,
            "policy_impact_nonzero_multiplier": "triangular(0.75, 1.00, 1.25)",
            "interaction_nonzero_multiplier": "triangular(0.50, 1.00, 1.50)",
            "interaction_scale": "uniform(0.40, 0.80)",
            "planner_weight_multiplier": "log-uniform(0.50, 1.50)",
            "policy_cost_multiplier": "log-uniform(0.50, 1.50)",
            "initial_gap_multiplier": "uniform(0.70, 1.30)",
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "openpyxl": openpyxl.__version__,
        },
        "notes": [
            "Negative signed states denote target shortfalls; reported remaining shortfall is max(-x, 0).",
            "The objective does not penalize meeting or exceeding a target, addressing the symmetric-overshoot problem of an unconstrained LQR.",
            "MDD-W has no usable within-country lagged observations in the supplied export; its persistence is a documented fallback calibration.",
            "The emissions-intensity state is a composite of four within-indicator standardized series.",
            "Sensitivity ranges are structured uncertainty ranges, not sampling confidence intervals for calibrated assumptions.",
        ],
    }
    with open(OUT_DIR / "model_diagnostics.json", "w", encoding="utf-8") as fh:
        json.dump(diagnostics, fh, indent=2)

    manifest = sorted(str(p.relative_to(ROOT)) for p in OUT_DIR.rglob("*") if p.is_file())
    (OUT_DIR / "output_manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"Replication completed. {len(manifest)} output files written to {OUT_DIR}")


if __name__ == "__main__":
    main()
