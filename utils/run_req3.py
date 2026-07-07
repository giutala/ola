"""
run_req3.py
-----------
Requirement 3: best-of-both-worlds bidding for N campaigns.

Algorithm: PrimalDualMultiCampaignAgent — Hedge as primal regret minimiser
(one per campaign, full feedback) coupled to a shared OGD step on the dual
variable lambda.  Conflicts are enforced both by MultiCampaignEnv.round and
by the suppression rule applied to Hedge counterfactual rewards (NB08
pattern).

Two experiments
---------------
A. Stochastic environment (MultiCampaignEnv, NB07 setup) — baseline is the
   fixed LP optimum (compute_clairvoyant_multi), computed once from the
   true win probabilities.

B. Adversarial / non-stationary environment (AdversarialMultiCampaignEnv)
   — baseline is OPT^A, the best FIXED distribution in hindsight (NB08
   cells 8-11), computed per trial from the empirical win probabilities
   of that trial's realised m-sequence (env.empirical_win_probabilities)
   fed into the same compute_clairvoyant_multi LP.  This is the benchmark
   a primal-dual (Hedge+OGD) regret minimiser actually has a provable
   sublinear-regret guarantee against, in both the stochastic and the
   adversarial regime -- the "best-of-both-worlds" property.  Comparing
   against the best DYNAMIC sequence in hindsight instead (a per-round-
   adaptive benchmark) makes regret linear by construction in a "highly
   non-stationary" environment, regardless of how good the agent is.

T=10_000 matches R1 / R2 for a valid cross-requirement regret comparison.
B scaled so rho=0.16 is unchanged.

Call from the notebook: run_req3()
"""

import logging

import numpy as np
import matplotlib.pyplot as plt

from utils.agents import PrimalDualMultiCampaignAgent
from utils.environments import MultiCampaignEnv, AdversarialMultiCampaignEnv
from utils.experiments import (
    compute_clairvoyant_multi,
    plot_budget,
    plot_lambda,
    plot_regret,
    run_primal_dual_trials,
    OUTPUTS_DIR,
)
from utils.req3_config import (
    VALUES, T, BUDGET, N_TRIALS, N_COMPETITORS, CONFLICT_EDGES,
    AVAILABLE_BIDS,
)

logger = logging.getLogger(__name__)

# ── Parameters ──────────────────────────────────────────────────────────────
# VALUES, T, BUDGET, N_TRIALS, N_COMPETITORS, CONFLICT_EDGES, AVAILABLE_BIDS
# all come from utils/req3_config.py, the single source of truth shared with
# precompute_clairvoyant.py. Change them there, not here, so the two scripts
# can never drift apart again.

# Non-stationarity pattern of the adversarial environment.
#   'drift'  : m_t ~ Beta with sinusoidal mean — changes every round
#   'shocks' : piecewise-stationary, distribution changes per block
# Switch this single constant to compare the two regimes.
ENV_MODE = "drift"  # "drift" or "shocks"



def run_req3():
    """Single entry point called from the notebook."""
    logger.info("=" * 60)
    logger.info("Requirement 3 – Best-of-both-worlds bidding")
    logger.info("=" * 60)
    logger.info(
        "Parameters | N=%d T=%d B=%.1f rho=%.4f env_mode=%s conflict_edges=%s",
        len(VALUES), T, BUDGET, BUDGET / T, ENV_MODE, CONFLICT_EDGES
    )

    # ── Factories shared across the two experiments ───────────────────────
    # We freeze a "reference" env (seed=0) only to read Ks and bid_sets,
    # which are deterministic functions of values + AVAILABLE_BIDS and
    # therefore the same for every trial.
    _env_ref_stoch = MultiCampaignEnv(
        values=VALUES, budget=BUDGET, T=T,
        available_bids=AVAILABLE_BIDS, n_competitors=N_COMPETITORS,
        conflict_edges=CONFLICT_EDGES, seed=0,
    )
    BID_SETS = _env_ref_stoch.bid_sets
    KS       = _env_ref_stoch.Ks
    N        = _env_ref_stoch.N

    def make_agent():
        return PrimalDualMultiCampaignAgent(
            N=N, Ks=KS, bid_sets=BID_SETS,
            T=T, budget=BUDGET, values=VALUES,
            conflict_edges=CONFLICT_EDGES,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Experiment A – Stochastic
    # ─────────────────────────────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Experiment A – Stochastic environment")
    logger.info("-" * 60)

    def env_factory_stoch(seed):
        return MultiCampaignEnv(
            values=VALUES, budget=BUDGET, T=T,
            available_bids=AVAILABLE_BIDS, n_competitors=N_COMPETITORS,
            conflict_edges=CONFLICT_EDGES, seed=seed,
        )

    # Stochastic baseline: fixed LP optimum.  Computed once.
    win_probs = _env_ref_stoch.win_probabilities()
    _, opt_stoch = compute_clairvoyant_multi(
        np.array(VALUES), BID_SETS, BUDGET / T, win_probs, CONFLICT_EDGES,
    )
    logger.info("Stochastic clairvoyant | per-round utility = %.4f", opt_stoch)

    res_stoch = run_primal_dual_trials(
        env_factory   = env_factory_stoch,
        agent_factory = make_agent,
        n_trials      = N_TRIALS,
        opt_per_round = opt_stoch,            # Mode B: fixed baseline
        name          = "req3_stochastic",
    )

    # ─────────────────────────────────────────────────────────────────────
    # Experiment B – Adversarial / non-stationary
    # ─────────────────────────────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Experiment B – Non-stationary environment")
    logger.info("-" * 60)

    def env_factory_adv(seed):
        return AdversarialMultiCampaignEnv(
            values=VALUES, budget=BUDGET, T=T,
            available_bids=AVAILABLE_BIDS,
            conflict_edges=CONFLICT_EDGES, seed=seed, mode=ENV_MODE,
        )

    # No opt_per_round given -> run_primal_dual_trials computes OPT^A (best
    # fixed distribution in hindsight) per trial from the empirical win
    # probabilities of that trial's realised m-sequence. See NB08 cells 8-11
    # and the run_primal_dual_trials docstring in experiments.py.
    res_adv = run_primal_dual_trials(
        env_factory   = env_factory_adv,
        agent_factory = make_agent,
        n_trials      = N_TRIALS,
        name          = "req3_adversarial",
    )

    # ─────────────────────────────────────────────────────────────────────
    # Comparison plots
    # ─────────────────────────────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Plotting")
    logger.info("-" * 60)

    results = {
        "Primal-Dual (stochastic)":   res_stoch,
        "Primal-Dual (adversarial)":  res_adv,
    }

    plot_regret(
        results=results,
        title="Req 3 – Cumulative Pseudo-Regret: Best-of-both-worlds",
        filename="r3/req3_regret.png",
        add_reference=True,
    )
    plot_budget(
        results=results, budget=BUDGET,
        title="Req 3 – Cumulative Cost",
        filename="r3/req3_budget.png",
    )
    plot_lambda(
        results=results,
        title="Req 3 – Lagrange multiplier $\\lambda_t$",
        filename="r3/req3_lambda.png",
    )

    # ─────────────────────────────────────────────────────────────────────
    # Annotated regret: highlight that the SAME agent achieves sublinear
    # regret in BOTH regimes.  This is the punchline of Requirement 3.
    # ─────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ts = np.arange(1, T + 1)
    for label, res, color in [
        ("Stochastic",  res_stoch, "C0"),
        ("Adversarial", res_adv,   "C1"),
    ]:
        mean   = res["mean_regret"]
        stderr = res["std_regret"] / np.sqrt(res["n_trials"])
        ax.plot(ts, mean, label=label, color=color)
        ax.fill_between(ts, mean - stderr, mean + stderr, alpha=0.25, color=color)

    # O(sqrt(T)) reference, scaled to the adversarial curve (the harder one)
    ref = np.sqrt(ts)
    ref = ref * (res_adv["mean_regret"][-1] / ref[-1])
    ax.plot(ts, ref, "k--", linewidth=1.2, label=r"$O(\sqrt{T})$")

    ax.set_xlabel("$t$")
    ax.set_ylabel("Cumulative Pseudo-Regret")
    ax.set_title("Req 3 – Best-of-both-worlds: same agent, two regimes")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = OUTPUTS_DIR / "r3" / "req3_regret_annotated.png"
    plt.savefig(path, dpi=150)
    logger.info("Saved annotated plot to %s", path)
    plt.show()
    plt.close()

    # ─────────────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Final pseudo-regret (mean over %d trials):", N_TRIALS)
    logger.info("  Stochastic   : %.2f", res_stoch["mean_regret"][-1])
    logger.info("  Adversarial  : %.2f", res_adv["mean_regret"][-1])
    logger.info("Final cumulative cost:")
    logger.info("  Stochastic   : %.2f / %.0f",
                res_stoch["mean_cumcost"][-1], BUDGET)
    logger.info("  Adversarial  : %.2f / %.0f",
                res_adv["mean_cumcost"][-1], BUDGET)
    logger.info("=" * 60)
    logger.info("Requirement 3 complete.")

    return {"stochastic": res_stoch, "adversarial": res_adv}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    run_req3()