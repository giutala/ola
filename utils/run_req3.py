"""
run_req3.py
-----------
Requirement 3: best-of-both-worlds bidding for N campaigns.

Algorithm: PrimalDualMultiCampaignAgent — one Hedge regret minimiser per
campaign (full feedback) coupled to a shared OGD dual variable for the budget.

Two experiments demonstrate the best-of-both-worlds property:

  A. Stochastic environment (MultiCampaignEnv) — benchmark is the fixed LP
     optimum computed once from the true win probabilities.

  B. Adversarial / non-stationary environment (AdversarialMultiCampaignEnv,
     mode='drift') — benchmark is OPT^A, the best FIXED distribution in
     hindsight, computed per trial from the empirical win probabilities of
     that trial's realised m-sequence. This is the benchmark against which a
     primal-dual (Hedge+OGD) agent has a provable sublinear-regret guarantee
     in both stochastic and adversarial settings.

Parameters are imported from req3_config.py (shared with req2, req4, and the
precompute scripts) so all requirements operate on the same problem instance.

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
# Shared problem parameters (VALUES, T, BUDGET, etc.) are imported from
# req3_config.py. Modify them there to keep all requirements in sync.

# Non-stationarity mode: 'drift' (sinusoidal mean, changes every round)
# or 'shocks' (piecewise-stationary blocks).
ENV_MODE = "drift"

# Budget pacing: if True, uses adaptive rho_t = remaining_budget / remaining_rounds
# instead of the fixed rho = B/T.
BUDGET_PACING = True

# OGD learning rate for the dual variable lambda. 0.017 was found to minimise
# final regret with budget_pacing=True on the 'drift' environment.
OGD_ETA = 0.017



def run_req3():
    """Single entry point called from the notebook."""
    logger.info("=" * 60)
    logger.info("Requirement 3 – Best-of-both-worlds bidding")
    logger.info("=" * 60)
    logger.info(
        "Parameters | N=%d T=%d B=%.1f rho=%.4f env_mode=%s budget_pacing=%s "
        "ogd_eta=%s conflict_edges=%s",
        len(VALUES), T, BUDGET, BUDGET / T, ENV_MODE, BUDGET_PACING, OGD_ETA,
        CONFLICT_EDGES
    )

    # ── Factories shared across the two experiments ───────────────────────
    # A reference env (seed=0) is used only to read Ks and bid_sets, which
    # are deterministic functions of values + AVAILABLE_BIDS.
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
            budget_pacing=BUDGET_PACING,
            ogd_eta=OGD_ETA,
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

    win_probs = _env_ref_stoch.win_probabilities()
    _, opt_stoch = compute_clairvoyant_multi(
        np.array(VALUES), BID_SETS, BUDGET / T, win_probs, CONFLICT_EDGES,
    )
    logger.info("Stochastic clairvoyant | per-round utility = %.4f", opt_stoch)

    res_stoch = run_primal_dual_trials(
        env_factory   = env_factory_stoch,
        agent_factory = make_agent,
        n_trials      = N_TRIALS,
        opt_per_round = opt_stoch,
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

    # opt_per_round=None → run_primal_dual_trials computes OPT^A per trial
    # from the empirical win probabilities (env.empirical_win_probabilities()).
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
        add_reference=False,
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