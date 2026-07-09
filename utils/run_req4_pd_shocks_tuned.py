"""
run_req4_pd_shocks_tuned.py
----------------------------
ADDITIONAL, comparison-only script -- does NOT replace utils/run_req4.py.

Motivation
----------
Empirical diagnosis (see the project discussion / notebook) found that
PrimalDualMultiCampaignAgent's dual variable lambda_t gets "stuck" high on
the 'shocks' environment: it jumps to ~0.74 within the FIRST stationary
block (BEFORE any regime change has even happened) and then stays there
for the rest of the horizon even while the agent is under-spending for
four consecutive intervals -- the OGD gradient should be pulling lambda
back down during sustained under-spending, and empirically it barely does.
This is different from Requirement 3's 'drift' mode, where the same
hyperparameters (hedge_eta=default, ogd_eta=0.017, budget_pacing=True) let
lambda settle into a stable, moderate equilibrium instead.

utils/run_req4.py deliberately keeps Requirement 3's agent CONFIGURATION
unchanged (project spec: "the primal-dual method", not "a new primal-dual
method") -- that is the correct thing to submit as the answer to
Requirement 4's "Compare" question. This file exists ALONGSIDE it, as an
explicit, separate experiment: what if the SAME agent CLASS were
re-tuned specifically for discrete (piecewise-constant) non-stationarity
instead of reusing Requirement 3's drift-tuned hyperparameters? This lets
the two approaches be compared directly, not to decide which one is "the"
Requirement 4 submission.

Hypothesis under test
----------------------
hedge_eta controls how fast the PRIMAL (Hedge) weights can move away from
a wrong, regime-specific preference. Requirement 3's default
(hedge_eta = sqrt(log(K_max)/T) ~ 0.01517) was tuned/validated only
against continuous drift, where no single round ever requires a large
correction. Under a discrete shock, the opposite may be true: a LARGER
hedge_eta could let Hedge unlearn a stale preference within one 2000-round
block instead of carrying it into the next one. This script sweeps
hedge_eta and ogd_eta JOINTLY (not ogd_eta alone, which is all the earlier
tune_ogd_eta() in run_req4.py checked) specifically on 'shocks'.

Usage
-----
    python -m utils.run_req4_pd_shocks_tuned          # tunes, then runs the comparison
"""

import logging

import numpy as np

from utils.agents import (
    PrimalDualMultiCampaignAgent,
    SlidingWindowCombinatorialUCBAgent,
    CUSUMCombinatorialUCBAgent,
)
from utils.environments import AdversarialMultiCampaignEnv
from utils.experiments import (
    plot_regret, plot_lambda, run_nonstationary_trials, OUTPUTS_DIR,
)
from utils.req4_config import (
    VALUES, T, BUDGET, N_TRIALS, CONFLICT_EDGES, AVAILABLE_BIDS,
    N_INTERVALS, BLOCK_SIZE, U_T, SW_WINDOW, SHOCKS_MODE_PARAMS,
)
from utils.run_req4 import BUDGET_PACING, PD_OGD_ETA as REQ3_OGD_ETA

logger = logging.getLogger(__name__)

# Requirement 3's (drift-tuned) configuration, reused unchanged in
# utils/run_req4.py -- kept here too, as the "unchanged" reference point
# for the comparison this script exists to make.
REQ3_HEDGE_ETA = None  # None -> class default sqrt(log(K_max)/T) ~= 0.01517

# Filled in by tune_pd_for_shocks() -- see the logged sweep results for
# how these were chosen (3 rounds of sweeping at T_smoke=2000, budget
# rescaled to preserve rho):
#
#   Round 1 -- coarse grid, hedge_mult in {0.5,1,2,4,8,16} x ogd_eta in
#   {0.005,0.01,0.017,0.03,0.05}: regret DECREASES MONOTONICALLY with
#   hedge_mult at every ogd_eta tested, still improving at the top of the
#   range (16x: regret=242.25 vs 1x/default: regret=355.85, at the same
#   ogd_eta=0.017) -- i.e. Requirement 3's hedge_eta is measurably too
#   slow for 'shocks'. ogd_eta=0.017 wins at every hedge_mult -- same
#   value Requirement 3 converged to independently.
#
#   Round 2 -- extended hedge_mult range {16,32,64,128,256} at
#   ogd_eta=0.017: hedge_mult=32 (eta~1.086) makes Hedge's multiplicative
#   weight update numerically UNSTABLE -- weights underflow to all-zero,
#   `weights / weights.sum()` divides 0/0 -> NaN -> np.random.choice
#   crashes. This is a hard ceiling, not a soft tradeoff.
#
#   Round 3 -- fine sweep near the boundary {16,19,22,25,28,31}:
#   hedge_mult=16 -> regret=222.73, hedge_mult=19 -> regret=221.23
#   (+0.7%, essentially a plateau), hedge_mult=22 -> CRASHES (NaN).
#   The instability boundary is seed-dependent (some trials' loss values
#   push exp(-eta*loss) to underflow before others), so 19 is already
#   uncomfortably close. hedge_mult=16 is kept: captures ~99% of the
#   achievable improvement while leaving a real safety margin before the
#   numerical cliff.
#
# ogd_eta is unchanged from Requirement 3's own value -- every sweep
# above confirms 0.017 remains best even after hedge_eta is retuned.
SHOCKS_HEDGE_ETA = 16.0 * 0.015174271293851465   # ~= 0.24279 (16x the T=10000 default)
SHOCKS_OGD_ETA = 0.017


def tune_pd_for_shocks(hedge_mults=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
                        ogd_etas=(0.005, 0.01, 0.017, 0.03, 0.05),
                        n_trials=10, T_smoke=None, budget_pacing=True):
    """
    Joint grid search over (hedge_eta, ogd_eta) on the 'shocks' environment.

    hedge_mults are multipliers of the class default
    sqrt(log(K_max)/T_use) (recomputed at whatever T_smoke is used, so the
    multiplier -- not the absolute value -- is what transfers to the full
    T=10000 run). ogd_etas are absolute values, same convention as
    run_req4.tune_ogd_eta.

    Ranks candidates by:
      1. final regret vs the piecewise expected clairvoyant (primary metric)
      2. logs final budget utilisation and final lambda for every
         candidate, so a candidate that "wins" on regret but still shows
         the lambda-stuck pathology can be spotted, not just accepted
         blindly because the number looks good.

    Returns a dict {(hedge_mult, ogd_eta): result_dict}.
    """
    T_use = T_smoke if T_smoke is not None else T
    budget_use = BUDGET * T_use / T
    block_use = int(BLOCK_SIZE * T_use / T)
    logger.info("tune_pd_for_shocks | hedge_mults=%s ogd_etas=%s n_trials=%d T=%d budget=%.1f",
                hedge_mults, ogd_etas, n_trials, T_use, budget_use)

    def env_factory(seed):
        return AdversarialMultiCampaignEnv(
            values=VALUES, budget=budget_use, T=T_use, available_bids=AVAILABLE_BIDS,
            conflict_edges=CONFLICT_EDGES, seed=seed, mode="shocks",
            block_size=block_use, n_regimes=N_INTERVALS,
        )

    _env_ref = env_factory(0)
    N, Ks, bid_sets = _env_ref.N, _env_ref.Ks, _env_ref.bid_sets
    K_max = max(Ks)
    hedge_default = float(np.sqrt(np.log(max(K_max, 2)) / T_use))

    results = {}
    for hm in hedge_mults:
        hedge_eta = hm * hedge_default
        for oe in ogd_etas:

            def make_agent(hedge_eta=hedge_eta, oe=oe):
                return PrimalDualMultiCampaignAgent(
                    N=N, Ks=Ks, bid_sets=bid_sets, T=T_use, budget=budget_use, values=VALUES,
                    conflict_edges=CONFLICT_EDGES, hedge_eta=hedge_eta, ogd_eta=oe,
                    budget_pacing=budget_pacing,
                )

            res = run_nonstationary_trials(
                env_factory, make_agent, n_trials,
                name=f"tune_pd_shocks_h{hm}_o{oe}",
                compute_opt_a=False, compute_piecewise=True,
            )
            final_regret = res["mean_regret_piecewise"][-1]
            final_cost = res["mean_cumcost"][-1]
            final_lmbd = res["mean_lmbd"][-1] if "mean_lmbd" in res else float("nan")
            results[(hm, oe)] = dict(regret=final_regret, cost=final_cost,
                                      cost_pct=100 * final_cost / budget_use, lmbd=final_lmbd)
            logger.info("  hedge_mult=%5.1f (eta=%.5f) ogd_eta=%.4f -> regret=%7.2f cost=%.1f%% lambda_final=%.3f",
                        hm, hedge_eta, oe, final_regret, results[(hm, oe)]["cost_pct"], final_lmbd)

    best = min(results, key=lambda k: results[k]["regret"])
    logger.info("Best (hedge_mult, ogd_eta) = %s -> %s", best, results[best])
    return results


def make_pd_unchanged_agent(N, Ks, bid_sets):
    """Requirement 3's configuration, reused unchanged (same as utils/run_req4.py)."""
    return PrimalDualMultiCampaignAgent(
        N=N, Ks=Ks, bid_sets=bid_sets, T=T, budget=BUDGET, values=VALUES,
        conflict_edges=CONFLICT_EDGES, hedge_eta=REQ3_HEDGE_ETA, ogd_eta=REQ3_OGD_ETA,
        budget_pacing=BUDGET_PACING,
    )


def make_pd_shocks_tuned_agent(N, Ks, bid_sets):
    """Re-tuned specifically for 'shocks' -- see tune_pd_for_shocks()."""
    if SHOCKS_HEDGE_ETA is None or SHOCKS_OGD_ETA is None:
        raise RuntimeError(
            "SHOCKS_HEDGE_ETA / SHOCKS_OGD_ETA are not set -- run "
            "tune_pd_for_shocks() first and hardcode the winning values "
            "at the top of this module (mirroring how PD_OGD_ETA was "
            "hardcoded in run_req4.py after its own sweep)."
        )
    return PrimalDualMultiCampaignAgent(
        N=N, Ks=Ks, bid_sets=bid_sets, T=T, budget=BUDGET, values=VALUES,
        conflict_edges=CONFLICT_EDGES, hedge_eta=SHOCKS_HEDGE_ETA, ogd_eta=SHOCKS_OGD_ETA,
        budget_pacing=BUDGET_PACING,
    )


def run_req4_pd_comparison():
    """
    Full-scale (T=10000, N_TRIALS trials) comparison of FOUR agents on the
    SAME 'shocks' environment:
      1. Sliding-Window Combinatorial-UCB   (unchanged from run_req4.py)
      2. CUSUM Combinatorial-UCB             (unchanged from run_req4.py)
      3. Primal-Dual -- Req 3 config, UNCHANGED (hedge_eta=default, ogd_eta=0.017)
      4. Primal-Dual -- SHOCKS-TUNED (hedge_eta/ogd_eta from tune_pd_for_shocks())

    Reports regret against the piecewise expected clairvoyant (this
    project's primary Requirement 4 benchmark) and the lambda trajectories
    of both Primal-Dual variants side by side, so the "lambda gets stuck"
    pathology can be checked visually for whether re-tuning fixed it.
    """
    if SHOCKS_HEDGE_ETA is None or SHOCKS_OGD_ETA is None:
        raise RuntimeError("Run tune_pd_for_shocks() and hardcode the winning values first.")

    logger.info("=" * 60)
    logger.info("Requirement 4 -- Primal-Dual hyperparameter comparison (unchanged vs shocks-tuned)")
    logger.info("=" * 60)

    (OUTPUTS_DIR / "r4").mkdir(parents=True, exist_ok=True)

    _env_ref = AdversarialMultiCampaignEnv(
        values=VALUES, budget=BUDGET, T=T, available_bids=AVAILABLE_BIDS,
        conflict_edges=CONFLICT_EDGES, seed=0, mode="shocks", **SHOCKS_MODE_PARAMS,
    )
    N, Ks, bid_sets = _env_ref.N, _env_ref.Ks, _env_ref.bid_sets

    def env_factory(seed):
        return AdversarialMultiCampaignEnv(
            values=VALUES, budget=BUDGET, T=T, available_bids=AVAILABLE_BIDS,
            conflict_edges=CONFLICT_EDGES, seed=seed, mode="shocks", **SHOCKS_MODE_PARAMS,
        )

    def make_sw_agent():
        return SlidingWindowCombinatorialUCBAgent(
            N=N, Ks=Ks, T=T, budget=BUDGET, values=VALUES,
            conflict_edges=CONFLICT_EDGES, W=SW_WINDOW,
        )

    def make_cusum_agent():
        return CUSUMCombinatorialUCBAgent(
            N=N, Ks=Ks, T=T, budget=BUDGET, values=VALUES,
            conflict_edges=CONFLICT_EDGES, U_T=U_T,
        )

    common = dict(compute_opt_a=False, compute_piecewise=True)

    logger.info("-" * 60); logger.info("Sliding-Window Combinatorial-UCB")
    res_sw = run_nonstationary_trials(env_factory, make_sw_agent, N_TRIALS,
                                       name="req4_sw_cucb_pdcompare", **common)

    logger.info("-" * 60); logger.info("CUSUM Combinatorial-UCB")
    res_cusum = run_nonstationary_trials(env_factory, make_cusum_agent, N_TRIALS,
                                          name="req4_cusum_cucb_pdcompare", **common)

    logger.info("-" * 60)
    logger.info("Primal-Dual -- Req 3 config UNCHANGED (hedge_eta=default, ogd_eta=%.4f)", REQ3_OGD_ETA)
    res_pd_unchanged = run_nonstationary_trials(
        env_factory, lambda: make_pd_unchanged_agent(N, Ks, bid_sets), N_TRIALS,
        name="req4_pd_unchanged", **common,
    )

    logger.info("-" * 60)
    logger.info("Primal-Dual -- SHOCKS-TUNED (hedge_eta=%.5f, ogd_eta=%.4f)",
                SHOCKS_HEDGE_ETA, SHOCKS_OGD_ETA)
    res_pd_tuned = run_nonstationary_trials(
        env_factory, lambda: make_pd_shocks_tuned_agent(N, Ks, bid_sets), N_TRIALS,
        name="req4_pd_shocks_tuned", **common,
    )

    results = {
        "Sliding-Window Combinatorial-UCB": res_sw,
        "CUSUM Combinatorial-UCB": res_cusum,
        "Primal-Dual (Req 3 config, unchanged)": res_pd_unchanged,
        "Primal-Dual (shocks-tuned)": res_pd_tuned,
    }
    piecewise_results = {
        label: {**res, "mean_regret": res["mean_regret_piecewise"], "std_regret": res["std_regret_piecewise"]}
        for label, res in results.items() if "mean_regret_piecewise" in res
    }
    plot_regret(results=piecewise_results,
                title="Req 4 -- Primal-Dual: Req3 config vs shocks-tuned (vs Piecewise Clairvoyant)",
                filename="r4/req4_regret_pd_hparam_compare.png", add_reference=False)

    plot_lambda(results={
        "Primal-Dual (Req 3 config, unchanged)": res_pd_unchanged,
        "Primal-Dual (shocks-tuned)": res_pd_tuned,
    }, title="Req 4 -- $\\lambda_t$: Req3 config vs shocks-tuned",
       filename="r4/req4_lambda_pd_hparam_compare.png")

    logger.info("=" * 60)
    logger.info("Final regret vs piecewise expected clairvoyant (mean over %d trials):", N_TRIALS)
    for label, res in results.items():
        logger.info("  %-42s regret=%8.2f cost=%.2f/%.0f (%.1f%%)",
                    label, res["mean_regret_piecewise"][-1], res["mean_cumcost"][-1], BUDGET,
                    100 * res["mean_cumcost"][-1] / BUDGET)
    logger.info("=" * 60)

    return {"sw": res_sw, "cusum": res_cusum, "pd_unchanged": res_pd_unchanged, "pd_shocks_tuned": res_pd_tuned}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    tune_pd_for_shocks()
