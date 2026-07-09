"""
run_req4.py
-----------
Requirement 4: slightly non-stationary environment, multiple campaigns.

Environment: AdversarialMultiCampaignEnv(mode='shocks') — the same N, VALUES,
BUDGET, T, and bid grid as Requirements 2 and 3, reparameterised for 5 long
piecewise-stationary blocks (block_size=2000).

Three strategies compared on the same environment:
  1. SlidingWindowCombinatorialUCBAgent  — window W=2000 (one block length)
  2. CUSUMCombinatorialUCBAgent          — CUSUM detector on the win indicator
  3. PrimalDualMultiCampaignAgent        — Requirement 3 agent, budget_pacing=True

Three benchmarks reported:
  - PRIMARY   : piecewise expected clairvoyant — knows block boundaries and true
                block distributions, not individual m_t realisations. The natural
                target for SW-UCB and CUSUM-UCB (Garivier & Moulines 2011).
  - SECONDARY : OPT^A — best fixed distribution in hindsight (continuity with Req 3).
  - REFERENCE : dynamic/prophet — knows every realised m_t; inflates regret by a
                term linear in T via Jensen's inequality (see compute_clairvoyant_dynamic_multi).

Call from the notebook / CLI: run_req4()
"""

import logging

from utils.agents import (
    PrimalDualMultiCampaignAgent,
    SlidingWindowCombinatorialUCBAgent,
    CUSUMCombinatorialUCBAgent,
)
from utils.environments import AdversarialMultiCampaignEnv
from utils.experiments import (
    plot_regret, plot_budget, plot_lambda, plot_resets_histogram,
    load_clairvoyant_cache, run_nonstationary_trials, OUTPUTS_DIR,
)
from utils.req4_config import (
    VALUES, T, BUDGET, N_TRIALS, CONFLICT_EDGES, AVAILABLE_BIDS,
    N_INTERVALS, BLOCK_SIZE, U_T, SW_WINDOW, SHOCKS_MODE_PARAMS,
    make_cache_key,
)

logger = logging.getLogger(__name__)

# Cache key for the dynamic/prophet reference curve, auto-derived from
# req4_config parameters. Run `python -m utils.precompute_clairvoyant_req4`
# once to populate the cache; if missing, the prophet curve is skipped
# (only the reference benchmark, not required for the primary diagnostic).
CLAIRVOYANT_CACHE_KEY = make_cache_key(mode="shocks", mode_params=SHOCKS_MODE_PARAMS)

# Budget pacing: rho_t = remaining_budget / remaining_rounds (adaptive).
BUDGET_PACING = True

# OGD learning rate for the dual variable lambda. Tuned empirically via
# tune_ogd_eta() on the 'shocks' environment with budget_pacing=True;
# the same value was found independently optimal for 'drift' in run_req3.py.
PD_OGD_ETA = 0.017


def run_req4(pd_ogd_eta=None):
    eta = pd_ogd_eta if pd_ogd_eta is not None else PD_OGD_ETA

    logger.info("=" * 60)
    logger.info("Requirement 4 - Slightly Non-Stationary, Multiple Campaigns")
    logger.info("=" * 60)
    logger.info("Parameters | N=%d T=%d B=%.1f rho=%.4f n_intervals=%d block_size=%d "
                "sw_window=%d U_T=%d pd_ogd_eta=%.4f budget_pacing=%s",
                len(VALUES), T, BUDGET, BUDGET / T, N_INTERVALS, BLOCK_SIZE,
                SW_WINDOW, U_T, eta, BUDGET_PACING)

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

    def make_pd_agent():
        return PrimalDualMultiCampaignAgent(
            N=N, Ks=Ks, bid_sets=bid_sets, T=T, budget=BUDGET, values=VALUES,
            conflict_edges=CONFLICT_EDGES, ogd_eta=eta, budget_pacing=BUDGET_PACING,
        )

    cache = load_clairvoyant_cache(CLAIRVOYANT_CACHE_KEY)
    if not cache:
        logger.warning(
            "No dynamic/prophet clairvoyant cache found for key=%s -- run "
            "`python -m utils.precompute_clairvoyant_req4` to populate the "
            "upper-bound reference curve. The primary (piecewise-expected) "
            "and secondary (OPT^A) benchmarks below do not need it.",
            CLAIRVOYANT_CACHE_KEY,
        )

    common_kwargs = dict(clairvoyant_cache=cache or None, compute_opt_a=True, compute_piecewise=True)

    logger.info("-" * 60); logger.info("Sliding-Window Combinatorial-UCB")
    res_sw = run_nonstationary_trials(env_factory, make_sw_agent, N_TRIALS,
                                       name="req4_sw_cucb", **common_kwargs)

    logger.info("-" * 60); logger.info("CUSUM Combinatorial-UCB")
    res_cusum = run_nonstationary_trials(env_factory, make_cusum_agent, N_TRIALS,
                                          name="req4_cusum_cucb", **common_kwargs)

    logger.info("-" * 60); logger.info("Primal-Dual (Requirement 3, budget_pacing=%s, ogd_eta=%.4f)",
                                        BUDGET_PACING, eta)
    res_pd = run_nonstationary_trials(env_factory, make_pd_agent, N_TRIALS,
                                       name="req4_primal_dual", **common_kwargs)

    results = {
        "Sliding-Window Combinatorial-UCB": res_sw,
        "CUSUM Combinatorial-UCB": res_cusum,
        "Primal-Dual (Req 3)": res_pd,
    }

    # --- PRIMARY diagnostic: piecewise expected clairvoyant -----------------
    piecewise_results = {
        label: {**res, "mean_regret": res["mean_regret_piecewise"], "std_regret": res["std_regret_piecewise"]}
        for label, res in results.items() if "mean_regret_piecewise" in res
    }
    if piecewise_results:
        plot_regret(results=piecewise_results,
                    title="Req 4 - Cumulative Regret vs Piecewise Expected Clairvoyant (primary)",
                    filename="r4/req4_regret_piecewise.png", add_reference=False)

    # --- SECONDARY: OPT^A, same methodology as Requirement 3 ----------------
    opt_a_results = {
        label: {**res, "mean_regret": res["mean_regret_opt_a"], "std_regret": res["std_regret_opt_a"]}
        for label, res in results.items() if "mean_regret_opt_a" in res
    }
    if opt_a_results:
        plot_regret(results=opt_a_results,
                    title="Req 4 - Cumulative Regret vs OPT$^A$ (secondary, matches Req 3)",
                    filename="r4/req4_regret_opta.png", add_reference=False)

    # --- REFERENCE upper bound: dynamic / prophet clairvoyant ---------------
    plot_regret(results=results,
                title="Req 4 - Cumulative Regret vs Dynamic/Prophet Clairvoyant (reference upper bound)",
                filename="r4/req4_regret_prophet.png", add_reference=False)

    plot_budget(results=results, budget=BUDGET, title="Req 4 - Cumulative Cost",
                filename="r4/req4_budget.png")

    if "resets_per_trial" in res_cusum:
        plot_resets_histogram(res_cusum["resets_per_trial"],
                               title="Req 4 - How often did the CUSUM detector fire?",
                               filename="r4/req4_cusum_resets.png")

    if "mean_lmbd" in res_pd:
        plot_lambda(results={"Primal-Dual (Req 3)": res_pd},
                    title="Req 4 - Lagrange multiplier $\\lambda_t$",
                    filename="r4/req4_lambda.png")

    logger.info("=" * 60)
    logger.info("Final regret (mean over %d trials):", N_TRIALS)
    logger.info("  %-22s %10s %10s %10s", "", "piecewise", "OPT^A", "prophet")
    for label, res in results.items():
        pw = res.get("mean_regret_piecewise", [float("nan")])[-1]
        oa = res.get("mean_regret_opt_a", [float("nan")])[-1]
        pr = res["mean_regret"][-1]
        logger.info("  %-22s %10.2f %10.2f %10.2f", label, pw, oa, pr)
    if "mean_resets" in res_cusum:
        logger.info("Mean CUSUM resets per trial: %.1f", res_cusum["mean_resets"])
    logger.info("Final cumulative cost:")
    for label, res in results.items():
        logger.info("  %-22s %.2f / %.0f", label, res["mean_cumcost"][-1], BUDGET)
    logger.info("=" * 60)
    logger.info("Requirement 4 complete.")

    return {"sw": res_sw, "cusum": res_cusum, "pd": res_pd}


def tune_ogd_eta(candidates=(0.005, 0.01, 0.017, 0.028, 0.04, 0.06),
                  n_trials=10, T_smoke=None, budget_pacing=True):
    """
    Empirically compare PrimalDualMultiCampaignAgent's ogd_eta (absolute
    value, not a c/sqrt(T) multiplier) WITH budget_pacing on, on the
    'shocks' environment specifically. See module docstring / PD_OGD_ETA
    comment above for the result.

    Uses a smaller n_trials (and optionally a shorter T, with budget
    rescaled to preserve rho) than the full run for speed; final regret
    should be re-checked with the full run_req4() settings before
    reporting numbers.
    """
    T_use = T_smoke if T_smoke is not None else T
    budget_use = BUDGET * T_use / T
    logger.info("tune_ogd_eta | candidates=%s n_trials=%d T=%d budget=%.1f (rho=%.4f) budget_pacing=%s",
                candidates, n_trials, T_use, budget_use, budget_use / T_use, budget_pacing)

    def env_factory(seed):
        return AdversarialMultiCampaignEnv(
            values=VALUES, budget=budget_use, T=T_use, available_bids=AVAILABLE_BIDS,
            conflict_edges=CONFLICT_EDGES, seed=seed, mode="shocks",
            block_size=int(BLOCK_SIZE * T_use / T), n_regimes=N_INTERVALS,
        )

    _env_ref = env_factory(0)
    N, Ks, bid_sets = _env_ref.N, _env_ref.Ks, _env_ref.bid_sets

    results = {}
    for eta in candidates:

        def make_agent(eta=eta):
            return PrimalDualMultiCampaignAgent(
                N=N, Ks=Ks, bid_sets=bid_sets, T=T_use, budget=budget_use, values=VALUES,
                conflict_edges=CONFLICT_EDGES, ogd_eta=eta, budget_pacing=budget_pacing,
            )

        res = run_nonstationary_trials(env_factory, make_agent, n_trials,
                                        name=f"tune_ogd_eta_pace_{eta}",
                                        compute_opt_a=True, compute_piecewise=True)
        final_regret = res["mean_regret_piecewise"][-1] if "mean_regret_piecewise" in res else res["mean_regret"][-1]
        final_cost = res["mean_cumcost"][-1]
        results[eta] = (final_regret, final_cost)
        logger.info("  ogd_eta=%.4f -> final regret(piecewise)=%.2f cost=%.2f/%.0f",
                    eta, final_regret, final_cost, budget_use)

    best_eta = min(results, key=lambda e: results[e][0])
    logger.info("Best candidate: ogd_eta=%.4f (regret=%.2f)", best_eta, results[best_eta][0])
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    run_req4()
