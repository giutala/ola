"""
run_req4_pd_req3_eta.py
-----------------------
Separate Requirement 4 comparison run for the primal-dual agent using the
Requirement 3 learning-rate choice.

This script intentionally does not overwrite the standard Requirement 4 run:

  - pickle name: req4_primal_dual_req3_eta_results.pkl
  - figures: outputs/r4/req3_eta_comparison/

The environment remains the Requirement 4 shocks environment. Only the
PrimalDualMultiCampaignAgent learning-rate configuration changes:

  - hedge_eta: None -> default sqrt(log(K_max) / T), as in Requirement 3
  - ogd_eta: 0.017, as in Requirement 3
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.show = lambda *args, **kwargs: None

from utils.agents import PrimalDualMultiCampaignAgent
from utils.environments import AdversarialMultiCampaignEnv
from utils.experiments import (
    OUTPUTS_DIR,
    load_clairvoyant_cache,
    plot_budget,
    plot_lambda,
    plot_regret,
    run_nonstationary_trials,
)
from utils.run_req3 import OGD_ETA as REQ3_OGD_ETA
from utils.req4_config import (
    AVAILABLE_BIDS,
    BLOCK_SIZE,
    BUDGET,
    CONFLICT_EDGES,
    N_INTERVALS,
    N_TRIALS,
    SHOCKS_MODE_PARAMS,
    T,
    VALUES,
    make_cache_key,
)

logger = logging.getLogger(__name__)

OUT_DIR = OUTPUTS_DIR / "r4" / "req3_eta_comparison"
RESULT_NAME = "req4_primal_dual_req3_eta"
BUDGET_PACING = True


def _plot_average_regret(results: dict, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    T_len = len(next(iter(results.values()))["mean_regret"])
    ts = np.arange(1, T_len + 1)

    for label, res in results.items():
        mean = res["mean_regret"] / ts
        stderr = (res["std_regret"] / np.sqrt(res["n_trials"])) / ts
        ax.plot(ts, mean, label=label)
        ax.fill_between(ts, mean - stderr, mean + stderr, alpha=0.2)

    ax.set_xlabel("$t$")
    ax.set_ylabel("Average pseudo-regret $R_t/t$")
    ax.set_title("Req 4 - Primal-Dual with Req 3 eta: Average Regret")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = OUT_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved average regret plot to %s", path)


def _write_summary(res: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "req4_pd_req3_eta_summary.csv"
    piecewise = res.get("mean_regret_piecewise", res["mean_regret"])[-1]
    opt_a = res.get("mean_regret_opt_a", [np.nan])[-1]
    prophet = res["mean_regret"][-1]
    with open(path, "w") as f:
        f.write(
            "setting,hedge_eta,ogd_eta,budget_pacing,n_trials,"
            "final_regret_piecewise,final_regret_opt_a,final_regret_prophet,"
            "final_cost,remaining_budget,final_lambda\n"
        )
        f.write(
            "Req4 shocks with Req3 eta,"
            f"default_sqrt_logK_over_T,{REQ3_OGD_ETA:.6f},{BUDGET_PACING},"
            f"{res['n_trials']},{piecewise:.6f},{opt_a:.6f},{prophet:.6f},"
            f"{res['mean_cumcost'][-1]:.6f},{BUDGET - res['mean_cumcost'][-1]:.6f},"
            f"{res.get('mean_lmbd', [np.nan])[-1]:.6f}\n"
        )
    logger.info("Saved summary to %s", path)


def run_req4_pd_req3_eta(n_trials: int | None = None):
    trials = n_trials if n_trials is not None else N_TRIALS
    logger.info("=" * 60)
    logger.info("Req 4 comparison - Primal-Dual with Requirement 3 eta")
    logger.info("=" * 60)
    logger.info(
        "Parameters | T=%d B=%.1f rho=%.4f blocks=%d block_size=%d "
        "hedge_eta=default ogd_eta=%.4f budget_pacing=%s trials=%d",
        T, BUDGET, BUDGET / T, N_INTERVALS, BLOCK_SIZE,
        REQ3_OGD_ETA, BUDGET_PACING, trials,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env_ref = AdversarialMultiCampaignEnv(
        values=VALUES,
        budget=BUDGET,
        T=T,
        available_bids=AVAILABLE_BIDS,
        conflict_edges=CONFLICT_EDGES,
        seed=0,
        mode="shocks",
        **SHOCKS_MODE_PARAMS,
    )
    N, Ks, bid_sets = env_ref.N, env_ref.Ks, env_ref.bid_sets

    def env_factory(seed):
        return AdversarialMultiCampaignEnv(
            values=VALUES,
            budget=BUDGET,
            T=T,
            available_bids=AVAILABLE_BIDS,
            conflict_edges=CONFLICT_EDGES,
            seed=seed,
            mode="shocks",
            **SHOCKS_MODE_PARAMS,
        )

    def make_agent():
        return PrimalDualMultiCampaignAgent(
            N=N,
            Ks=Ks,
            bid_sets=bid_sets,
            T=T,
            budget=BUDGET,
            values=VALUES,
            conflict_edges=CONFLICT_EDGES,
            hedge_eta=None,
            ogd_eta=REQ3_OGD_ETA,
            budget_pacing=BUDGET_PACING,
        )

    cache_key = make_cache_key(mode="shocks", mode_params=SHOCKS_MODE_PARAMS)
    cache = load_clairvoyant_cache(cache_key)
    if not cache:
        logger.warning("Dynamic/prophet cache missing; prophet reference may be skipped.")

    res = run_nonstationary_trials(
        env_factory,
        make_agent,
        trials,
        name=RESULT_NAME,
        clairvoyant_cache=cache or None,
        compute_opt_a=True,
        compute_piecewise=True,
    )

    standalone = {"Primal-Dual (Req 3 eta)": res}

    if "mean_regret_piecewise" in res:
        plot_regret(
            results={
                "Primal-Dual (Req 3 eta)": {
                    **res,
                    "mean_regret": res["mean_regret_piecewise"],
                    "std_regret": res["std_regret_piecewise"],
                }
            },
            title="Req 4 - Primal-Dual with Req 3 eta vs Piecewise Clairvoyant",
            filename="r4/req3_eta_comparison/regret_piecewise.png",
            add_reference=False,
        )

    if "mean_regret_opt_a" in res:
        plot_regret(
            results={
                "Primal-Dual (Req 3 eta)": {
                    **res,
                    "mean_regret": res["mean_regret_opt_a"],
                    "std_regret": res["std_regret_opt_a"],
                }
            },
            title="Req 4 - Primal-Dual with Req 3 eta vs OPT$^A$",
            filename="r4/req3_eta_comparison/regret_opta.png",
            add_reference=False,
        )

    plot_regret(
        results=standalone,
        title="Req 4 - Primal-Dual with Req 3 eta vs Dynamic/Prophet",
        filename="r4/req3_eta_comparison/regret_prophet.png",
        add_reference=False,
    )
    _plot_average_regret(standalone, "average_regret_prophet.png")
    plot_budget(
        results=standalone,
        budget=BUDGET,
        title="Req 4 - Primal-Dual with Req 3 eta: Cumulative Cost",
        filename="r4/req3_eta_comparison/budget.png",
    )
    if "mean_lmbd" in res:
        plot_lambda(
            results=standalone,
            title="Req 4 - Primal-Dual with Req 3 eta: $\\lambda_t$",
            filename="r4/req3_eta_comparison/lambda.png",
        )

    _write_summary(res)

    logger.info("Final regret piecewise: %.2f", res.get("mean_regret_piecewise", res["mean_regret"])[-1])
    if "mean_regret_opt_a" in res:
        logger.info("Final regret OPT^A: %.2f", res["mean_regret_opt_a"][-1])
    logger.info("Final regret prophet: %.2f", res["mean_regret"][-1])
    logger.info("Final cost: %.2f / %.0f", res["mean_cumcost"][-1], BUDGET)
    logger.info("Done. Outputs in %s", OUT_DIR)
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    run_req4_pd_req3_eta()
