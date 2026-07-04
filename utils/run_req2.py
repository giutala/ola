"""
run_req2.py
-----------
Requirement 2: multiple campaigns, stochastic environment.

Algorithm: CombinatorialUCBAgent — extends UCBLikeAgent (NB07 cell 40)
to N campaigns with a shared budget LP oracle and conflict graph,
mirroring the UCBMatchingAgent structure from NB09 cell 30.

T=10_000 matches R1 for a valid cross-requirement regret comparison.
B scaled proportionally so rho=0.16 is unchanged.

Call from the notebook: run_req2()
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from utils.agents import CombinatorialUCBAgent
from utils.environments import MultiCampaignEnv
from utils.experiments import (
    compute_clairvoyant_multi,
    plot_average_regret,
    plot_budget,
    plot_multi_competing_bid_distributions,
    plot_pairwise_joint_bid_distributions,
    plot_regret,
    run_multi_campaign_trials,
    OUTPUTS_DIR,
)

logger = logging.getLogger(__name__)

# ── Parameters ────────────────────────────────────────────────────────────
VALUES = [0.8, 0.6, 0.9, 0.7]
T = 10_000  # matches R1 for valid comparison
BUDGET = 1_600.0  # rho = 1600/10000 = 0.16 (unchanged from before)
N_TRIALS = 20
N_COMPETITORS = [3, 3, 3, 3]
CONFLICT_EDGES = [(0, 1), (2, 3)]
AVAILABLE_BIDS = np.linspace(0, 1, 11)

# Exploration ends when every (i,k) pair has been pulled at least once.
# With N=4 campaigns and K_i<=11 bids each, N*K_max = 44 rounds minimum,
# but the greedy fallback fires until ALL LCBs > 0, which takes longer.
# We annotate the approximate transition empirically.
N_ARMS_TOTAL = 4 * 11  # upper bound; actual Ks may differ per campaign


def run_req2():
    """Single entry point called from the notebook."""
    logger.info("=" * 60)
    logger.info("Requirement 2 – Multiple Campaigns, Stochastic Environment")
    logger.info("=" * 60)
    logger.info(
        "Parameters | N=%d T=%d B=%.1f rho=%.4f conflict_edges=%s",
        len(VALUES),
        T,
        BUDGET,
        BUDGET / T,
        CONFLICT_EDGES,
    )

    # ── Environment ───────────────────────────────────────────────────────
    env = MultiCampaignEnv(
        values=VALUES,
        budget=BUDGET,
        T=T,
        available_bids=AVAILABLE_BIDS,
        n_competitors=N_COMPETITORS,
        conflict_edges=CONFLICT_EDGES,
        seed=0,
    )
    rho = BUDGET / T
    Ks = env.Ks
    N = env.N

    # ── Clairvoyant ───────────────────────────────────────────────────────
    win_prob_list = env.win_probabilities()
    _, opt_utility = compute_clairvoyant_multi(
        values=np.array(VALUES),
        bid_sets=env.bid_sets,
        rho=rho,
        win_prob_list=win_prob_list,
        conflict_edges=CONFLICT_EDGES,
    )
    logger.info("Clairvoyant | opt_utility_per_round=%.4f", opt_utility)

    # ── Combinatorial UCB ─────────────────────────────────────────────────
    results = run_multi_campaign_trials(
        env=env,
        agent_factory=lambda: CombinatorialUCBAgent(
            N=N,
            Ks=Ks,
            T=T,
            budget=BUDGET,
            values=VALUES,
            conflict_edges=CONFLICT_EDGES,
        ),
        opt_utility_per_round=opt_utility,
        n_trials=N_TRIALS,
        name="req2_comb_ucb",
    )

    # ── Standard plots ────────────────────────────────────────────────────
    plot_regret(
        results={"Combinatorial UCB": results},
        title="Req 2 – Cumulative Pseudo-Regret: Multiple Campaigns",
        filename="req2_regret.png",
        add_reference=False,
    )
    plot_budget(
        results={"Combinatorial UCB": results},
        budget=BUDGET,
        title="Req 2 – Cumulative Cost: Multiple Campaigns",
        filename="req2_budget.png",
    )
    plot_average_regret(
        results={"Combinatorial UCB": results},
        title="Req 2 – Average Pseudo-Regret: Multiple Campaigns",
        filename="req2_average_regret.png",
    )
    plot_multi_competing_bid_distributions(
        env=env,
        title="Req 2 – Highest Competing Bid Distributions",
        filename="req2_highest_competing_bid_distributions.png",
    )
    plot_pairwise_joint_bid_distributions(
        env=env,
        title="Req 2 – Pairwise Joint Distributions of Highest Competing Bids",
        filename="req2_pairwise_joint_bid_distributions.png",
    )

    # ── Annotated regret: mark exploration/exploitation transition ─────────
    # The transition is where the regret slope visibly changes — we find it
    # as the point of maximum second derivative of the mean regret.
    mean_regret = results["mean_regret"]
    stderr = results["std_regret"] / np.sqrt(results["n_trials"])
    ts = np.arange(1, T + 1)

    second_deriv = np.diff(mean_regret, n=2)
    # Smooth before finding peak to avoid noise
    smooth_width = max(1, T // 200)
    kernel = np.ones(smooth_width) / smooth_width
    smoothed_d2 = np.convolve(second_deriv, kernel, mode="same")
    # Look only in first third of run where transition should occur
    search_end = T // 3
    kink_t = int(np.argmax(smoothed_d2[:search_end])) + 2  # +2 for diff offset

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ts, mean_regret, label="Combinatorial UCB", color="C0")
    ax.fill_between(
        ts,
        mean_regret - stderr,
        mean_regret + stderr,
        alpha=0.25,
        color="C0",
        label="±stderr",
    )

    # Annotate the transition
    ax.axvline(kink_t, color="gray", linestyle=":", linewidth=1.2)
    ax.annotate(
        f"Exploration ends\n$t\\approx{kink_t}$",
        xy=(kink_t, mean_regret[kink_t]),
        xytext=(kink_t + T * 0.04, mean_regret[kink_t] * 0.6),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="gray"),
        color="gray",
    )
    logger.info("Exploration/exploitation transition annotated at t=%d", kink_t)

    ax.set_xlabel("$t$")
    ax.set_ylabel("Cumulative Pseudo-Regret")
    ax.set_title("Req 2 – Cumulative Pseudo-Regret: Multiple Campaigns (annotated)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = OUTPUTS_DIR / "req2_regret_annotated.png"
    plt.savefig(path, dpi=150)
    logger.info("Saved annotated plot to %s", path)
    plt.show()
    plt.close()

    logger.info("Requirement 2 complete.")
