"""
run_req1.py
-----------
Requirement 1: single campaign, stochastic environment.

Runs two budget scenarios to demonstrate the budget-awareness of UCBlike:
  - Generous budget: B=1600, rho=0.16 → constraint non-binding for the
                                         unconstrained best bid b=0.6, whose
                                         expected cost is 0.1296.
  - Tight budget:    B=400,  rho=0.04 → constraint binding, UCB1 overshoots,
                                         UCBlike stops at t≈4000 (linear regret
                                         but zero budget violation).

Call from the notebook: run_req1()
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils.agents import UCB1BiddingAgent, UCBLikeBiddingAgent
from utils.environments import SingleCampaignEnv
from utils.experiments import (
    compute_clairvoyant_single,
    compute_ucb1_gap_upper_bound,
    plot_average_regret,
    plot_budget,
    plot_competing_bid_distribution,
    plot_regret,
    plot_chosen_bids,
    plot_ucb1_bound_check,
    plot_ucb1_bound_ratio,
    run_single_campaign_trials,
    OUTPUTS_DIR,
)

logger = logging.getLogger(__name__)

# ── Shared parameters ─────────────────────────────────────────────────────
VALUE = 0.8
T = 10_000
N_TRIALS = 30
N_COMPETITORS = 3
# Coarser grid keeps the assignment's "small discrete bid set" assumption and
# makes the UCB gap-dependent behavior visible at T=10_000. With a 0.1 grid,
# bids 0.5 and 0.6 have very close expected rewards, so finite-time bounds are
# true but visually uninformative.
AVAILABLE_BIDS = np.linspace(0, 1, 6)

# Two budget scenarios
BUDGET_GENEROUS = 1600.0  # rho = 0.16  — non-binding; best bid cost is 0.1296
BUDGET_TIGHT = 400.0  # rho = 0.04  — constraint binding


def _expected_rewards_and_costs(env):
    win_probs = env.win_probabilities()
    rewards = (VALUE - env.available_bids) * win_probs
    costs = env.available_bids * win_probs
    return rewards, costs


def _run_scenario(budget, label_suffix, name_suffix):
    """Run one budget scenario, return (results_ucb1, results_ucblike, opt, K, env)."""
    env = SingleCampaignEnv(
        value=VALUE,
        budget=budget,
        T=T,
        available_bids=AVAILABLE_BIDS,
        n_competitors=N_COMPETITORS,
        seed=0,
    )
    K = env.K
    rho = budget / T
    win_probs = env.win_probabilities()
    _, opt_utility, exp_payment = compute_clairvoyant_single(
        env.available_bids,
        VALUE,
        rho,
        win_probs,
    )
    logger.info(
        "Scenario %s | rho=%.4f opt=%.4f exp_payment=%.4f",
        label_suffix,
        rho,
        opt_utility,
        exp_payment,
    )

    r_ucb1 = run_single_campaign_trials(
        env=env,
        agent_factory=lambda: UCB1BiddingAgent(K=K, T=T, range=VALUE),
        opt_utility_per_round=opt_utility,
        n_trials=N_TRIALS,
        name=f"req1_ucb1_{name_suffix}",
    )
    r_ucbl = run_single_campaign_trials(
        env=env,
        agent_factory=lambda: UCBLikeBiddingAgent(K=K, B=budget, T=T, range=VALUE),
        opt_utility_per_round=opt_utility,
        n_trials=N_TRIALS,
        name=f"req1_ucblike_{name_suffix}",
    )
    expected_rewards, expected_costs = _expected_rewards_and_costs(env)
    return r_ucb1, r_ucbl, opt_utility, K, env, rho, expected_rewards, expected_costs


def _plot_comparison(res_gen, res_tight, filename):
    """
    Two-panel figure: generous ρ (left) vs tight ρ (right).
    Each panel shows UCB1 and UCBlike regret. No theoretical upper bound is
    drawn here because the tight-budget panel uses a constrained LP benchmark,
    while the standard UCB1 bound applies to unconstrained stochastic MAB.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    ts = np.arange(1, T + 1)

    panels = [
        (axes[0], res_gen, r"Generous budget $\rho=0.16$", BUDGET_GENEROUS),
        (axes[1], res_tight, r"Tight budget $\rho=0.04$", BUDGET_TIGHT),
    ]

    for ax, (r_ucb1, r_ucbl), title, budget in panels:
        for label, res, color in [
            ("UCB1 (no budget)", r_ucb1, "C0"),
            ("UCB-like (budget)", r_ucbl, "C1"),
        ]:
            mean = res["mean_regret"]
            stderr = res["std_regret"] / np.sqrt(res["n_trials"])
            ax.plot(ts, mean, label=label, color=color)
            ax.fill_between(ts, mean - stderr, mean + stderr, alpha=0.25, color=color)

        ax.set_title(title)
        ax.set_xlabel("$t$")
        ax.set_ylabel("Cumulative Pseudo-Regret")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Req 1 — Regret: Generous vs Tight Budget", fontsize=12)
    plt.tight_layout()
    path = OUTPUTS_DIR / filename
    plt.savefig(path, dpi=150)
    logger.info("Saved comparison plot to %s", path)
    plt.show()
    plt.close()


def _plot_cost_comparison(res_gen, res_tight, filename):
    """Two-panel cost plot: generous ρ (left) vs tight ρ (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ts = np.arange(1, T + 1)

    for ax, (r_ucb1, r_ucbl), title, budget in [
        (axes[0], res_gen, r"Generous budget $\rho=0.16$", BUDGET_GENEROUS),
        (axes[1], res_tight, r"Tight budget $\rho=0.04$", BUDGET_TIGHT),
    ]:
        ax.plot(ts, r_ucb1["mean_cumcost"], label="UCB1 (no budget)", color="C0")
        ax.plot(ts, r_ucbl["mean_cumcost"], label="UCB-like (budget)", color="C1")
        ax.axhline(
            budget, color="red", linestyle="--", linewidth=1.2, label="Budget $B$"
        )
        ax.set_title(title)
        ax.set_xlabel("$t$")
        ax.set_ylabel(r"$\sum c_t$")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Req 1 — Cumulative Cost: Generous vs Tight Budget", fontsize=12)
    plt.tight_layout()
    path = OUTPUTS_DIR / filename
    plt.savefig(path, dpi=150)
    logger.info("Saved cost comparison to %s", path)
    plt.show()
    plt.close()


def run_req1():
    """Single entry point called from the notebook."""
    logger.info("=" * 60)
    logger.info("Requirement 1 – Single Campaign, Stochastic Environment")
    logger.info("=" * 60)

    # ── Scenario 1: generous budget (rho=0.16) ────────────────────────────
    logger.info(
        "--- Scenario: generous budget (B=%.0f, rho=%.2f) ---",
        BUDGET_GENEROUS,
        BUDGET_GENEROUS / T,
    )
    (
        r_ucb1_gen,
        r_ucbl_gen,
        opt_gen,
        K_gen,
        env_gen,
        rho_gen,
        rewards_gen,
        costs_gen,
    ) = _run_scenario(
        BUDGET_GENEROUS, "generous", "generous"
    )

    # ── Scenario 2: tight budget (rho=0.04) ───────────────────────────────
    logger.info(
        "--- Scenario: tight budget (B=%.0f, rho=%.2f) ---",
        BUDGET_TIGHT,
        BUDGET_TIGHT / T,
    )
    (
        r_ucb1_tight,
        r_ucbl_tight,
        opt_tight,
        K_tight,
        env_tight,
        rho_tight,
        rewards_tight,
        costs_tight,
    ) = (
        _run_scenario(BUDGET_TIGHT, "tight", "tight")
    )

    # ── Combined comparison plots ──────────────────────────────────────────
    _plot_comparison(
        res_gen=(r_ucb1_gen, r_ucbl_gen),
        res_tight=(r_ucb1_tight, r_ucbl_tight),
        filename="req1_regret_comparison.png",
    )
    _plot_cost_comparison(
        res_gen=(r_ucb1_gen, r_ucbl_gen),
        res_tight=(r_ucb1_tight, r_ucbl_tight),
        filename="req1_budget_comparison.png",
    )
    plot_competing_bid_distribution(
        env=env_gen,
        title="Req 1 — Distribution of the Highest Competing Bid",
        filename="req1_highest_competing_bid_distribution.png",
    )

    # ── Individual plots (generous scenario, for the report) ──────────────
    plot_regret(
        results={"UCB1 (no budget)": r_ucb1_gen, "UCB-like (budget)": r_ucbl_gen},
        title=r"Req 1 — Regret: Generous budget ($\rho=0.16$)",
        filename="req1_regret_generous.png",
        add_reference=False,
    )
    plot_regret(
        results={"UCB1 (no budget)": r_ucb1_tight, "UCB-like (budget)": r_ucbl_tight},
        title=r"Req 1 — Regret: Tight budget ($\rho=0.04$)",
        filename="req1_regret_tight.png",
        add_reference=False,
    )

    ucb1_bound = compute_ucb1_gap_upper_bound(
        expected_rewards=rewards_gen,
        reward_range=VALUE,
        T=T,
    )
    plot_ucb1_bound_check(
        results=r_ucb1_gen,
        upper_bound=ucb1_bound,
        title=r"Req 1 — UCB1 bound check, non-binding budget ($\rho=0.16$)",
        filename="req1_ucb1_true_upper_bound.png",
    )
    plot_ucb1_bound_ratio(
        results=r_ucb1_gen,
        upper_bound=ucb1_bound,
        title=r"Req 1 — UCB1 regret / true upper bound ($\rho=0.16$)",
        filename="req1_ucb1_bound_ratio.png",
    )
    plot_average_regret(
        results={"UCB1 (no budget)": r_ucb1_gen, "UCB-like (budget)": r_ucbl_gen},
        title=r"Req 1 — Average regret: non-binding budget ($\rho=0.16$)",
        filename="req1_average_regret_generous.png",
    )
    plot_average_regret(
        results={"UCB1 (no budget)": r_ucb1_tight, "UCB-like (budget)": r_ucbl_tight},
        title=r"Req 1 — Average regret: tight budget ($\rho=0.04$)",
        filename="req1_average_regret_tight.png",
    )
    plot_budget(
        results={"UCB1 (no budget)": r_ucb1_tight, "UCB-like (budget)": r_ucbl_tight},
        budget=BUDGET_TIGHT,
        title=r"Req 1 — Cost: Tight budget ($\rho=0.04$)",
        filename="req1_budget_tight.png",
    )

    # ── Chosen bids diagnostic (generous scenario) ────────────────────────
    diag_env = SingleCampaignEnv(
        VALUE, BUDGET_GENEROUS, T, AVAILABLE_BIDS, N_COMPETITORS, seed=42
    )
    diag_agent = UCBLikeBiddingAgent(K=K_gen, B=BUDGET_GENEROUS, T=T, range=VALUE)
    for _ in range(T):
        k = diag_agent.pull_arm()
        f_t, c_t, _ = diag_env.round(k)
        diag_agent.update(f_t, c_t)

    plot_chosen_bids(
        agent=diag_agent,
        available_bids=diag_env.available_bids,
        title="Req 1 — UCB-like: Chosen Bids (generous budget)",
        filename="req1_chosen_bids.png",
    )

    logger.info("Requirement 1 complete.")
