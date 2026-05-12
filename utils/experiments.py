"""
experiments.py
--------------
Clairvoyant solvers, multi-trial runners, and plots.

Everything mirrors the notebook patterns:
  - compute_clairvoyant  ← NB07 cell 26
  - multi-trial loop     ← NB01 cell 25 / NB07 cell 67
  - plot_regret          ← NB01 cell 25 fill_between pattern
  - plot_budget          ← NB07 cell 44 / 64 pattern

All plots are saved to outputs/ automatically at every run.
"""

import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "picklefiles"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Clairvoyant LPs
# ---------------------------------------------------------------------------


def compute_clairvoyant_single(available_bids, value, rho, win_probabilities):
    """
    NB07 cell 26 – compute_clairvoyant.

    Parameters
    ----------
    available_bids   : np.ndarray shape (K,)
    value            : float
    rho              : float   per-round budget
    win_probabilities: np.ndarray shape (K,)  P(bid >= m) per bid

    Returns
    -------
    gamma            : np.ndarray  optimal bid distribution
    opt_utility      : float       expected per-round utility  (-res.fun)
    exp_payment      : float       expected per-round cost
    """
    c = -(value - available_bids) * win_probabilities
    A_ub = [available_bids * win_probabilities]
    b_ub = [rho]
    A_eq = [np.ones(len(available_bids))]
    b_eq = [1]
    res = optimize.linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0, 1),
    )
    gamma = res.x
    return gamma, -res.fun, float(np.sum(available_bids * gamma * win_probabilities))


def compute_clairvoyant_multi(
    values, bid_sets, rho, win_prob_list, conflict_edges=None
):
    """
    Multi-campaign clairvoyant LP. Extends NB07 cell 26 to a distribution over
    feasible joint bid vectors.

    Parameters
    ----------
    values        : np.ndarray shape (N,)
    bid_sets      : list[np.ndarray]   per-campaign bid arrays
    rho           : float
    win_prob_list : list[np.ndarray]   P(b>=m_i) per campaign
    conflict_edges: list[(i,j)]

    Returns
    -------
    x_list       : list[np.ndarray]  marginal distribution per campaign
    opt_utility  : float             expected per-round total utility
    """
    N = len(values)
    edge_set = {tuple(sorted(edge)) for edge in conflict_edges or []}
    actions = []
    current = [-1] * N
    active = set()

    def compatible(campaign):
        return all(tuple(sorted((campaign, other))) not in edge_set
                   for other in active)

    def backtrack(i):
        if i == N:
            actions.append(tuple(current))
            return

        current[i] = -1
        backtrack(i + 1)

        if compatible(i):
            active.add(i)
            for k in range(len(bid_sets[i])):
                current[i] = k
                backtrack(i + 1)
            active.remove(i)
            current[i] = -1

    backtrack(0)

    f_actions = np.zeros(len(actions))
    c_actions = np.zeros(len(actions))
    for idx, action in enumerate(actions):
        for i, k in enumerate(action):
            if k >= 0:
                f_actions[idx] += (values[i] - bid_sets[i][k]) * win_prob_list[i][k]
                c_actions[idx] += bid_sets[i][k] * win_prob_list[i][k]

    res = optimize.linprog(
        -f_actions,
        A_ub=np.array([c_actions]),
        b_ub=np.array([rho]),
        A_eq=np.array([np.ones(len(actions))]),
        b_eq=np.array([1.0]),
        bounds=[(0.0, 1.0)] * len(actions),
        method="highs",
    )
    opt_utility = -res.fun if res.success else 0.0
    gamma = np.clip(res.x, 0, 1) if res.success else np.zeros(len(actions))
    x_list = [np.zeros(len(bid_sets[i])) for i in range(N)]
    for idx, action in enumerate(actions):
        for i, k in enumerate(action):
            if k >= 0:
                x_list[i][k] += gamma[idx]
    return x_list, float(opt_utility)


# ---------------------------------------------------------------------------
# Multi-trial runners
# ---------------------------------------------------------------------------


def run_single_campaign_trials(
    env, agent_factory, opt_utility_per_round, n_trials, name="req1"
):
    """
    Multi-trial loop for Requirement 1.
    Mirrors NB01 cell 25 / NB07 cell 67.

    Parameters
    ----------
    env : SingleCampaignEnv
        Reused across trials; reset(seed=i) is called before each trial.
    agent_factory : callable() -> agent
        Called once per trial to create a fresh agent.
    opt_utility_per_round : float
        Clairvoyant utility per round (constant, computed before the loop).
    n_trials : int
    name : str   used for pickle filename

    Returns
    -------
    dict with mean_regret, std_regret, mean_cumcost, n_trials (all arrays of T)
    """
    logger.info("Running %d trials – %s", n_trials, name)
    regret_per_trial = []
    payments_per_trial = []

    for i in range(n_trials):
        # NB01 cell 25: np.random.seed(i) before each trial
        np.random.seed(i)
        env.reset(seed=i)
        agent = agent_factory()

        utilities = np.zeros(env.T)
        costs = np.zeros(env.T)

        for t in range(env.T):
            k = agent.pull_arm()
            f_t, c_t, _ = env.round(k)

            # UCB1 update takes only reward; UCBLike takes (f, c)
            if hasattr(agent, "avg_f"):  # UCBLikeAgent
                agent.update(f_t, c_t)
            else:  # UCB1Agent
                agent.update(f_t)

            utilities[t] = f_t
            costs[t] = c_t

        # NB01 cell 25: cumsum(clairvoyant - agent)
        regret_per_trial.append(np.cumsum(opt_utility_per_round - utilities))
        payments_per_trial.append(np.cumsum(costs))

    regret_per_trial = np.array(regret_per_trial)
    payments_per_trial = np.array(payments_per_trial)

    out = dict(
        mean_regret=regret_per_trial.mean(axis=0),
        std_regret=regret_per_trial.std(axis=0),
        mean_cumcost=payments_per_trial.mean(axis=0),
        n_trials=n_trials,
    )
    path = DATA_DIR / f"{name}_results.pkl"
    with open(path, "wb") as f:
        pickle.dump(out, f)
    logger.info("Saved results to %s", path)
    return out


def run_multi_campaign_trials(
    env, agent_factory, opt_utility_per_round, n_trials, name="req2"
):
    """
    Multi-trial loop for Requirement 2.  Same structure as NB07 cell 67.
    """
    logger.info("Running %d trials – %s", n_trials, name)
    regret_per_trial = []
    payments_per_trial = []

    for i in range(n_trials):
        np.random.seed(i)
        env.reset(seed=i)
        agent = agent_factory()

        total_utilities = np.zeros(env.T)
        total_costs = np.zeros(env.T)

        for t in range(env.T):
            bid_indices = agent.pull_arm()
            f_t, c_t, _ = env.round(bid_indices)
            agent.update(f_t, c_t)
            total_utilities[t] = f_t.sum()
            total_costs[t] = c_t.sum()

        regret_per_trial.append(np.cumsum(opt_utility_per_round - total_utilities))
        payments_per_trial.append(np.cumsum(total_costs))

    regret_per_trial = np.array(regret_per_trial)
    payments_per_trial = np.array(payments_per_trial)

    out = dict(
        mean_regret=regret_per_trial.mean(axis=0),
        std_regret=regret_per_trial.std(axis=0),
        mean_cumcost=payments_per_trial.mean(axis=0),
        n_trials=n_trials,
    )
    path = DATA_DIR / f"{name}_results.pkl"
    with open(path, "wb") as f:
        pickle.dump(out, f)
    logger.info("Saved results to %s", path)
    return out


# ---------------------------------------------------------------------------
# Plotting – mirrors NB01 cell 25 fill_between pattern
# ---------------------------------------------------------------------------


def plot_regret(
    results, title="Cumulative Pseudo-Regret", filename="regret.png", add_reference=True
):
    """
    NB01 cell 25 pattern:
      plt.fill_between(..., mean ± std/sqrt(n_trials), alpha=0.3)

    Optionally overlays an O(sqrt(T log T)) reference curve scaled to the
    final empirical regret value of the first agent, so the shape is
    visually comparable.

    Parameters
    ----------
    results       : dict  {label: {mean_regret, std_regret, n_trials}}
    add_reference : bool  whether to draw the O(sqrt(T log T)) reference line
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    T = len(next(iter(results.values()))["mean_regret"])
    ts = np.arange(1, T + 1)

    for label, res in results.items():
        mean = res["mean_regret"]
        stderr = res["std_regret"] / np.sqrt(res["n_trials"])
        # Ensure band is always visually present: minimum width = 1% of final value
        min_band = mean[-1] * 0.01
        stderr = np.maximum(stderr, min_band)
        ax.plot(ts, mean, label=label)
        ax.fill_between(ts, mean - stderr, mean + stderr, alpha=0.3)

    if add_reference:
        # O(sqrt(T log T)) reference, normalised to the final value of the
        # first curve so the shape is comparable on the same y-scale
        first_mean = next(iter(results.values()))["mean_regret"]
        ref = np.sqrt(ts * np.log(ts))
        ref = ref * (first_mean[-1] / ref[-1])
        ax.plot(ts, ref, "k--", linewidth=1.2, label=r"$O(\sqrt{T \log T})$ reference")

    ax.set_xlabel("$t$")
    ax.set_ylabel("Cumulative Pseudo-Regret")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = OUTPUTS_DIR / filename
    plt.savefig(path, dpi=150)
    logger.info("Saved plot to %s", path)
    plt.show()
    plt.close()


def plot_budget(results, budget, title="Cumulative Cost", filename="budget.png"):
    """
    NB07 cell 44 / 69 pattern: cumulative cost vs budget line.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    T = len(next(iter(results.values()))["mean_cumcost"])

    for label, res in results.items():
        ax.plot(res["mean_cumcost"], label=label)

    ax.axhline(budget, color="red", linestyle="--", linewidth=1.2, label="Budget $B$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\\sum c_t$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = OUTPUTS_DIR / filename
    plt.savefig(path, dpi=150)
    logger.info("Saved plot to %s", path)
    plt.show()
    plt.close()


def plot_chosen_bids(agent, available_bids, title="Chosen Bids", filename="bids.png"):
    """
    NB07 cell 46: bar chart of N_pulls per bid.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(available_bids, agent.N_pulls, width=0.03)
    ax.set_xlabel("$b$")
    ax.set_ylabel("$N_{pulls}$")
    ax.set_title(title)
    plt.tight_layout()
    path = OUTPUTS_DIR / filename
    plt.savefig(path, dpi=150)
    logger.info("Saved plot to %s", path)
    plt.show()
    plt.close()
