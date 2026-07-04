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


def compute_ucb1_gap_upper_bound(expected_rewards, reward_range, T):
    """
    Gap-dependent UCB1 regret upper bound for bounded rewards in [0, reward_range].

    This is a real theoretical upper bound, not a visually rescaled reference
    curve. It is intentionally loose, as finite-time UCB1 constants usually are.
    """
    expected_rewards = np.asarray(expected_rewards, dtype=float)
    best = float(np.max(expected_rewards))
    gaps = best - expected_rewards
    positive_gaps = gaps[gaps > 1e-12]
    ts = np.arange(1, T + 1)
    if len(positive_gaps) == 0:
        return np.zeros(T)

    log_ts = np.log(np.maximum(ts, 2))
    exploration = sum((8 * reward_range**2 * log_ts) / gap for gap in positive_gaps)
    constant = sum((1 + np.pi**2 / 3) * gap for gap in positive_gaps)
    return exploration + constant


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
    results,
    title="Cumulative Pseudo-Regret",
    filename="regret.png",
    add_reference=False,
    upper_bound=None,
):
    """
    NB01 cell 25 pattern:
      plt.fill_between(..., mean ± std/sqrt(n_trials), alpha=0.3)

    Optionally overlays a user-provided upper bound. The function does not
    rescale reference curves to the empirical regret, because that would make
    the visual guide look like a theoretical guarantee when it is not one.

    Parameters
    ----------
    results       : dict  {label: {mean_regret, std_regret, n_trials}}
    add_reference : bool  whether to draw an unscaled sqrt(T log T) guide
    upper_bound   : tuple(label, values), optional true bound to draw
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
        ref = np.sqrt(ts * np.log(np.maximum(ts, 2)))
        ax.plot(ts, ref, "k:", linewidth=1.2, label=r"Unscaled $\sqrt{t\log t}$ guide")

    if upper_bound is not None:
        bound_label, bound_values = upper_bound
        ax.plot(ts, bound_values, "k--", linewidth=1.2, label=bound_label)

    ax.set_xlabel("$t$")
    ax.set_ylabel("Cumulative Pseudo-Regret")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = OUTPUTS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    logger.info("Saved plot to %s", path)
    plt.show()
    plt.close()


def plot_ucb1_bound_check(results, upper_bound, title, filename):
    """Plot empirical UCB1 pseudo-regret against the true gap-dependent bound."""
    fig, ax = plt.subplots(figsize=(9, 5))
    T = len(results["mean_regret"])
    ts = np.arange(1, T + 1)
    mean = results["mean_regret"]
    stderr = results["std_regret"] / np.sqrt(results["n_trials"])

    ax.plot(ts, mean, label="Empirical UCB1 pseudo-regret", color="C0")
    ax.fill_between(ts, mean - stderr, mean + stderr, alpha=0.25, color="C0")
    ax.plot(ts, upper_bound, "k--", linewidth=1.2, label="UCB1 gap-dependent upper bound")
    ax.set_xlabel("$t$")
    ax.set_ylabel("Cumulative Pseudo-Regret")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = OUTPUTS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    logger.info("Saved UCB1 bound check to %s", path)
    plt.show()
    plt.close()


def plot_ucb1_bound_ratio(results, upper_bound, title, filename):
    """Plot empirical regret divided by the upper bound; values <= 1 satisfy it."""
    fig, ax = plt.subplots(figsize=(9, 5))
    T = len(results["mean_regret"])
    ts = np.arange(1, T + 1)
    ratio = results["mean_regret"] / upper_bound

    ax.plot(ts, ratio, label=r"$R_t / \mathrm{UB}_t$", color="C2")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Bound threshold")
    ax.set_xlabel("$t$")
    ax.set_ylabel("Empirical regret / upper bound")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = OUTPUTS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    logger.info("Saved UCB1 bound ratio to %s", path)
    plt.show()
    plt.close()


def plot_average_regret(results, title, filename):
    """Plot R_t / t to make no-regret behavior visually explicit."""
    fig, ax = plt.subplots(figsize=(9, 5))
    T = len(next(iter(results.values()))["mean_regret"])
    ts = np.arange(1, T + 1)

    for label, res in results.items():
        mean = res["mean_regret"] / ts
        stderr = (res["std_regret"] / np.sqrt(res["n_trials"])) / ts
        ax.plot(ts, mean, label=label)
        ax.fill_between(ts, mean - stderr, mean + stderr, alpha=0.25)

    ax.set_xlabel("$t$")
    ax.set_ylabel(r"Average Pseudo-Regret $R_t/t$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = OUTPUTS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    logger.info("Saved average regret plot to %s", path)
    plt.show()
    plt.close()


def plot_competing_bid_distribution(env, title, filename):
    """
    Plot the empirical distribution of the highest competing bid and the
    theoretical Beta(k, 1) model implied by max of k Uniform[0, 1] bids.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xs = np.linspace(0, 1, 400)
    k = env.n_competitors
    theoretical_cdf = xs**k
    theoretical_pdf = k * xs ** (k - 1)

    axes[0].hist(
        env.m,
        bins=40,
        density=True,
        alpha=0.45,
        color="C0",
        label="Empirical highest bid",
    )
    axes[0].plot(xs, theoretical_pdf, "k--", linewidth=1.5, label=rf"Beta({k}, 1) PDF")
    axes[0].set_xlabel("Highest competing bid $m_t$")
    axes[0].set_ylabel("Density")
    axes[0].set_title("PDF")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    sorted_m = np.sort(env.m)
    empirical_cdf = np.arange(1, len(sorted_m) + 1) / len(sorted_m)
    axes[1].plot(sorted_m, empirical_cdf, color="C0", label="Empirical CDF")
    axes[1].plot(xs, theoretical_cdf, "k--", linewidth=1.5, label=rf"Beta({k}, 1) CDF")
    axes[1].set_xlabel("Highest competing bid $m_t$")
    axes[1].set_ylabel("Cumulative probability")
    axes[1].set_title("CDF")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    path = OUTPUTS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    logger.info("Saved competing bid distribution plot to %s", path)
    plt.show()
    plt.close()


def plot_multi_competing_bid_distributions(env, title, filename):
    """
    Plot Req2's stochastic model: one highest competing bid distribution per
    campaign plus the empirical correlation matrix of those maxima.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xs = np.linspace(0, 1, 400)

    for i in range(env.N):
        sorted_m = np.sort(env.m[i])
        empirical_cdf = np.arange(1, len(sorted_m) + 1) / len(sorted_m)
        k = env.n_competitors[i]
        axes[0].plot(sorted_m, empirical_cdf, label=f"Campaign {i} empirical")
        axes[0].plot(xs, xs**k, "--", linewidth=1.1, label=rf"Campaign {i} Beta({k}, 1)")

    axes[0].set_xlabel("Highest competing bid $m_{i,t}$")
    axes[0].set_ylabel("Cumulative probability")
    axes[0].set_title("Per-campaign CDFs")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    corr = np.corrcoef(env.m)
    im = axes[1].imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    axes[1].set_title("Empirical correlation of highest bids")
    axes[1].set_xlabel("Campaign")
    axes[1].set_ylabel("Campaign")
    axes[1].set_xticks(range(env.N))
    axes[1].set_yticks(range(env.N))
    for i in range(env.N):
        for j in range(env.N):
            axes[1].text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    path = OUTPUTS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    logger.info("Saved multi-campaign bid distribution plot to %s", path)
    plt.show()
    plt.close()


def plot_pairwise_joint_bid_distributions(env, title, filename):
    """
    Plot pairwise empirical joint distributions of highest competing bids.

    The full joint distribution over N campaigns is N-dimensional; for N=4 we
    visualize all six 2-D marginals. Since campaigns are generated
    independently, the pairwise clouds should show no structural dependence.
    """
    pairs = [(i, j) for i in range(env.N) for j in range(i + 1, env.N)]
    n_cols = 3
    n_rows = int(np.ceil(len(pairs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (i, j) in zip(axes, pairs):
        h = ax.hist2d(
            env.m[i],
            env.m[j],
            bins=35,
            range=[[0, 1], [0, 1]],
            cmap="Blues",
            density=True,
        )
        ax.set_title(f"Campaigns {i} and {j}")
        ax.set_xlabel(r"$m_{%d,t}$" % i)
        ax.set_ylabel(r"$m_{%d,t}$" % j)
        ax.grid(True, linestyle="--", alpha=0.25)
        fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(pairs):]:
        ax.axis("off")

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    path = OUTPUTS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    logger.info("Saved pairwise joint bid distribution plot to %s", path)
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
    path.parent.mkdir(parents=True, exist_ok=True)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    logger.info("Saved plot to %s", path)
    plt.show()
    plt.close()
