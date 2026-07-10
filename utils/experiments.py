"""
experiments.py
--------------
Clairvoyant LP solvers, multi-trial runners, and plotting utilities.

Clairvoyants
  compute_clairvoyant_single           Single-campaign LP (Requirement 1).
  compute_clairvoyant_multi            Multi-campaign LP (Requirements 2–4).
  compute_clairvoyant_dynamic_multi    Dynamic/prophet oracle (reference upper bound).
  compute_piecewise_expected_clairvoyant   Piecewise oracle (Requirement 4 primary).

Trial runners
  run_single_campaign_trials     Requirement 1 loop.
  run_multi_campaign_trials      Requirement 2 loop.
  run_primal_dual_trials         Requirement 3 full-feedback loop.
  run_nonstationary_trials       Requirement 4 loop with multiple benchmarks.

All plots are auto-saved to outputs/ at every call.
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


def compute_clairvoyant_single(
    available_bids: np.ndarray,
    value: float,
    rho: float,
    win_probabilities: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Single-campaign clairvoyant LP.

    Solves:
        max_gamma  sum_b gamma_b (value - bid_b) P(win_b)
        s.t.       sum_b gamma_b bid_b P(win_b) <= rho
                   sum_b gamma_b = 1,  gamma_b >= 0

    Parameters
    ----------
    available_bids    : np.ndarray shape (K,)
    value             : float
    rho               : float   per-round budget rate
    win_probabilities : np.ndarray shape (K,)   P(bid >= m) per bid

    Returns
    -------
    gamma       : np.ndarray   optimal bid distribution
    opt_utility : float        expected per-round utility
    exp_payment : float        expected per-round cost
    """
    K = len(available_bids)
    res = optimize.linprog(
        -(value - available_bids) * win_probabilities,
        A_ub=[available_bids * win_probabilities],
        b_ub=[rho],
        A_eq=[np.ones(K)],
        b_eq=[1.0],
        bounds=[(0.0, 1.0)] * K,
        method="highs",
    )
    gamma = res.x
    return gamma, -res.fun, float(np.sum(available_bids * gamma * win_probabilities))


def compute_ucb1_gap_upper_bound(
    expected_rewards: np.ndarray,
    reward_range: float,
    T: int,
) -> np.ndarray:
    """
    Gap-dependent UCB1 regret upper bound for bounded rewards in [0, reward_range].

    Returns the true finite-time bound summed over all suboptimal arms. The
    bound is intentionally loose at short horizons; its role is to confirm
    that empirical regret stays below the guarantee, not to predict the level.
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
    values: np.ndarray,
    bid_sets: list[np.ndarray],
    rho: float,
    win_prob_list: list[np.ndarray],
    conflict_edges: list[tuple[int, int]] | None = None,
) -> tuple[list[np.ndarray], float]:
    """
    Multi-campaign clairvoyant LP over a distribution on feasible joint bid vectors.

    Solves:
        max_gamma  sum_a gamma_a sum_i (v_i - b_{i,k}) P(win_{i,k})   [for action a]
        s.t.       sum_a gamma_a sum_i b_{i,k} P(win_{i,k}) <= rho
                   sum_a gamma_a = 1,  gamma_a >= 0

    Each joint action a is a conflict-graph-feasible bid vector.

    Parameters
    ----------
    values        : np.ndarray shape (N,)
    bid_sets      : list[np.ndarray]   per-campaign bid arrays
    rho           : float              per-round budget rate
    win_prob_list : list[np.ndarray]   P(b >= m_i) per campaign
    conflict_edges : list[tuple[int, int]], optional

    Returns
    -------
    x_list      : list[np.ndarray]   marginal bid distribution per campaign
    opt_utility : float              expected per-round total utility
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
    env,
    agent_factory,
    opt_utility_per_round: float,
    n_trials: int,
    name: str = "req1",
) -> dict:
    """
    Multi-trial loop for Requirement 1.

    Parameters
    ----------
    env : SingleCampaignEnv
        Reused across trials; reset(seed=i) is called at the start of each.
    agent_factory : callable() -> agent
        Called once per trial to produce a fresh agent instance.
    opt_utility_per_round : float
        Clairvoyant per-round utility (computed once before the loop).
    n_trials : int
    name : str
        Stem for the pickle filename saved to data/picklefiles/.

    Returns
    -------
    dict
        mean_regret, std_regret, mean_cumcost (all shape (T,)), n_trials.
    """
    logger.info("Running %d trials – %s", n_trials, name)
    regret_per_trial = []
    payments_per_trial = []

    for i in range(n_trials):
        np.random.seed(i)
        env.reset(seed=i)
        agent = agent_factory()

        utilities = np.zeros(env.T)
        costs = np.zeros(env.T)

        for t in range(env.T):
            k = agent.pull_arm()
            f_t, c_t, _ = env.round(k)

            if hasattr(agent, "avg_f"):
                agent.update(f_t, c_t)
            else:
                agent.update(f_t)

            utilities[t] = f_t
            costs[t] = c_t

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
    env,
    agent_factory,
    opt_utility_per_round: float,
    n_trials: int,
    name: str = "req2",
) -> dict:
    """Multi-trial loop for Requirement 2. Same structure as run_single_campaign_trials."""
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
    results: dict,
    title: str = "Cumulative Pseudo-Regret",
    filename: str = "regret.png",
    add_reference: bool = False,
    upper_bound: tuple | None = None,
) -> None:
    """
    Plot mean cumulative pseudo-regret with ±stderr bands.

    Parameters
    ----------
    results      : dict   {label: {mean_regret, std_regret, n_trials}}
    add_reference: bool   draw an unscaled sqrt(t log t) guide
    upper_bound  : tuple(label, values), optional  true bound to overlay
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


# ---------------------------------------------------------------------------
# Requirements 3 and 4 -- non-stationary clairvoyants, multi-trial runners
# ---------------------------------------------------------------------------


def compute_clairvoyant_dynamic_multi(
    m_seq: np.ndarray,
    values: np.ndarray,
    bid_sets: list[np.ndarray],
    budget: float,
    conflict_edges: list[tuple[int, int]] | None = None,
) -> tuple[float, float]:
    """
    Dynamic clairvoyant ("prophet") oracle: knows every realised m_t in advance.

    Solves one LP over the full horizon:

        max  sum_{t,i,k} y_{t,i,k} (v_i - b_{i,k}) 1[b_{i,k} >= m_{t,i}]
        s.t. sum_{t,i,k} y_{t,i,k} b_{i,k} 1[b_{i,k} >= m_{t,i}] <= B
             sum_k y_{t,i,k} <= 1                     for all t, i
             sum_k y_{t,i,k} + sum_k y_{t,j,k} <= 1    for all t, (i,j) in E
             0 <= y_{t,i,k} <= 1

    This oracle's per-round foreknowledge of m_t means it can react to each
    specific draw within a stationary block, not just to the block's distribution.
    By Jensen's inequality, this inflates regret by a term linear in T regardless
    of learner quality (the prophet premium). It is kept as a reference upper-bound
    only; the primary Requirement 4 diagnostic is the piecewise expected clairvoyant.

    This LP has ~440k variables for T=10000, N=4, K=11. Use the precompute scripts
    to cache results rather than solving inside the trial loop.

    Returns
    -------
    opt_utility_total    : float   total utility over the horizon
    opt_utility_per_round: float   opt_utility_total / T
    """
    from scipy.sparse import csr_matrix

    m_seq = np.asarray(m_seq)
    N, T = m_seq.shape
    values = np.asarray(values, dtype=float)
    Ks = [len(bs) for bs in bid_sets]
    offsets = [0] + list(np.cumsum(Ks))
    NK = offsets[-1]
    n_vars = T * NK
    edges = conflict_edges or []

    f_flat = np.zeros(n_vars)
    c_flat = np.zeros(n_vars)
    for t in range(T):
        m_t = m_seq[:, t]
        for i in range(N):
            wins = bid_sets[i] >= m_t[i]
            base = t * NK + offsets[i]
            f_flat[base:base + Ks[i]] = (values[i] - bid_sets[i]) * wins
            c_flat[base:base + Ks[i]] = bid_sets[i] * wins

    rows, cols, data = [], [], []
    nz = np.where(c_flat > 0)[0]
    rows.extend([0] * len(nz)); cols.extend(nz.tolist()); data.extend(c_flat[nz].tolist())

    row_idx = 1
    for t in range(T):
        for i in range(N):
            base = t * NK + offsets[i]
            rows.extend([row_idx] * Ks[i])
            cols.extend(range(base, base + Ks[i]))
            data.extend([1.0] * Ks[i])
            row_idx += 1

    for t in range(T):
        for (ei, ej) in edges:
            base_i = t * NK + offsets[ei]
            base_j = t * NK + offsets[ej]
            rows.extend([row_idx] * (Ks[ei] + Ks[ej]))
            cols.extend(range(base_i, base_i + Ks[ei]))
            cols.extend(range(base_j, base_j + Ks[ej]))
            data.extend([1.0] * (Ks[ei] + Ks[ej]))
            row_idx += 1

    A_ub = csr_matrix((data, (rows, cols)), shape=(row_idx, n_vars))
    b_ub = np.empty(row_idx)
    b_ub[0] = budget
    b_ub[1:] = 1.0

    res = optimize.linprog(-f_flat, A_ub=A_ub, b_ub=b_ub, bounds=(0.0, 1.0), method="highs")
    if not res.success:
        logger.warning("Dynamic clairvoyant LP failed: %s", res.message)
        return 0.0, 0.0

    opt_total = -float(res.fun)
    logger.info("Dynamic clairvoyant (prophet) | T=%d N=%d total_utility=%.3f per_round=%.4f",
                T, N, opt_total, opt_total / T)
    return opt_total, opt_total / T


def _build_feasible_joint_actions(bid_sets, conflict_edges=None):
    """Enumerate joint actions with abstention (-1) and conflict constraints."""
    N = len(bid_sets)
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
    return actions


def compute_piecewise_expected_clairvoyant(env) -> tuple[float, np.ndarray]:
    """
    Piecewise expected clairvoyant — the PRIMARY Requirement 4 benchmark.

    Knows block boundaries and true block distributions (from
    env.piecewise_win_probabilities()) but NOT the realised m_t each round.
    Solves one LP with one mixed action per block, coupled by the shared budget:

        max_{x_{s,a}>=0}  sum_s |I_s| sum_a x_{s,a} E_s[utility(a)]
        s.t.              sum_s |I_s| sum_a x_{s,a} E_s[cost(a)] <= B
                          sum_a x_{s,a} = 1   for every block s

    This benchmark sits between OPT^A (one fixed distribution for the whole
    horizon, no regime awareness) and the dynamic/prophet oracle (knows every
    realised m_t). It rewards an agent for tracking regime changes without
    providing per-round foreknowledge — the natural target for SW-UCB and
    CUSUM-UCB, which have literature tracking guarantees against the best
    per-segment action (e.g. Garivier & Moulines 2011).

    Parameters
    ----------
    env : AdversarialMultiCampaignEnv with mode='shocks'

    Returns
    -------
    opt_total           : float           expected total utility over horizon
    expected_per_round  : np.ndarray (T,) blockwise expected utility per round
    """
    blocks = env.piecewise_win_probabilities()
    actions = _build_feasible_joint_actions(env.bid_sets, env.conflict_edges)
    n_blocks = len(blocks)
    n_actions = len(actions)
    n_vars = n_blocks * n_actions

    values = np.asarray(env.values, dtype=float)
    f = np.zeros(n_vars)
    c = np.zeros(n_vars)
    lengths = np.zeros(n_blocks, dtype=float)

    for s, (start, end, win_prob_list) in enumerate(blocks):
        length = end - start
        lengths[s] = length
        for a_idx, action in enumerate(actions):
            var_idx = s * n_actions + a_idx
            for i, k in enumerate(action):
                if k >= 0:
                    p_win = win_prob_list[i][k]
                    bid = env.bid_sets[i][k]
                    f[var_idx] += length * (values[i] - bid) * p_win
                    c[var_idx] += length * bid * p_win

    A_eq = np.zeros((n_blocks, n_vars))
    for s in range(n_blocks):
        A_eq[s, s * n_actions:(s + 1) * n_actions] = 1.0

    res = optimize.linprog(
        -f,
        A_ub=np.array([c]),
        b_ub=np.array([env.budget]),
        A_eq=A_eq,
        b_eq=np.ones(n_blocks),
        bounds=[(0.0, 1.0)] * n_vars,
        method="highs",
    )
    if not res.success:
        logger.warning("Piecewise expected clairvoyant LP failed: %s", res.message)
        return 0.0, np.zeros(env.T)

    gamma = np.clip(res.x, 0.0, 1.0)
    expected_per_round = np.zeros(env.T)
    for s, (start, end, _) in enumerate(blocks):
        sl = slice(s * n_actions, (s + 1) * n_actions)
        block_total_utility = float(np.dot(gamma[sl], f[sl]))
        expected_per_round[start:end] = block_total_utility / lengths[s]

    opt_total = -float(res.fun)
    logger.info("Piecewise expected clairvoyant | T=%d blocks=%d total_utility=%.3f per_round=%.4f",
                env.T, n_blocks, opt_total, opt_total / env.T)
    return opt_total, expected_per_round


def load_clairvoyant_cache(path_or_key):
    """
    Load a clairvoyant cache produced by precompute_clairvoyant.py.

    Parameters
    ----------
    path_or_key : str | Path
        Either an absolute path to the pickle, or just the 12-char hex
        key (the script will look for it in DATA_DIR).

    Returns
    -------
    dict[int, dict]  seed -> {opt_total, opt_per_round}, or {} if missing.
    """
    p = Path(path_or_key)
    if not p.is_absolute() and not p.exists():
        p = DATA_DIR / f"clairvoyant_dyn_{path_or_key}.pkl"
    if not p.exists():
        logger.warning("No cache at %s -- will compute on the fly (slow).", p)
        return {}
    with open(p, "rb") as f:
        cache = pickle.load(f)
    logger.info("Loaded clairvoyant cache from %s (%d entries)", p, len(cache))
    return cache


def plot_lambda(results: dict, title: str = r"Lagrange multiplier $\lambda_t$", filename: str = "lambda.png") -> None:
    """Plot the Lagrange multiplier trajectory, one line per agent."""
    fig, ax = plt.subplots(figsize=(9, 5))
    T = len(next(iter(results.values()))["mean_lmbd"])
    ts = np.arange(1, T + 1)
    for label, res in results.items():
        mean = res["mean_lmbd"]
        stderr = res["std_lmbd"] / np.sqrt(res["n_trials"])
        ax.plot(ts, mean, label=label)
        ax.fill_between(ts, mean - stderr, mean + stderr, alpha=0.25)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\lambda_t$")
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


def plot_resets_histogram(resets_per_trial, title="CUSUM resets per trial",
                           filename="cusum_resets.png"):
    """Histogram of total CUSUM resets fired per trial, across all cells."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = range(0, max(resets_per_trial + [1]) + 2)
    ax.hist(resets_per_trial, bins=bins)
    ax.set_xlabel("Total CUSUM resets fired (per trial)")
    ax.set_ylabel("Number of trials")
    ax.set_title(title)
    plt.tight_layout()
    path = OUTPUTS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    logger.info("Saved plot to %s", path)
    plt.close()


def run_primal_dual_trials(
    env_factory,
    agent_factory,
    n_trials: int,
    opt_per_round: float | None = None,
    name: str = "req3",
) -> dict:
    """
    Multi-trial loop for PrimalDualMultiCampaignAgent (Requirement 3).

    Baseline selection:
      - opt_per_round provided  → fixed per-round clairvoyant (stochastic regime).
      - opt_per_round is None   → OPT^A per trial: best FIXED distribution in
        hindsight, computed from env.empirical_win_probabilities() fed into
        compute_clairvoyant_multi. This is the benchmark a primal-dual
        (Hedge + OGD) agent has a provable sublinear-regret guarantee against
        in both stochastic and adversarial settings.

    Differences from run_multi_campaign_trials:
      - Calls agent.update(f_t, c_t, m_t) with full feedback.
      - Creates a fresh env per trial via env_factory(seed=i).
      - Tracks per-trial lambda trajectories for plotting.

    Parameters
    ----------
    env_factory    : callable(seed: int) -> env
    agent_factory  : callable() -> PrimalDualMultiCampaignAgent
    n_trials       : int
    opt_per_round  : float | None
    name           : str   pickle filename stem

    Returns
    -------
    dict with mean_regret, std_regret, mean_cumcost, mean_lmbd, std_lmbd (T,), n_trials.
    """
    mode = "stochastic (fixed OPT)" if opt_per_round is not None else "adversarial (per-trial OPT^A)"
    logger.info("Running %d trials - %s - %s", n_trials, name, mode)

    regret_per_trial = []
    payments_per_trial = []
    lmbd_per_trial = []

    for i in range(n_trials):
        np.random.seed(i)
        env = env_factory(seed=i)
        agent = agent_factory()

        if opt_per_round is not None:
            trial_opt = float(opt_per_round)
        else:
            win_probs = env.empirical_win_probabilities()
            _, trial_opt = compute_clairvoyant_multi(
                env.values, env.bid_sets, env.rho, win_probs, env.conflict_edges,
            )

        utilities = np.zeros(env.T)
        costs = np.zeros(env.T)

        for t in range(env.T):
            A_t = agent.pull_arm()
            f_t, c_t, m_t = env.round(A_t)
            agent.update(f_t, c_t, m_t)
            utilities[t] = f_t.sum()
            costs[t] = c_t.sum()

        regret_per_trial.append(np.cumsum(trial_opt - utilities))
        payments_per_trial.append(np.cumsum(costs))

        lmbds = np.asarray(agent.lmbds_history, dtype=float)
        if lmbds.size < env.T:
            pad_value = lmbds[-1] if lmbds.size > 0 else 0.0
            lmbds = np.pad(lmbds, (0, env.T - lmbds.size), constant_values=pad_value)
        lmbd_per_trial.append(lmbds[:env.T])

    regret_per_trial = np.array(regret_per_trial)
    payments_per_trial = np.array(payments_per_trial)
    lmbd_per_trial = np.array(lmbd_per_trial)

    out = dict(
        mean_regret=regret_per_trial.mean(axis=0),
        std_regret=regret_per_trial.std(axis=0),
        mean_cumcost=payments_per_trial.mean(axis=0),
        mean_lmbd=lmbd_per_trial.mean(axis=0),
        std_lmbd=lmbd_per_trial.std(axis=0),
        n_trials=n_trials,
    )
    path = DATA_DIR / f"{name}_results.pkl"
    with open(path, "wb") as f:
        pickle.dump(out, f)
    logger.info("Saved results to %s", path)
    return out


def run_nonstationary_trials(
    env_factory,
    agent_factory,
    n_trials: int,
    name: str = "req4",
    clairvoyant_cache: dict | None = None,
    compute_opt_a: bool = True,
    compute_piecewise: bool = True,
) -> dict:
    """
    Requirement 4 multi-trial loop with up to three benchmarks.

    Creates a fresh env and agent per trial; dispatches between semi-bandit
    agents (update(f, c)) and full-feedback agents (update(f, c, m)) based
    on whether the agent has a hedge_agents attribute.

    Benchmarks computed per trial:
      - dynamic/prophet  (mean_regret)         — always; cached if cache given.
        Reference upper bound only; see compute_clairvoyant_dynamic_multi.
      - piecewise expected  (mean_regret_piecewise) — if compute_piecewise=True
        and env supports mode='shocks'. PRIMARY Requirement 4 diagnostic.
      - OPT^A  (mean_regret_opt_a)             — if compute_opt_a=True.
        Same methodology as Requirement 3; kept for cross-requirement continuity.

    Parameters
    ----------
    env_factory       : callable(seed: int) -> env
    agent_factory     : callable() -> agent
    n_trials          : int
    name              : str   pickle filename stem
    clairvoyant_cache : dict {seed: {opt_per_round: float}}, optional
    compute_opt_a     : bool
    compute_piecewise : bool

    Returns
    -------
    dict with mean_regret, std_regret, mean_cumcost, n_trials, plus
    mean_regret_piecewise/std_regret_piecewise, mean_regret_opt_a/std_regret_opt_a,
    mean_lmbd/std_lmbd, and resets_per_trial (agent-dependent).
    """
    logger.info("Running %d trials - %s (cache=%s, opt_a=%s, piecewise=%s)",
                n_trials, name, "yes" if clairvoyant_cache else "no",
                compute_opt_a, compute_piecewise)

    regret_per_trial = []
    regret_piecewise_per_trial = []
    regret_opt_a_per_trial = []
    payments_per_trial = []
    lmbd_per_trial = []
    resets_per_trial = []

    for i in range(n_trials):
        np.random.seed(i)
        env = env_factory(seed=i)
        agent = agent_factory()
        full_feedback = hasattr(agent, "hedge_agents")

        if clairvoyant_cache is not None and i in clairvoyant_cache:
            trial_opt = float(clairvoyant_cache[i]["opt_per_round"])
        else:
            _, trial_opt = compute_clairvoyant_dynamic_multi(
                m_seq=env.m, values=env.values, bid_sets=env.bid_sets,
                budget=env.budget, conflict_edges=env.conflict_edges,
            )

        piecewise_expected = None
        if compute_piecewise and hasattr(env, "piecewise_win_probabilities"):
            try:
                _, piecewise_expected = compute_piecewise_expected_clairvoyant(env)
            except ValueError:
                piecewise_expected = None

        opt_a = None
        if compute_opt_a:
            win_probs = env.empirical_win_probabilities()
            _, opt_a = compute_clairvoyant_multi(
                env.values, env.bid_sets, env.rho, win_probs, env.conflict_edges,
            )

        utilities = np.zeros(env.T)
        costs = np.zeros(env.T)

        for t in range(env.T):
            A_t = agent.pull_arm()
            f_t, c_t, m_t = env.round(A_t)
            if full_feedback:
                agent.update(f_t, c_t, m_t)
            else:
                agent.update(f_t, c_t)
            utilities[t] = f_t.sum()
            costs[t] = c_t.sum()

        regret_per_trial.append(np.cumsum(trial_opt - utilities))
        if piecewise_expected is not None:
            regret_piecewise_per_trial.append(np.cumsum(piecewise_expected - utilities))
        if opt_a is not None:
            regret_opt_a_per_trial.append(np.cumsum(opt_a - utilities))
        payments_per_trial.append(np.cumsum(costs))

        if full_feedback:
            lmbds = np.asarray(agent.lmbds_history, dtype=float)
            if lmbds.size < env.T:
                pad_value = lmbds[-1] if lmbds.size > 0 else 0.0
                lmbds = np.pad(lmbds, (0, env.T - lmbds.size), constant_values=pad_value)
            lmbd_per_trial.append(lmbds[:env.T])

        if hasattr(agent, "n_resets"):
            resets_per_trial.append(sum(int(n.sum()) for n in agent.n_resets))

    regret_per_trial = np.array(regret_per_trial)
    payments_per_trial = np.array(payments_per_trial)

    out = dict(
        mean_regret=regret_per_trial.mean(axis=0),
        std_regret=regret_per_trial.std(axis=0),
        mean_cumcost=payments_per_trial.mean(axis=0),
        n_trials=n_trials,
    )
    if regret_piecewise_per_trial:
        arr = np.array(regret_piecewise_per_trial)
        out["mean_regret_piecewise"] = arr.mean(axis=0)
        out["std_regret_piecewise"] = arr.std(axis=0)
    if regret_opt_a_per_trial:
        arr = np.array(regret_opt_a_per_trial)
        out["mean_regret_opt_a"] = arr.mean(axis=0)
        out["std_regret_opt_a"] = arr.std(axis=0)
    if lmbd_per_trial:
        lmbd_arr = np.array(lmbd_per_trial)
        out["mean_lmbd"] = lmbd_arr.mean(axis=0)
        out["std_lmbd"] = lmbd_arr.std(axis=0)
    if resets_per_trial:
        out["resets_per_trial"] = resets_per_trial
        out["mean_resets"] = float(np.mean(resets_per_trial))

    path = DATA_DIR / f"{name}_results.pkl"
    with open(path, "wb") as f:
        pickle.dump(out, f)
    logger.info("Saved results to %s", path)
    return out
