"""
environments.py
---------------
Auction environments for Requirements 1 and 2.

Generative model follows NB07 exactly:
  - n_competitors bids are drawn i.i.d. from Uniform[0,1] each round
  - m_t = max of those bids  (NB07 cell 14)
  - the win-probability for bid b is therefore Beta(n_competitors, 1).cdf(b)
    (NB07 cell 22: "the maximum among k uniformly distributed r.v.s is a
     beta r.v. with alpha=k and beta=1")

Both environments pre-generate ALL competitor bids at __init__ time,
following the same pattern as BernoulliEnvironment in NB01 cell 4.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "picklefiles"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Requirement 1 – single campaign
# ---------------------------------------------------------------------------

class SingleCampaignEnv:
    """
    Stochastic first-price auction for one campaign.

    Follows the setup in NB07 cells 14 and 43:
      - n_competitors competitors each bid ~ Uniform[0, 1] i.i.d.
      - m_t = max of their bids
      - learner wins if bid >= m_t, pays bid, earns (value - bid)

    All competitor bids are pre-generated at init (NB01 cell 4 pattern).

    Parameters
    ----------
    value : float
        Learner's value per won auction.
    budget : float
        Total budget B.
    T : int
        Time horizon.
    available_bids : np.ndarray
        Discrete bid set. Bids > value are excluded here, following
        NB07 cell 43: good_bids = available_bids[available_bids <= my_valuation]
    n_competitors : int
        Number of competitors (k in Beta(k,1)). Default 3 matches NB07.
    seed : int, optional
    """

    def __init__(self, value, budget, T, available_bids,
                 n_competitors=3, seed=None):
        self.value = value
        self.budget = budget
        self.T = T
        # NB07 cell 43: restrict to bids <= value
        self.available_bids = np.asarray(available_bids)
        self.available_bids = self.available_bids[self.available_bids <= value]
        self.K = len(self.available_bids)
        self.n_competitors = n_competitors
        self.rho = budget / T
        self.seed = seed

        rng = np.random.default_rng(seed)
        # NB07 cell 14: other_bids ~ Uniform[0,1], shape (n_competitors, T)
        self.other_bids = rng.uniform(0, 1, size=(n_competitors, T))
        # m_t = max competing bid each round
        self.m = self.other_bids.max(axis=0)
        self.t = 0

        logger.info(
            "SingleCampaignEnv | value=%.2f B=%.1f T=%d K=%d rho=%.4f n_comp=%d",
            value, budget, T, self.K, self.rho, n_competitors,
        )

    def round(self, bid_index):
        """
        Play one round.

        Parameters
        ----------
        bid_index : int
            Index into self.available_bids.

        Returns
        -------
        f_t : float   (value - bid) * I[won]   — NB07 cell 43
        c_t : float   bid * I[won]              — NB07 cell 43
        m_t : float   max competing bid revealed after the round
        """
        if self.t >= self.T:
            raise RuntimeError(f"Episode finished after T={self.T} rounds.")
        bid = self.available_bids[bid_index]
        m_t = self.m[self.t]
        my_win = int(bid >= m_t)                   # NB07 cell 43
        f_t = (self.value - bid) * my_win          # NB07 cell 43
        c_t = bid * my_win                         # NB07 cell 43
        self.t += 1
        return f_t, c_t, m_t

    def reset(self, seed=None):
        """Re-draw competitor bids and reset the round counter (for multi-trial loops)."""
        s = seed if seed is not None else self.seed
        rng = np.random.default_rng(s)
        self.other_bids = rng.uniform(0, 1, size=(self.n_competitors, self.T))
        self.m = self.other_bids.max(axis=0)
        self.t = 0

    def win_probabilities(self):
        """
        Exact P(bid >= m) per bid via Beta(k,1) CDF.
        NB07 cell 22: stats.beta.cdf(available_bids, n_advertisers, 1)
        """
        return stats.beta.cdf(self.available_bids, self.n_competitors, 1)

    def save(self, name="single_campaign_env"):
        path = DATA_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved env to %s", path)
        return path


# ---------------------------------------------------------------------------
# Requirement 2 – multiple campaigns with conflict graph
# ---------------------------------------------------------------------------

class MultiCampaignEnv:
    """
    N independent first-price auctions with a shared budget and a conflict
    graph.

    Each campaign i has its own set of competitors bidding Uniform[0,1].
    All competitor bids are pre-generated at init.

    Conflict graph
    --------------
    Two campaigns connected by an edge cannot both be bid on in the same round
    (project spec p.7). Invalid bid vectors are rejected.

    Parameters
    ----------
    values : list[float]
        Per-campaign values v_i.
    budget : float
        Total shared budget B.
    T : int
        Time horizon.
    available_bids : np.ndarray
        Discrete bid set shared across campaigns (before per-campaign filtering).
    n_competitors : list[int], optional
        Competitors per campaign.  Default 3 each (NB07 default).
    conflict_edges : list[tuple[int, int]], optional
    seed : int, optional
    """

    def __init__(self, values, budget, T, available_bids,
                 n_competitors=None, conflict_edges=None, seed=None):
        self.values = np.asarray(values)
        self.N = len(values)
        self.budget = budget
        self.T = T
        self.rho = budget / T
        self.conflict_edges = conflict_edges or []
        self.seed = seed

        n_comp = n_competitors if n_competitors is not None else [3] * self.N
        self.n_competitors = n_comp

        # Per-campaign bid sets: bids <= v_i  (NB07 cell 43)
        all_bids = np.asarray(available_bids)
        self.bid_sets = [all_bids[all_bids <= v] for v in self.values]
        self.Ks = [len(bs) for bs in self.bid_sets]

        rng = np.random.default_rng(seed)
        # Shape: list of arrays (n_comp_i, T)
        self.other_bids = [
            rng.uniform(0, 1, size=(n_comp[i], T))
            for i in range(self.N)
        ]
        # m[i, t] = max competing bid for campaign i at round t
        self.m = np.vstack([ob.max(axis=0) for ob in self.other_bids])
        self.t = 0

        logger.info(
            "MultiCampaignEnv | N=%d T=%d B=%.1f rho=%.4f conflict_edges=%s",
            self.N, T, budget, self.rho, self.conflict_edges,
        )

    def round(self, bid_indices):
        """
        Play one round across all campaigns.

        Parameters
        ----------
        bid_indices : list[int]  length N
            bid_indices[i] = index into self.bid_sets[i], or -1 to abstain.

        Returns
        -------
        f_t : np.ndarray shape (N,)  per-campaign utility
        c_t : np.ndarray shape (N,)  per-campaign cost
        m_t : np.ndarray shape (N,)  max competing bids
        """
        if self.t >= self.T:
            raise RuntimeError(f"Episode finished after T={self.T} rounds.")

        m_t = self.m[:, self.t]

        bids = np.array([
            self.bid_sets[i][bid_indices[i]] if bid_indices[i] >= 0
            else -1.0
            for i in range(self.N)
        ])

        active = bids >= 0
        for (ei, ej) in self.conflict_edges:
            if active[ei] and active[ej]:
                raise ValueError(
                    f"Campaigns {ei} and {ej} are incompatible and cannot "
                    "both receive bids in the same round."
                )

        won = (bids >= m_t) & active

        f_t = np.where(won, self.values - bids, 0.0)
        c_t = np.where(won, bids, 0.0)
        self.t += 1
        return f_t, c_t, m_t

    def reset(self, seed=None):
        s = seed if seed is not None else self.seed
        rng = np.random.default_rng(s)
        self.other_bids = [
            rng.uniform(0, 1, size=(self.n_competitors[i], self.T))
            for i in range(self.N)
        ]
        self.m = np.vstack([ob.max(axis=0) for ob in self.other_bids])
        self.t = 0

    def win_probabilities(self):
        """
        True P(b >= m_i) per campaign and bid.
        Returns list of np.ndarray, one per campaign (NB07 cell 22).
        """
        return [
            stats.beta.cdf(self.bid_sets[i], self.n_competitors[i], 1)
            for i in range(self.N)
        ]

    def save(self, name="multi_campaign_env"):
        path = DATA_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved env to %s", path)
        return path


# ---------------------------------------------------------------------------
# Requirement 3 – multiple campaigns, HIGHLY non-stationary environment
# ---------------------------------------------------------------------------

class AdversarialMultiCampaignEnv:
    """
    N first-price auctions with a shared budget, conflict graph, and a
    HIGHLY non-stationary sequence of highest competing bids m_t.

    Project spec (Requirement 3, p.15):
      "Build a highly non-stationary environment. At a high level, it should
       include a non-stochastic sequence of highest competing bids for each
       campaign (e.g., sampled from a distribution that changes quickly
       over time)."

    Drop-in compatible with `MultiCampaignEnv`:
      - same `round(bid_indices)` -> (f_t, c_t, m_t) signature
      - same conflict-graph handling (tie-break on utility)
      - same pre-generation pattern so trials are reproducible

    The ONLY difference is how the (N, T) array `self.m` is built. Two
    modes are provided:

      'drift'  : m_t ~ Beta(alpha_t, beta_t) whose mean drifts as a high-
                 frequency sinusoid -> the distribution changes every round.
                 Per-campaign random phase shift desynchronises the N
                 campaigns. The 'highly non-stationary' default.

      'shocks' : the horizon is partitioned in tiny blocks of length
                 `block_size`. In each block, m_t is i.i.d. from a regime
                 drawn uniformly at random from a pre-built menu of
                 `n_regimes` Beta distributions. Many regime switches per
                 trajectory => non-stationary at all timescales.

    Note on full feedback (project spec p.16): the env *returns* m_t after
    the round, exactly like MultiCampaignEnv. Any bidding strategy can
    therefore reconstruct counterfactual rewards (v_i - b)*1[b >= m_t] and
    costs b*1[b >= m_t] for every b in the bid set -- this is the full-
    feedback signal the primal-dual agent of Req 3 needs.

    Parameters
    ----------
    values : list[float]
        Per-campaign values v_i.
    budget : float
        Total shared budget B.
    T : int
        Time horizon.
    available_bids : np.ndarray
        Discrete bid set shared across campaigns (before per-campaign
        filtering by v_i).
    conflict_edges : list[tuple[int,int]], optional
    seed : int, optional
    mode : {'drift', 'shocks'}
    drift_cycles : float
        For 'drift'. Number of full sinusoid periods in [0, T]. Higher =
        faster non-stationarity. Default 10 (period ~ T/10).
    drift_amplitude : float
        For 'drift'. Half-range of the mean's sinusoid in (0, 0.5). The
        Beta mean oscillates in [base_mean - amp, base_mean + amp].
    base_mean : float
        For 'drift'. Center of the Beta-mean sinusoid. Default 0.5.
    beta_concentration : float
        Sum alpha+beta of the Beta. Higher => more peaked m_t around its
        current mean. Shared across regimes in 'shocks' mode.
    block_size : int
        For 'shocks'. Length of each piecewise-constant regime.
    n_regimes : int
        For 'shocks'. Size of the menu of (alpha, beta) regimes.
    """

    SUPPORTED_MODES = ("drift", "shocks")

    def __init__(self, values, budget, T, available_bids,
                 conflict_edges=None, seed=None,
                 mode="drift",
                 drift_cycles=10.0,
                 drift_amplitude=0.35,
                 base_mean=0.5,
                 beta_concentration=8.0,
                 block_size=25,
                 n_regimes=4):

        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"mode={mode!r} not in {self.SUPPORTED_MODES}"
            )

        self.values = np.asarray(values, dtype=float)
        self.N = len(values)
        self.budget = budget
        self.T = T
        self.rho = budget / T
        self.conflict_edges = conflict_edges or []
        self.seed = seed
        self.mode = mode
        self.drift_cycles = drift_cycles
        self.drift_amplitude = drift_amplitude
        self.base_mean = base_mean
        self.beta_concentration = beta_concentration
        self.block_size = block_size
        self.n_regimes = n_regimes

        # Per-campaign bid sets: bids <= v_i  (NB07 cell 43, same as Req 2)
        all_bids = np.asarray(available_bids, dtype=float)
        self.bid_sets = [all_bids[all_bids <= v] for v in self.values]
        self.Ks = [len(bs) for bs in self.bid_sets]

        # Build the (N, T) sequence of m_t
        rng = np.random.default_rng(seed)
        if mode == "drift":
            self.m = self._build_drift(rng)
        else:  # 'shocks'
            self.m = self._build_shocks(rng)

        self.t = 0

        logger.info(
            "AdversarialMultiCampaignEnv | N=%d T=%d B=%.1f rho=%.4f "
            "mode=%s conflict_edges=%s",
            self.N, T, budget, self.rho, mode, self.conflict_edges,
        )

    # ---- m_t generators --------------------------------------------------

    def _build_drift(self, rng):
        """
        Per-round Beta(alpha_t, beta_t) sampling, with mean drifting as a
        high-frequency sinusoid. Each campaign gets its own phase shift so
        the N campaigns are NOT synchronised (more realistic / harder).
        """
        T = self.T
        ts = np.arange(T)
        phases = rng.uniform(0, 2 * np.pi, size=self.N)
        m = np.empty((self.N, T))
        s = self.beta_concentration  # alpha + beta
        for i in range(self.N):
            angle = 2 * np.pi * self.drift_cycles * ts / T + phases[i]
            mean_t = self.base_mean + self.drift_amplitude * np.sin(angle)
            mean_t = np.clip(mean_t, 1e-3, 1 - 1e-3)
            alpha_t = mean_t * s
            beta_t = (1.0 - mean_t) * s
            m[i] = rng.beta(alpha_t, beta_t)
        return m

    def _build_shocks(self, rng):
        """
        Piecewise-stationary in tiny blocks. n_regimes random (alpha, beta)
        pairs are pre-drawn at init; each block picks one uniformly at
        random and samples i.i.d. from it.
        """
        T = self.T
        regime_means = rng.uniform(0.1, 0.9, size=self.n_regimes)
        s = self.beta_concentration
        regimes = [(mu * s, (1 - mu) * s) for mu in regime_means]

        m = np.empty((self.N, T))
        n_blocks = (T + self.block_size - 1) // self.block_size
        for i in range(self.N):
            for b in range(n_blocks):
                start = b * self.block_size
                end = min(start + self.block_size, T)
                a, bb = regimes[rng.integers(0, self.n_regimes)]
                m[i, start:end] = rng.beta(a, bb, size=end - start)
        return m

    # ---- Same interaction protocol as MultiCampaignEnv -------------------

    def round(self, bid_indices):
        """
        Play one round across all campaigns.

        Parameters
        ----------
        bid_indices : list[int]   length N
            bid_indices[i] = index into self.bid_sets[i], or -1 to abstain.

        Returns
        -------
        f_t : np.ndarray (N,)    per-campaign utility (v_i - b_i) * 1[won]
        c_t : np.ndarray (N,)    per-campaign cost     b_i * 1[won]
        m_t : np.ndarray (N,)    max competing bids (full-feedback signal)
        """
        if self.t >= self.T:
            raise RuntimeError(f"Episode finished after T={self.T} rounds.")

        m_t = self.m[:, self.t]
        bids = np.array([
            self.bid_sets[i][bid_indices[i]] if bid_indices[i] >= 0 else -1.0
            for i in range(self.N)
        ])
        won = (bids >= m_t) & (bids >= 0)

        # Conflict graph: keep only the higher-utility winner per edge
        for (ei, ej) in self.conflict_edges:
            if won[ei] and won[ej]:
                u_i = self.values[ei] - bids[ei]
                u_j = self.values[ej] - bids[ej]
                if u_i >= u_j:
                    won[ej] = False
                else:
                    won[ei] = False

        f_t = np.where(won, self.values - bids, 0.0)
        c_t = np.where(won, bids, 0.0)
        self.t += 1
        return f_t, c_t, m_t

    def reset(self, seed=None):
        """Re-draw the m sequence and reset the round counter."""
        s = seed if seed is not None else self.seed
        rng = np.random.default_rng(s)
        if self.mode == "drift":
            self.m = self._build_drift(rng)
        else:  # 'shocks'
            self.m = self._build_shocks(rng)
        self.t = 0

    def empirical_win_probabilities(self):
        """
        For non-stationary envs there is no SINGLE win-probability per bid.
        Returns the *empirical* P(b >= m_i) over the full horizon -- this
        is what a 'best fixed bid in hindsight' baseline optimises on.
        Returns list[np.ndarray], one per campaign.
        """
        return [
            (self.bid_sets[i][:, None] >= self.m[i][None, :]).mean(axis=1)
            for i in range(self.N)
        ]

    def save(self, name="adversarial_multi_campaign_env"):
        path = DATA_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved env to %s", path)
        return path
