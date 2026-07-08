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
        if len(bid_indices) != self.N:
            raise ValueError(
                f"Expected {self.N} bid indices, got {len(bid_indices)}."
            )
        for i, k in enumerate(bid_indices):
            if k < -1 or k >= self.Ks[i]:
                raise ValueError(
                    f"Invalid bid index {k} for campaign {i}; expected -1 "
                    f"or an index in [0, {self.Ks[i] - 1}]."
                )

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
# Requirements 3 and 4 -- non-stationary, N campaigns, conflict graph
# ---------------------------------------------------------------------------

class NonStationaryMultiCampaignEnv:
    """
    N first-price auctions, shared budget, conflict graph, and a highest
    competing bid m_t whose distribution changes over time.

    Two generation modes:

      'drift'  : m_t ~ Beta(alpha_t, beta_t), whose mean follows a
                 sinusoid that completes `drift_cycles` full periods over
                 the horizon -- the distribution moves EVERY round. Each
                 campaign gets its own random phase so the N campaigns are
                 not synchronised. This is the "highly" non-stationary
                 regime (Requirement 3).

      'shocks' : the horizon is cut into blocks of `block_size` rounds.
                 Each block draws one of `n_regimes` pre-built Beta
                 regimes uniformly at random and samples m_t i.i.d. from
                 it for the whole block -- i.e. EXACTLY the project spec
                 for Requirement 4 (p.18): "rounds are partitioned in
                 intervals, in each interval the distribution ... is
                 fixed, each interval has a different distribution."
                 Few, long blocks = "slightly" non-stationary (R4); many,
                 short blocks = "highly" non-stationary (an alternative to
                 'drift' for R3).

    Conflict resolution: unlike MultiCampaignEnv (which raises on a
    violation and expects the agent to never produce one), this class
    resolves conflicts itself by keeping the higher-utility winner --
    Requirement 3's primal-dual agent computes COUNTERFACTUAL rewards for
    every bid of every campaign (full feedback), so it needs the
    environment's realised outcome to reflect the same resolution rule
    its own update() assumes, rather than raising and refusing to proceed.

    Parameters
    ----------
    values, budget, T, available_bids, conflict_edges, seed
        Same as MultiCampaignEnv.
    mode : {'drift', 'shocks'}
    drift_cycles, drift_amplitude, base_mean, beta_concentration
        'drift' parameters -- see _build_drift.
    block_size, n_regimes
        'shocks' parameters -- see _build_shocks.
    """

    SUPPORTED_MODES = ("drift", "shocks")

    def __init__(self, values, budget, T, available_bids,
                 conflict_edges=None, seed=None,
                 mode="drift",
                 drift_cycles=10.0, drift_amplitude=0.35, base_mean=0.5,
                 beta_concentration=8.0,
                 block_size=25, n_regimes=4):
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"mode={mode!r} not in {self.SUPPORTED_MODES}")

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

        all_bids = np.asarray(available_bids, dtype=float)
        self.bid_sets = [all_bids[all_bids <= v] for v in self.values]
        self.Ks = [len(bs) for bs in self.bid_sets]

        self._draw(seed)
        self.t = 0

        logger.info(
            "NonStationaryMultiCampaignEnv | N=%d T=%d B=%.1f rho=%.4f mode=%s "
            "conflict_edges=%s",
            self.N, T, budget, self.rho, mode, self.conflict_edges,
        )

    def _draw(self, seed):
        rng = np.random.default_rng(seed)
        self.shock_blocks = None
        if self.mode == "drift":
            self.m = self._build_drift(rng)
        else:
            self.m = self._build_shocks(rng)

    def _build_drift(self, rng):
        T = self.T
        ts = np.arange(T)
        phases = rng.uniform(0, 2 * np.pi, size=self.N)
        s = self.beta_concentration
        m = np.empty((self.N, T))
        for i in range(self.N):
            angle = 2 * np.pi * self.drift_cycles * ts / T + phases[i]
            mean_t = np.clip(self.base_mean + self.drift_amplitude * np.sin(angle), 1e-3, 1 - 1e-3)
            m[i] = rng.beta(mean_t * s, (1.0 - mean_t) * s)
        return m

    def _build_shocks(self, rng):
        T = self.T
        regime_means = rng.uniform(0.1, 0.9, size=self.n_regimes)
        s = self.beta_concentration
        regimes = [(mu * s, (1 - mu) * s) for mu in regime_means]

        m = np.empty((self.N, T))
        n_blocks = (T + self.block_size - 1) // self.block_size
        self.shock_blocks = []
        for i in range(self.N):
            campaign_blocks = []
            for b in range(n_blocks):
                start = b * self.block_size
                end = min(start + self.block_size, T)
                a, bb = regimes[rng.integers(0, self.n_regimes)]
                campaign_blocks.append((start, end, float(a), float(bb)))
                m[i, start:end] = rng.beta(a, bb, size=end - start)
            self.shock_blocks.append(campaign_blocks)
        return m

    def piecewise_win_probabilities(self):
        """
        Return analytical P(b >= m_i) for each shock block.

        This is the distributional counterpart of `empirical_win_probabilities`:
        it uses the Beta parameters that generated each stationary block,
        rather than the realised sampled values m_{i,t}.
        """
        if self.mode != "shocks" or self.shock_blocks is None:
            raise ValueError("piecewise_win_probabilities is only available for mode='shocks'.")

        from scipy.stats import beta

        n_blocks = len(self.shock_blocks[0])
        out = []
        for b in range(n_blocks):
            start, end = self.shock_blocks[0][b][:2]
            block_probs = []
            for i in range(self.N):
                i_start, i_end, alpha, beta_param = self.shock_blocks[i][b]
                if (i_start, i_end) != (start, end):
                    raise RuntimeError("Shock block boundaries are inconsistent across campaigns.")
                block_probs.append(beta.cdf(self.bid_sets[i], alpha, beta_param))
            out.append((start, end, block_probs))
        return out

    def round(self, bid_indices):
        if self.t >= self.T:
            raise RuntimeError(f"Episode finished after T={self.T} rounds.")

        m_t = self.m[:, self.t]
        bids = np.array([
            self.bid_sets[i][bid_indices[i]] if bid_indices[i] >= 0 else -1.0
            for i in range(self.N)
        ])
        won = (bids >= m_t) & (bids >= 0)

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
        self._draw(seed if seed is not None else self.seed)
        self.t = 0

    def empirical_win_probabilities(self, start=0, end=None):
        """
        Empirical P(b >= m_i) over rounds [start, end) -- used by the
        piecewise / dynamic clairvoyant, since there is no single true
        win-probability once the environment is non-stationary.
        """
        end = self.T if end is None else end
        m_slice = self.m[:, start:end]
        return [
            (self.bid_sets[i][:, None] >= m_slice[i][None, :]).mean(axis=1)
            for i in range(self.N)
        ]
