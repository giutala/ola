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
