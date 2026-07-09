"""
environments.py
---------------
Auction environments for Requirements 1 through 4.

Generative model: n_competitors bids are drawn i.i.d. from Uniform[0,1] each
round; m_t = max of those bids. The win probability for bid b is therefore
P(b >= m_t) = Beta(n_competitors, 1).cdf(b), i.e. b^k for k competitors.

All environments pre-generate the full competitor-bid array at construction
time so that trials are reproducible and the round() method is O(1).
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

    Each round t, n_competitors competitors bid i.i.d. from Uniform[0,1].
    The learner wins if its bid >= m_t (max competing bid) and earns
    (value - bid); cost is bid * 1[won].

    Parameters
    ----------
    value : float
        Learner's value per won auction.
    budget : float
        Total budget B.
    T : int
        Time horizon.
    available_bids : np.ndarray
        Discrete bid set. Bids strictly above value are excluded.
    n_competitors : int
        Number of competitors k; win probability = Beta(k, 1).cdf(bid).
    seed : int, optional
    """

    def __init__(
        self,
        value: float,
        budget: float,
        T: int,
        available_bids: np.ndarray,
        n_competitors: int = 3,
        seed: int | None = None,
    ) -> None:
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
        self.other_bids = rng.uniform(0, 1, size=(n_competitors, T))
        self.m = self.other_bids.max(axis=0)
        self.t = 0

        logger.info(
            "SingleCampaignEnv | value=%.2f B=%.1f T=%d K=%d rho=%.4f n_comp=%d",
            value, budget, T, self.K, self.rho, n_competitors,
        )

    def round(self, bid_index: int) -> tuple[float, float, float]:
        """
        Play one round.

        Parameters
        ----------
        bid_index : int
            Index into self.available_bids.

        Returns
        -------
        f_t : float   (value - bid) * 1[won]
        c_t : float   bid * 1[won]
        m_t : float   max competing bid revealed after the round
        """
        if self.t >= self.T:
            raise RuntimeError(f"Episode finished after T={self.T} rounds.")
        bid = self.available_bids[bid_index]
        m_t = self.m[self.t]
        my_win = int(bid >= m_t)
        f_t = (self.value - bid) * my_win
        c_t = bid * my_win
        self.t += 1
        return f_t, c_t, m_t

    def reset(self, seed=None):
        """Re-draw competitor bids and reset the round counter (for multi-trial loops)."""
        s = seed if seed is not None else self.seed
        rng = np.random.default_rng(s)
        self.other_bids = rng.uniform(0, 1, size=(self.n_competitors, self.T))
        self.m = self.other_bids.max(axis=0)
        self.t = 0

    def win_probabilities(self) -> np.ndarray:
        """Exact P(bid >= m) per bid: Beta(n_competitors, 1).cdf(bid)."""
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
    N independent first-price auctions with a shared budget and a conflict graph.

    Each campaign i has its own competitors bidding Uniform[0,1]. Competitor
    bids are pre-generated at construction. Campaigns connected by a conflict
    edge cannot both receive bids in the same round; invalid bid vectors raise
    ValueError.

    Parameters
    ----------
    values : list[float]
        Per-campaign values v_i.
    budget : float
        Total shared budget B.
    T : int
        Time horizon.
    available_bids : np.ndarray
        Discrete bid set shared across campaigns (bids > v_i excluded per campaign).
    n_competitors : list[int], optional
        Competitors per campaign. Default: 3 each.
    conflict_edges : list[tuple[int, int]], optional
    seed : int, optional
    """

    def __init__(
        self,
        values: list[float],
        budget: float,
        T: int,
        available_bids: np.ndarray,
        n_competitors: list[int] | None = None,
        conflict_edges: list[tuple[int, int]] | None = None,
        seed: int | None = None,
    ) -> None:
        self.values = np.asarray(values)
        self.N = len(values)
        self.budget = budget
        self.T = T
        self.rho = budget / T
        self.conflict_edges = conflict_edges or []
        self.seed = seed

        n_comp = n_competitors if n_competitors is not None else [3] * self.N
        self.n_competitors = n_comp

        all_bids = np.asarray(available_bids)
        self.bid_sets = [all_bids[all_bids <= v] for v in self.values]
        self.Ks = [len(bs) for bs in self.bid_sets]

        rng = np.random.default_rng(seed)
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

    def win_probabilities(self) -> list[np.ndarray]:
        """
        Exact P(b >= m_i) per campaign and bid via Beta(k_i, 1).cdf.

        Returns list of np.ndarray, one array per campaign.
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
    non-stationary sequence of highest competing bids m_t.

    Drop-in compatible with MultiCampaignEnv (same round() signature and
    conflict-graph handling). The only difference is how the (N, T) array
    self.m is built. Two modes are supported:

      'drift'  : m_t ~ Beta(alpha_t, beta_t) whose mean follows a high-
                 frequency sinusoid with campaign-specific random phase
                 shifts. The distribution changes every round.

      'shocks' : the horizon is partitioned into blocks of length block_size.
                 Each block draws m_t i.i.d. from one of n_regimes Beta
                 distributions chosen uniformly at random. Suitable for
                 Requirement 4's piecewise-stationary setting.

    Full feedback: round() returns m_t after each round, enabling any agent
    to reconstruct counterfactual rewards (v_i - b)*1[b >= m_t] and costs
    b*1[b >= m_t] for all bids in the bid set.

    Parameters
    ----------
    values : list[float]
        Per-campaign values v_i.
    budget : float
        Total shared budget B.
    T : int
        Time horizon.
    available_bids : np.ndarray
        Discrete bid set shared across campaigns (bids > v_i excluded per campaign).
    conflict_edges : list[tuple[int, int]], optional
    seed : int, optional
    mode : {'drift', 'shocks'}
    drift_cycles : float
        For 'drift'. Number of full sinusoid periods in [0, T].
    drift_amplitude : float
        For 'drift'. Half-amplitude of the sinusoidal mean (in (0, 0.5)).
    base_mean : float
        For 'drift'. Centre of the sinusoidal mean. Default 0.5.
    beta_concentration : float
        Beta concentration parameter (alpha + beta). Shared across modes.
    block_size : int
        For 'shocks'. Number of rounds per piecewise-constant block.
    n_regimes : int
        For 'shocks'. Number of distinct Beta regimes to sample from.
    """

    SUPPORTED_MODES = ("drift", "shocks")

    def __init__(
        self,
        values: list[float],
        budget: float,
        T: int,
        available_bids: np.ndarray,
        conflict_edges: list[tuple[int, int]] | None = None,
        seed: int | None = None,
        mode: str = "shocks",
        drift_cycles: float = 10.0,
        drift_amplitude: float = 0.35,
        base_mean: float = 0.5,
        beta_concentration: float = 8.0,
        block_size: int = 25,
        n_regimes: int = 4,
    ) -> None:

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

        all_bids = np.asarray(available_bids, dtype=float)
        self.bid_sets = [all_bids[all_bids <= v] for v in self.values]
        self.Ks = [len(bs) for bs in self.bid_sets]

        # Build the (N, T) sequence of m_t
        rng = np.random.default_rng(seed)
        self.shock_blocks = None   # set by _build_shocks; used by piecewise_win_probabilities
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

    def _build_drift(self, rng: np.random.Generator) -> np.ndarray:
        """
        Sample m_t from Beta(alpha_t, beta_t) per round, where the mean follows
        a high-frequency sinusoid with a campaign-specific random phase offset
        so the N campaigns are not synchronised.
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

    def _build_shocks(self, rng: np.random.Generator) -> np.ndarray:
        """
        Piecewise-stationary in blocks. n_regimes random Beta(alpha, beta)
        distributions are pre-drawn; each block i.i.d. samples from one chosen
        uniformly at random. Records (start, end, alpha, beta) per (campaign, block)
        in self.shock_blocks so the piecewise expected clairvoyant can be computed
        analytically from the true parameters rather than empirical estimates.
        """
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
        self.shock_blocks = None
        if self.mode == "drift":
            self.m = self._build_drift(rng)
        else:  # 'shocks'
            self.m = self._build_shocks(rng)
        self.t = 0

    def empirical_win_probabilities(self) -> list[np.ndarray]:
        """
        Empirical P(b >= m_i) over the full realised horizon per campaign.

        In a non-stationary environment there is no single true win probability;
        this time-average is what the best fixed bid distribution in hindsight
        (OPT^A) optimises against.

        Returns list[np.ndarray], one array per campaign.
        """
        return [
            (self.bid_sets[i][:, None] >= self.m[i][None, :]).mean(axis=1)
            for i in range(self.N)
        ]

    def piecewise_win_probabilities(self) -> list[tuple[int, int, list[np.ndarray]]]:
        """
        Analytical P(b >= m_i) per stationary block, for mode='shocks' only.

        Uses the exact Beta(alpha, beta) parameters that generated each block
        (recorded in self.shock_blocks) to compute the true win probability
        analytically, without sampling noise or per-round foreknowledge of m_t.

        This is the distributional input to compute_piecewise_expected_clairvoyant:
        an oracle that knows block boundaries and true block distributions, but
        not the realised competing bids — the natural benchmark for a piecewise-
        stationary environment against which SW-UCB and CUSUM-UCB have literature
        tracking guarantees (e.g. Garivier & Moulines 2011).

        Returns
        -------
        list[tuple[int, int, list[np.ndarray]]]
            One (start, end, win_prob_list) tuple per block, where
            win_prob_list[i] gives P(b >= m_i) for every b in self.bid_sets[i].
        """
        if self.mode != "shocks" or self.shock_blocks is None:
            raise ValueError("piecewise_win_probabilities is only available for mode='shocks'.")

        from scipy.stats import beta as beta_dist

        n_blocks = len(self.shock_blocks[0])
        out = []
        for b in range(n_blocks):
            start, end = self.shock_blocks[0][b][:2]
            block_probs = []
            for i in range(self.N):
                i_start, i_end, alpha, beta_param = self.shock_blocks[i][b]
                if (i_start, i_end) != (start, end):
                    raise RuntimeError("Shock block boundaries are inconsistent across campaigns.")
                block_probs.append(beta_dist.cdf(self.bid_sets[i], alpha, beta_param))
            out.append((start, end, block_probs))
        return out

    def save(self, name="adversarial_multi_campaign_env"):
        path = DATA_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved env to %s", path)
        return path
