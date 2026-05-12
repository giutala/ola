"""
agents.py
---------
Bidding agents for Requirements 1 and 2.

Every class mirrors the corresponding notebook implementation as closely
as possible.

Requirement 1
-------------
UCB1BiddingAgent      ← NB01 cell 39  (UCB1Agent, range=value)
UCBLikeBiddingAgent   ← NB07 cell 40  (UCBLikeAgent, range=value)

Requirement 2
-------------
CombinatorialUCBAgent ← NB09 cell 30  (UCBMatchingAgent) adapted for
                        N campaigns with a shared budget LP oracle
                        instead of linear_sum_assignment
"""

import logging
import pickle
from pathlib import Path

import numpy as np
from scipy import optimize

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "picklefiles"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Requirement 1 – UCB1 (budget-unaware)
# ---------------------------------------------------------------------------

class UCB1BiddingAgent:
    """
    UCB1 applied to the bid set, ignoring the budget constraint.

    Mirrors NB01 cell 39 (UCB1Agent) with range = value, following
    NB07 cell 43 where range=my_valuation is used.

    Parameters
    ----------
    K : int     number of available bids
    T : int     time horizon
    range : float   reward range; set to value (NB07 cell 43)
    """

    def __init__(self, K, T, range=1):
        # NB01 cell 39 – field names kept identical
        self.K = K
        self.T = T
        self.range = range
        self.a_t = None
        self.average_rewards = np.zeros(K)
        self.N_pulls = np.zeros(K)
        self.t = 0
        logger.info("UCB1BiddingAgent | K=%d T=%d range=%.2f", K, T, range)

    def pull_arm(self):
        """NB01 cell 39: pull each arm once first, then follow UCB."""
        if self.t < self.K:
            self.a_t = self.t
        else:
            ucbs = (self.average_rewards
                    + self.range * np.sqrt(2 * np.log(self.T) / self.N_pulls))
            self.a_t = int(np.argmax(ucbs))
        return self.a_t

    def update(self, r_t):
        """
        NB01 cell 39 update: incremental mean on reward only.
        UCB1 does not track cost.
        """
        self.N_pulls[self.a_t] += 1
        self.average_rewards[self.a_t] += (
            (r_t - self.average_rewards[self.a_t]) / self.N_pulls[self.a_t]
        )
        self.t += 1

    def save(self, name="ucb1_bidding"):
        path = DATA_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved agent to %s", path)
        return path


# ---------------------------------------------------------------------------
# Requirement 1 – UCB-like (budget-aware, LP-based)
# ---------------------------------------------------------------------------

class UCBLikeBiddingAgent:
    """
    Budget-aware bidding agent for a single campaign.

    Direct port of UCBLikeAgent from NB07 cell 40.  Field names, update
    rules, LP formulation, greedy fallback, and budget-stop condition are
    all identical to the notebook.

    Parameters
    ----------
    K : int     number of bids (after restricting to bids <= value)
    B : float   total budget
    T : int     time horizon
    range : float   reward range = value  (NB07 cell 43)
    """

    def __init__(self, K, B, T, range=1):
        # NB07 cell 40 – field names kept identical
        self.K = K
        self.T = T
        self.range = range
        self.a_t = None          # index, not the actual bid
        self.avg_f = np.zeros(K)
        self.avg_c = np.zeros(K)
        self.N_pulls = np.zeros(K)
        self.budget = B
        self.rho = B / T
        self.t = 0
        logger.info(
            "UCBLikeBiddingAgent | K=%d T=%d B=%.1f rho=%.4f range=%.2f",
            K, T, B, self.rho, range,
        )

    def pull_arm(self):
        """NB07 cell 40: budget stop → init phase → LP sampling."""
        # NB07 cell 40: if budget < 1, bid 0 (index 0)
        if self.budget < 1:
            self.a_t = 0
            return 0
        # NB07 cell 40: pull each arm once before UCB kicks in
        if self.t < self.K:
            self.a_t = self.t
            return self.a_t
        # NB07 cell 40: compute UCBs and LCBs (NO max(0,...) on LCB)
        f_ucbs = self.avg_f + self.range * np.sqrt(2 * np.log(self.T) / self.N_pulls)
        c_lcbs = self.avg_c - self.range * np.sqrt(2 * np.log(self.T) / self.N_pulls)
        gamma_t = self._compute_opt(f_ucbs, c_lcbs)
        self.a_t = int(np.random.choice(self.K, p=gamma_t))
        return self.a_t

    def _compute_opt(self, f_ucbs, c_lcbs):
        """
        NB07 cell 40 compute_opt:
          - if any c_lcb <= 0: go greedy on f_ucbs (no LP)
          - otherwise: solve LP
        """
        # NB07 cell 40: "if np.sum(c_lcbs <= np.zeros(len(c_lcbs)))"
        if np.sum(c_lcbs <= np.zeros(len(c_lcbs))):
            gamma = np.zeros(self.K)
            gamma[int(np.argmax(f_ucbs))] = 1
            return gamma

        # NB07 cell 40: LP formulation
        c    = -f_ucbs                       # minimise negative utility
        A_ub = [c_lcbs]                      # budget constraint
        b_ub = [self.rho]
        A_eq = [np.ones(self.K)]             # simplex
        b_eq = [1]
        res  = optimize.linprog(
            c, A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=(0, 1),
        )
        return res.x

    def update(self, f_t, c_t):
        """NB07 cell 40: incremental mean update for both f and c."""
        self.N_pulls[self.a_t] += 1
        n = self.N_pulls[self.a_t]
        self.avg_f[self.a_t] += (f_t - self.avg_f[self.a_t]) / n
        self.avg_c[self.a_t] += (c_t - self.avg_c[self.a_t]) / n
        self.budget -= c_t
        self.t += 1

    def save(self, name="ucblike_bidding"):
        path = DATA_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved agent to %s", path)
        return path


# ---------------------------------------------------------------------------
# Requirement 2 – Combinatorial UCB
# ---------------------------------------------------------------------------

class CombinatorialUCBAgent:
    """
    Combinatorial UCB for N campaigns with a shared budget and a conflict graph.

    Structure mirrors UCBMatchingAgent (NB09 cell 30):
      - avg_f[i,k], avg_c[i,k], N_pulls[i,k] track each (campaign, bid) pair
      - unexplored arms get large_value = (1 + sqrt(2*log(T)/1))*10  (NB09 cell 30)
      - oracle = LP (instead of linear_sum_assignment) over feasible joint
        actions to handle the shared budget and conflict graph constraints
      - update receives per-campaign utilities and costs (semi-bandit feedback
        analogous to NB09 cell 30's per-edge reward)

    The LP oracle extends the single-campaign UCBLikeAgent LP (NB07 cell 40)
    to multiple campaigns:

        max  sum_a p_a * UCB_f(a)
        s.t. sum_a p_a * LCB_c(a) <= rho      [shared budget]
             sum_a p_a = 1                    [distribution over actions]
             p_a >= 0

    Every joint action a is built so that it contains at most one bid per
    campaign and never contains both endpoints of a conflict edge.  This makes
    incompatibility constraints hold in every realised round, not only in
    expectation through marginal probabilities.

    Parameters
    ----------
    N : int                     number of campaigns
    Ks : list[int]              number of bids per campaign
    T : int                     time horizon
    budget : float              total shared budget B
    values : list[float]        per-campaign values (used for range)
    conflict_edges : list[(i,j)]
    """

    def __init__(self, N, Ks, T, budget, values, conflict_edges=None):
        self.N = N
        self.Ks = Ks            # list of K_i
        self.T = T
        self.budget = budget
        self.values = np.asarray(values)
        self.rho = budget / T
        self.conflict_edges = conflict_edges or []

        # NB09 cell 30 field names adapted for 2-D (campaign × bid)
        self.avg_f   = [np.zeros(Ks[i]) for i in range(N)]
        self.avg_c   = [np.zeros(Ks[i]) for i in range(N)]
        self.N_pulls = [np.zeros(Ks[i]) for i in range(N)]

        self.joint_actions = self._build_joint_actions()
        self.A_t = None     # list of bid indices, one per campaign
        self.t = 0

        logger.info(
            "CombinatorialUCBAgent | N=%d Ks=%s T=%d B=%.1f rho=%.4f edges=%s",
            N, Ks, T, budget, self.rho, self.conflict_edges,
        )

    def pull_arm(self):
        """
        Compute UCB/LCB matrices, run LP oracle, sample one joint action.

        Mirrors NB09 cell 30's pull_arm:
          - unexplored arms get large_value = (1 + sqrt(2*log(T)/1))*10
          - explored arms get mean + confidence interval
        Then LP replaces linear_sum_assignment as the combinatorial oracle.
        """
        # NB07 cell 40: budget stop
        if self.budget < 1:
            self.A_t = [-1] * self.N
            return self.A_t

        # NB09 cell 30: large_value for unexplored arms
        large_value = (1 + np.sqrt(2 * np.log(self.T) / 1)) * 10

        f_ucb_list = []
        c_lcb_list = []

        for i in range(self.N):
            range_i = self.values[i]            # NB07 cell 43: range = value
            f_ucb_i = np.where(
                self.N_pulls[i] == 0,
                large_value,
                self.avg_f[i] + range_i * np.sqrt(2 * np.log(self.T) / np.maximum(self.N_pulls[i], 1)),
            )
            # NB07 cell 40: LCB with NO max(0,...) clipping
            c_lcb_i = np.where(
                self.N_pulls[i] == 0,
                0.0,
                self.avg_c[i] - range_i * np.sqrt(2 * np.log(self.T) / np.maximum(self.N_pulls[i], 1)),
            )
            f_ucb_list.append(f_ucb_i)
            c_lcb_list.append(c_lcb_i)

        gamma_t = self._solve_lp(f_ucb_list, c_lcb_list)
        action_idx = int(np.random.choice(len(self.joint_actions), p=gamma_t))
        self.A_t = list(self.joint_actions[action_idx])

        return self.A_t

    def _build_joint_actions(self):
        """Enumerate all round-feasible bid vectors, including abstentions."""
        actions = []
        current = [-1] * self.N
        active = set()
        edge_set = {tuple(sorted(edge)) for edge in self.conflict_edges}

        def compatible(campaign):
            return all(tuple(sorted((campaign, other))) not in edge_set
                       for other in active)

        def backtrack(i):
            if i == self.N:
                actions.append(tuple(current))
                return

            current[i] = -1
            backtrack(i + 1)

            if compatible(i):
                active.add(i)
                for k in range(self.Ks[i]):
                    current[i] = k
                    backtrack(i + 1)
                active.remove(i)
                current[i] = -1

        backtrack(0)
        return actions

    def _action_scores(self, f_ucb_list, c_lcb_list):
        utilities = np.zeros(len(self.joint_actions))
        costs = np.zeros(len(self.joint_actions))
        for idx, action in enumerate(self.joint_actions):
            for i, k in enumerate(action):
                if k >= 0:
                    utilities[idx] += f_ucb_list[i][k]
                    costs[idx] += c_lcb_list[i][k]
        return utilities, costs

    def _greedy_feasible_distribution(self, f_ucb_list):
        utilities, _ = self._action_scores(
            f_ucb_list,
            [np.zeros(self.Ks[i]) for i in range(self.N)],
        )
        gamma = np.zeros(len(self.joint_actions))
        gamma[int(np.argmax(utilities))] = 1.0
        return gamma

    def _solve_lp(self, f_ucb_list, c_lcb_list):
        """
        Joint LP oracle. Extends NB07 cell 40's compute_opt to a distribution
        over round-feasible joint actions.

        Returns one probability vector over self.joint_actions.
        """
        # NB07 cell 40 fallback, adapted to choose one feasible joint action.
        any_non_positive = any(
            np.sum(c_lcb_list[i] <= 0) > 0 for i in range(self.N)
        )
        if any_non_positive:
            return self._greedy_feasible_distribution(f_ucb_list)

        f_actions, c_actions = self._action_scores(f_ucb_list, c_lcb_list)

        res = optimize.linprog(
            -f_actions,
            A_ub=np.array([c_actions]),
            b_ub=np.array([self.rho]),
            A_eq=np.array([np.ones(len(self.joint_actions))]),
            b_eq=np.array([1.0]),
            bounds=[(0.0, 1.0)] * len(self.joint_actions),
            method="highs",
        )

        if not res.success:
            logger.warning("Joint LP failed (%s), using feasible greedy fallback.", res.message)
            return self._greedy_feasible_distribution(f_ucb_list)

        gamma = np.clip(res.x, 0, 1)
        gamma /= gamma.sum()
        return gamma

    def update(self, utilities, costs):
        """
        NB09 cell 30 update pattern: update all arms in A_t (semi-bandit).
        utilities, costs: np.ndarray shape (N,)
        """
        for i, k in enumerate(self.A_t):
            if k < 0:
                continue
            self.N_pulls[i][k] += 1
            n = self.N_pulls[i][k]
            self.avg_f[i][k] += (utilities[i] - self.avg_f[i][k]) / n
            self.avg_c[i][k] += (costs[i]     - self.avg_c[i][k]) / n
        self.budget -= costs.sum()
        self.t += 1

    def save(self, name="combinatorial_ucb"):
        path = DATA_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved agent to %s", path)
        return path
