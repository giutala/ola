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

    Budget-aware extension of UCB1.  The agent is optimistic on utility
    through UCB estimates, and handles the budget through an LP over
    distributions on bids.  Unlike the original notebook fallback, the LP is
    always attempted: negative/over-large confidence artifacts are not allowed
    to bypass the budget constraint.

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
        """Budget stop → init phase → LP sampling.

        The objective remains UCB-like on utility.  The budget constraint uses
        the empirical expected cost, rather than a lower confidence bound that
        can collapse to zero for many arms and make the LP effectively
        budget-unaware in early rounds.
        """
        # All admissible bids are <= value <= 1 in this project, so when the
        # remaining budget is below 1 we stop bidding safely with bid index 0.
        if self.budget < 1:
            self.a_t = 0
            return 0

        # Pull each arm once before UCB kicks in.
        if self.t < self.K:
            self.a_t = self.t
            return self.a_t

        # Time-dependent confidence, as allowed by the course slides.  This
        # avoids making the early confidence radius depend on a large fixed T.
        beta = self.range * np.sqrt(
            2 * np.log(max(self.t, 2)) / self.N_pulls
        )
        f_ucbs = np.minimum(self.range, self.avg_f + beta)

        # Budget-aware practical constraint: use the empirical expected cost.
        # This avoids artificial zero costs caused by avg_c - beta < 0.
        cost_for_constraint = np.maximum(0.0, self.avg_c)

        gamma_t = self._compute_opt(f_ucbs, cost_for_constraint)
        self.a_t = int(np.random.choice(self.K, p=gamma_t))
        return self.a_t

    def _compute_opt(self, f_ucbs, cost_for_constraint):
        """Solve the budget-aware LP over distributions on bids.

        max_gamma sum_b gamma_b f_ucb[b]
        s.t.      sum_b gamma_b cost[b] <= rho
                  sum_b gamma_b = 1, gamma_b >= 0

        scipy.linprog minimizes, so the objective is -f_ucbs.  If the LP fails
        numerically, fall back to the best utility among empirically feasible
        bids; if none is feasible, use the lowest-cost bid.
        """
        res = optimize.linprog(
            -f_ucbs,
            A_ub=np.array([cost_for_constraint]),
            b_ub=np.array([self.rho]),
            A_eq=np.array([np.ones(self.K)]),
            b_eq=np.array([1.0]),
            bounds=[(0.0, 1.0)] * self.K,
            method="highs",
        )

        if res.success and res.x is not None and np.all(np.isfinite(res.x)):
            gamma = np.clip(res.x, 0.0, 1.0)
            if gamma.sum() > 0:
                gamma /= gamma.sum()
                return gamma

        logger.warning("Single-campaign LP failed; using safe fallback.")
        feasible = np.where(cost_for_constraint <= self.rho + 1e-12)[0]
        gamma = np.zeros(self.K)
        if len(feasible) > 0:
            gamma[int(feasible[np.argmax(f_ucbs[feasible])])] = 1.0
        else:
            gamma[int(np.argmin(cost_for_constraint))] = 1.0
        return gamma

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
        s.t. sum_a p_a * empirical_cost(a) <= rho  [shared budget]
             sum_a p_a = 1                         [distribution over actions]
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

        f_ucb_list = []
        cost_list = []

        for i in range(self.N):
            range_i = self.values[i]
            pulls = np.maximum(self.N_pulls[i], 1)
            beta_i = range_i * np.sqrt(2 * np.log(max(self.t, 2)) / pulls)

            # Unexplored pairs receive the largest admissible utility optimism
            # for campaign i.  Explored pairs use a clipped UCB.
            f_ucb_i = np.where(
                self.N_pulls[i] == 0,
                range_i,
                np.minimum(range_i, self.avg_f[i] + beta_i),
            )

            # Use empirical cost in the budget constraint.  Unexplored pairs
            # are optimistic with cost 0, but no negative LCB can bypass the LP.
            cost_i = np.maximum(0.0, self.avg_c[i])
            f_ucb_list.append(f_ucb_i)
            cost_list.append(cost_i)

        gamma_t = self._solve_lp(f_ucb_list, cost_list)
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

    def _action_scores(self, f_ucb_list, cost_list):
        utilities = np.zeros(len(self.joint_actions))
        costs = np.zeros(len(self.joint_actions))
        for idx, action in enumerate(self.joint_actions):
            for i, k in enumerate(action):
                if k >= 0:
                    utilities[idx] += f_ucb_list[i][k]
                    costs[idx] += cost_list[i][k]
        return utilities, costs

    def _solve_lp(self, f_ucb_list, cost_list):
        """Joint LP oracle over round-feasible joint actions.

        The LP is always attempted.  If it fails numerically, the fallback is
        budget-safe with respect to the same empirical-cost vector: choose the
        best feasible joint action, or the minimum-cost joint action if none is
        feasible.
        """
        f_actions, c_actions = self._action_scores(f_ucb_list, cost_list)

        res = optimize.linprog(
            -f_actions,
            A_ub=np.array([c_actions]),
            b_ub=np.array([self.rho]),
            A_eq=np.array([np.ones(len(self.joint_actions))]),
            b_eq=np.array([1.0]),
            bounds=[(0.0, 1.0)] * len(self.joint_actions),
            method="highs",
        )

        if res.success and res.x is not None and np.all(np.isfinite(res.x)):
            gamma = np.clip(res.x, 0.0, 1.0)
            if gamma.sum() > 0:
                gamma /= gamma.sum()
                return gamma

        msg = getattr(res, "message", "unknown error")
        logger.warning("Joint LP failed (%s), using safe fallback.", msg)
        gamma = np.zeros(len(self.joint_actions))
        feasible = np.where(c_actions <= self.rho + 1e-12)[0]
        if len(feasible) > 0:
            gamma[int(feasible[np.argmax(f_actions[feasible])])] = 1.0
        else:
            gamma[int(np.argmin(c_actions))] = 1.0
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
