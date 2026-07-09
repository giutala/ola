"""
agents.py
---------
Bidding agents for all four project requirements.

Requirement 1
-------------
UCB1BiddingAgent       Budget-unaware UCB1 baseline.
UCBLikeBiddingAgent    Budget-aware UCB with LP oracle.

Requirement 2
-------------
CombinatorialUCBAgent  Multi-campaign UCB with shared budget and conflict graph.

Requirement 3
-------------
_HedgeAgent                     Per-campaign Hedge regret minimiser (internal).
PrimalDualMultiCampaignAgent    Best-of-both-worlds primal-dual agent.

Requirement 4
-------------
SlidingWindowCombinatorialUCBAgent  Combinatorial UCB with sliding-window statistics.
CUSUMCombinatorialUCBAgent          Combinatorial UCB with per-cell CUSUM change detection.
"""

import logging
import pickle
from collections import deque as _deque
from pathlib import Path
from typing import Optional

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

    Tracks average rewards per bid and uses an upper confidence bound to
    balance exploration and exploitation. This agent is budget-unaware and
    serves as a baseline for Requirement 1.

    Parameters
    ----------
    K : int
        Number of available bids.
    T : int
        Time horizon.
    range : float
        Reward range; set to the campaign value.
    """

    def __init__(self, K: int, T: int, range: float = 1.0) -> None:
        self.K = K
        self.T = T
        self.range = range
        self.a_t: Optional[int] = None
        self.average_rewards = np.zeros(K)
        self.N_pulls = np.zeros(K)
        self.t = 0
        logger.info("UCB1BiddingAgent | K=%d T=%d range=%.2f", K, T, range)

    def pull_arm(self) -> int:
        """Pull each arm once first, then follow UCB indices."""
        if self.t < self.K:
            self.a_t = self.t
        else:
            ucbs = (self.average_rewards
                    + self.range * np.sqrt(2 * np.log(self.T) / self.N_pulls))
            self.a_t = int(np.argmax(ucbs))
        return self.a_t

    def update(self, r_t: float) -> None:
        """Incremental mean update on reward. Does not track cost."""
        self.N_pulls[self.a_t] += 1
        self.average_rewards[self.a_t] += (
            (r_t - self.average_rewards[self.a_t]) / self.N_pulls[self.a_t]
        )
        self.t += 1

    def save(self, name: str = "ucb1_bidding") -> Path:
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

    At each round, solves an LP over bid distributions to find the randomised
    strategy that maximises UCB-optimistic expected utility subject to the
    per-round budget constraint rho = B/T. Cost is estimated via empirical
    means rather than an LCB to avoid spurious zero-cost artifacts during
    early exploration.

    Parameters
    ----------
    K : int
        Number of bids (after restricting to bids <= value).
    B : float
        Total budget.
    T : int
        Time horizon.
    range : float
        Reward range; set to the campaign value.
    """

    def __init__(self, K: int, B: float, T: int, range: float = 1.0) -> None:
        self.K = K
        self.T = T
        self.range = range
        self.a_t: Optional[int] = None
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

    def pull_arm(self) -> int:
        """
        Select a bid: budget stop → init phase → LP sampling.

        The LP maximises UCB-optimistic utility subject to the empirical
        expected cost staying within rho. If the LP fails numerically, the
        fallback selects the best feasible bid deterministically.
        """
        # All admissible bids are <= value <= 1, so budget < 1 means no
        # affordable bid remains; return the abstain action (index 0).
        if self.budget < 1:
            self.a_t = 0
            return 0

        # Pull each arm once before UCB kicks in.
        if self.t < self.K:
            self.a_t = self.t
            return self.a_t

        # Time-dependent confidence radius to avoid over-optimism at early t.
        beta = self.range * np.sqrt(
            2 * np.log(max(self.t, 2)) / self.N_pulls
        )
        f_ucbs = np.minimum(self.range, self.avg_f + beta)

        # Use empirical expected cost in the budget constraint to avoid
        # negative LCB values that would make the LP budget-unaware.
        cost_for_constraint = np.maximum(0.0, self.avg_c)

        gamma_t = self._compute_opt(f_ucbs, cost_for_constraint)
        self.a_t = int(np.random.choice(self.K, p=gamma_t))
        return self.a_t

    def _compute_opt(self, f_ucbs: np.ndarray, cost_for_constraint: np.ndarray) -> np.ndarray:
        """
        Solve the budget-aware LP over distributions on bids.

            max_gamma  sum_b gamma_b f_ucb[b]
            s.t.       sum_b gamma_b cost[b] <= rho
                       sum_b gamma_b = 1,  gamma_b >= 0
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

    def update(self, f_t: float, c_t: float) -> None:
        """Incremental mean update for both utility and cost."""
        self.N_pulls[self.a_t] += 1
        n = self.N_pulls[self.a_t]
        self.avg_f[self.a_t] += (f_t - self.avg_f[self.a_t]) / n
        self.avg_c[self.a_t] += (c_t - self.avg_c[self.a_t]) / n
        self.budget -= c_t
        self.t += 1

    def save(self, name: str = "ucblike_bidding") -> Path:
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

    Maintains per-(campaign, bid) statistics and at each round solves an LP
    over all round-feasible joint bid vectors to find the optimal randomised
    strategy subject to the shared budget constraint:

        max  sum_a p_a * UCB_f(a)
        s.t. sum_a p_a * empirical_cost(a) <= rho  [shared budget]
             sum_a p_a = 1                          [valid distribution]
             p_a >= 0

    Each joint action a is a bid vector that contains at most one bid per
    campaign and never places bids on both endpoints of any conflict edge.
    Budget cost uses empirical means to avoid spurious zero-cost artifacts
    from negative LCB values during early exploration.

    Parameters
    ----------
    N : int
        Number of campaigns.
    Ks : list[int]
        Number of bids per campaign.
    T : int
        Time horizon.
    budget : float
        Total shared budget B.
    values : list[float]
        Per-campaign values (used for UCB clipping range).
    conflict_edges : list[tuple[int, int]], optional
        Pairs of campaigns that cannot both receive bids in the same round.
    """

    def __init__(
        self,
        N: int,
        Ks: list[int],
        T: int,
        budget: float,
        values: list[float],
        conflict_edges: list[tuple[int, int]] | None = None,
    ) -> None:
        self.N = N
        self.Ks = Ks
        self.T = T
        self.budget = budget
        self.values = np.asarray(values)
        self.rho = budget / T
        self.conflict_edges = conflict_edges or []

        self.avg_f   = [np.zeros(Ks[i]) for i in range(N)]
        self.avg_c   = [np.zeros(Ks[i]) for i in range(N)]
        self.N_pulls = [np.zeros(Ks[i]) for i in range(N)]

        self.joint_actions = self._build_joint_actions()
        self.A_t: Optional[list] = None
        self.t = 0

        logger.info(
            "CombinatorialUCBAgent | N=%d Ks=%s T=%d B=%.1f rho=%.4f edges=%s",
            N, Ks, T, budget, self.rho, self.conflict_edges,
        )

    def pull_arm(self) -> list:
        """
        Compute UCB matrices, run the LP oracle, and sample one joint action.

        Unexplored arms receive the maximum admissible utility as their
        optimistic estimate. Explored arms use empirical mean plus a
        time-dependent confidence radius.
        """
        if self.budget < 1:
            self.A_t = [-1] * self.N
            return self.A_t

        f_ucb_list = []
        cost_list = []

        for i in range(self.N):
            range_i = self.values[i]
            pulls = np.maximum(self.N_pulls[i], 1)
            beta_i = range_i * np.sqrt(2 * np.log(max(self.t, 2)) / pulls)

            f_ucb_i = np.where(
                self.N_pulls[i] == 0,
                range_i,
                np.minimum(range_i, self.avg_f[i] + beta_i),
            )
            cost_i = np.maximum(0.0, self.avg_c[i])
            f_ucb_list.append(f_ucb_i)
            cost_list.append(cost_i)

        gamma_t = self._solve_lp(f_ucb_list, cost_list)
        action_idx = int(np.random.choice(len(self.joint_actions), p=gamma_t))
        self.A_t = list(self.joint_actions[action_idx])

        return self.A_t

    def _build_joint_actions(self) -> list:
        """Enumerate all round-feasible bid vectors via backtracking."""
        actions = []
        current = [-1] * self.N
        active: set = set()
        edge_set = {tuple(sorted(edge)) for edge in self.conflict_edges}

        def compatible(campaign: int) -> bool:
            return all(tuple(sorted((campaign, other))) not in edge_set
                       for other in active)

        def backtrack(i: int) -> None:
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

    def _action_scores(
        self,
        f_ucb_list: list[np.ndarray],
        cost_list: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        utilities = np.zeros(len(self.joint_actions))
        costs = np.zeros(len(self.joint_actions))
        for idx, action in enumerate(self.joint_actions):
            for i, k in enumerate(action):
                if k >= 0:
                    utilities[idx] += f_ucb_list[i][k]
                    costs[idx] += cost_list[i][k]
        return utilities, costs

    def _solve_lp(
        self,
        f_ucb_list: list[np.ndarray],
        cost_list: list[np.ndarray],
    ) -> np.ndarray:
        """
        Joint LP oracle over round-feasible joint actions.

        Falls back to the best empirically feasible joint action if the LP
        fails numerically; if none is feasible, selects the minimum-cost
        joint action.
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

    def update(self, utilities: np.ndarray, costs: np.ndarray) -> None:
        """
        Semi-bandit update: update statistics for every arm pulled this round.

        Parameters
        ----------
        utilities : np.ndarray shape (N,)
        costs     : np.ndarray shape (N,)
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

    def save(self, name: str = "combinatorial_ucb") -> Path:
        path = DATA_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved agent to %s", path)
        return path


# ---------------------------------------------------------------------------
# Requirement 3 – Best-of-Both-Worlds: Primal-Dual with multiple campaigns
# ---------------------------------------------------------------------------

class _HedgeAgent:
    """
    Hedge (exponential weights) regret minimiser with full feedback.

    Maintains a weight vector over K arms updated multiplicatively after each
    round. The distribution returned by get_distribution() is the normalised
    weight vector.

    Parameters
    ----------
    K : int
        Number of arms.
    eta : float
        Learning rate.
    """

    def __init__(self, K: int, eta: float) -> None:
        self.K   = K
        self.eta = eta
        self.weights = np.ones(K, dtype=float)

    def get_distribution(self) -> np.ndarray:
        """Return the normalised probability vector (shape K)."""
        w = self.weights
        return w / w.sum()

    def update(self, loss_t: np.ndarray) -> None:
        """
        Multiplicative-weights update.

        Parameters
        ----------
        loss_t : np.ndarray shape (K,)
            Per-arm losses, must be in [0, 1].
        """
        self.weights *= np.exp(-self.eta * loss_t)


class PrimalDualMultiCampaignAgent:
    """
    Best-of-both-worlds bidding agent for N campaigns.

    Implements the primal-dual framework with one Hedge regret minimiser per
    campaign (primal) and a shared OGD step on the Lagrange multiplier for
    the budget constraint (dual). The agent achieves sublinear regret against
    the best fixed bid distribution in hindsight (OPT^A) in both stochastic
    and adversarial environments — the "best-of-both-worlds" property.

    Architecture
    ------------
    Primal  -- one Hedge agent per campaign. Observing m_t each round allows
               counterfactual utilities to be computed for all bids without
               importance weighting (full feedback).
    Dual    -- one shared Lagrange multiplier lambda in [0, 1/rho] via OGD.

    Lagrangian
    ----------
        L(x, lambda) = sum_{i,k} x_{ik} (f_{t,ik} - lambda * c_{t,ik}) + lambda * rho

    where f_{t,ik} = (v_i - b_{ik}) * 1[b_{ik} >= m_{t,i}]
          c_{t,ik} = b_{ik}         * 1[b_{ik} >= m_{t,i}]

    Hedge loss normalisation (mapped to [0, 1])
    -------------------------------------------
        loss_{t,ik} = (v_i - (f_{t,ik} - lambda_t * c_{t,ik})) / (v_i + lambda_t * max_bid_i)

    Using the current lambda_t in the denominator (rather than lambda_max)
    keeps the loss signal meaningful as lambda rises during the run.

    Dual OGD update
    ---------------
        lambda_{t+1} = clip[0, 1/rho](lambda_t - eta_D * (rho_t - sum_i c_{t,i}))

    where rho_t = B/T (fixed) or remaining_budget/remaining_rounds (budget pacing).

    Conflict graph
    --------------
    After independent sampling from each campaign's Hedge distribution,
    the lower-utility campaign in each conflicting pair is forced to abstain.
    The same tie-breaking rule (higher utility wins, ties go to lower index)
    is applied when computing counterfactual rewards for Hedge.

    Dual gradient signal
    --------------------
    The OGD gradient uses the realised total cost c_t.sum() rather than the
    sum of per-campaign expected costs E[c_i | x_t]. This correctly accounts
    for conflict suppression already applied by the environment; using the
    per-campaign expectation would double-count costs for conflicting pairs,
    causing lambda to overshoot and Hedge to converge to the no-bid arm.

    Parameters
    ----------
    N : int
        Number of campaigns.
    Ks : list[int]
        Number of bids per campaign.
    bid_sets : list[np.ndarray]
        Actual bid values per campaign.
    T : int
        Time horizon.
    budget : float
        Total shared budget B.
    values : list[float]
        Per-campaign valuations v_i.
    conflict_edges : list[tuple[int, int]], optional
        Pairs of campaigns that cannot both bid in the same round.
    hedge_eta : float, optional
        Hedge learning rate. Default: sqrt(log(K_max) / T).
    ogd_eta : float, optional
        OGD learning rate for lambda. Default: 1 / sqrt(T).
    budget_pacing : bool
        If True, uses adaptive rho_t = remaining_budget / remaining_rounds
        instead of the fixed rho = B/T. Default: False.
    """

    def __init__(
        self,
        N: int,
        Ks: list[int],
        bid_sets: list[np.ndarray],
        T: int,
        budget: float,
        values: list[float],
        conflict_edges: list[tuple[int, int]] | None = None,
        hedge_eta: float | None = None,
        ogd_eta: float | None = None,
        budget_pacing: bool = False,
    ) -> None:
        self.N  = N
        self.Ks = Ks
        self.bid_sets      = [np.asarray(bs) for bs in bid_sets]
        self.T             = T
        self.budget        = float(budget)
        self.rho           = budget / T
        self.values        = np.asarray(values, dtype=float)
        self.conflict_edges = conflict_edges or []
        # Budget pacing: when True, the OGD target uses rho_t =
        # remaining_budget / remaining_rounds instead of fixed rho = B/T,
        # which self-corrects the spending pace throughout the horizon.
        self.budget_pacing = budget_pacing
        self._edge_set = {frozenset(e) for e in self.conflict_edges}

        K_max = max(Ks)
        self.hedge_eta = float(hedge_eta) if hedge_eta is not None else float(
            np.sqrt(np.log(max(K_max, 2)) / T)
        )
        self.ogd_eta = float(ogd_eta) if ogd_eta is not None else 1.0 / np.sqrt(T)

        self._lmbd_max    = 1.0 / self.rho
        self._reward_range = 1.0 + self._lmbd_max

        self.hedge_agents = [
            _HedgeAgent(Ks[i], self.hedge_eta) for i in range(N)
        ]

        # Lambda starts at 0 so Hedge first learns to maximise utility
        # unconstrained; it rises as the budget constraint becomes active.
        self.lmbd = 0.0

        self.A_t: Optional[list] = None
        self.x_t: Optional[list] = None
        self.t       = 0
        self.N_pulls = [np.zeros(Ks[i]) for i in range(N)]

        self.lmbds_history:   list[float] = []
        self.cost_history:    list[float] = []
        self.utility_history: list[float] = []

        logger.info(
            "PrimalDualMultiCampaignAgent | N=%d Ks=%s T=%d B=%.1f "
            "rho=%.4f hedge_eta=%.5f ogd_eta=%.5f budget_pacing=%s edges=%s",
            N, Ks, T, budget, self.rho,
            self.hedge_eta, self.ogd_eta, self.budget_pacing, self.conflict_edges,
        )

    # --- Action selection --------------------------------------------------

    def pull_arm(self) -> list:
        """
        Sample one bid per campaign from each Hedge distribution and resolve
        conflicts proactively before returning the action vector.
        """
        # 1. Budget depletion: abstain from all campaigns
        if self.budget < 1:
            self.A_t = [-1] * self.N
            self.x_t = []
            for i in range(self.N):
                xi = np.zeros(self.Ks[i])
                xi[0] = 1.0
                self.x_t.append(xi)
            return self.A_t

        # 2. Sample independently from each Hedge distribution
        self.x_t = [self.hedge_agents[i].get_distribution() for i in range(self.N)]
        self.A_t = []
        for i in range(self.N):
            k = int(np.random.choice(self.Ks[i], p=self.x_t[i]))
            self.A_t.append(k)

        # 3. Resolve conflicts: the lower-utility campaign in each conflicting
        #    pair abstains (ties broken in favour of the lower-index campaign).
        active = np.array(self.A_t) >= 0
        for (ei, ej) in self.conflict_edges:
            if active[ei] and active[ej]:
                u_i = self.values[ei] - self.bid_sets[ei][self.A_t[ei]]
                u_j = self.values[ej] - self.bid_sets[ej][self.A_t[ej]]
                if u_i >= u_j:
                    self.A_t[ej] = -1
                    active[ej] = False
                else:
                    self.A_t[ei] = -1
                    active[ei] = False

        # 4. Update pull counts only for campaigns that survived conflict resolution
        for i in range(self.N):
            if self.A_t[i] >= 0:
                self.N_pulls[i][self.A_t[i]] += 1

        return self.A_t

    # --- Update ------------------------------------------------------------

    def update(
        self,
        f_t: np.ndarray,
        c_t: np.ndarray,
        m_t: np.ndarray,
    ) -> None:
        """
        Full-feedback primal-dual update.

        When the budget is depleted, both primal and dual updates are skipped
        to avoid artifacts in the lambda trajectory plot.

        Parameters
        ----------
        f_t : np.ndarray shape (N,)   per-campaign utility this round
        c_t : np.ndarray shape (N,)   per-campaign cost this round
        m_t : np.ndarray shape (N,)   max competing bid per campaign (full feedback)
        """
        if self.budget < 1:
            self.lmbds_history.append(self.lmbd)
            self.cost_history.append(0.0)
            self.utility_history.append(0.0)
            self.t += 1
            return

        assert self.A_t is not None, "pull_arm() must be called before update()"

        # === Step 1: Dual OGD update =======================================
        # Use the realised total cost c_t.sum() as the gradient signal.
        # This correctly accounts for conflict suppression already applied
        # by the environment; using the sum of per-campaign expected costs
        # would double-count costs for conflicting pairs and drive lambda too high.
        lmbd_before = self.lmbd
        realised_cost = float(c_t.sum())

        # Dual target: fixed rho = B/T or, with budget pacing enabled,
        # the adaptive rho_t = remaining_budget / remaining_rounds.
        if self.budget_pacing:
            rho_t = max(self.budget, 0.0) / max(self.T - self.t, 1)
        else:
            rho_t = self.rho
        grad = rho_t - realised_cost

        self.lmbd = float(np.clip(
            self.lmbd - self.ogd_eta * grad,
            0.0, self._lmbd_max,
        ))

        # === Step 2: Primal Hedge update ===================================
        # Pre-compute the potential utility of every campaign j — needed to
        # apply the conflict-graph tie-breaking rule in counterfactual rewards.
        A_t = self.A_t
        potential_u = np.zeros(self.N)
        for j in range(self.N):
            aj = A_t[j]
            if aj >= 0 and self.bid_sets[j][aj] >= m_t[j]:
                potential_u[j] = self.values[j] - self.bid_sets[j][aj]

        for i in range(self.N):
            wins_i   = (self.bid_sets[i] >= m_t[i]).astype(float)
            f_full_i = (self.values[i] - self.bid_sets[i]) * wins_i
            c_full_i = self.bid_sets[i] * wins_i

            # Apply the conflict graph to counterfactual rewards, mirroring
            # the environment's round() rule: when both i and j would win,
            # the higher-utility campaign keeps the win (ties go to i).
            for j in range(self.N):
                if i == j:
                    continue
                if frozenset((i, j)) not in self._edge_set:
                    continue
                uj = potential_u[j]
                if uj > 0:
                    loses_conflict = f_full_i < uj      # strict <: ties → i
                    f_full_i[loses_conflict] = 0.0
                    c_full_i[loses_conflict] = 0.0

            primal_reward = f_full_i - lmbd_before * c_full_i

            # Loss normalised to [0, 1] using the current lambda.
            # Reward range: upper = v_i (win, cost → 0),
            #               lower = -lmbd_before * max_bid (loss at highest bid).
            # Using lmbd_max instead would compress all losses to ~0,
            # making Hedge updates negligibly small.
            max_r   = self.values[i]
            range_r = max_r + lmbd_before * float(self.bid_sets[i].max())
            if range_r < 1e-10:
                range_r = max_r
            loss_i = (max_r - primal_reward) / range_r
            loss_i = np.clip(loss_i, 0.0, 1.0)

            self.hedge_agents[i].update(loss_i)

        # === Budget tracking + history =====================================
        self.budget -= realised_cost
        self.lmbds_history.append(self.lmbd)
        self.cost_history.append(realised_cost)
        self.utility_history.append(float(f_t.sum()))
        self.t += 1

    # --- Persistence -------------------------------------------------------

    def save(self, name: str = "primal_dual_multi") -> Path:
        path = DATA_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved agent to %s", path)
        return path


# ---------------------------------------------------------------------------
# Requirement 4 -- Combinatorial-UCB extended with a sliding window / CUSUM
# ---------------------------------------------------------------------------

class SlidingWindowCombinatorialUCBAgent(CombinatorialUCBAgent):
    """
    CombinatorialUCBAgent with statistics restricted to a trailing window of W rounds.

    The window is over TIME, not per-arm pull counts: every cell forgets
    observations older than W rounds regardless of how often that cell was
    pulled. This allows adaptation to regime changes at the cost of higher
    variance for infrequently-pulled arms. Window statistics are maintained in
    O(1) amortised time per round via a shared deque of per-round records.

    Confidence radius uses log(min(t, W)) to remain valid before the window
    fills up (t < W).

    Parameters
    ----------
    W : int, optional
        Window length. Default: 2 * sqrt(T). For environments with long
        stationary blocks, setting W to the block length reduces variance
        without sacrificing adaptability.
    """

    def __init__(
        self,
        N: int,
        Ks: list[int],
        T: int,
        budget: float,
        values: list[float],
        conflict_edges: list[tuple[int, int]] | None = None,
        W: int | None = None,
    ) -> None:
        super().__init__(N, Ks, T, budget, values, conflict_edges)
        self.W = int(W) if W is not None else int(2 * np.sqrt(T))
        if self.W < 1:
            raise ValueError("W must be >= 1")

        self.sum_f = [np.zeros(Ks[i]) for i in range(N)]
        self.sum_c = [np.zeros(Ks[i]) for i in range(N)]
        self.win_pulls = [np.zeros(Ks[i]) for i in range(N)]
        self.history: _deque = _deque()

        logger.info("SlidingWindowCombinatorialUCBAgent | window=%d", self.W)

    def pull_arm(self) -> list:
        if self.budget < 1:
            self.A_t = [-1] * self.N
            return self.A_t

        log_term = np.log(min(max(self.t, 2), self.W))

        f_ucb_list, cost_list = [], []
        for i in range(self.N):
            range_i = self.values[i]
            n_i = self.win_pulls[i]
            n_safe = np.maximum(n_i, 1)
            avg_f_i = np.where(n_i == 0, 0.0, self.sum_f[i] / n_safe)
            avg_c_i = np.where(n_i == 0, 0.0, self.sum_c[i] / n_safe)
            beta_i = range_i * np.sqrt(2 * log_term / n_safe)

            f_ucb_i = np.where(n_i == 0, range_i, np.minimum(range_i, avg_f_i + beta_i))
            cost_i = np.maximum(0.0, avg_c_i)
            f_ucb_list.append(f_ucb_i)
            cost_list.append(cost_i)

        gamma_t = self._solve_lp(f_ucb_list, cost_list)
        action_idx = int(np.random.choice(len(self.joint_actions), p=gamma_t))
        self.A_t = list(self.joint_actions[action_idx])
        return self.A_t

    def update(self, utilities: np.ndarray, costs: np.ndarray) -> None:
        record = []
        for i, k in enumerate(self.A_t):
            if k < 0:
                continue
            f, c = float(utilities[i]), float(costs[i])
            record.append((i, k, f, c))
            self.sum_f[i][k] += f
            self.sum_c[i][k] += c
            self.win_pulls[i][k] += 1

            # Lifetime stats kept for diagnostics only.
            self.N_pulls[i][k] += 1
            n = self.N_pulls[i][k]
            self.avg_f[i][k] += (f - self.avg_f[i][k]) / n
            self.avg_c[i][k] += (c - self.avg_c[i][k]) / n

        self.history.append(record)
        if len(self.history) > self.W:
            for i, k, f, c in self.history.popleft():
                self.sum_f[i][k] -= f
                self.sum_c[i][k] -= c
                self.win_pulls[i][k] -= 1

        self.budget -= costs.sum()
        self.t += 1


class CUSUMCombinatorialUCBAgent(CombinatorialUCBAgent):
    """
    CombinatorialUCBAgent with a per-(campaign, bid) CUSUM change detector.

    Detects regime changes by running a Page (1954) CUSUM test on the win
    indicator w = 1[bid >= m_t] for each cell independently. When a cell's
    CUSUM statistic exceeds the threshold h, that cell's statistics are reset
    so pull_arm treats it as unexplored again.

    Detection signal: the win indicator w = 1 if (utility + cost) > 0 else 0,
    recoverable exactly from the semi-bandit (utility, cost) feedback. Using
    the win indicator (a function of m_t's distribution) rather than the raw
    utility (also affected by the fixed campaign value) keeps the detector
    sensitive to distributional shifts in the competing bids.

    CUSUM per cell
    --------------
    After M initial pulls establish a reference mean mu0:
        g+ = max(0, g+ + (w - mu0 - eps))
        g- = max(0, g- + (mu0 - w - eps))
    Alarm fires if g+ > h or g- > h; on alarm the cell is fully reset.

    Extra exploration
    -----------------
    With probability alpha, a uniformly random feasible joint action is played
    regardless of the LP, serving as a safety net for slow drifts that CUSUM
    may not detect promptly.

    Parameters
    ----------
    U_T : int
        Upper bound on the number of regime changes. Used to derive M, h,
        and alpha if not provided explicitly.
    M : int, optional
        Burn-in period for the reference mean. Default: max(log(T/U_T), 5).
    h : float, optional
        Detection threshold. Default: 2 * log(T/U_T).
    alpha : float, optional
        Exploration probability. Default: sqrt(U_T * log(T/U_T) / T).
    eps : float
        CUSUM slack term. Default: 0.05.
    """

    def __init__(
        self,
        N: int,
        Ks: list[int],
        T: int,
        budget: float,
        values: list[float],
        conflict_edges: list[tuple[int, int]] | None = None,
        U_T: int = 5,
        M: int | None = None,
        h: float | None = None,
        alpha: float | None = None,
        eps: float = 0.05,
    ) -> None:
        super().__init__(N, Ks, T, budget, values, conflict_edges)

        U_T = max(int(U_T), 1)
        self.U_T = U_T
        self.M = int(M) if M is not None else max(int(np.log(T / U_T)), 5)
        self.h = float(h) if h is not None else 2.0 * np.log(T / U_T)
        self.alpha = float(alpha) if alpha is not None else float(np.sqrt(U_T * np.log(T / U_T) / T))
        self.eps = float(eps)

        self.cell_history = [[[] for _ in range(Ks[i])] for i in range(N)]
        self.n_resets = [np.zeros(Ks[i]) for i in range(N)]
        self.reset_log: list[tuple[int, int, int]] = []

        logger.info(
            "CUSUMCombinatorialUCBAgent | N=%d Ks=%s T=%d U_T=%d M=%d h=%.3f alpha=%.4f eps=%.3f B=%.1f",
            N, Ks, T, U_T, self.M, self.h, self.alpha, self.eps, budget,
        )

    def pull_arm(self) -> list:
        if self.budget < 1:
            self.A_t = [-1] * self.N
            return self.A_t

        if np.random.random() <= self.alpha:
            idx = np.random.randint(len(self.joint_actions))
            self.A_t = list(self.joint_actions[idx])
            return self.A_t

        return super().pull_arm()

    def update(self, utilities: np.ndarray, costs: np.ndarray) -> None:
        for i, k in enumerate(self.A_t):
            if k < 0:
                continue
            # Win indicator: recoverable from (utility, cost) without needing m_t.
            w = 1.0 if (utilities[i] + costs[i]) > 1e-9 else 0.0
            self.cell_history[i][k].append(w)
            if self._change_detected(i, k):
                self._reset_cell(i, k)

        super().update(utilities, costs)

    def _change_detected(self, i: int, k: int) -> bool:
        hist = self.cell_history[i][k]
        if len(hist) <= self.M:
            return False
        mu0 = float(np.mean(hist[:self.M]))
        gp = gm = 0.0
        for w in hist[self.M:]:
            gp = max(0.0, gp + (w - mu0 - self.eps))
            gm = max(0.0, gm + (mu0 - w - self.eps))
            if gp > self.h or gm > self.h:
                return True
        return False

    def _reset_cell(self, i: int, k: int) -> None:
        self.avg_f[i][k] = 0.0
        self.avg_c[i][k] = 0.0
        self.N_pulls[i][k] = 0
        self.cell_history[i][k] = []
        self.n_resets[i][k] += 1
        self.reset_log.append((self.t, i, k))
