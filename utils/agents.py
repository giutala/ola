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


# ---------------------------------------------------------------------------
# Requirement 3 – Best-of-Both-Worlds: Primal-Dual with multiple campaigns
# ---------------------------------------------------------------------------

class _HedgeAgent:
    """
    Hedge (exponential weights) with full feedback.

    Direct port of HedgeAgent from NB08 cell 13, extended to work with
    any reward range (not just [0,1]).

    Parameters
    ----------
    K : int         number of arms
    eta : float     learning rate  (NB08: sqrt(log(K)/T))
    """

    def __init__(self, K: int, eta: float):
        self.K   = K
        self.eta = eta
        self.weights = np.ones(K, dtype=float)

    def get_distribution(self) -> np.ndarray:
        """Return normalised probability vector (shape K)."""
        w = self.weights
        return w / w.sum()

    def update(self, loss_t: np.ndarray) -> None:
        """
        Multiplicative-weights update.  loss_t must be in [0, 1].
        NB08 cell 13: weights *= exp(-eta * loss_t)
        """
        self.weights *= np.exp(-self.eta * loss_t)


class PrimalDualMultiCampaignAgent:
    """
    Best-of-both-worlds bidding agent for N campaigns.

    Implements the primal-dual framework from NB08 (OGDHedgeSingleKnapsackAgent),
    extended to multiple campaigns with a shared budget constraint and a conflict
    graph.

    Architecture
    ------------
    Primal  – one Hedge agent per campaign (full feedback => we observe m_t
              and can compute counterfactual utilities for every bid k)
    Dual    – one shared OGD variable lambda in [0, 1/rho] for the shared budget

    Lagrangian (NB08)
    -----------------
      L(x, lambda) = sum_{i,k} x_{ik} (f_{t,ik} - lambda * c_{t,ik}) + lambda*rho

    where  f_{t,ik} = (v_i - b_{ik}) * I[b_{ik} >= m_{t,i}]
           c_{t,ik} = b_{ik}          * I[b_{ik} >= m_{t,i}]

    Primal loss for Hedge (rescaled to [0,1] -- NB08 pattern)
    ---------------------------------------------------------
      loss^P_{t,ik} = (lambda_max - (f_{t,ik} - lambda_t * c_{t,ik}))
                      / (1 + lambda_max)

    This maps the reward range [-lambda_max, 1] linearly onto [0, 1].

    Dual OGD update (NB08, modified for conflict graph)
    ----------------------------------------------------
      lambda_{t+1} = clip[0, 1/rho](lambda_t - eta_D * (rho - sum_i c_{t,i}))

    We use the realized total cost sum_i c_{t,i} (an unbiased estimator of the
    expected cost) instead of E_{x_t}[c_t] computed bid-by-bid.  Reason: the
    realized cost already incorporates the suppressions made by the conflict
    graph in MultiCampaignEnv.round, while the bid-by-bid expectation would
    ignore them.  Trade-off: higher gradient variance, mitigated by eta_D = 1/sqrt(T).

    Full feedback
    -------------
    By observing m_t (highest competitor bid per campaign) we reconstruct
    f_{t,ik} and c_{t,ik} for *all* bids without sampling -- unlike bandit
    feedback, which would require importance-weighted estimates.

    Conflict graph
    --------------
    Conflicts are enforced by the environment (MultiCampaignEnv.round).
    The Hedge agents use per-campaign marginal rewards (treating each campaign
    independently), which is the standard simplification when decomposing
    the primal minimizer across campaigns.

    Parameters
    ----------
    N            : int             number of campaigns
    Ks           : list[int]       K_i = number of bids in campaign i
    bid_sets     : list[ndarray]   actual bid values per campaign
    T            : int             time horizon
    budget       : float           total shared budget B
    values       : list[float]     per-campaign valuations v_i
    conflict_edges : list[(i,j)]   optional conflict graph
    hedge_eta    : float           Hedge LR  (default sqrt(log K_max / T))
    ogd_eta      : float           OGD LR    (default 1 / sqrt(T))
    """

    def __init__(
        self,
        N: int,
        Ks: list,
        bid_sets: list,
        T: int,
        budget: float,
        values: list,
        conflict_edges=None,
        hedge_eta: Optional[float] = None,
        ogd_eta: Optional[float] = None,
        budget_pacing: bool = False,
    ):
        self.N  = N
        self.Ks = Ks
        self.bid_sets      = [np.asarray(bs) for bs in bid_sets]
        self.T             = T
        self.budget        = float(budget)
        self.rho           = budget / T
        self.values        = np.asarray(values, dtype=float)
        self.conflict_edges = conflict_edges or []
        # Budget pacing (extension beyond NB08): when True, the dual OGD target
        # uses rho_t = remaining_budget / remaining_rounds instead of the fixed
        # rho = B/T.  This self-corrects the spending pace so the budget lasts
        # to T (no early-exhaustion regret tail) and unused budget is spent.
        # When False the agent uses the exact NB08 fixed-rho gradient.
        self.budget_pacing = budget_pacing
        # Normalised edge set: frozenset for symmetric, O(1) lookup
        self._edge_set = {frozenset(e) for e in self.conflict_edges}

        # --- Learning rates (NB08 default formulas) ------------------------
        K_max = max(Ks)
        self.hedge_eta = float(hedge_eta) if hedge_eta is not None else float(
            np.sqrt(np.log(max(K_max, 2)) / T)
        )
        self.ogd_eta = float(ogd_eta) if ogd_eta is not None else 1.0 / np.sqrt(T)
        #self.ogd_eta= 0.022
        
        # --- Reward range for loss normalisation ---------------------------
        # primal reward in [-1/rho, 1]  =>  total range = 1 + 1/rho
        self._lmbd_max    = 1.0 / self.rho
        self._reward_range = 1.0 + self._lmbd_max

        # --- Primal: one Hedge per campaign --------------------------------
        self.hedge_agents = [
            _HedgeAgent(Ks[i], self.hedge_eta) for i in range(N)
        ]

        # --- Dual: shared Lagrange multiplier ------------------------------
        # Start at 0: Hedge first learns to maximise utility unconstrained,
        # then lambda rises until the budget constraint is met.
        # NB08 uses lmbd=1 which works for rho≈0.4 (lmbd_max=2.5), but with
        # rho=0.05 (lmbd_max=20) starting at 1 immediately penalises costs
        # so heavily that Hedge converges to no-bid before lambda can correct.
        self.lmbd = 0.0

        # --- State ---------------------------------------------------------
        self.A_t: Optional[list] = None          # sampled actions (length N)
        self.x_t     = None          # distributions at current round
        self.t       = 0
        self.N_pulls = [np.zeros(Ks[i]) for i in range(N)]

        # --- History tracking (for Requirement 3 plots, NB08-style) --------
        # NB08 plots: lambda trajectory, cumulative costs, regret over time
        self.lmbds_history = []      # lambda_t at each round
        self.cost_history  = []      # total cost spent at each round
        self.utility_history = []    # total utility received at each round

        logger.info(
            "PrimalDualMultiCampaignAgent | N=%d Ks=%s T=%d B=%.1f "
            "rho=%.4f hedge_eta=%.5f ogd_eta=%.5f budget_pacing=%s edges=%s",
            N, Ks, T, budget, self.rho,
            self.hedge_eta, self.ogd_eta, self.budget_pacing, self.conflict_edges,
        )

    # --- Action selection --------------------------------------------------

    def pull_arm(self) -> list:
        """
        Get distributions from each Hedge, sample one bid per campaign,
        and proactively resolve conflicts before sending to the environment.
        """
        # 1. Budget depletion check
        if self.budget < 1:
            self.A_t = [-1] * self.N
            self.x_t = []
            for i in range(self.N):
                xi = np.zeros(self.Ks[i])
                xi[0] = 1.0
                self.x_t.append(xi)
            return self.A_t

        # 2. Ottieni le distribuzioni e campiona in modo indipendente
        self.x_t = [self.hedge_agents[i].get_distribution() for i in range(self.N)]
        self.A_t = []
        for i in range(self.N):
            k = int(np.random.choice(self.Ks[i], p=self.x_t[i]))
            self.A_t.append(k)
            
        # 3. RISOLUZIONE PROATTIVA DEI CONFLITTI LATO AGENTE
        # Se l'agente ha pescato azioni in conflitto, annulla quella con utilità minore
        active = np.array(self.A_t) >= 0
        for (ei, ej) in self.conflict_edges:
            if active[ei] and active[ej]:
                # Calcoliamo l'utilità potenziale massima (v - b)
                u_i = self.values[ei] - self.bid_sets[ei][self.A_t[ei]]
                u_j = self.values[ej] - self.bid_sets[ej][self.A_t[ej]]
                
                # Chi ha utilità minore si astiene (impostato a -1)
                if u_i >= u_j:
                    self.A_t[ej] = -1
                    active[ej] = False
                else:
                    self.A_t[ei] = -1
                    active[ei] = False

        # 4. Aggiorna i contatori (N_pulls) SOLO per le azioni sopravvissute ai conflitti
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
        Full-feedback primal-dual update, conflict-graph aware.

        When the budget is depleted, both updates are skipped: the agent is in
        a terminal state, and updating lambda would produce artefacts in the
        trajectory plot (lambda would drift downward as cost stays at 0).
        """
        # === Budget-depleted: terminal state, skip updates =================
        if self.budget < 1:
            # Still log so plots stay aligned to round index
            self.lmbds_history.append(self.lmbd)
            self.cost_history.append(0.0)
            self.utility_history.append(0.0)
            self.t += 1
            return

        assert self.A_t is not None, "pull_arm() must be called before update()"

        # === Step 1: Dual OGD update =======================================
        # Use the realised total cost c_t.sum() as the OGD gradient signal.
        # The conflict graph is already enforced by the environment, so
        # c_t.sum() is an unbiased estimate of the true expected cost under
        # the joint distribution (including conflict suppression).  Using the
        # sum of per-campaign expected costs E[c_i | x_t] instead overestimates
        # the true cost by roughly 2x (both endpoints of each conflict edge are
        # counted), which drives lambda far too high and makes Hedge converge
        # to the no-bid arm.
        lmbd_before = self.lmbd
        realised_cost = float(c_t.sum())

        # Dual target: fixed rho = B/T (NB08 cell 16/42) or, with pacing on,
        # the adaptive rho_t = remaining_budget / remaining_rounds.  At this
        # point self.budget is still the pre-spend residual and self.t has not
        # been incremented yet, so (T - t) is exactly the number of rounds
        # remaining (this one included).  No upper cap on rho_t by design.
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
        # Pre-compute the "potential" utility of every other campaign j, i.e.
        # the utility j would obtain with its sampled action a_j IF that bid
        # beats the competitors (m_t[j]).  Needed to simulate the conflict
        # graph in the counterfactual rewards used by Hedge.
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

            # --- Conflict graph applied to counterfactual rewards ---
            # Mirrors MultiCampaignEnv.round: when both i and j win, the
            # higher-utility campaign keeps the win (ties go to i, exactly
            # like the env's `u_i >= u_j` rule).
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

            # --- Loss normalisation to [0,1] -----------------------------
            # Reward range at the CURRENT lambda:
            #   upper bound = v_i  (bid wins, cost → 0)
            #   lower bound = -lmbd_before * max_bid  (highest bid wins)
            # Using lmbd_max in the denominator instead would compress all
            # losses to ~[0, v_i/(v_i + lmbd_max*max_bid)] ≈ [0, 0.05],
            # making Hedge updates negligibly small and preventing learning.
            max_r   = self.values[i]
            range_r = max_r + lmbd_before * float(self.bid_sets[i].max())
            if range_r < 1e-10:
                range_r = max_r
            loss_i = (max_r - primal_reward) / range_r
            loss_i = np.clip(loss_i, 0.0, 1.0)

            self.hedge_agents[i].update(loss_i)

        # === Budget tracking + history =====================================
        realised_cost = float(c_t.sum())
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
#
# Both classes subclass CombinatorialUCBAgent (Requirement 2, above) DIRECTLY
# -- no workaround needed, since this file's CombinatorialUCBAgent is the
# fixed version (empirical mean cost in the budget LP, no premature greedy
# fallback -- see its own docstring / the "Budget-aware practical
# constraint" comment in UCBLikeBiddingAgent above). Only the STATISTICS
# feeding the LP change (windowed / since-last-reset instead of
# full-history); the joint-action enumeration, LP oracle, and safe fallback
# are all inherited unchanged.

from collections import deque as _deque


class SlidingWindowCombinatorialUCBAgent(CombinatorialUCBAgent):
    """
    CombinatorialUCBAgent restricted to a trailing window of W ROUNDS
    (Practical/10_nonstationary_bandits.ipynb cell 23, SW-UCB), applied per
    (campaign, bid) cell.

    The window is in TIME, not "last W pulls of this arm": an
    under-explored cell must still forget stale data after W rounds, or
    "recent" would silently mean "long ago" for it. O(1) amortised updates
    via a shared deque of per-round records.

    Confidence radius uses log(min(t, W)) -- the anytime-correct version:
    before the window has even filled up (t < W), log(t) is the honest
    bound, not the fully-warmed-up log(W).

    Parameters
    ----------
    W : int, optional
        Window length. Default: Practical/10 cell 34's rule of thumb
        W = 2*sqrt(T). TUNE against the environment's regime-switch period
        (see req4_config.SW_WINDOW): with sum(K_i) cells substantially
        larger than the lab's toy K=3, the textbook default under-samples
        and keeps "forgetting" cells that have not actually changed.
    """

    def __init__(self, N, Ks, T, budget, values, conflict_edges=None, W=None):
        super().__init__(N, Ks, T, budget, values, conflict_edges)
        self.W = int(W) if W is not None else int(2 * np.sqrt(T))
        if self.W < 1:
            raise ValueError("W must be >= 1")

        self.sum_f = [np.zeros(Ks[i]) for i in range(N)]
        self.sum_c = [np.zeros(Ks[i]) for i in range(N)]
        self.win_pulls = [np.zeros(Ks[i]) for i in range(N)]
        self.history = _deque()

        logger.info("SlidingWindowCombinatorialUCBAgent | window=%d", self.W)

    def pull_arm(self):
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

    def update(self, utilities, costs):
        record = []
        for i, k in enumerate(self.A_t):
            if k < 0:
                continue
            f, c = float(utilities[i]), float(costs[i])
            record.append((i, k, f, c))
            self.sum_f[i][k] += f
            self.sum_c[i][k] += c
            self.win_pulls[i][k] += 1

            # Lifetime stats, diagnostics only.
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
    CombinatorialUCBAgent + a per-(campaign,bid) CUSUM change detector
    (Practical/10_nonstationary_bandits.ipynb cell 46, CUSUM-UCB; Page,
    1954).

    Detection signal: the WIN INDICATOR w = 1[bid >= m_t], recovered
    exactly from the semi-bandit (utility, cost) pair as
    w = 1 if (utility + cost) > 0 else 0. This is the principled signal to
    test for a shift: what actually changes across regimes is the win
    PROBABILITY (m_t's distribution), not the campaign's fixed value.

    CUSUM statistic per cell, reset independently on detection:
      - first M pulls after a (re)start build the reference mean mu0
      - after that: g+ = max(0, g+ + (w - mu0 - eps))
                    g- = max(0, g- + (mu0 - w - eps))
        alarm if g+ > h or g- > h (eps is the standard CUSUM slack term
        against mistaking noise for a shift, Page 1954).

    On alarm: reset ONLY that cell's N_pulls/avg_f/avg_c to 0 -- pull_arm
    (inherited unchanged) then treats it as unexplored again automatically.

    Extra safety-net exploration (Practical/10 cell 46's alpha): with
    probability alpha, ignore the LP and play a uniformly random FEASIBLE
    joint action -- a guard against slow drifts too gradual for CUSUM to
    flag.

    Parameters
    ----------
    U_T : int
        Prior upper bound on the number of regime changes any cell can
        undergo -- used to derive M, h, alpha if not given explicitly.
    """

    def __init__(self, N, Ks, T, budget, values, conflict_edges=None,
                 U_T=5, M=None, h=None, alpha=None, eps=0.05):
        super().__init__(N, Ks, T, budget, values, conflict_edges)

        U_T = max(int(U_T), 1)
        self.U_T = U_T
        self.M = int(M) if M is not None else max(int(np.log(T / U_T)), 5)
        self.h = float(h) if h is not None else 2.0 * np.log(T / U_T)
        self.alpha = float(alpha) if alpha is not None else float(np.sqrt(U_T * np.log(T / U_T) / T))
        self.eps = float(eps)

        self.cell_history = [[[] for _ in range(Ks[i])] for i in range(N)]
        self.n_resets = [np.zeros(Ks[i]) for i in range(N)]
        self.reset_log = []

        logger.info(
            "CUSUMCombinatorialUCBAgent | N=%d Ks=%s T=%d U_T=%d M=%d h=%.3f alpha=%.4f eps=%.3f B=%.1f",
            N, Ks, T, U_T, self.M, self.h, self.alpha, self.eps, budget,
        )

    def pull_arm(self):
        if self.budget < 1:
            self.A_t = [-1] * self.N
            return self.A_t

        if np.random.random() <= self.alpha:
            idx = np.random.randint(len(self.joint_actions))
            self.A_t = list(self.joint_actions[idx])
            return self.A_t

        return super().pull_arm()

    def update(self, utilities, costs):
        for i, k in enumerate(self.A_t):
            if k < 0:
                continue
            w = 1.0 if (utilities[i] + costs[i]) > 1e-9 else 0.0
            self.cell_history[i][k].append(w)
            if self._change_detected(i, k):
                self._reset_cell(i, k)

        super().update(utilities, costs)

    def _change_detected(self, i, k):
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

    def _reset_cell(self, i, k):
        self.avg_f[i][k] = 0.0
        self.avg_c[i][k] = 0.0
        self.N_pulls[i][k] = 0
        self.cell_history[i][k] = []
        self.n_resets[i][k] += 1
        self.reset_log.append((self.t, i, k))
