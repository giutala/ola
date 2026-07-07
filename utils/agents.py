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
        hedge_eta: float = None,
        ogd_eta: float = None,
    ):
        self.N  = N
        self.Ks = Ks
        self.bid_sets      = [np.asarray(bs) for bs in bid_sets]
        self.T             = T
        self.budget        = float(budget)
        self.rho           = budget / T
        self.values        = np.asarray(values, dtype=float)
        self.conflict_edges = conflict_edges or []
        # Normalised edge set: frozenset for symmetric, O(1) lookup
        self._edge_set = {frozenset(e) for e in self.conflict_edges}

        # --- Learning rates (NB08 default formulas) ------------------------
        K_max = max(Ks)
        self.hedge_eta = float(hedge_eta) if hedge_eta is not None else float(
            np.sqrt(np.log(max(K_max, 2)) / T)
        )
        self.ogd_eta = float(ogd_eta) if ogd_eta is not None else 1.0 / np.sqrt(T)

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
        self.A_t     = None          # sampled actions (length N)
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
            "rho=%.4f hedge_eta=%.5f ogd_eta=%.5f edges=%s",
            N, Ks, T, budget, self.rho,
            self.hedge_eta, self.ogd_eta, self.conflict_edges,
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
        grad = self.rho - realised_cost

        self.lmbd = float(np.clip(
            self.lmbd - self.ogd_eta * grad,
            0.0, self._lmbd_max,
        ))

        # === Step 2: Primal Hedge update ===================================
        # Pre-compute the "potential" utility of every other campaign j, i.e.
        # the utility j would obtain with its sampled action a_j IF that bid
        # beats the competitors (m_t[j]).  Needed to simulate the conflict
        # graph in the counterfactual rewards used by Hedge.
        potential_u = np.zeros(self.N)
        for j in range(self.N):
            aj = self.A_t[j]
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
