# OLA – Online Learning Applications Project

**Course:** Online Learning Applications – M. Castiglioni  
**Goal:** Design online learning algorithms to bid in first-price auctions under a shared budget constraint.

---

## Setup

```bash
# Install uv (once): https://docs.astral.sh/uv/
uv sync          # installs all deps + creates venv
```

---

## Project Structure

```
ola/
├── deliverables/
│   ├── req1_single_campaign.ipynb         ← Requirement 1 notebook
│   ├── req2_multi_campaign.ipynb          ← Requirement 2 notebook
│   ├── req3_best_of_both_world.ipynb      ← Requirement 3 notebook
│   └── req4_slightly_nonstationary.ipynb  ← Requirement 4 notebook
├── utils/
│   ├── agents.py                          ← all bidding agents (req1–req4)
│   ├── environments.py                    ← SingleCampaignEnv, MultiCampaignEnv, AdversarialMultiCampaignEnv
│   ├── experiments.py                     ← trial runners, LP clairvoyants, plotting helpers
│   ├── req3_config.py                     ← shared problem parameters for req2–req4
│   ├── req4_config.py                     ← req4-specific parameters (block_size, SW_WINDOW, …)
│   ├── run_req1.py                        ← Requirement 1 pipeline
│   ├── run_req2.py                        ← Requirement 2 pipeline
│   ├── run_req3.py                        ← Requirement 3 pipeline
│   ├── run_req4.py                        ← Requirement 4 pipeline
│   ├── precomputed_clairvoyant.py         ← one-shot cache builder for req3 dynamic clairvoyant
│   └── precompute_clairvoyant_req4.py     ← one-shot cache builder for req4 dynamic clairvoyant
├── data/picklefiles/                      ← cached clairvoyant LP solutions
├── outputs/                               ← generated plots (req1/, req2/, r3/, r4/)
├── pyproject.toml
└── README.md
```

---

## Shared Problem Instance

Requirements 2–4 all operate on the **same** problem instance, defined in `utils/req3_config.py`:

| Parameter | Value |
|-----------|-------|
| N (campaigns) | 4 |
| T (rounds) | 10 000 |
| B (budget) | 1 600 |
| ρ = B/T | 0.16 |
| Values | [0.8, 0.8, 0.9, 0.9] |
| Bid grid | linspace(0, 1, 11) |
| Competitors per campaign | 3 (each) |
| Conflict edges | (0,1), (2,3) |

Requirement 1 uses the same base bid grid, `linspace(0, 1, 11)`, but with a single campaign. Because the campaign value is `v=0.8`, `SingleCampaignEnv` filters the effective bid set to `{0.0, 0.1, ..., 0.8}`. The generous scenario keeps ρ=0.16, while the tight scenario uses ρ=0.04 to make the budget constraint binding.

---

## Requirements

### Requirement 1 – Single campaign, stochastic environment

**Environment:** `SingleCampaignEnv` — first-price auction; each competitor bid is i.i.d. Uniform[0,1], so the highest competing bid m_t has distribution Beta(k, 1), where k is the number of competitors.

**Agents:**

- **UCB1** (`UCB1BiddingAgent`) — budget-unaware; treats each bid level as a MAB arm; maximises cumulative utility with no budget constraint.
- **UCB-like** (`UCBLikeBiddingAgent`) — budget-aware; maintains optimistic reward estimates and empirical cost estimates, then solves a 1-campaign LP at each round to find the budget-feasible randomised bid.

**Two budget scenarios** (deliberately different within Requirement 1, to illustrate the role of the constraint):

| Scenario | B | ρ | Effect |
|----------|------|------|--------|
| Generous | 1600 | 0.16 | Constraint non-binding; both agents converge to the same bid |
| Tight | 400 | 0.04 | Constraint binding; UCB1 overspends, UCB-like respects the budget |

**Clairvoyant benchmark:** LP optimum over the true win probabilities with the per-round budget ρ as cost constraint.

---

### Requirement 2 – Multiple campaigns, stochastic environment

**Environment:** `MultiCampaignEnv` — N=4 independent first-price auctions with a shared budget B=1600 and a conflict graph (campaigns 0–1 and 2–3 are pairwise exclusive).

**Agent:** `CombinatorialUCBAgent` — extends the single-campaign UCB-like approach to N campaigns. At each round it builds optimistic utility estimates and empirical cost estimates per (campaign, bid) cell, solves a joint LP over all N campaigns subject to the shared budget and conflict-graph constraints, then samples a feasible joint action.

**Clairvoyant benchmark:** `compute_clairvoyant_multi` — LP optimum over the true win probabilities with the shared budget constraint and conflict edges.

---

### Requirement 3 – Best-of-both-worlds, multiple campaigns

**Environment:** same N, B, T, values, bid grid, and conflict graph as Requirement 2.  Two sub-experiments:

- **Stochastic:** `MultiCampaignEnv` (i.i.d. competing bids).
- **Adversarial / non-stationary:** `AdversarialMultiCampaignEnv(mode='drift')` — the mean of the highest competing bid drifts sinusoidally, changing every round.

**Agent:** `PrimalDualMultiCampaignAgent` — one Hedge (exponential-weights) regret minimiser per campaign coupled to a shared OGD dual variable λ for the budget.  The Lagrangian is:

$$L(\mathbf{x}, \lambda) = \sum_{i=1}^N v_i \langle \mathbf{x}_i, \mathbf{w}_{i,t} \rangle - \lambda \left(\sum_{i} \langle \mathbf{x}_i, \mathbf{c}_{i,t} \rangle - \rho\right)$$

Key settings: `budget_pacing=True` (adaptive ρ_t = remaining_budget / remaining_rounds), `ogd_eta=0.017`.

**Clairvoyant benchmark:**

- **Stochastic experiment:** `compute_clairvoyant_multi` on the true stationary win probabilities.
- **Adversarial experiment:** OPT^A — best *fixed* distribution in hindsight, computed per trial from `env.empirical_win_probabilities()` fed into `compute_clairvoyant_multi`.  This is the benchmark against which Hedge + OGD has a provable sublinear pseudo-regret guarantee in both stochastic and adversarial settings.

The dynamic (prophet) clairvoyant — which knows every realised competing bid m_t — is **not** used as the adversarial benchmark because it inflates regret by a Jensen-gap term that is linear in T regardless of learner quality, making it impossible to achieve sublinear regret against it in a first-price auction.

---

### Requirement 4 – Slightly non-stationary environment, multiple campaigns

**Environment:** `AdversarialMultiCampaignEnv(mode='shocks')` — same N, B, T, values, bid grid, and conflict graph as Requirements 2–3, reparameterised for piecewise-stationary dynamics: 5 blocks of length 2 000, with the competing-bid distribution changing at each block boundary.

**Agents:**

- `SlidingWindowCombinatorialUCBAgent` — trailing window W=500 (one quarter of a block); discards observations older than W rounds so stale data is flushed quickly after regime changes.
- `CUSUMCombinatorialUCBAgent` — per-cell Page (1954) CUSUM change detector on the win indicator; resets statistics when a change is detected.
- `PrimalDualMultiCampaignAgent` — Requirement 3 agent included as a reference point, retuned for the shocks environment (`budget_pacing=True`, `hedge_eta=0.16`, `ogd_eta=0.003`).

**Benchmark hierarchy:**

| Tier | Description | Function |
|------|-------------|----------|
| PRIMARY | Piecewise expected clairvoyant — knows block boundaries and true per-block distributions; does **not** know individual m_t realisations | `compute_piecewise_expected_clairvoyant` |
| SECONDARY | OPT^A — best fixed distribution in hindsight (continuity with Req 3) | `compute_clairvoyant_multi` + `empirical_win_probabilities` |
| REFERENCE | Dynamic/prophet — knows every realised m_t; inflates regret by a linear-in-T Jensen gap | `compute_clairvoyant_dynamic_multi` |

The piecewise expected clairvoyant is the natural target for SW-UCB and CUSUM-UCB because it corresponds to the best policy a learner that knows only block boundaries (not individual m_t) could achieve.

**Results (mean over 20 trials, T=10 000):**

| Agent | Regret vs Piecewise | Regret vs OPT^A | Regret vs Prophet | Final cost |
|-------|---------------------|-----------------|-------------------|------------|
| Primal-Dual (Req 3) | **1 426.10** | **626.75** | **2 699.69** | 1 456.37 / 1 600 |
| CUSUM Combinatorial-UCB | 1 429.43 | 630.08 | 2 703.02 | 1 598.70 / 1 600 |
| Sliding-Window Combinatorial-UCB | 1 950.41 | 1 151.06 | 3 224.00 | 1 599.23 / 1 600 |

The ranking is identical under all three benchmarks in this final run: Primal-Dual and CUSUM are essentially tied, with Primal-Dual slightly lower in regret but more conservative in budget usage. Sliding-Window is worse with W=500 because it adapts quickly after shocks but keeps fewer samples during each stationary block.

---

## Running

Open any notebook and run all cells. Each notebook calls a single entry point:

```python
from utils.run_req1 import run_req1; run_req1()
from utils.run_req2 import run_req2; run_req2()
from utils.run_req3 import run_req3; run_req3()
from utils.run_req4 import run_req4; run_req4()
```

Plots are saved under `outputs/req1/`, `outputs/req2/`, `outputs/r3/`, `outputs/r4/`.  
Pickled results land in `data/picklefiles/`.

### Precomputing the dynamic clairvoyant (optional)

The dynamic/prophet LP takes 5–30 s per trial. To cache it once and skip the solve on subsequent runs:

```bash
# Requirement 3 (all adversarial modes)
cd utils/
python precomputed_clairvoyant.py

# Requirement 4 (shocks mode only)
python -m utils.precompute_clairvoyant_req4
```

If the cache is absent, the reference (prophet) curve is simply omitted; the primary and secondary benchmarks do not require it.
