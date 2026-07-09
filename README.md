# OLA – Online Learning Applications Project

**Course:** Online Learning Applications – M. Castiglioni  
**Goal:** Design online learning algorithms to bid on advertising campaigns under budget constraints.

---

## Setup

```bash
# Install uv (once): https://docs.astral.sh/uv/
uv sync          # installs all deps + creates venv
```

## Project Structure

```
ola/
├── deliverables/
│   ├── req1_single_campaign.ipynb   ← Requirement 1
│   └── req2_multi_campaign.ipynb    ← Requirement 2
├── docs/
│   ├── reports/                     ← generated PDF reports
│   └── req1_req2_corrections.tex
├── utils/
│   ├── environments.py              ← SingleCampaignEnv, MultiCampaignEnv
│   ├── agents.py                    ← UCB1BiddingAgent, UCBLikeBiddingAgent, CombinatorialUCBAgent
│   ├── experiments.py               ← clairvoyants, trial runners, plots
│   ├── run_req1.py                  ← full R1 pipeline (called from notebook)
│   └── run_req2.py                  ← full R2 pipeline (called from notebook)
├── data/picklefiles/                ← saved results
├── outputs/
│   ├── req1/                        ← Requirement 1 plots
│   └── req2/                        ← Requirement 2 plots
├── pyproject.toml
└── README.md
```

---

## Requirements implemented

### Requirement 1 – Single campaign, stochastic environment ✅
- **Environment:** `SingleCampaignEnv` — first-price auction, competing bids ~ Beta(α, β) i.i.d.
- **UCB1 (budget-unaware):** treats each bid as a MAB arm, maximises utility with no budget constraint.
- **UCB-like (budget-aware):** maintains UCB on reward and LCB on cost per bid, solves LP at each round to find the optimal randomised bid within the budget.

### Requirement 2 – Multiple campaigns, stochastic environment ✅
- **Environment:** `MultiCampaignEnv` — N independent first-price auctions, shared budget, conflict graph (non-compatible campaigns).
- **Combinatorial-UCB:** extends the single-campaign UCB-like approach. Solves a joint LP over all N campaigns respecting the shared budget and the conflict-graph constraints.

### Requirement 3 – Best-of-both-worlds, multiple campaigns ✅
- **Environment:** `AdversarialMultiCampaignEnv(mode='drift')` — same campaigns/conflict graph as Requirement 2, but the highest competing bid drifts every round.
- **Primal-Dual:** one Hedge regret-minimiser per campaign (full feedback) + one shared OGD dual variable for the budget, with `budget_pacing=True` (adaptive `rho_t = remaining_budget/remaining_rounds`) and `ogd_eta=0.017`. Benchmarked against OPT$^A$ (best fixed distribution in hindsight) — the benchmark Hedge actually has a provable sublinear-regret guarantee against.
- See `deliverables/req3_best_of_both_world.ipynb`.

### Requirement 4 – Slightly non-stationary environment, multiple campaigns ✅
- **Environment:** `AdversarialMultiCampaignEnv(mode='shocks')` — same class as Requirement 3, reparameterised for FEW, LONG intervals (`block_size=2000` → 5 intervals over `T=10000`).
- **Two new agents** (added to `utils/agents.py`'s "Requirement 4" section): `SlidingWindowCombinatorialUCBAgent` (trailing window `W=2000`) and `CUSUMCombinatorialUCBAgent` (CUSUM change detector on the win indicator) — both subclass Requirement 2's `CombinatorialUCBAgent` directly.
- **Compared against Requirement 3's `PrimalDualMultiCampaignAgent`** (same `budget_pacing`, `ogd_eta` re-tuned for this environment — same value found independently for both regimes).
- **Benchmark:** the primary diagnostic is the *piecewise expected clairvoyant* (`compute_piecewise_expected_clairvoyant` in `utils/experiments.py`), not the dynamic/realised clairvoyant — see `docs/Req4_Linear_Regret_Baseline.tex` and `docs/Req4_Baseline_Code_Fix_Practical.tex` for why: the dynamic oracle's round-by-round foreknowledge inflates regret by a term that is linear in $T$ regardless of learner quality. OPT$^A$ (Requirement 3's own benchmark) and the dynamic oracle are also reported, as secondary/reference curves.

  | Agent | Regret vs Piecewise (primary) | Regret vs OPT$^A$ | Regret vs Prophet (reference) | Cumulative cost | Budget used |
  |---|---|---|---|---|---|
  | **CUSUM Combinatorial-UCB** | **1443.90** (best) | **644.55** | **2717.49** | 1598.91 / 1600 | 99.9% |
  | Sliding-Window Combinatorial-UCB | 1532.01 | 732.66 | 2805.60 | 1595.52 / 1600 | 99.7% |
  | Primal-Dual (Req 3, `budget_pacing=True`) | 1777.40 | 978.05 | 3050.99 | 1193.92 / 1600 | 74.6% |

  The ranking (CUSUM < Sliding-Window < Primal-Dual) is identical under all three benchmarks. Primal-Dual's gap is driven by budget under-utilisation (74.6% spent vs ~99.8% for the other two) — `budget_pacing` fixes this cleanly at short horizons but only marginally at $T=10000$, an unresolved and explicitly documented "unexpected result". See `deliverables/req4_slightly_nonstationary.ipynb` for the full discussion.

- **Additional investigation (beyond the spec)**: is Primal-Dual's under-spending fixable by retuning `hedge_eta`/`ogd_eta` specifically for `shocks`? `utils/run_req4_pd_shocks_tuned.py` runs a dedicated hyperparameter search and a 4-agent comparison (included as a clearly-labelled extra section at the end of `req4_slightly_nonstationary.ipynb`). Short answer: retuning closes most of the budget gap (74.6% → 93.1%) but barely moves the regret (1777.40 → 1751.38, +1.5%) — it *relocates* the pacing failure rather than fixing it, evidence that the gap is structural (Hedge has no forgetting mechanism matched to discrete regime changes, unlike Sliding-Window/CUSUM), not a missed tuning step. Full write-up, tuning log, and theoretical argument (static vs. dynamic regret) in **`docs/Requirement4_Exam_Report.pdf`**.

---

## Running

Open any notebook and run all cells. Each notebook calls a single function from `utils/`:

```python
# req1_single_campaign.ipynb
from utils.run_req1 import run_req1
run_req1()

# req2_multi_campaign.ipynb
from utils.run_req2 import run_req2
run_req2()

# req3_best_of_both_world.ipynb
from utils.run_req3 import run_req3
run_req3()

# req4_slightly_nonstationary.ipynb
from utils.run_req4 import run_req4
run_req4()

# Optional, beyond the spec (~30-40 min, see README section above):
# from utils.run_req4_pd_shocks_tuned import run_req4_pd_comparison
# run_req4_pd_comparison()
```

Plots are saved by requirement in `outputs/req1/`, `outputs/req2/`, `outputs/r3/`, `outputs/r4/`.
PDF reports live in `docs/reports/`. Results are pickled to `data/picklefiles/`.
