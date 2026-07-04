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
```

Plots are saved by requirement in `outputs/req1/` and `outputs/req2/`.
PDF reports live in `docs/reports/`. Results are pickled to `data/picklefiles/`.
