"""
Utility package for first-price auction bidding agents.

Modules
-------
agents          — UCB1, UCB-like, Combinatorial UCB, SW-UCB, CUSUM-UCB, Primal-Dual
environments    — SingleCampaignEnv, MultiCampaignEnv, AdversarialMultiCampaignEnv
experiments     — trial runners, clairvoyant LP solvers, plotting helpers
req3_config     — shared problem parameters (VALUES, T, BUDGET, …) for req2–req4
req4_config     — req4-specific parameters (block_size, SW_WINDOW, …), extends req3_config
run_req1        — Requirement 1 entry point (single campaign, stochastic)
run_req2        — Requirement 2 entry point (multiple campaigns, stochastic)
run_req3        — Requirement 3 entry point (best-of-both-worlds, primal-dual)
run_req4        — Requirement 4 entry point (non-stationary, SW-UCB / CUSUM-UCB)
"""
