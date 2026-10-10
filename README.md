# Controller-agnostic runtime safety certificates for tool-using AI agents

Code, raw results and reproduction scripts for

> A. Baheri and P. Wei, "Controller-Agnostic Runtime Safety Certificates for
> Tool-Using AI Agents via Data-Driven Robust Reachability," *IEEE Control
> Systems Letters*, 2026.
> 
A runtime filter ("shield") sits between a language-model agent and its
environment and permits, in each abstract state, a subset of the tool calls.
The shield is synthesized from ordinary logs and comes with a certificate: with
probability at least 1 − δ over the logs, every controller that respects the
filter reaches an irreversible (unsafe) outcome within the task horizon with
probability at most ε. The certificate is a robust finite-horizon reach–avoid
bound over a distribution-free L1 confidence set around the empirical
transition kernel (Sections II–III of the letter), and the synthesis is a greedy
search for the least restrictive filter meeting the budget (Algorithm 1).

Everything in the letter's Section V is reproduced here: Experiment 1 and the
convergence sentence, Table II, Table III, Table IV (from the raw output of the
language-model sweep) and Experiment 4, plus Figure 1.

## Layout

```
shield/                 the method (numpy only)
  robust.py             L1 radius (5), closed form of Proposition 1, robust reach-avoid
                        value iteration (8), exact worst case on a known kernel
  grid.py               grid benchmark, exploratory logging, committed-prefix estimator
                        (Section III-A), Algorithm 1
  sandbox.py            file-system sandbox, competent agent, prefix logging, the synthesis
                        variant of Table III / Experiment 3, the four baseline shields
  markov.py             the Markov diagnostic of Experiment 4 (G-test, BH, L1 diameter,
                        permutation null)
experiments/
  exp1_grid_soundness.py        Experiment 1, convergence, 50.5% vs 25.5%      (~40 s)
  exp2_grid_pac_validation.py   Table II, 100 datasets per M                  (~10 min per M)
  table3_sandbox_baselines.py   Table III                                     (~20 s)
  exp3_llm_sweep_colab.py       Experiment 3: the Colab cell (GPU)            (30-45 min on a T4)
  exp3_summarize.py             Table IV and the text's numbers from the raw sweep output
  exp4_markov_grid.py           Experiment 4 on the grid                      (~3 min)
  exp4_markov_sandbox.py        Experiment 4 on the sandbox                   (~2 min)
  make_fig1.py                  Figure 1                                      (~10 s)
results/                outputs of every script above, as shipped with the letter
  results/logs/         the console output of each step of reproduce.sh
tests/test_core.py      unit and regression tests (~10 s)
reproduce.sh            runs everything but the GPU sweep, in order
```

## Setup

Python 3.10 or later with `numpy` and `scipy` (`matplotlib` for Figure 1):

```
pip install -r requirements.txt
python3 tests/test_core.py        # or: python3 -m pytest tests
bash reproduce.sh                 # about 35 minutes on one core
```

Experiment 3 needs a GPU: open `experiments/exp3_llm_sweep_colab.py` as one
cell in Google Colab (Runtime > Change runtime type > T4 GPU) and run it. It
installs `torch`, `transformers`, `accelerate` and `bitsandbytes` if needed,
downloads the open-weights Qwen2.5 models from the Hugging Face hub and writes
`llm_sweep_results.json`. No API key or account is needed.

## What each number in the letter comes from

| Letter | Script | Expected output |
|---|---|---|
| Sec. V-A, no shield: worst controller reaches S_u w.p. about 1 | `exp1_grid_soundness.py` | 0.9999 |
| Experiment 1: certificate 0.072, exact worst case 0.009; adversary 0.010, competent 0.000, random 0.000 | `exp1_grid_soundness.py` | same |
| convergence 0.189 (M=3,000) → 0.023 (M=120,000), log–log slope −0.70 | `exp1_grid_soundness.py` | same |
| 50.5% of live pairs forbidden, 25.5% if all samples could be used | `exp1_grid_soundness.py` | 101/200 vs 51/200 |
| 1.5 of 200 rows below budget at M=2,500 | `exp2_grid_pac_validation.py` | 1.5 |
| Table II, M=2,500: 1.00, 1.00, 0.076, 0.002 | `exp2_grid_pac_validation.py --M 2500` | same (112 pairs forbidden on average) |
| Table II, M=12,000: 1.00, 1.00, 0.074, 0.008 | `exp2_grid_pac_validation.py --M 12000` | same (67 pairs forbidden on average) |
| Sec. V-B: least-visited covered row has 9,118 observations | `table3_sandbox_baselines.py` | 9118 (seed 0) |
| Table III (all five rows), 4.2% vs 80.2% and 8.3% | `table3_sandbox_baselines.py` | same |
| Table IV and every number in the Experiment 3 text (39 of 150 injected, 84.7%, 2%, 0.29%, 31 of 39, 6.5%, 3.3 → 38.7, 31–47, …) | `exp3_summarize.py --check` | 18 statements verified |
| Experiment 4, grid: 0 of 160 reject, η̂ = 0.14 below the null 0.17 | `exp4_markov_grid.py` | 160, 0, 0.136, 0.169 |
| Experiment 4, sandbox: 0 of 15, 0.08 against 0.20; confirmation deleted: 4 of 7, 0.18 above the null | `exp4_markov_sandbox.py` | 15/0/0.084/0.201 and 7/4/0.184/0.155 |
| Extended version, Table V (both contexts) | `exp4_markov_*.py` | see `results/exp4_markov_*.json` |
| Figure 1 | `make_fig1.py` | `results/fig1.pdf` |

The `results/` folder holds the outputs the tables were made from, so each
script can also be checked against them without rerunning (Experiment 2 is the
only slow one).

## The estimator: read this before changing anything

The certificate is built on a *committed prefix* (Section III-A). For every
row (s, a) a budget n₀(s, a) is fixed before the data are seen, and the
empirical kernel is the law of the **first** n₀(s, a) successors recorded at
that row:

    Phat(s'|s,a) = #{ j <= n_0(s,a) : X_j^{s,a} = s' } / n_0(s,a).

Surplus observations are discarded; the total count N(s, a) is used only to
test whether the budget was met, and a row below budget receives the vacuous
confidence set (radius 2). This is what makes the radius (5) valid for logs
collected in closed loop (Lemma 2). Aggregate transition counts are therefore
not sufficient input: logging must keep the order of transitions or accumulate
prefix counts online. The canonical implementations are
`grid.behavior_counts_prefix` / `grid.prefix_estimator` / `robust.budget_beta`
and `sandbox.log_counts_prefix` / `sandbox.estimator_from_logs`. The routines
`grid.behavior_counts`, `grid.estimate_kernel` and `robust.weissman_beta`
implement the superseded estimator that uses every observation; they exist only
for the comparison in Experiment 1 ("if all samples could be used").

Budgets: n₀ = ⌊0.7 M T / R⌋ for every grid row (R = 200 rows; logging is
exploratory by construction); n₀ = M in the sandbox, where the competent agent
visits only three rows, each at least 9,118 times at M = 4,000, and the hazard
rows not at all.

## Two synthesis routines

* `grid.synthesize_shield` is **Algorithm 1** as printed: at each step it
  evaluates V̄_T(s₀) for every candidate removal (line 5), ranks removals that
  meet the budget first (least restrictive first) and the rest by risk reduction
  per unit of task occupancy (line 6), and commits the top-ranked removal that
  keeps the goal reachable, or the top one if none does (line 7). All grid
  experiments use it.
* `sandbox.synth` is the simpler **variant** stated in Section V-B and used for
  Table III and Experiment 3: it ranks every permitted non-HALT action at every
  occupied state by its robust Q-value, never commits a removal that would
  disconnect the goal, and falls back to the all-HALT shield. Corollary 2 covers
  it. The sandbox and language-model permissiveness figures come from this
  variant, not from the printed algorithm.

Conventions: Table III reports forbidden pairs as a share of the 96 *live*
pairs (16 live states × 6 tools); the sweep script prints the same four fences
as 4/102 because its denominator includes the absorbing goal state. The union
bound of (5) ranges over R = 102 rows in the sandbox (every non-unsafe,
non-sink state, the goal included) and R = 200 on the grid.

## Experiment 3: the language-model sweep

`exp3_llm_sweep_colab.py` is self-contained by design (it is pasted into one
Colab cell), so it carries its own copies of the sandbox, the logging, the
estimator and `synth`; `tests/test_core.py` checks that those copies build
exactly the shield the library builds. The shield is the one of Table III,
built once from 4,000 scripted-agent episodes (seed 20000) and never refitted.

Each episode draws its injection flag and environment noise from
`default_rng(777 + episode index)`, so the 39 injected episodes are the same
for every controller. A forbidden call is blocked, the model is told so and
asked again, up to three times; HALT is executed if the fourth proposal is also
forbidden. Blocked proposals do not advance the step count, and the
intervention rate is blocked proposals per executed call. The model chooses
among the six tool names by its next-token distribution restricted to them
(no parsing); at τ = 0 the argmax, at τ = 1 a sample.

The reported sweep ran in two Colab sessions on a T4, whose raw outputs are
`results/exp3_llm_sweep_qwen1.5b.json` (scripted control, 1.5B at τ = 0 and 1)
and `results/exp3_llm_sweep_qwen3b_7b.json` (scripted control, 3B and 7B at
τ = 0 and 1; the 7B in 4-bit NF4). The scripted control is identical in both
files, which checks that both sessions ran the same harness.
`exp3_summarize.py --check` recomputes Table IV and the eighteen statements the
text makes about this experiment from these files. The pooled bound on the
average unsafe rate over the 1050 shielded episodes is a one-sided 95%
Clopper–Pearson bound, 0.285%, which the letter rounds up to 0.29%.

## Notes on reproducibility

* Everything but the sweep is deterministic given the seeds in the scripts
  (numpy's `default_rng`); the outputs in `results/` were produced with numpy
  2.5 and scipy 1.18. Dataset `sd` at sample size `M` on the grid uses seed
  `40000 + 131 M + sd`; the convergence curve uses `sd = 1000..1004`; sandbox
  seed `i` uses `20000 + i` for the logs and `50000 + i` for the Dirichlet
  posterior samples.
* The sweep itself is not bit-reproducible across GPU drivers and library
  versions (the τ = 0 runs depend only on argmax decisions and are stable in
  practice; the τ = 1 runs sample).
* Experiment 4 on the grid logs 40,000 rollouts with seed 7. The number of
  testable rows (160) and the rejection count (0) do not depend on the seed;
  η̂ and its permutation null move in the second decimal (0.12–0.16 against
  0.16–0.17 over seeds 0–59). With the previous state and action as context no
  grid row has enough data to be testable at the threshold of 25 observations
  per context.
* Algorithm 1 at M = 2,500 (n₀ = 140) sits on a plateau: with radii this large
  no single removal lowers V̄_T(s₀) below 1 for the first hundred or so steps,
  so ties in line 6 are broken by state order and the shields are restrictive
  (112 of 200 pairs forbidden on average). The certificate is exact regardless.
* `robust._inner_max_core` is the closed form of Proposition 1 and is checked
  against a scalar reference on random rows by the tests. An earlier version of
  the batched evaluation of line 5 (used for the shields of the submitted
  manuscript) did not cap the mass moved onto the top successor by the room
  left there and so under-valued candidates when n₀ is small; the certificates
  were always recomputed exactly, and the only published number it touched is
  the M = 2,500 average certificate of Table II (0.083 → 0.076).

## License and citation

MIT License (see `LICENSE`). If you use this code, please cite the letter
(`CITATION.cff`).
