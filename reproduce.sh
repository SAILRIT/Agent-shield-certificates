#!/usr/bin/env bash
# Reproduces every number of the letter that does not need a GPU, in order,
# writing the console output of each step to results/logs/. Experiment 3 (the
# language-model sweep) runs in Google Colab; see README.md. About 35 minutes
# on one CPU core, most of it Experiment 2.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/logs

echo "== tests"
python3 tests/test_core.py                                  | tee results/logs/tests.txt

echo "== Experiment 1: soundness and convergence on the grid (Section V-A)"
python3 experiments/exp1_grid_soundness.py --out results/exp1_grid_soundness.json \
                                                             | tee results/logs/exp1_grid_soundness.txt

echo "== Experiment 2: the certificate over 100 datasets (Table II)"
python3 experiments/exp2_grid_pac_validation.py --M 2500  --from 0 --to 100 --fresh \
                                                             | tee results/logs/exp2_grid_pac_validation_M2500.txt
python3 experiments/exp2_grid_pac_validation.py --M 12000 --from 0 --to 100 --fresh \
                                                             | tee results/logs/exp2_grid_pac_validation_M12000.txt

echo "== Table III: five shields from the same sandbox logs (Section V-B)"
python3 experiments/table3_sandbox_baselines.py              | tee results/logs/table3_sandbox_baselines.txt

echo "== Experiment 3: Table IV from the raw Colab output, checked against the text"
python3 experiments/exp3_summarize.py --check                | tee results/logs/exp3_summary.txt

echo "== Experiment 4: the Markov diagnostic (Section V-B)"
python3 experiments/exp4_markov_grid.py                      | tee results/logs/exp4_markov_grid.txt
python3 experiments/exp4_markov_sandbox.py                   | tee results/logs/exp4_markov_sandbox.txt

echo "== Figure 1"
python3 experiments/make_fig1.py --out results/fig1.pdf      | tee results/logs/make_fig1.txt

echo "done; outputs are in results/"
