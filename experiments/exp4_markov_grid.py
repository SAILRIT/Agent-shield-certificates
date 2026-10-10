"""Experiment 4 (Section V-B) on the grid benchmark: the Markov diagnostic on
an abstraction that is Markov by construction.

40,000 exploratory rollouts of 16 steps are logged with their context. Each of
the 200 rows (40 live states x 5 actions) is tested with the previous abstract
state as context (the primary diagnostic of the letter); the previous state
and action together are also evaluated for the comparison in the extended
version, where that richer context exhausts the data.

    python experiments/exp4_markov_grid.py

The logging seed is 7. The number of testable rows and the rejection count
are stable across seeds; eta_hat and its permutation null move in the second
decimal (eta_hat 0.12-0.16, null 0.16-0.17 over seeds 0-59).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shield import grid as G                       # noqa: E402
from shield import markov as mt                    # noqa: E402

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "results", "exp4_markov_grid.json")


def main(out_path=DEFAULT_OUT, M=40000, T=G.T, seed=7):
    env = G.make_env()
    rec = mt.grid_log_with_context(env, M, T, np.random.default_rng(seed))
    print(f"logged transitions with a predecessor: {rec.shape[0]} "
          f"({M} rollouts of {T} steps, seed {seed})")
    results = {}
    for ctx in ("s", "sa"):
        v = mt.reduce_context_to_state(rec, env["A"]) if ctx == "s" else rec
        rep = mt.markov_report(v, mt.supports_from(v), min_ctx=25, n_perm=60, seed=2)
        results[ctx] = rep
        print(f"\n--- context = {'previous state' if ctx == 's' else 'previous state and action'} ---")
        for k, val in rep.items():
            print(f"   {k:12s} {val:.4f}" if isinstance(val, float) else f"   {k:12s} {val}")
    json.dump(results, open(out_path, "w"), indent=1)
    print(f"\nresults written to {out_path}")
    return results


if __name__ == "__main__":
    main()
