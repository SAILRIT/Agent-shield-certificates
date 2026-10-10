"""Experiment 4 (Section V-B) on the sandbox: the Markov diagnostic on the
letter's four-predicate abstraction, and on a deliberately coarsened one that
drops the predicate "confirmation obtained".

40,000 episodes of the competent agent with 30% epsilon-greedy exploration
are logged with their context; each transition is tested with the previous
abstract state as context (the primary diagnostic of the letter) and, for the
comparison of the extended version, with the previous state and action.

    python experiments/exp4_markov_sandbox.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shield import markov as mt                    # noqa: E402
from shield import sandbox as sx                   # noqa: E402

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "results", "exp4_markov_sandbox.json")


def log_ctx(env, M, T, rng, explore=0.30):
    """Episodes recording (s_prev, a_prev, s, a, s') in the fine abstraction."""
    P, start = env["P"], env["start"]
    cum = np.cumsum(P, axis=2)
    s = np.full(M, start, dtype=np.int64)
    sp = np.full(M, -1, dtype=np.int64)
    ap = np.zeros(M, dtype=np.int64)
    rec = []
    for _ in range(T):
        a = np.array([sx.competent_action(int(x)) for x in s])
        rnd = rng.random(M) < explore
        a = np.where(rnd, rng.integers(0, sx.A, size=M), a)
        u = rng.random(M)
        s2 = (u[:, None] < cum[s, a]).argmax(axis=1)
        ok = sp >= 0
        if ok.any():
            rec.append(np.stack([sp[ok], ap[ok], s[ok], a[ok], s2[ok]], axis=1))
        absorb = (s2 == sx.UNSAFE) | (s2 == sx.SINK) | (s2 == sx.GOAL)
        sp = np.where(absorb, -1, s)
        ap = np.where(absorb, 0, a)
        s = np.where(absorb, start, s2)
    return np.concatenate(rec, axis=0)


def coarsen(s):
    """Drop the 'confirmation obtained' bit; terminals keep their identity."""
    out = np.empty_like(s)
    for i, x in enumerate(s):
        x = int(x)
        if x >= 16:
            out[i] = x
        else:
            loc, w, c, d = sx._dec(x)
            out[i] = (loc << 2) | (w << 1) | d      # 8 coarse live states
    return out


def view(rec5, ctx="s"):
    """Collapse the 5-column record to (context, s, a, s')."""
    c = rec5[:, 0] if ctx == "s" else rec5[:, 0] * sx.A + rec5[:, 1]
    return np.stack([c, rec5[:, 2], rec5[:, 3], rec5[:, 4]], axis=1)


def main(out_path=DEFAULT_OUT):
    env = sx.build_fs_env()
    rec = log_ctx(env, 40000, 10, np.random.default_rng(11))
    print(f"logged transitions with a predecessor: {rec.shape[0]}")
    crec = np.stack([coarsen(rec[:, 0]), rec[:, 1], coarsen(rec[:, 2]),
                     rec[:, 3], coarsen(rec[:, 4])], axis=1)
    results = {}
    for label, r5 in [("fine", rec), ("coarse", crec)]:
        for ctx in ("s", "sa"):
            v = view(r5, ctx)
            rep = mt.markov_report(v, mt.supports_from(v), min_ctx=25, n_perm=60, seed=2)
            results[f"{label}|{ctx}"] = rep
            what = "paper's 4 predicates" if label == "fine" else "confirmation predicate dropped"
            print(f"\n--- {what} | context = {'previous state' if ctx == 's' else 'previous state and action'} ---")
            for k, val in rep.items():
                print(f"   {k:12s} {val:.4f}" if isinstance(val, float) else f"   {k:12s} {val}")
    json.dump(results, open(out_path, "w"), indent=1)
    print(f"\nresults written to {out_path}")
    return results


if __name__ == "__main__":
    main()
