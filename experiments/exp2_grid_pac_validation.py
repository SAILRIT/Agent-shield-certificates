"""Experiment 2 (Table II): the certificate validated as a PAC statement.

For each sample size M and each of n datasets drawn independently, a fresh
shield is synthesized by Algorithm 1 from that dataset alone, and its TRUE
worst-case reach probability V*_T(s0) is computed on the known grid kernel.
Reported per M: the fraction of datasets with V* <= Vbar (the bound holds),
the fraction with V* <= eps (the budget is met), and the averages of Vbar and
V*. Both fractions should be at least 1 - delta = 0.95.

Results accumulate in a JSON file keyed by M and dataset index, so the
computation can be split into chunks and resumed; datasets already in the
file are skipped unless --fresh is given:

    python experiments/exp2_grid_pac_validation.py --M 2500 --from 0 --to 100 --fresh
    python experiments/exp2_grid_pac_validation.py --M 12000 --from 0 --to 100 --fresh
    python experiments/exp2_grid_pac_validation.py --summary

Each dataset takes a few seconds on one CPU core (Algorithm 1 is the
expensive step), so 100 datasets per M take about ten minutes.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shield import grid as G                      # noqa: E402
from shield.robust import true_reach_avoid        # noqa: E402

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "results", "exp2_grid_pac_validation.json")


def run_chunk(M, lo, hi, out_path, eps=G.EPS, fresh=False):
    env = G.make_env()
    s0, unsafe, sink, P = env["start"], env["unsafe"], env["sink"], env["P"]
    live, n_rows = G.live_rows(env)
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    key = str(M)
    if fresh:
        out[key] = {}
    out.setdefault(key, {})
    for sd in range(lo, hi):
        if str(sd) in out[key]:
            continue
        t0 = time.time()
        r = G.certified_shield_from_logs(env, M, G.dataset_seed(M, sd), eps=eps)
        Vstar = float(true_reach_avoid(P, r["D"], unsafe, sink, G.T)[s0])
        out[key][str(sd)] = dict(
            Vbar=r["Vbar"], Vstar=Vstar,
            rows_below_budget=int((~r["covered"][live]).sum()),
            forbidden=int((~r["D"][live]).sum()),
            seconds=round(time.time() - t0, 2))
        json.dump(out, open(out_path, "w"), indent=1)
        print(f"M={M} dataset {sd}: Vbar={r['Vbar']:.4f} V*={Vstar:.4f} "
              f"forbidden={out[key][str(sd)]['forbidden']}/{n_rows} "
              f"({time.time() - t0:.1f}s)", flush=True)
    return out


def summarize(out, eps=G.EPS, n_rows=200):
    print(f"\nTable II (eps={eps}, delta={G.DELTA}, T={G.T})")
    print(f"{'M':>8} {'datasets':>9} {'Pr[V*<=Vbar]':>13} {'Pr[V*<=eps]':>12} "
          f"{'mean Vbar':>10} {'mean V*':>8} {'max V*':>7} {'rows<n0':>8} {'forbidden':>10}")
    for key in sorted(out, key=int):
        v = out[key]
        vb = np.array([x["Vbar"] for x in v.values()])
        vs = np.array([x["Vstar"] for x in v.values()])
        uc = np.array([x["rows_below_budget"] for x in v.values()])
        fb = np.array([x["forbidden"] for x in v.values()])
        print(f"{int(key):>8} {len(v):>9} {(vs <= vb + 1e-9).mean():>13.2f} "
              f"{(vs <= eps + 1e-9).mean():>12.2f} {vb.mean():>10.3f} {vs.mean():>8.3f} "
              f"{vs.max():>7.3f} {uc.mean():>8.1f} {fb.mean():>6.1f}/{n_rows}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=None, help="number of logged episodes")
    ap.add_argument("--from", dest="lo", type=int, default=0)
    ap.add_argument("--to", dest="hi", type=int, default=100)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--summary", action="store_true", help="only print the table")
    ap.add_argument("--fresh", action="store_true",
                    help="recompute the datasets of this M even if the file already has them")
    args = ap.parse_args()
    if args.M is not None and not args.summary:
        out = run_chunk(args.M, args.lo, args.hi, args.out, fresh=args.fresh)
    else:
        out = json.load(open(args.out))
    summarize(out)
