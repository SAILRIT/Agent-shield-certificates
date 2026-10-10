"""Table III (Section V-B): five shields built from the same sandbox logs.

For each of n_seed seeds, M = 4,000 episodes of the competent agent are logged
and the following shields are built from them (eps = 0.05, T = 10,
delta = 0.05): count threshold (forbid pairs seen fewer than 10 times),
Dirichlet-95% credible region, plug-in (certainty equivalence), the
policy-specific certificate (robust evaluation of the logging controller only;
it forbids nothing) and the robust shield of the letter. The all-HALT shield
and "no shield" are included for reference. Reported per shield, averaged
over the seeds: the certificate, the exact worst-case (adversarial) risk on
the true kernel, the share of the 96 live pairs forbidden, and the fraction
of seeds in which a certificate is violated; plus, on the last seed's shield
with common random numbers, the competent agent's success rate, intervention
rate and mean episode length over n_mc episodes.

    python experiments/table3_sandbox_baselines.py [--seeds 200] [--mc 20000]

A scripted ReAct-style agent with prompt injection (not in the letter) is run
behind each shield as well.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shield import sandbox as sx                   # noqa: E402

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "results", "table3_sandbox_baselines.json")
NAMES = ["no shield", "all-HALT", "count-thr.", "Dirichlet-95%",
         "plug-in MDP", "policy-spec.", "robust (ours)"]


def run(eps=sx.EPS, T=sx.T, M=sx.M, delta=sx.DELTA, n_seed=200, n_mc=20000,
        n_react=5000, n_min=10, p_inj=0.30, q_detour=0.15, verbose=True):
    env = sx.build_fs_env()
    s0 = env["start"]
    weight = sx.competent_occupancy(env, T)
    per_seed = {n: dict(cert=[], adv=[], forb=[], viol=0) for n in NAMES}
    D_last = {}
    min_covered_N = []
    for sd in range(n_seed):
        est = sx.estimator_from_logs(env, M=M, T=T, delta=delta, seed=20_000 + sd)
        pad, beta, Phat, counts = est["pad"], est["beta"], est["Phat"], est["full"]
        min_covered_N.append(int(est["total"][est["covered"]].min()))
        shields = {
            "no shield": (sx.D_all(env), None),
            "all-HALT": (sx.D_halt(env), 0.0),
            "count-thr.": (sx.shield_count_threshold(counts, env, n_min), None),
        }
        Dd, cd = sx.shield_dirichlet(counts, env, s0, T, eps, weight,
                                     np.random.default_rng(50_000 + sd))
        shields["Dirichlet-95%"] = (Dd, cd)
        Dp, Vp, _ = sx.synth(pad, np.zeros_like(beta), env, s0, T, eps, weight)
        shields["plug-in MDP"] = (Dp, float(Vp[s0]))
        shields["policy-spec."] = (sx.D_all(env), sx.policy_specific_cert(Phat, beta, env, T))
        Dr, Vr, _ = sx.synth(pad, beta, env, s0, T, eps, weight)
        shields["robust (ours)"] = (Dr, float(Vr[s0]))
        for n, (D, cert) in shields.items():
            r = per_seed[n]
            a = sx.adversarial_risk(env, D, T)
            r["adv"].append(a)
            r["forb"].append(sx.forb_frac(D))
            r["viol"] += (cert is not None) and (a > eps + 1e-9)
            if cert is not None:
                r["cert"].append(cert)
            D_last[n] = D

    table = {}
    for n in NAMES:
        r = per_seed[n]
        m = sx.competent_metrics(env, D_last[n], T, n_mc, np.random.default_rng(7))
        table[n] = dict(
            cert=(float(np.mean(r["cert"])) if r["cert"] else None),
            adv=float(np.mean(r["adv"])),
            succ=m["succ"], intv=m["intv"], length=m["length"],
            forb=float(np.mean(r["forb"])),
            viol_rate=r["viol"] / n_seed if r["cert"] else None)

    react, react_trace = {}, []
    for n in ["no shield", "count-thr.", "Dirichlet-95%", "plug-in MDP", "robust (ours)"]:
        rr = sx.react_rollout(env, D_last[n], T, n_react, np.random.default_rng(97),
                              p_inj=p_inj, q_detour=q_detour,
                              trace_first_injected=(n == "robust (ours)"))
        react[n] = {k: v for k, v in rr.items() if k != "trace"}
        if n == "robust (ours)":
            react_trace = rr["trace"]

    fences = sx.fenced_actions(env, D_last["robust (ours)"])
    out = dict(params=dict(eps=eps, T=T, M=M, delta=delta, n_seed=n_seed, n_mc=n_mc,
                           n_react=n_react, n_min=n_min, p_inj=p_inj, q_detour=q_detour),
               least_visited_covered_row=dict(min=int(min(min_covered_N)), max=int(max(min_covered_N)),
                                              seed0=min_covered_N[0]),
               table=table, react=react,
               robust_fences=[list(map(int, f)) for f in fences],
               robust_fences_described=[sx.describe_pair(s, a) for s, a in fences])

    if verbose:
        print(f"=== sandbox: eps={eps}, T={T}, M={M}, delta={delta}, seeds={n_seed} ===")
        print(f"least-visited row meeting its budget: {min_covered_N[0]} observations (seed 0); "
              f"range over seeds {min(min_covered_N)}-{max(min_covered_N)}\n")
        print(f"{'shield':<15}{'cert':>7}{'adv':>7}{'succ':>7}{'forb%':>8}{'intv%':>8}{'len':>6}{'viol%':>7}")
        for n in NAMES:
            v = table[n]
            cs = "  --" if v["cert"] is None else f"{v['cert']:.2f}"
            vs = "  --" if v["viol_rate"] is None else f"{100 * v['viol_rate']:.0f}"
            print(f"{n:<15}{cs:>7}{v['adv']:>7.2f}{v['succ']:>7.2f}{100 * v['forb']:>8.1f}"
                  f"{100 * v['intv']:>8.1f}{v['length']:>6.1f}{vs:>7}")
        print("\nthe robust shield forbids:")
        for d in out["robust_fences_described"]:
            print("  " + d)
        print(f"\n--- scripted ReAct-style agent (p_inj={p_inj}, q_detour={q_detour}, {n_react} episodes) ---")
        print(f"{'shield':<15}{'unsafe':>8}{'succ':>7}{'intv%':>8}{'blk/ep':>8}{'len':>6}")
        for n, v in react.items():
            print(f"{n:<15}{v['unsafe']:>8.3f}{v['succ']:>7.2f}{100 * v['intv']:>8.1f}"
                  f"{v['blocked_per_ep']:>8.2f}{v['length']:>6.1f}")
        print("\n--- excerpt: first injected episode under the robust shield ---")
        for line in react_trace[:14]:
            print(line)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--mc", type=int, default=20000)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    res = run(n_seed=args.seeds, n_mc=args.mc)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nresults written to {args.out}")
