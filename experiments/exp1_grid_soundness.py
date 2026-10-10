"""Experiment 1 (Section V-A): soundness on the grid benchmark, and the
convergence of the certificate with the size of the logs.

  (a) One shield is synthesized by Algorithm 1 from M = 12,000 logged episodes
      (eps = 0.10). Its certificate Vbar_T(s0) is compared with the exact
      worst-case risk V*_T(s0) on the known kernel, and three controllers are
      run through it for 40,000 episodes each: an adversary playing the exact
      worst-case admissible policy, a competent goal-seeking controller and a
      uniformly random one.
  (b) The certificate of that FIXED shield is recomputed from fresh logs of
      increasing size M (five datasets per M) and the slope of
      log(Vbar - V*) against log M is reported (Corollary 1 guarantees -1/2).
  (c) The price of the committed budget: the share of live pairs the shield
      forbids under the estimator of Section III-A, against the share under
      the superseded estimator that uses every observation.

    python experiments/exp1_grid_soundness.py [--figure results/fig1.png]

NOTE on exactness: the letter's shield comes from ONE dataset at M = 12,000,
dataset index 0 of the repository's seeding convention
(rng = default_rng(40000 + 131*M + index)).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shield import grid as G                                           # noqa: E402
from shield.robust import (build_padded, budget_beta, robust_reach_avoid_fast,  # noqa: E402
                           true_reach_avoid, weissman_beta)

N_MC = 40000


def mc_unsafe(env, D, select, n_ep, seed, T):
    """Fraction of n_ep episodes from the start in which the controller
    `select(states, t, rng)` reaches the unsafe set within T steps."""
    P, unsafe, sink, s0 = env["P"], env["unsafe"], env["sink"], env["start"]
    rng = np.random.default_rng(seed)
    cum = np.cumsum(P, axis=2)
    s = np.full(n_ep, s0, dtype=np.int64)
    hit = np.zeros(n_ep, dtype=bool)
    done = np.zeros(n_ep, dtype=bool)
    for t in range(T):
        a = select(s, t, rng)
        u = rng.random(n_ep)
        s2 = (u[:, None] < cum[s, a]).argmax(axis=1)
        s2 = np.where(done, s, s2)
        hit |= unsafe[s2] & ~done
        done |= unsafe[s2] | (s2 == sink)
        s = s2
    return float(hit.mean())


def main(figure=None, out_json=None):
    env = G.make_env()
    s0, unsafe, sink, P = env["start"], env["unsafe"], env["sink"], env["P"]
    S, A = env["nstate"], env["A"]
    live, n_rows = G.live_rows(env)
    results = {}

    # worst controller without a shield
    D_all = np.ones((S, A), bool)
    D_all[unsafe] = False
    D_all[sink] = False
    v_none = float(true_reach_avoid(P, D_all, unsafe, sink, G.T)[s0])
    print(f"worst-case unsafe-reach probability from s0 without a shield: {v_none:.4f}")
    results["no_shield_worst_case"] = v_none

    # ---- (a) the shield of the letter -------------------------------------
    r = G.certified_shield_from_logs(env, 12000, G.dataset_seed(12000, 0))
    D_fix = r["D"]
    Vstar, adv_pol = true_reach_avoid(P, D_fix, unsafe, sink, G.T, return_policy=True)
    Vstar = float(Vstar[s0])
    print(f"\nshield from M=12,000 episodes (eps={G.EPS}): certified Vbar_T(s0)={r['Vbar']:.3f}, "
          f"exact worst case V*_T(s0)={Vstar:.3f}")
    results["shield"] = dict(Vbar=r["Vbar"], Vstar=Vstar,
                             forbidden=int((~D_fix[live]).sum()), n_rows=n_rows,
                             n0=r["n0"], D=D_fix.tolist(), V=r["V"].tolist())

    comp_pol = G.goal_seeking_policy(env)
    perm = [np.where(D_fix[s])[0] for s in range(S)]

    def sel_adversarial(s, t, rng):
        a = adv_pol[min(t, G.T - 1)][s]
        return np.where(a < 0, G.A_HALT, a)

    def sel_competent(s, t, rng):
        out = np.empty_like(s)
        for i, si in enumerate(s):
            a = rng.choice(A, p=comp_pol[int(si)])
            out[i] = a if D_fix[int(si), a] else G.A_HALT
        return out

    def sel_random(s, t, rng):
        out = np.empty_like(s)
        for i, si in enumerate(s):
            p = perm[int(si)]
            out[i] = rng.choice(p) if p.size else G.A_HALT
        return out

    print(f"\nthree controllers through the same shield ({N_MC} episodes each):")
    results["controllers"] = {}
    for name, sel, sd in [("adversarial", sel_adversarial, 11),
                          ("competent", sel_competent, 12),
                          ("random", sel_random, 13)]:
        rate = mc_unsafe(env, D_fix, sel, N_MC, sd, G.T)
        results["controllers"][name] = rate
        print(f"  {name:<12} unsafe-reach {rate:.3f}  "
              f"({'within the certificate' if rate <= r['Vbar'] + 1e-9 else 'ABOVE THE CERTIFICATE'})")

    # ---- (b) certificate of the fixed shield vs log size --------------------
    print("\ncertificate of the fixed shield recomputed from fresh logs (5 datasets per M):")
    Ms = [3000, 6000, 12000, 24000, 48000, 120000]
    means = []
    results["convergence"] = {}
    for M in Ms:
        vals = []
        n0v = G.row_budget(M, G.T, n_rows)
        n0 = np.full((S, A), n0v, dtype=np.int64)
        for sd in range(5):
            rng = np.random.default_rng(G.dataset_seed(M, 1000 + sd))
            pref, total = G.behavior_counts_prefix(env, M, G.T, rng, n0)
            Phat, cov = G.prefix_estimator(pref, total, n0)
            beta = budget_beta(cov, env["support_sizes"], G.DELTA, n_rows, n0v)
            pad = build_padded(Phat, env["support"], env["support_sizes"])
            V, _ = robust_reach_avoid_fast(pad, beta, D_fix, unsafe, sink, G.T)
            vals.append(float(V[s0]))
        means.append(float(np.mean(vals)))
        results["convergence"][str(M)] = dict(mean=means[-1], min=float(np.min(vals)), max=float(np.max(vals)))
        print(f"  M={M:>7,}: Vbar mean={np.mean(vals):.3f}  [{np.min(vals):.3f}, {np.max(vals):.3f}]")
    slope = float(np.polyfit(np.log10(Ms), np.log10(np.array(means) - Vstar), 1)[0])
    results["slope"] = slope
    print(f"  log-log slope of (Vbar - V*) against M: {slope:.2f}  (Corollary 1: -0.5)")

    # ---- (c) the price of the committed budget ------------------------------
    rng2 = np.random.default_rng(G.dataset_seed(12000, 0))
    counts = G.behavior_counts(env, 12000, G.T, rng2)          # same rollouts
    assert (counts.sum(axis=2) == r["total"]).all()
    Phat2, N2 = G.estimate_kernel(counts)
    beta2 = weissman_beta(N2, env["support_sizes"], G.DELTA, n_rows)
    pad2 = build_padded(Phat2, env["support"], env["support_sizes"])
    D2, V2, _ = G.synthesize_shield(pad2, beta2, unsafe, sink, s0, G.T, G.EPS, G.A_HALT,
                                    weight=G.nominal_occupancy(env, G.T), protect=env["goal"])
    f1 = 100.0 * (~D_fix[live]).sum() / n_rows
    f2 = 100.0 * (~D2[live]).sum() / n_rows
    results["forbidden_share"] = dict(committed_prefix=f1, all_samples=f2)
    print(f"\nlive pairs forbidden: {f1:.1f}% with the committed-prefix estimator "
          f"(n0={r['n0']} of {int(r['total'][live].min())}-{int(r['total'][live].max())} "
          f"observations per row), {f2:.1f}% if every observation could be used")

    if out_json:
        json.dump(results, open(out_json, "w"), indent=1)
        print(f"\nresults written to {out_json}")
    if figure:
        draw_figure(env, D_fix, r["V"], figure)
    return results


def draw_figure(env, D, V, path):
    """Figure 1(b): the certified risk Vbar_T(s) of each cell and the forbidden
    moves (black segments)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nr, nc = env["nrows"], env["ncols"]
    grid = np.full((nr, nc), np.nan)
    for s in range(nr * nc):
        grid[divmod(s, nc)] = V[s]
    fig, ax = plt.subplots(figsize=(4.2, 4))
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1)
    for (r, c) in env["hazard"]:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=True, color="firebrick"))
    for s in range(nr * nc):
        if env["unsafe"][s]:
            continue
        r, c = divmod(s, nc)
        for a, (dr, dc) in G.MOVE_DELTA.items():
            if not D[s, a]:
                if dr == -1:
                    ax.plot([c - 0.4, c + 0.4], [r - 0.45, r - 0.45], "k", lw=3)
                elif dr == 1:
                    ax.plot([c - 0.4, c + 0.4], [r + 0.45, r + 0.45], "k", lw=3)
                elif dc == 1:
                    ax.plot([c + 0.45, c + 0.45], [r - 0.4, r + 0.4], "k", lw=3)
                else:
                    ax.plot([c - 0.45, c - 0.45], [r - 0.4, r + 0.4], "k", lw=3)
    sr, sc = divmod(env["start"], nc)
    gr, gc = env["goal_cell"]
    ax.text(sc, sr, "$s_0$", ha="center", va="center", color="w", fontsize=11)
    ax.text(gc, gr, "$\\star$", ha="center", va="center", color="w", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, label="certified $\\bar V_T(s)$")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"figure written to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", default=None, help="write Fig. 1(b) to this path")
    ap.add_argument("--out", default=None, help="write the results to this JSON file")
    args = ap.parse_args()
    main(figure=args.figure, out_json=args.out)
