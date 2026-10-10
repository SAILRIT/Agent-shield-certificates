"""Figure 1 of the letter: (a) the grid benchmark, (b) the certified risk
Vbar_T(s) of every cell and the fence of the shield of Experiment 1
(M = 12,000 logged episodes, eps = 0.10, T = 16, dataset index 0).

    python experiments/make_fig1.py [--out results/fig1.pdf]

Text is embedded as TrueType (no Type 3 fonts).
"""
import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shield import grid as G                        # noqa: E402

HAZARD = "#c0504d"
START = "#3f6fb5"
GOAL = "#e8893b"


def cell_axes(ax, nr, nc):
    ax.set_xlim(-0.5, nc - 0.5)
    ax.set_ylim(nr - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for i in range(nc + 1):
        ax.plot([i - 0.5, i - 0.5], [-0.5, nr - 0.5], color="0.75", lw=0.6, zorder=1)
    for i in range(nr + 1):
        ax.plot([-0.5, nc - 0.5], [i - 0.5, i - 0.5], color="0.75", lw=0.6, zorder=1)


def hazard_cells(ax, env):
    for (r, c) in env["hazard"]:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=HAZARD, zorder=2))
        ax.text(c, r, "×", ha="center", va="center", color="white",
                fontsize=11, fontweight="bold", zorder=3)


def markers(ax, env, nc):
    sr, sc = divmod(env["start"], nc)
    gr, gc = env["goal_cell"]
    ax.add_patch(plt.Circle((sc, sr), 0.26, color=START, zorder=4))
    ax.text(sc, sr, "S", ha="center", va="center", color="white", fontsize=7,
            fontweight="bold", zorder=5)
    ax.plot(gc, gr, marker="*", color=GOAL, markersize=13, zorder=4, linestyle="none")


def fence(ax, env, D):
    nr, nc = env["nrows"], env["ncols"]
    for s in range(nr * nc):
        if env["unsafe"][s]:
            continue
        r, c = divmod(s, nc)
        for a, (dr, dc) in G.MOVE_DELTA.items():
            if D[s, a]:
                continue
            if dr == -1:
                ax.plot([c - 0.5, c + 0.5], [r - 0.5, r - 0.5], "k", lw=2.4, zorder=6,
                        solid_capstyle="butt")
            elif dr == 1:
                ax.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5], "k", lw=2.4, zorder=6,
                        solid_capstyle="butt")
            elif dc == 1:
                ax.plot([c + 0.5, c + 0.5], [r - 0.5, r + 0.5], "k", lw=2.4, zorder=6,
                        solid_capstyle="butt")
            else:
                ax.plot([c - 0.5, c - 0.5], [r - 0.5, r + 0.5], "k", lw=2.4, zorder=6,
                        solid_capstyle="butt")


def main(out):
    env = G.make_env()
    nr, nc = env["nrows"], env["ncols"]
    r = G.certified_shield_from_logs(env, 12000, G.dataset_seed(12000, 0))
    D, V = r["D"], r["V"]
    vmax = float(np.ceil(V[:nr * nc][~env["unsafe"][:nr * nc]].max() * 20) / 20)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.25),
                                   gridspec_kw=dict(width_ratios=[1, 1.22], wspace=0.12))
    # (a) the workspace
    cell_axes(ax1, nr, nc)
    hazard_cells(ax1, env)
    markers(ax1, env, nc)
    ax1.set_title("(a)  Tool-agent workspace", fontsize=9)
    # (b) certified risk and fence
    cmap = LinearSegmentedColormap.from_list("risk", ["#1a7a3c", "#f2e58a", "#b22222"])
    grid = np.full((nr, nc), np.nan)
    for s in range(nr * nc):
        if not env["unsafe"][s]:
            grid[divmod(s, nc)] = V[s]
    im = ax2.imshow(grid, cmap=cmap, vmin=0, vmax=vmax, zorder=0,
                    extent=(-0.5, nc - 0.5, nr - 0.5, -0.5))
    cell_axes(ax2, nr, nc)
    hazard_cells(ax2, env)
    fence(ax2, env, D)
    markers(ax2, env, nc)
    ax2.set_title("(b)  Certified risk $\\bar V_T(s)$ and shield fence", fontsize=9)
    cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
    cb.set_label("$\\bar V_T(s)$ (worst-case reach prob.)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    # legend
    handles = [
        plt.Line2D([], [], marker="o", color=START, markersize=7, linestyle="none", label="start $s_0$"),
        plt.Line2D([], [], marker="*", color=GOAL, markersize=10, linestyle="none", label="goal"),
        plt.Rectangle((0, 0), 1, 1, color=HAZARD, label="unsafe (hazard)"),
        plt.Line2D([], [], color="k", lw=2.4, label="forbidden action (fence)"),
    ]
    fig.text(0.5, 0.095, "hazard = irreversible tool call (absorbing, unsafe)",
             ha="center", va="center", fontsize=7.5)
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, -0.005), handletextpad=0.5, columnspacing=1.6)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.2)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"figure written to {out}: shield certifies Vbar_T(s0)={r['Vbar']:.3f}, "
          f"forbids {int((~D[r['live']]).sum())} of {r['n_rows']} live pairs, colour scale to {vmax}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "results", "fig1.pdf"))
    main(ap.parse_args().out)
