"""The diagnostic of Experiment 4: testing Assumption 1 (abstract Markovness)
on logs.

A1 says that, given the current abstract state and action, the recorded
successor is independent of anything earlier in the transcript. The test
conditions on a chosen CONTEXT, by default the previous abstract state, and
compares the successor distributions across contexts row by row:

  * a likelihood-ratio (G) test of independence per row, with Benjamini-
    Hochberg control of the false discovery rate over the testable rows;
  * eta_hat, the largest L1 distance between the successor distributions
    under two contexts (the L1 diameter), the empirical counterpart of the
    tolerance eta of Proposition 2, calibrated against a permutation null that
    shuffles the context labels within each row.

Failing to reject is not evidence of Markovness: only visited rows can be
tested, and dependence invisible to the chosen context is invisible to the
test. The test can refute a predicate set; it cannot certify one.
"""
import numpy as np
from scipy import stats


# ----------------------------------------------------------------------------
# Logging with context (grid)
# ----------------------------------------------------------------------------
def grid_log_with_context(env, M, T, rng):
    """Exploratory rollouts on the grid, recording (context, s, a, s') with the
    context = previous state * A + previous action; the first transition after
    a (re)start has no predecessor and is not recorded."""
    P, nstate, A = env["P"], env["nstate"], env["A"]
    unsafe, sink = env["unsafe"], env["sink"]
    safe = np.where(~unsafe)[0]
    safe = safe[safe != sink]
    cum = np.cumsum(P, axis=2)
    s = rng.choice(safe, size=M)
    s_prev = np.full(M, -1, dtype=np.int64)
    a_prev = np.zeros(M, dtype=np.int64)
    rec = []
    for _ in range(T):
        a = rng.integers(0, A, size=M)
        u = rng.random(M)
        s2 = (u[:, None] < cum[s, a]).argmax(axis=1)
        ok = s_prev >= 0
        if ok.any():
            ctx = s_prev[ok] * A + a_prev[ok]
            rec.append(np.stack([ctx, s[ok], a[ok], s2[ok]], axis=1))
        absorbed = unsafe[s2] | (s2 == sink)
        s_prev = np.where(absorbed, -1, s)
        a_prev = np.where(absorbed, 0, a)
        s = np.where(absorbed, rng.choice(safe, size=M), s2)
    return np.concatenate(rec, axis=0) if rec else np.zeros((0, 4), np.int64)


def reduce_context_to_state(rec, A):
    """Replace the (previous state, previous action) context by the previous
    state alone, the primary diagnostic of the letter."""
    out = rec.copy()
    out[:, 0] = out[:, 0] // A
    return out


def supports_from(rec):
    """Observed successor set of every row (s, a) present in the record."""
    sup = {}
    for s, a in set(map(tuple, rec[:, 1:3])):
        m = (rec[:, 1] == s) & (rec[:, 2] == a)
        sup[(int(s), int(a))] = sorted(set(rec[m, 3].tolist()))
    return sup


# ----------------------------------------------------------------------------
# The test
# ----------------------------------------------------------------------------
def row_table(ctx, nxt, support, min_ctx):
    """Contingency table context x successor for one row, keeping the contexts
    with at least min_ctx observations."""
    cols = {v: j for j, v in enumerate(support)}
    keep = np.isin(nxt, support)
    ctx, nxt = ctx[keep], nxt[keep]
    uc, inv = np.unique(ctx, return_inverse=True)
    tab = np.zeros((uc.size, len(support)), float)
    np.add.at(tab, (inv, np.array([cols[v] for v in nxt])), 1.0)
    return tab[tab.sum(axis=1) >= min_ctx]


def gtest(tab):
    """Likelihood-ratio test of independence; (G, df, p), or None when the
    table is degenerate or the chi-square approximation is unreliable (more
    than 20% of expected counts below 5)."""
    n = tab.sum()
    if tab.shape[0] < 2 or tab.shape[1] < 2 or n == 0:
        return None
    E = np.outer(tab.sum(1), tab.sum(0)) / n
    if (E < 5).mean() > 0.20:
        return None
    nz = tab > 0
    G = 2.0 * np.sum(tab[nz] * np.log(tab[nz] / E[nz]))
    df = (tab.shape[0] - 1) * (tab.shape[1] - 1)
    return G, df, float(stats.chi2.sf(G, df))


def l1_diameter(tab):
    """Largest L1 distance between any two context-conditional successor
    distributions: the quantity Proposition 2 assumes a bound eta on."""
    rows = tab / tab.sum(1, keepdims=True)
    if rows.shape[0] < 2:
        return 0.0
    d = np.abs(rows[:, None, :] - rows[None, :, :]).sum(axis=2)
    return float(d.max())


def bh(pvals, alpha=0.05):
    """Benjamini-Hochberg; boolean rejection vector."""
    p = np.asarray(pvals, float)
    m = p.size
    if m == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    rej = np.zeros(m, bool)
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        rej[order[: kmax + 1]] = True
    return rej


def markov_report(rec, supports, min_ctx=25, alpha=0.05, n_perm=200, seed=0):
    """rec: (context, s, a, s') rows; supports: {(s, a): [successors]}.
    Returns the number of testable rows, the number rejected (BH at level
    alpha), eta_hat, the 95th percentile of eta_hat under the permutation null,
    the excess over it and the median per-row diameter."""
    rng = np.random.default_rng(seed)
    pvals, gaps, keys = [], [], []
    for (s, a), sup in supports.items():
        m = (rec[:, 1] == s) & (rec[:, 2] == a)
        if m.sum() < 2 * min_ctx:
            continue
        tab = row_table(rec[m, 0], rec[m, 3], list(sup), min_ctx)
        out = gtest(tab)
        if out is None:
            continue
        pvals.append(out[2])
        gaps.append(l1_diameter(tab))
        keys.append((s, a))
    pvals = np.array(pvals)
    gaps = np.array(gaps)
    rej = bh(pvals, alpha)
    null = []
    for _ in range(n_perm):
        g = []
        for (s, a), sup in supports.items():
            m = (rec[:, 1] == s) & (rec[:, 2] == a)
            if m.sum() < 2 * min_ctx:
                continue
            ctx = rec[m, 0].copy()
            rng.shuffle(ctx)
            tab = row_table(ctx, rec[m, 3], list(sup), min_ctx)
            if tab.shape[0] >= 2 and tab.shape[1] >= 2:
                g.append(l1_diameter(tab))
        if g:
            null.append(max(g))
    null = np.array(null)
    null95 = float(np.percentile(null, 95)) if null.size else float("nan")
    eta_hat = float(gaps.max()) if gaps.size else 0.0
    return dict(n_rows=len(keys), rejected=int(rej.sum()),
                rej_rate=float(rej.mean()) if rej.size else 0.0,
                eta_hat=eta_hat, eta_null95=null95,
                eta_excess=max(0.0, eta_hat - null95),
                median_gap=float(np.median(gaps)) if gaps.size else 0.0)
