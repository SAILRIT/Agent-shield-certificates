"""Core numerical primitives: the L1 confidence radius, the inner maximisation
over an L1 ball (Proposition 1), robust finite-horizon reach-avoid value
iteration (Section III-B), and the exact worst-case recursion on a known kernel.

All arrays follow the (state, action[, successor]) layout. A kernel row (s, a)
is represented on its known support K_{s,a} by the padded arrays produced by
``build_padded``.
"""
import numpy as np


# ----------------------------------------------------------------------------
# Confidence radii (Section III-A)
# ----------------------------------------------------------------------------
def budget_beta(covered, support_sizes, delta, n_rows, n0):
    """Radius (5) at the committed budget n_0 for every row whose budget was
    met; rows below budget receive the vacuous radius 2 (the whole simplex).

    covered       : (S, A) bool, N(s,a) >= n_0(s,a)
    support_sizes : (S, A) known support cardinalities k(s,a)
    delta         : confidence parameter of the certificate
    n_rows        : number R of rows the union bound ranges over
    n0            : scalar or (S, A) committed budget
    """
    k = np.clip(np.asarray(support_sizes, dtype=np.float64), 2, None)
    lt = np.log((2.0 ** k - 2.0) * n_rows / delta)
    n0 = np.asarray(n0, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.sqrt(2.0 * lt / n0)
    return np.minimum(np.where(covered, beta, 2.0), 2.0)


def weissman_beta(counts, support_sizes, delta, n_rows):
    """SUPERSEDED for certification: the radius sized at the realised,
    trajectory-dependent count N(s,a). Kept only for the comparison of
    Experiment 1 ("if all samples could be used") and for reading old logs."""
    counts = np.asarray(counts, dtype=np.float64)
    k = np.clip(np.asarray(support_sizes, dtype=np.float64), 2, None)
    log_term = np.log((np.power(2.0, k) - 2.0) * n_rows / delta)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.sqrt(2.0 * log_term / counts)
    beta = np.where(counts > 0, beta, 2.0)
    return np.minimum(beta, 2.0)


# ----------------------------------------------------------------------------
# Inner maximisation over the L1 ball (Proposition 1)
# ----------------------------------------------------------------------------
def robust_inner_max(phat, V, beta):
    """Reference (scalar) implementation of
        sigma(C; V) = sup { p . V : ||p - phat||_1 <= beta, p in the simplex }.
    The supremum moves up to beta/2 of mass from the lowest-value atoms to the
    highest-value atom. Used as the oracle in the unit tests."""
    phat = np.asarray(phat, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    k = V.shape[0]
    if k == 1:
        return float(V[0])
    if beta >= 2.0 - 1e-12:
        return float(V.max())
    budget = min(beta / 2.0, 1.0)
    if budget <= 0.0:
        return float(phat @ V)
    p = phat.copy()
    order_desc = np.argsort(V, kind="stable")[::-1]
    order_asc = order_desc[::-1]
    add_left = budget
    for i in order_desc:
        if add_left <= 1e-15:
            break
        room = 1.0 - p[i]
        d = room if room < add_left else add_left
        p[i] += d
        add_left -= d
    added = budget - add_left
    rem_left = added
    for i in order_asc:
        if rem_left <= 1e-15:
            break
        d = p[i] if p[i] < rem_left else rem_left
        p[i] -= d
        rem_left -= d
    return float(p @ V)


def build_padded(Phat, support, support_sizes):
    """Pad every row's support to a common width so the backup vectorises.
    Returns supp_idx (S,A,kmax) successor indices, supp_mask (S,A,kmax) and
    phat_pad (S,A,kmax) the estimated probabilities on the support."""
    S, A, _ = Phat.shape
    kmax = int(support_sizes.max())
    supp_idx = np.zeros((S, A, kmax), dtype=np.int64)
    supp_mask = np.zeros((S, A, kmax), dtype=bool)
    phat_pad = np.zeros((S, A, kmax), dtype=np.float64)
    for s in range(S):
        for a in range(A):
            idx = support[s][a]
            k = len(idx)
            supp_idx[s, a, :k] = idx
            supp_mask[s, a, :k] = True
            phat_pad[s, a, :k] = Phat[s, a, idx]
    return dict(supp_idx=supp_idx, supp_mask=supp_mask, phat_pad=phat_pad,
                kmax=kmax, Phat=Phat)


def _inner_max_core(Vs, phat_pad, supp_mask, budget):
    """Closed form of Proposition 1 on padded rows, for any leading batch
    shape. Vs, phat_pad, supp_mask: (..., kmax); budget = min(beta/2, 1): (...).

    Sort the support by value, move m = min(budget, 1 - phat_top) of mass onto
    the highest-value atom, and take it from the lowest-value atoms in
    ascending order. This is exact: the top atom can hold at most 1, and the
    other atoms hold exactly 1 - phat_top >= m between them, so the drain never
    touches the atom being filled."""
    sortkey = np.where(supp_mask, -Vs, np.inf)
    order = np.argsort(sortkey, axis=-1, kind="stable")
    ph_s = np.take_along_axis(np.broadcast_to(phat_pad, Vs.shape), order, axis=-1)
    V_s = np.take_along_axis(Vs, order, axis=-1)
    m_s = np.take_along_axis(np.broadcast_to(supp_mask, Vs.shape), order, axis=-1)
    ph_s = np.where(m_s, ph_s, 0.0)
    V_s = np.where(m_s, V_s, 0.0)
    base = (ph_s * V_s).sum(axis=-1)
    m = np.minimum(budget, 1.0 - ph_s[..., 0])
    ph_rev = ph_s[..., ::-1]
    V_rev = V_s[..., ::-1]
    cum_before = np.cumsum(ph_rev, axis=-1) - ph_rev
    rem = np.clip(m[..., None] - cum_before, 0.0, ph_rev)
    Q = base + m * V_s[..., 0] - (rem * V_rev).sum(axis=-1)
    # A vacuous row (radius 2, i.e. budget 1) is the whole simplex over the
    # support, whatever its centre: the supremum is the largest value there.
    return np.where(budget >= 1.0 - 1e-12, V_s[..., 0], Q)


def robust_backup_vec(Vprev, pad, beta):
    """Q[s,a] = sigma(C_{s,a}; Vprev) for every row, as an (S, A) array."""
    supp_idx, supp_mask, phat_pad = pad["supp_idx"], pad["supp_mask"], pad["phat_pad"]
    Vs = np.where(supp_mask, Vprev[supp_idx], 0.0)
    budget = np.minimum(beta / 2.0, 1.0)
    return _inner_max_core(Vs, phat_pad, supp_mask, budget)


def robust_backup_batched(Vb, pad, beta):
    """The same backup for a batch of value functions Vb of shape (B, S);
    returns (B, S, A)."""
    supp_idx, supp_mask, phat_pad = pad["supp_idx"], pad["supp_mask"], pad["phat_pad"]
    Vs = np.where(supp_mask[None], Vb[:, supp_idx], 0.0)
    budget = np.broadcast_to(np.minimum(beta / 2.0, 1.0)[None], Vs.shape[:-1])
    return _inner_max_core(Vs, phat_pad[None], supp_mask[None], budget)


# ----------------------------------------------------------------------------
# Value iteration
# ----------------------------------------------------------------------------
def robust_reach_avoid_fast(pad, beta, D, unsafe, sink, T):
    """Robust finite-horizon reach-avoid value iteration (8) under shield D.
    Returns (Vbar_T (S,), Q (S,A)) where Q is the last backup
    sigma(C_{s,a}; Vbar_{T-1}), the quantity line 4 of Algorithm 1 ranks."""
    Vprev = unsafe.astype(np.float64).copy()
    NEG = -1e9
    Q = None
    for _ in range(T):
        Q = robust_backup_vec(Vprev, pad, beta)
        Vnew = np.clip(np.where(D, Q, NEG).max(axis=1), 0.0, 1.0)
        Vnew[unsafe] = 1.0
        Vnew[sink] = 0.0
        Vprev = Vnew
    return Vprev, Q


def eval_removals_batched(D, cand_sa, pad, beta, unsafe, sink, T):
    """Vbar_T under each of the candidate shields D minus one (s,a); returns
    (B, S). Line 5 of Algorithm 1, evaluated for all candidates at once."""
    S, A = D.shape
    B = len(cand_sa)
    Db = np.broadcast_to(D, (B, S, A)).copy()
    for b, (s, a) in enumerate(cand_sa):
        Db[b, s, a] = False
    Vb = np.broadcast_to(unsafe.astype(np.float64), (B, S)).copy()
    NEG = -1e9
    for _ in range(T):
        Qb = robust_backup_batched(Vb, pad, beta)
        Vb = np.clip(np.where(Db, Qb, NEG).max(axis=2), 0.0, 1.0)
        Vb[:, unsafe] = 1.0
        Vb[:, sink] = 0.0
    return Vb


def true_reach_avoid(P, D, unsafe, sink, T, return_policy=False):
    """Exact worst-case reach-avoid probability on a KNOWN kernel (beta = 0),
    i.e. V*_T of (3); optionally the worst-case (adversarial) policy, indexed
    [t, s] for t = 0..T-1."""
    S, A, _ = P.shape
    Vprev = unsafe.astype(np.float64).copy()
    pol = np.zeros((T, S), dtype=np.int64)
    for t in range(T):
        Vnew = np.zeros(S)
        for s in range(S):
            if unsafe[s]:
                Vnew[s] = 1.0
                pol[T - 1 - t, s] = -1
                continue
            if s == sink:
                Vnew[s] = 0.0
                pol[T - 1 - t, s] = -1
                continue
            best, barg = -1.0, 0
            for a in range(A):
                if not D[s, a]:
                    continue
                qa = P[s, a] @ Vprev
                if qa > best:
                    best, barg = qa, a
            Vnew[s] = best
            pol[T - 1 - t, s] = barg
        Vprev = Vnew
    return (Vprev, pol) if return_policy else Vprev


# ----------------------------------------------------------------------------
# Helpers shared by the synthesis routines
# ----------------------------------------------------------------------------
def occupancy(Phat, pol, unsafe, sink, s0, T):
    """Expected visit counts of each state from s0 under the deterministic
    policy ``pol`` on the estimated kernel, over T steps."""
    S = Phat.shape[0]
    Ppi = Phat[np.arange(S), pol]
    occ = np.zeros(S)
    d = np.zeros(S)
    d[s0] = 1.0
    for _ in range(T):
        occ += d
        d = d @ Ppi
        d = d.copy()
        d[unsafe] = 0.0
        d[sink] = 0.0
    return occ


def reachable(D, supp_idx, supp_mask, s0, unsafe, sink):
    """States reachable from s0 along the supports of permitted actions."""
    S, A, _ = supp_idx.shape
    seen = np.zeros(S, dtype=bool)
    seen[s0] = True
    stack = [s0]
    while stack:
        s = stack.pop()
        if unsafe[s] or s == sink:
            continue
        row = D[s]
        for a in range(A):
            if not row[a]:
                continue
            for s2 in supp_idx[s, a][supp_mask[s, a]]:
                if not seen[s2] and not unsafe[s2] and s2 != sink:
                    seen[s2] = True
                    stack.append(s2)
    return seen
