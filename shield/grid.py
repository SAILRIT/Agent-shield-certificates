"""The grid benchmark of Section V-A: the environment, exploratory logging with
the committed-prefix estimator of Section III-A, the nominal task occupancy
used as the permissiveness weight, and Algorithm 1.
"""
import numpy as np

from .robust import (build_padded, budget_beta, eval_removals_batched,
                     occupancy, reachable, robust_reach_avoid_fast)

# grid actions
N, E, S, W, HALT = 0, 1, 2, 3, 4
A_HALT = HALT
MOVE_DELTA = {N: (-1, 0), E: (0, 1), S: (1, 0), W: (0, -1)}

DELTA = 0.05      # confidence parameter of the certificate
T = 16            # task horizon
EPS = 0.10        # risk budget


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
def build_gridworld(nrows=5, ncols=5, slip=0.15,
                    hazard_cells=((0, 3), (0, 4), (1, 3), (1, 4)),
                    start_cell=(4, 0), goal_cell=(4, 4)):
    """A gridworld whose cells are the abstract states; a sink models HALT.
    Each move succeeds with probability 1 - slip and otherwise executes one of
    the three other moves; moving into a wall leaves the state unchanged."""
    ncell = nrows * ncols
    nstate = ncell + 1
    sink = ncell
    A = 5

    def cid(r, c):
        return r * ncols + c

    hazard = set(hazard_cells)
    unsafe = np.zeros(nstate, dtype=bool)
    for (r, c) in hazard:
        unsafe[cid(r, c)] = True
    P = np.zeros((nstate, A, nstate))

    def move(r, c, a):
        dr, dc = MOVE_DELTA[a]
        rr, cc = r + dr, c + dc
        return (rr, cc) if (0 <= rr < nrows and 0 <= cc < ncols) else (r, c)

    for r in range(nrows):
        for c in range(ncols):
            s = cid(r, c)
            if unsafe[s]:
                P[s, :, s] = 1.0
                continue
            for a in (N, E, S, W):
                dist = {}
                rr, cc = move(r, c, a)
                dist[cid(rr, cc)] = dist.get(cid(rr, cc), 0.0) + (1.0 - slip)
                others = [x for x in (N, E, S, W) if x != a]
                for o in others:
                    orr, occ = move(r, c, o)
                    dist[cid(orr, occ)] = dist.get(cid(orr, occ), 0.0) + slip / len(others)
                for s2, pr in dist.items():
                    P[s, a, s2] += pr
            P[s, HALT, sink] = 1.0
    P[sink, :, sink] = 1.0

    support = [[np.where(P[s, a] > 0)[0] for a in range(A)] for s in range(nstate)]
    support_sizes = np.array([[len(support[s][a]) for a in range(A)]
                              for s in range(nstate)])
    assert np.allclose(P.sum(axis=2), 1.0)
    return dict(P=P, unsafe=unsafe, sink=sink, nstate=nstate, A=A,
                nrows=nrows, ncols=ncols, start=cid(*start_cell),
                goal=cid(*goal_cell), hazard=hazard, support=support,
                support_sizes=support_sizes, goal_cell=goal_cell, slip=slip)


def make_env():
    """The benchmark of the letter: 7 x 7 cells, the central 3 x 3 block
    unsafe, slip 0.16, start bottom-left, goal top-right."""
    return build_gridworld(nrows=7, ncols=7, slip=0.16,
                           hazard_cells=tuple((r, c) for r in range(2, 5)
                                              for c in range(2, 5)),
                           start_cell=(6, 0), goal_cell=(0, 6))


def live_rows(env):
    """Boolean mask of the live states and the number R of rows (live states
    times actions) the union bound of (5) ranges over."""
    live = ~env["unsafe"]
    live[env["sink"]] = False
    return live, int(live.sum()) * env["A"]


def row_budget(M, T, n_rows):
    """The budget n_0 = floor(0.7 M T / R) of Section V-A, the same for every row."""
    return int(0.7 * M * T / n_rows)


# ----------------------------------------------------------------------------
# Logging and the committed-prefix estimator
# ----------------------------------------------------------------------------
def behavior_counts_prefix(env, M, T, rng, n0):
    """M exploratory rollouts of T steps: uniform-random start over the safe
    cells, uniform-random actions, restart on absorption. Returns
    (prefix_counts, total_counts): prefix_counts[s,a,s'] counts the successors
    among the FIRST n_0(s,a) visits of row (s,a), in the order of simulated time
    and then rollout index; total_counts[s,a] is N(s,a), used only to test
    whether the budget was met."""
    P, nstate, A = env["P"], env["nstate"], env["A"]
    unsafe, sink = env["unsafe"], env["sink"]
    safe_cells = np.where(~unsafe)[0]
    safe_cells = safe_cells[safe_cells != sink]
    cum = np.cumsum(P, axis=2)
    s = rng.choice(safe_cells, size=M)
    idx = np.arange(M, dtype=np.int64)
    rows_l, keys_l, nxt_l = [], [], []
    for t in range(T):
        a = rng.integers(0, A, size=M)
        u = rng.random(M)
        s2 = (u[:, None] < cum[s, a]).argmax(axis=1)
        rows_l.append(s.astype(np.int64) * A + a)
        keys_l.append(t * M + idx)            # time first, then rollout index
        nxt_l.append(s2.astype(np.int64))
        absorbed = unsafe[s2] | (s2 == sink)
        s = np.where(absorbed, rng.choice(safe_cells, size=M), s2)
    rows = np.concatenate(rows_l)
    keys = np.concatenate(keys_l)
    nxt = np.concatenate(nxt_l)
    nrow = nstate * A
    total = np.bincount(rows, minlength=nrow).reshape(nstate, A)
    order = np.lexsort((keys, rows))          # by row, then by arrival order
    rs = rows[order]
    starts = np.searchsorted(rs, np.arange(nrow), side="left")
    rank = np.arange(rs.size) - starts[rs]    # 0-based arrival index within row
    n0flat = np.asarray(n0, dtype=np.int64).reshape(-1)
    sel = order[rank < n0flat[rs]]
    pref = np.zeros((nstate, A, nstate), dtype=np.int64)
    np.add.at(pref, (rows[sel] // A, rows[sel] % A, nxt[sel]), 1)
    return pref, total


def behavior_counts(env, M, T, rng):
    """SUPERSEDED for certification: aggregate counts N(s,a,s') over ALL visits
    of the same rollouts (same seed stream as behavior_counts_prefix). Kept for
    the comparison of Experiment 1, "if all samples could be used"."""
    P, nstate, A = env["P"], env["nstate"], env["A"]
    unsafe, sink = env["unsafe"], env["sink"]
    safe_cells = np.where(~unsafe)[0]
    safe_cells = safe_cells[safe_cells != sink]
    counts = np.zeros((nstate, A, nstate), dtype=np.int64)
    s = rng.choice(safe_cells, size=M)
    cum = np.cumsum(P, axis=2)
    for _ in range(T):
        a = rng.integers(0, A, size=M)
        u = rng.random(M)
        s2 = (u[:, None] < cum[s, a]).argmax(axis=1)
        np.add.at(counts, (s, a, s2), 1)
        absorbed = unsafe[s2] | (s2 == sink)
        s2 = np.where(absorbed, rng.choice(safe_cells, size=M), s2)
        s = s2
    return counts


def estimate_kernel(counts):
    """SUPERSEDED for certification: the empirical kernel centred on all
    observations (paired with weissman_beta at the realised counts)."""
    N_sa = counts.sum(axis=2, keepdims=True)
    Phat = np.divide(counts, N_sa, out=np.zeros_like(counts, dtype=np.float64),
                     where=N_sa > 0)
    return Phat, N_sa[:, :, 0]


def prefix_estimator(pref, total, n0):
    """The estimator of Section III-A: Phat(.|s,a) is the empirical law of the
    first n_0(s,a) successors. Returns (Phat, covered) with covered[s,a] true
    when the budget was met. A row below budget receives the vacuous
    confidence set from budget_beta, so its centre is immaterial; it is set to
    the empirical law of the observations it does have (all of which lie in the
    prefix), or to zero when it was never visited."""
    n0 = np.asarray(n0, dtype=np.float64)
    total = np.asarray(total, dtype=np.float64)
    covered = total >= n0
    denom = np.where(covered, n0, np.maximum(total, 1.0))[:, :, None]
    Phat = np.divide(pref, denom, out=np.zeros(pref.shape, dtype=np.float64),
                     where=denom > 0)
    return Phat, covered


def goal_seeking_policy(env):
    """The stochastic goal-seeking, hazard-avoiding reference policy (S, A):
    with probability 0.8 a safe move that reduces the Manhattan distance to the
    goal, with probability 0.2 another safe move; HALT where no safe move
    exists. It is the competent controller of Experiment 1 and the policy whose
    occupancy weights Algorithm 1."""
    S, A = env["nstate"], env["A"]
    unsafe, sink = env["unsafe"], env["sink"]
    ncols = env["ncols"]
    gr, gc = env["goal_cell"]
    pol = np.zeros((S, A))
    for s in range(S):
        if unsafe[s] or s == sink:
            pol[s, A_HALT] = 1.0
            continue
        r, c = divmod(s, ncols)
        safe_moves = []
        for a, (dr, dc) in MOVE_DELTA.items():
            rr, cc = r + dr, c + dc
            if 0 <= rr < env["nrows"] and 0 <= cc < ncols and not unsafe[rr * ncols + cc]:
                safe_moves.append((a, abs(rr - gr) + abs(cc - gc)))
        if not safe_moves:
            pol[s, A_HALT] = 1.0
            continue
        dmin = min(d for _, d in safe_moves)
        best = [a for a, d in safe_moves if d == dmin]
        others = [a for a, d in safe_moves if d != dmin]
        for a in best:
            pol[s, a] += 0.8 / len(best)
        for a in (others if others else best):
            pol[s, a] += 0.2 / len(others if others else best)
    return pol


def nominal_occupancy(env, T):
    """State-action occupancy of the goal-seeking policy over T steps from the
    start, the weight c(s,a) of Algorithm 1: forbidding an action the task
    relies on costs more, so the shield fences the hazard rather than moating
    the start."""
    P, S = env["P"], env["nstate"]
    unsafe, sink = env["unsafe"], env["sink"]
    pol = goal_seeking_policy(env)
    Ppi = np.einsum("sa,sat->st", pol, P)
    occ = np.zeros(S)
    d = np.zeros(S)
    d[env["start"]] = 1.0
    for _ in range(T):
        occ += d
        d = d @ Ppi
        d = d.copy()
        d[unsafe] = 0.0
        d[sink] = 0.0
    occ[unsafe] = 0.0
    occ[sink] = 0.0
    return occ[:, None] * pol


# ----------------------------------------------------------------------------
# Algorithm 1
# ----------------------------------------------------------------------------
def synthesize_shield(pad, beta, unsafe, sink, s0, T, eps, a_halt,
                      weight=None, protect=None):
    """Algorithm 1 of the letter (greedy least-restrictive certified shield).

    Starting from the permit-all shield, while Vbar_T(s0) > eps:
      line 4: at every live state s with positive occupancy under the current
              worst-case policy, the candidate is the permitted action other
              than HALT with the largest robust Q-value sigma(C_{s,a}; Vbar_{T-1});
      line 5: Vbar_T(s0) is evaluated for each candidate removal;
      line 6: candidates that bring Vbar_T(s0) within the budget rank first,
              the one leaving the highest Vbar_T(s0) first (the least
              restrictive); the rest rank by risk reduction per unit of
              weight c(s,a);
      line 7: the top-ranked removal that keeps `protect` (the goal) reachable
              is committed, or the top-ranked one if none does.
    Returns (D, Vbar, Q) with Vbar the certified value function of D."""
    S, A = beta.shape
    Phat = pad["Phat"]
    supp_idx, supp_mask = pad["supp_idx"], pad["supp_mask"]
    if weight is None:
        weight = np.ones((S, A))
    w0 = 1e-3 * float(weight.max() + 1e-12)
    D = np.ones((S, A), dtype=bool)
    D[unsafe, :] = False
    D[sink, :] = False
    live = ~unsafe
    live[sink] = False
    V, Q = robust_reach_avoid_fast(pad, beta, D, unsafe, sink, T)
    while V[s0] > eps:
        pol = np.where(D, Q, -np.inf).argmax(axis=1)
        occ = occupancy(Phat, pol, unsafe, sink, s0, T)
        cand_sa, cand_cost = [], []
        for s in np.where(live & (D.sum(axis=1) > 1))[0]:
            perm = np.where(D[s])[0]
            perm = perm[perm != a_halt]
            if perm.size == 0 or occ[s] <= 1e-12:
                continue
            a_star = perm[int(np.argmax(Q[s, perm]))]
            cand_sa.append((s, a_star))
            cand_cost.append(float(weight[s, a_star]) + w0)
        if not cand_sa:
            break
        Vfin = eval_removals_batched(D, cand_sa, pad, beta, unsafe, sink, T)
        vs0 = Vfin[:, s0]
        cand_cost = np.asarray(cand_cost)
        benefit = float(V[s0]) - vs0
        is_reach = vs0 <= eps + 1e-12
        prim = np.where(is_reach, 0, 1)
        sec = np.where(is_reach, -vs0, -benefit / cand_cost)
        order = np.lexsort((sec, prim))
        j = int(order[0])
        if protect is not None:
            for jj in order:
                s, a = cand_sa[jj]
                D[s, a] = False
                keep = reachable(D, supp_idx, supp_mask, s0, unsafe, sink)[protect]
                D[s, a] = True
                if keep:
                    j = int(jj)
                    break
        s, a = cand_sa[j]
        D[s, a] = False
        V, Q = robust_reach_avoid_fast(pad, beta, D, unsafe, sink, T)
    return D, V, Q


def certified_shield_from_logs(env, M, seed, eps=EPS, T=T, delta=DELTA):
    """Log M exploratory episodes with the given seed, build the estimator of
    Section III-A with the budget of Section V-A and run Algorithm 1.
    Returns a dict with the shield D, its certificate, the data and radii."""
    live, n_rows = live_rows(env)
    n0v = row_budget(M, T, n_rows)
    n0 = np.full((env["nstate"], env["A"]), n0v, dtype=np.int64)
    rng = np.random.default_rng(seed)
    pref, total = behavior_counts_prefix(env, M, T, rng, n0)
    Phat, covered = prefix_estimator(pref, total, n0)
    beta = budget_beta(covered, env["support_sizes"], delta, n_rows, n0v)
    pad = build_padded(Phat, env["support"], env["support_sizes"])
    mu = nominal_occupancy(env, T)
    D, V, Q = synthesize_shield(pad, beta, env["unsafe"], env["sink"], env["start"],
                                T, eps, A_HALT, weight=mu, protect=env["goal"])
    return dict(D=D, Vbar=float(V[env["start"]]), V=V, Q=Q, pad=pad, beta=beta,
                Phat=Phat, covered=covered, total=total, n0=n0v, n_rows=n_rows,
                live=live)


def dataset_seed(M, sd, seed0=40000):
    """The seeding convention of the repository: dataset sd at sample size M."""
    return seed0 + 131 * int(M) + int(sd)
