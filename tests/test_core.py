"""Unit and regression tests. Run with  python -m pytest tests  (or python tests/test_core.py).

1. The vectorised closed form of Proposition 1 agrees with the scalar
   reference implementation on random rows, including ties, vacuous rows and
   single-atom supports; the batched backup agrees with the per-row one.
2. The primitives inlined in the self-contained Colab script of Experiment 3
   build exactly the shield the library builds.
3. Regression values of the letter: the shield of Experiment 1 and the
   robust and plug-in shields of Table III.
"""
import ast
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from shield import grid as G                                  # noqa: E402
from shield import robust as R                                # noqa: E402
from shield import sandbox as sx                              # noqa: E402


def test_inner_max_closed_form_matches_reference():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(5000):
        k = int(rng.integers(1, 7))
        p = rng.dirichlet(np.ones(k) * rng.choice([0.2, 1.0, 5.0]))
        if rng.random() < 0.3:                       # ties and zeros in phat
            p = np.round(p, 1)
            p = p / p.sum() if p.sum() > 0 else np.ones(k) / k
        V = rng.random(k)
        if rng.random() < 0.3:                       # ties in V
            V = np.round(V, 1)
        beta = float(rng.choice([0.0, 0.05, 0.2, 0.4, 0.8, 1.2, 1.9, 2.0]))
        ref = R.robust_inner_max(p, V, beta)
        Vs = np.zeros((1, 1, 6))
        ph = np.zeros((1, 1, 6))
        mk = np.zeros((1, 1, 6), bool)
        Vs[0, 0, :k], ph[0, 0, :k], mk[0, 0, :k] = V, p, True
        got = float(R._inner_max_core(Vs, ph, mk, np.array([[min(beta / 2, 1.0)]]))[0, 0])
        worst = max(worst, abs(got - ref))
    assert worst < 1e-12, worst


def test_batched_backup_matches_vectorised_backup():
    env = G.make_env()
    r = G.certified_shield_from_logs(env, 2500, G.dataset_seed(2500, 0))
    pad, beta = r["pad"], r["beta"]
    unsafe, sink = env["unsafe"], env["sink"]
    rng = np.random.default_rng(1)
    for _ in range(5):
        Vb = rng.random((4, env["nstate"]))
        Qb = R.robust_backup_batched(Vb, pad, beta)
        for b in range(4):
            assert np.allclose(Qb[b], R.robust_backup_vec(Vb[b], pad, beta), atol=1e-12)
    # and the batched evaluation of removals equals a full value iteration
    S, A = r["D"].shape
    D = np.ones((S, A), bool)
    D[unsafe] = False
    D[sink] = False
    cand = [(s, a) for s in np.where(r["live"])[0][:10] for a in range(A - 1)]
    Vb = R.eval_removals_batched(D, cand, pad, beta, unsafe, sink, G.T)
    for b, (s, a) in enumerate(cand):
        D2 = D.copy()
        D2[s, a] = False
        V2, _ = R.robust_reach_avoid_fast(pad, beta, D2, unsafe, sink, G.T)
        assert np.allclose(Vb[b], V2, atol=1e-12)


def _colab_namespace():
    """The function definitions of the Colab script, without its imports,
    settings or driver (which need torch and a GPU)."""
    path = os.path.join(ROOT, "experiments", "exp3_llm_sweep_colab.py")
    tree = ast.parse(open(path).read())
    keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            and n.name not in ("LocalLLM", "_load_model", "main", "run_episode")]
    assigns = [n for n in tree.body if isinstance(n, ast.Assign)
               and all(isinstance(t, (ast.Name, ast.Tuple)) for t in n.targets)
               and not any(isinstance(v, ast.Call) for v in ast.walk(n.value))]
    mod = ast.Module(body=assigns + keep, type_ignores=[])
    ns = {"np": np, "Counter": __import__("collections").Counter}
    exec(compile(mod, path, "exec"), ns)
    return ns


def test_colab_script_builds_the_same_shield():
    ns = _colab_namespace()
    env_c = ns["build_fs_env"]()
    env = sx.build_fs_env()
    assert np.allclose(env_c["P"], env["P"])
    # the inlined estimator and synthesis
    n0 = np.full((ns["NS"], ns["A"]), ns["M_LOG"], dtype=np.int64)
    pref, _full, Nsa = ns["log_counts_prefix"](env_c, ns["M_LOG"], ns["HORIZON"],
                                               np.random.default_rng(20000), n0)
    covered = Nsa >= n0
    den = np.where(covered, n0, 1.0)[:, :, None]
    Phat = np.divide(pref, den, out=np.zeros(pref.shape, float), where=den > 0)
    R_ = int((~env_c["unsafe"]).sum() - 1) * ns["A"]
    k = np.clip(np.asarray(env_c["support_sizes"], float), 2, None)
    lt = np.log((2.0 ** k - 2.0) * R_ / ns["DELTA"])
    beta = np.minimum(np.where(covered, np.sqrt(2 * lt / ns["M_LOG"]), 2.0), 2.0)
    pad = ns["build_padded"](Phat, env_c["support"], env_c["support_sizes"])
    w = ns["competent_occupancy"](env_c, ns["HORIZON"])
    Dc, Vc, _ = ns["synth"](pad, beta, env_c, env_c["start"], ns["HORIZON"], ns["EPS"], w)
    # the library
    est = sx.estimator_from_logs(env, seed=20000)
    assert np.allclose(est["Phat"], Phat) and np.allclose(est["beta"], beta)
    Dl, Vl, _ = sx.synth(est["pad"], est["beta"], env, env["start"], sx.T, sx.EPS,
                         sx.competent_occupancy(env, sx.T))
    assert (Dc == Dl).all()
    assert abs(float(Vc[env["start"]]) - float(Vl[env["start"]])) < 1e-12
    # the inlined backup is the library's
    V = np.random.default_rng(3).random(ns["NS"])
    assert np.allclose(ns["robust_backup_vec"](V, pad, beta), R.robust_backup_vec(V, pad, beta))
    assert ns["D_halt"](env_c).sum() == 16


def test_experiment1_shield_regression():
    env = G.make_env()
    r = G.certified_shield_from_logs(env, 12000, G.dataset_seed(12000, 0))
    Vstar = float(R.true_reach_avoid(env["P"], r["D"], env["unsafe"], env["sink"], G.T)[env["start"]])
    assert abs(r["Vbar"] - 0.0716) < 5e-4, r["Vbar"]
    assert abs(Vstar - 0.0094) < 5e-4, Vstar
    assert int((~r["D"][r["live"]]).sum()) == 101
    assert Vstar <= r["Vbar"]


def test_table3_shields_regression():
    env = sx.build_fs_env()
    est = sx.estimator_from_logs(env, seed=20000)
    w = sx.competent_occupancy(env, sx.T)
    Dr, Vr, _ = sx.synth(est["pad"], est["beta"], env, env["start"], sx.T, sx.EPS, w)
    assert sx.fenced_actions(env, Dr) == [(0, 1), (0, 3), (2, 3), (14, 3)]
    assert float(Vr[env["start"]]) == 0.0
    assert sx.adversarial_risk(env, Dr, sx.T) == 0.0
    assert abs(sx.forb_frac(Dr) - 4 / 96) < 1e-12
    Dp, Vp, _ = sx.synth(est["pad"], np.zeros_like(est["beta"]), env, env["start"], sx.T, sx.EPS, w)
    assert float(Vp[env["start"]]) == 0.0 and sx.adversarial_risk(env, Dp, sx.T) > 0.99
    assert int(est["total"][est["covered"]].min()) == 9118


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
