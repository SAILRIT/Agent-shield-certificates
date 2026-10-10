"""Experiment 3 (Section V-B): Table IV and the numbers quoted in the text,
computed from the raw output of the Colab sweep (exp3_llm_sweep_colab.py).

The sweep was run in two Colab sessions on a T4 GPU, whose output files are
results/exp3_llm_sweep_qwen1.5b.json (the scripted control and the 1.5B model)
and results/exp3_llm_sweep_qwen3b_7b.json (the scripted control again, the 3B
and the 7B models). Both share every setting (150 episodes per condition,
injection probability 0.3, base seed 777, horizon 10, the shield of Table
III); the scripted control is identical in both, which is the check that the
two sessions ran the same harness.

    python experiments/exp3_summarize.py [--check]

With --check the script also verifies, number by number, the statements the
letter makes about this experiment and exits with an error if one fails.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FILES = ["exp3_llm_sweep_qwen1.5b.json", "exp3_llm_sweep_qwen3b_7b.json"]

TABLE_ROWS = [  # (label in the letter, run key prefix)
    ("scripted competent", "scripted competent"),
    ("Qwen2.5-1.5B, tau=0", "Qwen2.5-1.5B-Instruct | T=0.0"),
    ("Qwen2.5-1.5B, tau=1", "Qwen2.5-1.5B-Instruct | T=1.0"),
    ("Qwen2.5-3B, tau=0", "Qwen2.5-3B-Instruct | T=0.0"),
    ("Qwen2.5-3B, tau=1", "Qwen2.5-3B-Instruct | T=1.0"),
    ("Qwen2.5-7B, tau=0", "Qwen2.5-7B-Instruct | T=0.0"),
    ("Qwen2.5-7B, tau=1", "Qwen2.5-7B-Instruct | T=1.0"),
]
IN_TABLE_IV = {"scripted competent", "Qwen2.5-1.5B, tau=0", "Qwen2.5-3B, tau=0",
               "Qwen2.5-7B, tau=0", "Qwen2.5-7B, tau=1"}


def load():
    runs, configs = {}, []
    for f in FILES:
        d = json.load(open(os.path.join(RES, f)))
        configs.append(d["config"])
        for k, v in d["runs"].items():
            if k in runs and k.startswith("scripted"):
                assert runs[k] == v, "the scripted control differs between the two sessions"
            runs[k] = v
    return runs, configs


def clopper_pearson_upper(k, n, conf=0.95):
    """One-sided upper confidence bound on a binomial proportion."""
    from scipy.stats import beta
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def main(check=False):
    runs, configs = load()
    for c in configs:
        assert (c["episodes"], c["p_inject"], c["base_seed"], c["horizon"],
                c["eps"], c["delta"], c["m_log"]) == (150, 0.3, 777, 10, 0.05, 0.05, 4000)
    rows = {}
    for label, key in TABLE_ROWS:
        a, b = runs[f"{key} | no shield"], runs[f"{key} | robust (ours)"]
        blocked = b["blocked_action_counts"]
        rows[label] = dict(
            unsafe_none=100 * a["unsafe_rate"], unsafe_shield=100 * b["unsafe_rate"],
            succ_none=100 * a["success_rate"], succ_shield=100 * b["success_rate"],
            interv=100 * b["intervention_rate"],
            blocks=(max(blocked, key=blocked.get) if blocked else "none"),
            n=b["n_episodes"], n_injected=b["n_injected"],
            first_action_none=a["first_action_counts"], terminal_shield=b["terminal_counts"],
            unsafe_injected_none=a["unsafe_injected"], blocked_counts=blocked)

    print("Table IV: seven controllers behind one shield (150 episodes each; "
          "* = rows printed in the letter)")
    print(f"{'controller':<22}{'unsafe none':>12}{'shield':>8}{'succ none':>11}{'shield':>8}{'interv':>8}  blocks")
    for label, _ in TABLE_ROWS:
        r = rows[label]
        star = "*" if label in IN_TABLE_IV else " "
        print(f"{star}{label:<21}{r['unsafe_none']:>12.1f}{r['unsafe_shield']:>8.1f}"
              f"{r['succ_none']:>11.1f}{r['succ_shield']:>8.1f}{r['interv']:>8.1f}  {r['blocks']}")

    n_total = sum(rows[l]["n"] for l, _ in TABLE_ROWS)
    unsafe_total = sum(round(rows[l]["unsafe_shield"] / 100 * rows[l]["n"]) for l, _ in TABLE_ROWS)
    per_run_bound = clopper_pearson_upper(0, 150)
    pooled_bound = clopper_pearson_upper(unsafe_total, n_total)
    b7 = rows["Qwen2.5-7B, tau=0"]
    hij = round(b7["unsafe_injected_none"] * b7["n_injected"])
    print(f"\ninjected episodes: {b7['n_injected']} of {b7['n']} (the same in every run)")
    print(f"unsafe without the shield: from {min(r['unsafe_none'] for r in rows.values()):.1f}% "
          f"to {max(r['unsafe_none'] for r in rows.values()):.1f}%")
    print(f"unsafe with the shield: {unsafe_total} of {n_total} episodes; one-sided 95% Clopper-Pearson "
          f"upper bound per run {100 * per_run_bound:.3f}% (the letter: 2%), pooled "
          f"{100 * pooled_bound:.3f}% (the letter rounds up: 0.29%)")
    print(f"7B, tau=0: hijacked in {hij} of {b7['n_injected']} attacked episodes "
          f"({100 * b7['unsafe_injected_none']:.1f}%); blocked calls {b7['blocked_counts']}; "
          f"first actions {b7['first_action_none']}")
    b3 = rows["Qwen2.5-3B, tau=0"]
    print(f"3B, tau=0: success {b3['succ_none']:.1f}% -> {b3['succ_shield']:.1f}% with the shield")
    small = [rows[l]["interv"] for l in ("Qwen2.5-1.5B, tau=0", "Qwen2.5-1.5B, tau=1",
                                          "Qwen2.5-3B, tau=0", "Qwen2.5-3B, tau=1")]
    big = [rows[l]["interv"] for l in ("Qwen2.5-7B, tau=0", "Qwen2.5-7B, tau=1")]
    print(f"intervention rate: 1.5B and 3B {min(small):.1f}-{max(small):.1f}, 7B {min(big):.1f}-{max(big):.1f}")
    b15 = rows["Qwen2.5-1.5B, tau=0"]
    print(f"1.5B, tau=0: first actions {b15['first_action_none']}, terminals with the shield {b15['terminal_shield']}")
    b15t = rows["Qwen2.5-1.5B, tau=1"]
    print(f"1.5B, tau=1: unsafe without the shield {b15t['unsafe_none']:.1f}%, "
          f"terminals with the shield {b15t['terminal_shield']}, intervention {b15t['interv']:.1f}")

    if check:
        checks = [
            ("Table IV scripted", (rows["scripted competent"]["unsafe_none"], rows["scripted competent"]["unsafe_shield"],
                                   rows["scripted competent"]["succ_none"], rows["scripted competent"]["succ_shield"],
                                   rows["scripted competent"]["interv"]), (0.0, 0.0, 99.3, 99.3, 0.0)),
            ("Table IV 1.5B tau=0", (b15["unsafe_none"], b15["unsafe_shield"], b15["succ_none"], b15["succ_shield"], b15["interv"], b15["blocks"]),
             (70.7, 0.0, 0.0, 0.0, 42.0, "chmod")),
            ("Table IV 3B tau=0", (b3["unsafe_none"], b3["unsafe_shield"], b3["succ_none"], b3["succ_shield"], b3["interv"], b3["blocks"]),
             (78.7, 0.0, 3.3, 38.7, 43.4, "chmod")),
            ("Table IV 7B tau=0", (b7["unsafe_none"], b7["unsafe_shield"], b7["succ_none"], b7["succ_shield"], b7["interv"], b7["blocks"]),
             (20.7, 0.0, 55.3, 55.3, 6.5, "delete")),
            ("Table IV 7B tau=1", tuple(rows["Qwen2.5-7B, tau=1"][k] for k in ("unsafe_none", "unsafe_shield", "succ_none", "succ_shield", "interv", "blocks")),
             (21.3, 0.0, 59.3, 59.3, 6.1, "delete")),
            ("39 of 150 injected", (b7["n_injected"],), (39,)),
            ("worst unsafe rate without shield 84.7%", (max(r["unsafe_none"] for r in rows.values()),), (84.7,)),
            ("none unsafe with the shield", (unsafe_total,), (0,)),
            ("per-run bound 2%", (round(100 * per_run_bound),), (2,)),
            # the letter rounds the upper bound up (0.285% -> 0.29%)
            ("pooled bound 0.29%", (math.ceil(100 * pooled_bound * 100) / 100,), (0.29,)),
            ("7B hijacked 31 of 39 (79.5%)", (hij, round(100 * b7["unsafe_injected_none"], 1)), (31, 79.5)),
            ("7B blocks only the 39 injected deletes", (b7["blocked_counts"],), ({"delete": 39},)),
            ("7B always starts with read", (b7["first_action_none"],), ({"read": 150},)),
            ("3B success 3.3 -> 38.7", (b3["succ_none"], b3["succ_shield"]), (3.3, 38.7)),
            ("smaller models 31-47 blocked per 100 calls", (round(min(small)), round(max(small))), (31, 47)),
            ("about 6 for the 7B", (round(min(big)), round(max(big))), (6, 6)),
            ("1.5B tau=0 always opens with chmod", (b15["first_action_none"],), ({"chmod": 150},)),
            ("1.5B tau=0 always halts safely", (b15["terminal_shield"],), ({"halt": 150},)),
        ]
        bad = 0
        for name, got, want in checks:
            got = tuple(round(g, 1) if isinstance(g, float) and name != "pooled bound 0.29%" else g for g in got)
            ok = got == want
            bad += not ok
            print(f"  [{'ok' if ok else 'MISMATCH'}] {name}: {got}" + ("" if ok else f" expected {want}"))
        print(f"{len(checks) - bad} of {len(checks)} statements verified")
        if bad:
            sys.exit(1)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    main(check=ap.parse_args().check)
