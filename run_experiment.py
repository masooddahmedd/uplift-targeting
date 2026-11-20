# Trains every learner on one outcome (visit or conversion) and evaluates on a held-out test set.
#
#   python run_experiment.py conversion        # results/conversion.json + results/scores_conversion.parquet
#
# Split: 60% train, 20% validation (early stopping only), 20% test, stratified on treatment x outcome
# with a fixed seed. Every reported number is on the test set.
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from uplift import FEATURES, Learners, evaluate_all

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
SEED = 7


def split(df, outcome):
    rng = np.random.default_rng(SEED)
    strata = df["treatment"].to_numpy() * 2 + df[outcome].to_numpy()
    part = np.empty(len(df), dtype=np.int8)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        rng.shuffle(idx)
        a, b = int(0.6 * len(idx)), int(0.8 * len(idx))
        part[idx[:a]], part[idx[a:b]], part[idx[b:]] = 0, 1, 2
    return part


def ate(t, y):
    rt, rc = y[t == 1].mean(), y[t == 0].mean()
    se = np.sqrt(rt * (1 - rt) / (t == 1).sum() + rc * (1 - rc) / (t == 0).sum())
    return {
        "treated_rate": float(rt),
        "control_rate": float(rc),
        "ate": float(rt - rc),
        "ate_ci95": [float(rt - rc - 1.96 * se), float(rt - rc + 1.96 * se)],
        "relative_lift": float((rt - rc) / rc),
    }


def main(outcome: str) -> None:
    t0 = time.time()
    df = pd.read_parquet(ROOT / "data" / "criteo.parquet")
    part = split(df, outcome)
    X = df[FEATURES].to_numpy(np.float32)
    t = df["treatment"].to_numpy(np.int8)
    y = df[outcome].to_numpy(np.int8)
    tr, va, te = part == 0, part == 1, part == 2
    print(outcome, "rows", len(df), "train/valid/test", tr.sum(), va.sum(), te.sum(), flush=True)

    L = Learners(X[tr], t[tr], y[tr], X[va], t[va], y[va], seed=SEED)
    timings = {}
    for name, step in [
        ("response", L.fit_response),
        ("S", L.fit_s),
        ("T", L.fit_t),
        ("X+DR", L.fit_x_and_dr),
    ]:
        s = time.time()
        step()
        timings[name] = round(time.time() - s, 1)
        print("fitted", name, timings[name], "s", flush=True)

    scores = L.scores(X[te])
    rng = np.random.default_rng(SEED)
    scores["random"] = rng.random(te.sum())
    tt, yt = t[te], y[te]

    res = {
        "outcome": outcome,
        "n": {"train": int(tr.sum()), "valid": int(va.sum()), "test": int(te.sum())},
        "propensity": L.p,
        "test_ate": ate(tt, yt),
        "timings_s": timings,
        "learners": {},
    }
    res["learners"] = evaluate_all(scores, tt, yt)
    for name, entry in res["learners"].items():
        print(
            name,
            "qini",
            round(entry["qini_coefficient"], 3),
            {k: round(v, 3) for k, v in entry["captured"].items()},
            flush=True,
        )

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{outcome}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    pd.DataFrame({"treatment": tt, "y": yt, **scores}).to_parquet(
        RESULTS / f"scores_{outcome}.parquet", index=False
    )
    print("done in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main(sys.argv[1])
