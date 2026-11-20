# Recomputes every test-set metric from the saved predictions (results/scores_*.parquet) without
# retraining, e.g. after a fix to the evaluation code.
#
#   python rescore.py
import json
from pathlib import Path

import pandas as pd

from uplift import evaluate_all

RES = Path(__file__).parent / "results"

for outcome in ("visit", "conversion"):
    jp = RES / f"{outcome}.json"
    res = json.loads(jp.read_text(encoding="utf-8"))
    df = pd.read_parquet(RES / f"scores_{outcome}.parquet")
    scores = {c: df[c].to_numpy() for c in df.columns if c not in ("treatment", "y")}
    res["learners"] = evaluate_all(scores, df["treatment"].to_numpy(), df["y"].to_numpy())
    jp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for name, e in res["learners"].items():
        print(
            outcome,
            name,
            "qini",
            round(e["qini_coefficient"], 3),
            {k: round(v, 3) for k, v in e["captured"].items()},
            "ties",
            round(e["tied_share"], 3),
        )
