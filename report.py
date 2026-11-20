# Charts and the money analysis, from results/*.json and results/scores_*.parquet.
#
#   python report.py            # results/*.png and results/money.json
#
# The DR-learner is the primary model, fixed before any test result was seen; the others are shown
# for comparison. Money is computed per 1 million customers from the held-out test set:
#   for the top-k users by predicted uplift we observe the treated and control conversion rates
#   (rt_k, rc_k) in that slice, so incremental conversions = (rt_k - rc_k) * n_k, and every treated
#   conversion in the slice redeems the discount, including the ones that would have happened anyway.
#   profit(k) = margin * incremental - discount * rt_k * n_k - contact_cost * n_k
# Treat-everyone is k = 100%, treat-nobody is k = 0. The scenario values are assumptions and are
# printed next to every number; the shares of conversions and treatments are not.
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uplift import rank_order

ROOT = Path(__file__).parent
RES = ROOT / "results"
PRIMARY = "DR"
ORDER = ["random", "response", "S", "T", "X", "DR"]
LABEL = {
    "random": "random",
    "response": "response model (who buys)",
    "S": "S-learner",
    "T": "T-learner",
    "X": "X-learner",
    "DR": "DR-learner",
}
COLOR = {
    "random": "#9aa3b2",
    "response": "#e0703a",
    "S": "#b9a3e3",
    "T": "#6aa6e8",
    "X": "#3fbf9f",
    "DR": "#1f4e8c",
}
SCENARIOS = [  # (name, margin per order, discount per redeemed coupon, cost per contact)
    ("coupon: $10 off a $40-margin order", 40.0, 10.0, 0.0),
    ("coupon: $5 off a $40-margin order", 40.0, 5.0, 0.0),
    ("ad: $0.02 per contact, $40 margin", 40.0, 0.0, 0.02),
]
N_CUSTOMERS = 1_000_000


def slice_rates(score, t, y, n_points=201):
    order = rank_order(score)
    t, y = t[order].astype(float), y[order].astype(float)
    n = len(t)
    nt, nc = np.cumsum(t), np.cumsum(1 - t)
    yt, yc = np.cumsum(t * y), np.cumsum((1 - t) * y)
    ks = np.linspace(0, 1, n_points)
    rt = np.zeros(n_points)
    rc = np.zeros(n_points)
    for i, k in enumerate(ks[1:], 1):
        j = max(int(k * n) - 1, 0)
        rt[i] = yt[j] / max(nt[j], 1)
        rc[i] = yc[j] / max(nc[j], 1)
    return ks, rt, rc


def profit_curve(ks, rt, rc, margin, discount, contact):
    n_k = ks * N_CUSTOMERS
    inc = (rt - rc) * n_k
    return margin * inc - discount * rt * n_k - contact * n_k, inc


def money(df, n_splits=20, seed=0):
    """Out-of-sample profit of each targeting rule.

    The share of customers to target is chosen on a random half of the test set and the profit is
    measured on the other half (both directions, n_splits random splits), so no method gets credit
    for a threshold tuned on the data it is scored on. Reported per 1M customers: mean and the
    5th-95th percentile across splits, plus the gain over treating everyone on the same half.
    """
    t, y = df["treatment"].to_numpy(), df["y"].to_numpy()
    rng = np.random.default_rng(seed)
    out = {
        name: {s[0]: {"gain": [], "profit": [], "share": [], "treat_all": []} for s in SCENARIOS}
        for name in ORDER
    }
    for _ in range(n_splits):
        half = rng.random(len(t)) < 0.5
        for a, b in ((half, ~half), (~half, half)):
            for name in ORDER:
                sc = df[name].to_numpy()
                ka, rta, rca = slice_rates(sc[a], t[a], y[a])
                kb, rtb, rcb = slice_rates(sc[b], t[b], y[b])
                for sname, m, d, c in SCENARIOS:
                    pa, _ = profit_curve(ka, rta, rca, m, d, c)
                    pb, _ = profit_curve(kb, rtb, rcb, m, d, c)
                    i = int(np.argmax(pa))
                    rec = out[name][sname]
                    rec["profit"].append(pb[i])
                    rec["treat_all"].append(pb[-1])
                    rec["gain"].append(pb[i] - pb[-1])
                    rec["share"].append(ka[i])
    summary = {}
    for name, per in out.items():
        summary[name] = {}
        for sname, rec in per.items():
            g = np.array(rec["gain"])
            summary[name][sname] = {
                "share_targeted_median": float(np.median(rec["share"])),
                "profit_mean": float(np.mean(rec["profit"])),
                "treat_all_profit_mean": float(np.mean(rec["treat_all"])),
                "gain_vs_treat_all_mean": float(g.mean()),
                "gain_p05_p95": [float(np.percentile(g, 5)), float(np.percentile(g, 95))],
            }
    return summary


def plot_qini(res, outcome):
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=150)
    ks = np.linspace(0, 1, 101)
    for name in ORDER:
        q = np.array(res["learners"][name]["qini"])
        share = q / q[-1] * 100
        lw = 2.8 if name == PRIMARY else 1.6
        ax.plot(
            ks * 100,
            share,
            color=COLOR[name],
            lw=lw,
            label=f"{LABEL[name]} (Qini {res['learners'][name]['qini_coefficient']:.2f})",
        )
    ax.axhline(100, color="#ccc", lw=0.8, ls="--")
    ax.set_xlabel("% of customers targeted (ranked by the model)")
    ax.set_ylabel(f"% of all extra {outcome}s captured")
    ax.set_title(f"Who to target: extra {outcome}s captured vs. customers targeted", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(RES / f"qini_{outcome}.png")
    plt.close(fig)


def plot_deciles(res, outcome):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, sharey=True)
    for ax, name in zip(axes, ["response", PRIMARY]):
        d = res["learners"][name]["deciles"]
        obs = [x["obs"] * 100 for x in d]
        ax.bar(range(1, 11), obs, color=COLOR[name])
        ax.axhline(
            res["test_ate"]["ate"] * 100, color="#555", ls="--", lw=1, label="average effect"
        )
        ax.axhline(0, color="#222", lw=0.6)
        ax.set_title(f"ranked by {LABEL[name]}", fontsize=10)
        ax.set_xlabel("decile (1 = targeted first)")
        ax.set_xticks(range(1, 11))
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(f"observed uplift in {outcome} rate (pp)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Observed effect of the incentive by decile ({outcome})", fontsize=11)
    fig.tight_layout()
    fig.savefig(RES / f"deciles_{outcome}.png")
    plt.close(fig)


def plot_profit(df, outcome):
    """Profit curves for a pay-per-redemption coupon and a pay-per-contact ad, side by side."""
    t, y = df["treatment"].to_numpy(), df["y"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    for ax, (sname, m, d, c) in zip(axes, [SCENARIOS[0], SCENARIOS[2]]):
        for name in ["random", "response", PRIMARY]:
            ks, rt, rc = slice_rates(df[name].to_numpy(), t, y)
            prof, _ = profit_curve(ks, rt, rc, m, d, c)
            ax.plot(
                ks * 100,
                prof / 1000,
                color=COLOR[name],
                lw=2.6 if name == PRIMARY else 1.6,
                label=LABEL[name],
            )
        ax.axhline(0, color="#222", lw=0.6)
        ax.set_xlabel("% of customers targeted")
        ax.set_title(sname.replace("$", r"\$"), fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"campaign profit, \$k per 1M customers")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Pay per redemption: give it to everyone.  Pay per contact: target the top ~30%.\n"
        "(test set; margin and costs are assumptions)",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(RES / f"profit_{outcome}.png")
    plt.close(fig)


def main():
    summary = {}
    for outcome in ("visit", "conversion"):
        jp = RES / f"{outcome}.json"
        if not jp.exists():
            continue
        res = json.loads(jp.read_text(encoding="utf-8"))
        df = pd.read_parquet(RES / f"scores_{outcome}.parquet")
        plot_qini(res, outcome)
        plot_deciles(res, outcome)
        if outcome == "conversion":  # a visit is not a sale, so money is computed on conversions
            plot_profit(df, outcome)
        summary[outcome] = {
            "test_ate": res["test_ate"],
            "anyway_share_of_treated_conversions": res["test_ate"]["control_rate"]
            / res["test_ate"]["treated_rate"],
            "learners": {
                n: {
                    "qini_coefficient": v["qini_coefficient"],
                    "qini_coefficient_ci95": v["qini_coefficient_ci95"],
                    "captured": v["captured"],
                    "captured_ci95": v["captured_ci95"],
                    "top_decile_obs_uplift": v["deciles"][0]["obs"],
                    "bottom_decile_obs_uplift": v["deciles"][-1]["obs"],
                }
                for n, v in res["learners"].items()
            },
            "money": money(df) if outcome == "conversion" else None,
        }
    (RES / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {
                o: {n: round(v["qini_coefficient"], 3) for n, v in s["learners"].items()}
                for o, s in summary.items()
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
