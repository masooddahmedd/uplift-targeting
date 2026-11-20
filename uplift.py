# Uplift learners and evaluation for a randomized experiment.
#
# Uplift (CATE) tau(x) = P(y=1 | x, treated) - P(y=1 | x, control): how much the incentive changes
# the chance that this user converts. Targeting by tau, not by P(y), is the point: a user who would
# convert anyway gains nothing from a coupon.
#
# Learners (all on LightGBM):
#   response  P(y | x) ignoring treatment. The common mistake, kept as a baseline.
#   S         one model with treatment as a feature; tau = f(x, 1) - f(x, 0)
#   T         separate models for treated and control; tau = mu1(x) - mu0(x)
#   X         Kunzel et al. (2019): impute individual effects with the other arm's model, regress
#             them, combine the two arms with the propensity
#   DR        doubly robust pseudo-outcome (Kennedy, 2020) regressed on x
# X and DR use 2-fold cross-fitting so no row's pseudo-outcome comes from a model that saw it.
#
# Evaluation needs only that treatment was randomized: within any slice of users ranked by a score,
# (treated conversion rate - control conversion rate) is an unbiased estimate of their uplift.
import lightgbm as lgb
import numpy as np

FEATURES = [f"f{i}" for i in range(12)]

CLS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 2000,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_jobs": 16,
    "verbose": -1,
}
REG = {**CLS, "objective": "regression"}


def fit(X, y, Xv, yv, params, weight=None, rounds=2000):
    model = lgb.train(
        params,
        lgb.Dataset(X, y, weight=weight),
        num_boost_round=rounds,
        valid_sets=[lgb.Dataset(Xv, yv)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def predict(model, X):
    return model.predict(X, num_iteration=model.best_iteration)


class Learners:
    """Fits every learner on (X, t, y) with a validation set for early stopping."""

    def __init__(self, X, t, y, Xv, tv, yv, seed=0):
        self.X, self.t, self.y = X, t, y
        self.Xv, self.tv, self.yv = Xv, tv, yv
        self.p = float(t.mean())  # propensity: constant, the experiment randomized
        self.rng = np.random.default_rng(seed)
        self.models = {}

    def fit_response(self):
        self.models["response"] = fit(self.X, self.y, self.Xv, self.yv, CLS)

    def fit_s(self):
        Xs = np.column_stack([self.X, self.t])
        Xvs = np.column_stack([self.Xv, self.tv])
        self.models["s"] = fit(Xs, self.y, Xvs, self.yv, CLS)

    def _arm(self, mask, vmask):
        return fit(self.X[mask], self.y[mask], self.Xv[vmask], self.yv[vmask], CLS)

    def fit_t(self):
        self.models["t1"] = self._arm(self.t == 1, self.tv == 1)
        self.models["t0"] = self._arm(self.t == 0, self.tv == 0)

    def fit_x_and_dr(self):
        # Two folds: arm models fitted on one fold produce pseudo-outcomes on the other.
        fold = self.rng.integers(0, 2, len(self.y))
        d1 = np.zeros(len(self.y))  # X-learner imputed effect, treated rows
        d0 = np.zeros(len(self.y))  # X-learner imputed effect, control rows
        phi = np.zeros(len(self.y))  # DR pseudo-outcome
        for k in (0, 1):
            tr, te = fold != k, fold == k
            m1 = fit(
                self.X[tr & (self.t == 1)],
                self.y[tr & (self.t == 1)],
                self.Xv[self.tv == 1],
                self.yv[self.tv == 1],
                CLS,
            )
            m0 = fit(
                self.X[tr & (self.t == 0)],
                self.y[tr & (self.t == 0)],
                self.Xv[self.tv == 0],
                self.yv[self.tv == 0],
                CLS,
            )
            mu1, mu0 = predict(m1, self.X[te]), predict(m0, self.X[te])
            yt, tt = self.y[te], self.t[te]
            d1[te] = np.where(tt == 1, yt - mu0, 0)
            d0[te] = np.where(tt == 0, mu1 - yt, 0)
            phi[te] = mu1 - mu0 + tt * (yt - mu1) / self.p - (1 - tt) * (yt - mu0) / (1 - self.p)
        # Validation targets for early stopping use the T-learner difference on the validation set.
        tv_hat = predict(self.models["t1"], self.Xv) - predict(self.models["t0"], self.Xv)
        on1, on0 = self.t == 1, self.t == 0
        self.models["x1"] = fit(self.X[on1], d1[on1], self.Xv, tv_hat, REG)
        self.models["x0"] = fit(self.X[on0], d0[on0], self.Xv, tv_hat, REG)
        self.models["dr"] = fit(self.X, phi, self.Xv, tv_hat, REG)

    def scores(self, X):
        """Uplift score per learner (higher = target first)."""
        m = self.models
        out = {}
        if "response" in m:
            out["response"] = predict(m["response"], X)
        if "s" in m:
            out["S"] = predict(m["s"], np.column_stack([X, np.ones(len(X))])) - predict(
                m["s"], np.column_stack([X, np.zeros(len(X))])
            )
        if "t1" in m:
            out["T"] = predict(m["t1"], X) - predict(m["t0"], X)
        if "x1" in m:
            # tau = g(x) tau0(x) + (1 - g(x)) tau1(x) with g = propensity (constant here)
            out["X"] = self.p * predict(m["x0"], X) + (1 - self.p) * predict(m["x1"], X)
        if "dr" in m:
            out["DR"] = predict(m["dr"], X)
        return out


# ---------------------------------------------------------------- evaluation
def rank_order(score, seed=0):
    """Indices from highest to lowest score, ties broken at random with a fixed seed.

    Ties matter: regression learners on rare outcomes can give a third of users the same score, and
    the data file is stored sorted by treatment, so a stable sort would order tied users by file
    position instead of by the model.
    """
    tiebreak = np.random.default_rng(seed).random(len(score))
    return np.lexsort((tiebreak, -np.asarray(score)))


def qini_curve(score, t, y, n_points=101, weight=None):
    """Incremental outcomes when treating the top fraction k of users ranked by score.

    Returns fractions k and Q(k) = Y_T(k) - Y_C(k) * N_T(k) / N_C(k), the estimated number of extra
    conversions among the top-k users, in the units of the treated count. Q(1) is the total uplift.
    """
    order = rank_order(score)
    w = None if weight is None else weight[order]
    return _qini_sorted(t[order], y[order], w, n_points)


def _qini_sorted(t, y, w=None, n_points=101):
    w = np.ones(len(t)) if w is None else w
    nt = np.cumsum(w * t)
    nc = np.cumsum(w * (1 - t))
    yt = np.cumsum(w * t * y)
    yc = np.cumsum(w * (1 - t) * y)
    idx = np.clip((np.linspace(0, 1, n_points) * len(t)).astype(int) - 1, 0, len(t) - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = yt[idx] - np.where(nc[idx] > 0, yc[idx] * nt[idx] / nc[idx], 0)
    q[0] = 0.0
    return np.linspace(0, 1, n_points), q


def uplift_curve_per_user(score, t, y, n_points=101):
    """Extra conversions per user in the whole population when only the top-k fraction is treated.

    Uses treated-vs-control rates within the top-k slice (an unbiased estimate under randomization)
    times the slice size, divided by the population size.
    """
    order = rank_order(score)
    t, y = t[order], y[order]
    n = len(t)
    nt, nc = np.cumsum(t), np.cumsum(1 - t)
    yt, yc = np.cumsum(t * y), np.cumsum((1 - t) * y)
    ks = np.linspace(0, 1, n_points)
    out = np.zeros(n_points)
    for i, k in enumerate(ks[1:], 1):
        j = max(int(k * n) - 1, 0)
        if nt[j] > 0 and nc[j] > 0:
            out[i] = (yt[j] / nt[j] - yc[j] / nc[j]) * (j + 1) / n
    return ks, out


def qini_coefficient(ks, q):
    """Area between the Qini curve and the random-targeting diagonal, as a share of total uplift."""
    total = q[-1]
    random_line = ks * total
    area = np.trapezoid(q - random_line, ks)
    return float(area / abs(total)) if total != 0 else float("nan")


def captured(ks, q, k):
    """Share of all incremental outcomes captured by treating the top fraction k."""
    i = int(np.argmin(np.abs(ks - k)))
    return float(q[i] / q[-1])


def uplift_by_decile(score, t, y, n_bins=10):
    """Observed uplift (treated rate - control rate) within each predicted-uplift decile."""
    order = rank_order(score)
    bins = np.array_split(order, n_bins)
    out = []
    for b in bins:
        tb, yb = t[b], y[b]
        rt = yb[tb == 1].mean() if (tb == 1).any() else np.nan
        rc = yb[tb == 0].mean() if (tb == 0).any() else np.nan
        out.append(
            {
                "pred": float(score[b].mean()),
                "obs": float(rt - rc),
                "treated_rate": float(rt),
                "control_rate": float(rc),
                "n": len(b),
            }
        )
    return out


def bootstrap(score, t, y, stat, n_boot=200, seed=0):
    """Poisson bootstrap of a statistic of the Qini curve; returns (2.5%, 97.5%) percentiles."""
    rng = np.random.default_rng(seed)
    order = rank_order(score)
    ts, ys = t[order].astype(np.float64), y[order].astype(np.float64)
    vals = []
    for _ in range(n_boot):
        w = rng.poisson(1.0, len(t)).astype(np.float64)
        ks, q = _qini_sorted(ts, ys, w)
        vals.append(stat(ks, q))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


KS = (0.1, 0.2, 0.3, 0.5)


def evaluate_all(scores, t, y, n_boot=200):
    """Qini coefficient, share captured at KS and per-decile uplift, with bootstrap intervals."""
    out = {}
    for name, sc in scores.items():
        ks, q = qini_curve(sc, t, y)
        out[name] = {
            "qini_coefficient": qini_coefficient(ks, q),
            "qini_coefficient_ci95": bootstrap(sc, t, y, qini_coefficient, n_boot),
            "captured": {str(k): captured(ks, q, k) for k in KS},
            "captured_ci95": {
                str(k): bootstrap(sc, t, y, lambda a, b, k=k: captured(a, b, k), n_boot) for k in KS
            },
            "deciles": uplift_by_decile(sc, t, y),
            "qini": q.tolist(),
            "tied_share": float(np.unique(sc, return_counts=True)[1].max() / len(sc)),
        }
    return out
