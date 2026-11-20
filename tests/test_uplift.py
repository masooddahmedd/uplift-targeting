# The metrics must be right before any model number means anything, so they are checked on a
# synthetic randomized experiment where the true uplift is known.
import numpy as np

from uplift import Learners, captured, qini_coefficient, qini_curve, uplift_by_decile


def synthetic(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12)).astype(np.float32)
    t = (rng.random(n) < 0.85).astype(np.int8)
    base = 0.05 + 0.03 * (X[:, 1] > 0)  # who converts anyway
    tau = np.where(X[:, 0] > 1.0, 0.10, 0.0)  # only ~16% of users respond to the incentive
    y = (rng.random(n) < base + t * tau).astype(np.int8)
    return X, t, y, tau


def test_true_uplift_ranks_capture_everything_early():
    _X, t, y, tau = synthetic()
    ks, q = qini_curve(tau + 1e-9 * np.random.default_rng(1).random(len(tau)), t, y)
    share = float((tau > 0).mean())
    assert captured(ks, q, round(share + 0.02, 2)) > 0.9


def test_random_ranking_has_near_zero_qini_coefficient():
    _X, t, y, _tau = synthetic()
    ks, q = qini_curve(np.random.default_rng(2).random(len(t)), t, y)
    assert abs(qini_coefficient(ks, q)) < 0.05


def test_response_model_signal_is_not_uplift():
    # Ranking by who converts anyway (X[:, 1]) should do no better than random at finding uplift.
    X, t, y, tau = synthetic()
    ks, q = qini_curve(X[:, 1], t, y)
    ks2, q2 = qini_curve(tau, t, y)
    assert qini_coefficient(ks, q) < 0.05 < qini_coefficient(ks2, q2)


def test_deciles_recover_the_true_effect():
    _X, t, y, tau = synthetic()
    d = uplift_by_decile(tau + 1e-9 * np.random.default_rng(3).random(len(tau)), t, y)
    assert abs(d[0]["obs"] - 0.10) < 0.02  # top decile is all responders
    assert abs(d[-1]["obs"]) < 0.02  # bottom decile has none


def test_t_learner_finds_the_responders():
    X, t, y, _tau = synthetic(n=120_000)
    Xv, tv, yv, _ = synthetic(n=40_000, seed=5)
    L = Learners(X, t, y, Xv, tv, yv)
    L.fit_t()
    score = L.scores(Xv)["T"]
    true_v = np.where(Xv[:, 0] > 1.0, 0.10, 0.0)
    assert np.corrcoef(score, true_v)[0, 1] > 0.5


def test_tied_scores_do_not_follow_file_order():
    # A constant score must behave like random targeting even when the rows are stored sorted by
    # treatment (as the Criteo file is). A stable sort would put every treated row first.
    _X, t, y, _tau = synthetic()
    order = np.argsort(-t, kind="stable")
    t, y = t[order], y[order]
    ks, q = qini_curve(np.zeros(len(t)), t, y)
    assert abs(qini_coefficient(ks, q)) < 0.05
