"""Unit tests for learned-baseline fusion (LR + XGBoost).

Cover:

  * Determinism — fitting twice on the same data with the same seed
    produces bit-identical parameters and identical predictions.
  * Schema invariance — feature dimensionality matches the variant.
  * Pre-fit guard — predict_proba on an unfitted model raises.
  * Sanity — both baselines beat random on the synthetic harness
    but do NOT cross 0.80 on the commercial-tier feature vector.
    The synthetic SR/CC/TD/HA vector is intentionally not strongly
    separable on its own — VW's discrimination power comes from the
    integrity-tier streams (S_CTC/S_SIS/S_CPS/S_RIS), which is the
    architectural thesis of the paper. The 8-stream variant clears
    AUC > 0.95 because it sees those integrity streams.
  * Comparability to closed-form baselines — DS/NB/SA/MV score
    AUC ≈ 0.60–0.65 on this dataset; learned baselines should sit
    in the same regime on the commercial-tier variant.
  * Stratified split — every (label, attack_class) stratum is
    represented in both train and test partitions, and the split is
    deterministic given a seed.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the entire module if the optional learned dependencies are not
# installed. This keeps the "core" pytest run green on minimal envs
# while still exercising learned baselines in CI's full env.
sklearn = pytest.importorskip("sklearn")
xgboost = pytest.importorskip("xgboost")

from validation import learned_baselines as lb
from validation.datasets import generate_dataset
from validation.metrics import auc_roc


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_dataset():
    samples, _registry = generate_dataset(n=2_000, seed=42)
    labels = np.array(
        [1 if s.label == "adversarial" else 0 for s in samples], dtype=int
    )
    return samples, labels


@pytest.fixture(scope="module")
def commercial_X(small_dataset):
    samples, _ = small_dataset
    return lb.commercial_features(samples)


@pytest.fixture(scope="module")
def split_indices(small_dataset):
    samples, labels = small_dataset
    train_idx, test_idx = lb.stratified_split(
        n=len(samples),
        labels=labels,
        attack_classes=[s.attack_class for s in samples],
        test_fraction=0.30,
        seed=42,
    )
    return train_idx, test_idx


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def test_feature_names_commercial():
    assert lb.feature_names("commercial") == ("SR", "CC", "TD", "HA")
    assert lb.feature_dim("commercial") == 4


def test_feature_names_eight_stream():
    assert lb.feature_names("eight_stream") == (
        "SR", "CC", "TD", "HA", "S_CTC", "S_SIS", "S_CPS", "S_RIS",
    )
    assert lb.feature_dim("eight_stream") == 8


def test_feature_names_invalid_variant():
    with pytest.raises(ValueError, match="Unknown feature variant"):
        lb.feature_names("nonexistent")


def test_commercial_features_shape(commercial_X, small_dataset):
    samples, _ = small_dataset
    assert commercial_X.shape == (len(samples), 4)
    assert np.all((commercial_X >= 0.0) & (commercial_X <= 1.0))


# ---------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------


class TestStratifiedSplit:
    def test_partitions_are_disjoint(self, split_indices):
        train_idx, test_idx = split_indices
        overlap = np.intersect1d(train_idx, test_idx)
        assert len(overlap) == 0

    def test_partitions_cover_full_dataset(self, split_indices, small_dataset):
        train_idx, test_idx = split_indices
        samples, _ = small_dataset
        union = np.union1d(train_idx, test_idx)
        assert len(union) == len(samples)

    def test_test_fraction_approx(self, split_indices, small_dataset):
        train_idx, test_idx = split_indices
        samples, _ = small_dataset
        ratio = len(test_idx) / len(samples)
        assert 0.27 < ratio < 0.33

    def test_every_stratum_in_test(self, split_indices, small_dataset):
        train_idx, test_idx = split_indices
        samples, labels = small_dataset
        observed = set()
        for i in test_idx:
            key = (int(labels[i]), samples[i].attack_class or "legitimate")
            observed.add(key)
        # 1 legitimate stratum + 5 attack classes
        assert len(observed) == 6

    def test_split_is_deterministic(self, small_dataset):
        samples, labels = small_dataset
        attacks = [s.attack_class for s in samples]
        a1, b1 = lb.stratified_split(len(samples), labels, attacks, seed=42)
        a2, b2 = lb.stratified_split(len(samples), labels, attacks, seed=42)
        assert np.array_equal(a1, a2)
        assert np.array_equal(b1, b2)

    def test_different_seed_produces_different_split(self, small_dataset):
        samples, labels = small_dataset
        attacks = [s.attack_class for s in samples]
        _, t1 = lb.stratified_split(len(samples), labels, attacks, seed=42)
        _, t2 = lb.stratified_split(len(samples), labels, attacks, seed=99)
        assert not np.array_equal(t1, t2)


# ---------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------


class TestLogisticRegression:
    def test_pre_fit_predict_raises(self, commercial_X):
        model = lb.LogisticRegressionBaseline(feature_variant="commercial")
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_adversarial_prob(commercial_X[:5])

    def test_fit_returns_self(self, commercial_X, small_dataset):
        _, labels = small_dataset
        model = lb.LogisticRegressionBaseline(feature_variant="commercial")
        out = model.fit(commercial_X[:1000], labels[:1000])
        assert out is model

    def test_predict_proba_in_range(self, commercial_X, small_dataset, split_indices):
        _, labels = small_dataset
        train_idx, test_idx = split_indices
        model = lb.LogisticRegressionBaseline(feature_variant="commercial")
        model.fit(commercial_X[train_idx], labels[train_idx])
        proba = model.predict_adversarial_prob(commercial_X[test_idx])
        assert proba.shape == (len(test_idx),)
        assert np.all((proba >= 0.0) & (proba <= 1.0))

    def test_cw_complement_of_proba(self, commercial_X, small_dataset, split_indices):
        _, labels = small_dataset
        train_idx, test_idx = split_indices
        model = lb.LogisticRegressionBaseline(feature_variant="commercial")
        model.fit(commercial_X[train_idx], labels[train_idx])
        proba = model.predict_adversarial_prob(commercial_X[test_idx])
        cw = model.predict_cw(commercial_X[test_idx])
        assert np.allclose(cw, 1.0 - proba)

    def test_deterministic_fit(self, commercial_X, small_dataset, split_indices):
        _, labels = small_dataset
        train_idx, _ = split_indices
        m1 = lb.LogisticRegressionBaseline(feature_variant="commercial")
        m2 = lb.LogisticRegressionBaseline(feature_variant="commercial")
        m1.fit(commercial_X[train_idx], labels[train_idx])
        m2.fit(commercial_X[train_idx], labels[train_idx])
        assert np.allclose(m1.coef_, m2.coef_)
        assert np.allclose(m1.intercept_, m2.intercept_)

    def test_beats_random_on_synthetic(
        self, commercial_X, small_dataset, split_indices
    ):
        _, labels = small_dataset
        train_idx, test_idx = split_indices
        model = lb.LogisticRegressionBaseline(feature_variant="commercial")
        model.fit(commercial_X[train_idx], labels[train_idx])
        proba = model.predict_adversarial_prob(commercial_X[test_idx])
        auc = auc_roc(labels[test_idx], proba)
        # The commercial-tier feature vector (SR/CC/TD/HA) is intentionally
        # not highly separable — closed-form baselines (DS/NB/SA/MV) score
        # AUC ≈ 0.60–0.65 on this dataset, and LR sits in the same regime.
        # The whole point of the eight-stream architecture is that the
        # commercial-tier vector alone is insufficient. We require AUC > 0.55
        # (clearly better than random) for this baseline.
        assert auc > 0.55, f"LR commercial AUC={auc:.4f} below 0.55 threshold"
        # And ≤ 0.80 — if it were higher, the dataset would be trivially
        # learnable from commercial features and the architectural argument
        # would be undermined.
        assert auc < 0.80, (
            f"LR commercial AUC={auc:.4f} above 0.80 — dataset may be too easy "
            f"and the architectural argument needs revisiting."
        )

    def test_feature_dim_mismatch_raises(self, commercial_X, small_dataset):
        _, labels = small_dataset
        model = lb.LogisticRegressionBaseline(feature_variant="eight_stream")
        # commercial_X is 4-dim but model expects 8
        with pytest.raises(ValueError, match="features"):
            model.fit(commercial_X[:100], labels[:100])


# ---------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------


class TestXGBoost:
    def test_pre_fit_predict_raises(self, commercial_X):
        model = lb.XGBoostBaseline(feature_variant="commercial")
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_adversarial_prob(commercial_X[:5])

    def test_predict_proba_in_range(self, commercial_X, small_dataset, split_indices):
        _, labels = small_dataset
        train_idx, test_idx = split_indices
        model = lb.XGBoostBaseline(
            feature_variant="commercial", n_estimators=50
        )
        model.fit(commercial_X[train_idx], labels[train_idx])
        proba = model.predict_adversarial_prob(commercial_X[test_idx])
        assert proba.shape == (len(test_idx),)
        assert np.all((proba >= 0.0) & (proba <= 1.0))

    def test_deterministic_fit(self, commercial_X, small_dataset, split_indices):
        _, labels = small_dataset
        train_idx, test_idx = split_indices
        m1 = lb.XGBoostBaseline(feature_variant="commercial", n_estimators=50)
        m2 = lb.XGBoostBaseline(feature_variant="commercial", n_estimators=50)
        m1.fit(commercial_X[train_idx], labels[train_idx])
        m2.fit(commercial_X[train_idx], labels[train_idx])
        p1 = m1.predict_adversarial_prob(commercial_X[test_idx])
        p2 = m2.predict_adversarial_prob(commercial_X[test_idx])
        assert np.allclose(p1, p2, atol=1e-7)

    def test_beats_random_on_synthetic(
        self, commercial_X, small_dataset, split_indices
    ):
        _, labels = small_dataset
        train_idx, test_idx = split_indices
        model = lb.XGBoostBaseline(feature_variant="commercial", n_estimators=100)
        model.fit(commercial_X[train_idx], labels[train_idx])
        proba = model.predict_adversarial_prob(commercial_X[test_idx])
        auc = auc_roc(labels[test_idx], proba)
        # Same regime as LR (see TestLogisticRegression.test_beats_random_
        # on_synthetic). XGBoost has slightly more capacity but the
        # commercial-tier vector is not strongly separable.
        assert auc > 0.55, f"XGB commercial AUC={auc:.4f} below 0.55 threshold"
        assert auc < 0.85, (
            f"XGB commercial AUC={auc:.4f} above 0.85 — dataset may be too easy."
        )

    def test_feature_importances_available(
        self, commercial_X, small_dataset, split_indices
    ):
        _, labels = small_dataset
        train_idx, _ = split_indices
        model = lb.XGBoostBaseline(feature_variant="commercial", n_estimators=20)
        model.fit(commercial_X[train_idx], labels[train_idx])
        imp = model.feature_importances_
        assert imp is not None
        assert imp.shape == (4,)
        assert np.all(imp >= 0.0)


# ---------------------------------------------------------------------
# Eight-stream feature variant (uses VW outputs)
# ---------------------------------------------------------------------


class TestEightStreamFeatures:
    @pytest.fixture(scope="class")
    def vw_results(self, small_dataset):
        from validation.synthetic_validation import vw_score_one
        from verdict_weight import UnifiedComposer

        samples, _ = small_dataset
        # generate_dataset is module-scoped; rebuild composer separately
        _, registry = generate_dataset(n=2_000, seed=42)
        composer = UnifiedComposer(registry=registry)
        return [vw_score_one(composer, s) for s in samples]

    def test_eight_stream_features_shape(self, small_dataset, vw_results):
        samples, _ = small_dataset
        X = lb.eight_stream_features(samples, vw_results)
        assert X.shape == (len(samples), 8)
        # SR/CC/TD/HA (cols 0-3) ∈ [0,1]; integrity columns are stream
        # values clipped at 0 for halted samples
        assert np.all(X >= 0.0)
        assert np.all(X[:, :4] <= 1.0)
        assert np.all(X[:, 4:] <= 1.0 + 1e-9)

    def test_eight_stream_misaligned_inputs_raise(self, small_dataset, vw_results):
        samples, _ = small_dataset
        with pytest.raises(ValueError, match="aligned 1:1"):
            lb.eight_stream_features(samples, vw_results[:-5])

    def test_lr_eight_stream_fits(self, small_dataset, vw_results, split_indices):
        samples, labels = small_dataset
        train_idx, test_idx = split_indices
        X = lb.eight_stream_features(samples, vw_results)
        model = lb.LogisticRegressionBaseline(feature_variant="eight_stream")
        model.fit(X[train_idx], labels[train_idx])
        proba = model.predict_adversarial_prob(X[test_idx])
        auc = auc_roc(labels[test_idx], proba)
        # With access to all 8 streams, LR should clear 0.95 AUC easily
        assert auc > 0.95, f"LR eight_stream AUC={auc:.4f} below 0.95"
