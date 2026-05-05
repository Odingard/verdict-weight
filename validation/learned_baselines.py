"""
Learned baseline fusion methods for the head-to-head comparison.

Implements two reference *learned* multi-source fusion baselines that
require fitting against labeled training data, contrasted with the
closed-form classical baselines in :mod:`validation.baselines`:

  * :class:`LogisticRegressionBaseline` — scikit-learn ``LogisticRegression``
    with deterministic solver and ``random_state=42``.
  * :class:`XGBoostBaseline` — ``xgboost.XGBClassifier`` with deterministic
    ``random_state=42`` and a single thread (``n_jobs=1``) for reproducibility.

Both baselines support two feature-set variants:

  * ``"commercial"`` — the four commercial-tier signals
    ``{SR, CC, TD, HA}`` (the same vector consumed by the closed-form
    fusion baselines). Fair-comparison feature set.
  * ``"eight_stream"`` — the four commercial signals **plus** the four
    integrity-tier stream values ``{S_CTC, S_SIS, S_CPS, S_RIS}`` from
    the unified composer. This answers the stronger question: "what if
    you simply *learn-fused* every VW stream output directly, without
    the unified composition rule and HALT semantics?"

Why include both feature variants?

  The closed-form baselines (DS / NB / SA / MV) only see commercial-tier
  evidence by construction — classical fusion theory has no notion of
  trajectory consistency, source independence, or cryptographic
  provenance. The first variant is therefore a methodologically faithful
  comparison.

  The eight-stream variant is the strongest possible learned baseline:
  it sees *every* signal VW sees, with no architectural disadvantage.
  If VW still dominates this baseline on adversarial samples, the
  delta is attributable to the **composition rule** (HALT propagation,
  tier-aware exponents γ/δ, and the structural relationship between
  S_RIS / S_CPS as HALT-class streams), not to feature availability.

Determinism

  Both baselines use ``random_state=42`` and a fixed solver. Single-
  threaded execution (``n_jobs=1``) is enforced to eliminate parallel-
  reduction non-determinism. Calling ``fit`` twice on the same data
  with the same seed produces bit-identical model parameters.

Optional dependency

  ``scikit-learn`` and ``xgboost`` are declared as the
  ``[learned]`` optional-dependency group in ``pyproject.toml``. They
  are not part of the core dependency set so the closed-form harness
  can run without them. Importing :mod:`validation.learned_baselines`
  without the dependencies installed will raise ``ImportError`` at
  the call site (not at module import) so the rest of the harness
  continues to work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


_COMMERCIAL_FEATURES: Tuple[str, ...] = ("SR", "CC", "TD", "HA")
_EIGHT_STREAM_FEATURES: Tuple[str, ...] = (
    "SR", "CC", "TD", "HA", "S_CTC", "S_SIS", "S_CPS", "S_RIS",
)


def feature_names(variant: str) -> Tuple[str, ...]:
    """Ordered feature names for the requested variant."""
    if variant == "commercial":
        return _COMMERCIAL_FEATURES
    if variant == "eight_stream":
        return _EIGHT_STREAM_FEATURES
    raise ValueError(
        f"Unknown feature variant: {variant!r}. "
        f"Expected 'commercial' or 'eight_stream'."
    )


def feature_dim(variant: str) -> int:
    """Number of features in the requested variant (4 or 8)."""
    return len(feature_names(variant))


@dataclass(frozen=True)
class TrainingReport:
    """Summary of a learned-baseline fit, written into the JSON output.

    Attributes
    ----------
    n_train, n_test:
        Sample counts post train/test split.
    n_features:
        Feature dimensionality (4 for ``commercial``, 8 for
        ``eight_stream``).
    feature_variant:
        ``"commercial"`` or ``"eight_stream"``.
    feature_names_:
        Ordered feature names; useful when reading the JSON later.
    train_seconds:
        Wall-clock fit time.
    """

    n_train: int
    n_test: int
    n_features: int
    feature_variant: str
    feature_names_: Tuple[str, ...]
    train_seconds: float


class _LearnedBaselineBase:
    """Common interface for learned baselines.

    Subclasses must set ``self._model`` in ``fit`` and implement
    :meth:`_predict_proba_internal`. The ``score`` method returns a
    suppression score in ``[0, 1]`` (lower = more suppressed = treated
    as adversarial), matching the convention used by the closed-form
    baselines and ``UnifiedResult.cw_certified``.
    """

    name: str = "LearnedBaselineBase"

    def __init__(self, feature_variant: str = "commercial", random_state: int = 42):
        self.feature_variant = feature_variant
        self.feature_names_: Tuple[str, ...] = feature_names(feature_variant)
        self.random_state = int(random_state)
        self._model = None  # set by fit
        self._fitted: bool = False
        self._training_report: Optional[TrainingReport] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LearnedBaselineBase":
        """Fit the model to features ``X`` and binary labels ``y``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix; columns must follow :attr:`feature_names_`.
        y : np.ndarray of shape (n_samples,)
            Binary labels: 1 = adversarial, 0 = legitimate.
        """
        import time

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {y.shape[0]}."
            )
        if X.shape[1] != len(self.feature_names_):
            raise ValueError(
                f"X has {X.shape[1]} features but variant "
                f"{self.feature_variant!r} expects "
                f"{len(self.feature_names_)} ({list(self.feature_names_)})."
            )

        t0 = time.perf_counter()
        self._fit_internal(X, y)
        elapsed = time.perf_counter() - t0
        self._fitted = True
        self._training_report = TrainingReport(
            n_train=int(X.shape[0]),
            n_test=0,
            n_features=int(X.shape[1]),
            feature_variant=self.feature_variant,
            feature_names_=self.feature_names_,
            train_seconds=float(elapsed),
        )
        return self

    def predict_adversarial_prob(self, X: np.ndarray) -> np.ndarray:
        """Probability that each row of ``X`` is adversarial.

        Returns
        -------
        np.ndarray of shape (n_samples,) in ``[0, 1]``.
        """
        if not self._fitted:
            raise RuntimeError(
                f"{self.name} has not been fitted. Call fit(X, y) first."
            )
        X = np.asarray(X, dtype=float)
        if X.shape[1] != len(self.feature_names_):
            raise ValueError(
                f"X has {X.shape[1]} features but model was trained "
                f"with {len(self.feature_names_)}."
            )
        return self._predict_proba_internal(X)

    def predict_cw(self, X: np.ndarray) -> np.ndarray:
        """Predicted Consequence Weight for each row of ``X``.

        ``CW = 1 - P(adversarial)``. Lower CW indicates higher
        suppression (more confident the sample is adversarial), matching
        the convention in :mod:`verdict_weight.unified` and
        :mod:`validation.baselines`.
        """
        return 1.0 - self.predict_adversarial_prob(X)

    @property
    def training_report(self) -> Optional[TrainingReport]:
        return self._training_report

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _fit_internal(self, X: np.ndarray, y: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError

    def _predict_proba_internal(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class LogisticRegressionBaseline(_LearnedBaselineBase):
    """L2-regularised logistic regression on stream features.

    The reference learned baseline reviewers most commonly request:
    a linear model with L2 regularisation, fit by lbfgs to the
    canonical limit. Deterministic given ``random_state=42``.

    Parameters
    ----------
    feature_variant:
        ``"commercial"`` (4-dim) or ``"eight_stream"`` (8-dim).
    random_state:
        Solver seed; default 42.
    C:
        Inverse regularisation strength; default 1.0 (sklearn default).
    max_iter:
        Solver iteration cap; default 1000 (large enough that lbfgs
        converges on every dataset in the harness — verified by tests).
    """

    name = "LOGISTIC_REGRESSION"

    def __init__(
        self,
        feature_variant: str = "commercial",
        random_state: int = 42,
        C: float = 1.0,
        max_iter: int = 1000,
    ):
        super().__init__(feature_variant=feature_variant, random_state=random_state)
        self.C = float(C)
        self.max_iter = int(max_iter)

    def _fit_internal(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression

        # ``n_jobs`` is intentionally omitted: lbfgs is single-threaded and
        # passing it explicitly emits a FutureWarning under scikit-learn 1.8+.
        self._model = LogisticRegression(
            solver="lbfgs",
            max_iter=self.max_iter,
            C=self.C,
            random_state=self.random_state,
        )
        self._model.fit(X, y)

    def _predict_proba_internal(self, X: np.ndarray) -> np.ndarray:
        proba = self._model.predict_proba(X)
        # Column 1 is P(class=1) = P(adversarial)
        return np.asarray(proba[:, 1], dtype=float)

    @property
    def coef_(self) -> Optional[np.ndarray]:
        if self._model is None:
            return None
        return np.asarray(self._model.coef_, dtype=float)

    @property
    def intercept_(self) -> Optional[np.ndarray]:
        if self._model is None:
            return None
        return np.asarray(self._model.intercept_, dtype=float)


class XGBoostBaseline(_LearnedBaselineBase):
    """Gradient-boosted decision-tree classifier (xgboost.XGBClassifier).

    The strongest learned tabular baseline reviewers commonly request.
    Configured for full reproducibility:

      * ``random_state=42`` seeds tree-building.
      * ``n_jobs=1`` disables parallel reduction (which is otherwise
        a source of non-determinism).
      * ``tree_method="hist"`` for stable, deterministic histogramming.

    Parameters
    ----------
    feature_variant:
        ``"commercial"`` (4-dim) or ``"eight_stream"`` (8-dim).
    random_state:
        Tree-building seed; default 42.
    n_estimators:
        Boost rounds; default 200 (validated by tests as past the
        convergence elbow on the synthetic harness).
    max_depth:
        Tree depth cap; default 6 (xgboost default).
    learning_rate:
        Step size; default 0.1 (xgboost default).
    """

    name = "XGBOOST"

    def __init__(
        self,
        feature_variant: str = "commercial",
        random_state: int = 42,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
    ):
        super().__init__(feature_variant=feature_variant, random_state=random_state)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)

    def _fit_internal(self, X: np.ndarray, y: np.ndarray) -> None:
        import xgboost as xgb

        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=1,
            tree_method="hist",
            objective="binary:logistic",
            eval_metric="logloss",
            verbosity=0,
        )
        self._model.fit(X, y)

    def _predict_proba_internal(self, X: np.ndarray) -> np.ndarray:
        proba = self._model.predict_proba(X)
        return np.asarray(proba[:, 1], dtype=float)

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        if self._model is None:
            return None
        return np.asarray(self._model.feature_importances_, dtype=float)


# ---------------------------------------------------------------------
# Feature-extraction helpers
# ---------------------------------------------------------------------


def commercial_features(samples: Sequence) -> np.ndarray:
    """Build the 4-dim commercial feature matrix from ``Sample`` records.

    Returns an ``(N, 4)`` array of ``[SR, CC, TD, HA]`` columns.
    """
    return np.asarray(
        [[s.SR, s.CC, s.TD, s.HA] for s in samples],
        dtype=float,
    )


def eight_stream_features(
    samples: Sequence,
    vw_results: Sequence[dict],
) -> np.ndarray:
    """Build the 8-dim eight-stream feature matrix.

    Combines commercial-tier signals from each ``Sample`` with the
    integrity-tier stream values produced by ``vw_score_one`` for the
    same sample. ``vw_results`` must be aligned with ``samples``.

    Halted samples (``r["halted"] == True``) carry zero values for
    the integrity stream that triggered the halt — this is the same
    convention used by ``UnifiedResult.cw_certified == None`` being
    coerced to ``0.0`` elsewhere in the harness.
    """
    if len(samples) != len(vw_results):
        raise ValueError(
            f"samples ({len(samples)}) and vw_results ({len(vw_results)}) "
            f"must be aligned 1:1."
        )

    rows = []
    for s, r in zip(samples, vw_results):
        ctc = float(r.get("S_CTC") or 0.0)
        sis = float(r.get("S_SIS") or 0.0)
        cps = float(r.get("S_CPS") or 0.0)
        ris = float(r.get("S_RIS") or 0.0)
        rows.append([s.SR, s.CC, s.TD, s.HA, ctc, sis, cps, ris])
    return np.asarray(rows, dtype=float)


def stratified_split(
    n: int,
    labels: np.ndarray,
    attack_classes: Sequence[Optional[str]],
    test_fraction: float = 0.30,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified train/test split by ``(label, attack_class)`` strata.

    Each stratum is shuffled with the same seeded RNG and split at
    ``test_fraction``. Strata with fewer than 2 samples are forced into
    the training set (no test sample is created from them) — this is
    deterministic and reproducible.

    Returns
    -------
    (train_idx, test_idx) : Tuple[np.ndarray, np.ndarray]
        Sorted index arrays into the original sample list.
    """
    if len(labels) != n or len(attack_classes) != n:
        raise ValueError(
            f"labels ({len(labels)}) and attack_classes "
            f"({len(attack_classes)}) must both equal n ({n})."
        )
    rng = np.random.default_rng(seed)

    # Group sample indices by (label, attack_class) stratum
    strata: dict = {}
    for i in range(n):
        key = (int(labels[i]), attack_classes[i] or "legitimate")
        strata.setdefault(key, []).append(i)

    train_idx_list = []
    test_idx_list = []
    for key in sorted(strata.keys(), key=lambda k: (k[0], str(k[1]))):
        idxs = np.array(strata[key], dtype=int)
        rng.shuffle(idxs)
        n_test = int(round(test_fraction * len(idxs)))
        if n_test < 1 and len(idxs) >= 2:
            n_test = 1
        if len(idxs) < 2:
            train_idx_list.extend(idxs.tolist())
            continue
        test_idx_list.extend(idxs[:n_test].tolist())
        train_idx_list.extend(idxs[n_test:].tolist())

    train_idx = np.array(sorted(train_idx_list), dtype=int)
    test_idx = np.array(sorted(test_idx_list), dtype=int)
    return train_idx, test_idx


__all__ = [
    "LogisticRegressionBaseline",
    "XGBoostBaseline",
    "TrainingReport",
    "feature_names",
    "feature_dim",
    "commercial_features",
    "eight_stream_features",
    "stratified_split",
]
