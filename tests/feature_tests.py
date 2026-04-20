"""Feature utility tests — L1 sweep, permutation importance, ablation, VIF.

Each public function accepts (model, X, y) and returns a structured result
object with per-feature scores and pass/fail flags.  All tests are
deterministic given fixed random seeds and have no side effects.

Usage (interactive):
    from tests.feature_tests import run_all_feature_tests
    results = run_all_feature_tests(model, X, y)

Usage (pytest):
    pytest tests/feature_tests.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "FeatureResult",
    "L1SweepResult",
    "PermutationImportanceResult",
    "AblationResult",
    "VIFResult",
    "l1_regularisation_sweep",
    "permutation_importance",
    "ablation_test",
    "vif_test",
    "run_all_feature_tests",
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class FeatureResult:
    feature: str
    score: float  # semantics differ per test — see each function's docstring
    passed: bool


@dataclass
class L1SweepResult:
    features: list[FeatureResult]
    c_values: np.ndarray   # shape (n_c,) — the logspace grid, ascending
    coef_paths: np.ndarray  # shape (n_c, n_features) — coef at each C
    feature_names: list[str]
    median_c: float        # pass/fail threshold: must survive at this C


@dataclass
class PermutationImportanceResult:
    features: list[FeatureResult]
    baseline_score: float
    n_repeats: int
    threshold: float


@dataclass
class AblationResult:
    features: list[FeatureResult]
    baseline_cv_score: float
    cv: int
    threshold: float


@dataclass
class VIFResult:
    features: list[FeatureResult]
    threshold: float


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------


def _validate_inputs(X: pd.DataFrame, y: pd.Series) -> None:
    assert isinstance(X, pd.DataFrame), "X must be a DataFrame"
    assert isinstance(y, pd.Series), "y must be a Series"
    assert len(X) == len(y), f"X and y must have same length, got {len(X)} vs {len(y)}"
    assert not X.isnull().any().any(), "X must not contain NaN"
    assert not y.isnull().any(), "y must not contain NaN"
    assert X.shape[1] > 0, "X must have at least one feature column"
    assert len(X) > 0, "X must have at least one row"


# ---------------------------------------------------------------------------
# Test 1: L1 regularisation sweep
# ---------------------------------------------------------------------------


def l1_regularisation_sweep(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    c_values: np.ndarray | None = None,
    random_state: int = 42,
) -> L1SweepResult:
    """Train L1-penalised logistic regression across a range of C values.

    For each feature, records the smallest C at which it retains a non-zero
    coefficient.  A feature *passes* if it survives regularisation at the
    median C value (medium regularisation strength).

    score = smallest C where the feature is non-zero (np.inf if always zeroed).
    passed = score <= median(c_values).

    ``model`` is accepted for API consistency but not used; fresh L1 pipelines
    are constructed internally so the penalty is guaranteed.
    """
    _validate_inputs(X, y)

    if c_values is None:
        c_values = np.logspace(-3, 3, 20)

    c_values = np.asarray(c_values, dtype=float)
    feature_names = list(X.columns)
    n_features = len(feature_names)
    coef_paths = np.zeros((len(c_values), n_features))

    for i, c in enumerate(c_values):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                l1_ratio=1,   # pure L1 (penalty kwarg deprecated in sklearn 1.8)
                solver="saga",
                C=float(c),
                max_iter=1000,
                random_state=random_state,
            )),
        ])
        pipe.fit(X, y)
        coef_paths[i] = pipe.named_steps["clf"].coef_[0]

    # For each feature find the smallest C at which its coefficient is non-zero.
    # c_values is ascending (low C = strong regularisation).
    median_c = float(np.median(c_values))
    feature_results: list[FeatureResult] = []
    for j, feat in enumerate(feature_names):
        nonzero_mask = coef_paths[:, j] != 0.0
        if nonzero_mask.any():
            first_survive = float(c_values[nonzero_mask][0])
        else:
            first_survive = np.inf
        feature_results.append(FeatureResult(
            feature=feat,
            score=first_survive,
            passed=(first_survive <= median_c),
        ))

    return L1SweepResult(
        features=feature_results,
        c_values=c_values,
        coef_paths=coef_paths,
        feature_names=feature_names,
        median_c=median_c,
    )


# ---------------------------------------------------------------------------
# Test 2: Permutation importance
# ---------------------------------------------------------------------------


def permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_repeats: int = 10,
    threshold: float = 0.001,
    random_state: int = 42,
) -> PermutationImportanceResult:
    """Measure importance by shuffling each feature and recording the log-loss increase.

    Uses the already-fitted ``model`` — no refitting occurs.

    score = mean log-loss increase over n_repeats shuffles (positive = important).
    passed = score > threshold.
    """
    _validate_inputs(X, y)
    assert hasattr(model, "predict_proba"), "model must implement predict_proba"

    rng = np.random.default_rng(random_state)
    y_arr = np.asarray(y)
    baseline_score = log_loss(y_arr, model.predict_proba(X)[:, 1])

    feature_results: list[FeatureResult] = []
    for col in X.columns:
        deltas = np.empty(n_repeats)
        for k in range(n_repeats):
            X_perm = X.copy()
            X_perm[col] = rng.permutation(X_perm[col].to_numpy())
            deltas[k] = log_loss(y_arr, model.predict_proba(X_perm)[:, 1]) - baseline_score
        mean_delta = float(deltas.mean())
        feature_results.append(FeatureResult(
            feature=col,
            score=mean_delta,
            passed=(mean_delta > threshold),
        ))

    return PermutationImportanceResult(
        features=feature_results,
        baseline_score=baseline_score,
        n_repeats=n_repeats,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Test 3: Ablation test
# ---------------------------------------------------------------------------


def ablation_test(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv: int = 5,
    threshold: float = 0.005,
    random_state: int = 42,
) -> AblationResult:
    """Train with each feature removed; record the cross-validated log-loss delta.

    Uses clone(model) so the original fitted model is never mutated.

    score = ablated_cv_loss - baseline_cv_loss (positive = feature was useful).
    passed = score > threshold.

    Complexity: O(n_features × cv) fits.  With 48 features and cv=5 that is
    245 total LogisticRegression fits.
    """
    _validate_inputs(X, y)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    # neg_log_loss returns negative values; negate to get positive log-loss.
    baseline_scores = cross_val_score(
        clone(model), X, y, cv=skf, scoring="neg_log_loss"
    )
    baseline_cv_score = float(-baseline_scores.mean())

    feature_results: list[FeatureResult] = []
    for col in X.columns:
        X_ablated = X.drop(columns=[col])
        scores = cross_val_score(
            clone(model), X_ablated, y, cv=skf, scoring="neg_log_loss"
        )
        ablated_cv_score = float(-scores.mean())
        delta = ablated_cv_score - baseline_cv_score
        feature_results.append(FeatureResult(
            feature=col,
            score=delta,
            passed=(delta > threshold),
        ))

    return AblationResult(
        features=feature_results,
        baseline_cv_score=baseline_cv_score,
        cv=cv,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Test 4: VIF
# ---------------------------------------------------------------------------


def vif_test(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    threshold: float = 5.0,
) -> VIFResult:
    """Compute the Variance Inflation Factor for each feature using pure numpy.

    VIF_i = 1 / (1 - R²_i) where R²_i comes from regressing feature i on all
    other features plus an intercept via least squares.

    ``model`` and ``y`` are accepted for API consistency but not used.

    score = VIF value (np.inf for perfect or zero-variance collinearity).
    passed = VIF < threshold (default 5.0).
    """
    _validate_inputs(X, y)

    X_arr = X.to_numpy(dtype=float)
    n_samples, n_features = X_arr.shape
    feature_names = list(X.columns)
    ones = np.ones((n_samples, 1))

    feature_results: list[FeatureResult] = []
    for i, feat in enumerate(feature_names):
        col_i = X_arr[:, i]

        if np.var(col_i) == 0.0:
            feature_results.append(FeatureResult(feature=feat, score=np.inf, passed=False))
            continue

        other_cols = np.delete(X_arr, i, axis=1)
        A = np.hstack([ones, other_cols])  # shape (n_samples, n_features)

        coeffs, _, _, _ = np.linalg.lstsq(A, col_i, rcond=None)
        fitted = A @ coeffs
        residuals = col_i - fitted

        ss_res = float(np.dot(residuals, residuals))
        ss_tot = float(np.dot(col_i - col_i.mean(), col_i - col_i.mean()))
        r_squared = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))

        if r_squared >= 1.0 - 1e-10:
            vif = np.inf
        else:
            vif = 1.0 / (1.0 - r_squared)

        feature_results.append(FeatureResult(
            feature=feat,
            score=vif,
            passed=(vif < threshold),
        ))

    return VIFResult(features=feature_results, threshold=threshold)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def run_all_feature_tests(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    l1_kwargs: dict[str, Any] | None = None,
    perm_kwargs: dict[str, Any] | None = None,
    ablation_kwargs: dict[str, Any] | None = None,
    vif_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all four feature utility tests and return results in a single dict.

    Per-test keyword arguments can be passed via *_kwargs dicts, e.g.:
        run_all_feature_tests(model, X, y, perm_kwargs={"n_repeats": 20})
    """
    return {
        "l1_sweep": l1_regularisation_sweep(model, X, y, **(l1_kwargs or {})),
        "permutation_importance": permutation_importance(model, X, y, **(perm_kwargs or {})),
        "ablation": ablation_test(model, X, y, **(ablation_kwargs or {})),
        "vif": vif_test(model, X, y, **(vif_kwargs or {})),
    }


# ---------------------------------------------------------------------------
# pytest wrapper
# ---------------------------------------------------------------------------


def _make_test_data(
    n_samples: int = 200,
    n_features: int = 5,
    random_state: int = 0,
) -> tuple:
    """Build a tiny inline DataFrame + fitted model for smoke tests."""
    rng = np.random.default_rng(random_state)
    X = pd.DataFrame(
        rng.standard_normal((n_samples, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series((rng.random(n_samples) > 0.5).astype(int), name="home_win")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X, y)
    return model, X, y


def test_l1_sweep_returns_result():
    model, X, y = _make_test_data()
    result = l1_regularisation_sweep(model, X, y)
    assert isinstance(result, L1SweepResult)
    assert len(result.features) == X.shape[1]
    assert result.coef_paths.shape == (len(result.c_values), X.shape[1])
    assert result.feature_names == list(X.columns)
    for fr in result.features:
        assert isinstance(fr.passed, bool)
        assert fr.score > 0


def test_l1_sweep_pass_threshold_is_median_c():
    model, X, y = _make_test_data()
    result = l1_regularisation_sweep(model, X, y)
    assert result.median_c == float(np.median(result.c_values))
    for fr in result.features:
        assert fr.passed == (fr.score <= result.median_c)


def test_permutation_importance_returns_result():
    model, X, y = _make_test_data()
    result = permutation_importance(model, X, y)
    assert isinstance(result, PermutationImportanceResult)
    assert len(result.features) == X.shape[1]
    assert result.baseline_score > 0
    assert result.n_repeats == 10


def test_permutation_importance_is_reproducible():
    model, X, y = _make_test_data()
    r1 = permutation_importance(model, X, y, random_state=7)
    r2 = permutation_importance(model, X, y, random_state=7)
    for f1, f2 in zip(r1.features, r2.features):
        assert f1.score == f2.score


def test_ablation_returns_result():
    model, X, y = _make_test_data()
    result = ablation_test(model, X, y)
    assert isinstance(result, AblationResult)
    assert len(result.features) == X.shape[1]
    assert result.baseline_cv_score > 0
    assert result.cv == 5


def test_vif_returns_result():
    model, X, y = _make_test_data()
    result = vif_test(model, X, y)
    assert isinstance(result, VIFResult)
    assert len(result.features) == X.shape[1]
    for fr in result.features:
        assert fr.score >= 1.0 or np.isinf(fr.score)


def test_vif_infinite_on_perfectly_collinear():
    model, X, y = _make_test_data()
    X_dup = X.copy()
    X_dup["feat_0_copy"] = X_dup["feat_0"]
    result = vif_test(model, X_dup, y)
    dup_fr = next(fr for fr in result.features if fr.feature == "feat_0_copy")
    assert np.isinf(dup_fr.score)
    assert not dup_fr.passed


def test_vif_flags_zero_variance():
    model, X, y = _make_test_data()
    X_const = X.copy()
    X_const["feat_0"] = 1.0
    result = vif_test(model, X_const, y)
    const_fr = next(fr for fr in result.features if fr.feature == "feat_0")
    assert np.isinf(const_fr.score)
    assert not const_fr.passed


def test_run_all_feature_tests_returns_all_keys():
    model, X, y = _make_test_data()
    results = run_all_feature_tests(model, X, y)
    assert set(results.keys()) == {"l1_sweep", "permutation_importance", "ablation", "vif"}


def test_input_validation_rejects_nan():
    import pytest as _pytest

    model, X, y = _make_test_data()
    X_bad = X.copy()
    X_bad.iloc[0, 0] = np.nan
    with _pytest.raises(AssertionError, match="NaN"):
        permutation_importance(model, X_bad, y)


def test_input_validation_rejects_mismatched_lengths():
    import pytest as _pytest

    model, X, y = _make_test_data()
    with _pytest.raises(AssertionError, match="same length"):
        vif_test(model, X, y.iloc[:-1])


