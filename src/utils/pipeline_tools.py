"""Utility functions for building and running machine learning pipelines."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import torch
from scipy.stats import randint, uniform
from scipy.stats.distributions import loguniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, make_scorer, precision_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    QuantileTransformer,
    StandardScaler,
    TargetEncoder,
)
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

if TYPE_CHECKING:
    from pathlib import Path

    from pandas import DataFrame as pdDataFrame
    from sklearn.model_selection import TimeSeriesSplit


# Filename used both when persisting CV results and when reading them back to
# refit on the full training set (see fit_full_model).
_RESULTS_FILENAME = "pipeline_results.json"


def _save_pipeline_results(
    best_score: float,
    best_params: dict[str, Any],
    dirpath: Path,
) -> None:
    """Save the best score and parameters to a JSON file.

    Args:
        best_score: The best score from GridSearchCV.
        best_params: The best parameters from GridSearchCV.
        dirpath: Directory to save the results.

    """
    dirpath.mkdir(parents=True, exist_ok=True)
    results = {
        "best_score": best_score,
        "best_params": best_params,
    }
    file_path = dirpath / _RESULTS_FILENAME
    with file_path.open("w") as f:
        json.dump(results, f, indent=4)


def _save_best_model(estimator: Pipeline, dirpath: Path) -> None:
    """Persist the fitted best pipeline (preprocessing + model) to disk.

    Saves the entire fitted Pipeline, so inference reuses the exact fitted
    preprocessing (TargetEncoder mappings, QuantileTransformer quantiles,
    imputer statistics) together with the trained classifier. Reload with
    ``joblib.load(path)`` (with ``src`` importable so the custom
    ``MCDropoutClassifier`` / ``_MCDropoutNet`` classes resolve). The PyTorch
    weights are pickled in place; ``predict_proba`` moves the module onto the
    resolved device on load, so the artifact reloads on CPU or GPU.

    Args:
        estimator: The pipeline fitted on the full training set (the model
            returned by ``fit_full_model``).
        dirpath: Directory to save the model into.

    """
    dirpath.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, dirpath / "best_model.joblib")


def _save_grid_search_params(grid_search: Any, dirpath: Path) -> None:
    """Save grid search parameters to a JSON file.

    Args:
        grid_search: The scikit-learn search object (e.g., RandomizedSearchCV).
        dirpath: Directory to save the parameters.

    """
    dirpath.mkdir(parents=True, exist_ok=True)
    params = grid_search.get_params()

    serializable_params = {}
    for key, value in params.items():
        try:
            json.dumps(value)
            serializable_params[key] = value
        except TypeError, OverflowError:
            serializable_params[key] = str(value)

    file_path = dirpath / "grid_search_params.json"
    with file_path.open("w") as f:
        json.dump(serializable_params, f, indent=4)


def save_pipeline_params(pipeline: Pipeline, dirpath: Path) -> None:
    """Save pipeline parameters to a JSON file.

    Args:
        pipeline: The scikit-learn Pipeline object.
        dirpath: Directory to save the parameters.

    """
    dirpath.mkdir(parents=True, exist_ok=True)
    params = pipeline.get_params()

    serializable_params = {}
    for key, value in params.items():
        try:
            json.dumps(value)
            serializable_params[key] = value
        except TypeError, OverflowError:
            serializable_params[key] = str(value)

    file_path = dirpath / "pipeline_params.json"

    with file_path.open("w") as f:
        json.dump(serializable_params, f, indent=4)


def xgboost_reference(
    categorical_features: list[str] | None = None,
) -> tuple[Pipeline, dict[str, Any]]:
    """Create a pipeline that encodes categorical features and passes through NaNs.

    Args:
        categorical_features: List of categorical feature names.

    Returns:
        A tuple containing the Pipeline and the parameter grid.

    """
    if categorical_features is None:
        categorical_features = []

    # necessary for non-numerical columns like strings
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                categorical_features,
            ),
        ],
        remainder="passthrough",
    )

    # CPU hist: the sklearn Pipeline feeds host (CPU) numpy, so device="cuda"
    # forced a CPU->GPU copy / DMatrix fallback every call. On this tabular size
    # CPU hist avoids that overhead and the device-mismatch warning.
    clf = XGBClassifier(
        objective="binary:logistic", tree_method="hist", device="cpu", n_jobs=2
    )
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ],
    )

    param_distributions = {
        "classifier__n_estimators": randint(100, 1000),
        "classifier__learning_rate": uniform(0.01, 0.3),
        "classifier__max_depth": randint(3, 12),
        "classifier__gamma": uniform(0, 10),
        "classifier__subsample": uniform(0.6, 0.4),  # Range [0.6, 1.0]
        "classifier__colsample_bytree": uniform(0.6, 0.4),
        "classifier__reg_alpha": [0, 0.001, 0.01, 0.1, 1],
        "classifier__reg_lambda": [0, 0.001, 0.01, 0.1, 1],
    }
    return pipe, param_distributions


def mlp_reference(
    categorical_features: list[str] | None = None,
    numeric_features: list[str] | None = None,
) -> tuple[Pipeline, dict[str, Any]]:
    """Create a pipeline that imputes NaNs and encodes categorical features.

    Args:
        categorical_features: List of categorical feature names.
        numeric_features: List of numeric feature names.

    Returns:
        A tuple containing the Pipeline and the parameter grid.

    """
    if categorical_features is None:
        categorical_features = []
    if numeric_features is None:
        numeric_features = []

    # Cross-fitted TargetEncoder (CV-safe) replaces OrdinalEncoder: it maps
    # categories to a meaningful, bounded scale (smoothed P(fraud|category))
    # instead of arbitrary integer codes, so high-cardinality columns no longer
    # enter the net unscaled and fake-ordered. StandardScaler then puts the
    # encoded values on the same ~unit-variance footing as the numerics.
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", TargetEncoder(target_type="binary", random_state=42)),
            ("scaler", StandardScaler()),
        ],
    )

    # QuantileTransformer is rank-based, so it handles both the heavy tails and
    # the zero-IQR sparsity that broke RobustScaler (see preprocessing_findings).
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "scaler",
                QuantileTransformer(output_distribution="normal", random_state=42),
            ),
        ],
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, categorical_features),
            ("num", num_transformer, numeric_features),
        ],
        remainder="drop",
    )

    clf = MLPClassifier(random_state=42, max_iter=500)
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ],
    )

    param_distributions = {
        "classifier__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
        "classifier__activation": ["tanh", "relu"],
        "classifier__solver": ["sgd", "adam"],
        "classifier__alpha": uniform(0.0001, 0.05),
        "classifier__learning_rate": ["constant", "adaptive"],
    }
    return pipe, param_distributions


class _MCDropoutNet(nn.Module):
    """Feed-forward network with dropout after every hidden layer.

    Dropout is kept active at inference time to enable Monte Carlo dropout,
    which approximates Bayesian posterior sampling over the network weights.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_layer_sizes: tuple[int, ...],
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        act_layer = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]

        layers: list[nn.Module] = []
        in_features = n_features
        for hidden in hidden_layer_sizes:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(act_layer())
            layers.append(nn.Dropout(p=dropout))
            in_features = hidden
        layers.append(nn.Linear(in_features, n_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw logits for a batch of inputs."""
        return self.network(x)


class MCDropoutClassifier(ClassifierMixin, BaseEstimator):
    """Bayesian deep learning classifier using Monte Carlo dropout.

    A scikit-learn compatible wrapper around a PyTorch network. Uncertainty is
    estimated by leaving dropout active during inference and averaging the
    softmax outputs over several stochastic forward passes.
    """

    def __init__(  # noqa: PLR0913
        self,
        hidden_layer_sizes: tuple[int, ...] = (100,),
        dropout: float = 0.2,
        activation: str = "relu",
        lr: float = 1e-3,
        alpha: float = 1e-4,
        max_iter: int = 100,
        batch_size: int = 4096,
        mc_samples: int = 30,
        class_weight: str | None = "balanced",
        prior_correction: bool = True,
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.activation = activation
        self.lr = lr
        self.alpha = alpha
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.mc_samples = mc_samples
        self.class_weight = class_weight
        self.prior_correction = prior_correction
        self.random_state = random_state
        self.device = device

    def _resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, x: Any, y: Any) -> MCDropoutClassifier:  # noqa: ANN401
        """Train the network on the given features and labels."""
        x, y = check_X_y(x, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = x.shape[1]

        torch.manual_seed(self.random_state)
        device = self._resolve_device()

        class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_indices = np.array([class_to_index[label] for label in y])

        self.module_ = _MCDropoutNet(
            n_features=self.n_features_in_,
            n_classes=len(self.classes_),
            hidden_layer_sizes=tuple(self.hidden_layer_sizes),
            dropout=self.dropout,
            activation=self.activation,
        ).to(device)

        x_tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        y_tensor = torch.as_tensor(y_indices, dtype=torch.long)
        loader = DataLoader(
            TensorDataset(x_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(
            self.module_.parameters(),
            lr=self.lr,
            weight_decay=self.alpha,
        )
        weight_tensor = None
        # Log of the true class priors, used to undo the balanced-weighting
        # prior shift at inference time (see predict_proba). None when no
        # weighting was applied, since there is no shift to correct.
        self.log_prior_ = None
        if self.class_weight == "balanced":
            counts = np.bincount(y_indices, minlength=len(self.classes_))
            weights = len(y_indices) / (len(self.classes_) * counts)
            weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
            self.log_prior_ = np.log(counts / len(y_indices))
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        self.module_.train()
        for _ in range(self.max_iter):
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(self.module_(x_batch), y_batch)
                loss.backward()
                optimizer.step()

        return self

    def _enable_mc_dropout(self) -> None:
        """Set the network to eval mode but keep dropout layers active."""
        self.module_.eval()
        for module in self.module_.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def _logit_offset(self, device: torch.device) -> torch.Tensor | None:
        """Return the prior-correction offset to add to logits, or None.

        The balanced weighting trains under a uniform prior, so adding back the
        log of the true class priors in logit space recovers calibrated
        probabilities. None when no correction applies.
        """
        if self.prior_correction and self.log_prior_ is not None:
            return torch.as_tensor(self.log_prior_, dtype=torch.float32, device=device)
        return None

    def predict_proba(self, x: Any) -> np.ndarray:  # noqa: ANN401
        """Return class probabilities averaged over MC dropout samples."""
        check_is_fitted(self)
        x = check_array(x)
        device = self._resolve_device()
        # Keep the (possibly just-unpickled) module and inputs on one device.
        self.module_.to(device)
        x_tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32).to(device)
        log_prior = self._logit_offset(device)

        self._enable_mc_dropout()
        probas = torch.zeros(
            (x_tensor.shape[0], len(self.classes_)),
            device=device,
        )
        with torch.no_grad():
            for _ in range(self.mc_samples):
                logits = self.module_(x_tensor)
                if log_prior is not None:
                    logits = logits + log_prior
                probas += torch.softmax(logits, dim=1)
        probas /= self.mc_samples

        return probas.cpu().numpy()

    def predict(self, x: Any) -> np.ndarray:  # noqa: ANN401
        """Return the most likely class for each sample."""
        proba = self.predict_proba(x)
        return self.classes_[np.argmax(proba, axis=1)]

    def uncertainty_metrics(self, x: Any) -> dict[str, np.ndarray]:  # noqa: ANN401
        """Return a menu of uncertainty metrics from a single MC-dropout run.

        All metrics are derived from the same N stochastic forward passes, so
        they are mutually consistent, and all are computed on the
        prior-corrected probabilities (when ``prior_correction`` is enabled).
        ``bald`` and ``epistemic_var`` measure disagreement across passes and
        are far less sensitive to the correction than the level-based metrics,
        but are not strictly invariant to it (the offset interacts with the
        softmax/entropy nonlinearity). Intended for the reject-mask workflow to
        consume directly
        (e.g. ``pipe[:-1].transform(X)`` then ``pipe[-1].uncertainty_metrics``,
        since Pipeline does not forward custom methods).

        Returns a dict with, per sample:
            mean_proba:          predictive mean probabilities (n_samples, n_classes)
            bayes_error:         1 - max_c p̄(c)  (total uncertainty, 0-1 flavor)
            predictive_entropy:  H(p̄)            (total uncertainty, entropy flavor)
            aleatoric:           E_i[H(p_i)]      (expected per-pass entropy)
            bald:                H(p̄) - E_i[H(p_i)]  (epistemic / mutual information)
            epistemic_var:       mean over classes of per-pass probability variance
        """
        check_is_fitted(self)
        x = check_array(x)
        device = self._resolve_device()
        # Keep the (possibly just-unpickled) module and inputs on one device.
        self.module_.to(device)
        x_tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32).to(device)
        log_prior = self._logit_offset(device)

        eps = 1e-12
        n_samples, n_classes = x_tensor.shape[0], len(self.classes_)
        p_sum = torch.zeros((n_samples, n_classes), device=device)
        p_sq_sum = torch.zeros((n_samples, n_classes), device=device)
        entropy_sum = torch.zeros(n_samples, device=device)

        self._enable_mc_dropout()
        with torch.no_grad():
            for _ in range(self.mc_samples):
                logits = self.module_(x_tensor)
                if log_prior is not None:
                    logits = logits + log_prior
                p = torch.softmax(logits, dim=1)
                p_sum += p
                p_sq_sum += p**2
                entropy_sum += -(p * torch.log(p + eps)).sum(dim=1)

        p_mean = p_sum / self.mc_samples
        aleatoric = entropy_sum / self.mc_samples
        predictive_entropy = -(p_mean * torch.log(p_mean + eps)).sum(dim=1)
        bald = predictive_entropy - aleatoric
        epistemic_var = (p_sq_sum / self.mc_samples - p_mean**2).mean(dim=1)
        bayes_error = 1.0 - p_mean.max(dim=1).values

        return {
            "mean_proba": p_mean.cpu().numpy(),
            "bayes_error": bayes_error.cpu().numpy(),
            "predictive_entropy": predictive_entropy.cpu().numpy(),
            "aleatoric": aleatoric.cpu().numpy(),
            "bald": bald.cpu().numpy(),
            "epistemic_var": epistemic_var.cpu().numpy(),
        }


def bdl_reference(
    categorical_features: list[str] | None = None,
    numeric_features: list[str] | None = None,
) -> tuple[Pipeline, dict[str, Any]]:
    """Create a Bayesian deep learning pipeline using MC dropout.

    Args:
        categorical_features: List of categorical feature names.
        numeric_features: List of numeric feature names.

    Returns:
        A tuple containing the Pipeline and the parameter grid.

    """
    if categorical_features is None:
        categorical_features = []
    if numeric_features is None:
        numeric_features = []

    # Cross-fitted TargetEncoder (CV-safe) replaces OrdinalEncoder: it maps
    # categories to a meaningful, bounded scale (smoothed P(fraud|category))
    # instead of arbitrary integer codes, so high-cardinality columns no longer
    # enter the net unscaled and fake-ordered. StandardScaler then puts the
    # encoded values on the same ~unit-variance footing as the numerics.
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", TargetEncoder(target_type="binary", random_state=42)),
            ("scaler", StandardScaler()),
        ],
    )

    # QuantileTransformer is rank-based, so it handles both the heavy tails and
    # the zero-IQR sparsity that broke RobustScaler (see preprocessing_findings).
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "scaler",
                QuantileTransformer(output_distribution="normal", random_state=42),
            ),
        ],
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, categorical_features),
            ("num", num_transformer, numeric_features),
        ],
        remainder="drop",
    )

    clf = MCDropoutClassifier()
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ],
    )

    param_distributions = {
        "classifier__hidden_layer_sizes": [(100, 50), (100, 100), (100, 100, 100)],
        "classifier__activation": ["tanh", "relu"],
        "classifier__dropout": uniform(0.1, 0.3),  # Range [0.1, 0.5]
        "classifier__lr": loguniform(1e-4, 5e-3),
        "classifier__alpha": loguniform(1e-6, 1e-2),
        "classifier__max_iter": randint(200, 600),
    }
    return pipe, param_distributions


def run_pipeline(  # noqa: PLR0913
    pipe: Pipeline,
    param_distributions: dict[str, Any],
    tscv: TimeSeriesSplit,
    x: pdDataFrame,
    y: Any,  # noqa: ANN401
    dirpath: Path,
    n_jobs: int = 4,
) -> None:
    """Run RandomizedSearchCV on a pipeline with time-series cross-validation.

    Args:
        pipe: The scikit-learn Pipeline object.
        param_distributions: The parameter distributions for RandomizedSearchCV.
        tscv: The TimeSeriesSplit object.
        x: The feature DataFrame.
        y: The target variable.
        dirpath: Directory to save the results.

    """
    pr_auc_scorer = make_scorer(
        average_precision_score, response_method="predict_proba"
    )

    # refit=False: the search only selects hyperparameters here. The single
    # full-training-set fit (and the saved best_model.joblib) is produced by
    # fit_full_model, so refitting best_estimator_ on all of x, y would be a
    # wasted training pass (notably for the BDL torch model).
    grid_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        cv=tscv,
        scoring=pr_auc_scorer,
        n_jobs=n_jobs,
        verbose=1,
        random_state=42,
        refit=False,
    )

    _save_grid_search_params(grid_search, dirpath)

    grid_search.fit(x, y)

    _save_pipeline_results(
        grid_search.best_score_,
        grid_search.best_params_,
        dirpath,
    )


def fit_full_model(
    pipe: Pipeline,
    x: pdDataFrame,
    y: Any,  # noqa: ANN401
    dirpath: Path,
) -> Pipeline:
    """Refit a pipeline on the full training set using the best CV params.

    Reads ``best_params`` from the ``pipeline_results.json`` written by
    ``run_pipeline`` in ``dirpath``, applies them to a fresh clone of ``pipe``,
    and fits once on the entire provided training set (no cross-validation
    holdout). The fitted model is persisted to ``dirpath`` via
    ``_save_best_model``. This is deliberately separate from the CV search: CV
    selects the hyperparameters, this turns them into the deployable model.

    Args:
        pipe: An unfitted pipeline of the same structure used during the search
            (e.g. the one returned by ``xgboost_reference``/``bdl_reference``).
        x: The full training feature DataFrame.
        y: The full training target.
        dirpath: Directory holding ``pipeline_results.json``; the fitted model
            is saved here too.

    Returns:
        The pipeline fitted on the full training set.

    """
    results_path = dirpath / _RESULTS_FILENAME
    with results_path.open() as f:
        best_params = json.load(f)["best_params"]

    model = clone(pipe)
    model.set_params(**best_params)
    model.fit(x, y)

    _save_best_model(model, dirpath)
    return model


# Filename used to persist the side-by-side test-set comparison (see
# compare_models_on_test).
_TEST_COMPARISON_FILENAME = "test_comparison.json"


def _binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute PR-AUC (from scores) and precision (from hard labels).

    PR-AUC is a ranking metric and needs the continuous positive-class score;
    precision needs the model's hard decision. Keeping the two inputs separate
    (rather than re-thresholding the score here) means precision reflects the
    model's own decision rule and stays consistent with ``y_score`` even for the
    stochastic BDL model, where ``y_pred`` is the ``argmax`` of the same pass.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probability of the positive (fraud) class.
        y_pred: The model's predicted hard labels.

    Returns:
        A dict with ``pr_auc``, ``precision`` and the supporting ``n_samples``.

    """
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "n_samples": int(len(y_true)),
    }


def evaluate_on_test(
    model: Pipeline,
    x_test: pdDataFrame,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Score a fitted pipeline on the held-out test set.

    Takes a single ``predict_proba`` pass and derives the hard label by
    ``argmax`` of those same probabilities — identical to ``model.predict`` but
    guaranteed to match the scores PR-AUC is computed from (and, for the BDL
    model, computed from one MC-dropout pass rather than a second independent
    one). Applies to any ``predict_proba`` pipeline: XGBoost, and the BDL model
    evaluated *without* a reject mask.

    Args:
        model: A pipeline already fitted on the full training set.
        x_test: The held-out test features (same column layout as training X).
        y_test: The held-out test labels.

    Returns:
        A dict of test metrics (``pr_auc``, ``precision``, ``n_samples``).

    """
    proba = model.predict_proba(x_test)
    y_pred = model.classes_[np.argmax(proba, axis=1)]
    return _binary_metrics(y_test, proba[:, 1], y_pred)


# Reductions for collapsing the reference set's per-sample Bayes Error into the
# single scalar that each test datum is compared against. Bayes Error is bounded
# in [0, 0.5] for binary classification, so "max" tends to saturate near 0.5 and
# reject nothing; "mean" gives a meaningful operating point to start from.
_BAYES_ERROR_REDUCTIONS = {
    "mean": np.mean,
    "median": np.median,
    "max": np.max,
}


def bayes_error_reference(
    model: Pipeline,
    x_reference: pdDataFrame,
    reduction: str = "mean",
) -> float:
    """Compute the reference Bayes Error from a reference (e.g. training) set.

    Infers ``x_reference`` through the fitted BDL pipeline (one MC-dropout pass
    via ``MCDropoutClassifier.uncertainty_metrics``) and reduces the per-sample
    Bayes Error to a single scalar. A test datum is later rejected when its own
    Bayes Error exceeds this reference (see ``evaluate_bdl_on_test``).

    Args:
        model: The BDL pipeline fitted on the full training set.
        x_reference: Features of the reference set (the full training set here).
        reduction: How to collapse the reference Bayes Errors into one scalar;
            one of ``"mean"``, ``"median"`` or ``"max"``.

    Returns:
        The reference Bayes Error scalar.

    """
    classifier = model[-1]
    x_pre = model[:-1].transform(x_reference)
    bayes_error = classifier.uncertainty_metrics(x_pre)["bayes_error"]
    return float(_BAYES_ERROR_REDUCTIONS[reduction](bayes_error))


def _masked_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    keep_mask: np.ndarray,
) -> dict[str, float | None]:
    """Score the retained subset, reporting coverage and rejection counts.

    Guards the empty case (every sample rejected), where PR-AUC and precision are
    undefined, by returning ``None`` for those metrics rather than raising.

    Args:
        y_true: Ground-truth binary labels for all test samples.
        y_score: Positive-class probability for all test samples.
        y_pred: Predicted hard labels for all test samples.
        keep_mask: Boolean mask, ``True`` where a sample is kept (not rejected).

    Returns:
        A dict with the retained-subset metrics plus ``coverage`` and
        ``n_rejected``.

    """
    result: dict[str, float | None] = {
        "coverage": float(keep_mask.mean()),
        "n_rejected": int((~keep_mask).sum()),
    }
    if not keep_mask.any():
        result.update({"pr_auc": None, "precision": None, "n_samples": 0})
        return result
    result.update(
        _binary_metrics(y_true[keep_mask], y_score[keep_mask], y_pred[keep_mask])
    )
    return result


def evaluate_bdl_on_test(
    model: Pipeline,
    x_test: pdDataFrame,
    y_test: np.ndarray,
    reference_bayes_error: float,
) -> tuple[dict[str, float], dict[str, float | None]]:
    """Evaluate the BDL pipeline on test, both without and with the reject mask.

    Runs a single MC-dropout pass over the test set via ``uncertainty_metrics``,
    so the no-mask metrics, the reject-mask metrics and the rejection decision
    all derive from the same predictive distribution (and the test set is not
    inferred twice). The hard label is the ``argmax`` of the predictive mean
    (matching ``predict``). A test datum is rejected when its Bayes Error exceeds
    ``reference_bayes_error``; the kept subset is scored for the masked metrics.

    Args:
        model: The BDL pipeline fitted on the full training set (its final step
            is an ``MCDropoutClassifier``).
        x_test: The held-out test features (same column layout as training X).
        y_test: The held-out test labels.
        reference_bayes_error: The reference scalar from
            ``bayes_error_reference``.

    Returns:
        ``(no_mask_metrics, reject_mask_metrics)``. The masked dict also carries
        ``coverage``, ``n_rejected`` and the ``reference_bayes_error`` applied.

    """
    classifier = model[-1]
    x_pre = model[:-1].transform(x_test)
    metrics = classifier.uncertainty_metrics(x_pre)
    proba = metrics["mean_proba"]
    y_score = proba[:, 1]
    y_pred = classifier.classes_[np.argmax(proba, axis=1)]

    no_mask = _binary_metrics(y_test, y_score, y_pred)

    # Reject any test datum more uncertain (higher Bayes Error) than the
    # training reference; keep the rest.
    keep_mask = metrics["bayes_error"] <= reference_bayes_error
    reject_mask = _masked_metrics(y_test, y_score, y_pred, keep_mask)
    reject_mask["reference_bayes_error"] = reference_bayes_error

    return no_mask, reject_mask


def compare_models_on_test(  # noqa: PLR0913
    xgb_model: Pipeline,
    bdl_model: Pipeline,
    x_reference: pdDataFrame,
    x_test: pdDataFrame,
    y_test: np.ndarray,
    dirpath: Path,
    reduction: str = "mean",
) -> dict[str, dict[str, float]]:
    """Evaluate XGBoost and BDL (with and without reject mask) on the test set.

    Surfaces PR-AUC and precision for three configurations side by side and
    persists them to ``test_comparison.json`` in ``dirpath`` for comparison:
    ``xgboost``, ``bdl_no_reject_mask`` and ``bdl_reject_mask``. The reject mask
    rejects a test datum whose Bayes Error exceeds the reference computed from
    ``x_reference``.

    ``x_reference`` no longer has to be the full training set: it is whatever
    matrix a :class:`~utils.reference_sets.ReferenceSpec` resolved to. Pass the
    matching ``reference_provenance`` (from ``resolve_reference``) so the saved
    comparison records *which* reference subset it was calibrated against, under a
    ``"reference"`` key, instead of leaving it implicit.

    Args:
        xgb_model: The XGBoost pipeline fitted on the full training set.
        bdl_model: The BDL pipeline fitted on the full training set.
        x_reference: The reference features for the BDL Bayes Error reference
            (the full training set, or a resolved subset of interest).
        x_test: The held-out test features (same column layout as training X).
        y_test: The held-out test labels.
        dirpath: Directory to write the comparison JSON into.
        reduction: How to reduce the reference Bayes Errors to one scalar
            (passed to ``bayes_error_reference``).
        reference_provenance: Optional provenance dict (from
            ``resolve_reference``) describing how ``x_reference`` was selected;
            embedded in the output under ``"reference"`` when given.

    Returns:
        A dict keyed by configuration name, each holding that config's metrics
        (plus a ``"reference"`` entry when ``reference_provenance`` is provided).

    """
    reference = bayes_error_reference(bdl_model, x_reference, reduction=reduction)
    bdl_no_mask, bdl_reject_mask = evaluate_bdl_on_test(
        bdl_model, x_test, y_test, reference
    )
    results: dict[str, Any] = {
        "xgboost": evaluate_on_test(xgb_model, x_test, y_test),
        "bdl_no_reject_mask": bdl_no_mask,
        "bdl_reject_mask": bdl_reject_mask,
    }

    dirpath.mkdir(parents=True, exist_ok=True)
    file_path = dirpath / _TEST_COMPARISON_FILENAME
    with file_path.open("w") as f:
        json.dump(results, f, indent=4)

    return results
