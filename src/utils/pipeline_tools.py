"""Utility functions for building and running machine learning pipelines."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import torch
from scipy.stats import randint, uniform
from scipy.stats.distributions import loguniform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, make_scorer
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
    file_path = dirpath / "pipeline_results.json"
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
        estimator: The fitted pipeline (e.g. ``grid_search.best_estimator_``).
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

    clf = XGBClassifier(objective="binary:logistic", device="cuda")
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
            return torch.as_tensor(
                self.log_prior_, dtype=torch.float32, device=device
            )
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

    grid_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        cv=tscv,
        scoring=pr_auc_scorer,
        n_jobs=n_jobs,
        verbose=1,
        random_state=42,
    )

    _save_grid_search_params(grid_search, dirpath)

    grid_search.fit(x, y)

    _save_pipeline_results(
        grid_search.best_score_,
        grid_search.best_params_,
        dirpath,
    )
    _save_best_model(grid_search.best_estimator_, dirpath)
