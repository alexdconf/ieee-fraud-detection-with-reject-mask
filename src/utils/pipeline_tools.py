"""Utility functions for building and running machine learning pipelines."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import average_precision_score, make_scorer
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
        except (TypeError, OverflowError):
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
        except (TypeError, OverflowError):
            serializable_params[key] = str(value)

    file_path = dirpath / "pipeline_params.json"

    with file_path.open("w") as f:
        json.dump(serializable_params, f, indent=4)


def pipeline_nans_passthrough(
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

    clf = XGBClassifier(objective="binary:logistic")
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


def pipeline_nans_imputed(
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

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ],
    )

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ],
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, categorical_features),
            ("num", num_transformer, numeric_features),
        ],
        remainder="passthrough",
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


def run_pipeline(  # noqa: PLR0913
    pipe: Pipeline,
    param_distributions: dict[str, Any],
    tscv: TimeSeriesSplit,
    x: pdDataFrame,
    y: Any,  # noqa: ANN401
    dirpath: Path,
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
    pr_auc_scorer = make_scorer(average_precision_score, response_method='predict_proba')

    grid_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        cv=tscv,
        scoring=pr_auc_scorer,
        n_jobs=-1,
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
