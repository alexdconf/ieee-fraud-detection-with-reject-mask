# Why both pandas and polars are in this repo

Short answer: **polars is the project's dataframe library; pandas appears only at
the sklearn/XGBoost boundary.** The split is deliberate, not an accident.

## Polars — everything up to the models

All data handling is polars (`src/utils/data_handlers.py`):

- `load_csv_data` → `pl.read_csv` (the raw IEEE-CIS CSVs).
- `holdout_test_split`, `time_series_split` — chronological sort + slicing.
- `holdout_test.parquet` is written/read with polars.
- column typing / null profiling / EDA, and the reject-mask reference
  reconstruction in `scripts/compare_saved_models.py`.

## Pandas — only the feature matrix `X`, at the model boundary

`features_and_target(df, target, timestamp)` ends with `.to_pandas()`, so the
feature matrix `X` that enters the pipelines is a pandas `DataFrame` (the target
`y` is plain numpy). The reason is the modeling stack:

- scikit-learn `ColumnTransformer` selects columns **by name** and the
  transformers (`SimpleImputer`, `TargetEncoder`, `QuantileTransformer`,
  `StandardScaler`, `OrdinalEncoder`) expect a pandas frame, and
- `XGBClassifier` consumes pandas/numpy.

Feeding polars through that chain isn't reliably supported across all of these, so
the conversion is done once, at the single point where data crosses from "our
code" into "the model pipelines." Everything downstream of that boundary —
`run_pipeline`, `fit_full_model`, `evaluate_on_test`, `compare_models_on_test`,
`trouble_reference` — therefore operates on the pandas `X`. That is why those
functions are typed `pdDataFrame` and why a filter like
`trouble_reference`'s `x[values >= threshold]` is pandas: its input is `X` and its
output feeds straight back into `model[:-1].transform(...)`.

## Rule of thumb

- New data loading / wrangling / IO → **polars**.
- Anything handed to or returned from a fitted sklearn/XGBoost pipeline → **pandas
  `X` / numpy `y`**, and keep the `.to_pandas()` conversion localized to
  `features_and_target` rather than sprinkling it around.

Making `X` polars end-to-end would mean dropping that `.to_pandas()` and verifying
the full transformer chain accepts polars — and it would require retraining (the
saved `best_model.joblib` artifacts were fit on pandas). Deferred unless that
cost is worth paying.
