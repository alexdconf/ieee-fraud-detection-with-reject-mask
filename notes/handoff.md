# Handoff / project status

Pick-up doc for a fresh agent. Last updated 2026-06-15.

## Goal

Classification-with-reject service for IEEE-CIS Fraud Detection (~3.5% positives,
heavily imbalanced, time-ordered). A Bayesian deep-learning model (MC-dropout)
provides predictive uncertainty that a downstream **reject mask** uses to abstain
on low-confidence / out-of-distribution transactions.

## Where things are (code)

All model/pipeline code is in `src/utils/pipeline_tools.py`. Entry point is
`src/main.py`; data/CV helpers in `src/utils/data_handlers.py`; paths in
`src/constants.py` (`TARGET="isFraud"`, `TIMESTAMP="TransactionDT"`).

Three model builders, each returning `(sklearn Pipeline, param_distributions)`
for `RandomizedSearchCV`:

- `xgboost_reference(categorical_features)` — XGBoost (GPU, `device="cuda"`),
  `remainder="passthrough"` (fine: XGB is scale-free).
- `mlp_reference(categorical_features, numeric_features)` — sklearn MLPClassifier.
- `bdl_reference(categorical_features, numeric_features)` — the BDL model
  (`MCDropoutClassifier`). **This is the main model.**

Both NN pipelines preprocess identically: numerics = `SimpleImputer(median) +
QuantileTransformer(output_distribution="normal")`; categoricals =
`SimpleImputer(most_frequent) + TargetEncoder(target_type="binary") +
StandardScaler`; `remainder="drop"`. (Was StandardScaler + OrdinalEncoder — see
`notes/categorical_and_numeric_encoding.md`.) XGB still uses OrdinalEncoder +
`passthrough`.

`MCDropoutClassifier` (a `ClassifierMixin, BaseEstimator` wrapper around the
PyTorch `_MCDropoutNet`):
- Key params: `hidden_layer_sizes`, `dropout`, `activation`, `lr`, `alpha`
  (Adam weight decay), `max_iter` (epochs), `batch_size=4096`, `mc_samples=30`,
  `class_weight="balanced"`, `prior_correction=True`, `random_state`, `device`.
- `predict_proba` averages softmax over `mc_samples` stochastic passes (dropout
  kept on at inference), with the prior-correction applied.
- `uncertainty_metrics(X)` returns a dict of reject-mask signals from one MC run:
  `mean_proba, bayes_error, predictive_entropy, aleatoric, bald, epistemic_var`.
  NOTE: Pipeline does not forward this method — call
  `pipe[:-1].transform(X); pipe[-1].uncertainty_metrics(Xt)`.

`run_pipeline(pipe, param_distributions, tscv, x, y, dirpath, n_jobs)` runs
`RandomizedSearchCV` (n_iter=10, `TimeSeriesSplit(n_splits=5)`) and writes, under
`dirpath` (= the per-model report dir, e.g. `reports/<ts>_transactions_only/
bdl_reference`): `pipeline_results.json` (best_score + best_params) and
`best_model.joblib` — the fitted `best_estimator_` (preprocessing + trained
model). Reload with `joblib.load` (needs `src` importable so `MCDropoutClassifier`
resolves); `predict_proba` moves the torch module onto the resolved device, so the
artifact reloads on CPU or GPU.

`main.py` currently runs **only `bdl_reference`** on transactions-only data; the
XGB and MLP blocks are commented out (toggle-by-comment workflow — this is why
ruff flags `mlp_reference`/`xgboost_reference` as unused imports; leave them).
`n_jobs=4` is the RAM/throughput sweet spot on this machine (~80% RAM, GPU fine).

## Latest result

**BDL PR-AUC ≈ 0.453** (full transactions-only run) after fixing the timestamp
leak and using StandardScaler. Up from ≈0.037 (base rate) when the leak was
present. PR-AUC = base rate is the signature of "no ranking signal."

**Preprocessing #1+#2 now implemented** (TargetEncoder + QuantileTransformer in
both NN pipelines). Subset head-to-head (BDL, 80k rows, time-ordered 80/20) shows
**+0.025 to +0.038 PR-AUC** over the old OrdinalEncoder+StandardScaler at matched
params, with tighter bounded magnitudes (max |x| 339→65) and no NaNs. The
full-data run vs the 0.453 baseline is **still pending** (the user runs the ~3h
jobs) — see `notes/categorical_and_numeric_encoding.md` for the table and the
exact command (`scripts/validate_preprocessing.py`).

## What was settled (see the other notes files)

- `notes/preprocessing_findings.md` — the `remainder="passthrough"` timestamp
  leak (root cause of base-rate scores), the MLP-missing-scaler bug, and why
  RobustScaler fails here (360/378 numeric cols have ~zero IQR → left unscaled).
- `notes/bdl_imbalance_calibration.md` — class imbalance options, why balanced
  weighting inflates probabilities, the analytic logit prior-correction
  (implemented), Bayes Error vs BALD for the reject mask, and the
  `uncertainty_metrics` design.
- `notes/categorical_and_numeric_encoding.md` — NEXT-UP #1+#2 (implemented):
  cross-fitted `TargetEncoder` + `StandardScaler` for categoricals,
  `QuantileTransformer(normal)` for numerics, why each, and subset PR-AUC results.

## Open work — NEXT UP: preprocessing improvements

Ranked, for the NN pipelines (`bdl_reference`/`mlp_reference`; leave XGB as-is):

1. ~~**Categorical encoding.**~~ **DONE.** Replaced OrdinalEncoder with sklearn's
   cross-fitted `TargetEncoder(target_type="binary")` + `StandardScaler` (CV-safe;
   bounded, meaningful scale). Entity embeddings (`nn.Embedding` in
   `_MCDropoutNet`) would be the next step up but were deferred.
2. ~~**Numeric scaler.**~~ **DONE.** Replaced StandardScaler with
   `QuantileTransformer(output_distribution="normal")` — handles heavy tails AND
   the zero-IQR sparsity that broke RobustScaler.
3. **Missingness as signal:** `add_indicator=True` on the imputers — null
   patterns are predictive in fraud. Cheap bolt-on. **(next)**
4. **Dimensionality (later):** `VarianceThreshold` to drop near-constant cols;
   PCA on the correlated V-column blocks.

#1 and #2 are implemented and pass a subset head-to-head (see Latest result and
`notes/categorical_and_numeric_encoding.md`). Outstanding: the full-data run vs
0.453 (user-run), then #3.

## Open work — reject mask & calibration (not yet built)

- The reject decision logic itself is **not implemented** — only the
  `uncertainty_metrics` signals exist. The intended workflow: compute a reference
  metric on a semantically-meaningful holdout (all-weird / all-perfect inputs),
  compute the same per-candidate over N MC passes, compare, decide by semantic
  context. Decide which signal drives rejection (BALD for OOD/"weird";
  Bayes Error / entropy for in-distribution ambiguity — they're complementary).
- Optional calibration check (reliability curve + Brier) on a fold.

## Gotchas

- `MCDropoutClassifier` must keep `ClassifierMixin` BEFORE `BaseEstimator` (MRO),
  or sklearn treats it as a regressor and the PR-AUC scorer rejects predict_proba.
- Don't change `predict_proba`'s return shape — the RandomizedSearchCV scorer
  needs `(n_samples, n_classes)`.
- Requires `torch` (GPU). `uv run` for everything; Python 3.14.
