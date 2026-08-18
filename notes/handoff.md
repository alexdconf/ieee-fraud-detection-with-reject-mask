# Handoff / project status

Pick-up doc for a fresh agent. Last updated 2026-06-30.

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

- `xgboost_reference(categorical_features)` — XGBoost (`tree_method="hist"`,
  `device="cpu"` — the sklearn Pipeline feeds host numpy, so CPU hist avoids a
  per-call CPU→GPU copy; see the comment in the builder),
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

Each model now runs in **three stages** (see `notes/reject_mask_evaluation.md`):

1. `run_pipeline(pipe, param_distributions, tscv, x, y, dirpath, n_jobs)` runs
   `RandomizedSearchCV` (n_iter=10, `TimeSeriesSplit(n_splits=5)`) on `train_df`
   and writes `pipeline_results.json` (best_score + best_params) under `dirpath`
   (= per-model report dir, e.g. `reports/<ts>_transactions_only/bdl_reference`).
   **`refit=False`** — the search only selects hyperparameters; it does not refit
   `best_estimator_` on all of `x, y` (that was a wasted full training pass).
2. `fit_full_model(pipe, x, y, dirpath)` reads back `best_params`, fits **once on
   all of `train_df`**, and saves `best_model.joblib` — the single deployable
   artifact and the only full-training fit (previously this came from
   `grid_search.best_estimator_`). Reload with `joblib.load` (needs `src`
   importable so `MCDropoutClassifier` resolves); `predict_proba` moves the torch
   module onto the resolved device, so the artifact reloads on CPU or GPU.
3. Test step — `compare_models_on_test` scores XGB and BDL (with/without reject
   mask) on the held-out test set; see the "reject mask" note and section below.

`main.py` runs **`xgboost_reference` + `bdl_reference`** on transactions-only data
(XGB un-commented so the test comparison has both models); `mlp_reference` stays
commented out (toggle-by-comment workflow — this is why ruff flags
`mlp_reference` as an unused import; leave it). `n_jobs=4` is the RAM/throughput
sweet spot on this machine (~80% RAM, GPU fine).

## Latest result

**BDL PR-AUC ≈ 0.456, XGBoost ≈ 0.530** (latest full transactions-only run,
`reports/20260621-132004_transactions_only`, no reject mask). BDL is up from
≈0.037 (base rate) before the timestamp leak was fixed — PR-AUC = base rate is the
signature of "no ranking signal."

**Preprocessing #1+#2 are implemented** (cross-fitted TargetEncoder +
QuantileTransformer in both NN pipelines), replacing OrdinalEncoder+StandardScaler:
tighter bounded magnitudes (max |x| 339→65) and no NaNs. Subset head-to-head (BDL,
80k rows, time-ordered 80/20) showed **+0.025 to +0.038 PR-AUC** at matched params
— see `notes/categorical_and_numeric_encoding.md` for the table.

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

#1 and #2 are implemented and reflected in the latest full run (see Latest result
and `notes/categorical_and_numeric_encoding.md`). Outstanding: #3 (missingness
indicators), then #4.

## Reject mask & test evaluation — DONE

Full detail in `notes/reject_mask_evaluation.md`.

- **Held-out test set:** `holdout_test_split` reserves the most recent 20% of
  `train_transaction.csv` (chronological); saved to `report_dir/holdout_test.parquet`.
- **Test comparison:** `compare_models_on_test` writes
  `report_dir/test_comparison.json` for `xgboost`, `bdl_no_reject_mask`, and
  `bdl_reject_mask`. Each carries **PR-AUC, precision, recall, accuracy, plus
  `precision_macro`/`recall_macro` and `precision_per_class`/`recall_per_class`**
  (per-class accuracy is omitted — it equals per-class recall). All metrics funnel
  through `_binary_metrics` (single source of truth); masked results also carry
  `coverage` and `n_rejected`.
- **Reject rule:** reference = Bayes Error over a *reference set*, reduced by
  `mean` (`reduction` param; `max` saturates because binary Bayes Error ≤ 0.5). A
  test datum is rejected iff its `bayes_error > reference`. One MC pass over the
  test set feeds both masked and unmasked metrics.
- **Reference set = `trouble_reference(bdl_model, x)`** (defined in `src/main.py`):
  the training rows the model is most uncertain about — top 5% by Bayes Error (the
  metric the mask thresholds; pass `metric="bald"` for the epistemic flavour,
  `quantile=` for the cut). This is the "all-weird" reference from
  `bdl_imbalance_calibration.md` §5 — the mask only rejects test data weirder than
  cases the model already struggles with. (Replaces the deleted
  `reference_sets.py`/`subset_finder.py` `ReferenceSpec` framework, removed as
  over-built for current needs.)

### Re-evaluating saved models — `scripts/compare_saved_models.py`

Re-runs the test-step comparison from saved `best_model.joblib` artifacts, **no
retraining**. Loads the XGB+BDL pipelines and `holdout_test.parquet` from a run
dir, rebuilds the reference, and writes a fresh `test_comparison.json` under
`<run>/recompare/`. The training features for the reference are not persisted, so
they are reconstructed deterministically via the same `holdout_test_split`.
`--reference`:
- `trouble` (default) — `trouble_reference` subset (`--metric`, `--quantile`).
- `all` — full training set, swept over Bayes-Error threshold multipliers
  (`_ALL_REFERENCE_MULTIPLIERS`), one labelled `bdl_reject_mask_x{m}` result each
  (via `evaluate_bdl_reject_sweep` — single MC pass, applies every threshold).
- `none` — no mask; XGB+BDL test metrics only (fast, skips the reference).
- `random` — abstention baseline: drop test rows uniformly to `--quantile` coverage
  (read as the retained fraction here), the floor a real reject rule must beat.
- `risk_coverage` — the matched-coverage risk–coverage sweep (see below); writes
  `risk_coverage.json`, prints the AURC league table, `--coverages`/`--bootstrap` tune
  the grid and error bars.

## Current line of inquiry — risk–coverage harness (the test_comparison work was exploratory)

The per-configuration `test_comparison.json` numbers (single threshold, single
coverage) were an **exploratory** first pass. They surfaced two confounds that make
those head-to-head numbers unsafe as model rankings, which motivated a dedicated
harness:

- **Calibration** — the hard-label metrics (precision/recall/accuracy) depend on each
  model's probability calibration, which differs (the BDL net trains under a balanced
  prior). Only PR-AUC (a threshold-free ranking score) is safe *across* models.
- **Prevalence** — every metric, PR-AUC included, drifts with the positive-class rate
  (fraud %), which the reject mask changes by dropping rows. So `bdl_no_reject_mask`
  vs `bdl_reject_mask` is not like-for-like; it is purely a risk–coverage tradeoff
  (the mask reuses the same `y_pred`, only subsetting rows).

**`risk_coverage_sweep`** (`pipeline_tools.py`;
`compare_saved_models.py --reference risk_coverage`; plotted by
`scripts/plot_risk_coverage.py`) compares **abstention rules within one model at
matched coverage**, where calibration and prevalence cancel. From one MC pass it
scores `random`, `bayes_error`, `predictive_entropy`, `bald`, `epistemic_var`,
`xgb_margin`. Glossary:
- **Coverage** = fraction of rows kept (not abstained on). **Risk–coverage curve** =
  risk vs. coverage as the rule rejects its most-uncertain rows first. **AURC** = Area
  Under that curve, one number per rule, lower better. **Risk = balanced error**
  (`1 − ½(TPR+TNR)`), prevalence-robust so curves at different coverages stay
  comparable.
- Since `bayes_error = 1 − max p̄` ≈ the confidence margin (not distinctively
  Bayesian), **the test is whether `bald`/`epistemic_var` beat
  `bayes_error`/`predictive_entropy` and `random`.** If not, MC-dropout isn't earning
  its cost here. This subsumes the old "add BALD to the reject mask" item — BALD is
  now *evaluated* as a ranking signal, though not yet wired as the default reject-mask
  metric in `main.py`.

**Result** (`reports/20260625-155136_transactions_only/recompare/risk_coverage.json`,
bootstrapped): `bald` AURC 0.0605 beats `random` 0.0641 (conservative z-test p≈0.012);
confidence rules 0.0805 are decisively *worse than random* (z≈16). The confidence rules
tank fraud recall (0.37→0.00 by 80% coverage) buying precision by discarding boundary
frauds; `bald` lifts precision with recall ~flat — a coverage/precision exchange, not a
precision/recall one. Test it with `scripts/analyze_risk_coverage.py` (paired bootstrap
when raw draws are present, conservative z-test otherwise). Figure:
`supplementary_material/risk_coverage.png`. Full detail:
`notes/reject_mask_evaluation.md` (last section).

Still open:
- **Rerun the sweep to refresh `risk_coverage.json` with `aurc_boot_samples`** (the
  current file predates that field) so `analyze_risk_coverage.py` can run the stronger
  *paired* bootstrap test instead of the conservative z-test. The conservative test
  already calls `bald` > `random` significant (p≈0.012); paired only strengthens it.
- Wiring BALD as the reject mask's *default* signal in `main.py` (only if the sweep
  shows it helps — Bayes Error collapses to the mean and carries no disagreement
  signal; see `bdl_imbalance_calibration.md` §5).
- Optional calibration check (reliability curve + Brier) on a fold.

## Gotchas

- `MCDropoutClassifier` must keep `ClassifierMixin` BEFORE `BaseEstimator` (MRO),
  or sklearn treats it as a regressor and the PR-AUC scorer rejects predict_proba.
- Don't change `predict_proba`'s return shape — the RandomizedSearchCV scorer
  needs `(n_samples, n_classes)`.
- Requires `torch` (GPU). `uv run` for everything; Python 3.14.
