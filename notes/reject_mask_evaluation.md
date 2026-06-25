# Reject mask & held-out test evaluation

Implemented 2026-06-21; reference + metrics updated 2026-06-25. Covers the
held-out test split, the full-data refit, and
the test-set comparison step that scores XGBoost vs BDL (with and without the
reject mask). The *conceptual* discussion of the Bayes Error metric and its
caveats lives in `notes/bdl_imbalance_calibration.md` §5 — this note documents
what is actually wired up in code.

All functions are in `src/utils/pipeline_tools.py`; the split helper is in
`src/utils/data_handlers.py`; everything is orchestrated in `src/main.py`.

## Pipeline stages (per model)

For each model `main.py` now runs three distinct stages:

1. **CV search** — `run_pipeline(...)` runs `RandomizedSearchCV`
   (`TimeSeriesSplit`, PR-AUC) on `train_df` and writes `pipeline_results.json`
   (best_score + best_params). **`refit=False`** here: the search only *selects*
   hyperparameters. It no longer refits `best_estimator_` on all of `x, y`, which
   was a wasted full training pass (notably expensive for the BDL torch model).
2. **Full-data refit** — `fit_full_model(pipe, x, y, dirpath)` reads back
   `best_params`, clones a fresh pipeline, fits **once on all of `train_df`**, and
   saves it as `best_model.joblib`. This is the single deployable artifact and the
   *only* full-training fit. (Previously `best_model.joblib` came from
   `grid_search.best_estimator_`; that line was removed.)
3. **Test step** — see below.

## Held-out test split

`holdout_test_split(df, timestamp, test_fraction=0.2)` (in `data_handlers.py`)
sorts by `TransactionDT` and reserves the **most recent 20%** as a labeled test
set, mirroring the forward-in-time CV. Everything upstream (EDA, CV, refit) only
sees `train_df`; the test set is never touched during training. The test rows are
persisted to `report_dir/holdout_test.parquet`.

`features_and_target(df, target, timestamp)` builds the test `(X, y)` with the
**same column layout** `time_series_split` produces (sorted, target excluded,
`TransactionDT` kept as a column — BDL drops it via `remainder="drop"`, XGB passes
it through). `time_series_split` now delegates to this helper so the two cannot
drift apart.

## Test comparison step

`compare_models_on_test(xgb_model, bdl_model, x_reference, x_test, y_test,
dirpath, reduction="mean")` writes `report_dir/test_comparison.json` with three
configurations:

- `xgboost` — `evaluate_on_test` (XGB `predict_proba`).
- `bdl_no_reject_mask` — BDL scored on the full test set.
- `bdl_reject_mask` — BDL scored on the **retained** subset after rejection.

### Metrics

`_binary_metrics` reports, from two deliberately different inputs:

- **PR-AUC** (`average_precision_score`) from the continuous positive-class
  score. A ranking metric — it *needs* probabilities, not hard labels.
- **Hard-label metrics** from the model's own decision: `precision`, `recall`,
  `accuracy` (aggregate), `precision_macro`/`recall_macro` (unweighted mean across
  classes), and `precision_per_class`/`recall_per_class` (keyed by label). The
  hard label is `argmax` of the **same** probabilities used for PR-AUC — identical
  to `.predict()`, but computed from one set of scores. This matters for the
  stochastic BDL model: a second `.predict()` call would be a *different*
  MC-dropout pass, desyncing the label from the score. No hand-rolled 0.5
  threshold (that would just re-implement `.predict()` and smuggle in a magic
  number; a non-default operating point, if ever wanted, should be tuned on
  validation, not the test set). Per-class accuracy is intentionally omitted — the
  share of a class's samples predicted correctly equals that class's recall.

The masked config additionally reports `coverage` (fraction kept), `n_rejected`,
and the `reference_bayes_error` applied. When every sample is rejected the metrics
come back `null` instead of raising.

## The reject mask (Bayes Error vs training reference)

The intended general workflow (see `bdl_imbalance_calibration.md` §5): infer a
reference set, get reject-mask metrics; infer the test set (N MC passes/datum),
get the same metrics; compare per-datum to decide reject/keep. The signal compared
is **Bayes Error**. The reference set started as *all of training* and is now
`trouble_reference` (see below).

- `bayes_error_reference(model, x_reference, reduction="mean")` infers the
  reference set through the fitted BDL pipeline (one `uncertainty_metrics` MC run)
  and reduces its per-sample Bayes Errors to a single scalar.
- `evaluate_bdl_on_test(model, x_test, y_test, reference)` runs **one** MC-dropout
  pass over the test set (`uncertainty_metrics`), so the no-mask metrics, the
  reject-mask metrics, and the rejection decision all share one predictive
  distribution and the test set is not inferred twice. **Reject rule: a test
  datum is rejected iff its `bayes_error > reference`** (kept iff `<=`).
- `_masked_metrics` scores the kept subset and guards the empty case (every
  sample rejected → all metrics returned as `null` instead of raising).

In `main.py` the reference is `trouble_reference(bdl_model, x)`: the training rows
the model is most uncertain about (top 5% by Bayes Error — the metric the mask
thresholds; `metric="bald"` selects the epistemic flavour, `quantile=` the cut).
The mask then only rejects test data weirder than cases the model already
struggles with. (Earlier this was the whole training matrix `x`; the
`reference_sets.py`/`subset_finder.py` `ReferenceSpec` framework that briefly
generalised it was removed as over-built.)

### Why `reduction="mean"` (not max)

Binary Bayes Error is `1 − max_c p̄(c)` and is **bounded in [0, 0.5]** (the
winning class always has p̄ ≥ 0.5). So a `max` reference saturates near 0.5 and
rejects essentially nothing. `mean` (reject test data more uncertain than the
*average* training datum) gives a meaningful starting operating point. `reduction`
is a parameter (`mean`/`median`/`max`); add a quantile reduction if a specific
target coverage is wanted. (Synthetic smoke check: mean reference → ~47% coverage.)

### Known caveats / deliberate choices

- **Reference is on data the model trained on**, so its Bayes Error is
  optimistically low. `trouble_reference` deliberately picks the most-uncertain
  training rows, pushing the threshold high (lenient); swapping to a held-off or
  otherwise different reference is just a different `x_reference` argument.
- **Bayes Error carries no epistemic signal.** Linearity of expectation collapses
  it to `1 − p̄` (the mean-probability flavor), and it is the metric *distorted*
  by the balanced-weighting inflation — see `bdl_imbalance_calibration.md` §5.
  Starting with Bayes Error is intentional; **BALD** is the disagreement/OOD
  alternative already exposed by `uncertainty_metrics` for the next iteration.

## Re-evaluation & threshold sweep (no retraining)

`scripts/compare_saved_models.py` re-runs this comparison from saved
`best_model.joblib` artifacts — `--reference trouble|all|none`. With
`--reference all` it sweeps the reject threshold over multipliers of the reference
scalar via `evaluate_bdl_reject_sweep` (one MC pass over the test set, every
threshold applied), writing one labelled `bdl_reject_mask_x{m}` result each;
`--reference none` skips the mask entirely. See `notes/handoff.md` for usage.

## Status of `main.py`

`main.py` now runs **both** `xgboost_reference` and `bdl_reference` on
transactions-only data (the XGB block was un-commented so the comparison has both
models); `mlp_reference` stays commented out. Each model gets CV search → full
refit → and both feed the single `compare_models_on_test` call.
