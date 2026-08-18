# Reject mask & held-out test evaluation

Implemented 2026-06-21; reference + metrics updated 2026-06-25; risk–coverage
harness added 2026-06-30. Covers the held-out test split, the full-data refit, and
the test-set comparison step that scores XGBoost vs BDL (with and without the
reject mask). The *conceptual* discussion of the Bayes Error metric and its
caveats lives in `notes/bdl_imbalance_calibration.md` §5 — this note documents
what is actually wired up in code.

**Framing:** the single-operating-point comparison below (one threshold, one
coverage, per-configuration `test_comparison.json`) was an **exploratory** first
pass. It did its job — it exposed *why* those head-to-head numbers are unsafe to
read as model rankings — and that motivated the matched-coverage **risk–coverage
harness** documented in the new section at the end. Read this note as: exploratory
eval first, then the harness it led to.

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
**same column layout** `time_series_split` produces (sorted; both the target and
`TransactionDT` excluded from X — the timestamp is only a sort/split key, never a
feature). `time_series_split` now delegates to this helper so the two cannot drift
apart. (Previously the timestamp was left in X, which BDL dropped via
`remainder="drop"` but XGB leaked in via `remainder="passthrough"`; dropping it at
the source fixes both.)

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

## Risk–coverage harness (matched-coverage abstention comparison)

Added 2026-06-30 (`risk_coverage_sweep` in `pipeline_tools.py`;
`--reference risk_coverage` in `compare_saved_models.py`; plotting in
`scripts/plot_risk_coverage.py`). This is the **current line of inquiry**; the
single-point eval above was the exploratory groundwork.

**Why it exists.** Two confounds make the exploratory comparisons unsafe to read
head-to-head:

1. **Calibration.** The hard-label metrics (precision/recall/accuracy) apply a fixed
   threshold to predicted probabilities, so they depend on each model's
   *calibration* — how well its probabilities match observed frequencies. XGBoost and
   the BDL net are not calibrated alike (the BDL net trains under a balanced class
   prior, shifting its probabilities), so a fixed-threshold comparison conflates
   ranking ability with calibration. Only a threshold-free ranking metric (PR-AUC) is
   safe *across models*.
2. **Prevalence.** Every metric — PR-AUC included — moves with the positive-class
   **prevalence** (fraud rate of the scored rows). The reject mask changes prevalence
   by dropping rows (PR-AUC's no-skill baseline *is* the prevalence), so
   `bdl_no_reject_mask` vs `bdl_reject_mask` is not like-for-like — the populations
   differ. Because the mask reuses the same `y_pred` and only subsets the rows, that
   pairing is purely a *risk–coverage* tradeoff and must be read as one.

**What it does.** Compares **abstention rules within one model at matched coverage**,
where calibration and prevalence are held fixed and cancel. Terms:

- **Coverage** — fraction of test rows kept (not abstained on); 1.0 = predict on all.
- **Risk–coverage curve** — risk vs. coverage as the rule abstains on its
  most-uncertain rows first; a good signal makes risk fall as coverage drops.
- **AURC** — Area Under the Risk–Coverage curve (trapezoidal; lower is better) — one
  number per rule. `_aurc` integrates only finite points, so an undefined risk at
  aggressive coverage drops out instead of poisoning the area.
- **Balanced error** — the risk used: `1 − ½(TPR + TNR)` (TPR = recall on fraud, TNR
  = recall on legit). Prevalence-robust — each class contributes equally regardless
  of how abstention reshaped the class balance — so curves at different coverages stay
  comparable. `_balanced_error` returns `nan` when the kept subset has lost a class
  (honest about degenerate coverage rather than averaging over the survivor); the
  default grid stops at 0.80 coverage to avoid that on this imbalance.

**Rules scored** (all from one MC-dropout pass; `random` and `xgb_margin` aside, all
keep the BDL predictions and differ only in *which rows* they reject):

| rule | rejects by | family |
| --- | --- | --- |
| `random` | uniform noise | floor every rule must beat |
| `bayes_error` | `1 − max p̄` (predictive confidence) | confidence (≈ free, non-Bayesian) |
| `predictive_entropy` | entropy of the mean prediction | confidence |
| `bald` | BALD — MC-pass disagreement (mutual info) | epistemic |
| `epistemic_var` | variance of per-pass probabilities | epistemic |
| `xgb_margin` | XGBoost confidence over its own predictions | cross-model baseline |

In the binary case `bayes_error = 1 − max p̄` is monotone with the confidence margin,
so it is essentially the *confidence* baseline, not a distinctively Bayesian one. **The
real test is whether `bald`/`epistemic_var` beat `bayes_error`/`predictive_entropy`
and `random`.** If they don't, the MC-dropout machinery isn't earning its cost here.
`--bootstrap N` resamples the test set for AURC/risk error bars (the MC pass is done
once; only the rows are resampled), so "beats random" can be judged against noise.
The resample `idx` is shared across rules each iteration, so the stored raw draws
(`aurc_boot_samples`) are **paired** — `scripts/analyze_risk_coverage.py` uses them
for a paired bootstrap AURC test (`AURC_baseline(b) − AURC_rule(b)` per draw), which
cancels the shared row-sampling noise. Without raw draws it falls back to an
independent-Gaussian z-test on `aurc_boot_mean/std`, which is *conservative* (the true
paired SE is smaller, so a significant result there is significant a fortiori).

**Result** (`reports/20260625-155136_transactions_only/recompare/risk_coverage.json`,
bootstrapped). On balanced-error AURC: `bald` 0.0605 beats `random` 0.0641
(conservative z-test p≈0.012 — significant; the JSON predates `aurc_boot_samples`, so
rerun the sweep to get the stronger *paired* test); the confidence rules
`bayes_error`/`predictive_entropy` 0.0805 are **decisively worse than random**
(z≈16). The mechanism is a recall/precision tradeoff that the confidence rules fall
into and `bald` largely escapes: as coverage drops to 0.80, `bayes_error` recall
(fraud catch rate) collapses 0.37→0.00 while precision →1.0 — it buys precision by
discarding essentially every fraud (uncertain ⇒ boundary frauds), so balanced error
blows up to 0.5. `bald` instead lifts precision 0.58→0.87 with recall roughly flat
(0.37→0.40, dipping only at the most aggressive coverage), so it is a coverage/precision
exchange, not a precision/recall one. Balanced error stays floored near 0.30 even for
`bald` because fraud recall never climbs past ~0.40. Figure:
`supplementary_material/risk_coverage.png`; tests via `scripts/analyze_risk_coverage.py`.

## Status of `main.py`

`main.py` now runs **both** `xgboost_reference` and `bdl_reference` on
transactions-only data (the XGB block was un-commented so the comparison has both
models); `mlp_reference` stays commented out. Each model gets CV search → full
refit → and both feed the single `compare_models_on_test` call.
