# ieee-fraud-detection-with-reject-mask
Implementing a classification with reject service for the IEEE-CIS Fraud Detection data set.

# Usage
`bash setup.sh`

`bash run.sh`

# Models

Three model builders in `src/utils/pipeline_tools.py`, each returning an sklearn
`Pipeline` + `param_distributions` for `RandomizedSearchCV`
(`TimeSeriesSplit`, PR-AUC scoring):

- `xgboost_reference` — XGBoost baseline (CPU hist).
- `mlp_reference` — sklearn MLP baseline.
- `bdl_reference` — the main model: a PyTorch MC-dropout Bayesian net
  (`MCDropoutClassifier`) whose predictive uncertainty feeds a reject mask.
  Exposes `uncertainty_metrics(X)` (predictive entropy, BALD, Bayes error, etc.).

Latest transactions-only run: BDL PR-AUC ≈ 0.456, XGBoost ≈ 0.530 (no reject
mask). `main.py` runs the XGBoost and BDL pipelines (MLP block commented out).

# Pipeline & evaluation

Each model runs three stages (see `notes/reject_mask_evaluation.md`):

1. **CV search** — `RandomizedSearchCV` (`TimeSeriesSplit`, PR-AUC, `refit=False`)
   on the training split; writes `pipeline_results.json`.
2. **Full-data refit** — `fit_full_model` refits the best params on all training
   data and saves the deployable `best_model.joblib`.
3. **Test step** — `compare_models_on_test` scores a chronologically held-out 20%
   test set and writes `test_comparison.json` for XGBoost, BDL (no reject mask),
   and BDL (with reject mask). Each carries **PR-AUC, precision, recall, accuracy,
   plus macro and per-class precision/recall**. Per-class keys are the `isFraud`
   labels — `"1"` = fraud, `"0"` = legit — and the flat `precision`/`recall` are
   the fraud (positive) class.

The **reject mask** computes a Bayes Error reference from a reference set and
rejects any test datum whose Bayes Error exceeds it; masked metrics are reported
over the retained subset alongside `coverage` and `n_rejected`. By default the
reference is `trouble_reference(bdl_model, x)` — the training rows the model is
most uncertain about (top 5% by Bayes Error) — so the mask only rejects test data
weirder than cases the model already struggles with.

## The held-out test set (`holdout_test.parquet`)

Each run writes `holdout_test.parquet` to its report directory — the held-out test
set itself. `holdout_test_split` sorts `train_transaction.csv` by `TransactionDT`
and reserves the **most recent 20%** of rows as the test set (a deterministic,
time-based cut — not a random or stratified sample, so its fraud rate is whatever
the latest transactions happen to have). Those rows — the full feature columns
plus the `isFraud` label — are saved to the parquet. Only this test slice is
persisted; the training 80% is reconstructed on demand from the CSV, which is how
`compare_saved_models.py` rebuilds the reject-mask reference while loading the test
set straight from the parquet.

# Re-evaluating saved models

`scripts/compare_saved_models.py` re-runs the test-step comparison from saved
`best_model.joblib` artifacts — no retraining. It loads the XGBoost and BDL
pipelines and the held-out test set from a run directory, rebuilds the
reject-mask reference, and writes a fresh `test_comparison.json` (under
`<run>/recompare/`).

```
uv run python scripts/compare_saved_models.py [REPORT_DIR] --reference {trouble,all,none}
```

- `trouble` (default) — reference = `trouble_reference` subset (`--metric`,
  `--quantile` tune which rows it selects).
- `all` — reference = full training set, swept over a set of Bayes-Error
  threshold multipliers, with one labelled `bdl_reject_mask_x{m}` result each.
- `none` — no reject mask; just XGBoost and BDL test metrics (fast, skips the
  training-set reference reconstruction).

# Project notes / handoff

Design decisions, debugging findings, and open work live in `notes/`:

- `notes/handoff.md` — **start here**: current state, what's settled, and what's next.
- `notes/preprocessing_findings.md` — the data-leak debugging and scaler findings.
- `notes/bdl_imbalance_calibration.md` — class imbalance, probability calibration,
  and the reject-mask uncertainty metrics.
- `notes/reject_mask_evaluation.md` — the held-out test split, full-data refit,
  and the implemented Bayes Error reject-mask evaluation.
- `notes/categorical_and_numeric_encoding.md` — categorical/numeric feature
  encoding choices for the pipelines.