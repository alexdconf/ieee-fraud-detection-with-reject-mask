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

Current best: BDL PR-AUC ≈ 0.453 (transactions-only). `main.py` runs the XGBoost
and BDL pipelines (MLP block commented out).

# Pipeline & evaluation

Each model runs three stages (see `notes/reject_mask_evaluation.md`):

1. **CV search** — `RandomizedSearchCV` (`TimeSeriesSplit`, PR-AUC, `refit=False`)
   on the training split; writes `pipeline_results.json`.
2. **Full-data refit** — `fit_full_model` refits the best params on all training
   data and saves the deployable `best_model.joblib`.
3. **Test step** — `compare_models_on_test` scores a chronologically held-out 20%
   test set and writes `test_comparison.json` with **PR-AUC + precision** for
   XGBoost, BDL (no reject mask), and BDL (with reject mask).

The **reject mask** (first cut) computes a Bayes Error reference over all of
training and rejects any test datum whose Bayes Error exceeds it; metrics are
reported over the retained subset alongside `coverage`.

# Project notes / handoff

Design decisions, debugging findings, and open work live in `notes/`:

- `notes/handoff.md` — **start here**: current state, what's settled, and what's next.
- `notes/preprocessing_findings.md` — the data-leak debugging and scaler findings.
- `notes/bdl_imbalance_calibration.md` — class imbalance, probability calibration,
  and the reject-mask uncertainty metrics.
- `notes/reject_mask_evaluation.md` — the held-out test split, full-data refit,
  and the implemented Bayes Error reject-mask evaluation.