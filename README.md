# ieee-fraud-detection-with-reject-mask
Implementing a classification with reject service for the IEEE-CIS Fraud Detection data set.

# Usage
`bash setup.sh`

`bash run.sh`

# Models

Three model builders in `src/utils/pipeline_tools.py`, each returning an sklearn
`Pipeline` + `param_distributions` for `RandomizedSearchCV`
(`TimeSeriesSplit`, PR-AUC scoring):

- `xgboost_reference` — XGBoost baseline (GPU).
- `mlp_reference` — sklearn MLP baseline.
- `bdl_reference` — the main model: a PyTorch MC-dropout Bayesian net
  (`MCDropoutClassifier`) whose predictive uncertainty feeds a reject mask.
  Exposes `uncertainty_metrics(X)` (predictive entropy, BALD, Bayes error, etc.).

Current best: BDL PR-AUC ≈ 0.453 (transactions-only). `main.py` currently runs
only the BDL pipeline; the XGB/MLP blocks are commented out.

# Project notes / handoff

Design decisions, debugging findings, and open work live in `notes/`:

- `notes/handoff.md` — **start here**: current state, what's settled, and what's next.
- `notes/preprocessing_findings.md` — the data-leak debugging and scaler findings.
- `notes/bdl_imbalance_calibration.md` — class imbalance, probability calibration,
  and the reject-mask uncertainty metrics.