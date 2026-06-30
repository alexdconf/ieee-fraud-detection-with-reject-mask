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
- `bdl_reference` — the main model: a PyTorch **MC-dropout** Bayesian net
  (`MCDropoutClassifier`). MC-dropout (Monte-Carlo dropout) keeps dropout active at
  inference and averages many stochastic forward passes, approximating a Bayesian
  net whose *spread* across passes estimates model uncertainty. That uncertainty
  feeds a reject mask. `uncertainty_metrics(X)` returns, from one MC run:
  **Bayes error** (`1 − max_c p̄(c)`, the chance the top-scoring class is wrong —
  total uncertainty), **predictive entropy** (entropy of the mean prediction — also
  total uncertainty), and **BALD** (Bayesian Active Learning by Disagreement — the
  mutual information between the prediction and the model weights, i.e. how much the
  MC passes *disagree*, isolating epistemic / model uncertainty).

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
   plus macro and per-class precision/recall**. (**PR-AUC** = area under the
   precision–recall curve, a threshold-free score from 0–1 of how well the model
   *ranks* fraud above legit.) Per-class keys are the `isFraud` labels — `"1"` =
   fraud, `"0"` = legit — and the flat `precision`/`recall` are the fraud (positive)
   class. These per-configuration numbers were an **exploratory** first pass: useful
   for sanity, but not safe to read as head-to-head model rankings — see
   *Risk–coverage evaluation* below for why, and what replaced them.

The **reject mask** computes a Bayes Error reference from a reference set and
rejects any test datum whose Bayes Error exceeds it; masked metrics are reported
over the retained subset alongside `coverage` and `n_rejected`. By default the
reference is `trouble_reference(bdl_model, x)` — the training rows the model is
most uncertain about (top 5% by Bayes Error) — so the mask only rejects test data
weirder than cases the model already struggles with. (This single-threshold reject
rule was likewise exploratory — see *Risk–coverage evaluation*.)

# Risk–coverage evaluation (current focus)

The comparisons above sit at a **single operating point** (one decision threshold,
one coverage). That data collection was **exploratory**, and it surfaced two reasons
those head-to-head numbers can mislead — which motivated the harness described here:

- The hard-label metrics (precision/recall/accuracy) depend on each model's
  probability **calibration** (how closely its predicted probabilities match
  observed frequencies), which differs across XGBoost and the BDL net — the BDL net
  trains under a balanced class prior, which shifts its probabilities. A
  fixed-threshold comparison therefore partly measures calibration, not ranking
  ability.
- Every metric, PR-AUC included, moves with the positive-class **prevalence** (the
  fraud rate of the rows being scored). The reject mask changes prevalence by
  dropping rows, so "before vs after masking" is not a like-for-like comparison.
  (PR-AUC's no-skill baseline *is* the prevalence, which is exactly why it drifts.)

The **risk–coverage harness** sidesteps both confounds by comparing *abstention
rules within one model at matched coverage*, where calibration and prevalence are
held constant and cancel out. Key terms:

- **Coverage** — the fraction of test transactions the model keeps (does not abstain
  on); 1.0 = predict on everything.
- **Risk–coverage curve** — error plotted against coverage as the model abstains on
  its most-uncertain cases first; a good uncertainty signal makes error fall steadily
  as coverage drops.
- **AURC** — Area Under the Risk–Coverage curve (lower is better); one summary number
  per rule.
- **Balanced error** — the risk plotted: `1 − ½(TPR + TNR)`, the mean of the two
  per-class error rates (TPR = true-positive rate = recall on fraud; TNR =
  true-negative rate = recall on legit). Unlike accuracy or PR-AUC it does *not* drift
  as abstention reshapes the class balance, so curves at different coverages stay
  comparable on one axis.

`scripts/compare_saved_models.py --reference risk_coverage` runs this from saved
models in **one MC-dropout pass**, scoring six abstention rules at identical
coverages and writing `risk_coverage.json`:

| rule | rejects rows by | what it represents |
| --- | --- | --- |
| `random` | a coin flip | the floor every real rule must beat |
| `bayes_error` | `1 − max p̄` (predictive confidence) | ≈ what a non-Bayesian model gives for free |
| `predictive_entropy` | entropy of the mean prediction | confidence, entropy flavour |
| `bald` | MC-pass disagreement (BALD) | genuinely epistemic (model) uncertainty |
| `epistemic_var` | variance of the per-pass probabilities | the other epistemic signal |
| `xgb_margin` | XGBoost's own confidence (its own predictions) | cross-model baseline |

The question it answers: do the **epistemic** rules (`bald`, `epistemic_var`) beat
the cheap **confidence** rules (`bayes_error`, `predictive_entropy`) and `random`? If
they do not, the MC-dropout apparatus is not earning its cost on this data.
`--bootstrap N` adds resampled error bars; `scripts/plot_risk_coverage.py` renders
the curves and the AURC league table to `supplementary_material/risk_coverage.png`.

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
uv run python scripts/compare_saved_models.py [REPORT_DIR] \
    --reference {trouble,all,none,random,risk_coverage}
```

- `trouble` (default) — reference = `trouble_reference` subset (`--metric`,
  `--quantile` tune which rows it selects).
- `all` — reference = full training set, swept over a set of Bayes-Error
  threshold multipliers, with one labelled `bdl_reject_mask_x{m}` result each.
- `none` — no reject mask; just XGBoost and BDL test metrics (fast, skips the
  training-set reference reconstruction).
- `random` — abstention baseline: drop test rows uniformly at random to
  `--quantile` coverage (here `--quantile` is read as the retained fraction), so a
  reject mask's gains can be checked against simply dropping the same fraction of
  rows. (No training-set reconstruction.)
- `risk_coverage` — the matched-coverage risk–coverage sweep over all six
  abstention rules (see *Risk–coverage evaluation* above); writes
  `risk_coverage.json` and prints the AURC league table. `--coverages` sets the
  coverage grid, `--bootstrap N` the error-bar resamples. Plot it with
  `scripts/plot_risk_coverage.py <run>/recompare/risk_coverage.json`.

# Project notes / handoff

Design decisions, debugging findings, and open work live in `notes/`:

- `notes/handoff.md` — **start here**: current state, what's settled, and what's next.
- `notes/preprocessing_findings.md` — the data-leak debugging and scaler findings.
- `notes/bdl_imbalance_calibration.md` — class imbalance, probability calibration,
  and the reject-mask uncertainty metrics.
- `notes/reject_mask_evaluation.md` — the held-out test split, full-data refit, the
  exploratory Bayes Error reject-mask evaluation, and the matched-coverage
  risk–coverage harness it motivated.
- `notes/categorical_and_numeric_encoding.md` — categorical/numeric feature
  encoding choices for the pipelines.
- `notes/pandas_and_polars.md` — why both libraries are here: polars for data
  handling, pandas only at the sklearn/XGBoost boundary.