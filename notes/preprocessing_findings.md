# Preprocessing findings (BDL / MLP neural-net pipelines)

Notes from debugging why the neural-net models scored at the fraud base rate in
`src/utils/pipeline_tools.py`. Context: IEEE fraud, transactions-only, ~3.5%
positives, PR-AUC scorer.

## 1. The data leak: unscaled TIMESTAMP reaching the network

**Symptom.** `bdl_reference` `best_score` ≈ 0.037 — essentially the fraud base
rate. Average precision of a model with no ranking signal *equals* the positive
prevalence, so the model wasn't ranking fraud at all. Not a tuning / weighting /
calibration issue (those affect probability levels, not ranking).

**Root cause.** A column leak through `remainder="passthrough"`:

- `time_series_split` returns `x` = all columns except the target, so `x` still
  contains `TransactionDT` (the TIMESTAMP / sort key).
- `get_numeric_and_categorical_columns` *excludes* TIMESTAMP from both the
  numeric and categorical lists (intent: don't use it as a feature).
- But the `ColumnTransformer` used `remainder="passthrough"`, which silently
  re-inserted `TransactionDT` — **unscaled** — into the model input.

`TransactionDT` has magnitude ~1e6. With every real feature StandardScaled to
~unit variance, one feature that large dominates the first layer and prevents the
net from learning. XGBoost was immune (scale-free), which is why **only the
neural nets** flatlined (BDL transactions 0.037, MLP merged 0.041).

**Evidence (40k-row subset, time-ordered 80/20 split, 60 epochs):**

| | max abs feature value | PR-AUC | base rate |
|---|---|---|---|
| AS-IS (timestamp passthrough) | 975,137 | 0.074 | 0.027 |
| timestamp dropped             | 186     | 0.282 | 0.027 |

Dropping the leaked column took PR-AUC from ~base-rate to ~10x base rate, same
model/params.

**Fix.** `remainder="drop"` in both `bdl_reference` and `mlp_reference`. Verified
safe: numeric + categorical lists already enumerate all 392 real features, so
`drop` removes *only* `TransactionDT`.

Note: `xgboost_reference` still uses `remainder="passthrough"` — left as-is, since
XGBoost is scale-free and TransactionDT may even be a useful split feature there.

## 2. MLP was also missing scaling entirely

Separate from the leak, `mlp_reference`'s numeric transformer originally only
imputed (no scaler) — so the sklearn `MLPClassifier` got *all* features unscaled.
Added a scaler so it can train as a neural net.

## 3. RobustScaler fails on this data (zero-IQR columns)

Tried `RobustScaler` for the NN scalers (outlier robustness). It made things
**worse** because the data is overwhelmingly sparse:

- 360 of 378 numeric columns have **~zero IQR** (V/C/D features are >75% identical
  values). RobustScaler divides by IQR; when IQR=0, sklearn sets the scale to 1.0,
  so those columns are **centered but not scaled**.
- Result: a ~4e5-magnitude feature (e.g. V160) survives unscaled — reintroducing
  the exact giant-magnitude problem the timestamp fix removed.

| scaler | max abs feature value | quick PR-AUC (BDL, 60ep subset) |
|---|---|---|
| RobustScaler  | 431,396 (V160, IQR=0 → unscaled) | 0.11 |
| StandardScaler| 186                              | 0.28 |

StandardScaler divides by std (nonzero even for sparse columns), keeping
everything bounded. The max-abs / zero-IQR finding is deterministic; the PR-AUC
numbers are noisy subset runs but directionally consistent.

**Decision.** Reverted both NN pipelines to `StandardScaler` for now.

If outlier robustness is still wanted later, the workable options (RobustScaler is
not one, given the sparsity):
- StandardScaler + clipping (e.g. clip to ±5 after scaling).
- QuantileTransformer / PowerTransformer (handle skew + sparsity).

## 4. Housekeeping

- Renamed `BDL_reference` → `bdl_reference`, `MLP_reference` → `mlp_reference`
  (snake_case, `# noqa: N802` removed); references updated in `main.py`.
- `mlp_reference` / `xgboost_reference` show as unused imports in `main.py` only
  because their run blocks are currently commented out (toggle-by-comment
  workflow) — not removed on purpose.

## Current state of the NN pipelines

Both `bdl_reference` and `mlp_reference`:
`SimpleImputer(median) + StandardScaler` on numerics,
`SimpleImputer(most_frequent) + OrdinalEncoder` on categoricals,
`remainder="drop"`.
