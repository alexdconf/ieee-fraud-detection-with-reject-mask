# NN preprocessing: TargetEncoder + QuantileTransformer

Notes from implementing NEXT-UP items #1 (categorical encoding) and #2 (numeric
scaler) from the handoff, in `bdl_reference` and `mlp_reference`
(`src/utils/pipeline_tools.py`). `xgboost_reference` left untouched (scale-free).

## What changed

Both NN pipelines previously fed the net:

- categoricals: `SimpleImputer(most_frequent) + OrdinalEncoder` — arbitrary
  integer codes, **unscaled and fake-ordered**; high-cardinality cols (`card1`
  ~thousands) entered as giant, meaningless magnitudes.
- numerics: `SimpleImputer(median) + StandardScaler` — heavy tails survive as
  large z-scores (max |x| ~339 on the subset).

Now:

- categoricals: `SimpleImputer(most_frequent) + TargetEncoder(target_type=
  "binary", random_state=42) + StandardScaler`.
- numerics: `SimpleImputer(median) + QuantileTransformer(output_distribution=
  "normal", random_state=42)`.

### Why TargetEncoder

- sklearn's `TargetEncoder` is **cross-fitted inside `fit_transform`** (internal
  CV), so the training rows are encoded with out-of-fold target means — CV-safe,
  no leakage. At `transform` time (test/inference) it uses the full-data
  encoding, and unseen categories fall back to the global mean.
- Output is a meaningful, bounded scale (smoothed `P(fraud|category)` ~[0,1])
  instead of arbitrary ordered codes. The trailing `StandardScaler` then centers
  it to ~unit variance so the encoded categoricals sit on the same footing as
  the quantile-normal numerics (otherwise their tiny variance gets under-weighted
  by the first layer).
- Frequency/count encoding was the cheaper alternative; entity embeddings would
  be best but need `nn.Embedding` in `_MCDropoutNet` (deferred).
- Pipeline mechanics verified: `ColumnTransformer` forwards `y` to the cat
  sub-pipeline, and `Pipeline` calls `fit_transform` on non-final steps, so the
  cross-fitting path is actually exercised.

### Why QuantileTransformer

- Rank-based, so it handles BOTH the heavy tails AND the zero-IQR sparsity that
  broke `RobustScaler` (360/378 numeric cols have ~zero IQR — see
  `preprocessing_findings.md`). Maps each column to a normal regardless of its
  raw distribution, keeping magnitudes bounded.

## Validation (subset, the way the notes did it)

`scripts/validate_preprocessing.py` — builds OLD and NEW pipelines with identical
model params on one time-ordered 80/20 holdout, so only the preprocessing
differs. Checks the transformed test matrix (NaNs, max |x|) and PR-AUC
(average precision).

BDL, first 80k rows (time-ordered), single 80/20 holdout, base rate ~0.026:

| epochs | config | PR-AUC | max \|x\| | NaNs |
|---|---|---|---|---|
| 60  | OLD (ordinal + standard) | 0.4242 | 339.4 | 0 |
| 60  | NEW (target + quantile)  | **0.4490** | 65.2 | 0 |
| 100 | OLD (ordinal + standard) | 0.4095 | 339.4 | 0 |
| 100 | NEW (target + quantile)  | **0.4477** | 65.2 | 0 |

- NEW is **+0.025 to +0.038 PR-AUC** over OLD, head-to-head.
- Magnitudes ~5x tighter (65 vs 339), no NaNs — matrix is clean and bounded.
- NEW already scores ~0.448 on just 64k training rows, in the neighborhood of the
  0.453 full-run baseline — encouraging, but **not** an apples-to-apples compare.

## Still to do (left for the user — full runs take ~3h)

- Full-dataset comparison vs the **0.453** baseline:
  `uv run python scripts/validate_preprocessing.py --model bdl --rows 0 --epochs 100`
  and ideally a full `run_pipeline` (RandomizedSearchCV) best_score on the new
  pipeline.
- MLP head-to-head not run by the agent (sklearn MLP on CPU over the full data is
  a heavy run); same script with `--model mlp`.
- Items #3 (`add_indicator=True` for missingness) and #4 (VarianceThreshold/PCA)
  from the handoff remain.
