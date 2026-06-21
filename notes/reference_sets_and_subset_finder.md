# Reference sets & subset finder (reject-mask `x_reference`)

Scaffolded 2026-06-21. Two capabilities layered on top of the existing
reject-mask workflow (`notes/reject_mask_evaluation.md`):

1. **Specify & capture the reference set** — make `x_reference` (the set the BDL
   Bayes-Error reference is computed from) a *declared, serializable* object, and
   record it in the report. Fully functional.
2. **Find subsets of interest** — propose and rank candidate reference subsets
   automatically. Plumbing functional; the "interestingness" scoring is a
   documented placeholder (the open research piece).

Motivation: the first cut hard-coded `x_reference = x` (all of training) and
recorded nothing about it. The intended workflow (`bdl_imbalance_calibration.md`
§5) is a *semantically-meaningful* reference — "all-weird / all-perfect" — so the
reference set needs to be both **definable** and **traceable**.

## Part 1 — `src/utils/reference_sets.py` (functional)

A `ReferenceSpec` declares *how* to pick the reference rows, not the rows
themselves, so it round-trips to JSON and can be proposed before any data is
touched:

```python
ReferenceSpec(name, description, selector, params, source)
```

- **`selector`** names a registered row-selection rule; **`params`** are its
  arguments. Register new ones with `@register_selector(name)`.
- Built-in selectors (`available_selectors()`):
  - `all` — the whole source (the first-cut behaviour; `all_training_reference()`
    returns this spec).
  - `by_label` — rows where the target equals a value (all-fraud / all-legit).
  - `feature_threshold` — rows passing a raw-column comparison (`column/op/value`).
  - `uncertainty_quantile` — **the "all-weird / all-perfect" reference**: score
    every row with one MC-dropout pass and keep a tail by quantile. `params`:
    `metric` (default `bald`), `quantile` (default `0.95`), `side`
    (`upper`=most-uncertain/weird, `lower`=most-confident/perfect). Needs the
    fitted BDL `model`.

Resolution + capture:

```python
x_reference, provenance = resolve_reference(spec, train_df, TARGET, TIMESTAMP, model=bdl_model)
save_reference_report(provenance, report_dir)   # writes reference_spec.json
```

- `resolve_reference` applies the selector, filters, and builds `x_reference` via
  `features_and_target` — so it has the **same column layout** as the training /
  test `X` and drops straight into `compare_models_on_test`.
- `provenance` records the spec, source/selected row counts, coverage, and a
  resolution timestamp. `compare_models_on_test(..., reference_provenance=...)`
  now embeds it under a `"reference"` key in `test_comparison.json`, and
  `reference_spec.json` is written alongside. **Every comparison is now traceable
  to the subset it was calibrated against.**

`select_mask(spec, df, model)` is the shared low-level primitive (mask over the
source rows) used by both resolution and the finder.

## Part 2 — `src/utils/subset_finder.py` (scaffold)

`find_subsets_of_interest(df, model, strategies, ..., dirpath, with_uncertainty)`
enumerates **strategies**, each proposing `ReferenceSpec`s; resolves each
(skipping proposals that match nothing); computes diagnostics; scores; ranks; and
writes `subsets_of_interest.json`. Each candidate's `spec` is ready to copy into
`main.py`'s reference step to actually use it.

- Strategies (`available_strategies()`): `by_label` (concrete), and
  `uncertainty_extremes` (sweeps upper/lower quantiles of an uncertainty metric →
  all-weird / all-perfect candidates). Register more with
  `@register_strategy(name)`.
- Diagnostics: sizes, coverage, base vs subset fraud rate, and — only when
  `with_uncertainty=True` — mean BALD over the subset (one extra MC pass each).

### What is real vs. TODO

Real: the full enumerate → resolve → diagnose → rank → report loop, the
label/threshold path, and the uncertainty-quantile selection.

**Open (clearly marked in code):**
- `_score_candidate` — the interestingness score is a **placeholder** (mean BALD
  if available, else departure from the base fraud rate). `subsets_of_interest.json`
  sets `"score_is_placeholder": true`. Replace with a defensible criterion:
  distributional distance of the subset from the bulk (MMD / energy distance in
  the embedding or uncertainty space), a held-out reject-mask quality metric, or
  cluster silhouette.
- `_strategy_uncertainty_extremes` — a fixed sweep, not *discovery*. The real task
  is choosing the metric/cut from the data (knee/second-mode in the metric's
  distribution; BALD vs predictive_entropy vs epistemic_var for separation), and a
  clustering-based proposer over the uncertainty / penultimate-layer space.
- Use **BALD**, not Bayes Error, as the default OOD signal — §5 shows Bayes Error
  collapses to the mean and carries no disagreement signal. (The selector default
  is already `bald`; the *reject mask itself* in `pipeline_tools` still thresholds
  Bayes Error — switching it to BALD is the matching next step there.)

## Wiring in `main.py`

The test step now: builds the default `all_training_reference()` spec → resolves
it (behaviour-identical to the old `x_reference = x`) → saves `reference_spec.json`
→ passes both `x_reference` and `reference_provenance` to `compare_models_on_test`.
A commented `find_subsets_of_interest(...)` block sits above it (off by default —
its model-driven strategies run MC passes over all of `train_df`; enable while
exploring, ideally on a subset). To calibrate against a subset of interest, swap
the spec (hand-written or promoted from the finder).

## New report artifacts

- `reference_spec.json` — the resolved reference provenance for the run.
- `test_comparison.json` — now carries a `"reference"` key (the same provenance).
- `subsets_of_interest.json` — ranked candidates (only when the finder is run).
