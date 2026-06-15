# BDL: Class Imbalance, Probability Inflation, and Calibration

Notes from working on `MCDropoutClassifier` / `BDL_reference` in
`src/utils/pipeline_tools.py`. Context: IEEE fraud detection (~3.5% positives),
a PyTorch MC-dropout Bayesian model used with a reject/abstain mask.

## 1. Class imbalance options

On ~3.5% fraud, an unweighted `CrossEntropyLoss` lets the model minimize loss by
predicting "not fraud" for everyone. The model collapses to the majority class
and PR-AUC sits near the base rate — the signature of "awful results."

Ways to attack it:

- **Loss weighting** (`class_weight` / `pos_weight`) — re-scale the loss so the
  rare class costs more. Training-side fix.
- **Resampling** — over/undersample or SMOTE (lives in preprocessing).
- **Threshold moving** — train unweighted, keep probabilities calibrated, just
  move the decision threshold below 0.5.

You generally pick one, not stack them. Weighting and resampling both work by
effectively changing the prior the model trains on (~3.5% → ~50/50).

### class_weight vs pos_weight (which fits our code)

- **`class_weight` → `CrossEntropyLoss(weight=...)`**: per-class weight tensor.
  Fits our architecture (n_classes logits + softmax + cross-entropy). **This is
  the route we used.**
- **`pos_weight` → `BCEWithLogitsLoss(pos_weight=...)`**: single scalar for a
  1-logit binary head. Does NOT plug into our CrossEntropyLoss path.

"Balanced" weight recipe (same as sklearn `class_weight="balanced"`):

    weight[c] = n_samples / (n_classes * count[c])

For binary, `pos_weight` would be `n_neg / n_pos` (≈ 27 here).

### Verified effect

- `class_weight=None`  → 0 predicted positives, PR-AUC ≈ 0.025 (collapse).
- `class_weight="balanced"` → learns the minority class, PR-AUC ≈ 0.523.

## 2. class_weight is a training-only mechanic

- It enters **only** in the loss during `fit`. `predict_proba` never touches it.
- BUT its effect is **baked into the learned weights**. You can't un-inflate an
  already-weighted model with an inference flag — you retrain unweighted or
  correct after the fact.

## 3. Side effect: inflated fraud probabilities

Balanced weighting trains as if classes were ~50/50, so the softmax estimates
P(fraud) **under that artificial prior** → predicted fraud probabilities come
out **systematically too high**. A "0.7" no longer means 70% of such cases are
fraud.

Key fact — the bias is a **constant shift in logit space**, nonlinear in
probability space:

    logit(p_inflated) = logit(p_true) + log(n_neg / n_pos)   # ≈ +3.3 here

PR-AUC is a **ranking** metric → unaffected by this level shift, so the scorer
never reveals the damage.

This is a side effect of the *weighting/resampling family* specifically.
Threshold-moving and post-hoc calibration do not inflate.

## 4. Calibration

**Calibration** = do predicted probabilities match observed frequencies (among
cases scored p≈0.8, ~80% are actually fraud).

- **Check**: reliability/calibration curve (`sklearn.calibration.calibration_curve`),
  ECE, Brier score — on a held-out fold.
- **Fix options**:
  - **Logit prior-correction** (analytic): subtract `log(n_neg/n_pos)` from the
    logits (equivalently add `log(true_prior_c)` per class). Exact for the bias
    *we* introduced, needs no held-out data, OOD-safe.
  - **`CalibratedClassifierCV`** (Platt/isotonic): more general, but needs a
    held-out set AND won't transfer to deliberately-weird OOD holdout inputs.

Matters for the reject mask: rejection thresholds on probability/uncertainty
only mean what we intend if the probabilities are honest.

## 5. The reject-mask workflow and the "Bayes Error" metric

Workflow: train BDL → infer a semantically-meaningful holdout (all-weird or
all-perfect) and compute a "Bayes Error" reference → for a candidate datum, run
N MC-dropout passes and compute its Bayes Error → compare and decide.

Metric chosen: `E[1 - P(C|x)]` (expectation, over outputs, of one minus the
chosen class's probability).

**Critical consequence — linearity of expectation collapses it:**

    E_i[1 - P_i(C|x)] = 1 - E_i[P_i(C|x)] = 1 - p̄(C|x)   # = min(p̄,1-p̄) binary

So the N passes only estimate the **mean**; the **variance/disagreement across
passes (epistemic uncertainty) is averaged away.** Two inputs with the same mean
score identically even if one had passes tightly agreeing and the other wildly
disagreeing.

Implications:
- This is the **mean-probability flavor**, i.e. the one **distorted nonlinearly**
  by the inflation — not the robust MC-disagreement flavor.
- Because `min(p,1-p)` is non-monotonic, the inflation **relocates** the
  max-uncertainty point from `p_true=0.5` to `p_true ≈ base rate`. A genuinely
  50/50 transaction scores as confident-fraud — backwards for rejection.
- Common-mode cancellation (reference + datum distorted the same way) gives
  partial cover, but it's **weakest in the tails** where the weird holdout lives.
- Watch `C`: argmax-of-mean → `1 - p̄(C)` (total uncertainty, responds to
  disagreement via the mean). Per-pass argmax → can actively **hide** epistemic
  uncertainty (passes flipping 0.99/0.01 score as confident).

**If epistemic signal is wanted**: use a disagreement measure like mutual
information / BALD: `H(p̄) − E_i[H(p_i)]`. Near-zero when passes agree, large when
they disagree → the right signal for OOD/"weird input" rejection.

Note on robustness to the weighting bias: BALD is **far less** sensitive to the
prior shift than the level-based metrics, but **not strictly invariant**. The
constant offset leaves disagreement unchanged in *logit* space, but BALD measures
it after the softmax/entropy nonlinearity, which the offset does affect. Measured:
toggling the correction moved mean P(fraud) 0.39→0.039 but BALD by ≤0.036 nats.

## 6. Decision: calibrate after the fact (what we implemented)

Chose the **analytic logit prior-correction** over `CalibratedClassifierCV`
(easier, exact for the introduced bias, no held-out data, OOD-safe).

Implementation in `MCDropoutClassifier`:
- `prior_correction: bool = True` constructor arg.
- `fit`: when `class_weight="balanced"`, store `self.log_prior_ = log(counts/N)`;
  else `None`.
- `predict_proba`: add `log_prior_` to logits before softmax on every MC pass.

Verified: mean P(fraud) goes 0.39 (inflated) → 0.039 ≈ 0.038 base rate.

## 7. Returning a menu of metrics for the workflow

Rather than baking one reject metric in, expose several and let the workflow
decide. `MCDropoutClassifier.uncertainty_metrics(X)` returns, from a *single*
N-pass run (so they're mutually consistent), all computed on the prior-corrected
probabilities:

- `mean_proba`          — predictive mean `p̄` (n_samples, n_classes)
- `bayes_error`         — `1 - max_c p̄(c)` (total uncertainty, 0-1 flavor)
- `predictive_entropy`  — `H(p̄)` (total uncertainty, entropy flavor)
- `aleatoric`           — `E_i[H(p_i)]`
- `bald`                — `H(p̄) - E_i[H(p_i)]` (epistemic / mutual information)
- `epistemic_var`       — mean over classes of per-pass probability variance

What does NOT break, and the constraints:
- **Do not change `predict_proba`** — it must stay `(n_samples, n_classes)` for
  the `RandomizedSearchCV` scorer. The menu lives in a *separate* method.
- **Pipeline does not forward custom methods.** Call via
  `Xt = pipe[:-1].transform(X); pipe[-1].uncertainty_metrics(Xt)`.
- Compute all metrics in one MC loop so the means behind each are identical
  (separate calls re-sample dropout → inconsistent).

Verified: `predictive_entropy == bald + aleatoric`, `bald >= 0`,
`bayes_error == 1 - max(mean_proba)`.

### Caveat on the monotonicity claim

Earlier claim: "monotonic shift → PR-AUC exactly unchanged." True for a single
deterministic forward pass. With **N averaged stochastic softmax passes** it does
NOT hold exactly — `mean_i sigmoid(z_i + c)` is not a strictly monotonic function
of `mean_i sigmoid(z_i)`. Measured (identical dropout masks): Spearman rank corr
≈ 0.99987, PR-AUC 0.4636 → 0.4695. So PR-AUC is **essentially** unchanged (well
within MC noise), **not provably invariant**.

### Still open / not done

- Calibration does not recover epistemic uncertainty; only BALD-style measures
  do. The current `1 - p̄(C)` metric still discards MC disagreement.
- Could add a calibration check (reliability curve + Brier) on a fold.
- `class_weight` and `prior_correction` are both sweepable in the grid if wanted.
