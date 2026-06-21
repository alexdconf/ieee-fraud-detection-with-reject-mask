"""Find candidate reference subsets ("subsets of interest") for the reject mask.

The reject mask is only as meaningful as the reference set it calibrates against
(``reference_sets.py``). This module is the *discovery* half: instead of the user
hand-writing a :class:`~utils.reference_sets.ReferenceSpec`, a set of **strategies**
each propose candidate specs, those candidates are resolved and summarised, and the
finder ranks them by an "interestingness" score so the promising ones surface.

Concretely a run produces ``subsets_of_interest.json`` — a ranked list of
candidates, each with the spec needed to *use* it as ``x_reference`` plus
diagnostics (size, coverage, label mix, optionally mean uncertainty). Promote a
candidate by copying its ``spec`` into ``main.py``'s reference step.

Scaffold status
---------------
The plumbing is real and runs end-to-end:

* the label/threshold strategies enumerate concrete, usable specs;
* every candidate is resolved through :func:`~utils.reference_sets.resolve_reference`
  so a proposal that cannot build a reference is dropped, not crashed on;
* diagnostics and a ranked report are written.

The genuinely open research lives in two clearly-marked spots:

* :func:`_strategy_uncertainty_extremes` — *which* uncertainty metric and quantile
  actually isolate OOD/"weird" transactions (start from BALD per
  ``notes/bdl_imbalance_calibration.md`` §5, but the data-driven choice is TODO),
  and richer proposals (clustering the embedding/uncertainty space) are not done.
* :func:`_score_candidate` — the "interestingness" score is a **placeholder**
  heuristic. Replace it with a real separation/OOD criterion before trusting the
  ranking. See its TODO.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import polars as pl

import constants
from utils.reference_sets import (
    ReferenceSpec,
    _model_uncertainty,
    resolve_reference,
    select_mask,
)

if TYPE_CHECKING:
    from pathlib import Path

    from sklearn.pipeline import Pipeline


# A strategy proposes candidate reference specs given the source frame and
# (optionally) the fitted BDL model. It only *proposes* — resolution, diagnostics,
# and scoring happen in find_subsets_of_interest. ``params`` lets a caller tune the
# strategy (quantile sweep, target column, ...).
SubsetStrategy = Callable[
    ["pl.DataFrame", "Pipeline | None", dict[str, Any]], "list[ReferenceSpec]"
]

_STRATEGY_REGISTRY: dict[str, SubsetStrategy] = {}


def register_strategy(name: str) -> Callable[[SubsetStrategy], SubsetStrategy]:
    """Register a subset-proposing strategy under ``name``.

    Raises:
        ValueError: If ``name`` is already registered.

    """

    def decorator(fn: SubsetStrategy) -> SubsetStrategy:
        if name in _STRATEGY_REGISTRY:
            msg = f"Strategy {name!r} is already registered."
            raise ValueError(msg)
        _STRATEGY_REGISTRY[name] = fn
        return fn

    return decorator


def available_strategies() -> list[str]:
    """Return the names of all registered strategies."""
    return sorted(_STRATEGY_REGISTRY)


@dataclass
class SubsetCandidate:
    """A proposed reference subset with diagnostics and a ranking score.

    Attributes:
        spec: The reusable :class:`~utils.reference_sets.ReferenceSpec`. Copy this
            into the reference step to actually use the subset.
        diagnostics: Summary stats for the subset (counts, coverage, label mix and,
            when computed, mean uncertainty).
        score: The (currently placeholder) interestingness score used to rank.

    """

    spec: ReferenceSpec
    diagnostics: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of this candidate."""
        return {
            "score": self.score,
            "spec": self.spec.to_dict(),
            "diagnostics": self.diagnostics,
        }


@register_strategy("by_label")
def _strategy_by_label(
    df: pl.DataFrame,
    model: Pipeline | None,  # noqa: ARG001
    params: dict[str, Any],
) -> list[ReferenceSpec]:
    """Propose one reference per distinct target value (all-fraud, all-legit).

    Fully concrete: a label-conditioned reference is a natural "subset of interest"
    and needs no model. ``params``: ``target`` (default ``constants.TARGET``).
    """
    target = params.get("target", constants.TARGET)
    specs: list[ReferenceSpec] = []
    for value in sorted(df.get_column(target).unique().to_list()):
        specs.append(
            ReferenceSpec(
                name=f"label_{target}_{value}",
                description=f"All rows where {target} == {value}.",
                selector="by_label",
                params={"target": target, "value": value},
            )
        )
    return specs


@register_strategy("uncertainty_extremes")
def _strategy_uncertainty_extremes(
    df: pl.DataFrame,  # noqa: ARG001
    model: Pipeline | None,
    params: dict[str, Any],
) -> list[ReferenceSpec]:
    """Propose "all-weird" (upper tail) and "all-perfect" (lower tail) references.

    Enumerates a small quantile sweep on one uncertainty metric (default BALD, the
    epistemic signal §5 recommends over Bayes Error). Each proposal is a usable
    ``uncertainty_quantile`` spec.

    TODO (open research): this is a fixed sweep, not *discovery*. The real task is
    choosing the metric and cut from the data — e.g. find the quantile where the
    metric's distribution has a knee/second mode (OOD cluster), or compare BALD vs
    predictive_entropy vs epistemic_var for separation. Clustering the uncertainty
    (or penultimate-layer) space into coherent "weird" groups would be a stronger
    proposer and belongs here too. Returns ``[]`` when no model is available.

    ``params``: ``metric`` (default ``"bald"``), ``upper_quantiles``
    (default ``(0.9, 0.95, 0.99)``), ``lower_quantiles`` (default
    ``(0.1, 0.05, 0.01)``).
    """
    if model is None:
        return []
    metric = params.get("metric", "bald")
    upper = params.get("upper_quantiles", (0.9, 0.95, 0.99))
    lower = params.get("lower_quantiles", (0.1, 0.05, 0.01))

    specs: list[ReferenceSpec] = []
    for q in upper:
        specs.append(
            ReferenceSpec(
                name=f"weird_{metric}_q{q}",
                description=f"Top tail: {metric} >= its {q:.0%} quantile (most OOD).",
                selector="uncertainty_quantile",
                params={"metric": metric, "quantile": q, "side": "upper"},
            )
        )
    for q in lower:
        specs.append(
            ReferenceSpec(
                name=f"perfect_{metric}_q{q}",
                description=f"Bottom tail: {metric} <= its {q:.0%} quantile (most confident).",
                selector="uncertainty_quantile",
                params={"metric": metric, "quantile": q, "side": "lower"},
            )
        )
    return specs


def _diagnostics(
    spec: ReferenceSpec,
    df: pl.DataFrame,
    provenance: dict[str, Any],
    model: Pipeline | None,
    target: str,
    with_uncertainty: bool,
) -> dict[str, Any]:
    """Summarise a resolved candidate subset.

    Cheap stats (sizes, coverage, fraud rates) always; the mean uncertainty over
    the subset only when ``with_uncertainty`` is set, since that costs an extra
    MC-dropout pass per candidate.
    """
    mask = select_mask(spec, df, model)
    subset = df.filter(pl.Series(values=mask))

    base_fraud = float(df.get_column(target).mean())
    subset_fraud = float(subset.get_column(target).mean()) if subset.height else None

    diagnostics: dict[str, Any] = {
        "selected_rows": provenance["selected_rows"],
        "source_rows": provenance["source_rows"],
        "coverage": provenance["coverage"],
        "base_fraud_rate": base_fraud,
        "subset_fraud_rate": subset_fraud,
    }

    # Optional, expensive: mean epistemic uncertainty (BALD) over the subset. One
    # extra MC pass per candidate, so it is opt-in (see find_subsets_of_interest).
    if with_uncertainty and model is not None:
        bald = _model_uncertainty(model, subset, target, "bald")
        diagnostics["mean_bald"] = float(np.mean(bald))

    return diagnostics


def _score_candidate(diagnostics: dict[str, Any]) -> float:
    """Rank a candidate by "interestingness".

    PLACEHOLDER. The real question — does this subset isolate the OOD/"weird"
    transactions the reject mask should calibrate against? — is open. The provisional
    heuristic below is just enough to produce a non-arbitrary ordering for the
    scaffold:

    * if mean epistemic uncertainty was computed, rank by it (more disagreement =
      more OOD-like = a more interesting "all-weird" reference);
    * otherwise rank by how far the subset's fraud rate departs from the base rate
      (a strongly label-skewed slice is at least a meaningful, non-random subset).

    TODO: replace with a defensible criterion — e.g. distributional distance of the
    subset from the bulk (MMD/energy distance in the embedding or uncertainty
    space), a held-out reject-mask quality metric, or silhouette of a discovered
    cluster. Until then, treat the ranking as a rough prioritisation, not a verdict.
    """
    if diagnostics.get("mean_bald") is not None:
        return float(diagnostics["mean_bald"])
    base = diagnostics.get("base_fraud_rate")
    subset = diagnostics.get("subset_fraud_rate")
    if base is not None and subset is not None:
        return abs(subset - base)
    return 0.0


# Filename used to persist the ranked candidates into a report dir.
_FINDER_FILENAME = "subsets_of_interest.json"

# Strategies run by default when the caller does not name a subset. Both are safe
# to run without a model (uncertainty_extremes simply yields nothing then).
_DEFAULT_STRATEGIES = ("by_label", "uncertainty_extremes")


def find_subsets_of_interest(  # noqa: PLR0913
    df: pl.DataFrame,
    model: Pipeline | None = None,
    strategies: list[str] | None = None,
    target: str = constants.TARGET,
    timestamp: str = constants.TIMESTAMP,
    dirpath: Path | None = None,
    with_uncertainty: bool = False,
) -> list[SubsetCandidate]:
    """Enumerate, resolve, score, and rank candidate reference subsets.

    For each requested strategy, every proposed :class:`ReferenceSpec` is resolved
    against ``df`` (proposals that match zero rows or otherwise fail to resolve are
    skipped, not raised), summarised by :func:`_diagnostics`, and scored by the
    placeholder :func:`_score_candidate`. The candidates are returned sorted by
    score (descending) and, if ``dirpath`` is given, written to
    ``subsets_of_interest.json``.

    Note: model-driven strategies (and ``with_uncertainty`` diagnostics) each run
    MC-dropout passes over ``df`` / the subset, so on the full training set this is
    not cheap — run it on a subset while exploring (the user runs the full jobs).

    Args:
        df: The source polars frame to draw subsets from (typically ``train_df``).
        model: The fitted BDL pipeline; required for uncertainty-based strategies.
        strategies: Strategy names to run (default: label + uncertainty extremes).
        target: Target column name.
        timestamp: Timestamp column used to build the resolved matrices.
        dirpath: If given, where to write ``subsets_of_interest.json``.
        with_uncertainty: Compute mean-BALD diagnostics per subset (extra MC pass
            each). Off by default to keep the scan cheap.

    Returns:
        Candidates sorted by descending score. Each ``spec`` is ready to drop into
        the reference step to use that subset as ``x_reference``.

    """
    strategy_names = strategies if strategies is not None else list(_DEFAULT_STRATEGIES)

    candidates: list[SubsetCandidate] = []
    for name in strategy_names:
        strategy = _STRATEGY_REGISTRY[name]
        for spec in strategy(df, model, {}):
            try:
                _, provenance = resolve_reference(spec, df, target, timestamp, model)
            except ValueError, KeyError:
                # A proposal that selects nothing / references a bad column is not
                # a candidate — skip it rather than aborting the whole scan.
                continue
            diagnostics = _diagnostics(
                spec, df, provenance, model, target, with_uncertainty
            )
            candidates.append(
                SubsetCandidate(
                    spec=spec,
                    diagnostics=diagnostics,
                    score=_score_candidate(diagnostics),
                )
            )

    candidates.sort(key=lambda c: c.score, reverse=True)

    if dirpath is not None:
        dirpath.mkdir(parents=True, exist_ok=True)
        report = {
            "strategies": strategy_names,
            "score_is_placeholder": True,
            "candidates": [c.to_dict() for c in candidates],
        }
        with (dirpath / _FINDER_FILENAME).open("w") as f:
            json.dump(report, f, indent=4)

    return candidates
