"""Specify, resolve, and record the BDL reject-mask reference set (``x_reference``).

The reject mask compares each test datum's uncertainty against a single scalar
computed from a *reference set* (``x_reference`` in ``pipeline_tools``). The first
cut hard-coded that reference to "all of training" (``main.py`` passed the whole
training matrix ``x``). This module turns the reference set into a *declared,
serializable* object — a :class:`ReferenceSpec` — so it can be:

* **defined** as a semantically-meaningful subset rather than "everything" (e.g.
  the "weirdest"/"most-confident" transactions — the "all-weird / all-perfect"
  reference discussed in ``notes/bdl_imbalance_calibration.md`` §5), and
* **captured in the report** for provenance, via :func:`save_reference_report`
  (writes ``reference_spec.json`` next to ``test_comparison.json``).

A spec names a *selector* (a registered row-selection rule) plus its ``params``.
:func:`resolve_reference` applies the selector to a source polars frame and
returns the pandas ``x_reference`` matrix (same column layout the training/test
``X`` uses, via :func:`features_and_target`) together with a provenance dict.

This is the resolution half of the workflow. The discovery half — *finding* which
subset to use — lives in ``subset_finder.py``, which proposes ``ReferenceSpec``
objects that resolve through here. New selectors register via
:func:`register_selector`, so a finder strategy and its matching selector can be
added together.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import polars as pl

import constants
from utils.data_handlers import features_and_target

if TYPE_CHECKING:
    from pathlib import Path

    from pandas import DataFrame as pdDataFrame
    from sklearn.pipeline import Pipeline


# A selector decides, for each row of the source frame, whether it belongs to the
# reference subset. It returns a boolean mask aligned to the *current* row order
# of ``df`` (resolution sorts afterwards), and may use the fitted BDL ``model`` to
# select on predictive uncertainty (``None`` for purely data-defined selectors).
ReferenceSelector = Callable[
    [pl.DataFrame, "Pipeline | None", dict[str, Any]], np.ndarray
]

_SELECTOR_REGISTRY: dict[str, ReferenceSelector] = {}


def register_selector(name: str) -> Callable[[ReferenceSelector], ReferenceSelector]:
    """Register a reference-subset selector under ``name``.

    Args:
        name: The key a :class:`ReferenceSpec` uses to look this selector up.

    Returns:
        A decorator that registers and returns the selector unchanged.

    Raises:
        ValueError: If ``name`` is already registered.

    """

    def decorator(fn: ReferenceSelector) -> ReferenceSelector:
        if name in _SELECTOR_REGISTRY:
            msg = f"Selector {name!r} is already registered."
            raise ValueError(msg)
        _SELECTOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_selector(name: str) -> ReferenceSelector:
    """Look up a registered selector by name.

    Args:
        name: The selector key.

    Returns:
        The registered selector callable.

    Raises:
        KeyError: If no selector is registered under ``name``.

    """
    try:
        return _SELECTOR_REGISTRY[name]
    except KeyError:
        msg = f"Unknown selector {name!r}. Available: {sorted(_SELECTOR_REGISTRY)}."
        raise KeyError(msg) from None


def available_selectors() -> list[str]:
    """Return the names of all registered selectors."""
    return sorted(_SELECTOR_REGISTRY)


@dataclass
class ReferenceSpec:
    """A declarative, serializable definition of a reject-mask reference set.

    The spec says *how* to select the reference rows (``selector`` + ``params``),
    not the rows themselves, so it round-trips to JSON for the report and can be
    proposed by the subset finder before any data is touched.

    Attributes:
        name: Short identifier used in report keys/filenames (keep filesystem- and
            JSON-key-safe).
        description: Human-readable meaning of the subset ("all training",
            "top 5% highest-BALD (most OOD)", ...).
        selector: Key of a registered selector (see :func:`available_selectors`).
        params: Keyword arguments handed to the selector.
        source: Which dataset the subset is drawn from ("train" by default). Free-
            form provenance label; resolution is against whatever frame is passed.

    """

    name: str
    description: str = ""
    selector: str = "all"
    params: dict[str, Any] = field(default_factory=dict)
    source: str = "train"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of this spec."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReferenceSpec:
        """Rebuild a spec from a dict produced by :meth:`to_dict`."""
        fields = {"name", "description", "selector", "params", "source"}
        return cls(**{k: v for k, v in data.items() if k in fields})


def all_training_reference() -> ReferenceSpec:
    """Return the default spec: the entire source (the first-cut behaviour).

    Resolving this against ``train_df`` reproduces exactly what ``main.py`` used to
    pass as ``x_reference`` (all of training), now named and recorded in the report
    instead of being implicit.
    """
    return ReferenceSpec(
        name="all_training",
        description="All training transactions (first-cut reference).",
        selector="all",
        source="train",
    )


def _features_only(df: pl.DataFrame, target: str) -> pdDataFrame:
    """Drop the target and return a pandas matrix in ``df``'s current row order.

    Selectors that score rows through the model need the feature matrix aligned to
    ``df`` *as given* (no sort), so the returned mask lines up with ``df``. The
    final, sorted ``x_reference`` is built separately by :func:`resolve_reference`
    via :func:`features_and_target` (matching the training/test layout).
    """
    return df.select(pl.all().exclude(target)).to_pandas()


def _model_uncertainty(
    model: Pipeline,
    df: pl.DataFrame,
    target: str,
    metric: str,
) -> np.ndarray:
    """Run one MC-dropout pass over ``df`` and return the chosen per-row metric.

    Mirrors how ``pipeline_tools.bayes_error_reference`` reaches the classifier:
    ``model[:-1].transform`` then ``model[-1].uncertainty_metrics`` (the Pipeline
    does not forward the custom method). The metric is aligned to ``df``'s current
    row order.
    """
    x = _features_only(df, target)
    x_pre = model[:-1].transform(x)
    metrics = model[-1].uncertainty_metrics(x_pre)
    if metric not in metrics:
        msg = f"Unknown uncertainty metric {metric!r}. Available: {sorted(metrics)}."
        raise KeyError(msg)
    return np.asarray(metrics[metric])


@register_selector("all")
def _select_all(
    df: pl.DataFrame,
    model: Pipeline | None,  # noqa: ARG001
    params: dict[str, Any],  # noqa: ARG001
) -> np.ndarray:
    """Select every row (reference = the whole source)."""
    return np.ones(df.height, dtype=bool)


@register_selector("by_label")
def _select_by_label(
    df: pl.DataFrame,
    model: Pipeline | None,  # noqa: ARG001
    params: dict[str, Any],
) -> np.ndarray:
    """Select rows whose target column equals ``params['value']``.

    Lets the reference be "all fraud" or "all legit" — a purely data-defined,
    label-conditioned subset. ``params``: ``value`` (required), ``target``
    (defaults to ``constants.TARGET``).
    """
    target = params.get("target", constants.TARGET)
    value = params["value"]
    return df.get_column(target).eq(value).to_numpy()


_COMPARATORS: dict[str, Callable[[pl.Series, Any], pl.Series]] = {
    "==": lambda s, v: s.eq(v),
    "!=": lambda s, v: s.ne(v),
    "<": lambda s, v: s.lt(v),
    "<=": lambda s, v: s.le(v),
    ">": lambda s, v: s.gt(v),
    ">=": lambda s, v: s.ge(v),
}


@register_selector("feature_threshold")
def _select_feature_threshold(
    df: pl.DataFrame,
    model: Pipeline | None,  # noqa: ARG001
    params: dict[str, Any],
) -> np.ndarray:
    """Select rows by comparing a raw feature column against a threshold.

    A purely data-defined slice (e.g. very large ``TransactionAmt``). ``params``:
    ``column`` (required), ``value`` (required), ``op`` (one of ``==``, ``!=``,
    ``<``, ``<=``, ``>``, ``>=``; default ``>=``). Nulls compare as ``False``.
    """
    column = params["column"]
    value = params["value"]
    op = params.get("op", ">=")
    if op not in _COMPARATORS:
        msg = f"Unknown op {op!r}. Available: {sorted(_COMPARATORS)}."
        raise KeyError(msg)
    mask = _COMPARATORS[op](df.get_column(column), value)
    return mask.fill_null(value=False).to_numpy()


@register_selector("uncertainty_quantile")
def _select_uncertainty_quantile(
    df: pl.DataFrame,
    model: Pipeline | None,
    params: dict[str, Any],
) -> np.ndarray:
    """Select the most- (or least-) uncertain rows by a BDL uncertainty metric.

    This is the "all-weird / all-perfect" reference from
    ``notes/bdl_imbalance_calibration.md`` §5, made concrete: score every row with
    one MC-dropout pass, then keep a tail by quantile.

    ``params``:
        metric: Which ``uncertainty_metrics`` key to threshold. Default ``"bald"``
            — the epistemic/disagreement signal §5 recommends for OOD/"weird"
            detection (Bayes Error carries no disagreement signal). Any key works:
            ``bald``, ``bayes_error``, ``predictive_entropy``, ``epistemic_var``,
            ``aleatoric``.
        quantile: The cut percentile in [0, 1]. Default ``0.95``.
        side: ``"upper"`` keeps rows ``>=`` the ``quantile``-th percentile (the
            most-uncertain tail → "all-weird"); ``"lower"`` keeps rows ``<=`` it
            (the most-confident tail → "all-perfect"). Default ``"upper"``.
        target: Target column to drop before inference (default
            ``constants.TARGET``).

    Requires a fitted BDL ``model`` (raises if ``None``).
    """
    if model is None:
        msg = "uncertainty_quantile selector requires a fitted BDL model."
        raise ValueError(msg)
    metric = params.get("metric", "bald")
    quantile = params.get("quantile", 0.95)
    side = params.get("side", "upper")
    target = params.get("target", constants.TARGET)

    values = _model_uncertainty(model, df, target, metric)
    threshold = float(np.quantile(values, quantile))
    if side == "upper":
        return values >= threshold
    if side == "lower":
        return values <= threshold
    msg = f"Unknown side {side!r}; expected 'upper' or 'lower'."
    raise ValueError(msg)


def select_mask(
    spec: ReferenceSpec,
    df: pl.DataFrame,
    model: Pipeline | None = None,
) -> np.ndarray:
    """Resolve a spec to a boolean row mask over ``df`` (current row order).

    The low-level primitive shared by :func:`resolve_reference` (which builds the
    matrix) and the subset finder's diagnostics (which only need the mask).

    Args:
        spec: The reference specification.
        df: The source polars frame.
        model: The fitted BDL pipeline, for selectors that score uncertainty.

    Returns:
        A boolean ``np.ndarray`` of length ``df.height``.

    Raises:
        ValueError: If the selector returns a mask of the wrong length.

    """
    selector = get_selector(spec.selector)
    mask = np.asarray(selector(df, model, spec.params), dtype=bool)
    if mask.shape[0] != df.height:
        msg = (
            f"Selector {spec.selector!r} returned a mask of length "
            f"{mask.shape[0]}, expected {df.height}."
        )
        raise ValueError(msg)
    return mask


def resolve_reference(
    spec: ReferenceSpec,
    df: pl.DataFrame,
    target: str = constants.TARGET,
    timestamp: str = constants.TIMESTAMP,
    model: Pipeline | None = None,
) -> tuple[pdDataFrame, dict[str, Any]]:
    """Resolve a :class:`ReferenceSpec` into an ``x_reference`` matrix + provenance.

    Applies the spec's selector to ``df``, filters to the selected rows, and builds
    the feature matrix with :func:`features_and_target` so ``x_reference`` has the
    **same column layout** the training/test ``X`` uses (target excluded, sorted by
    ``timestamp``, ``TransactionDT`` retained). The returned matrix is what
    ``pipeline_tools.bayes_error_reference`` / ``compare_models_on_test`` expect as
    ``x_reference``; the provenance dict is what :func:`save_reference_report`
    records.

    Args:
        spec: The reference specification.
        df: The source polars frame (typically ``train_df``).
        target: The target column name.
        timestamp: The column to sort by (kept in ``X``, like training).
        model: The fitted BDL pipeline, required only by model-driven selectors.

    Returns:
        ``(x_reference, provenance)``. ``provenance`` carries the spec, the source
        and selected row counts, the coverage, and a resolution timestamp.

    Raises:
        ValueError: If the selector matches zero rows (no reference to compute).

    """
    mask = select_mask(spec, df, model)
    n_selected = int(mask.sum())
    if n_selected == 0:
        msg = (
            f"Reference spec {spec.name!r} (selector {spec.selector!r}) "
            "matched 0 rows; cannot build a reference set."
        )
        raise ValueError(msg)

    subset = df.filter(pl.Series(values=mask))
    x_reference, _ = features_and_target(subset, target, timestamp)

    provenance = {
        "spec": spec.to_dict(),
        "source": spec.source,
        "source_rows": df.height,
        "selected_rows": n_selected,
        "coverage": float(n_selected / df.height),
        "resolved_at": datetime.now(tz=UTC).isoformat(),
    }
    return x_reference, provenance


# Filename used to persist the resolved reference provenance into a report dir.
_REFERENCE_FILENAME = "reference_spec.json"


def save_reference_report(provenance: dict[str, Any], dirpath: Path) -> None:
    """Write the resolved-reference provenance to ``reference_spec.json``.

    Captures *which* reference set fed the reject mask alongside the run's other
    artifacts, so a ``test_comparison.json`` can always be traced back to the
    subset it was calibrated against.

    Args:
        provenance: The dict returned by :func:`resolve_reference`.
        dirpath: Directory to write the report into (usually the run's report dir).

    """
    dirpath.mkdir(parents=True, exist_ok=True)
    with (dirpath / _REFERENCE_FILENAME).open("w") as f:
        json.dump(provenance, f, indent=4)
