"""Object and time splits for training, development and held-out evaluation.

Why a module of its own (and not a section of ``catalog.py``)
-------------------------------------------------------------
``catalog.py`` decides which element sets are usable at all; this module
decides who may *look* at them. The split is an instrument that guards the
held-out data, is driven by its own committed config
(``config/splits.toml``) and has its own test file, so a change to catalog
loading can never move an object across the held-out boundary by accident,
and the split can be reviewed on its own.

The locked rule (planning package, section 2): split by object and by time.
Test objects never appear in training; test epochs come after training epochs.

How it works
------------
1. **Objects.** Every NORAD ID gets a pseudo-random draw in [0, 1) from
   SHA-256 of ``"<seed>:<id>"``. The draw depends only on the seed and the ID,
   not on which other objects are in the catalog, so adding or removing objects
   never moves an existing object between sets. IDs listed as forced (the
   validation-case objects) are always held out. Of the rest, a draw below
   ``heldout_object_fraction`` holds the object out; a second, independent draw
   puts a ``dev_object_fraction`` share of the remaining objects in ``dev``.
2. **Rows.** Each (object, start epoch) row is labeled from its object's role,
   its start epoch and its prediction window ``[start, start + max_horizon]``:

   ================  =========================  ============================
   object role       start < cutoff             start >= cutoff
   ================  =========================  ============================
   held out          ``reserved_object``        ``test``
   train / dev       ``train`` / ``dev``, or    ``reserved_time``
                     ``excluded`` (see below)
   ================  =========================  ============================

   A training or dev row is ``excluded`` if its window reaches past the cutoff
   (so no training target lies at or after the cutoff), or if its window
   overlaps a validation event's exclusion window.
3. **Data end.** A row of *any* role is ``excluded`` if its window ends after
   the last fully observed space-weather day (the end of that UTC day). Past
   it the space-weather file holds only forecasts, so truth propagated there
   would be driven by predictions rather than measurements. The day comes
   from :meth:`orbit_correction.spaceweather.SpaceWeather.last_fully_observed_date`
   at labeling time and should be recorded with any split that is generated.

What the labels allow
---------------------
* ``train`` rows are the only rows a model is fitted on.
* ``dev`` rows are for development evaluation (tuning, model choice).
* ``test`` rows are the held-out evaluation set, left unexamined until a
  confirmatory run.
* ``reserved_object`` and ``reserved_time`` rows enter **neither training nor
  evaluation**. They are held out on one axis only (a held-out object before
  the cutoff, or a training object after it), so using them for training would
  leak held-out objects or epochs, and using them for evaluation would mix
  partly seen data into the held-out result.
* ``excluded`` rows are not used at all.

Nothing here reads data: it only labels IDs and epochs it is given.
"""

from __future__ import annotations

import hashlib
import tomllib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orbit_correction.frames import utc_to_tt_seconds
from orbit_correction.types import SPLIT_LABELS

_DAY_S = 86400.0


@dataclass(frozen=True)
class ValidationEvent:
    """A validation case: a close approach whose objects are held out of training."""

    name: str
    tca_utc: str
    norad_ids: tuple[int, ...]

    @property
    def tca_tt_s(self) -> float:
        return float(utc_to_tt_seconds(self.tca_utc))


@dataclass(frozen=True)
class SplitConfig:
    """Split parameters. Values are instruments; see ``config/splits.toml``.

    Attributes:
        status: ``"approved <date>"`` once Kaitlin approves the values.
        seed: seed for the object draws.
        heldout_object_fraction: share of (non-forced) objects held out.
        dev_object_fraction: share of the remaining objects used for development
            evaluation instead of training.
        time_cutoff_utc: ISO UTC instant separating training epochs from test epochs.
        max_horizon_days: prediction window length; a training window must end
            at or before the cutoff.
        forced_heldout_norad_ids: always held out (validation-case objects).
        event_exclusion_days_before, event_exclusion_days_after: training and dev
            windows overlapping ``[tca - before, tca + after]`` of any validation
            event are excluded. 0 and 0 turn the rule off except at the TCA itself.
        validation_events: the events, for the exclusion rule and the record.
    """

    status: str
    seed: int
    heldout_object_fraction: float
    dev_object_fraction: float
    time_cutoff_utc: str
    max_horizon_days: float
    forced_heldout_norad_ids: tuple[int, ...]
    event_exclusion_days_before: float
    event_exclusion_days_after: float
    validation_events: tuple[ValidationEvent, ...]

    def __post_init__(self) -> None:
        for name in ("heldout_object_fraction", "dev_object_fraction"):
            v = getattr(self, name)
            if not 0.0 <= v < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {v}")
        if self.max_horizon_days <= 0:
            raise ValueError("max_horizon_days must be positive")
        if self.event_exclusion_days_before < 0 or self.event_exclusion_days_after < 0:
            raise ValueError("event exclusion widths must be >= 0")
        missing = {i for e in self.validation_events for i in e.norad_ids} - set(
            self.forced_heldout_norad_ids
        )
        if missing:
            raise ValueError(f"validation-event objects not in forced_heldout_norad_ids: {missing}")

    @property
    def time_cutoff_tt_s(self) -> float:
        return float(utc_to_tt_seconds(self.time_cutoff_utc))


def load_split_config(path: str | Path) -> SplitConfig:
    """Read ``[splits]`` and its ``[[splits.validation_events]]`` from a TOML file."""
    with open(path, "rb") as f:
        raw = dict(tomllib.load(f)["splits"])
    events = tuple(
        ValidationEvent(e["name"], e["tca_utc"], tuple(int(i) for i in e["norad_ids"]))
        for e in raw.pop("validation_events", [])
    )
    raw["forced_heldout_norad_ids"] = tuple(int(i) for i in raw["forced_heldout_norad_ids"])
    return SplitConfig(validation_events=events, **raw)


# --------------------------------------------------------------------------------------
# Object roles
# --------------------------------------------------------------------------------------


def object_draw(norad_id: int, seed: int, salt: str = "heldout") -> float:
    """Deterministic pseudo-random number in [0, 1) for one object.

    SHA-256 of ``"<seed>:<salt>:<norad_id>"``, first 8 bytes as an unsigned
    integer divided by 2**64. Independent of every other object.
    """
    digest = hashlib.sha256(f"{seed}:{salt}:{int(norad_id)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def object_roles(norad_ids: Iterable[int], cfg: SplitConfig) -> dict[int, str]:
    """Map each NORAD ID to ``"heldout"``, ``"dev"`` or ``"train"``."""
    forced = set(cfg.forced_heldout_norad_ids)
    roles: dict[int, str] = {}
    for nid in {int(i) for i in norad_ids}:
        if nid in forced or object_draw(nid, cfg.seed, "heldout") < cfg.heldout_object_fraction:
            roles[nid] = "heldout"
        elif object_draw(nid, cfg.seed, "dev") < cfg.dev_object_fraction:
            roles[nid] = "dev"
        else:
            roles[nid] = "train"
    return roles


# --------------------------------------------------------------------------------------
# Row labels
# --------------------------------------------------------------------------------------


def data_end_tt_s(last_observed_date_utc: str) -> float:
    """End of the last fully observed space-weather day (next 00:00 UTC), in TT seconds."""
    day = np.datetime64(last_observed_date_utc, "D") + np.timedelta64(1, "D")
    return float(utc_to_tt_seconds(f"{day}T00:00:00"))


def assign_split_labels(
    norad_ids: np.ndarray,
    start_epoch_tt_s: np.ndarray,
    cfg: SplitConfig,
    *,
    last_observed_date_utc: str,
) -> np.ndarray:
    """Label each (object, start epoch) row with one of :data:`SPLIT_LABELS`.

    Args:
        norad_ids: shape (N,), integer NORAD IDs.
        start_epoch_tt_s: shape (N,), prediction starts, TT seconds since J2000.0.
        cfg: the split configuration.
        last_observed_date_utc: last fully observed space-weather day
            (``YYYY-MM-DD``), normally ``SpaceWeather.last_fully_observed_date()``.
            Windows of any role ending after the end of this day are ``excluded``.
            Required, so the rule cannot be skipped by omission.

    Returns:
        Shape (N,) array of label strings (object dtype).
    """
    ids = np.asarray(norad_ids).astype(np.int64)
    start = np.asarray(start_epoch_tt_s, dtype=np.float64)
    if ids.shape != start.shape or ids.ndim != 1:
        raise ValueError("norad_ids and start_epoch_tt_s must be 1-D and the same length")
    roles = object_roles(np.unique(ids), cfg)
    role = np.array([roles[int(i)] for i in ids], dtype=object)

    cutoff = cfg.time_cutoff_tt_s
    end = start + cfg.max_horizon_days * _DAY_S
    after = start >= cutoff

    crosses = (~after) & (end > cutoff)
    near_event = np.zeros(ids.shape, dtype=bool)
    for ev in cfg.validation_events:
        lo = ev.tca_tt_s - cfg.event_exclusion_days_before * _DAY_S
        hi = ev.tca_tt_s + cfg.event_exclusion_days_after * _DAY_S
        near_event |= (start <= hi) & (end >= lo)

    labels = np.empty(ids.shape, dtype=object)
    held = role == "heldout"
    labels[held & after] = "test"
    labels[held & ~after] = "reserved_object"
    labels[~held & after] = "reserved_time"
    trainable = ~held & ~after
    labels[trainable] = role[trainable]
    labels[trainable & (crosses | near_event)] = "excluded"
    labels[end > data_end_tt_s(last_observed_date_utc)] = "excluded"
    assert set(labels.tolist()) <= set(SPLIT_LABELS)
    return labels


def split_summary(labels: np.ndarray, norad_ids: np.ndarray) -> dict[str, dict[str, int]]:
    """Row and object counts per label, for logging a split."""
    labels = np.asarray(labels)
    ids = np.asarray(norad_ids)
    out: dict[str, dict[str, int]] = {}
    for lab, n in sorted(Counter(labels.tolist()).items()):
        out[lab] = {"rows": int(n), "objects": int(np.unique(ids[labels == lab]).size)}
    return out
