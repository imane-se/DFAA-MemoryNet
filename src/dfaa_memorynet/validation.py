"""Generic validation split utilities for DFAA-MemoryNet."""

from __future__ import annotations
import numpy as np


def loyo_splits(years):
    """Yield leave-one-year-out train/test index arrays."""
    years = np.asarray(years)
    for held_out_year in np.unique(years):
        test = np.flatnonzero(years == held_out_year)
        train = np.flatnonzero(years != held_out_year)
        yield int(held_out_year), train, test


def loeo_splits(event_ids, valid_event_mask=None):
    """Yield leave-one-event-out train/test index arrays.

    Parameters
    ----------
    event_ids
        One event identifier per temporal sample. Non-event samples may be
        encoded as empty string, None, or NaN.
    valid_event_mask
        Optional boolean mask specifying which samples belong to the event
        catalogue used for LOEO.

    Notes
    -----
    Event construction and event-buffer definitions are study-specific and are
    intentionally outside this model-only repository.
    """
    event_ids = np.asarray(event_ids, dtype=object)

    if valid_event_mask is None:
        valid_event_mask = np.array([
            (x is not None)
            and str(x) not in {"", "nan", "None"}
            for x in event_ids
        ], dtype=bool)
    else:
        valid_event_mask = np.asarray(valid_event_mask, dtype=bool)

    unique_events = np.unique(event_ids[valid_event_mask])

    for event in unique_events:
        test = np.flatnonzero(valid_event_mask & (event_ids == event))
        train = np.flatnonzero(~(event_ids == event))
        yield str(event), train, test
