"""Reading single cells and columns out of dataset tables.

pandas types a label lookup as possibly returning a ``Series``, because a
duplicated column label yields every matching column rather than one value. A
type checker cannot prove a label is unique, so every ``int(row["x"])`` in the
codebase is reported as passing a ``Series`` where a number is expected.

Silencing that per call site would also silence the real fault it stands for:
duplicated columns do occur in metadata written by mixed tool versions, and
then ``int()`` fails somewhere far from the cause with an error that names
neither the column nor the file. These helpers check for it once and name the
column, which resolves the diagnostic and the underlying hazard together.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def row_scalar(row: pd.Series, column: str, default: Any = None) -> Any:
    """Return one scalar metadata cell, rejecting duplicate-column results."""
    value = row.get(column, default)
    if isinstance(value, pd.Series):
        raise TypeError(f"Metadata column {column!r} is duplicated")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        raise ValueError(f"Metadata column {column!r} must contain scalars")
    return value


def column_values(frame: pd.DataFrame, column: str) -> list[Any]:
    """Return a vector-valued column with duplicate columns ruled out."""
    values = frame[column]
    if not isinstance(values, pd.Series):
        raise TypeError(f"Data column {column!r} is duplicated")
    return values.tolist()


__all__ = ["column_values", "row_scalar"]
