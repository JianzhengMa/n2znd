"""
Utilities for bulk operations on HistoryWindow instances.
"""

from __future__ import annotations

from typing import Iterable, Optional

from ..utils import HistoryWindow

__all__ = ["reset_windows", "append_windows"]


def reset_windows(windows: Iterable[Optional[HistoryWindow]], values: Iterable[Optional[object]]) -> None:
    for window, value in zip(windows, values):
        if window is None or value is None:
            continue
        window.seed([value])


def append_windows(windows: Iterable[Optional[HistoryWindow]], values: Iterable[Optional[object]]) -> None:
    for window, value in zip(windows, values):
        if window is None or value is None:
            continue
        window.append(value)
