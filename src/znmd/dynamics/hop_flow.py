"""
Utility helpers that orchestrate hopping probability evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence, Tuple

GradientTriple = Tuple[Any, Any, Any]
SingleGradientFetcher = Callable[[str], Tuple[GradientTriple, Any]]
DoubleGradientFetcher = Callable[[], Tuple[GradientTriple, Any, GradientTriple, Any]]


@dataclass
class HopOutcome:
    probability: float
    momentum: Any
    velocity_factor: float


@dataclass
class SingleHopResult:
    outcome: HopOutcome
    gradients: GradientTriple
    metadata: Any


@dataclass
class DoubleHopResult:
    to_lower: HopOutcome
    to_upper: HopOutcome
    gradients_up: GradientTriple
    gradients_down: GradientTriple
    metadata_up: Any
    metadata_down: Any


def _print_section(title: str, end: bool = False) -> None:
    if end:
        print("\n{0:*^63}\n".format(title))
    else:
        print("\n{0:*^63}".format(title))


def process_single_hop(
    workflow,
    hop_style: str,
    gradient_fetcher: SingleGradientFetcher,
    base_gradients: GradientTriple,
    q_history: Sequence[Any],
    previous_momentum: Any,
    energies: Tuple[Any, Any],
    *,
    begin_title: str,
    end_title: str,
):
    """
    Shared helper for single adjacent hops (U-D or D-U).

    Parameters
    ----------
    workflow:
        The workflow mixin (provides ``cal_hop_factor`` and ``print``).
    hop_style:
        Either ``'U-D'`` or ``'D-U'``.
    gradient_fetcher:
        Callable that returns the target-state gradient history plus metadata.
    base_gradients:
        Gradient history of the current state (sub-2, sub-1, current).
    q_history:
        Coordinate history tuple (sub-2, sub-1, current).
    previous_momentum:
        Momentum at step sub-1 (used inside ``cal_hop_factor``).
    energies:
        Tuple of (U_u, U_d).
    begin_title / end_title:
        Strings used to reproduce the original logging.
    """

    _print_section(begin_title)
    gradient_triplet, metadata = gradient_fetcher(hop_style)
    _print_section(end_title, end=True)

    if hop_style == "U-D":
        grad_upper = base_gradients
        grad_lower = gradient_triplet
    else:
        grad_upper = gradient_triplet
        grad_lower = base_gradients

    hop_p, new_P, v_factor = workflow.cal_hop_factor(
        hop_style,
        *q_history,
        previous_momentum,
        energies[0],
        energies[1],
        *grad_upper,
        *grad_lower,
    )
    return SingleHopResult(
        outcome=HopOutcome(hop_p, new_P, v_factor),
        gradients=gradient_triplet,
        metadata=metadata,
    )


def process_double_hop(
    workflow,
    gradient_fetcher: DoubleGradientFetcher,
    q_history: Sequence[Any],
    previous_momentum: Any,
    *,
    energies_ud: Tuple[Any, Any],
    energies_du: Tuple[Any, Any],
    base_gradients: GradientTriple,
    begin_title: str,
    end_title: str,
):
    """
    Shared helper for double hop events.

    Returns hop information for both U->D and D->U directions together with the
    target-state gradient histories and metadata objects.
    """

    _print_section(begin_title)
    grads_up, meta_up, grads_down, meta_down = gradient_fetcher()
    _print_section(end_title, end=True)

    hop_ud = workflow.cal_hop_factor(
        "U-D",
        *q_history,
        previous_momentum,
        energies_ud[0],
        energies_ud[1],
        *base_gradients,
        *grads_down,
    )
    hop_du = workflow.cal_hop_factor(
        "D-U",
        *q_history,
        previous_momentum,
        energies_du[0],
        energies_du[1],
        *grads_up,
        *base_gradients,
    )
    return DoubleHopResult(
        to_lower=HopOutcome(*hop_ud),
        to_upper=HopOutcome(*hop_du),
        gradients_up=grads_up,
        gradients_down=grads_down,
        metadata_up=meta_up,
        metadata_down=meta_down,
    )


__all__ = [
    "process_single_hop",
    "process_double_hop",
    "SingleHopResult",
    "DoubleHopResult",
    "HopOutcome",
]
