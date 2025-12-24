"""
Electronic structure interface classes.
"""

from .gamess_mrsf import GAMESSUS_MRSF
from .mndo2020 import MNDO2020
from .mlmd import MLMD as _MLFFBase

MLFF = _MLFFBase
MLMD = _MLFFBase  # Backward compatibility alias

__all__ = ["GAMESSUS_MRSF", "MNDO2020", "MLFF", "MLMD"]
