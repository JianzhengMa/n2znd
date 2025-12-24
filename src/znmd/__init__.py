"""
N2ZND (Neural Network Zhu–Nakamura Dynamics) package initialization.

This package hosts the main building blocks required for running
Zhu–Nakamura nonadiabatic molecular dynamics simulations:

- ``constants``: unit conversion helpers and periodic-table data
- ``engines``: electronic structure / force evaluators
- ``thermostat``: thermostat and ensemble control utilities
- ``dynamics``: integrators and high-level simulation drivers
"""

from functools import partial
import builtins as _builtins

from .utils.constants import Unit
from .interface import GAMESSUS_MRSF, MNDO2020, MLFF, MLMD
from .thermostat.ensembles import Temp_Ensemb
from .dynamics.zhu_nakamura import Zhu_Nakamura
from .utils.config_loader import load_config, create_simulator_from_yaml, normalize_config

# Force all print calls to flush immediately so log streaming stays responsive.
_builtins.print = partial(_builtins.print, flush=True)

__all__ = [
    "Unit",
    "GAMESSUS_MRSF",
    "MNDO2020",
    "MLFF",
    "MLMD",
    "Temp_Ensemb",
    "Zhu_Nakamura",
    "load_config",
    "create_simulator_from_yaml",
    "normalize_config",
]
