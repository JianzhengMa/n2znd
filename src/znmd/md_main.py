"""
Entry point for running Zhu–Nakamura dynamics from the command line.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running either as ``python -m znmd.md_main`` or ``python md_main.py``.

from . import Zhu_Nakamura, load_config, normalize_config

DEFAULT_CONFIG: Dict[str, Any] = {
    "program_name": "MNDO2020",  # "GAMESSUS_MRSF" | "MNDO2020" | "MLFF"
    "hop_method": "ZN",  # "ZN" | "ZN_sim"
    "cal_type": "znmd",  # "bomd" | "znmd"
    "init_input": "E.inp",  # Initial input file name of electronic structure calculation
    "init_momentum": "momentum.inp",  # Initial momentum file name
    "atom_n": 8,  # Total atom number
    "total_time": 600,  # fs
    "state_n": 2,  # Number of electronic states to monitor
    "initial_state": 2,  # Initial electronic state of molecular dynamics
    "time_step": 0.5,  # fs
    "hopping_threshold_value": 0.3,  # eV
    "temp_dictory": "/public/home/majianzheng/softwares/mndo2020",
    "cpu_core": 64,
    "initial_temperature": 300,  # K
    "target_temperature": 300,  # K
    "fix_center": True,
    "fix_MB_temper": True,
    "berendsen_taut": 100,
    "andersen_prob": 0.1,
    "ensemble_type": "Verlet",
    "scal_step": 2,
    "net_num": 2,
    "net_path": "/public/home/majianzheng/net",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Zhu–Nakamura molecular dynamics simulation.")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a YAML configuration file. When omitted, a built-in example is used.",
    )
    parser.add_argument(
        "--set",
        metavar="KEY=VALUE",
        action="append",
        help="Override configuration entries (can be repeated).",
    )
    return parser.parse_args()


def _parse_overrides(pairs: Optional[List[str]]) -> Dict[str, Any]:
    if not pairs:
        return {}
    overrides: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid override '{pair}', expected KEY=VALUE format.")
        key, value = pair.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def build_configuration(args: argparse.Namespace) -> Dict[str, Any]:
    overrides = _parse_overrides(args.set)
    if args.config:
        return load_config(args.config, overrides=overrides)
    return normalize_config(DEFAULT_CONFIG, overrides=overrides)


def main() -> None:
    args = parse_args()
    config = build_configuration(args)
    simulator = Zhu_Nakamura(**config)
    simulator.main()


if __name__ == "__main__":
    main()
