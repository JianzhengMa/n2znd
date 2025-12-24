import os
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..utils.constants import Unit, PERIODIC_TABLE

__all__ = ["MLMD"]


class MLMD:
    def __init__(
        self,
        init_input: str,
        init_momentum: str,
        atom_n: int,
        state_n: int,
        initial_state: int,
        net_num: int,
        net_path: str,
        cpu_core: int,
    ):
        self.init_input = init_input
        self.init_momentum = init_momentum
        self.atom_n = atom_n
        self.state_n = state_n
        self.initial_state = initial_state
        self.net_num = net_num
        self.net_path = net_path
        self.cpu_core = cpu_core
        self._calculator_cache: Dict[str, object] = {}
        self._species_type_map: Optional[Dict[str, str]] = None

    def atom_symbol(self):
        from ase.io import read

        atoms = read(self.init_input, format="xyz")
        return atoms.get_chemical_symbols()

    def atom_mass(self):
        masses = np.zeros((self.atom_n, 1))
        symbols = self.atom_symbol()
        for i in range(self.atom_n):
            masses[i] = PERIODIC_TABLE[symbols[i]] * Unit.amu_to_au
        return masses

    def check_in(self, filename):
        if os.path.exists(filename):
            print(f"{filename} is exists\n")
        else:
            print(f"ERROR: {filename} is not exists,please check it!\n")
            exit(0)

    def check_atom_n(self):
        if self.atom_n == len(self.atom_symbol()):
            print("Atomic number check successful!\n")
        else:
            print(
                f"Error: Please ensure the atomic number of {self.init_input} "
                "and molecular dynamics input file are the same.\n"
            )
            exit(0)

    def check_momentum_file(self):
        atom_mom_sym = []
        with open(self.init_momentum) as fh:
            for line in fh:
                result = re.search(r"[A-Za-z]{1,2}(\s+-?\d+\.\d+){3}", line)
                if result:
                    atom_mom_sym.append(result.group().split()[0])
        if len(atom_mom_sym) != self.atom_n:
            print(
                f"Error: Please ensure the atomic number of {self.init_momentum} "
                "and molecular dynamics input file are the same."
            )
            exit(0)
        symbols = self.atom_symbol()
        if any(atom_mom_sym[i] != symbols[i] for i in range(self.atom_n)):
            print(
                f"Error: Please ensure the atom symbol of {self.init_momentum} "
                f"and {self.init_input} are same.\n"
            )
            exit(0)
        print("Atomic symbol of momentum input file check successful!\n")

    def check_num_net(self):
        if self.net_num != self.state_n:
            print("Error: please ensure the net_num and state_n are same!")
            exit(0)
        for idx in range(self.net_num):
            path = Path(self.net_path) / f"{idx + 1}.pth"
            if not path.exists():
                print(f"Error: please check the net path {path}! \n")
                exit(0)

    def _species_mapping(self, atoms):
        if self._species_type_map is None:
            self._species_type_map = {s: s for s in atoms.get_chemical_symbols()}
        return self._species_type_map

    def _get_calculator(self, net_path, atoms):
        calc = self._calculator_cache.get(net_path)
        if calc is None:
            from nequip.ase import NequIPCalculator

            calc = NequIPCalculator.from_deployed_model(
                model_path=net_path,
                set_global_options=True,
                species_to_type_name=self._species_mapping(atoms),
            )
            self._calculator_cache[net_path] = calc
        return calc

    def run_mlps(self, net_path, atoms):
        calc = self._get_calculator(net_path, atoms)
        atoms.set_calculator(calc=calc)
        return (
            atoms.get_potential_energy() / Unit.au_to_ev,
            atoms.get_forces() / Unit.Ha_Bohr_to_ev_ang,
        )

    def run(self, atoms):
        state_list = []
        energy_list = []
        grad_list = []
        for i in range(self.net_num):
            net_dictory = str(Path(self.net_path) / f"{i + 1}.pth")
            energy, force = self.run_mlps(net_dictory, atoms)
            state_list.append(str(i + 1))
            energy_list.append(energy)
            grad_list.append(-force)
        return dict(zip(state_list, energy_list)), dict(zip(state_list, grad_list))

    def print_matrix(self, matrix, var_name, step, time):
        if isinstance(time, str):
            string = f" {var_name} at step {step} Time = {time}(fs) "
        else:
            string = f" {var_name} at step {step} Time = {time:.2f}(fs) "
        print(f"\n{string:*^63}\n")
        for idx in range(self.atom_n):
            print(
                f"{self.atom_symbol()[idx]:2s}"
                f"{matrix[idx][0]:20.10f}{matrix[idx][1]:20.10f}{matrix[idx][2]:20.10f}"
            )

    def ini_momentum(self, filename):
        momentum = []
        with open(filename) as fh:
            for line in fh:
                result = re.search(r"[A-Za-z]{1,2}(\s+-?\d+\.\d+){3}", line)
                if result:
                    momentum.append(
                        [format(float(i), ">.10f") for i in result.group().split()[1:4]]
                    )
        return np.array(momentum).astype(float)

    def clear_file(self):
        for file in [
            "coordinate.xyz",
            "energy.log",
            "grdient.log",
            "momentum.log",
            "temperature.log",
            "state.log",
            "ConIs.xyz",
            "hopping_flag.log",
        ]:
            if os.path.exists(file):
                os.remove(file)
