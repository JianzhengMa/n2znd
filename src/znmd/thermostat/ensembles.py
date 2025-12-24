import os
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..utils.constants import Unit, PERIODIC_TABLE, NUM_LAB

__all__ = ["Temp_Ensemb"]


@dataclass
class Temp_Ensemb:
    program_name: str
    init_input: str
    atom_n: int
    time_step: float
    initial_temperature: float
    target_temperature: float
    fix_center: bool
    berendsen_taut: float
    andersen_prob: float
    ensemble_type: Literal["Verlet", "Velo_Scal", "Berendsen", "Berendsen_Smooth", "Andersen"]
    scal_step: int
    fix_MB_temper: bool

    def atom_symbol_GAMESS_MRSF(self):
        symbols = []
        with open(self.init_input) as fh:
            for line in fh:
                result = re.search(r"([a-zA-Z]{1,2})(\s+\d+(\.\d+)?)(\s+(-?)\d+\.\d+){3}", line)
                if result:
                    symbols.append(result.group().split()[0])
        return symbols

    def atom_symbol_MNDO2020(self):
        atom_num = []
        with open(self.init_input) as fh:
            for line in fh:
                result = re.search(r"((\s+)?\d+)((\s+(-?)\d+\.\d+)(\s+\d+)){3}", line)
                if result:
                    atom_num.append(result.group().split()[0])
        if atom_num and int(atom_num[-1]) == 0:
            atom_num.pop()
        return [NUM_LAB[int(num)] for num in atom_num]

    def atom_symbol_MLMD(self):
        symbols = []
        with open(self.init_input) as fh:
            for line in fh:
                result = re.search(r"(\s+)?([a-zA-Z]{1,2})(\s+(-?)\d+\.\d+){3}", line)
                if result:
                    symbols.append(result.group().split()[0])
        return symbols

    def atom_mass(self):
        masses = np.zeros((self.atom_n, 1))
        for idx in range(self.atom_n):
            if self.program_name == "GAMESSUS_MRSF":
                name = self.atom_symbol_GAMESS_MRSF()[idx]
            elif self.program_name == "MNDO2020":
                name = self.atom_symbol_MNDO2020()[idx]
            else:
                name = self.atom_symbol_MLMD()[idx]
            masses[idx] = PERIODIC_TABLE[name] * Unit.amu_to_au
        return masses

    def get_temperature(self, momentum):
        kin_energy = np.zeros((self.atom_n, 3))
        masses = self.atom_mass()
        for idx in range(self.atom_n):
            kin_energy[idx] = np.power(momentum[idx], 2) / (2 * masses[idx])
        kin_sum = np.sum(kin_energy) * Unit.au_to_ev
        if self.fix_center:
            dof = 3 * self.atom_n - 3
        else:
            dof = 3 * self.atom_n
        return kin_sum / (0.5 * dof * Unit.K_B)

    def fix_center_velo(self, momentum):
        return momentum - momentum.sum(axis=0) / float(len(momentum))

    def boltzmann_random(self, width, size):
        x = np.random.random_sample(size=size)
        y = np.random.random_sample(size=size)
        return width * np.cos(2 * np.pi * x) * np.sqrt(-2 * np.log(1 - y))

    def maxwell_boltzmann_velo(self, temperature):
        mass_matrix = np.repeat(self.atom_mass(), 3).reshape(self.atom_n, 3)
        temp = temperature * Unit.K_B / Unit.au_to_ev
        width = np.sqrt(temp / mass_matrix)
        return self.boltzmann_random(width, size=(self.atom_n, 3))

    def maxwell_boltzmann_momentum(self, temperature):
        mass_matrix = np.repeat(self.atom_mass(), 3).reshape(self.atom_n, 3)
        velocities = self.maxwell_boltzmann_velo(temperature)
        momentum = velocities * mass_matrix
        if self.fix_MB_temper:
            factor = np.sqrt(self.target_temperature / self.get_temperature(momentum))
            momentum = momentum * factor
        if self.fix_center:
            momentum = self.fix_center_velo(momentum)
        return momentum

    def Velo_Scal(self, momentum, loop, nsw):
        if loop % self.scal_step == 0:
            target = self.initial_temperature + (self.target_temperature - self.initial_temperature) * loop / nsw
            Lambda = np.sqrt(target / self.get_temperature(momentum))
            momentum = momentum * Lambda
        if self.fix_center:
            momentum = self.fix_center_velo(momentum)
        return momentum

    def berendsen(self, momentum):
        taut_scl = self.time_step / self.berendsen_taut
        old_temperature = self.get_temperature(momentum)
        temperature_scal = np.sqrt(1.0 + (self.target_temperature / old_temperature - 1.0) * taut_scl)
        momentum = temperature_scal * momentum
        if self.fix_center:
            momentum = self.fix_center_velo(momentum)
        return momentum

    def berendsen_smooth(self, momentum):
        taut_scl = self.time_step / self.berendsen_taut
        old_temperature = self.get_temperature(momentum)
        temperature_scal = np.sqrt(1.0 + (self.target_temperature / old_temperature - 1.0) * taut_scl)
        temperature_scal = min(max(temperature_scal, 0.9), 1.1)
        momentum = temperature_scal * momentum
        if self.fix_center:
            momentum = self.fix_center_velo(momentum)
        return momentum

    def andersen(self, momentum):
        mass_matrix = np.repeat(self.atom_mass(), 3).reshape(self.atom_n, 3)
        velocities = momentum / mass_matrix
        random_velocity = self.maxwell_boltzmann_velo(self.target_temperature)
        mask = np.random.random_sample(size=velocities.shape) <= self.andersen_prob
        velocities[mask] = random_velocity[mask]
        momentum = velocities * mass_matrix
        if self.fix_center:
            momentum = self.fix_center_velo(momentum)
        return momentum

    def ensemble(self, momentum, loop, nsw):
        if self.ensemble_type == "Verlet":
            if self.fix_center:
                momentum = self.fix_center_velo(momentum)
            return momentum
        if self.ensemble_type == "Velo_Scal":
            return self.Velo_Scal(momentum, loop, nsw)
        if self.ensemble_type == "Berendsen":
            return self.berendsen(momentum)
        if self.ensemble_type == "Berendsen_Smooth":
            return self.berendsen_smooth(momentum)
        if self.ensemble_type == "Andersen":
            return self.andersen(momentum)
        print("\nError:\nPlease input the correct ensemble_type!!!\n")
        os._exit(0)
