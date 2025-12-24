import os
import re
import subprocess
from pathlib import Path

import numpy as np

from ..utils.constants import Unit, PERIODIC_TABLE, NUM_LAB

__all__ = ["MNDO2020"]


class MNDO2020:
    def __init__(
        self,
        init_input: str,
        init_momentum: str,
        atom_n: int,
        state_n: int,
        initial_state: int,
        temp_dictory: str,
    ):
        self.init_input = init_input
        self.init_momentum = init_momentum
        self.atom_n = atom_n
        self.state_n = state_n
        self.initial_state = initial_state
        self.temp_dictory = temp_dictory

    def atom_symbol(self):
        atom_nums = []
        with open(self.init_input) as fh:
            for line in fh:
                result = re.search(r"((\s+)?\d+)((\s+(-?)\d+\.\d+)(\s+\d+)){3}", line)
                if result:
                    atom_nums.append(result.group().split()[0])
        if atom_nums and int(atom_nums[-1]) == 0:
            atom_nums.pop()
        return [NUM_LAB[int(num)] for num in atom_nums]

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

    def check_out(self, filename):
        with open(filename) as fh:
            for line in fh:
                result = re.search(r"     TIME FOR GRADIENT EVALUATION", line)
                if result:
                    return result.group()

    def check_state(self):
        lines = open(self.init_input, "r").readlines()
        initial_state_file = 1
        for line in lines:
            for token in line.split():
                if "LROOT" in token or "lroot" in token:
                    initial_state_file = int(token.split("=")[1])
        if self.initial_state == initial_state_file:
            print("State check successful!\n")
        else:
            print(
                f"Error: Please ensure the initial state of {self.init_input} "
                "and molecular dynamics input file initial_state are the same.\n"
            )
            exit(0)

    def check_atom_n(self):
        input_atom_n = len(self.atom_symbol())
        if self.atom_n == input_atom_n:
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

    def run(self, filename):
        print("\nStart ab initio calculation!")
        exec_dir = self.temp_dictory
        prefix = Path(filename).stem
        executable = Path(exec_dir) / "mndo2020"
        output_path = Path(f"{prefix}.out")

        def _run_checked():
            try:
                with open(filename, "r", encoding="utf-8") as inp, output_path.open("w", encoding="utf-8") as out:
                    subprocess.run(
                        [str(executable)],
                        check=True,
                        stdin=inp,
                        stdout=out,
                        stderr=subprocess.STDOUT,
                    )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Command '{executable}' failed with code {exc.returncode}; see {output_path} for details."
                ) from exc

        _run_checked()

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

    def get_coordinate(self, filename):
        coordinate = []
        with open(filename) as fh:
            for line in fh:
                result = re.search(r"((\s+)?\d+)((\s+(-?)\d+\.\d+)(\s+\d+)){3}", line)
                if result:
                    if int(result.group().split()[0]) != 0:
                        coordinate.append(
                            [
                                float(result.group().split()[1]),
                                float(result.group().split()[3]),
                                float(result.group().split()[5]),
                            ]
                        )
        return np.array(coordinate).astype(float) / Unit.au_to_ang

    def get_gradient(self, filename):
        force_conver = (1 / (Unit.au_to_ev * Unit.ev_to_kcal_mol)) / (1 / Unit.au_to_ang)
        gradient = []
        command = (
            f"grep -A {self.atom_n + 7} '     TIME FOR GRADIENT EVALUATION' {filename}"
        )
        data = os.popen(command).read().splitlines()
        for line in data:
            result = re.search(r"(\s+\d+){2}(\s+(-?)\d+\.\d+){6}", line)
            if result:
                tokens = result.group().split()
                gradient.append(
                    [
                        float(tokens[5]) * force_conver,
                        float(tokens[6]) * force_conver,
                        float(tokens[7]) * force_conver,
                    ]
                )
        return np.array(gradient).astype(float)

    def get_state(self, filename):
        command = f"grep 'Correlation energy for CI root' {filename}"
        data = os.popen(command).read().strip()
        if data:
            tokens = data.split()
            if len(tokens) >= 6:
                return tokens[5].split(":")[0]
        return None

    def state_energy(self, filename, state):
        symbols = []
        energies = []
        command = (
            "sed -n '/ Results of CI calculation:/,"
            "/ AO basis set for the evaluation of spectroscopic observables:/p' "
            + filename
        )
        data = os.popen(command).read().splitlines()
        for line in data:
            result = re.search(r"(\s+)State.+", line)
            if result:
                tokens = result.group().split()
                symbols.append(tokens[1].split(",")[0])
                energies.append(float(tokens[8]) / Unit.au_to_ev)
        energy_dict = dict(zip(symbols, energies))
        if state:
            return energy_dict.get(str(state))
        return energy_dict

    def replace_coordinate(self, filename_initial, filename_new, coordinate_new):
        coordinate_new = coordinate_new * Unit.au_to_ang
        with open(filename_initial) as f1, open(filename_new, "w+") as f2:
            idx = 0
            for line in f1:
                result = re.search(r"((\s+)?\d+)((\s+(-?)\d+\.\d+)(\s+\d+)){3}", line)
                if result is None:
                    f2.write(line)
                elif (
                    result.group().split()[0] == "0"
                    and result.group().split()[2] == "0"
                    and result.group().split()[4] == "0"
                    and result.group().split()[6] == "0"
                ):
                    f2.write(line)
                else:
                    label = format(result.group().split()[0], ">5s")
                    coordinate_1 = format(coordinate_new[idx][0], ">18.10f")
                    fix_label_1 = format(result.group().split()[2], ">3s")
                    coordinate_2 = format(coordinate_new[idx][1], ">18.10f")
                    fix_label_2 = format(result.group().split()[4], ">3s")
                    coordinate_3 = format(coordinate_new[idx][2], ">18.10f")
                    fix_label_3 = format(result.group().split()[6], ">3s")
                    replaced = re.sub(
                        result.group(),
                        label
                        + coordinate_1
                        + fix_label_1
                        + coordinate_2
                        + fix_label_2
                        + coordinate_3
                        + fix_label_3,
                        line,
                    )
                    f2.write(replaced)
                    idx += 1

    def replace_state(self, filename_initial, filename_new, state_new):
        with open(filename_initial) as f1, open(filename_new, "w+") as f2:
            for line in f1:
                result = re.search(r".+((lroot)|(LROOT)).+", line)
                if result is None:
                    f2.write(line)
                else:
                    tokens = result.group().split()
                    updated = []
                    for token in tokens:
                        if "LROOT" in token or "lroot" in token:
                            name, _ = token.split("=")
                            updated.append(f"{name}={state_new}")
                        else:
                            updated.append(token)
                    f2.write(" " + " ".join(updated) + "\n")

    def gen_filename(self, prefix):
        return prefix + ".inp", prefix + ".out"

    def remove_file(self, prefix):
        inp = prefix + ".inp"
        out = prefix + ".out"
        if os.path.exists(inp):
            os.remove(inp)
        else:
            print(f"No such file or directory: '{inp}'\n")
        if os.path.exists(out):
            os.remove(out)
        else:
            print(f"No such file or directory: '{out}'\n")

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
