import os
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..utils.constants import Unit, PERIODIC_TABLE

__all__ = ["GAMESSUS_MRSF"]


class GAMESSUS_MRSF:
    _rungms_template_cache: Dict[Path, Tuple[List[str], int]] = {}
    def __init__(
        self,
        init_input: str,
        init_momentum: str,
        atom_n: int,
        state_n: int,
        initial_state: int,
        temp_dictory: str,
        cpu_core: int,
    ):
        self.init_input = init_input
        self.init_momentum = init_momentum
        self.atom_n = atom_n
        self.state_n = state_n
        self.initial_state = initial_state
        self.temp_dictory = temp_dictory
        self.cpu_core = cpu_core

    # ---------- basic helpers ----------

    def atom_symbol(self) -> List[str]:
        symbols = []
        with open(self.init_input) as fh:
            for line in fh:
                result = re.search(r"([a-zA-Z]{1,2})(\s+\d+(\.\d+)?)(\s+(-?)\d+\.\d+){3}", line)
                if result:
                    symbols.append(result.group().split()[0])
        return symbols

    def atom_mass(self):
        masses = np.zeros((self.atom_n, 1))
        symbols = self.atom_symbol()
        for i in range(self.atom_n):
            masses[i] = PERIODIC_TABLE[symbols[i]] * Unit.amu_to_au
        return masses

    def check_in(self, filename: str):
        if os.path.exists(filename):
            print(f"{filename} is exists\n")
        else:
            print(f"ERROR: {filename} is not exists,please check it!\n")
            exit(0)

    def check_out(self, filename: str):
        with open(filename) as fh:
            lines = fh.readlines()
            for line in reversed(lines):
                result = re.search("UNITS ARE HARTREE/BOHR", line)
                if result:
                    return result.group()

    def check_state(self):
        with open(self.init_input) as fh:
            initial_state_file = 1
            for line in fh:
                result = re.search(r"((\$TDDFT)|(\$tddft)).+((\$END)|(\$end))", line)
                if result:
                    for token in result.group().split():
                        if "IROOT" in token or "iroot" in token:
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
        atom_mom_n = len(atom_mom_sym)
        if atom_mom_n == self.atom_n:
            print("Atomic number of momentum input file check successful!")
        else:
            print(
                f"Error: Please ensure the atomic number of {self.init_momentum} "
                "and molecular dynamics input file are the same."
            )
            exit(0)
        target_symbols = self.atom_symbol()
        if any(atom_mom_sym[i] != target_symbols[i] for i in range(self.atom_n)):
            print(
                f"Error: Please ensure the atom symbol of {self.init_momentum} "
                f"and {self.init_input} are same.\n"
            )
            exit(0)
        print("Atomic symbol of momentum input file check successful!\n")

    # ---------- execution ----------

    @classmethod
    def _get_rungms_template(cls, exec_dir: Path) -> Tuple[List[str], int]:
        key = exec_dir.resolve()
        if key not in cls._rungms_template_cache:
            source = key / "rungms"
            with source.open("r", encoding="utf-8") as fh:
                lines = fh.readlines()
            mode = source.stat().st_mode
            cls._rungms_template_cache[key] = (lines, mode)
        return cls._rungms_template_cache[key]

    def run(self, filename: str):
        print("\nStart ab initio calculation!")
        file_prefix = Path(filename).stem
        tmp_run_dir = file_prefix + str(random.randint(1, int(1e16)))
        cpu_core = str(self.cpu_core)
        exec_dir = Path(self.temp_dictory)
        restart_dir = exec_dir / "restart"
        run_dir = restart_dir / tmp_run_dir
        rungms_tmp = run_dir / "rungms-tmp"

        restart_dir.mkdir(parents=True, exist_ok=True)
        if run_dir.exists():
            print(f"{tmp_run_dir} exists")
            shutil.rmtree(run_dir)
        run_dir.mkdir()

        template_lines, template_mode = self._get_rungms_template(exec_dir)
        lines = list(template_lines)
        insert_line = f"set TmpRunDir={tmp_run_dir}\n"
        position = 1 if len(lines) > 1 else len(lines)
        lines.insert(position, insert_line)
        with rungms_tmp.open("w", encoding="utf-8") as handler:
            handler.writelines(lines)
        os.chmod(rungms_tmp, template_mode)

        def _run_checked(args, **kwargs):
            cmd_str = " ".join(map(str, args))
            try:
                subprocess.run(args, check=True, **kwargs)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"Command '{cmd_str}' failed with code {exc.returncode}") from exc

        def _clear_ipc_semaphores():
            try:
                result = subprocess.run(["ipcs", "-s"], check=True, capture_output=True, text=True)
            except (FileNotFoundError, subprocess.CalledProcessError):
                return
            for line in result.stdout.splitlines()[3:]:
                parts = line.split()
                if len(parts) >= 2:
                    sem_id = parts[1]
                    _run_checked(["ipcrm", "-s", sem_id])

        log_path = Path(f"{file_prefix}.out")
        try:
            _clear_ipc_semaphores()
            with log_path.open("w", encoding="utf-8") as out_fh:
                _run_checked(
                    [str(rungms_tmp), filename, "00", cpu_core],
                    stdout=out_fh,
                    stderr=subprocess.STDOUT,
                )
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    # ---------- I/O utilities ----------

    def print_matrix(self, matrix, var_name, step, time):
        if isinstance(time, str):
            string = f" {var_name} at step {step} Time = {time}(fs) "
        else:
            string = f" {var_name} at step {step} Time = {time:.2f}(fs) "
        print(f"\n{string:*^63}\n")
        for idx in range(self.atom_n):
            print(
                f"{self.atom_symbol()[idx]:2s}{matrix[idx][0]:20.10f}"
                f"{matrix[idx][1]:20.10f}{matrix[idx][2]:20.10f}"
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
                result = re.search(r"([a-zA-Z]{1,2})(\s+\d+(\.\d+)?)(\s+(-?)\d+\.\d+){3}", line)
                if result:
                    coordinate.append([float(i) for i in result.group().split()[2:5]])
        return np.array(coordinate).astype(float) / Unit.au_to_ang

    def get_gradient(self, filename):
        gradient = []
        command = f"grep -A {self.atom_n} 'UNITS ARE HARTREE/BOHR' {filename}"
        data = os.popen(command).read().splitlines()
        for line in data:
            result = re.search(r"(\s+\d+\s+)([a-zA-Z]{1,2})(\s+(-?)\d+\.\d+){3}", line)
            if result:
                gradient.append([float(i) for i in result.group().split()[2:5]])
        return np.array(gradient).astype(float)

    def get_state(self, filename):
        command = f"grep 'SELECTING EXCITED STATE IROOT=' {filename}"
        data = os.popen(command).read().strip()
        if data:
            tokens = data.split()
            if len(tokens) >= 5:
                return tokens[4]
        return None

    def state_energy(self, filename, state):
        symbols = []
        energies = []
        command = "sed -n '/SUMMARY OF MRSF-DFT RESULTS/,/SELECTING EXCITED STATE IROOT/p' " + filename
        data = os.popen(command).read().splitlines()
        for line in data:
            result = re.search(r"(\s+\d+\s+)([a-zA-Z]{1,2})(\s+(-?)\d+\.\d+){7}", line)
            if result:
                components = result.group().split()
                symbols.append(components[0])
                energies.append(float(components[2]))
        energy_dict = dict(zip(symbols, energies))
        if state:
            return energy_dict.get(str(state))
        return energy_dict

    def replace_coordinate(self, filename_initial, filename_new, coordinate_new):
        coordinate_new = coordinate_new * Unit.au_to_ang
        symbols = self.atom_symbol()
        with open(filename_initial) as f1, open(filename_new, "w+") as f2:
            counter = 0
            for line in f1:
                result = re.search(r"([a-zA-Z]{1,2})(\s+\d+(\.\d+)?)(\s+(-?)\d+\.\d+){3}", line)
                if result is None:
                    f2.write(line)
                else:
                    symbol = format(symbols[counter], "<2s")
                    label = result.group().split()[1]
                    c1 = format(coordinate_new[counter][0], ">18.10f")
                    c2 = format(coordinate_new[counter][1], ">18.10f")
                    c3 = format(coordinate_new[counter][2], ">18.10f")
                    replaced = re.sub(
                        result.group(),
                        f"{symbol}  {label}{c1}{c2}{c3}",
                        line,
                    )
                    f2.write(replaced)
                    counter += 1

    def replace_state(self, filename_initial, filename_new, state_new):
        with open(filename_initial) as f1, open(filename_new, "w+") as f2:
            for line in f1:
                result = re.search(r"((\$TDDFT)|(\$tddft)).+((\$END)|(\$end))", line)
                if result is None:
                    f2.write(line)
                else:
                    tokens = result.group().split()
                    updated = []
                    for token in tokens:
                        if "IROOT" in token or "iroot" in token:
                            name, _ = token.split("=")
                            updated.append(f"{name}={state_new}")
                        else:
                            updated.append(token)
                    f2.write(" " + " ".join(updated) + "\n")

    def gen_filename(self, prefix):
        return prefix + ".inp", prefix + ".out"

    def remove_file(self, prefix):
        inp = Path(prefix + ".inp")
        out = Path(prefix + ".out")
        if inp.exists():
            inp.unlink()
        else:
            print(f"No such file or directory: '{inp}'\n")
        if out.exists():
            out.unlink()
        else:
            print(f"No such file or directory: '{out}'\n")

    def clear_file(self):
        for file_name in [
            "coordinate.xyz",
            "energy.log",
            "grdient.log",
            "momentum.log",
            "temperature.log",
            "state.log",
            "ConIs.xyz",
            "hopping_flag.log",
        ]:
            path = Path(file_name)
            if path.exists():
                path.unlink()
