import numpy as np
import warnings
from functools import partial

from ..utils.constants import Unit
from ..interface import GAMESSUS_MRSF, MNDO2020, MLFF
from ..thermostat import Temp_Ensemb
from ..utils.io_utils import write_energy as log_energy
from ..utils.io_utils import write_coordinates as log_coordinates
from ..utils.io_utils import write_temperature as log_temperature
from ..utils.io_utils import write_state as log_state
from .integrators import IntegratorMixin
from .hop_manager import HopMixin
from .gradient_manager import GradientMixin
from .workflows import WorkflowMixin

np.set_printoptions(precision=12, suppress=True)
print = partial(print, flush=True)


class Zhu_Nakamura(IntegratorMixin, HopMixin, GradientMixin, WorkflowMixin):
    def __init__(
        self,
        atom_n,
        program_name="GAMESSUS_MRSF",
        hop_method="ZN",
        cal_type="bomd",
        init_input="trans.inp",
        init_momentum="momentum.inp",
        total_time=1000,
        state_n=2,
        initial_state=1,
        time_step=0.5,
        hopping_threshold_value=0.3,
        temp_dictory="/public/home/majianzheng/softwares/gamess",
        cpu_core=64,
        initial_temperature=300,
        target_temperature=300,
        fix_center=True,
        fix_MB_temper=True,
        berendsen_taut=100,
        andersen_prob=0.1,
        ensemble_type="Verlet",
        scal_step=1,
        net_path="./netpath",
        net_num=2,
    ):
        self.atom_n = atom_n
        self.program_name = program_name
        self.hop_method = hop_method
        self.cal_type = cal_type
        self.init_input = init_input
        self.init_momentum = init_momentum
        self.total_time = total_time
        self.state_n = state_n
        self.initial_state = initial_state
        self.time_step = time_step
        self.hopping_threshold_value = hopping_threshold_value
        self.unit = Unit()
        self.au_time_step = self.time_step / self.unit.au_to_fs
        self.temp_dictory = temp_dictory
        self.cpu_core = cpu_core
        self.initial_temperature = initial_temperature
        self.target_temperature = target_temperature
        self.fix_center = fix_center
        self.fix_MB_temper = fix_MB_temper
        self.berendsen_taut = berendsen_taut
        self.andersen_prob = andersen_prob
        self.ensemble_type = ensemble_type
        self.scal_step = scal_step
        self.net_num = net_num
        self.net_path = net_path

        normalized_program = self.program_name
        if normalized_program == "MLMD":
            warnings.warn(
                "program_name='MLMD' is deprecated, use 'MLFF' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            normalized_program = "MLFF"

        if normalized_program == "GAMESSUS_MRSF":
            self.mdif = GAMESSUS_MRSF(
                init_input=self.init_input,
                init_momentum=self.init_momentum,
                atom_n=self.atom_n,
                state_n=self.state_n,
                initial_state=self.initial_state,
                temp_dictory=self.temp_dictory,
                cpu_core=self.cpu_core,
            )
        elif normalized_program == "MNDO2020":
            self.mdif = MNDO2020(
                init_input=self.init_input,
                init_momentum=self.init_momentum,
                atom_n=self.atom_n,
                state_n=self.state_n,
                initial_state=self.initial_state,
                temp_dictory=self.temp_dictory,
            )
        elif normalized_program == "MLFF":
            self.mdif = MLFF(
                init_input=self.init_input,
                init_momentum=self.init_momentum,
                atom_n=self.atom_n,
                state_n=self.state_n,
                initial_state=self.initial_state,
                net_num=self.net_num,
                net_path=self.net_path,
                cpu_core=self.cpu_core,
            )
        else:
            raise ValueError(f"Unsupported program_name {self.program_name}")

        self.program_name = normalized_program

        self.te = Temp_Ensemb(
            program_name=self.program_name,
            init_input=self.init_input,
            atom_n=self.atom_n,
            time_step=self.time_step,
            initial_temperature=self.initial_temperature,
            target_temperature=self.target_temperature,
            fix_center=self.fix_center,
            berendsen_taut=self.berendsen_taut,
            andersen_prob=self.andersen_prob,
            ensemble_type=self.ensemble_type,
            scal_step=self.scal_step,
            fix_MB_temper=self.fix_MB_temper,
        )

    def write_energy(self, filename, energy, step, time):
        log_energy(filename, energy, step, time)

    def write_matrix(self, filename, matrix, step, time):
        log_coordinates(filename, self.mdif.atom_symbol(), matrix, step, time)

    def write_temperature(self, filename, temperature, step, time):
        log_temperature(filename, temperature, step, time)

    def write_state(self, filename, state, step, time):
        log_state(filename, state, step, time)

    def main(self):
        if self.cal_type == "bomd":
            if self.program_name == "MLFF":
                self.ml_bomd_on_the_fly()
            else:
                self.bomd_on_the_fly()
        elif self.cal_type == "znmd":
            if self.program_name == "MLFF":
                self.ml_znmd()
            else:
                self.znmd_on_the_fly()
        elif self.cal_type == "mlzn":
            warnings.warn(
                "cal_type='mlzn' is deprecated; use program_name='MLFF' with cal_type='znmd'.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.ml_znmd()
        else:
            print("\nError: Please specify the correct cal_type!\n")
            exit(0)
