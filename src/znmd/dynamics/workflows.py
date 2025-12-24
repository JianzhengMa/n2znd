from .workflow_znmd import ZNMDWorkflowMixin
from .workflow_bomd import BOMDWorkflowMixin
from .workflow_ml import MLWorkflowMixin


class WorkflowMixin(ZNMDWorkflowMixin, BOMDWorkflowMixin, MLWorkflowMixin):
    """Aggregate workflow mixins for different simulation modes."""

    def _log_bomd_step(
        self,
        coordinate_matrix,
        gradient_matrix,
        momentum_matrix,
        energy_dict,
        current_state,
        temperature,
        loop,
        current_time,
    ):
        """统一打印与输出 BOMD 步骤数据，减少不同流程之间的重复代码。"""
        state_key = str(current_state)
        if state_key not in energy_dict:
            raise KeyError(f"State {state_key} not found in energy dictionary.")

        kin_energy = self.cal_E_Kin(momentum_matrix)
        energy_current_state = energy_dict[state_key]
        total_energy = energy_current_state + kin_energy
        energy_list = list(energy_dict.values())

        self.mdif.print_matrix(coordinate_matrix * self.unit.au_to_ang, "Coordinate", str(loop), current_time)
        self.mdif.print_matrix(gradient_matrix, "Gradient", str(loop), current_time)
        self.mdif.print_matrix(momentum_matrix, "Momentun", str(loop), current_time)

        print('\n{0:*^63}\n'.format(" Summary information "))
        print("Step:{0}  time:{1:.2f}(fs)".format(loop, current_time))
        print("Current electronic state:{0}".format(state_key))
        print("Static electronic structure calculation total energy(a.u.):{0}".format(energy_current_state))
        print("Kinetic energy(eV):{0}".format(kin_energy * self.unit.au_to_ev))
        print("Temperature: {0} K".format(temperature))
        print("Total energy (a.u.):{0}".format(total_energy))

        self.write_matrix("coordinate.xyz", coordinate_matrix * self.unit.au_to_ang, loop, current_time)
        self.write_matrix("grdient.log", gradient_matrix, loop, current_time)
        self.write_matrix("momentum.log", momentum_matrix, loop, current_time)
        self.write_energy(
            "energy.log",
            [total_energy, energy_current_state, kin_energy] + energy_list,
            loop,
            current_time,
        )
        self.write_temperature("temperature.log", temperature, loop, current_time)
